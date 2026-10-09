"""Stage 6 — make the pieces: narration audio and one picture per shot.

Order matters. Narration is produced **first**, its real duration is measured, and
only then is the timeline recomputed and the visuals cut to it. The other order --
cut picture, then speak over it -- is how you end up with a shot list whose last
word is clipped or a 0.4s freeze frame at the tail.

Provider work runs in a thread pool because it is latency-bound, not CPU-bound; the
ffmpeg work that follows is deliberately serial so a single encode owns the machine.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import Any

from ..captions import plan_cues
from ..errors import PipelineError
from ..models import AssetSet, Shot, ShotAsset, StageName, VisualKind
from ..providers.tts import estimate_speech_seconds
from ..timing import MIN_SHOT as _MIN_SHOT


class GenerateStage:
    name = StageName.GENERATE

    def run(self, ctx: Any) -> AssetSet:
        board = ctx.artifacts.storyboard
        if board is None or not board.shots:
            raise PipelineError("no storyboard to generate against", stage="generate")
        shots = board.shots
        size = (board.width, board.height)

        with cf.ThreadPoolExecutor(max_workers=max(1, ctx.settings.max_workers)) as pool:
            narration = list(pool.map(lambda shot: self._narration(ctx, shot), shots))

        self._retime(ctx, shots, narration)
        board.captions = plan_cues(
            shots,
            vertical=board.height > board.width,
            max_words=int(ctx.job.request.caption_max_words)
            if ctx.job.request.caption_max_words
            else None,
        )
        for shot in shots:
            shot.captions = [
                c for c in board.captions if shot.start - 0.05 <= c.start < shot.end
            ]
        # The retimed board is what the edit will render, so it *is* the storyboard
        # artifact from here on -- and a copy stays in generate/ for the audit trail.
        ctx.save("storyboard.retimed.json", board)
        ctx.save_to(StageName.STORYBOARD, "storyboard.json", board)

        with cf.ThreadPoolExecutor(max_workers=max(1, ctx.settings.max_workers)) as pool:
            visuals = list(pool.map(lambda shot: self._visual(ctx, shot, size), shots))

        assets: list[ShotAsset] = []
        rows = zip(shots, narration, visuals, strict=True)
        for shot, (audio, n_provider, n_dur), (video, v_provider, note) in rows:
            assets.append(
                ShotAsset(
                    shot_index=shot.index,
                    video_path=video,
                    is_still=bool(video and video.suffix.lower() in {".png", ".jpg", ".jpeg"}),
                    audio_path=audio,
                    duration=round(shot.duration, 3),
                    narration_provider=n_provider,
                    visual_provider=v_provider,
                    notes=note or (f"narration {n_dur:.2f}s" if n_dur else ""),
                )
            )
        total = shots[-1].end if shots else 0.0
        budget = float(ctx.job.request.target_seconds or board.duration)
        if total > budget * 1.12:
            ctx.emit(
                f"generated {total:.1f}s against a {budget:.0f}s budget "
                f"(+{total - budget:.1f}s) -- trim lines or raise the target",
                level="warning",
            )
        ctx.emit(f"{len(assets)} shots ready, {total:.1f}s total")
        return AssetSet(assets=assets, total_seconds=round(total, 3), budget_seconds=budget)

    # -- narration -------------------------------------------------------------

    def _narration(self, ctx: Any, shot: Shot) -> tuple[Path | None, str, float]:
        out = ctx.path(f"narration_{shot.index:03d}.mp3")
        if not shot.narration.strip():
            return None, "none", 0.0
        if ctx.tts is not None and ctx.tts.available:
            try:
                ctx.tts.synthesize(shot.narration, out)
                measured = ctx.ffmpeg.duration(out)
                ctx.save(f"narration_{shot.index:03d}.txt", shot.narration)
                return out, ctx.tts.backend, measured
            except Exception as exc:
                ctx.emit(f"tts failed on shot {shot.index} ({exc}); falling back to silence", level="warning")
        wav = ctx.path(f"narration_{shot.index:03d}.wav")
        seconds = estimate_speech_seconds(shot.narration, wpm=float(ctx.job.request.wpm or 150.0))
        ctx.ffmpeg.silence(wav, seconds)
        return wav, "silent", seconds

    def _retime(self, ctx: Any, shots: list[Shot], narration: list[tuple[Path | None, str, float]]) -> None:
        """Rebuild the timeline from measured audio, then re-cut captions to match.

        Shot durations come out summing exactly to the render length, with lead-in and
        breath kept *inside* the first/last shot. The assembly pass concatenates
        back-to-back, so any gap expressed between shots would silently shorten the
        picture and slide every caption out of sync.
        """
        from ..timing import apply, retime_from_audio

        measured: list[float | None] = []
        for audio, provider, seconds in narration:
            if provider == "silent" or audio is None:
                measured.append(None)  # fall back to the estimate used to size the file
            else:
                measured.append(ctx.ffmpeg.duration(audio) or seconds or None)
        lead_in = float(ctx.settings.lead_in)
        timings = retime_from_audio(
            shots,
            measured,
            wpm=float(ctx.job.request.wpm or 150.0),
            lead_in=lead_in,
            min_shot=_MIN_SHOT,
        )
        apply(shots, timings)
        ctx.emit(f"timeline set: {len(shots)} shots, {timings[-1].end:.2f}s out")

    # -- picture ---------------------------------------------------------------

    def _visual(self, ctx: Any, shot: Shot, size: tuple[int, int]) -> tuple[Path | None, str, str]:
        """Return ``(file, provider, note)``; ``file`` may be an mp4, an image or None."""
        spec = shot.visual
        kind = VisualKind(spec.kind) if not isinstance(spec.kind, VisualKind) else spec.kind
        out = ctx.path(f"visual_{shot.index:03d}.mp4")
        if out.exists() and ctx.use_cache():
            return out, "cached", "reused existing visual"

        if kind is VisualKind.SOURCE_CLIP and spec.source_start is not None:
            source = ctx.artifacts.source
            if source is not None and source.has_media:
                try:
                    ctx.ffmpeg.run(
                        [
                            "-ss", f"{spec.source_start:.3f}",
                            "-i", str(source.media_path),
                            "-t", f"{max(0.2, spec.source_end - spec.source_start):.3f}",
                            "-an", "-vf", f"fps={30}",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                            "-pix_fmt", "yuv420p", str(out),
                        ],
                        timeout=900,
                    )
                    return out, "source-cut", f"b-roll {spec.source_start:.1f}-{spec.source_end:.1f}s"
                except Exception as exc:
                    ctx.emit(f"source cut failed on shot {shot.index} ({exc}); card instead", level="warning")
            kind = VisualKind.CARD

        if kind is VisualKind.GENERATED and ctx.images is not None and ctx.images.available:
            png = ctx.path(f"visual_{shot.index:03d}.png")
            try:
                ctx.images.generate(spec.prompt, png)
                # Handed to the assembly pass as a still: encoding a looped image here
                # would be a wasted generation of x264 loss.
                return png, "image-gen", spec.prompt[:80]
            except Exception as exc:
                ctx.emit(f"image gen failed on shot {shot.index} ({exc}); card instead", level="warning")

        # CARD: no file at all. The assembly pass draws the background and the global
        # ASS carries the on-screen text, so there is nothing worth caching on disk.
        return None, "procedural-card", "no image provider; flat card + overlay text"
