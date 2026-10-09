"""Stage 7 — the edit. One ffmpeg invocation, one encode pass.

The whole timeline is assembled in a single filtergraph (see :mod:`ytpipe.assembly`):
per-shot normalise → concat → burn captions → loudness → encode. That is one
generation of x264 loss instead of two, and it removes the intermediate container
entirely, so there is no keyframe-alignment or stream-copy class of failure left to
hit on someone else's machine.

Loudness target is -14 LUFS integrated / -1.5 dBTP: what YouTube measures against.
Going louder does not make a video louder, it makes YouTube quieter and adds
clipping on their re-encode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..assembly import AssemblySpec, ShotInput, build_graph, encode_args
from ..captions import CaptionStyle, build_ass, to_srt
from ..errors import RenderError
from ..models import CaptionCue, CaptionFiles, RenderResult, StageName

_STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class EditStage:
    name = StageName.EDIT

    def run(self, ctx: Any) -> RenderResult:
        board = ctx.artifacts.storyboard
        assets = ctx.artifacts.assets
        if board is None or assets is None:
            raise RenderError("storyboard or assets missing before edit")
        by_index = {a.shot_index: a for a in assets.assets}
        notes: list[str] = []
        started = time.monotonic()

        transition = str(ctx.job.request.transition or "dip")
        if transition not in {"cut", "dip", "xfade"}:
            notes.append(f"unknown transition {transition!r}; using 'dip'")
            transition = "dip"

        caption_files = self._captions(ctx, board)
        shot_inputs: list[ShotInput] = []
        first_speech = float(getattr(board.shots[0], "speech_start", 0.0) or 0.0)
        for shot in board.shots:
            asset = by_index.get(shot.index)
            video = Path(asset.video_path) if asset and asset.video_path else None
            if video is not None and not video.exists():
                notes.append(f"shot {shot.index}: visual {video.name} missing; drawn as a card")
                video = None
            audio = Path(asset.audio_path) if asset and asset.audio_path else None
            if audio is not None and not audio.exists():
                notes.append(f"shot {shot.index}: narration {audio.name} missing; silence")
                audio = None
            shot_inputs.append(
                ShotInput(
                    index=shot.index,
                    seconds=max(0.25, shot.duration),
                    video=str(video) if video else None,
                    audio=str(audio) if audio else None,
                    background=shot.visual.background or "0x0b1220",
                    zoom=float(shot.visual.zoom or 1.0),
                    pan=str(shot.visual.pan or "none"),
                    still=bool(video and (asset.is_still or video.suffix.lower() in _STILL_SUFFIXES)),
                    audio_delay=first_speech if shot.index == board.shots[0].index else 0.0,
                )
            )

        music = Path(board.music) if board.music else None
        if music is not None and not music.exists():
            notes.append(f"music bed {music} not found; mixed without it")
            music = None

        spec = AssemblySpec(
            shots=shot_inputs,
            width=board.width,
            height=board.height,
            fps=board.fps,
            transition=transition,
            ass=str(caption_files.ass) if caption_files.ass and ctx.job.request.captions else None,
            music_path=str(music) if music else None,
            music_gain_db=float(ctx.job.request.music_gain_db),
            loudness_i=board.loudness_target,
            tail_seconds=max(0.0, float(ctx.settings.tail_seconds)),
        )
        graph, inputs = build_graph(spec)
        final = ctx.output("final.mp4")
        final.parent.mkdir(parents=True, exist_ok=True)
        defaults = ctx.settings.render_defaults()
        ctx.ffmpeg.run(
            [
                *inputs,
                "-filter_complex",
                graph,
                *encode_args(
                    spec,
                    str(final),
                    vmap="[vout]",
                    amap="[aout]",
                    crf=int(defaults["crf"]),
                    preset=str(defaults["preset"]),
                ),
            ],
            timeout=3600,
        )
        render_seconds = time.monotonic() - started

        info = ctx.ffmpeg.probe(final)
        if not final.exists() or final.stat().st_size < 1024:
            raise RenderError("ffmpeg produced no output")
        if not info.has_audio:
            notes.append("output has no audio stream")
        if info.duration and info.duration < 1.0:
            raise RenderError(f"rendered output is only {info.duration:.2f}s")
        drift = abs((info.duration or 0.0) - spec.total_seconds)
        if drift > 0.4:
            notes.append(
                f"rendered {info.duration:.2f}s vs planned {spec.total_seconds:.2f}s "
                f"({drift:.2f}s drift)"
            )

        thumb = ctx.output("thumb.jpg")
        try:
            ctx.ffmpeg.thumbnail(
                final,
                thumb,
                at=min(1.5, max(0.1, (info.duration or 1) * 0.06)),
                size=(board.width, board.height),
            )
        except Exception as exc:
            notes.append(f"thumbnail failed: {exc}")

        render = RenderResult(
            path=final,
            duration=round(info.duration or spec.total_seconds, 3),
            width=info.width or board.width,
            height=info.height or board.height,
            fps=int(round(info.fps or board.fps)),
            size_bytes=final.stat().st_size,
            video_codec=info.vcodec or "h264",
            audio_codec=info.acodec or "aac",
            captions=caption_files,
            thumbnail_path=thumb if thumb.exists() else None,
            shot_count=len(board.shots),
            render_seconds=round(render_seconds, 2),
            warnings=notes,
        )
        ctx.emit(
            f"{render.width}x{render.height} {render.duration:.2f}s "
            f"{render.size_bytes / 1e6:.2f} MB in {render.render_seconds:.1f}s"
        )
        for note in notes:
            ctx.emit(note, level="warning")
        return render

    # -- caption sidecars ------------------------------------------------------

    @staticmethod
    def _captions(ctx: Any, board: Any) -> CaptionFiles:
        """Write the ASS that drives the burn and an SRT for the platform's own captions.

        Two files, one cue list: the burned version is what viewers see when the video
        is muted (most of them), the SRT is uploaded so the platform can index the words
        and offer a toggle. Letting them disagree is the usual source of "the captions
        are out of sync" bug reports.
        """
        out_dir = ctx.output("final.srt").parent
        out_dir.mkdir(parents=True, exist_ok=True)
        srt, ass = out_dir / "final.srt", out_dir / "final.ass"
        vertical = board.height > board.width
        overlays = [
            CaptionCue(start=s.start, end=s.end, text=s.on_screen)
            for s in board.shots
            if s.on_screen and s.on_screen.strip()
        ]
        style = CaptionStyle(
            max_words=3 if vertical else 5,
            size=int(board.height * 0.041),
            margin_v=int(board.height * 0.17),
        )
        ass.write_text(
            build_ass(
                board.captions,
                width=board.width,
                height=board.height,
                style=style,
                overlays=overlays,
            ),
            encoding="utf-8",
        )
        srt.write_text(to_srt(board.captions), encoding="utf-8")
        return CaptionFiles(srt=srt, ass=ass)
