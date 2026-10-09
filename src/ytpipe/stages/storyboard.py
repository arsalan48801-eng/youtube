"""Stage 5 — turn a script into a shot list the renderer can execute.

Two decisions live here, and they are the whole reason the stage exists:

* **What picture goes under each line.** A line whose keywords match a moment in the
  source gets that moment as b-roll (re-cropped, audio stripped); everything else
  gets a generated still or a procedural card. Reusing the source's own best
  visuals is cheaper and closer to the original's rhythm than generating new ones.
* **Where captions break.** Cue boundaries are chosen here so the renderer and the
  SRT sidecar agree; `generate` recomputes them once narration audio exists.
"""

from __future__ import annotations

import re
from typing import Any

from ..captions import clean, plan_cues
from ..errors import PipelineError
from ..models import (
    Aspect,
    CaptionCue,
    Script,
    ScriptLine,
    Shot,
    StageName,
    Storyboard,
    VisualKind,
    VisualSpec,
)

_STOP = frozenset(
    """
    the a an and or but if then than that this those these of to in on for with as at by is
    are was were be been being you your it its we our they their i he she do does did not
    no so just
    """.split()
)
_MIN_SOURCE_SPAN = 1.2


def content_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", text) if w.lower() not in _STOP}


class StoryboardStage:
    name = StageName.STORYBOARD

    def run(self, ctx: Any) -> Storyboard:
        script: Script | None = ctx.artifacts.script
        if script is None or not script.lines:
            raise PipelineError("no script to storyboard", stage="storyboard")

        request = ctx.job.request
        aspect = Aspect(request.aspect) if not isinstance(request.aspect, Aspect) else request.aspect
        base_w, base_h = aspect.size
        # Draft renders: same graph, same timing, fewer pixels. Only the canvas moves,
        # so a 0.5 pass is a truthful preview of the real one.
        scale = max(0.2, min(1.0, float(ctx.settings.render_scale or 1.0)))
        width = max(16, int(base_w * scale) // 2 * 2)
        height = max(16, int(base_h * scale) // 2 * 2)
        if scale < 1.0:
            ctx.emit(f"draft canvas {width}x{height} (render_scale={scale})")
        fps = 30
        source = ctx.artifacts.source
        transcript = ctx.artifacts.transcript
        candidates = self._source_candidates(source, transcript)

        shots: list[Shot] = []
        used: set[tuple[float, float]] = set()  # source spans already spent
        for line in script.lines:
            visual = self._visual_for(
                ctx, line, script, candidates, span=line.duration, used=used
            )
            shots.append(
                Shot(
                    index=line.index,
                    start=line.start,
                    end=line.end,
                    # The speech window is where captions and the lead-in live. Dropping
                    # it here is what made the hook start flush with the picture.
                    speech_start=line.speech_start,
                    speech_end=line.speech_end,
                    narration=line.voiceover,
                    on_screen=line.on_screen,
                    visual=visual,
                )
            )

        cues = plan_cues(
            shots,
            vertical=height > width,
            max_words=int(request.caption_max_words) if request.caption_max_words else None,
        )
        self._attach(shots, cues)

        storyboard = Storyboard(
            aspect=aspect,
            fps=fps,
            width=width,
            height=height,
            shots=shots,
            captions=cues,
            music=str(request.music_path) if request.music_path else None,
        )
        ctx.emit(
            f"{len(shots)} shots over {storyboard.duration:.1f}s on {width}x{height}, "
            f"{len(cues)} caption cues"
        )
        return storyboard

    # -- visuals --------------------------------------------------------------

    def _visual_for(
        self,
        ctx: Any,
        line: ScriptLine,
        script: Script,
        candidates: list[tuple[float, float, set[str]]],
        *,
        span: float,
        used: set[tuple[float, float]],
    ) -> VisualSpec:
        words = content_words(line.voiceover)
        best: tuple[float, float, float] | None = None  # score, candidate start, end
        if candidates and words:
            for start, end, kw in candidates:
                overlap = len(words & kw) / max(1, len(words))
                if any(_overlaps(start, start + max(span, _MIN_SOURCE_SPAN), u) for u in used):
                    overlap -= 0.45  # still usable, but prefer fresh footage
                if best is None or overlap > best[0]:
                    best = (overlap, start, end)
        if best and best[0] >= 0.34:
            length = max(_MIN_SOURCE_SPAN, min(span + 0.4, ctx.settings.max_source_span))
            limit = float(source_duration(ctx)) or float("inf")
            start = _fresh_start(best[1], best[2], length, used, limit)
            end = min(start + length, limit) if limit != float("inf") else start + length
            if end - start < _MIN_SOURCE_SPAN:
                start, end = 0.0, min(length, limit)
            used.add((start, end))
            return VisualSpec(
                kind=VisualKind.SOURCE_CLIP,
                source_start=round(start, 3),
                source_end=round(end, 3),
                zoom=1.0,
                pan=_pan_for(line.index),
                prompt=line.visual,
            )

        prompt = line.visual or line.voiceover
        if ctx.images is not None and ctx.images.available:
            return VisualSpec(
                kind=VisualKind.GENERATED,
                prompt=_image_prompt(prompt, script),
                zoom=1.08,
                pan=_pan_for(line.index),
            )
        return VisualSpec(
            kind=VisualKind.CARD,
            prompt=clean(line.on_screen or prompt)[:80],
            zoom=1.0,
            pan="none",
            background=_card_background(line.index, line.role),
        )

    @staticmethod
    def _source_candidates(source: Any, transcript: Any) -> list[tuple[float, float, set[str]]]:
        """Windows of source footage indexed by the words spoken over them.

        Chunked to a few seconds of speech rather than taken per caption cue, because
        an auto-caption track has one cue per word and a human upload has one per
        paragraph -- matching a script line against a single word would pick footage
        by accident.
        """
        if source is None or not source.has_media or transcript is None:
            return []
        out: list[tuple[float, float, set[str]]] = []
        buf: list[str] = []
        start = end = 0.0
        for seg in transcript.segments:
            if not buf:
                start = seg.start
            buf.extend(seg.text.split())
            end = seg.end
            if len(buf) >= 10 or (end - start) >= 4.0:
                words = content_words(" ".join(buf))
                if words:
                    out.append((start, end, words))
                buf = []
        if buf and content_words(" ".join(buf)):
            out.append((start, end, content_words(" ".join(buf))))
        return out
    @staticmethod
    def _attach(shots: list[Shot], cues: list[CaptionCue]) -> None:
        for shot in shots:
            shot.captions = [c for c in cues if c.start >= shot.start - 0.05 and c.start < shot.end]


def source_duration(ctx: Any) -> float:
    source = ctx.artifacts.source
    return float(source.duration) if source is not None and source.duration else 0.0


def _fresh_start(
    cand_start: float, cand_end: float, length: float, used: set[tuple[float, float]], limit: float
) -> float:
    """Advance past footage already cut, inside the same matched passage.

    Two lines matching the same passage should still not show the identical frames
    back to back: the *passage* is what matched, the exact window is negotiable.
    """
    start = cand_start
    for _ in range(len(used) + 1):
        clash = [hi for (lo, hi) in used if min(start + length, hi) - max(start, lo) > 0.25 * length]
        if not clash:
            return min(start, max(0.0, limit - length)) if limit != float("inf") else start
        start = max(clash) + 0.1
        if limit != float("inf") and start + length > limit:
            return cand_start  # nowhere left to go: overlap is better than a blank shot
    return cand_start


def _overlaps(a: float, b: float, used: tuple[float, float]) -> bool:
    lo, hi = used
    return min(b, hi) - max(a, lo) > 0.25 * (b - a)


def _pan_for(index: int) -> str:
    return ["left-right", "up-down", "right-left", "down-up"][index % 4]


def _card_background(index: int, role: str) -> str:
    if role == "hook":
        return "0x0b1220"
    if role == "cta":
        return "0x123a2a"
    palette = ["0x101826", "0x1a1030", "0x0d2430", "0x241018"]
    return palette[index % len(palette)]


def _image_prompt(visual: str, script: Script) -> str:
    return (
        f"{visual.strip().rstrip('.')}. Vertical 9:16 cinematic still for a short-form video "
        "titled "
        f"'{script.title[:70]}'. Photorealistic, shallow depth of field, motivated light, "
        "no text, no watermark, no logos, no captions."
    )
