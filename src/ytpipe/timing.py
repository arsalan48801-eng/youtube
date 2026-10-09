"""Timeline maths, in one place.

The writer needs provisional timings for lines, the generate stage needs real timings
once narration audio exists, and the caption planner needs to know where the *speech*
sits inside each shot. All three must agree, or burned-in captions drift against the
picture by the accumulated padding -- which is the single most common defect in
auto-edited video and the kind of bug that is invisible in a unit test on one stage.

So: one function, two callers, and the rule it encodes is that **shot durations sum
exactly to the output timeline** (no gaps), with lead-in and breath handled as padding
inside the shot rather than as empty space between shots. A `concat` filter throws away
inter-shot gaps; it cannot throw away padding.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .providers.tts import estimate_speech_seconds

#: Silence kept after the last word, so a cut never lands on a syllable.
TAIL_PAD = 0.22
#: A shot shorter than this reads as a flash, not a cut.
MIN_SHOT = 1.15


@dataclass
class Timing:
    start: float
    end: float
    speech_start: float
    speech_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan(
    texts: list[str],
    *,
    wpm: float = 150.0,
    lead_in: float = 0.35,
    tail_pad: float = TAIL_PAD,
    min_shot: float = MIN_SHOT,
) -> list[Timing]:
    """Lay out consecutive shots from their narration text, gaplessly.

    ``lead_in`` is absorbed into the *first* shot (its audio is delayed by the same
    amount at render time), not inserted as empty space before it.
    """
    out: list[Timing] = []
    t = 0.0
    for i, text in enumerate(texts):
        spoken = estimate_speech_seconds(text, wpm=wpm, padding=0.0)
        head = lead_in if i == 0 else 0.0
        duration = max(min_shot, head + spoken + tail_pad)
        start, end = t, round(t + duration, 3)
        out.append(
            Timing(
                start=round(start, 3),
                end=end,
                speech_start=round(start + head, 3),
                speech_end=round(min(end, start + head + spoken), 3),
            )
        )
        t = end
    return out


def apply(items: Iterable[Any], timings: list[Timing]) -> None:
    """Write a layout back onto models that expose ``start``/``end`` (and optionally
    ``speech_start``/``speech_end``)."""
    for item, timing in zip(items, timings, strict=True):
        item.start = timing.start
        item.end = timing.end
        if hasattr(item, "speech_start"):
            item.speech_start = timing.speech_start
            item.speech_end = timing.speech_end


def retime_from_audio(
    items: list[Any],
    measured: list[float | None],
    *,
    wpm: float = 150.0,
    lead_in: float = 0.35,
    tail_pad: float = TAIL_PAD,
    min_shot: float = MIN_SHOT,
) -> list[Timing]:
    """Replace estimated speech lengths with measured audio lengths.

    A shot whose narration file is 3.1s long must be at least 3.1s of picture; using
    the estimate instead is how auto-edits end up talking over the next scene.
    """
    out: list[Timing] = []
    t = 0.0
    for i, (item, seconds) in enumerate(zip(items, measured, strict=True)):
        estimated = estimate_speech_seconds(item.narration or "", wpm=wpm, padding=0.0)
        spoken = seconds if seconds and seconds > 0 else estimated
        head = lead_in if i == 0 else 0.0
        duration = max(min_shot, head + float(spoken) + tail_pad)
        start, end = t, round(t + duration, 3)
        out.append(
            Timing(
                start=round(start, 3),
                end=end,
                speech_start=round(start + head, 3),
                speech_end=round(min(end, start + head + float(spoken)), 3),
            )
        )
        t = end
    return out
