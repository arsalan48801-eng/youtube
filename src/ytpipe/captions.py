"""Caption generation: transcript cues -> punchy, animated ASS + a plain SRT sidecar.

Why ASS instead of ``drawtext``: the ffmpeg build we can ship from PyPI has libass
but not drawtext, and more importantly ASS gives us per-word highlight timing
(``\\k``), pop-in animation (``\\fad``) and a boxed background from one stylesheet
that the whole render inherits. Re-cutting a video's caption *feel* is then a
template change, not a filtergraph rewrite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .ffmpeg import ass_time
from .models import CaptionCue

_TAG_RE = re.compile(r"<[^>]+>|\{[^}]*\}")
_WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    """Strip VTT/HTML tags, speaker labels and filler that hurts readability."""
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"^\s*>>\s*", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return _WS_RE.sub(" ", text).strip()


@dataclass
class CaptionStyle:
    """Caption look. Tuned for 9:16; scaled to whatever canvas it is handed."""

    font: str = "DejaVu Sans"
    size: int = 78
    primary: str = "&H00FFFFFF"
    highlight: str = "&H0000E7FF"
    outline: str = "&H00101010"
    box: str = "&H99000000"
    alignment: int = 2  # bottom-centre
    margin_v: int = 330
    outline_width: int = 6
    shadow: int = 0
    bold: bool = True
    max_words: int = 4
    max_seconds: float = 2.4
    min_seconds: float = 0.55
    gap: float = 0.02
    pop_in_ms: int = 90
    pop_out_ms: int = 70

    def for_canvas(self, height: int) -> CaptionStyle:
        """Keep the caption block a constant fraction of frame height."""
        factor = height / 1920.0
        if abs(factor - 1.0) < 0.01:
            return self
        return CaptionStyle(
            **{
                **self.__dict__,
                "size": max(24, int(self.size * factor)),
                "margin_v": int(self.margin_v * factor),
                "outline_width": max(2, int(self.outline_width * factor)),
            }
        )


def regroup(
    cues: list[CaptionCue],
    *,
    max_words: int = 4,
    max_seconds: float = 2.4,
    min_seconds: float = 0.55,
) -> list[CaptionCue]:
    """Re-chunk long caption lines into short on-screen bursts.

    Long lines either wrap off-frame on a phone or force a tiny font. We pack words
    greedily up to ``max_words`` while the group still fits ``max_seconds``, then let
    a trailing short group stretch to ``min_seconds`` so it does not flash.
    """
    out: list[CaptionCue] = []
    buf_words: list[str] = []
    start = end = 0.0

    def flush(final: bool = False) -> None:
        nonlocal buf_words, start, end
        if not buf_words:
            return
        s, e = start, end
        if final and (e - s) < min_seconds:
            e = s + min_seconds
        out.append(
            CaptionCue(
                start=round(s, 3),
                end=round(e, 3),
                text=" ".join(buf_words),
                words=list(buf_words),
            )
        )
        buf_words = []

    for cue in cues:
        text = clean(cue.text)
        if not text:
            continue
        words = cue.words or text.split()
        span = max(0.05, cue.end - cue.start)
        per_word = span / max(1, len(words))
        for i, word in enumerate(words):
            w_start = cue.start + i * per_word
            w_end = cue.start + (i + 1) * per_word
            if not buf_words:
                start, end = w_start, w_end
                buf_words.append(word)
                continue
            too_many = len(buf_words) >= max_words
            too_long = (w_end - start) > max_seconds
            if too_many or too_long:
                flush()
                start, end = w_start, w_end
                buf_words = [word]
            else:
                buf_words.append(word)
                end = w_end
        if _ends_sentence(text):
            # A finished sentence is a hard boundary. Packing the next line's opening
            # words onto it ("matters. Three") reads as a typo even when the timing
            # is right, and it hides the beat a cut would normally land on.
            flush()
    flush(final=True)
    return _dedupe_overlap(out)


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


def _dedupe_overlap(cues: list[CaptionCue]) -> list[CaptionCue]:
    """Enforce a monotonic timeline with a breathing gap so cues never blink."""
    out: list[CaptionCue] = []
    for cue in cues:
        if out and cue.start < out[-1].end:
            out[-1].end = round(max(out[-1].start + 0.12, cue.start - 0.02), 3)
        out.append(cue.model_copy(deep=True))
        out[-1].end = max(out[-1].end, out[-1].start + 0.2)
    return out


def plan_cues(
    shots: list[Any],
    *,
    vertical: bool = True,
    max_words: int | None = None,
    min_seconds: float = 0.5,
) -> list[CaptionCue]:
    """Split each shot's narration into on-screen bursts.

    Vertical canvases read 3 words comfortably at 9:16 with a boxed background; a
    wide canvas fits 5. Getting this wrong is the most common reason captions look
    amateur, so the rule lives here rather than in each stage.
    """
    words = max_words or (3 if vertical else 5)
    return regroup(
        [
            CaptionCue(
                start=getattr(s, "speech_start", 0.0) or s.start,
                end=getattr(s, "speech_end", 0.0) or s.end,
                text=s.narration,
            )
            for s in shots
            if s.narration
        ],
        max_words=words,
        max_seconds=min(2.6, max(1.1, words * 0.42)),
        min_seconds=min_seconds,
    )


def to_srt(cues: list[CaptionCue]) -> str:
    blocks = []
    for i, cue in enumerate(cues, start=1):
        h1, m1, s1 = _srt_parts(cue.start)
        h2, m2, s2 = _srt_parts(cue.end)
        blocks.append(f"{i}\n{h1}:{m1}:{s1} --> {h2}:{m2}:{s2}\n{cue.text}\n")
    return "\n".join(blocks)


def _srt_parts(seconds: float) -> tuple[str, str, str]:
    total = max(0.0, seconds)
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    rest = total % 60
    return f"{h:02d}", f"{m:02d}", f"{rest:06.3f}".replace(".", ",")


def to_ass(cues: list[CaptionCue], *, width: int, height: int, style: CaptionStyle) -> str:
    """ASS with per-word ``\\k`` highlighting so the caption tracks the voiceover."""
    style = style.for_canvas(height)
    head = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\nYCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: CAP,{style.font},{style.size},{style.primary},{style.highlight},"
        f"{style.outline},{style.box},{-1 if style.bold else 0},0,0,0,100,100,0.6,0,4,"
        f"{style.outline_width},{style.shadow},{style.alignment},42,42,{style.margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for cue in cues:
        dur = max(0.2, cue.end - cue.start)
        words = cue.words or cue.text.split()
        if not words:
            continue
        # Uniform \k units are centiseconds; distribute them across the words so the
        # highlight lands with the syllable rate rather than at the cue midpoint.
        units = max(len(words), int(round(dur * 100)))
        base, extra = divmod(units, len(words))
        timing, acc = [], 0
        for i in range(len(words)):
            n = base + (1 if i < extra else 0)
            acc += n
            timing.append(n)
        body = "".join(
            f"{{\\kf{n}}}{_escape(word)}" for word, n in zip(words, timing, strict=True)
        )
        fade = f"{{\\fad({style.pop_in_ms},{style.pop_out_ms})}}"
        events.append(
            f"Dialogue: 0,{ass_time(cue.start)},{ass_time(cue.end)},CAP,,0,0,0,,"
            f"{fade}{body}"
        )
    return head + "\n".join(events) + "\n"


def _escape(word: str) -> str:
    return word.replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(
    cues: list[CaptionCue],
    *,
    width: int,
    height: int,
    style: CaptionStyle,
    overlays: list[CaptionCue] | None = None,
) -> str:
    """One ASS document: spoken captions at the bottom, key phrases up top.

    Two layers because they do different jobs. Captions are for sound-off viewers
    and follow the voice word by word; the overlay is a 1-4 word emphasis card that
    lives where it cannot collide with them. Putting both on one timeline means a
    single burn pass and no z-fighting over the safe area.
    """
    style = style.for_canvas(height)
    longest = max((len(c.text) for c in (overlays or [])), default=8)
    # Fit the widest overlay to the canvas instead of letting libass clip it: a 3-word
    # phrase at 5.5% of height overflows 1080px, and a clipped title looks broken.
    pop_size = min(
        int(height * 0.055),
        max(int(height * 0.028), int((width * 0.86) / max(6.0, longest * 0.56))),
    )
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\nYCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: CAP,{style.font},{style.size},{style.primary},{style.highlight},"
        f"{style.outline},{style.box},{-1 if style.bold else 0},0,0,0,100,100,0.6,0,4,"
        f"{style.outline_width},{style.shadow},{style.alignment},42,42,{style.margin_v},1\n"
        f"Style: POP,{style.font},{pop_size},&H00FFFFFF,&H0000E7FF,&H00000000,&HE0000000,"
        f"-1,0,0,0,100,100,1.5,0,3,4,0,8,60,60,{max(90, int(height * 0.075))},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [_caption_event(c, style) for c in cues]
    for cue in overlays or []:
        text = _escape(cue.text)
        if not text:
            continue
        events.append(
            (
                cue.start,
                f"Dialogue: 1,{ass_time(cue.start)},{ass_time(cue.end)},POP,,0,0,0,,"
                f"{{\\fad(120,90)}}{{\\t(0,140,\\fscx112\\fscy112)}}{{\\t(140,260,"
                f"\\fscx100\\fscy100)}}{text}",
            )
        )
    events.sort(key=lambda row: row[0])
    return header + "\n".join(line for _t, line in events) + "\n"


def _caption_event(cue: CaptionCue, style: CaptionStyle) -> tuple[float, str]:
    dur = max(0.2, cue.end - cue.start)
    words = cue.words or cue.text.split()
    units = max(len(words), int(round(dur * 100)))
    base, extra = divmod(units, max(1, len(words)))
    timings = [base + (1 if i < extra else 0) for i in range(len(words))]
    body = "".join(f"{{\\kf{n}}}{_escape(word)} " for word, n in zip(words, timings, strict=True)).rstrip()
    fade = f"{{\\fad({style.pop_in_ms},{style.pop_out_ms})}}"
    return (
        cue.start,
        f"Dialogue: 0,{ass_time(cue.start)},{ass_time(cue.end)},CAP,,0,0,0,,{fade}{body}",
    )
