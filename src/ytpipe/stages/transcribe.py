"""Stage 2 — a clean, word-timed transcript.

Segment granularity follows the source: a sentence-per-cue upload track stays that
way, and an auto-caption track with per-word tags keeps one segment per word, because
throwing the word boundaries away is what makes burned-in captions drift. The
analyser assembles sentences from whatever it is given.

Preference order, cheapest first:
1. the caption track that came with the download (free, human-quality when present),
2. local ASR via faster-whisper if the operator installed it,
3. nothing -- an empty transcript still lets the pipeline write a script from the
   title/description brief, and that is better than a hard failure.

YouTube's auto-caption VTT repeats the previous line in every cue and tags each
word with a timestamp. Both are handled here, because downstream caption
re-timing depends on word boundaries being real.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..captions import clean
from ..errors import TranscribeError
from ..models import CaptionSegment, StageName, Transcript

_TIME_RE = re.compile(r"(?:(\d{2}):)?(\d{2}):(\d{2})[.,](\d{1,3})")
_CUE_RE = re.compile(r"(\d{2}:)?\d{2}:\d{2}[.,]\d{1,3}\s*-->\s*(\d{2}:)?\d{2}:\d{2}[.,]\d{1,3}")
_WORD_TAG_RE = re.compile(r"<((?:\d{2}:)?\d{2}:\d{2}[.,]\d{1,3})>")


def _secs(token: str) -> float:
    m = _TIME_RE.search(token)
    if not m:
        return 0.0
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int((ms or "0").ljust(3, "0")) / 1000.0


def _unescape(text: str) -> str:
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&gt;", ">"), ("&lt;", "<"), ("\\N", " ")):
        text = text.replace(a, b)
    return text


def parse_webvtt(raw: str) -> list[CaptionSegment]:
    """Cue-level parse, with word timestamps flattened into their own segments."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", raw)
    segments: list[CaptionSegment] = []
    previous_tail = ""
    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "Kind:", "Language:", "STYLE", "::")):
            lines = lines[1:] if lines else lines
            if not lines:
                continue
        cue_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if cue_idx is None:
            continue
        header = lines[cue_idx]
        try:
            left, right = header.split("-->", 1)
        except ValueError:  # pragma: no cover - guarded by cue_idx search
            continue
        start, end = _secs(left.strip().split()[-1]), _secs(right.strip().split()[0])
        if end <= start:
            end = start + 1.2
        body = " ".join(_unescape(ln) for ln in lines[cue_idx + 1 :]).strip()
        if not body:
            continue

        if tags := _WORD_TAG_RE.findall(body):
            words = [w for w in _WORD_TAG_RE.sub("", body).split() if w]
            times = [_secs(t) for t in tags[: len(words)]]
            for i, word in enumerate(words):
                w_start = times[i] if i < len(times) else start
                w_end = times[i + 1] if i + 1 < len(times) else end
                segments.append(
                    CaptionSegment(
                        start=round(w_start, 3),
                        end=round(max(w_start + 0.08, w_end), 3),
                        text=word,
                    )
                )
            continue

        text = re.sub(r"<[^>]+>", " ", body)
        text = clean(text)
        # Rolling captions: cue N repeats the tail of cue N-1. Keep only the delta.
        if previous_tail and text and (text.startswith(previous_tail) or text in previous_tail):
            text = text[len(previous_tail) :].strip() if text.startswith(previous_tail) else ""
        if not text:
            continue
        previous_tail = (previous_tail + " " + text).strip()[-90:]
        segments.append(CaptionSegment(start=round(start, 3), end=round(end, 3), text=text))
    return _dedupe(segments)


def parse_srt(raw: str) -> list[CaptionSegment]:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    segments: list[CaptionSegment] = []
    for block in re.split(r"\n{2,}", raw.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if "-->" not in lines[0]:
            continue
        left, right = lines[0].split("-->", 1)
        start, end = _secs(left.strip()), _secs(right.strip())
        text = clean(re.sub(r"<[^>]+>", " ", _unescape(" ".join(lines[1:]))))
        if text:
            segments.append(CaptionSegment(start=round(start, 3), end=round(end, 3), text=text))
    return _dedupe(segments)


def _dedupe(segments: list[CaptionSegment]) -> list[CaptionSegment]:
    out: list[CaptionSegment] = []
    for seg in segments:
        if out and out[-1].text == seg.text and abs(out[-1].start - seg.start) < 0.01:
            out[-1].end = max(out[-1].end, seg.end)
            continue
        out.append(seg.model_copy(deep=True))
    return out


class TranscribeStage:
    name = StageName.TRANSCRIBE

    def run(self, ctx: Any) -> Transcript:
        source = ctx.artifacts.source
        if source is None:
            raise TranscribeError("fetch stage produced no source")

        if source.subtitle_path and Path(source.subtitle_path).exists():
            path = Path(source.subtitle_path)
            raw = path.read_text(encoding="utf-8", errors="replace")
            segs = parse_srt(raw) if path.suffix.lower() == ".srt" else parse_webvtt(raw)
            if segs:
                ctx.emit(f"parsed {len(segs)} cues from {path.name}")
                return Transcript(
                    source_id=source.video_id,
                    language=source.language or "en",
                    provider=f"caption:{path.suffix.lstrip('.')}",
                    segments=segs,
                )
            ctx.emit(f"{path.name} parsed to zero cues", level="warning")

        if source.media_path and ctx.settings.asr_enabled:
            transcript = self._asr(ctx, Path(source.media_path))
            if transcript is not None:
                return transcript

        ctx.emit(
            "no caption track and no local ASR; continuing with an empty transcript "
            "(install faster-whisper or set YTPIPE_ASR_ENABLED=1 for audio transcription)",
            level="warning",
        )
        return Transcript(source_id=source.video_id, provider="none", segments=[])

    def _asr(self, ctx: Any, media: Path) -> Transcript | None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return None
        audio = ctx.path("audio-for-asr.wav")
        ctx.ffmpeg.extract_audio(media, audio)
        ctx.emit(f"transcribing with whisper:{ctx.settings.asr_model}")
        model = WhisperModel(ctx.settings.asr_model, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(audio), word_timestamps=True, vad_filter=True)
        out: list[CaptionSegment] = []
        for seg in segments:
            words = list(seg.words or [])
            if words:
                for i, word in enumerate(words):
                    nxt = words[i + 1].start if i + 1 < len(words) else seg.end
                    out.append(
                        CaptionSegment(
                            start=round(float(word.start), 3),
                            end=round(max(float(word.start) + 0.08, float(nxt)), 3),
                            text=word.word.strip(),
                        )
                    )
            else:
                out.append(
                    CaptionSegment(
                        start=round(float(seg.start), 3),
                        end=round(float(seg.end), 3),
                        text=seg.text.strip(),
                    )
                )
        return Transcript(
            source_id=ctx.artifacts.source.video_id if ctx.artifacts.source else None,
            provider=f"whisper:{ctx.settings.asr_model}",
            language=_info.language if hasattr(_info, "language") else None,
            segments=_dedupe(out),
        )
