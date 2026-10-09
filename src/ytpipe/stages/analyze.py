"""Stage 3 — read the source like an editor, not a summariser.

The point is not "what was this video about" (a summariser answers that and every
generated video then feels like a book report). The point is *why it held people*:
line length, cadence, where the first payoff lands, how often it says "you".
Those become numeric constraints the writer must respect.

The heuristics below always run and are deterministic; an LLM, when configured,
only refines the beat labelling and the qualitative notes. A paid API is never a
hard dependency for structure.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from ..errors import PipelineError
from ..models import Beat, Brief, StageName, Transcript

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers herself him himself his how i if
    in into is it its itself just me more most my myself no nor not of off on once only or
    other ought our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with you your
    yours yourself yourselves one two three first second third really actually kind things
    thing going get got lets let's gonna want need much many still even always never lot
    bit okay ok yeah see make made take put give back well right good new like would could
    """.split()
)
_FILLER = re.compile(r"\b(us|um+|uh+|like|you know|i mean|sort of|kind of)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PROMO_RE = re.compile(
    r"\b(welcome back|smash the like|subscribe|like and comment|link in bio|check out my|"
    r"merch|use my code|thanks for watching|patreon)\b",
    re.I,
)
_NUM_RE = re.compile(r"\b\d[\d,.]*\s?(%|percent|x|k|m|bn|billion|million|fps|gb|mb|ms|w|kw|times)?\b", re.I)


def syllables(word: str) -> int:
    """Cheap English syllable estimate; good enough for a readability index."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    vowel_groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")):
        vowel_groups -= 1
    return max(1, vowel_groups)


def flesch_kincaid(sentences: list[str]) -> float:
    words = [w for s in sentences for w in re.findall(r"[A-Za-z']+", s)]
    if not words or not sentences:
        return 0.0
    syl = sum(syllables(w) for w in words)
    return round(0.39 * (len(words) / len(sentences)) + 11.8 * (syl / len(words)) - 15.59, 2)


def keywords(text: str, *, limit: int = 18) -> list[str]:
    """Frequent non-stopword unigrams, then bigrams that survived the same filter."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS and not w.isdigit())
    if not counts:
        return []
    top = [w for w, _ in counts.most_common(limit)]
    grams = Counter()
    for a, b in zip(words, words[1:], strict=False):  # deliberate pairwise bigram scan
        if a in counts and b in counts and counts[a] > 1 and counts[b] > 1:
            grams[f"{a} {b}"] += 1
    strong = [g for g, n in grams.most_common(6) if n >= 3]
    out: list[str] = []
    for item in top + strong:
        if item not in out:
            out.append(item)
    return out[: limit + 6]


def sentences_of(transcript: Transcript) -> list[tuple[float, float, str]]:
    """(start, end, text) per sentence, assembled from cue boundaries."""
    out: list[tuple[float, float, str]] = []
    buf: list[str] = []
    start: float | None = None
    end = 0.0
    for seg in transcript.segments:
        if start is None:
            start = seg.start
        end = seg.end
        for piece in _SENT_SPLIT.split(seg.text):
            if not piece.strip():
                continue
            buf.append(piece.strip())
            if piece.rstrip().endswith((".", "!", "?")):
                out.append((start or 0.0, end, " ".join(buf).strip()))
                buf = []
                start = None
    if buf and out:
        out[-1] = (out[-1][0], end, (out[-1][2] + " " + " ".join(buf)).strip())
    elif buf:
        out.append((0.0, end, " ".join(buf).strip()))
    return [(s, e, t) for s, e, t in out if len(t.split()) >= 2]


def classify_hook(text: str) -> str:
    low = text.lower()
    if "?" in text:
        return "question"
    if _NUM_RE.search(text):
        return "stat"
    if re.search(r"\b(stop|never|don't|nobody|wrong|mistake|lie|secret|myth)\b", low):
        return "contrarian"
    if re.search(r"\b(how|why|what)\b", low):
        return "curiosity"
    if re.search(r"\b(i (spent|tested|tried|built|bought)|after \d)\b", low):
        return "experiment"
    return "statement"


def tone_of(transcript_text: str, *, question_ratio: float, second_person: float, contrarian: float) -> str:
    if contrarian > 0.04:
        return "contrarian"
    if question_ratio > 0.22:
        return "conversational"
    if second_person > 0.12:
        return "instructional"
    if re.search(r"\b(data|study|research|evidence|benchmark|compared|results)\b", transcript_text.lower()):
        return "analytical"
    return "documentary"


def mine_sentences(
    sentences: list[tuple[float, float, str]], kw: Counter[str], *, limit: int = 14
) -> list[str]:
    """Pick the lines worth keeping: keyword-dense, not too long, not intro filler."""
    scored: list[tuple[float, int, str]] = []
    for i, (_start, _end, text) in enumerate(sentences):
        words = re.findall(r"[A-Za-z']+", text.lower())
        useful = [w for w in words if w not in STOPWORDS]
        if not useful or len(words) < 5 or len(words) > 34:
            continue
        if _PROMO_RE.search(text):
            # Never a "fact": an outro asked as a favour carries no information and
            # putting it in the new script reads like the channel is begging.
            continue
        density = sum(kw.get(w, 0) for w in useful) / max(1, len(words))
        # Concrete facts outrank vibes: numbers, names, comparatives.
        bonus = 0.35 if _NUM_RE.search(text) else 0.0
        bonus += 0.15 if re.search(r"\b(because|so that|which means|the reason)\b", text.lower()) else 0.0
        penalty = 0.3 if i == 0 else 0.0  # "welcome back to the channel"
        penalty += 0.2 if re.search(r"\b(subscribe|like and comment|smash)\b", text.lower()) else 0.0
        scored.append((density + bonus - penalty, i, text))
    scored.sort(key=lambda row: (-row[0], row[1]))
    chosen = sorted(scored[:limit], key=lambda row: row[1])
    return [t for _s, _i, t in chosen]


class AnalyzeStage:
    name = StageName.ANALYZE

    def run(self, ctx: Any) -> Brief:
        transcript = ctx.artifacts.transcript
        source = ctx.artifacts.source
        if transcript is None:
            raise PipelineError("transcript missing", stage="analyze")

        sentences = sentences_of(transcript)
        text = transcript.text or (source.description if source else "")
        # `tokens` is the pacing unit: a numeral is spoken just like a word, so
        # excluding "120" from a word count would under-budget the read.
        tokens = text.split()
        words = re.findall(r"[A-Za-z']+", text.lower())
        duration = max(1.0, transcript.duration or (source.duration if source else 1.0))
        question_count = sum(1 for _s, _e, t in sentences if "?" in t)
        second_person = sum(1 for w in words if w in {"you", "your", "yours", "yourself"})
        contrarian = sum(1 for w in words if w in {"stop", "never", "wrong", "myth", "lie", "mistake"})
        kw_counts = Counter(w for w in words if w not in STOPWORDS)
        kw_list = keywords(text)

        brief = Brief(
            title=(source.title if source else "") or "Untitled",
            topic=ctx.job.request.topic or (source.title if source else "") or "the topic",
            words=len(tokens),
            wpm=round(len(tokens) / (duration / 60.0), 1) if duration else 150.0,
            sentence_count=len(sentences),
            avg_sentence_words=round(len(tokens) / max(1, len(sentences)), 2),
            question_ratio=round(question_count / max(1, len(sentences)), 3),
            second_person_ratio=round(second_person / max(1, len(words)), 4),
            readability=flesch_kincaid([t for _s, _e, t in sentences]),
            keywords=kw_list,
            beats=self._beats(sentences, duration),
            top_sentences=mine_sentences(sentences, kw_counts),
            sentence_durations=[round(e - s, 3) for s, e, _t in sentences],
            transcript_covered=bool(sentences),
        )
        brief.hook_line = sentences[0][2] if sentences else ""
        brief.hook_seconds = round(sentences[0][1] - sentences[0][0], 2) if sentences else 0.0
        brief.hook_style = classify_hook(brief.hook_line) if sentences else "statement"
        brief.tone = tone_of(
            text,
            question_ratio=brief.question_ratio,
            second_person=brief.second_person_ratio,
            contrarian=contrarian / max(1, len(words)),
        )
        brief.cta_line = self._cta(sentences)
        brief.retention_notes = self._notes(brief, duration)

        if ctx.llm is not None and ctx.llm.available:
            try:
                brief = self._augment(ctx, brief, sentences[:80])
            except Exception as exc:  # structure already exists; enrichment is optional
                ctx.emit(f"llm analysis skipped ({exc})", level="warning")

        ctx.emit(
            f"brief: {brief.words}w, {brief.wpm:.0f} wpm, {brief.sentence_count} lines, "
            f"hook={brief.hook_style} ({brief.hook_seconds:.1f}s), tone={brief.tone}, "
            f"FK grade {brief.readability}"
        )
        return brief

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _beats(sentences: list[tuple[float, float, str]], duration: float) -> list[Beat]:
        """Segment the source into narratively-labelled beats.

        Roles come from position, which is how retention actually behaves on short
        content: hook lands in the first seconds, payoff must appear before the
        mid-point, the CTA only ever lives at the tail.
        """
        if not sentences:
            return []
        target = max(4, min(12, int(duration // 12) + 2))
        chunk = max(1, math.ceil(len(sentences) / target))
        beats: list[Beat] = []
        for i in range(0, len(sentences), chunk):
            span = sentences[i : i + chunk]
            start, end = span[0][0], span[-1][1]
            body = " ".join(t for _s, _e, t in span)
            first = len(beats) == 0
            last = end >= sentences[-1][1] - 0.01
            if first:
                role = "hook"
            elif last and re.search(r"\b(subscribe|follow|comment|link|next video)\b", body.lower()):
                role = "cta"
            elif len(beats) == 1:
                role = "setup"
            elif last or i + chunk >= len(sentences):
                role = "payoff"
            else:
                role = "body"
            beats.append(
                Beat(
                    start=round(start, 3),
                    end=round(end, 3),
                    role=role,
                    summary=body[:240],
                )
            )
        return beats

    @staticmethod
    def _cta(sentences: list[tuple[float, float, str]]) -> str:
        tail = sentences[-4:] if len(sentences) > 4 else sentences
        for _s, _e, text in reversed(tail):
            if re.search(r"\b(subscribe|follow|comment|watch this|check out|link in)\b", text.lower()):
                return text
        return ""

    @staticmethod
    def _notes(brief: Brief, duration: float) -> list[str]:
        notes: list[str] = []
        if brief.hook_seconds and brief.hook_seconds <= 2.5:
            notes.append(f"hook is over in {brief.hook_seconds:.1f}s - keep line 1 under 10 words")
        if brief.avg_sentence_words:
            notes.append(
                f"source averages {brief.avg_sentence_words:.1f} words per line; "
                "match that or go shorter"
            )
        if brief.wpm:
            notes.append(f"narration pace ~{brief.wpm:.0f} wpm")
        if brief.question_ratio > 0.15:
            notes.append(f"{brief.question_ratio:.0%} of lines are questions - ask at least twice")
        if brief.second_person_ratio > 0.05:
            notes.append("address the viewer directly (you/your) in most lines")
        if not brief.cta_line:
            notes.append("source had no explicit CTA; add one only at the very end")
        if duration and duration < 25:
            notes.append("source is very short: single idea, no section breaks")
        return notes

    def _augment(self, ctx: Any, brief: Brief, sentences: list[tuple[float, float, str]]) -> Brief:
        system = (
            "You are a short-form video editor. Given a transcript, return JSON with: "
            '{"hook_type": string, "tone": string, "beats": [{"role": '
            '"hook|setup|body|payoff|cta", "summary": string, "start": number, "end": number}], '
            '"why_it_works": [string, string, string]}. Judge craft, not topic. No prose.'
        )
        user = json.dumps(
            {
                "title": brief.title,
                "duration_seconds": round(sentences[-1][1], 1) if sentences else 0,
                "lines": [{"t": round(s, 2), "text": t} for s, _e, t in sentences],
            },
            indent=1,
        )
        data = ctx.llm.complete_json(system, user, temperature=0.2)
        if isinstance(data, dict):
            if data.get("hook_type"):
                brief.hook_style = str(data["hook_type"])[:40]
            if data.get("tone"):
                brief.tone = str(data["tone"])[:40]
            if isinstance(data.get("why_it_works"), list):
                brief.retention_notes += [str(n)[:200] for n in data["why_it_works"][:4]]
            if isinstance(data.get("beats"), list) and data["beats"]:
                brief.beats = [
                    Beat(
                        start=float(b.get("start") or 0.0),
                        end=float(b.get("end") or 0.0),
                        role=str(b.get("role") or "body"),
                        summary=str(b.get("summary") or ""),
                    )
                    for b in data["beats"][:16]
                ]
        return brief
