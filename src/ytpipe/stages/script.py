"""Stage 4 — the script.

Two writers, one contract: both return a :class:`Script` whose lines are already
timed. Timing is *always* computed here from word counts, never taken from the
model, because a model that says "12 seconds" for a 9-word line is lying to the
edit stage and the picture will drift off the voiceover.

The offline writer is not a placeholder. It builds a real arc (hook -> promise ->
numbered points mined from the source -> payoff -> CTA), reuses only the
information-bearing fragments of the source, and respects the measured cadence from
the brief. Feed it the same brief twice and you get the same script, which is what
makes the surrounding code testable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import PipelineError, ScriptError
from ..models import Brief, Script, ScriptLine, StageName

_LEAD_FILLER = re.compile(
    r"^(?:and|but|so|now|okay|ok|well|um+|uh+|actually|basically|literally|obviously|look|right)"
    r"[,\s]+",
    re.I,
)
#: Only the whole framing clause is removed. Rewriting bare "we"/"us" would mangle
#: legitimate sentences, and "we tested four panels" is exactly the credibility line
#: worth keeping.
_SELF_REF = re.compile(
    r"\b(?:in|within|through) this (?:video|episode|tutorial|clip|guide)[,.]?\s*"
    r"(?:i|i'm|i've|we|you)?\s*(?:'’|')?(?:ll|will)?\s*"
    r"(?:take|show|walk|look|go|break|cover|explain|test|dive)?\w*",
    re.I,
)
_HASHTAG = re.compile(r"#\w+")
_SPONSOR = re.compile(
    r"\b(this video is (sponsored|brought to you)"
    r"|thanks to .{0,40} for sponsoring"
    r"|use my code|affiliate link)\b",
    re.I,
)


def condense(text: str, *, max_words: int = 14) -> str:
    """One breath-sized line: no filler, no self-reference, no ad copy."""
    text = _HASHTAG.sub(" ", text)
    text = _SPONSOR.sub(" ", text)
    for _ in range(3):
        text = _LEAD_FILLER.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    if len(text.split()) > max_words + 6:
        # Prefer a clause boundary over a hard mid-sentence cut.
        head = " ".join(text.split()[: max_words + 6])
        for sep in (". ", "; ", " - ", ", "):
            if sep in head[max_words - 3 :]:
                candidate = head[: head.rfind(sep) + len(sep)].strip()
                if len(candidate.split()) >= max(4, int(max_words * 0.6)):
                    text = candidate
                    break
        else:
            text = " ".join(text.split()[:max_words]).rstrip(",;:") + "."
    if not text.endswith((".", "!", "?")):
        text += "."
    return text[0].upper() + text[1:] if text else text


def strip_self_reference(text: str) -> str:
    """Drop the "in this video I'll" frame without touching ordinary pronouns.

    Only the whole framing clause is removed: rewriting every "we" would mangle
    legitimate sentences (and "we tested it" is exactly the credibility we want).
    """
    cleaned = _SELF_REF.sub("", text, count=1)
    cleaned = re.sub(r"^[,;:\s]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return text
    return cleaned[0].upper() + cleaned[1:]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def assign_timings(lines: list[ScriptLine], wpm: float, *, lead_in: float = 0.35) -> None:
    """Provisional in-place timeline from word counts.

    `generate` replaces this with measured audio lengths. Both go through
    :mod:`ytpipe.timing` so the two layouts cannot disagree about where the gaps are.
    """
    from ..timing import apply, plan

    apply(lines, plan([ln.voiceover for ln in lines], wpm=wpm, lead_in=lead_in))


def on_screen_for(text: str, *, max_chars: int = 17) -> str:
    """A 1-3 word overlay. Longer and it competes with the captions for the same air.

    Chosen from the informative tail of the line, then trimmed to the widest word if it
    still does not fit, because a clipped overlay is worse than a terse one.
    """
    words = re.findall(r"[A-Za-z0-9'’%$.:-]+", text)
    keep = [w for w in words if len(w) > 3 or w.isdigit() or "%" in w] or words
    out = " ".join(keep[-3:] if len(keep) > 3 else keep)
    out = out.strip(" .,;:").title()
    if len(out) > max_chars:
        longest = max(keep, key=len) if keep else ""
        out = (longest or out[:max_chars]).strip(" .,;:").title()
    return out[:max_chars]


class ScriptStage:
    name = StageName.SCRIPT

    def run(self, ctx: Any) -> Script:
        brief: Brief | None = ctx.artifacts.brief
        if brief is None:
            raise ScriptError("analyze stage produced no brief")
        request = ctx.job.request
        target = float(request.target_seconds or 45.0)
        wpm = float(request.wpm or brief.wpm or 150.0)
        budget = int(target * wpm / 60.0)

        if ctx.llm is not None and ctx.llm.available:
            try:
                script = self._via_llm(ctx, brief, target=target, wpm=wpm, budget=budget)
                ctx.emit(f"llm script: {script.word_count} words / {budget} budget")
            except PipelineError:
                raise
            except Exception as exc:
                ctx.emit(f"llm writer failed ({exc}); using the offline writer", level="warning")
                script = self._offline(ctx, brief, target=target, wpm=wpm, budget=budget)
        else:
            script = self._offline(ctx, brief, target=target, wpm=wpm, budget=budget)
            ctx.emit(f"offline writer: {script.word_count} words / {budget} budget")

        if not script.lines:
            raise ScriptError("writer returned no lines")
        ctx.save("prompt.txt", script.prompts_used[-1] if script.prompts_used else "(offline)")
        return script

    # -- llm ------------------------------------------------------------------

    def _via_llm(self, ctx: Any, brief: Brief, *, target: float, wpm: float, budget: int) -> Script:
        system = (
            "You write voiceover scripts for faceless short-form video (Shorts/Reels/TikTok).\n"
            "HARD RULES:\n"
            f"- {budget} words total, +/-10% (this is a {target:.0f}s read at {wpm:.0f} wpm).\n"
            "- One idea per line. Maximum 14 words per line. 8-14 lines.\n"
            "- Line 1 is the hook: no 'hello', no 'in this video', no 'welcome'.\n"
            "- Use second person. Concrete nouns and numbers beat adjectives.\n"
            "- No emojis, no hashtags, no camera directions, no music cues.\n"
            "- `on_screen` is 1-4 words only. `visual` describes one shot to cut to.\n"
            "- Last line is the only CTA, and it must not say 'smash the like button'.\n"
            f"- Match this brief's measured cadence: tone={brief.tone}, hook style={brief.hook_style}, "
            f"avg {brief.avg_sentence_words:.0f} words/line, readability grade {brief.readability}.\n"
            "Return JSON only, shaped exactly like:\n"
            '{"title": str, "description": str, "tags": [str], "lines": '
            '[{"role": "hook|setup|body|payoff|cta", "voiceover": str, "on_screen": str, '
            '"visual": str}]}'
        )
        user = json.dumps(
            {
                "topic": ctx.job.request.topic or brief.topic,
                "title_hint": ctx.job.request.title_hint,
                "aspect": ctx.job.request.aspect.value
                if hasattr(ctx.job.request.aspect, "value")
                else str(ctx.job.request.aspect),
                "source_title": brief.title,
                "brief": {
                    "hook_line": brief.hook_line,
                    "hook_style": brief.hook_style,
                    "tone": brief.tone,
                    "wpm": brief.wpm,
                    "keywords": brief.keywords[:12],
                    "retention_notes": brief.retention_notes,
                },
                "source_facts_you_may_reuse": brief.top_sentences[:10],
                "instruction": (
                    "Rewrite, do not summarise. Reuse only facts and numbers from "
                    "'source_facts_you_may_reuse'; everything else is yours. Never "
                    "repeat a source sentence verbatim for more than 5 words."
                ),
            },
            indent=1,
        )
        data = ctx.llm.complete_json(system, user, temperature=ctx.settings.llm.temperature)
        if not isinstance(data, dict):
            raise ScriptError(f"expected a JSON object, got {type(data).__name__}")
        raw_lines = data.get("lines") or data.get("script") or []
        lines: list[ScriptLine] = []
        for i, item in enumerate(raw_lines):
            if isinstance(item, str):
                item = {"role": "body", "voiceover": item}
            voice = condense(str(item.get("voiceover") or item.get("text") or ""))
            if not voice:
                continue
            lines.append(
                ScriptLine(
                    index=i,
                    role=str(item.get("role") or ("hook" if i == 0 else "body")),
                    voiceover=voice,
                    on_screen=str(item.get("on_screen") or "").strip() or on_screen_for(voice),
                    visual=str(item.get("visual") or "").strip(),
                )
            )
        if not lines:
            raise ScriptError("JSON had no usable lines")
        script = Script(
            title=str(data.get("title") or "").strip()[:100] or brief.title[:100],
            description=str(data.get("description") or "").strip()[:5000],
            tags=[str(t).strip().lower() for t in (data.get("tags") or []) if str(t).strip()][:16],
            lines=lines,
            provider="llm",
            model=ctx.llm.model,
            target_seconds=target,
            wpm=wpm,
            prompts_used=[system + "\n\n" + user],
        )
        _enforce_budget(ctx, script.lines, budget, target=target, wpm=wpm)
        assign_timings(script.lines, wpm)
        return self._finalize(ctx, script, brief, target, wpm)

    # -- offline --------------------------------------------------------------

    def _offline(self, ctx: Any, brief: Brief, *, target: float, wpm: float, budget: int) -> Script:
        """Template arc over mined source material. Deterministic by construction."""
        topic = (ctx.job.request.topic or brief.topic or brief.title or "this").strip()
        topic = topic.rstrip(".!")
        kws = [k for k in brief.keywords if len(k) > 3]
        facts = [strip_self_reference(condense(s, max_words=13)) for s in brief.top_sentences]
        facts = [f for f in facts if f and len(f.split()) >= 4]

        lines: list[ScriptLine] = []

        def add(role: str, voice: str, *, visual: str = "", screen: str | None = None) -> None:
            voice = voice.strip()
            if not voice:
                return
            lines.append(
                ScriptLine(
                    index=len(lines),
                    role=role,
                    voiceover=voice,
                    on_screen=(screen or on_screen_for(voice))[:20],
                    visual=visual or f"tight shot illustrating: {voice[:60]}",
                )
            )

        # HOOK -- mirror whatever the source did, since that is what measured well.
        hook = self._hook(topic, brief, kws)
        add("hook", hook, visual=f"cold open, motion already running: {topic.lower()[:48]}")

        # PROMISE -- why staying is worth it. Only promise numbers if the source
        # actually gave some; a "3 numbers" hook over a video with zero figures is a
        # retention cliff at second 12.
        numeric = [f for f in facts if re.search(r"\d", f)]
        count = min(len(numeric), 3)
        if count >= 2:
            add(
                "setup",
                f"{_WORDS[count]} numbers explain {topic} better than any opinion.",
                visual="text overlay on a clean background",
                screen=f"{count} Numbers",
            )
        else:
            add(
                "setup",
                f"Here is the part of {topic} that decides the outcome.",
                visual="text overlay on a clean background",
                screen="The Part",
            )

        # BODY -- mined facts, one per line, each with its own cut. The final fact is
        # held back for the payoff so the closer is the source's own closing sentence.
        pool = (facts[:-1] if len(facts) >= 3 else facts)[:8] or self._fallback_points(topic, kws)
        if not facts:
            ctx.emit(
                "transcript had no reusable lines; writing from the topic alone", level="warning"
            )
        for fact in pool:
            if _word_total(lines) + len(fact.split()) > budget:
                break
            add("body", fact, visual=f"b-roll of {_pick_noun(fact) or topic}")

        # PAYOFF -- what to do with all of it.
        payoff = facts[-1] if len(facts) >= 2 else self._payoff(topic, kws)
        add("payoff", payoff, visual="slow push-in on the result", screen="Takeaway")

        # CTA -- one line, never earlier.
        add(
            "cta",
            f"Follow for more on {topic.lower().rstrip('.!')}.",
            visual="end card with channel mark",
            screen="Follow",
        )

        _enforce_budget(ctx, lines, budget, target=target, wpm=wpm)
        assign_timings(lines, wpm)

        title = self._title(topic, brief, kws)
        description = "\n".join(
            [
                hook,
                "",
                *[f"- {ln.voiceover}" for ln in lines if ln.role == "body"][:4],
                "",
                f"Made from a {brief.words}-word source transcript. {len(lines)} lines, "
                f"~{wpm:.0f} wpm.",
            ]
        )
        return Script(
            title=title,
            description=description,
            tags=_dedupe(
                [*(k.lower() for k in kws[:8]), topic.lower().split()[0] if topic.split() else "short"]
            ),
            lines=lines,
            provider="offline-template",
            target_seconds=target,
            wpm=wpm,
            prompts_used=[f"(offline) arc=hook,setup,body*{len(pool)},payoff,cta budget={budget}w"],
        )

    # -- shared parts ---------------------------------------------------------

    @staticmethod
    def _hook(topic: str, brief: Brief, kws: list[str]) -> str:
        style = brief.hook_style
        first = brief.hook_line
        if style == "question" and first:
            return condense(first, max_words=12)
        if style == "stat" and first:
            return condense(first, max_words=12)
        if style == "contrarian":
            return f"Everything said about {topic.lower()} skips the part that matters."
        if kws:
            return f"{kws[0].capitalize()} is the reason {topic.lower()} keeps failing."
        return f"Here is what nobody shows you about {topic.lower()}."

    @staticmethod
    def _fallback_points(topic: str, kws: list[str]) -> list[str]:
        """Used when the source gave nothing reusable: three generic-but-grammatical beats."""
        seeds = [k for k in kws[:3] if len(k) > 3] or ["the setup", "the cost", "the fix"]
        templates = [
            "{} decides the outcome before you start.",
            "Most people measure {} the wrong way.",
            "Change {} once and the rest follows.",
        ]
        return [templates[i % len(templates)].format(seed) for i, seed in enumerate(seeds)]

    @staticmethod
    def _payoff(topic: str, kws: list[str]) -> str:
        lever = _specific_keyword(kws) or "one variable"
        return f"Fix {lever} first; everything else is noise."

    @staticmethod
    def _title(topic: str, brief: Brief, kws: list[str]) -> str:
        lever = _specific_keyword(kws)
        hint = topic.strip().rstrip(":")
        return (f"{hint}: the {lever} nobody checks" if lever else hint)[:98]


    def _finalize(self, ctx: Any, script: Script, brief: Brief, target: float, wpm: float) -> Script:
        if not script.title:
            script.title = brief.title[:100] or "Untitled"
        if not script.description:
            script.description = script.lines[0].voiceover if script.lines else ""
        if not script.tags:
            script.tags = [k.lower() for k in brief.keywords[:8]]
        if script.duration > target * 1.25:
            ctx.emit(
                f"script reads {script.duration:.1f}s against a {target:.0f}s target",
                level="warning",
            )
        return script


def _enforce_budget(
    ctx: Any, lines: list[ScriptLine], budget: int, *, target: float = 0.0, wpm: float = 150.0
) -> None:
    """Trim to what the render can actually hold. Body lines go first.

    Cutting the longest body line -- never the hook or the CTA -- keeps the arc intact:
    an over-long video loses more retention than one missing detail does.

    Two limits are enforced, because they fail differently. The word budget catches a
    writer that rambles. The *shot floor* catches the opposite case: every line costs at
    least ``MIN_SHOT`` seconds of picture, so a five-beat arc physically cannot fit in six
    seconds no matter how few words it uses. Without the second pass the timeline simply
    overruns the target and the edit renders a 11s clip nobody asked for.
    """
    from ..providers.tts import estimate_speech_seconds
    from ..timing import MIN_SHOT, TAIL_PAD

    total = _word_total(lines)
    if total > int(budget * 1.1):
        droppable = [ln for ln in lines if ln.role == "body"]
        for line in sorted(droppable, key=lambda ln: -len(ln.voiceover.split())):
            if total <= budget:
                break
            lines.remove(line)
            total = _word_total(lines)
            ctx.emit(f"budget: dropped {line.voiceover[:40]!r} (words)", level="warning")

    if target > 0:
        def floor(ls: list[ScriptLine]) -> float:
            return sum(
                max(MIN_SHOT, estimate_speech_seconds(ln.voiceover, wpm=wpm, padding=0.0) + TAIL_PAD)
                for ln in ls
            )

        while floor(lines) > target * 1.12:
            # Hook and CTA are never candidates; within the rest the longest line goes
            # first, which is the biggest cut for the least arc damage. Two lines is the
            # smallest arc that still reads as a video rather than as a caption.
            droppable = [ln for ln in lines if ln.role in {"body", "setup", "payoff"}]
            if not droppable or len(lines) <= 2:
                break
            worst = max(droppable, key=lambda ln: floor([ln]))
            lines.remove(worst)
            ctx.emit(
                f"budget: dropped {worst.voiceover[:40]!r} "
                f"({floor(lines):.1f}s fits a {target:.0f}s video; {len(lines)} beats)",
                level="warning",
            )

    for i, line in enumerate(lines):
        line.index = i


def _word_total(lines: list[ScriptLine]) -> int:
    return sum(len(ln.voiceover.split()) for ln in lines)


#: Words that are frequent in transcripts but useless as the noun in a payoff line.
#: Frequent in transcripts, useless as the noun in a hook or a title.
_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

_VAGUE = {
    "people", "thing", "things", "video", "time", "way", "lot", "stuff", "day", "days",
    "year", "years", "part", "parts", "rest", "everything", "something", "nothing",
    "guys", "wrong", "right", "first", "last", "best", "new", "real", "same", "other",
    "many", "more", "most", "better", "good", "bad", "big", "small", "one", "two",
    "actually", "really", "just", "like", "know", "think", "going", "want", "need",
}


def _specific_keyword(kws: list[str]) -> str:
    """The most concrete keyword, preferring one that is already in the topic.

    Raw frequency is the wrong ranking here: the most common content word in a
    transcript is usually a vague one ("people"), and putting it into "Fix X first"
    is what makes auto-written scripts sound machine-made.
    """
    if not kws:
        return ""
    candidates = [k for k in kws if k.lower() not in _VAGUE and len(k) > 3]
    if not candidates:
        return ""
    topic_words = {k.lower() for k in kws}
    return max(candidates, key=lambda k: (len(k), k.lower() in topic_words))


def _pick_noun(sentence: str) -> str:
    skip = {"this", "that", "with", "from"}
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", sentence) if w.lower() not in skip]
    return words[0] if words else ""


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
