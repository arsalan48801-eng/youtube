"""Timing, analysis and the offline writer: the deterministic brain of the pipeline.

These are the stages where a wrong assumption becomes an unusable video, so they are
tested without ffmpeg, without a network and without an LLM.
"""

from __future__ import annotations

import math
import re

import pytest

from ytpipe.models import (
    Aspect,
    Brief,
    CaptionSegment,
    JobRequest,
    PipelineArtifacts,
    Script,
    ScriptLine,
    SourceVideo,
    StageName,
    Transcript,
)
from ytpipe.pipeline import StageContext
from ytpipe.stages.analyze import (
    AnalyzeStage,
    classify_hook,
    flesch_kincaid,
    keywords,
    mine_sentences,
    sentences_of,
)
from ytpipe.stages.script import ScriptStage, condense, on_screen_for, strip_self_reference
from ytpipe.stages.storyboard import StoryboardStage
from ytpipe.state import JobStore
from ytpipe.timing import MIN_SHOT, TAIL_PAD, apply, plan, retime_from_audio


def make_ctx(settings, tmp_path, stage, *, request=None, artifacts=None):
    """A StageContext wired to a throwaway store, so `emit`/`save` behave for real."""
    store = JobStore(tmp_path / "db.sqlite3")
    job = store.create(request or JobRequest(source="x"), job_id="t-" + stage.value,
                       workdir=tmp_path / "job")
    ctx = StageContext(
        job=job,
        settings=settings,
        store=store,
        artifacts=artifacts if artifacts is not None else PipelineArtifacts(),
        stage=stage,
    )
    return ctx


def transcript_from(lines: list[str], per_line: float = 3.0) -> Transcript:
    segments = []
    for i, line in enumerate(lines):
        segments.append(
            CaptionSegment(start=i * per_line, end=i * per_line + per_line - 0.1, text=line)
        )
    return Transcript(segments=segments, provider="test")


class TestTiming:
    def test_shots_are_gapless_and_sum_to_the_timeline(self):
        texts = ["one two three four", "five six", "seven eight nine"]
        timings = plan(texts, wpm=150, lead_in=0.35)
        for prev, nxt in zip(timings, timings[1:], strict=False):
            assert nxt.start == pytest.approx(prev.end)
        assert sum(t.duration for t in timings) == pytest.approx(timings[-1].end)

    def test_lead_in_is_absorbed_by_the_first_shot_not_left_as_a_gap(self):
        """An empty gap at the head survives as *padding*, but a gap between shots is
        deleted by `concat`, which slides every later caption off the picture."""
        timings = plan(["alpha beta", "gamma delta"], wpm=150, lead_in=0.4)
        assert timings[0].start == 0.0
        assert timings[0].speech_start == pytest.approx(0.4)
        assert timings[0].duration >= 0.4
        assert timings[1].speech_start == pytest.approx(timings[1].start)

    def test_measured_audio_replaces_the_estimate(self):
        class Shot:
            def __init__(self, text):
                self.narration, self.start, self.end = text, 0.0, 0.0
                self.speech_start = self.speech_end = 0.0

        shots = [Shot("a b"), Shot("c d e f g h i j k l m n")]
        plan_out = plan([s.narration for s in shots], wpm=150)
        apply(shots, plan_out)
        long_estimate = shots[1].end - shots[1].start
        timings = retime_from_audio(shots, [0.9, 5.5], wpm=150, lead_in=0.0)
        assert timings[1].duration > long_estimate
        assert timings[1].duration == pytest.approx(5.5 + TAIL_PAD, abs=0.02)

    def test_minimum_shot_length_is_enforced(self):
        timings = plan(["hi"], wpm=400, min_shot=MIN_SHOT)
        assert timings[0].duration >= MIN_SHOT

    def test_total_never_shrinks_below_the_sum_of_speech(self):
        texts = ["some words to speak here"] * 6
        timings = plan(texts, wpm=150, lead_in=0.35)
        assert timings[-1].end >= sum(len(t.split()) / 150 * 60 for t in texts)


class TestAnalyze:
    def test_hook_classification(self):
        assert classify_hook("Why do monitors fail?") == "question"
        assert classify_hook("We tested 4 panels for 30 days.") == "stat"
        assert classify_hook("Everything about refresh rate is a lie.") == "contrarian"
        assert classify_hook("Nice.") == "statement"

    def test_readability_and_keywords(self):
        sentences = ["The cable fails first.", "Bandwidth decides the outcome today."]
        assert flesch_kincaid(sentences) > 0
        kws = keywords("cable cable cable bandwidth bandwidth buy buy monitor", limit=4)
        assert kws[0] in {"cable", "bandwidth", "buy"}
        assert "the" not in kws

    def test_sentences_are_reassembled_from_word_cues(self):
        t = transcript_from(["the", "cable", "fails. then", "we fix it"], per_line=1.0)
        sentences = sentences_of(t)
        assert sentences[0][2].startswith("the cable fails.")
        assert sentences[0][0] == pytest.approx(0.0)

    def test_mine_sentences_skips_intro_and_cta_noise(self):
        lines = [
            "Welcome back to the channel everyone.",
            "Only two kept the refresh rate locked at 120 hertz.",
            "The failure was never the panel, it was the cable bandwidth.",
            "Please subscribe and smash the like button now.",
        ]
        sentences = [(i * 3.0, i * 3.0 + 2.9, line) for i, line in enumerate(lines)]
        mined = mine_sentences(sentences, {"refresh": 3, "cable": 3, "bandwidth": 3})
        assert any("120 hertz" in m or "cable" in m for m in mined)
        assert not any("smash" in m.lower() for m in mined)

    def test_brief_measures_cadence_that_the_writer_must_match(self, settings, tmp_path):
        t = transcript_from(
            [
                "Why does the cheap cable fail first under load?",
                "We measured four panels at 120 hertz for thirty days straight.",
                "Two locked the refresh rate, two dropped frames every ninety seconds.",
                "The fix costs eleven dollars and takes one minute.",
            ]
        )
        ctx = make_ctx(settings, tmp_path, StageName.ANALYZE)
        ctx.artifacts.transcript = t
        brief = AnalyzeStage().run(ctx)
        assert brief.words == len(t.text.split())
        assert brief.wpm > 0 and math.isfinite(brief.wpm)
        assert brief.hook_style == "question"
        assert brief.beats and brief.beats[0].role == "hook"
        assert brief.beats[-1].end == pytest.approx(t.duration, abs=0.2)
        assert any("wpm" in n for n in brief.retention_notes)
        assert brief.top_sentences


class TestOfflineWriter:
    def test_condense_trims_filler_and_length(self):
        assert not condense("So, actually the panel is fine").lower().startswith(("so", "actually"))
        long = " ".join(f"word{i}" for i in range(40))
        out = condense(long, max_words=12)
        assert len(out.split()) <= 18
        assert out.endswith((".", "!", "?"))

    def test_strip_self_reference_only_removes_the_framing_clause(self):
        kept = strip_self_reference("We tested four panels at 120 hertz.")
        assert kept.lower().startswith("we tested")
        framed = strip_self_reference("In this video we test four panels.")
        assert "in this video" not in framed.lower()
        assert framed[0].isupper()

    def test_on_screen_text_stays_short(self):
        out = on_screen_for("The spec sheet hides one number that actually matters")
        assert len(out) <= 17 and out

    def test_writer_produces_an_arc_in_order_and_respects_the_budget(self, settings, tmp_path):
        brief = Brief(
            title="Monitor buying mistakes",
            topic="monitor buying mistakes",
            hook_style="contrarian",
            tone="analytical",
            words=60,
            wpm=160.0,
            keywords=["monitor", "cable", "bandwidth", "refresh"],
            top_sentences=[
                "Only two kept the refresh rate locked at 120 hertz.",
                "The failure was never the panel, it was the cable.",
                "So measure before you spend and ignore the marketing.",
            ],
        )
        ctx = make_ctx(
            settings, tmp_path, StageName.SCRIPT,
            request=JobRequest(source="x", target_seconds=30, wpm=160),
            artifacts=PipelineArtifacts(brief=brief),
        )
        script = ScriptStage().run(ctx)
        roles = [ln.role for ln in script.lines]
        assert roles[0] == "hook" and roles[-1] == "cta" and "payoff" in roles
        assert all(ln.voiceover for ln in script.lines)
        assert script.word_count <= int(30 * 160 / 60 * 1.15)
        assert script.provider == "offline-template"
        # timings must be monotonic and gapless so the assembly can trust them
        for prev, nxt in zip(script.lines, script.lines[1:], strict=False):
            assert nxt.start == pytest.approx(prev.end)
        assert not any("in this video" in ln.voiceover.lower() for ln in script.lines)
        assert not any(len(ln.on_screen) > 20 for ln in script.lines)

    def test_no_false_number_promise_when_the_source_has_no_numbers(self, settings, tmp_path):
        brief = Brief(
            topic="patience",
            top_sentences=["Wait before you decide anything.", "Quiet practice beats talent."],
        )
        ctx = make_ctx(settings, tmp_path, StageName.SCRIPT, artifacts=PipelineArtifacts(brief=brief))
        script = ScriptStage().run(ctx)
        assert not re.search(r"\b\d+ numbers\b", script.voiceover.lower())

    def test_deterministic_for_the_same_input(self, settings, tmp_path):
        def run(name):
            brief = Brief(
                topic="cables",
                top_sentences=["A is true.", "B is false.", "C follows."],
                keywords=["cable"],
            )
            ctx = make_ctx(
                settings, tmp_path / name, StageName.SCRIPT, artifacts=PipelineArtifacts(brief=brief)
            )
            return ScriptStage().run(ctx).voiceover

        assert run("a") == run("b")


class TestStoryboard:
    def test_vertical_canvas_gets_shorter_caption_groups_than_wide(self, settings, tmp_path):
        def build(aspect, name):
            ctx = make_ctx(
                settings,
                tmp_path / name,
                StageName.STORYBOARD,
                request=JobRequest(source="x", aspect=aspect),
                artifacts=PipelineArtifacts(script=_tiny_script()),
            )
            return StoryboardStage().run(ctx)

        vertical, wide = build(Aspect.VERTICAL, "v"), build(Aspect.HORIZONTAL, "w")
        assert (vertical.width, vertical.height) == (270, 480)  # settings.render_scale=0.25
        assert wide.width > wide.height
        v_max = max(len(c.words) for c in vertical.captions)
        w_max = max(len(c.words) for c in wide.captions)
        assert v_max <= w_max

    def test_source_footage_is_preferred_and_not_reused_for_every_shot(self, settings, tmp_path, source_clip):
        t = transcript_from(["the cable fails under load today", "the monitor brightness matters more"])
        script = _tiny_script(
            lines=[
                ("body", "The cable fails under load."),
                ("body", "The monitor brightness matters."),
                ("body", "Nothing here matches the source at all xyzzy."),
            ]
        )
        ctx = make_ctx(
            settings, tmp_path, StageName.STORYBOARD,
            artifacts=PipelineArtifacts(
                script=script, transcript=t,
                source=SourceVideo(media_path=source_clip, duration=18.0),
            ),
        )
        board = StoryboardStage().run(ctx)
        kinds = [s.visual.kind for s in board.shots]
        assert "source_clip" in kinds, "matching lines should reuse source b-roll"
        assert kinds[-1] in ("card", "generated"), "an unmatched line must not steal source footage"
        spans = [(s.visual.source_start, s.visual.source_end) for s in board.shots if s.visual.source_start is not None]
        assert len(set(spans)) == len(spans), "the same source window should not be cut twice"
        for start, end in spans:
            assert 0 <= start < end <= 18.0

    def test_shots_keep_the_speech_window_that_captions_are_planned_on(self, settings, tmp_path):
        """The picture window and the speech window are different things.

        Shot 0 is 5.1s of video but the voice starts at 0.35; captions planned on the
        full window would flash the first words while the previous shot is still up.
        """
        script = _tiny_script(lines=[("hook", "Nobody checks the cable."), ("cta", "Follow for more.")])
        script.lines[0].speech_start = 0.35
        script.lines[0].speech_end = 2.6
        script.lines[1].start = 5.1
        script.lines[1].end = 7.4
        script.lines[1].speech_start = 5.1
        script.lines[1].speech_end = 7.1
        ctx = make_ctx(
            settings, tmp_path, StageName.STORYBOARD,
            artifacts=PipelineArtifacts(script=script),
        )
        board = StoryboardStage().run(ctx)
        assert board.shots[0].speech_start == pytest.approx(0.35)
        assert board.shots[1].speech_start == pytest.approx(5.1)
        assert min(c.start for c in board.captions) == pytest.approx(0.35)
        assert max(c.end for c in board.captions) <= 7.4 + 1e-6


class TestBudget:
    def test_short_targets_lose_beats_instead_of_overrunning(self, settings, tmp_path):
        """A 6s video cannot hold a five-beat arc: every line costs >= MIN_SHOT of picture.

        The writer used to hand the edit 11 seconds of script for a 6 second request.
        Cutting a beat is the honest answer; silently overrunning is not.
        """
        t = transcript_from(
            [
                "Why does the cheap cable fail first under load?",
                "We measured four panels at 120 hertz for thirty days straight.",
                "Two locked the refresh rate, two dropped frames every ninety seconds.",
                "The fix costs eleven dollars and takes one minute to apply.",
                "Most people simply buy the wrong cable and blame the monitor.",
            ]
        )
        brief = AnalyzeStage().run(make_ctx(settings, tmp_path / "an", StageName.ANALYZE,
                                           artifacts=PipelineArtifacts(transcript=t)))
        assert brief.top_sentences

        def write(target: float) -> Script:
            ctx = make_ctx(
                settings, tmp_path / f"s{target}", StageName.SCRIPT,
                request=JobRequest(source="x", topic="cable failures", target_seconds=target, wpm=210.0),
                artifacts=PipelineArtifacts(brief=brief),
            )
            return ScriptStage().run(ctx)

        long_form, short = write(30.0), write(6.0)
        assert len(short.lines) < len(long_form.lines)
        assert short.lines[0].role == "hook" and short.lines[-1].role == "cta"
        assert {ln.role for ln in short.lines} == {"hook", "cta"} or "payoff" in {
            ln.role for ln in short.lines
        }
        assert sum(ln.duration for ln in short.lines) <= 6.0 * 1.3
        assert sum(ln.duration for ln in long_form.lines) > sum(ln.duration for ln in short.lines)


def _tiny_script(*, lines: list[tuple[str, str]] | None = None):
    specs = lines or [("hook", "Nobody checks the cable."), ("body", "Bandwidth decides it."), ("cta", "Follow for more.")]
    return Script(
        title="Cable truth",
        lines=[
            ScriptLine(index=i, role=role, voiceover=text, on_screen=text.split()[0], start=i * 3.0, end=i * 3.0 + 3.0)
            for i, (role, text) in enumerate(specs)
        ],
    )
