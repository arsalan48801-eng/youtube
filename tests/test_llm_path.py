"""The configured-provider paths, driven by fake HTTP clients.

These matter because they are the paths that only run on a machine with API keys, and
they are exactly where a malformed model response or a long narration should degrade
gracefully instead of failing the job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ytpipe.config import LLMSettings
from ytpipe.models import (
    Brief,
    JobRequest,
    PipelineArtifacts,
    StageName,
)
from ytpipe.pipeline import StageContext
from ytpipe.providers.llm import ChatLLM
from ytpipe.stages.script import ScriptStage


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = {}

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(kwargs)
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


SCRIPT_JSON = {
    "title": "The cable nobody checks",
    "description": "Why cheap cables ruin good monitors.",
    "tags": ["monitors", "cables", "bandwidth"],
    "lines": [
        {"role": "hook", "voiceover": "Your monitor is fine. The cable is not.", "on_screen": "Not the panel", "visual": "close up on a connector"},
        {"role": "setup", "voiceover": "Two of four panels dropped frames under load.", "on_screen": "2 of 4", "visual": "test bench"},
        {"role": "body", "voiceover": "Bandwidth caps out long before the panel does.", "on_screen": "Bandwidth", "visual": "signal path diagram"},
        {"role": "payoff", "voiceover": "Measure the link, then buy the panel.", "on_screen": "Measure first", "visual": "meter reading"},
        {"role": "cta", "voiceover": "Follow for the full test method.", "on_screen": "Follow", "visual": "end card"},
    ],
}


def fake_llm(payload, *, api_key="sk-test") -> ChatLLM:
    return ChatLLM(
        LLMSettings(api_key=api_key, model="fake-model"),
        client=FakeClient(FakeResponse(payload={"choices": [{"message": {"content": json.dumps(payload)}}]})),
    )


def ctx_for(tmp_path, settings, stage, *, artifacts=None, request=None):
    from ytpipe.state import JobStore

    store = JobStore(tmp_path / "db.sqlite3")
    job = store.create(
        request or JobRequest(source="x", topic="cables"), job_id="llm-test", workdir=tmp_path / "job"
    )
    return StageContext(
        job=job,
        settings=settings,
        store=store,
        artifacts=artifacts or PipelineArtifacts(),
        stage=stage,
    )


def test_llm_script_is_used_and_retimed_locally(tmp_path, settings):
    brief = Brief(
        title="Monitor buying mistakes", topic="monitor buying mistakes", wpm=150.0,
        avg_sentence_words=9.0, readability=6.0, keywords=["cable", "monitor"],
        hook_style="contrarian", tone="analytical", retention_notes=[],
    )
    ctx = ctx_for(tmp_path, settings, StageName.SCRIPT, artifacts=PipelineArtifacts(brief=brief))
    ctx.with_providers(llm=fake_llm(SCRIPT_JSON))
    script = ScriptStage().run(ctx)

    assert script.provider == "llm" and script.model == "fake-model"
    assert [ln.role for ln in script.lines] == ["hook", "setup", "body", "payoff", "cta"]
    assert script.lines[0].voiceover.startswith("Your monitor is fine")
    # Timings come from word counts, never from whatever the model claimed. The
    # lead-in lives *inside* shot 0 (speech_start > start), because a gap between
    # shots would be deleted by concat and slide every later caption off the picture.
    assert script.lines[0].start == pytest.approx(0.0)
    assert script.lines[0].speech_start == pytest.approx(0.35, abs=0.01)
    for prev, nxt in zip(script.lines, script.lines[1:], strict=False):
        assert nxt.start == pytest.approx(prev.end)
        assert nxt.end > nxt.start
        assert nxt.speech_start == pytest.approx(nxt.start)
    assert script.tags[:2] == ["monitors", "cables"]
    assert script.description
    # The stage keeps the exact prompt it sent, so a weird script is debuggable.
    # (script.json itself is written by the orchestrator -- see test_pipeline.)
    assert (ctx.dir / "prompt.txt").read_text(encoding="utf-8").startswith("You write voiceover scripts")


def test_prompt_carries_the_brief_constraints(tmp_path, settings):
    ctx = ctx_for(tmp_path, settings, StageName.SCRIPT, artifacts=PipelineArtifacts(brief=Brief(topic="cables", wpm=150, avg_sentence_words=8)))
    llm = fake_llm(SCRIPT_JSON)
    ctx.with_providers(llm=llm)
    ScriptStage().run(ctx)
    user = llm._client.calls[0]["json"]["messages"][1]["content"]
    payload = json.loads(user)
    assert payload["topic"] == "cables"
    assert "source_facts_you_may_reuse" in payload
    assert payload["aspect"] == "9:16"


def test_model_junk_falls_back_to_the_offline_writer(tmp_path, settings):
    """A bad completion must cost a retry, not the whole job."""
    ctx = ctx_for(tmp_path, settings, StageName.SCRIPT, artifacts=PipelineArtifacts(brief=Brief(topic="cables", top_sentences=["Measure first, buy later."])))
    broken = ChatLLM(
        LLMSettings(api_key="sk-test"),
        client=FakeClient(FakeResponse(status_code=200, payload={"choices": [{"message": {"content": "I can't do that."}}]})),
    )
    ctx.with_providers(llm=broken)
    script = ScriptStage().run(ctx)
    assert script.provider == "offline-template"
    assert script.lines and script.lines[0].role == "hook"


def test_budget_trims_overlong_model_scripts(tmp_path, settings):
    payload = dict(SCRIPT_JSON)
    payload["lines"] = SCRIPT_JSON["lines"] + [
        {"role": "body", "voiceover": " ".join(["extra"] * 25), "on_screen": "Extra", "visual": "b-roll"}
        for _ in range(6)
    ]
    ctx = ctx_for(
        tmp_path, settings, StageName.SCRIPT,
        artifacts=PipelineArtifacts(brief=Brief(topic="cables", wpm=150)),
        request=JobRequest(source="x", target_seconds=10, wpm=150),
    )
    ctx.with_providers(llm=fake_llm(payload))
    script = ScriptStage().run(ctx)
    budget = int(10 * 150 / 60)
    assert script.word_count <= budget * 1.2
    assert script.lines[0].role == "hook" and script.lines[-1].role == "cta"


def test_measured_tts_duration_rewrites_the_timeline(tmp_path, settings, source_clip):
    """Narration length drives picture length: the shot must stretch to fit the audio."""
    from ytpipe.models import Shot, Storyboard, VisualKind, VisualSpec
    from ytpipe.stages.generate import GenerateStage

    shots = [
        Shot(index=0, start=0.0, end=1.0, narration="One two three.", visual=VisualSpec(kind=VisualKind.CARD)),
        Shot(index=1, start=1.0, end=2.0, narration="Four five six seven eight.", visual=VisualSpec(kind=VisualKind.CARD)),
    ]
    board = Storyboard(shots=shots, width=270, height=480)
    ctx = ctx_for(
        tmp_path, settings, StageName.GENERATE,
        artifacts=PipelineArtifacts(storyboard=board),
        request=JobRequest(source="x", wpm=150),
    )

    class FakeTTS:
        """Returns 1.5s of audio for every line, so length must come from the file."""

        available = True
        backend = "fake-tts"

        def synthesize(self, text, target, *, speed=None):
            from ytpipe.ffmpeg import FFmpeg

            FFmpeg(settings.ffmpeg_binary).tone(target, 1.5, freq=300)
            return target

    ctx.with_providers(tts=FakeTTS())
    assets = GenerateStage().run(ctx)
    assert ctx.artifacts.storyboard.shots[0].duration >= 1.5
    assert ctx.artifacts.storyboard.shots[1].start == pytest.approx(
        ctx.artifacts.storyboard.shots[0].end
    )
    assert assets.assets[0].narration_provider == "fake-tts"
    assert all(a.audio_path and Path(a.audio_path).exists() for a in assets.assets)
    # captions were rebuilt against the new speech windows
    cues = ctx.artifacts.storyboard.captions
    assert cues and max(c.end for c in cues) <= ctx.artifacts.storyboard.shots[-1].end + 0.01


def test_tts_failure_degrades_to_silence(tmp_path, settings):
    from ytpipe.models import Shot, Storyboard, VisualSpec
    from ytpipe.stages.generate import GenerateStage

    shots = [Shot(index=0, start=0.0, end=1.0, narration="Hello there.", visual=VisualSpec())]
    ctx = ctx_for(
        tmp_path, settings, StageName.GENERATE,
        artifacts=PipelineArtifacts(storyboard=Storyboard(shots=shots, width=270, height=480)),
    )

    class BrokenTTS:
        available = True
        backend = "broken"

        def synthesize(self, *a, **k):
            raise RuntimeError("quota exceeded")

    ctx.with_providers(tts=BrokenTTS())
    assets = GenerateStage().run(ctx)
    assert assets.assets[0].narration_provider == "silent"
    assert Path(assets.assets[0].audio_path).exists(), "render still needs a timed track"
    assert ctx.artifacts.storyboard.shots[0].duration > 0
