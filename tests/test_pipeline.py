"""The orchestrator: planning, caching, retries, failure isolation, publish gating."""

from __future__ import annotations

import json
import time

import pytest

from ytpipe.errors import FetchError, PipelineError, ScriptError
from ytpipe.models import JobRequest, RenderResult, StageName, StageRun, Status
from ytpipe.pipeline import Pipeline
from ytpipe.state import JobStore


class DummyStage:
    def __init__(self, name, result=None, error=None, delay=0.0):
        self.name = name
        self.result = result
        self.error = error
        self.delay = delay
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            ctx.save(f"{self.name.value}.json", self.result)
        return self.result


class TestPlanning:
    def test_stage_order_is_the_documented_pipeline(self, settings):
        pipe = Pipeline(settings)
        order = [StageName(s.name).value for s in pipe.plan(JobRequest(source="x", publish=True))]
        assert order == [s.value for s in StageName.ordered()]

    def test_publish_is_opt_in(self, settings):
        pipe = Pipeline(settings)
        order = [StageName(s.name) for s in pipe.plan(JobRequest(source="x"))]
        assert StageName.PUBLISH not in order
        assert StageName.EDIT in order

    def test_until_stops_early_and_only_selects(self, settings):
        pipe = Pipeline(settings)
        up_to = [StageName(s.name) for s in pipe.plan(JobRequest(source="x", until=StageName.SCRIPT))]
        assert up_to == [StageName.FETCH, StageName.TRANSCRIBE, StageName.ANALYZE, StageName.SCRIPT]
        subset = [
            StageName(s.name)
            for s in pipe.plan(JobRequest(source="x", only=[StageName.SCRIPT, StageName.EDIT]))
        ]
        assert subset == [StageName.SCRIPT, StageName.EDIT]


class TestExecution:
    def test_artifacts_reach_the_next_stage_and_are_persisted(self, settings, tmp_path):
        from ytpipe.models import Script, ScriptLine

        script = Script(title="T", lines=[ScriptLine(index=0, role="hook", voiceover="Hello there.", start=0.0, end=1.0)])
        stages = [DummyStage(StageName.SCRIPT, result=script)]
        pipe = Pipeline(settings, stages=stages)
        job = pipe.run(JobRequest(source="x", until=StageName.SCRIPT, only=[StageName.SCRIPT]))
        assert job.status == Status.DONE
        run = job.stages[StageName.SCRIPT.value]
        assert run.status == Status.DONE and run.attempts == 1
        assert run.duration_s >= 0
        assert job.stages[StageName.SCRIPT.value].artifact.endswith("script.json")
        assert pipe.load_artifact(job, StageName.SCRIPT).title == "T"

    def test_render_result_is_persisted_and_reloadable(self, settings, tmp_path):
        fake = RenderResult(
            path=tmp_path / "v.mp4", duration=12.34, width=1080, height=1920, shot_count=5
        )
        (tmp_path / "v.mp4").write_bytes(b"x" * 10)
        pipe = Pipeline(settings, stages=[DummyStage(StageName.EDIT, result=fake)])
        job = pipe.create(JobRequest(source="x", only=[StageName.EDIT]))
        pipe.execute(job)
        assert job.status == Status.DONE
        loaded = pipe.load_artifact(job, StageName.EDIT)
        assert loaded.duration == 12.34 and loaded.shot_count == 5

    def test_retryable_errors_are_retried_then_recorded(self, settings):
        err = FetchError("transient")
        flaky = DummyStage(StageName.FETCH, error=err)
        pipe = Pipeline(settings, stages=[flaky])
        job = pipe.run(JobRequest(source="x", only=[StageName.FETCH]))
        assert job.status == Status.FAILED
        assert "transient" in job.error
        assert flaky.calls == settings.max_retries + 1
        run = job.stages[StageName.FETCH.value]
        assert run.attempts == settings.max_retries + 1
        assert any("attempt" in line for line in run.log)

    def test_unexpected_errors_are_not_retried(self, settings):
        boom = DummyStage(StageName.SCRIPT, error=ValueError("bug in stage"))
        pipe = Pipeline(settings, stages=[boom])
        job = pipe.run(JobRequest(source="x", only=[StageName.SCRIPT]))
        assert job.status == Status.FAILED and boom.calls == 1
        assert "ValueError" in job.error

    def test_failure_halts_the_run_and_later_stays_pending(self, settings):
        stages = [
            DummyStage(StageName.ANALYZE, error=PipelineError("nope")),
            DummyStage(StageName.SCRIPT),
        ]
        pipe = Pipeline(settings, stages=stages)
        job = pipe.run(JobRequest(source="x", only=[StageName.ANALYZE, StageName.SCRIPT]))
        assert job.status == Status.FAILED
        assert stages[1].calls == 0
        assert job.stages[StageName.SCRIPT.value].status == Status.PENDING

    def test_resume_skips_finished_stages(self, settings, tmp_path):
        from ytpipe.models import Script

        first = DummyStage(StageName.SCRIPT, result=Script(title="cached"))
        pipe = Pipeline(settings, stages=[first])
        job = pipe.create(JobRequest(source="x", only=[StageName.SCRIPT]))
        pipe.execute(job)
        assert first.calls == 1
        reloaded = pipe.store.get(job.id)
        reloaded.request.resume = True
        pipe.execute(reloaded)
        assert first.calls == 1, "a completed stage must not be re-run on resume"
        assert any("reused" in line for line in reloaded.stages[StageName.SCRIPT.value].log)

    def test_resume_retries_only_the_failed_stage(self, settings):
        from ytpipe.models import Script

        calls = {"n": 0}

        class FlakyOnce(DummyStage):
            def run(self, ctx):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ScriptError("first attempt fails")
                return super().run(ctx)

        pipe = Pipeline(settings, stages=[FlakyOnce(StageName.SCRIPT, result=Script(title="ok"))])
        job = pipe.create(JobRequest(source="x", only=[StageName.SCRIPT]))
        job.stages[StageName.SCRIPT.value] = StageRun(
            stage=StageName.SCRIPT, status=Status.FAILED, error="boom"
        )
        pipe.store.save_stage(job.id, job.stages[StageName.SCRIPT.value])
        job.request.resume = True
        pipe.execute(job)
        assert job.status == Status.DONE


class TestStateStore:
    def test_roundtrip_keeps_request_and_stage_detail(self, settings, tmp_path):
        store = JobStore(settings.db_path())
        req = JobRequest(source="https://youtu.be/x", target_seconds=61.5, aspect="16:9")
        store.create(req, job_id="j1", workdir=tmp_path / "j1")
        run = StageRun(
            stage=StageName.FETCH, status=Status.DONE, duration_s=1.5, attempts=2, artifact="/a/b"
        )
        run.log.append("hello")
        store.save_stage("j1", run)
        loaded = store.get("j1")
        assert loaded.request.target_seconds == 61.5
        assert loaded.request.aspect.value == "16:9"
        assert loaded.stages[StageName.FETCH.value].attempts == 2
        assert loaded.stages[StageName.FETCH.value].log == ["hello"]
        assert (tmp_path / "j1" / "stages").is_dir() and (tmp_path / "j1" / "out").is_dir()

    def test_events_are_ordered_and_cursor_paginates(self, settings):
        store = JobStore(settings.db_path())
        store.create(JobRequest(source="x"), job_id="j2", workdir=settings.job_dir("j2"))
        for i in range(5):
            store.event("j2", f"event {i}", stage="script")
        rows, cursor = store.events_after("j2", 0, limit=2)
        assert [r["message"] for r in rows] == ["event 0", "event 1"]
        rest, _ = store.events_after("j2", cursor)
        assert [r["message"] for r in rest] == ["event 2", "event 3", "event 4"]

    def test_missing_job_returns_none(self, settings):
        assert JobStore(settings.db_path()).get("nope") is None

    def test_stage_log_is_bounded(self, settings):
        store = JobStore(settings.db_path())
        store.create(JobRequest(source="x"), job_id="j3", workdir=settings.job_dir("j3"))
        run = StageRun(stage=StageName.EDIT, log=[f"line {i}" for i in range(500)])
        store.save_stage("j3", run)
        assert len(store.get("j3").stages[StageName.EDIT.value].log) == 40


class TestPublishGate:
    def test_later_stages_can_run_alone_against_earlier_artifacts(self, settings, local_source):
        """`--only publish` reuses the render from disk instead of re-running everything."""
        pipe = Pipeline(settings)
        job = pipe.run(
            JobRequest(
                source=str(local_source), topic="cable failures", target_seconds=6.0,
                wpm=210.0, offline=True, until=StageName.EDIT,
            ),
            job_id="pub-rerun",
        )
        assert job.status == Status.DONE, job.error
        reloaded = pipe.store.get("pub-rerun")
        reloaded.request = JobRequest(
            source=str(local_source), topic="cable failures", target_seconds=6.0,
            wpm=210.0, offline=True, only=[StageName.PUBLISH], publish=True,
        )
        pipe.execute(reloaded)
        assert reloaded.status == Status.DONE, reloaded.error
        assert (reloaded.workdir / "out" / "publish.json").exists()

    def test_publish_after_full_run_is_a_dry_run_without_credentials(self, settings, local_source):
        settings.workdir = settings.workdir  # keep isolated
        pipe = Pipeline(settings)
        job = pipe.run(
            JobRequest(
                source=str(local_source),
                topic="cable failures",
                target_seconds=6.0,
                wpm=210.0,
                offline=True,
                publish=True,
                privacy="unlisted",
            ),
            job_id="pub-full",
        )
        assert job.status == Status.DONE, job.error
        manifest = job.workdir / "out" / "publish.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["quota_units"] == 1600
        assert data["body"]["status"]["privacyStatus"] == "unlisted"
        assert data["media"]["bytes"] > 1000
        assert data["body"]["snippet"]["title"]
        result = pipe.load_artifact(job, StageName.PUBLISH)
        assert result.status == "dry-run"
        assert "credential" in (result.detail or "").lower() or "dry run" in (result.detail or "").lower()

class TestReRender:
    def test_edit_only_reuses_the_measured_timeline(self, settings, local_source):
        """A re-render must not fall back to the pre-measurement storyboard.

        `generate` measures the narration, and with it the lead-in that `edit` delays
        shot 0 by. If the artifact on disk still described the estimate, `--only edit`
        would produce a subtly different video than the run that produced it.
        """
        pipe = Pipeline(settings)
        req = {
            "source": str(local_source), "topic": "cable failures",
            "target_seconds": 6.0, "wpm": 210.0, "offline": True,
        }
        job = pipe.run(JobRequest(**req), job_id="reedit")
        assert job.status == Status.DONE, job.error
        first = pipe.load_artifact(job, StageName.EDIT)

        board = json.loads((job.workdir / "stages/storyboard/storyboard.json").read_text(encoding="utf-8"))
        assert board["shots"][0]["speech_start"] > 0.0, "lead-in must reach the artifact"

        again = pipe.run(JobRequest(**{**req, "only": [StageName.EDIT]}), job_id="reedit")
        assert again.status == Status.DONE, again.error
        second = pipe.load_artifact(again, StageName.EDIT)
        assert second.duration == pytest.approx(first.duration)
        assert again.workdir == job.workdir

