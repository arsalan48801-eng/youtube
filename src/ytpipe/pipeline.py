"""The orchestrator.

Stages are plain objects with a ``run(ctx)`` method, registered by name. The
orchestrator owns everything that is not the actual work: ordering, per-stage
artifact caching (so a failed render does not re-spend a fetch or an LLM call),
retries with backoff, progress events, and the on-disk job layout.

    work/jobs/<job-id>/
      stages/fetch/     source.mp4  source.json   captions.vtt
      stages/script/    script.json prompt.txt
      stages/generate/  shot_003.mp4 narration_003.wav
      stages/edit/      shots/ final.ass
      out/              final.mp4  final.srt  thumb.jpg  publish.json
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import Settings, build_settings
from .errors import PipelineError
from .ffmpeg import FFmpeg
from .models import Job, JobRequest, PipelineArtifacts, StageName, StageRun, Status, utcnow
from .state import JobStore, new_job_id

log = logging.getLogger("ytpipe")


@runtime_checkable
class Stage(Protocol):
    name: StageName

    def run(self, ctx: StageContext) -> Any:  # pragma: no cover - interface
        ...


@dataclass
class StageContext:
    """Everything a stage is allowed to touch. Keeps stages honest and testable."""

    job: Job
    settings: Settings
    store: JobStore
    artifacts: PipelineArtifacts
    stage: StageName
    ffmpeg: FFmpeg = field(default_factory=FFmpeg)
    _llm: Any = field(default=None, init=False, repr=False)
    _tts: Any = field(default=None, init=False, repr=False)
    _images: Any = field(default=None, init=False, repr=False)

    def with_providers(self, *, llm: Any = None, tts: Any = None, images: Any = None) -> StageContext:
        """Inject prepared providers (mostly a test seam for fake LLM/TTS clients)."""
        self._llm, self._tts, self._images = llm, tts, images
        return self

    # -- filesystem -----------------------------------------------------------

    @property
    def dir(self) -> Path:
        path = self.job.stage_dir(self.stage)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, name: str, *, mkdir: bool = True) -> Path:
        target = self.dir / name
        if mkdir:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def output(self, name: str = "final.mp4") -> Path:
        """Destination in the job's out/ folder (the deliverables, not stage scratch)."""
        target = self.job.output(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def out_dir(self) -> Path:
        self.job.output("x").parent.mkdir(parents=True, exist_ok=True)
        return self.job.output("x").parent

    # -- typed IO -------------------------------------------------------------

    def emit(self, message: str, *, level: str = "info") -> None:
        getattr(log, level if level in {"info", "warning", "error", "debug"} else "info")(
            "%s [%s] %s", self.job.id, self.stage.value, message
        )
        run = self.job.stages.get(self.stage.value)
        if run is not None:
            run.log.append(f"{utcnow().strftime('%H:%M:%S')} {message}")
            run.log = run.log[-40:]
        self.store.event(self.job.id, message, stage=self.stage.value, level=level)

    def save(self, name: str, value: Any) -> Path:
        """Persist a stage artifact so a later run can skip this stage, and so a human
        can read what happened. Models become JSON; plain strings stay plain text
        (a prompt wrapped in JSON quotes is not worth opening)."""
        return self.save_to(self.stage, name, value)

    def save_to(self, stage: StageName, name: str, value: Any) -> Path:
        """Write into another stage's artifact folder.

        Exists because `generate` measures the narration and therefore *owns* the final
        timeline: the storyboard artifact must describe what will really be rendered, or
        a later `--only edit` would re-render from pre-measurement timings.
        """
        target = self.job.stage_dir(stage) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            target.write_text(value, encoding="utf-8")
            return target
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def load(self, name: str, model: type[Any]) -> Any | None:
        target = self.dir / name
        if not target.exists():
            return None
        try:
            return model.model_validate(json.loads(target.read_text(encoding="utf-8")))
        except Exception as exc:  # corrupt artifact -> recompute rather than die
            self.emit(f"discarding unreadable {name}: {exc}", level="warning")
            return None

    def use_cache(self) -> bool:
        return bool(self.job.request.resume)

    # -- providers (None when not configured, so stages can branch once) ------

    @property
    def llm(self):  # -> ChatLLM | None
        if self._llm is not None:
            return self._llm
        if not self.settings.llm_available:
            return None
        from .providers.llm import ChatLLM

        if self._llm is None:
            self._llm = ChatLLM(self.settings.llm)
        return self._llm

    @property
    def tts(self):  # -> SpeechProvider | None
        if self._tts is not None:
            return self._tts
        if not self.settings.tts_available:
            return None
        from .providers.tts import SpeechProvider

        if self._tts is None:
            self._tts = SpeechProvider(self.settings.tts)
        return self._tts

    @property
    def images(self):  # -> ImageProvider | None
        if self._images is not None:
            return self._images
        if not self.settings.images_available:
            return None
        from .providers.images import ImageProvider

        if self._images is None:
            self._images = ImageProvider(self.settings.images)
        return self._images

    @staticmethod
    def now():
        return utcnow()


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        store: JobStore | None = None,
        stages: list[Stage] | None = None,
    ) -> None:
        self.settings = settings or build_settings()
        self.settings.ensure_dirs()
        self.store = store or JobStore(self.settings.db_path())
        self.ffmpeg = FFmpeg(self.settings.ffmpeg_binary)
        self.stages = list(stages if stages is not None else default_stages())
        self._by_name = {StageName(s.name): s for s in self.stages}

    # -- planning -------------------------------------------------------------

    def plan(self, request: JobRequest) -> list[Stage]:
        order = [self._by_name[n] for n in StageName.ordered() if n in self._by_name]
        if request.only:
            wanted = {StageName(s) for s in request.only}
            order = [s for s in order if StageName(s.name) in wanted]
        if request.until:
            limit = StageName(request.until)
            order = [s for s in order if _rank(StageName(s.name)) <= _rank(limit)]
        if not request.publish:
            # Publishing is opt-in on the request, not opt-out: a typo in a shell
            # history should never push a video to a channel.
            order = [s for s in order if StageName(s.name) is not StageName.PUBLISH]
        return order

    # -- execution ------------------------------------------------------------

    def create(self, request: JobRequest, *, job_id: str | None = None) -> Job:
        job_id = job_id or new_job_id()
        workdir = self.settings.job_dir(job_id)
        job = Job(
            id=job_id,
            request=request,
            workdir=workdir,
            status=Status.PENDING,
            stages={
                StageName(s.name).value: StageRun(stage=StageName(s.name)) for s in self.stages
            },
        )
        return self.store.register(job)

    def execute(self, job: Job) -> PipelineArtifacts:
        """Run the planned stages, caching any that already produced artifacts."""
        artifacts = self._load_existing(job)
        self.store.update(job.id, status=Status.RUNNING, error=None)
        job.status = Status.RUNNING
        failure: str | None = None

        for stage in self.plan(job.request):
            name = StageName(stage.name)
            run = job.stages.setdefault(name.value, StageRun(stage=name))
            if run.status is Status.DONE and job.request.resume:
                run.log.append("reused cached artifact")
                self.store.save_stage(job.id, run)
                continue
            ctx = StageContext(
                job=job,
                settings=self.settings,
                store=self.store,
                artifacts=artifacts,
                stage=name,
                ffmpeg=self.ffmpeg,
            )
            self._run_stage(stage, ctx, run, artifacts)
            self.store.save_stage(job.id, run)
            if run.status is Status.FAILED:
                failure = run.error or "stage failed"
                break

        status = Status.FAILED if failure else Status.DONE
        self.store.update(job.id, status=status, error=failure)
        job.status = status
        job.error = failure
        job.updated_at = utcnow()
        return artifacts

    def _run_stage(
        self, stage: Stage, ctx: StageContext, run: StageRun, artifacts: PipelineArtifacts
    ) -> None:
        name = StageName(stage.name)
        run.status = Status.RUNNING
        run.started_at = utcnow()
        run.error = None
        started = time.monotonic()
        self.store.save_stage(ctx.job.id, run)

        # A prior run of this stage may already have the answer.
        cached = self._cached_result(name, ctx)
        if cached is not None and ctx.use_cache():
            setattr(artifacts, _ATTR[name], cached)
            run.status, run.finished_at = Status.DONE, utcnow()
            run.duration_s = round(time.monotonic() - started, 3)
            ctx.emit(f"reused {name.value} from cache")
            return

        attempts = max(1, self.settings.max_retries + 1)
        error: str | None = None
        for attempt in range(attempts):
            run.attempts = attempt + 1
            try:
                result = stage.run(ctx)
                if result is not None and hasattr(result, "model_dump"):
                    setattr(artifacts, _ATTR[name], result)
                    run.artifact = str(self._artifact_path(name, ctx, result))
                run.status = Status.DONE
                run.duration_s = round(time.monotonic() - started, 3)
                run.finished_at = utcnow()
                ctx.emit(f"{name.value} complete in {run.duration_s:.2f}s")
                return
            except PipelineError as exc:
                error = str(exc)
                run.log.append(f"attempt {attempt + 1}: {exc}")
                if not exc.retryable or attempt + 1 >= attempts:
                    break
                backoff = self.settings.retry_backoff * (2**attempt)
                ctx.emit(f"{name.value} failed ({exc}); retrying in {backoff:.1f}s", level="warning")
                time.sleep(backoff)
            except Exception as exc:  # unexpected -> keep the type for the operator
                error = f"{type(exc).__name__}: {exc}"
                run.log.append(f"attempt {attempt + 1}: {exc!r}")
                ctx.emit(traceback.format_exc(limit=3), level="error")
                break

        run.status = Status.FAILED
        run.error = error or "stage failed"
        run.finished_at = utcnow()
        run.duration_s = round(time.monotonic() - started, 3)
        ctx.emit(f"{name.value} failed: {run.error}", level="error")

    # -- artifacts ------------------------------------------------------------

    def _load_existing(self, job: Job) -> PipelineArtifacts:
        """Restore artifacts a previous run wrote, so `--only` can start mid-pipeline.

        Without this, "re-render only" (`--only edit,publish`) would have to re-spend
        the fetch and the LLM call, and an interrupted job could not be finished by
        hand. Loading is permissive: a stage that is about to run simply overwrites
        what we read here.
        """
        artifacts = PipelineArtifacts()
        models = _models()
        for stage, (model_name, filename) in _ARTIFACT_FILES.items():
            path = job.stage_dir(stage) / filename
            if not path.exists():
                continue
            try:
                payload = models[model_name].model_validate_json(path.read_text(encoding="utf-8"))
                setattr(artifacts, _ATTR[stage], payload)
            except Exception as exc:
                log.debug("%s: ignoring stale %s (%s)", job.id, filename, exc)
        return artifacts

    def load_artifact(self, job: Job, stage: StageName) -> Any | None:
        """Read a stage's persisted artifact without re-running anything.

        Exists so the API and the tests can inspect a finished job from disk alone --
        a run's decisions are only auditable if they can be read back.
        """
        entry = _ARTIFACT_FILES.get(StageName(stage))
        if not entry:
            return None
        model_name, filename = entry
        return StageContext(
            job=job, settings=self.settings, store=self.store,
            artifacts=PipelineArtifacts(), stage=StageName(stage), ffmpeg=self.ffmpeg,
        ).load(filename, _models()[model_name])

    def _cached_result(self, name: StageName, ctx: StageContext) -> Any | None:
        entry = _ARTIFACT_FILES.get(name)
        if not entry:
            return None
        model_name, filename = entry
        assert model_name in _models()
        return ctx.load(filename, _models()[model_name])

    def _artifact_path(self, name: StageName, ctx: StageContext, result: Any) -> Path:
        entry = _ARTIFACT_FILES.get(name)
        if entry and hasattr(result, "model_dump"):
            return ctx.save(entry[1], result)
        return ctx.path(f"{name.value}.json")

    # -- entry points ---------------------------------------------------------

    def run(self, request: JobRequest, *, job_id: str | None = None) -> Job:
        job = self.create(request, job_id=job_id)
        self.execute(job)
        return job

    def resume(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(f"no such job: {job_id}")
        job.request.resume = True
        self.execute(job)
        return job


_RANK = {name: i for i, name in enumerate(StageName.ordered())}


def _rank(name: StageName) -> int:
    return _RANK[StageName(name)]


_ATTR = {
    StageName.FETCH: "source",
    StageName.TRANSCRIBE: "transcript",
    StageName.ANALYZE: "brief",
    StageName.SCRIPT: "script",
    StageName.STORYBOARD: "storyboard",
    StageName.GENERATE: "assets",
    StageName.EDIT: "render",
    StageName.PUBLISH: "publish",
}

_ARTIFACT_FILES = {
    StageName.FETCH: ("SourceVideo", "source.json"),
    StageName.TRANSCRIBE: ("Transcript", "transcript.json"),
    StageName.ANALYZE: ("Brief", "brief.json"),
    StageName.SCRIPT: ("Script", "script.json"),
    StageName.STORYBOARD: ("Storyboard", "storyboard.json"),
    StageName.GENERATE: ("AssetSet", "assets.json"),
    StageName.EDIT: ("RenderResult", "render.json"),
    StageName.PUBLISH: ("PublishResult", "publish.json"),
}

_MODELS: dict[str, Any] = {}


def _models() -> dict[str, Any]:
    """Lazily map artifact model names to classes (avoids an import cycle)."""
    if not _MODELS:
        from . import models as m

        for model_name, _file in _ARTIFACT_FILES.values():
            _MODELS[model_name] = getattr(m, model_name)
    return _MODELS


def default_stages() -> list[Stage]:
    from .stages import build_default_stages

    return build_default_stages()
