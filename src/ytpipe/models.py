"""Domain models shared by every stage of the pipeline.

Everything a stage consumes and produces is a pydantic model, which gives us
free validation, JSON round-tripping (used for artifacts on disk) and a stable
contract between stages that are otherwise independently swappable.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Aspect(str, enum.Enum):
    """Output canvas. Vertical is the default because the target is Shorts/Reels."""

    VERTICAL = "9:16"
    HORIZONTAL = "16:9"
    SQUARE = "1:1"

    @property
    def size(self) -> tuple[int, int]:
        return {
            Aspect.VERTICAL: (1080, 1920),
            Aspect.HORIZONTAL: (1920, 1080),
            Aspect.SQUARE: (1080, 1080),
        }[self]


class StageName(str, enum.Enum):
    FETCH = "fetch"
    TRANSCRIBE = "transcribe"
    ANALYZE = "analyze"
    SCRIPT = "script"
    STORYBOARD = "storyboard"
    GENERATE = "generate"
    EDIT = "edit"
    PUBLISH = "publish"

    @classmethod
    def ordered(cls) -> list[StageName]:
        return [
            cls.FETCH,
            cls.TRANSCRIBE,
            cls.ANALYZE,
            cls.SCRIPT,
            cls.STORYBOARD,
            cls.GENERATE,
            cls.EDIT,
            cls.PUBLISH,
        ]


class Status(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class _Model(BaseModel):
    """Base for every model: tolerate forward-compatible extras, keep enums as enums.

    ``use_enum_values`` is deliberately NOT enabled. ``Aspect``/``Status`` are
    str-mixin enums, whose ``__hash__`` is the member name rather than the string
    value -- flattening them to plain str silently breaks dict lookups keyed by the
    enum (e.g. ``Aspect.size``). Validation coerces incoming strings to members.
    """

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------
# Stage 1-2: source + transcript
# --------------------------------------------------------------------------


class SourceVideo(_Model):
    """The raw material: whatever we pulled off YouTube (or a local file)."""

    video_id: str | None = None
    url: str | None = None
    title: str = ""
    channel: str | None = None
    description: str = ""
    duration: float = 0.0
    publish_date: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    language: str | None = None
    media_path: Path | None = None
    poster_path: Path | None = None
    subtitle_path: Path | None = None
    stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_media(self) -> bool:
        return self.media_path is not None and Path(self.media_path).exists()


class CaptionSegment(_Model):
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def words(self) -> list[str]:
        return self.text.split()


class Transcript(_Model):
    source_id: str | None = None
    language: str | None = None
    provider: str = "unknown"
    segments: list[CaptionSegment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def window(self, start: float, end: float) -> list[CaptionSegment]:
        """Caption cues overlapping [start, end) — used to cut burned-in subs."""
        return [s for s in self.segments if s.end > start and s.start < end]


# --------------------------------------------------------------------------
# Stage 3: analysis of what made the source work
# --------------------------------------------------------------------------


class Beat(_Model):
    """A narrative beat lifted from the source: a span, its job, its hook."""

    start: float = 0.0
    end: float = 0.0
    role: str = "body"  # hook | setup | payoff | cta | body
    summary: str = ""


class Brief(_Model):
    """What we learned from the source, expressed as *constraints* for the writer.

    The writer never copies the transcript; it inherits the shape that made the
    source hold attention (line length, cadence, hook type, which beats carried the
    payload) and the factual sentences worth keeping.
    """

    title: str = ""
    topic: str = ""
    hook_style: str = ""
    tone: str = ""
    words: int = 0
    wpm: float = 150.0
    sentence_count: int = 0
    avg_sentence_words: float = 0.0
    question_ratio: float = 0.0
    second_person_ratio: float = 0.0
    beats: list[Beat] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    retention_notes: list[str] = Field(default_factory=list)

    # -- extras the writer consumes directly --------------------------------
    readability: float = 0.0  # Flesch-Kincaid grade level
    hook_line: str = ""
    hook_seconds: float = 0.0
    cta_line: str = ""
    top_sentences: list[str] = Field(default_factory=list)
    sentence_durations: list[float] = Field(default_factory=list)
    transcript_covered: bool = True


# --------------------------------------------------------------------------
# Stage 4-5: the script we will actually shoot
# --------------------------------------------------------------------------


class ScriptLine(_Model):
    """One spoken line, with its on-screen text and its b-roll instruction."""

    index: int = 0
    role: str = "body"
    voiceover: str = ""
    on_screen: str = ""
    visual: str = ""
    start: float = 0.0
    end: float = 0.0
    #: Where the words actually land inside [start, end]. The picture owns the whole
    #: window; captions and audio are placed against this narrower one.
    speech_start: float = 0.0
    speech_end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Script(_Model):
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    lines: list[ScriptLine] = Field(default_factory=list)
    provider: str = "unknown"
    model: str | None = None
    target_seconds: float = 45.0
    wpm: float = 150.0
    prompts_used: list[str] = Field(default_factory=list)

    @property
    def voiceover(self) -> str:
        return " ".join(ln.voiceover for ln in self.lines)

    @property
    def word_count(self) -> int:
        return len(self.voiceover.split())

    @property
    def duration(self) -> float:
        return self.lines[-1].end if self.lines else 0.0


class VisualKind(str, enum.Enum):
    SOURCE_CLIP = "source_clip"  # cut from the fetched video
    GENERATED = "generated"      # still/image from an image provider
    CARD = "card"                # procedural text card, drawn by ffmpeg
    SILENT = "silent"            # placeholder, keeps timing honest


class VisualSpec(_Model):
    kind: VisualKind = VisualKind.CARD
    prompt: str = ""
    source_start: float | None = None
    source_end: float | None = None
    zoom: float = 1.06
    pan: str = "left-right"
    background: str = "0x0b1220"


class CaptionCue(_Model):
    start: float
    end: float
    text: str
    words: list[str] = Field(default_factory=list)


class Shot(_Model):
    index: int
    start: float
    end: float
    speech_start: float = 0.0
    speech_end: float = 0.0
    narration: str = ""
    on_screen: str = ""
    visual: VisualSpec = Field(default_factory=VisualSpec)
    captions: list[CaptionCue] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Storyboard(_Model):
    aspect: Aspect = Aspect.VERTICAL
    fps: int = 30
    width: int = 1080
    height: int = 1920
    shots: list[Shot] = Field(default_factory=list)
    captions: list[CaptionCue] = Field(default_factory=list)
    music: str | None = None
    loudness_target: float = -14.0  # LUFS, YouTube's recommendation

    @property
    def duration(self) -> float:
        return self.shots[-1].end if self.shots else 0.0


# --------------------------------------------------------------------------
# Stage 6-8: assets, the render, the upload
# --------------------------------------------------------------------------


class ShotAsset(_Model):
    shot_index: int
    video_path: Path | None = None
    audio_path: Path | None = None
    duration: float = 0.0
    #: True when ``video_path`` is an image to be looped, so the assembly pass feeds
    #: it through ``-loop 1`` instead of pre-encoding a clip it would only re-encode.
    is_still: bool = False
    narration_provider: str = "silent"
    visual_provider: str = "procedural"
    notes: str = ""


class AssetSet(_Model):
    assets: list[ShotAsset] = Field(default_factory=list)
    total_seconds: float = 0.0
    budget_seconds: float = 0.0


class CaptionFiles(_Model):
    srt: Path | None = None
    ass: Path | None = None


class RenderResult(_Model):
    path: Path
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: int = 30
    size_bytes: int = 0
    video_codec: str = "h264"
    audio_codec: str = "aac"
    captions: CaptionFiles = Field(default_factory=CaptionFiles)
    thumbnail_path: Path | None = None
    shot_count: int = 0
    render_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class PublishResult(_Model):
    status: str = "dry-run"
    video_id: str | None = None
    url: str | None = None
    title: str = ""
    privacy: str = "private"
    manifest_path: Path | None = None
    uploaded_at: datetime | None = None
    detail: str = ""


class StageRun(_Model):
    stage: StageName
    status: Status = Status.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float = 0.0
    attempts: int = 0
    artifact: str | None = None
    error: str | None = None
    log: list[str] = Field(default_factory=list)


class JobRequest(_Model):
    source: str
    topic: str | None = None
    aspect: Aspect = Aspect.VERTICAL
    target_seconds: float = 45.0
    wpm: float = 150.0
    title_hint: str | None = None
    offline: bool = False
    #: Upload for real. False keeps stage 8 a manifest-only dry run.
    publish: bool = False
    privacy: str | None = None  # private | unlisted | public
    #: Caption/ASR language preference, e.g. "en", "es", "pt-BR".
    language: str | None = None
    only: list[StageName] | None = None
    until: StageName | None = None
    resume: bool = False
    #: Explicit caption track for a local source, so the file name does not have to match
    #: the clip's. Ignored for a YouTube URL, which already knows its own tracks.
    caption_track: Path | None = None
    #: Read the link instead of downloading it: metadata + caption track only. The edit
    #: then has no source footage to reuse, so visuals fall back to cards and stills.
    transcript_only: bool = False
    #: Optional music bed, ducked under the voiceover at the final pass.
    music_path: Path | None = None
    music_gain_db: float = -18.0
    #: Style overrides for the burned-in captions.
    caption_max_words: int | None = None
    captions: bool = True
    transition: str = "dip"  # cut | dip | xfade


class Job(_Model):
    """One end-to-end run. Persisted to sqlite; artifacts on disk."""

    id: str
    request: JobRequest
    status: Status = Status.PENDING
    workdir: Path
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    stages: dict[str, StageRun] = Field(default_factory=dict)
    error: str | None = None

    def output(self, name: str = "final.mp4") -> Path:
        return self.workdir / "out" / name

    def stage_dir(self, stage: StageName | str) -> Path:
        return self.workdir / "stages" / str(getattr(stage, "value", stage))

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PipelineArtifacts(_Model):
    """Everything produced by a run, handed stage-to-stage."""

    source: SourceVideo | None = None
    transcript: Transcript | None = None
    brief: Brief | None = None
    script: Script | None = None
    storyboard: Storyboard | None = None
    assets: AssetSet | None = None
    render: RenderResult | None = None
    publish: PublishResult | None = None
