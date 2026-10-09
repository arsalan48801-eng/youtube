"""Runtime configuration.

One object, built from env vars / .env, injected into every stage. Nothing in
`stages/` reads `os.environ` directly — that is what makes the providers easy to
swap in tests and what keeps `--offline` a single switch.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Any OpenAI-compatible /chat/completions endpoint works here."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_s: float = 120.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class TTSSettings(BaseSettings):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini-tts"
    voice: str = "nova"
    speed: float = 1.0
    timeout_s: float = 180.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class ImageSettings(BaseSettings):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-image-1"
    size: str = "1024x1536"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class YouTubeSettings(BaseSettings):
    """Data API v3 upload credentials. Without them publish stays a dry run."""

    client_id: str = ""
    client_secret: str = ""
    refresh_token: str = ""
    privacy: str = "private"  # private | unlisted | public
    category_id: str = "27"  # Education
    made_for_kids: bool = False
    #: Set True when the output contains realistic synthetic media; YouTube exposes
    #: this as an AI-disclosure flag and Studio expects it to be declared.
    contains_synthetic_media: bool = True
    dry_run: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YTPIPE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    workdir: Path = Path("work")
    log_level: str = "INFO"

    #: Force deterministic, no-network code paths everywhere.
    offline: bool = False
    #: Render even when a shot has no narration audio.
    allow_silent: bool = True
    #: Longest b-roll window taken from the source, in seconds. Short windows keep
    #: the cut rhythm alive and reduce the "reused clip" feel.
    max_source_span: float = 3.5
    ffmpeg_binary: str | None = None
    #: Cap concurrent provider calls during `generate`.
    max_workers: int = 4
    #: Local ASR fallback when a video has no caption track. Opt-in: pulls a model.
    asr_enabled: bool = False
    asr_model: str = "base.en"
    #: Extra hold after the last shot. Shorts are often trimmed at the tail by the
    #: platform, so a beat of padding protects the closing frame.
    tail_seconds: float = 0.25
    #: Silence before the first word, in seconds. Zero makes the cut feel abrupt.
    lead_in: float = 0.35
    #: Canvas multiplier. 0.5 renders 540x960 for a ~4x faster draft pass.
    render_scale: float = 1.0
    #: Retries *after* the first attempt, for stages flagged retryable.
    max_retries: int = 2
    retry_backoff: float = 1.5
    #: Stage 8 is destructive/irreversible in the real world, so it stays off unless
    #: both keys and an explicit switch are present.
    allow_publish: bool = False

    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)
    youtube: YouTubeSettings = Field(default_factory=YouTubeSettings)

    # -- paths ---------------------------------------------------------------

    @property
    def jobs_dir(self) -> Path:
        return self.workdir / "jobs"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def db_path(self) -> Path:
        return self.workdir / "ytpipe.sqlite3"

    def ensure_dirs(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # -- providers -----------------------------------------------------------

    @property
    def llm_available(self) -> bool:
        return not self.offline and self.llm.enabled

    @property
    def tts_available(self) -> bool:
        return not self.offline and self.tts.enabled

    @property
    def images_available(self) -> bool:
        return not self.offline and self.images.enabled

    @property
    def publish_available(self) -> bool:
        yt = self.youtube
        return not self.offline and bool(yt.refresh_token and yt.client_id and not yt.dry_run)

    def render_defaults(self) -> dict[str, object]:
        return {"crf": 20, "preset": "veryfast", "fps": 30, "bitrate": "6M"}

    def describe_providers(self) -> dict[str, str]:
        return {
            "llm": f"{self.llm.model}" if self.llm_available else "offline template writer",
            "tts": f"{self.tts.model}/{self.tts.voice}" if self.tts_available else "silent timing track",
            "images": self.images.model if self.images_available else "procedural cards",
            "publish": "YouTube Data API v3" if self.publish_available else "dry run (manifest only)",
        }

    # -- binaries ------------------------------------------------------------

    @property
    def ffmpeg(self) -> str:
        """Resolved ffmpeg path: explicit setting, PATH, then the PyPI wheel."""
        if self.ffmpeg_binary:
            return self.ffmpeg_binary
        found = shutil.which("ffmpeg")
        if found:
            return found
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # pragma: no cover - only when the wheel is absent
            raise FileNotFoundError(
                "no ffmpeg found; install it or set YTPIPE_FFMPEG_BINARY"
            ) from None


def _nested(prefix: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Collect `YTPIPE_LLM_*`-style vars into the kwargs of a sub-settings object."""
    out: dict[str, str] = {}
    src = env if env is not None else os.environ
    head = f"{prefix}_"
    for key, value in src.items():
        if key.startswith(head):
            out[key[len(head) :].lower()] = value
    return out


def build_settings(env: dict[str, str] | None = None, **overrides: object) -> Settings:
    """Build Settings from `YTPIPE_*` environment variables plus a local .env.

    Scalar fields are mapped generically by their annotation, so adding a setting to
    the model makes it configurable -- an explicit key list here silently ignored
    `YTPIPE_RENDER_SCALE` for a while, which is exactly the bug this avoids.
    """
    src = dict(env) if env is not None else dict(os.environ)
    if Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            src.setdefault(key.strip(), value.strip().strip("'\""))

    kwargs: dict[str, object] = {}
    for name, field in Settings.model_fields.items():
        if name in {"llm", "tts", "images", "youtube"}:
            continue
        raw = src.get(f"YTPIPE_{name.upper()}")
        if raw is None:
            continue
        kwargs[name] = _coerce(raw, field.annotation)
    for prefix, model in (
        ("LLM", LLMSettings),
        ("TTS", TTSSettings),
        ("IMAGE", ImageSettings),
        ("YOUTUBE", YouTubeSettings),
    ):
        sub = _nested(f"YTPIPE_{prefix}", src)
        kwargs["images" if prefix == "IMAGE" else prefix.lower()] = model(**sub)

    kwargs.update(overrides)
    return Settings(**kwargs)


def _coerce(raw: str, annotation: object) -> object:
    """Cast an env string to the field's declared type, tolerating Optional[...] unions."""
    types = {
        t for t in getattr(annotation, "__args__", (annotation,)) if t is not type(None)
    }
    target = next(iter(types), str)
    if target is bool:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if target is int:
        return int(raw)
    if target is float:
        return float(raw)
    if target is Path:
        return Path(raw)
    return str(raw)


@lru_cache
def get_settings() -> Settings:
    return build_settings()
