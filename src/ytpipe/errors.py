"""Typed stage errors. `retryable` drives the orchestrator's backoff policy."""

from __future__ import annotations


class PipelineError(Exception):
    stage: str = "pipeline"
    retryable: bool = False

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if stage:
            self.stage = stage

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.stage}] {self.message}"


class ConfigError(PipelineError):
    stage = "config"


class FetchError(PipelineError):
    stage = "fetch"
    retryable = True


class TranscribeError(PipelineError):
    stage = "transcribe"
    retryable = True


class ScriptError(PipelineError):
    stage = "script"
    retryable = True


class RenderError(PipelineError):
    stage = "edit"


class PublishError(PipelineError):
    stage = "publish"
    retryable = True
