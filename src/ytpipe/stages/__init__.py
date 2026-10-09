"""Stage registry. Order here is the order the orchestrator runs."""

from __future__ import annotations

from .analyze import AnalyzeStage
from .edit import EditStage
from .fetch import FetchStage
from .generate import GenerateStage
from .publish import PublishStage
from .script import ScriptStage
from .storyboard import StoryboardStage
from .transcribe import TranscribeStage

__all__ = [
    "FetchStage",
    "TranscribeStage",
    "AnalyzeStage",
    "ScriptStage",
    "StoryboardStage",
    "GenerateStage",
    "EditStage",
    "PublishStage",
    "build_default_stages",
]


def build_default_stages() -> list:
    return [
        FetchStage(),
        TranscribeStage(),
        AnalyzeStage(),
        ScriptStage(),
        StoryboardStage(),
        GenerateStage(),
        EditStage(),
        PublishStage(),
    ]
