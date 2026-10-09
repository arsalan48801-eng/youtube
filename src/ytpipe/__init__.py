"""ytpipe — fetch a source video, learn from it, cut a new one, ship it.

    from ytpipe.config import build_settings
    from ytpipe.models import JobRequest
    from ytpipe.pipeline import Pipeline

    pipe = Pipeline(build_settings())
    job = pipe.run(JobRequest(source="https://youtu.be/dQw4w9WgXcQ", target_seconds=45))

Public surface is intentionally small: `Pipeline` for library use, `ytpipe` (CLI) and
the FastAPI app for everything else.
"""

from .config import Settings, build_settings
from .errors import (
    ConfigError,
    FetchError,
    PipelineError,
    PublishError,
    RenderError,
    ScriptError,
    TranscribeError,
)
from .models import (
    Aspect,
    Job,
    JobRequest,
    PipelineArtifacts,
    StageName,
    Status,
)
from .pipeline import Pipeline, StageContext

__version__ = "0.1.0"

__all__ = [
    "Aspect",
    "ConfigError",
    "FetchError",
    "Job",
    "JobRequest",
    "Pipeline",
    "PipelineArtifacts",
    "PipelineError",
    "PublishError",
    "RenderError",
    "ScriptError",
    "Settings",
    "StageContext",
    "StageName",
    "Status",
    "TranscribeError",
    "__version__",
    "build_settings",
]
