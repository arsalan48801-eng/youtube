"""Shared fixtures.

The demo clip is *generated* by ffmpeg rather than committed, so the suite has no
binary blobs and still exercises a real decode/encode path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

from ytpipe.config import build_settings  # noqa: E402
from ytpipe.ffmpeg import FFmpeg  # noqa: E402
from ytpipe.models import JobRequest  # noqa: E402
from ytpipe.pipeline import Pipeline  # noqa: E402

TRANSCRIPT_LINES = [
    "Most people buy the wrong monitor and it costs them a year.",
    "The spec sheet hides one number that actually matters.",
    "We tested four panels at 120 hertz for thirty days.",
    "Only two kept the refresh rate locked under load.",
    "The failure was never the panel, it was the cable.",
    "Fix the bandwidth first and everything else follows.",
]

VTT = "WEBVTT\n\n" + "\n\n".join(
    f"00:00:{i * 3:02d}.000 --> 00:00:{i * 3 + 3:02d}.000\n{line}"
    for i, line in enumerate(TRANSCRIPT_LINES)
)


@pytest.fixture(scope="session")
def ffmpeg() -> FFmpeg:
    ff = FFmpeg()
    try:
        ff.version()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"ffmpeg unavailable: {exc}")
    return ff


@pytest.fixture(scope="session")
def source_clip(ffmpeg: FFmpeg, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """18s of testsrc2 at 1280x720 with a tone, plus a matching VTT."""
    clip = tmp_path_factory.mktemp("src") / "demo.mp4"
    ffmpeg.run(
        [
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=18",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=18",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
            "-c:a", "aac", "-shortest", str(clip),
        ],
        timeout=300,
    )
    assert clip.exists() and clip.stat().st_size > 1000
    return clip


@pytest.fixture
def transcript_vtt(tmp_path: Path) -> Path:
    path = tmp_path / "demo.vtt"
    path.write_text(VTT, encoding="utf-8")
    return path


@pytest.fixture
def local_source(source_clip: Path, transcript_vtt: Path, tmp_path: Path) -> Path:
    """A directory holding clip + sibling captions, as `--source` would receive it."""
    target = tmp_path / "media"
    target.mkdir(exist_ok=True)
    clip = target / "demo.mp4"
    subprocess.run(["cp", str(source_clip), str(clip)], check=True)
    (target / "demo.vtt").write_text(transcript_vtt.read_text(encoding="utf-8"), encoding="utf-8")
    return clip


@pytest.fixture
def settings(tmp_path: Path) -> object:
    """Offline, half-resolution draft canvas, no providers, isolated workdir."""
    return build_settings(
        env={
            "YTPIPE_WORKDIR": str(tmp_path / "work"),
            "YTPIPE_OFFLINE": "1",
            "YTPIPE_RENDER_SCALE": "0.25",
            "YTPIPE_MAX_RETRIES": "1",
            "YTPIPE_RETRY_BACKOFF": "0.01",
        }
    )


@pytest.fixture
def pipeline(settings) -> Pipeline:
    return Pipeline(settings)


@pytest.fixture
def job_request(local_source: Path) -> JobRequest:
    return JobRequest(
        source=str(local_source),
        topic="monitor buying mistakes",
        target_seconds=12.0,
        wpm=170.0,
        offline=True,
    )
