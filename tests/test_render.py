"""Real renders. These are the only tests that prove the pipeline makes a playable file.

They run ffmpeg (skipped if it is missing) at a draft canvas so the suite stays
fast, and they assert on the properties that actually break in production: total
length vs the storyboard, caption timings never exceeding the video, audio presence,
and the frame geometry.
"""

from __future__ import annotations

import pytest

from ytpipe.ffmpeg import FFmpeg
from ytpipe.models import JobRequest, StageName, Status


def _ffmpeg_or_none():
    try:
        ff = FFmpeg()
        ff.version()
        return ff
    except Exception:
        return None


HAVE_FFMPEG = _ffmpeg_or_none() is not None
pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg is not installed")


@pytest.fixture
def render_job(pipeline, local_source, tmp_path):
    """One offline end-to-end run, reused by every assertion in this module."""
    job = pipeline.run(
        JobRequest(
            source=str(local_source),
            topic="cable failures",
            target_seconds=10.0,
            wpm=180.0,
            offline=True,
        ),
        job_id="render-main",
    )
    return job


def test_end_to_end_render_is_playable(pipeline, render_job, ffmpeg):
    assert render_job.status == Status.DONE, render_job.error
    render = pipeline.store.get(render_job.id)
    assert all(
        r.status == Status.DONE for name, r in render.stages.items() if name != "publish"
    )

    info = ffmpeg.probe(render_job.output("final.mp4"))
    storyboard = pipeline.load_artifact(render_job, StageName.STORYBOARD)

    assert info.width == 270 and info.height == 480  # 9:16 at render_scale 0.25
    assert info.has_audio, "a silent export is a failed export"
    assert int(round(info.fps)) == 30
    assert info.vcodec == "h264" and info.acodec == "aac"

    planned = storyboard.shots[-1].end + pipeline.settings.tail_seconds
    assert info.duration == pytest.approx(planned, abs=0.2), "picture length must equal the timeline"
    script = pipeline.load_artifact(render_job, StageName.SCRIPT)
    roles = [ln.role for ln in script.lines]
    assert len(storyboard.shots) == len(roles) >= 3
    assert roles[0] == "hook" and roles[-1] == "cta", "trimming may cut a beat, never an end"
    assert storyboard.duration <= render_job.request.target_seconds * 1.15 + 0.5


def test_captions_never_outrun_the_picture(pipeline, render_job, ffmpeg):
    """The drift bug: concat drops inter-shot gaps, so absolute caption times can end
    up past the end of the video -- the last cue gets cut off and nobody notices."""
    storyboard = pipeline.load_artifact(render_job, StageName.STORYBOARD)
    duration = ffmpeg.duration(render_job.output("final.mp4"))
    assert storyboard.captions
    assert max(c.end for c in storyboard.captions) <= duration + 0.05
    assert min(c.start for c in storyboard.captions) >= 0.0
    srt = render_job.output("final.srt").read_text(encoding="utf-8")
    assert srt.count("-->") == len(storyboard.captions)
    ass = render_job.output("final.ass").read_text(encoding="utf-8")
    assert ass.count("Dialogue:") >= len(storyboard.captions)
    assert "Style: POP," in ass and "Style: CAP," in ass


def test_generated_files_are_all_present(pipeline, render_job):
    out = render_job.output("final.mp4").parent
    for name in ("final.mp4", "final.srt", "final.ass", "thumb.jpg"):
        assert (out / name).exists(), name
    assert (out / "final.mp4").stat().st_size > 20_000
    assert (out / "thumb.jpg").stat().st_size > 1000


@pytest.mark.parametrize("transition", ["cut", "dip", "xfade"])
def test_each_transition_renders_without_warnings(pipeline, local_source, transition, tmp_path):
    job = pipeline.run(
        JobRequest(
            source=str(local_source),
            topic="cable failures",
            target_seconds=8.0,
            wpm=190.0,
            offline=True,
            transition=transition,
        ),
        job_id=f"render-{transition}",
    )
    assert job.status == Status.DONE, job.error
    render = pipeline.load_artifact(job, StageName.EDIT)
    assert render.warnings == [], render.warnings
    assert render.duration > 2.0
    assert render.shot_count >= 3


def test_music_bed_is_mixed_in(pipeline, ffmpeg, local_source, tmp_path):
    bed = tmp_path / "bed.mp3"
    ffmpeg.tone(bed, 20.0, freq=180, gain_db=-12)
    job = pipeline.run(
        JobRequest(
            source=str(local_source),
            topic="cable failures",
            target_seconds=8.0,
            wpm=190.0,
            offline=True,
            music_path=bed,
        ),
        job_id="render-music",
    )
    assert job.status == Status.DONE, job.error
    info = ffmpeg.probe(job.output("final.mp4"))
    render = pipeline.load_artifact(job, StageName.EDIT)
    assert info.has_audio and render.warnings == []


def test_horizontal_canvas_is_honoured(pipeline, local_source):
    job = pipeline.run(
        JobRequest(
            source=str(local_source),
            topic="cable failures",
            target_seconds=8.0,
            wpm=190.0,
            offline=True,
            aspect="16:9",
        ),
        job_id="render-wide",
    )
    assert job.status == Status.DONE, job.error
    render = pipeline.load_artifact(job, StageName.EDIT)
    assert render.width > render.height
