"""The front door: how a `--source` string becomes a fetch plan, and what a failure says.

These are cheap to get wrong and expensive to debug remotely, so they are tested
directly -- the network itself is never touched.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from ytpipe.errors import FetchError
from ytpipe.models import JobRequest
from ytpipe.stages.fetch import explain_failure, resolve_source


class TestResolveSource:
    def test_a_shorts_url_is_passed_through_untouched(self):
        url = "https://youtube.com/shorts/CRGWDiIgMZ4?si=qDhfATJ2BcsFGGwF"
        assert resolve_source(url) == (url, None)

    def test_a_bare_id_becomes_a_canonical_watch_url(self):
        assert resolve_source("CRGWDiIgMZ4") == ("https://www.youtube.com/watch?v=CRGWDiIgMZ4", None)

    def test_an_existing_file_is_a_local_source(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"ftyp")
        url, local = resolve_source(str(clip))
        assert url is None
        assert local == str(clip.resolve())

    def test_a_missing_path_or_junk_is_rejected_with_the_value_that_failed(self):
        with pytest.raises(FetchError) as exc:
            resolve_source("  not a url, not a file ")
        assert "not a url" in str(exc.value)

    def test_offline_mode_says_what_to_do_instead(self, settings, tmp_path):
        """The one fetch failure that is a feature, not a bug, must not read like a crash."""
        from ytpipe.models import JobRequest, PipelineArtifacts, StageName
        from ytpipe.pipeline import StageContext
        from ytpipe.stages import FetchStage
        from ytpipe.state import JobStore

        store = JobStore(tmp_path / "db.sqlite3")
        job = store.create(
            JobRequest(source="https://youtu.be/CRGWDiIgMZ4", offline=True),
            job_id="off",
            workdir=tmp_path / "off",
        )
        ctx = StageContext(
            job=job, settings=settings, store=store,
            artifacts=PipelineArtifacts(), stage=StageName.FETCH,
        )
        with pytest.raises(FetchError) as exc:
            FetchStage().run(ctx)
        assert "local file" in str(exc.value)


class TestExplainFailure:
    def test_network_block_is_reported_as_a_block_with_a_way_out(self):
        raw = (
            "ERROR: CRGWDiIgMZ4: Unable to download API page: TLS/SSL connection "
            "has been closed (EOF) (_ssl.c:992); please report this issue on "
            "https://github.com/yt-dlp/yt-dlp/issues?q= , filling out the appropriate issue "
            "template. Confirm you are on the latest version using  yt-dlp -U"
        )
        msg = explain_failure(raw)
        assert "unreachable" in msg
        assert "ytpipe run /path/to/clip.mp4" in msg
        assert "github.com/yt-dlp" not in msg, "issue-tracker boilerplate is not for users"
        assert len(msg) < 400

    def test_a_bad_video_id_stays_terse_and_gives_no_hint(self):
        msg = explain_failure("ERROR: [youtube] CRGWDiIgMZ4: Video unavailable")
        assert msg.startswith("ERROR: [youtube]")
        assert "unreachable" not in msg

    def test_empty_input_still_produces_a_message(self):
        assert explain_failure("") == "yt-dlp failed"


VTT = """WEBVTT

00:00:00.000 --> 00:00:03.000
The spec sheet hides one number that matters.

00:00:03.000 --> 00:00:06.000
We measured four panels for thirty days.
"""


class _FakeDL:
    """Stands in for yt_dlp.YoutubeDL: records the options, drops a caption file."""

    last_options: dict = {}
    last_download: bool | None = None
    captions = True

    def __init__(self, options):
        _FakeDL.last_options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        _FakeDL.last_download = download
        out = Path(_FakeDL.last_options["outtmpl"]).parent
        if type(self).captions:
            (out / "source.en.vtt").write_text(VTT, encoding="utf-8")
            (out / "source.en.vtt.info.json").write_text("{}", encoding="utf-8")
        if download:
            media = out / "source.mp4"
            media.write_bytes(b"ftyp" * 64)
            self.requested = str(media)
        return {
            "id": "CRGWDiIgMZ4",
            "title": "The monitor spec nobody checks",
            "duration": 31.0,
            "uploader": "Panel Test",
            "webpage_url": "https://www.youtube.com/watch?v=CRGWDiIgMZ4",
            "requested_downloads": [{"filepath": getattr(self, "requested", "")}],
        }


@pytest.fixture
def fake_ydl(monkeypatch):
    """Install the fake as the `yt_dlp` module the stage imports inside its function."""
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = _FakeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    _FakeDL.captions = True
    return _FakeDL


def _fetch(settings, tmp_path, **request_kwargs):
    from ytpipe.models import PipelineArtifacts, StageName
    from ytpipe.pipeline import StageContext
    from ytpipe.stages import FetchStage
    from ytpipe.state import JobStore

    settings.offline = False
    store = JobStore(tmp_path / "db.sqlite3")
    job = store.create(
        JobRequest(source="https://youtube.com/shorts/CRGWDiIgMZ4?si=qDhfATJ2BcsFGGwF", **request_kwargs),
        job_id="ydl",
        workdir=tmp_path / "job",
    )
    ctx = StageContext(job=job, settings=settings, store=store, artifacts=PipelineArtifacts(), stage=StageName.FETCH)
    return FetchStage().run(ctx), ctx


class TestTranscriptOnly:
    """`--transcript-only` reads the link instead of downloading it."""

    def test_no_media_is_requested_and_the_track_is_picked_up(
        self, settings, tmp_path, fake_ydl, caplog
    ):
        caplog.set_level(logging.INFO, logger="ytpipe")
        source, _ = _fetch(settings, tmp_path, transcript_only=True)
        assert fake_ydl.last_download is False, "extract_info must be called with download=False"
        assert fake_ydl.last_options.get("skip_download") is True
        assert "format" not in fake_ydl.last_options, "no stream selection when nothing is downloaded"
        assert not fake_ydl.last_options.get("writethumbnail")
        assert source.media_path is None and not source.has_media
        assert source.duration == 31.0 and source.channel == "Panel Test"
        assert source.subtitle_path and source.subtitle_path.exists()
        assert "no download" in caplog.text, "the log must say nothing was fetched"

    def test_the_empty_download_leaves_the_rest_of_the_pipeline_card_only(self, settings, tmp_path, fake_ydl):
        source, _ = _fetch(settings, tmp_path, transcript_only=True)
        from ytpipe.stages.storyboard import StoryboardStage

        assert StoryboardStage._source_candidates(source, None) == []

    def test_a_video_without_captions_says_so_and_points_at_the_way_in(
        self, settings, tmp_path, fake_ydl
    ):
        fake_ydl.captions = False
        with pytest.raises(FetchError) as exc:
            _fetch(settings, tmp_path, transcript_only=True)
        msg = str(exc.value)
        assert "no caption track" in msg and "--transcript-only" in msg

    def test_a_normal_fetch_still_downloads_the_media(self, settings, tmp_path, fake_ydl):
        source, _ = _fetch(settings, tmp_path)
        assert fake_ydl.last_download is True
        assert "1080" in fake_ydl.last_options["format"]
        assert source.has_media
class TestLocalCaptionTrack:
    """A local clip and its captions do not have to share a file name."""

    def _run(self, settings, tmp_path, clip, **kwargs):
        from ytpipe.models import PipelineArtifacts, StageName
        from ytpipe.pipeline import StageContext
        from ytpipe.stages import FetchStage
        from ytpipe.state import JobStore

        store = JobStore(tmp_path / "db.sqlite3")
        job = store.create(JobRequest(source=str(clip), **kwargs), job_id="loc", workdir=tmp_path / "j")
        ctx = StageContext(job=job, settings=settings, store=store, artifacts=PipelineArtifacts(), stage=StageName.FETCH)
        return FetchStage().run(ctx)

    def test_a_sibling_track_is_still_found_automatically(self, settings, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        (tmp_path / "clip.vtt").write_text(VTT, encoding="utf-8")
        source = self._run(settings, tmp_path, clip)
        assert source.subtitle_path and source.subtitle_path.name == "clip.vtt"

    def test_an_explicit_track_wins_over_the_sibling_search(self, settings, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        (tmp_path / "clip.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nwrong track\n", encoding="utf-8")
        chosen = tmp_path / "captions-i-actually-have.srt"
        chosen.write_text("1\n00:00:00,000 --> 00:00:02,000\nThe real one.\n", encoding="utf-8")
        source = self._run(settings, tmp_path, clip, caption_track=chosen)
        assert source.subtitle_path == chosen

    def test_a_missing_or_wrong_kind_of_track_is_refused_by_name(self, settings, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        with pytest.raises(FetchError) as exc:
            self._run(settings, tmp_path, clip, caption_track=tmp_path / "nope.vtt")
        assert "does not exist" in str(exc.value)
        with pytest.raises(FetchError) as exc:
            self._run(settings, tmp_path, clip, caption_track=clip)
        assert "not a .vtt/.srt/.ass" in str(exc.value)


def _make_clip(path: Path) -> Path:
    """A real, tiny, probeable file: fetch reads its duration, so bytes must be valid."""
    from ytpipe.ffmpeg import FFmpeg

    FFmpeg().run(
        ["-f", "lavfi", "-i", "testsrc2=size=192x108:rate=12:duration=2", "-c:v", "libx264",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", str(path)],
        timeout=120,
    )
    return path
