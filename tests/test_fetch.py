"""The front door: how a `--source` string becomes a fetch plan, and what a failure says.

These are cheap to get wrong and expensive to debug remotely, so they are tested
directly -- the network itself is never touched.
"""

from __future__ import annotations

import pytest

from ytpipe.errors import FetchError
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
