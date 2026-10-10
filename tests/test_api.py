"""HTTP surface: queueing, polling, artifact serving and path safety."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ytpipe.api import create_app
from ytpipe.models import StageName, Status


@pytest.fixture
def client(settings):
    app = create_app(settings, max_workers=2)
    with TestClient(app) as c:
        yield c


def wait_for(client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in (Status.DONE.value, Status.FAILED.value) and not job["running"]:
            return job
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


class TestMeta:
    def test_healthz_reports_ffmpeg_and_provider_selection(self, client):
        data = client.get("/healthz").json()
        assert data["ok"] is True
        assert "ffmpeg" in data and data["ffmpeg"].startswith("ffmpeg version")
        assert data["providers"]["publish"].startswith("dry run")
        assert data["providers"]["llm"] == "offline template writer"
        assert data["offline"] is True

    def test_dashboard_is_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "ytpipe" in resp.text and "<video" in resp.text

    def test_openapi_lists_the_job_routes(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for route in ("/jobs", "/jobs/{job_id}", "/jobs/{job_id}/events", "/jobs/{job_id}/video"):
            assert route in paths


class TestJobLifecycle:
    def test_validation_errors(self, client, local_source):
        assert client.post("/jobs", json={}).status_code == 422
        assert client.post("/jobs", json={"source": str(local_source), "target_seconds": 2}).status_code == 422
        assert client.post("/jobs", json={"source": str(local_source), "aspect": "4:3"}).status_code == 422

    def test_queue_poll_and_read_results(self, client, local_source):
        resp = client.post(
            "/jobs",
            json={
                "source": str(local_source),
                "topic": "cable failures",
                "target_seconds": 10,
                "wpm": 190,
                "offline": True,
                "until": "script",
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert body["poll"].startswith("/jobs/")

        job = wait_for(client, body["job_id"])
        assert job["status"] == "done", job["error"]
        assert set(job["stages"]) == {s.value for s in StageName}
        assert job["stages"]["script"]["status"] == "done"
        assert job["stages"]["edit"]["status"] == "pending"
        assert job["artifacts"]["brief"]["wpm"] > 0

        script = client.get(f"/jobs/{body['job_id']}/script").json()
        assert script["lines"] and script["lines"][0]["role"] == "hook"
        assert all(ln["end"] >= ln["start"] for ln in script["lines"])

        events = client.get(f"/jobs/{body['job_id']}/events").json()["events"]
        assert any(e["stage"] == "script" for e in events)
        assert any("complete" in e["message"] for e in events)

    def test_listing_shows_recent_jobs(self, client, local_source):
        client.post("/jobs", json={"source": str(local_source), "offline": True, "until": "fetch"})
        data = client.get("/jobs").json()
        assert data["count"] >= 1
        assert data["jobs"][0]["source"].endswith("demo.mp4")

    def test_script_endpoint_conflicts_before_it_exists(self, client, local_source):
        resp = client.post("/jobs", json={"source": str(local_source), "offline": True, "until": "analyze"})
        job_id = wait_for(client, resp.json()["job_id"])["id"]
        assert client.get(f"/jobs/{job_id}/script").status_code == 409

    def test_resume_reruns_a_failed_job(self, client, tmp_path):
        resp = client.post("/jobs", json={"source": str(tmp_path / "missing.mp4"), "offline": True})
        job = wait_for(client, resp.json()["job_id"])
        assert job["status"] == "failed"
        assert client.post(f"/jobs/{job['id']}/resume").status_code == 200
        assert wait_for(client, job["id"])["status"] == "failed"


class TestFiles:
    def test_video_is_served_once_written(self, client, local_source):
        resp = client.post("/jobs", json={"source": str(local_source), "offline": True, "until": "fetch"})
        job_id = wait_for(client, resp.json()["job_id"])["id"]
        out = Path(client.app.state.ctx.pipeline.settings.workdir) / "jobs" / job_id / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "final.mp4").write_bytes(b"\x00" * 4096)

        got = client.get(f"/jobs/{job_id}/video")
        assert got.status_code == 200
        assert got.headers["content-type"] == "video/mp4"
        assert len(got.content) == 4096

    def test_missing_video_is_404(self, client, local_source):
        resp = client.post("/jobs", json={"source": str(local_source), "offline": True, "until": "fetch"})
        job_id = wait_for(client, resp.json()["job_id"])["id"]
        assert client.get(f"/jobs/{job_id}/video").status_code == 404

    @pytest.mark.parametrize(
        "name",
        ["../../../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "/etc/passwd", "../work/ytpipe.sqlite3"],
    )
    def test_path_traversal_is_refused(self, client, local_source, name):
        resp = client.post("/jobs", json={"source": str(local_source), "offline": True, "until": "fetch"})
        job_id = wait_for(client, resp.json()["job_id"])["id"]
        got = client.get(f"/jobs/{job_id}/file/{name}")
        assert got.status_code in (400, 404)
        assert "root:" not in got.text

    def test_unknown_jobs_are_404(self, client):
        for route in ("", "/events", "/video", "/script"):
            assert client.get(f"/jobs/does-not-exist{route}").status_code == 404


def test_transcript_only_is_accepted_and_persisted(client, local_source):
    """The HTTP body must expose what the CLI exposes, or the two drift apart."""
    r = client.post(
        "/jobs",
        json={
            "source": str(local_source),
            "topic": "cable failures",
            "target_seconds": 10,
            "wpm": 190,
            "offline": True,
            "transcript_only": True,
            "until": "script",
        },
    )
    assert r.status_code == 202, r.text
    job = wait_for(client, r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["transcript_only"] is True
    assert job["stages"]["script"]["status"] == "done"

class TestUpload:
    """`POST /upload` is how a clip you already have enters the pipeline."""

    def test_a_real_clip_is_stored_probed_and_usable_as_a_source(self, client, local_source):
        resp = client.post(
            "/upload",
            files={"file": ("my clip (1).mp4", local_source.read_bytes(), "video/mp4")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert 10.0 < body["duration_s"] < 26.0, "the probe result must come back to the client"
        assert body["aspect"] in ("9:16", "16:9", "1:1")
        assert body["path"].endswith(".mp4") and Path(body["path"]).exists()
        assert "my-clip-1" in Path(body["path"]).name.replace(".mp4", ""), "client names are cleaned"
        assert body["caption_track"] is None

        job = client.post(
            "/jobs",
            json={
                "source": body["path"], "offline": True, "topic": "cable failures",
                "target_seconds": 10, "wpm": 190, "until": "storyboard",
            },
        )
        assert job.status_code == 202
        done = wait_for(client, job.json()["job_id"])
        assert done["status"] == "done", done["error"]
        assert done["stages"]["storyboard"]["status"] == "done"
        assert done["stages"]["fetch"]["status"] == "done", "a local clip still runs fetch (to probe)"

    def test_uploads_are_content_addressed_so_the_same_bytes_are_one_file(self, client, local_source):
        data = local_source.read_bytes()
        first = client.post("/upload", files={"file": ("a.mp4", data, "video/mp4")}).json()["path"]
        second = client.post("/upload", files={"file": ("b.mp4", data, "video/mp4")}).json()["path"]
        assert Path(first).name.split("-", 1)[0] == Path(second).name.split("-", 1)[0]
        assert len(list(Path(first).parent.glob("*.mp4"))) == 1

    def test_captions_come_along_and_reach_the_transcript(self, client, local_source):
        vtt = local_source.with_suffix(".vtt").read_bytes()
        resp = client.post(
            "/upload",
            files={
                "file": ("clip.mp4", local_source.read_bytes(), "video/mp4"),
                "subs": ("whatever-i-named-it.srt", vtt.replace(b"WEBVTT", b""), "text/plain"),
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["caption_track"] and Path(body["caption_track"]).exists()
        job = client.post(
            "/jobs",
            json={
                "source": body["path"], "caption_track": body["caption_track"], "offline": True,
                "topic": "cable failures", "target_seconds": 10, "wpm": 190, "until": "transcribe",
            },
        )
        done = wait_for(client, job.json()["job_id"])
        assert done["status"] == "done", done["error"]
        assert done["stages"]["transcribe"]["status"] == "done"

    def test_a_file_that_is_not_a_video_is_rejected_with_an_actionable_detail(self, client, settings):
        resp = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert resp.status_code == 415
        assert ".mp4" in resp.json()["detail"]

        # .mp4 on the outside, garbage on the inside: the probe is what catches this.
        resp = client.post("/upload", files={"file": ("fake.mp4", b"not a video at all", "video/mp4")})
        assert resp.status_code == 415
        assert "readable video" in resp.json()["detail"] or "no readable video stream" in resp.json()["detail"]
        assert not list(settings.uploads_dir.glob("*fake.mp4")), "rejected uploads are removed"

    def test_oversized_uploads_are_cut_off_at_the_limit(self, client, settings, local_source):
        # The app holds the same Settings object, so this is the documented env knob
        # (YTPIPE_UPLOAD_MAX_MB) exercised at runtime rather than a test-only hook.
        settings.upload_max_mb = 0.05
        size = len(local_source.read_bytes())
        resp = client.post("/upload", files={"file": ("big.mp4", local_source.read_bytes(), "video/mp4")})
        assert resp.status_code == 413, resp.text
        assert "MB limit" in resp.json()["detail"]
        assert size > 50_000, "the fixture has to be bigger than the limit for this to mean anything"
        assert not list(settings.uploads_dir.glob("*.part")), "a rejected upload leaves no partial file"
        assert not list(settings.uploads_dir.glob("*.mp4"))

    def test_a_client_cannot_escape_the_upload_directory_with_a_name(self, client, settings, local_source):
        resp = client.post(
            "/upload",
            files={"file": ("../../../../etc/passwd.mp4", local_source.read_bytes(), "video/mp4")},
        )
        assert resp.status_code == 200
        stored = Path(resp.json()["path"])
        assert stored.parent == settings.uploads_dir
        assert stored.name == Path(stored.name).name and ".." not in stored.name
