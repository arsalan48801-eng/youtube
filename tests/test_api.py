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
