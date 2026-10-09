"""Job persistence: sqlite for state, the filesystem for artifacts.

sqlite because the pipeline is a queue-shaped workload that outlives any single
process: a render can take longer than a request/response cycle, workers should be
resumable, and the API must answer `GET /jobs/{id}` for a job that another process
created. Artifacts (video, wav, json blobs) stay on disk and are referenced by path.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Job, JobRequest, StageName, StageRun, Status, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    workdir     TEXT NOT NULL,
    request     TEXT NOT NULL,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_runs (
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    started_at  TEXT,
    finished_at TEXT,
    duration_s  REAL NOT NULL DEFAULT 0,
    artifact    TEXT,
    error       TEXT,
    log         TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (job_id, stage)
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL,
    ts      TEXT NOT NULL,
    stage   TEXT,
    level   TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id, id);
"""


def new_job_id() -> str:
    return utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


class JobStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- jobs -----------------------------------------------------------------

    def create(
        self, request: JobRequest, *, job_id: str | None = None, workdir: Path | None = None
    ) -> Job:
        job_id = job_id or new_job_id()
        job = Job(id=job_id, request=request, workdir=workdir or Path(f"work/jobs/{job_id}"))
        return self._insert(job)

    def register(self, job: Job) -> Job:
        return self._insert(job)

    def _insert(self, job: Job) -> Job:
        job.workdir.mkdir(parents=True, exist_ok=True)
        (job.workdir / "stages").mkdir(parents=True, exist_ok=True)
        (job.workdir / "out").mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO jobs (id, status, workdir, request, error, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?,?)",
                (
                    job.id,
                    Status(job.status).value,
                    str(job.workdir),
                    job.request.model_dump_json(),
                    job.error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
            for name, run in job.stages.items():
                self._write_stage(conn, job.id, name, run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            runs = conn.execute(
                "SELECT * FROM stage_runs WHERE job_id = ? ORDER BY stage", (job_id,)
            ).fetchall()
        job = Job(
            id=row["id"],
            request=JobRequest.model_validate(json.loads(row["request"])),
            status=Status(row["status"]),
            workdir=Path(row["workdir"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            error=row["error"],
        )
        for run in runs:
            job.stages[run["stage"]] = StageRun(
                stage=StageName(run["stage"]),
                status=Status(run["status"]),
                started_at=datetime.fromisoformat(run["started_at"]) if run["started_at"] else None,
                finished_at=(
                    datetime.fromisoformat(run["finished_at"]) if run["finished_at"] else None
                ),
                duration_s=run["duration_s"],
                attempts=run["attempts"],
                artifact=run["artifact"],
                error=run["error"],
                log=json.loads(run["log"]),
            )
        return job

    def update(self, job_id: str, *, status: Status | None = None, error: str | None = None) -> None:
        sets, vals = ["updated_at = ?"], [utcnow().isoformat()]
        if status is not None:
            sets.insert(0, "status = ?")
            vals.insert(0, Status(status).value)
        if error is not None:
            sets.append("error = ?")
            vals.append(error)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", (*vals, job_id))

    def list_jobs(self, limit: int = 50) -> list[Job]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [job for r in rows if (job := self.get(r["id"])) is not None]

    # -- stages ---------------------------------------------------------------

    def _write_stage(
        self, conn: sqlite3.Connection, job_id: str, stage: str, run: StageRun
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO stage_runs (job_id, stage, status, attempts, started_at, "
            "finished_at, duration_s, artifact, error, log) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                str(stage),
                Status(run.status).value,
                run.attempts,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                run.duration_s,
                run.artifact,
                run.error,
                json.dumps(run.log[-40:]),
            ),
        )

    def save_stage(self, job_id: str, run: StageRun) -> None:
        stage = str(getattr(run.stage, "value", run.stage))
        with self._lock, self._connect() as conn:
            self._write_stage(conn, job_id, stage, run)
            conn.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (utcnow().isoformat(), job_id))

    # -- events ---------------------------------------------------------------

    def event(self, job_id: str, message: str, *, stage: str | None = None, level: str = "info") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO events (job_id, ts, stage, level, message) VALUES (?,?,?,?,?)",
                (job_id, utcnow().isoformat(), stage, level, message),
            )

    def events_after(
        self, job_id: str, cursor: int = 0, limit: int = 200
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, ts, stage, level, message FROM events WHERE job_id = ? AND id > ? "
                "ORDER BY id LIMIT ?",
                (job_id, cursor, limit),
            ).fetchall()
        return [dict(r) for r in rows], (rows[-1]["id"] if rows else cursor)
