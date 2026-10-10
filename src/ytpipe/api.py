"""HTTP API + job dashboard.

Built as a factory (``create_app(settings)``) rather than module globals, so a test
or a second deployment gets its own workdir, thread pool and job store instead of
inheriting whatever the first import created.

Long renders do not fit a request/response cycle, so ``POST /jobs`` returns a job id
immediately and the client polls ``/jobs/{id}`` (or streams ``/jobs/{id}/events``).
Work runs on a small thread pool -- one render is already ffmpeg-heavy, so a bigger
pool trades latency for context switching.

The dashboard at ``/`` is deliberately dependency-free HTML: it is the surface you
want when a run is half-finished on a remote box.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings, build_settings
from .models import Aspect, Job, JobRequest, StageName, Status
from .pipeline import Pipeline
from .state import JobStore

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ytpipe · jobs</title>
<style>
:root{--bg:#0b0e14;--fg:#e6edf3;--dim:#8b98a5;--card:#131823;--line:#1f2734;--ok:#3fb950;--err:#f85149;--run:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{display:flex;gap:18px;align-items:center;padding:16px 22px;border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(11,14,20,.9);backdrop-filter:blur(8px);z-index:2}
h1{font-size:16px;margin:0;letter-spacing:.3px}h1 span{color:var(--dim);font-weight:400}.meta{color:var(--dim);font-size:12px;font-family:ui-monospace,monospace}
main{padding:22px;display:grid;grid-template-columns:340px 1fr;gap:22px;align-items:start}
form{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;display:grid;gap:10px;position:sticky;top:70px}
label{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);display:grid;gap:5px}
input,select{background:#0d1117;border:1px solid var(--line);color:var(--fg);border-radius:8px;padding:9px 10px;font:inherit;width:100%}
button{background:#238636;border:0;color:#fff;border-radius:8px;padding:11px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.6;cursor:default}.row{display:flex;gap:10px}.row>*{flex:1}
.check{text-transform:none;letter-spacing:0;display:flex;gap:8px;align-items:center;color:var(--fg);font-size:13px}
.check input{width:auto}
.jobs{display:grid;gap:12px}
.job{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;display:grid;gap:12px;grid-template-columns:1fr 200px}
.job.sel{outline:1px solid #388bfd}.job h3{margin:0;font-size:15px;word-break:break-word}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{border:1px solid var(--line);border-radius:999px;padding:3px 9px;font-size:11px;color:var(--dim);font-family:ui-monospace,monospace;cursor:default}
.chip.done{color:var(--ok);border-color:#1d4228}.chip.failed{color:var(--err);border-color:#4d1f20}.chip.running{color:var(--run);border-color:#4d3b12}
video{width:200px;border-radius:10px;border:1px solid var(--line);background:#000;grid-row:span 4}
.script{grid-column:1/-1;border-top:1px solid var(--line);padding-top:10px;display:grid;gap:4px}
.line{display:grid;grid-template-columns:64px 1fr 120px;gap:10px;font-size:12.5px;padding:3px 0;border-bottom:1px dashed #161d29}
.line b{color:var(--dim);font-weight:500;font-family:ui-monospace,monospace}.line i{color:#79c0ff;font-style:normal;text-align:right}
.err{color:var(--err);font-size:12px;white-space:pre-wrap;margin-top:6px}
.empty{color:var(--dim);padding:40px;text-align:center;border:1px dashed var(--line);border-radius:12px}
.log{grid-column:1/-1;background:#0d1117;border-radius:8px;padding:10px;font:11.5px/1.6 ui-monospace,monospace;color:var(--dim);max-height:180px;overflow:auto;white-space:pre-wrap}
@media(max-width:900px){main{grid-template-columns:1fr}form{position:static}}
</style></head><body>
<header><h1>ytpipe <span>· fetch → script → cut → publish</span></h1><div id="health" class="meta"></div></header>
<main>
<form id="f">
  <label>source<input name="source" placeholder="https://youtu.be/…  or /path/to/clip.mp4" required></label>
  <label>topic<input name="topic" placeholder="what the new video argues"></label>
  <div class="row">
    <label>aspect<select name="aspect"><option value="9:16">9:16</option><option value="16:9">16:9</option><option value="1:1">1:1</option></select></label>
    <label>seconds<input name="target_seconds" type="number" value="45" min="8" max="600"></label>
  </div>
  <div class="row">
    <label>wpm<input name="wpm" type="number" value="152" min="80" max="320"></label>
    <label>transition<select name="transition"><option>dip</option><option>cut</option><option>xfade</option></select></label>
  </div>
  <label>stop after<select name="until"><option value="">end (render)</option><option>transcribe</option><option>analyze</option><option>script</option><option>storyboard</option><option>generate</option></select></label>
  <label class="check"><input type="checkbox" name="offline"> offline (no fetch, no providers)</label>
  <label class="check"><input type="checkbox" name="draft"> draft render (half resolution)</label>
  <label class="check"><input type="checkbox" name="transcript_only"> read the link only (no media download)</label>
  <button type="submit">Queue job</button>
  <div id="msg" class="meta"></div>
</form>
<section class="jobs" id="jobs"><div class="empty">loading…</div></section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=t=>String(t??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let sel=null, opened=new Set();
async function j(u,o){const r=await fetch(u,o);if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||('HTTP '+r.status));return r.json()}
$('#f').onsubmit=async e=>{e.preventDefault();
  const fd=new FormData(e.target); const d={};
  for(const [k,v] of fd.entries()){ if(k==='offline'||k==='draft'||k==='transcript_only'){continue} if(v!=='') d[k]=v }
  d.offline=fd.get('offline')!==null;
  // an unchecked box is absent from FormData and a checked one is the string "on", so
  // booleans are read from the form by presence, never by value.
  d.transcript_only=fd.get('transcript_only')!==null;
  if(fd.get('draft')) d.target_seconds=Math.min(+d.target_seconds||45,12);
  d.target_seconds=+d.target_seconds; d.wpm=+d.wpm;
  $('#msg').textContent='queuing…'; const btn=$('#f').querySelector('button'); btn.disabled=true;
  try{const r=await j('/jobs',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(d)});
    sel=r.job_id; $('#msg').textContent='queued '+r.job_id; refresh();}
  catch(err){$('#msg').textContent='error: '+err.message}
  finally{btn.disabled=false}};
function chip(n,s){return `<span class="chip ${s.status}${s.error?' failed':''}" title="${esc(s.error||'')}">${n}${s.seconds?' '+s.seconds+'s':''}</span>`}
async function refresh(){
  let l;
  try{ l=await j('/jobs') }catch(err){ $('#health').textContent='api unreachable'; return }
  try{
    const h=await j('/healthz');
    $('#health').textContent='ffmpeg '+String(h.ffmpeg||'').split(' ')[1]+' · '+(h.offline?'offline':'online')+' · llm: '+h.providers.llm;
  }catch(e){}
  const box=$('#jobs');
  if(!l.jobs.length){box.innerHTML='<div class="empty">no jobs yet — queue one on the left</div>';return}
  box.innerHTML=l.jobs.map(x=>`<article class="job${sel===x.id?' sel':''}" data-id="${x.id}">
    <div><h3>${esc(x.topic||x.source)}</h3>
    <div class="meta">${x.id} · ${x.status}${x.running?' · running':''} · ${x.aspect} · ${x.target_seconds}s target</div>
    <div class="chips">${Object.entries(x.stages).map(([n,s])=>chip(n,s)).join('')}</div>
    ${x.error?`<div class="err">${esc(x.error)}</div>`:''}</div>
    ${x.has_video?`<video src="/jobs/${x.id}/video" controls preload="metadata" onclick="event.stopPropagation()"></video>`:'<div></div>'}
    ${sel===x.id?`<div class="script" data-script="${x.id}">loading…</div><div class="log" data-log="${x.id}"></div>`:''}
    ${x.outputs&&x.outputs.length?`<div class="meta" style="grid-column:1/-1">${x.outputs.map(o=>`<a href="/jobs/${x.id}/file/${o}" style="color:#58a6ff;margin-right:10px">${esc(o)}</a>`).join('')}</div>`:''}
  </article>`).join('');
  box.querySelectorAll('.job').forEach(el=>el.onclick=()=>{sel=sel===el.dataset.id?null:el.dataset.id;refresh()});
  for(const el of box.querySelectorAll('[data-script]')){
    try{const s=await j('/jobs/'+el.dataset.script+'/script');
      el.innerHTML=`<div class="meta">${esc(s.title)} · ${esc(s.provider)} · ${s.word_count} words</div>`+
        s.lines.map(l=>`<div class="line"><b>${(l.start||0).toFixed(1)}–${(l.end||0).toFixed(1)}</b><span>${esc(l.voiceover)}</span><i>${esc(l.on_screen)}</i></div>`).join('');
    }catch(e){el.textContent='no script for this job yet'}
  }
  for(const el of box.querySelectorAll('[data-log]')){
    try{const ev=await j('/jobs/'+el.dataset.log+'/events');
      el.innerHTML=ev.events.slice(-40).map(e=>`${e.ts.slice(11,19)} ${(e.stage||'-').padEnd(10)} ${esc(e.message)}`).join('\\n');
    }catch(e){}
  }
}
refresh();setInterval(refresh,1800);
</script></body></html>
"""


#: Job ids double as directory names, so they are restricted before any path is built.
#: This allows `--job-id my-brand-video` while still making `..` and separators impossible.
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AppContext:
    """Per-app state: pipeline, worker pool, and which jobs are in flight."""

    def __init__(self, settings: Settings | None = None, max_workers: int = 2) -> None:
        self.pipeline = Pipeline(settings or build_settings())
        self.max_workers = max_workers
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ytpipe")
        self.running: set[str] = set()

    @property
    def store(self) -> JobStore:
        return self.pipeline.store

    def execute(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:  # pragma: no cover - deleted between submit and start
            return
        self.running.add(job_id)
        try:
            self.pipeline.execute(job)
        finally:
            self.running.discard(job_id)


class JobCreate(BaseModel):
    source: str = Field(..., description="YouTube URL, 11-char id, or a path readable by the server")
    topic: str | None = None
    title_hint: str | None = None
    aspect: Aspect = Aspect.VERTICAL
    target_seconds: float = Field(45.0, ge=8.0, le=600.0)
    wpm: float = Field(152.0, ge=80.0, le=320.0)
    language: str | None = None
    offline: bool = False
    #: Metadata + caption track only: no media download. Renders become card-driven.
    transcript_only: bool = False
    publish: bool = False
    privacy: str = "private"
    captions: bool = True
    transition: str = "dip"
    music_path: Path | None = None
    until: StageName | None = None
    only: list[StageName] | None = None
    resume: bool = False

    def to_request(self) -> JobRequest:
        return JobRequest(**self.model_dump())


def _summary(ctx: AppContext, job: Job) -> dict[str, Any]:
    stages = {
        name: {
            "status": getattr(run.status, "value", run.status),
            "seconds": round(run.duration_s, 2),
            "attempts": run.attempts,
            "error": run.error,
            "artifact": run.artifact,
            "log": run.log[-6:],
        }
        for name, run in job.stages.items()
    }
    return {
        "id": job.id,
        "status": getattr(job.status, "value", job.status),
        "error": job.error,
        "source": job.request.source,
        "topic": job.request.topic,
        "aspect": getattr(job.request.aspect, "value", job.request.aspect),
        "target_seconds": job.request.target_seconds,
        "offline": job.request.offline,
        "publish": job.request.publish,
        "transcript_only": job.request.transcript_only,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "running": job.id in ctx.running,
        "stages": stages,
        "has_video": (job.workdir / "out" / "final.mp4").exists(),
        "outputs": _outputs(job),
        "workdir": str(job.workdir),
    }


def _outputs(job: Job) -> list[str]:
    out = job.workdir / "out"
    if not out.exists():
        return []
    return sorted(p.name for p in out.iterdir() if p.is_file())


_MIME = {
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".srt": "application/x-subrip",
    ".ass": "text/plain",
    ".json": "application/json",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def _artifacts_preview(job: Job) -> dict[str, Any]:
    """Small views onto stage output, so the UI renders without touching ffmpeg."""
    base = job.workdir / "stages"
    out: dict[str, Any] = {}
    for stage, filename, key in (
        ("fetch", "source.json", "source"),
        ("analyze", "brief.json", "brief"),
        ("edit", "render.json", "render"),
        ("publish", "publish.json", "publish"),
    ):
        path = base / stage / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - torn write
            continue
        out[key] = data if key in ("publish",) else _trim(key, data)
    return out


def _trim(key: str, data: dict[str, Any]) -> dict[str, Any]:
    if key == "brief":
        return {
            k: data.get(k)
            for k in ("title", "tone", "hook_style", "wpm", "words", "readability", "retention_notes")
        }
    if key == "source":
        return {k: data.get(k) for k in ("title", "channel", "duration", "view_count", "video_id")}
    if key == "render":
        return {
            k: data.get(k)
            for k in ("duration", "width", "height", "fps", "size_bytes", "shot_count", "warnings")
        }
    return data


def create_app(settings: Settings | None = None, *, max_workers: int = 2) -> FastAPI:
    """Build the ASGI app. Pass `settings` to isolate the workdir (tests, staging)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.ctx.pool.shutdown(wait=False)

    app = FastAPI(
        title="ytpipe",
        version="0.1.0",
        description=(
            "Faceless video pipeline: fetch a source, learn from its transcript, "
            "rewrite a script, generate assets, edit with ffmpeg, publish to YouTube."
        ),
        lifespan=lifespan,
    )
    app.state.ctx = AppContext(settings, max_workers=max_workers)
    ctx: AppContext = app.state.ctx

    def require(job_id: str) -> Job:
        job = ctx.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return job

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        settings_ = ctx.pipeline.settings
        try:
            from .ffmpeg import FFmpeg

            ffmpeg = FFmpeg(settings_.ffmpeg_binary).version()
        except Exception as exc:
            ffmpeg = f"unavailable: {exc}"
        return {
            "ok": bool(ffmpeg and not ffmpeg.startswith("unavailable")),
            "ffmpeg": ffmpeg,
            "offline": settings_.offline,
            "render_scale": settings_.render_scale,
            "providers": settings_.describe_providers(),
            "workers": ctx.max_workers,
            "running": sorted(ctx.running),
            "workdir": str(settings_.workdir.resolve()),
        }

    @app.post("/jobs", status_code=202)
    def create_job(payload: JobCreate) -> dict[str, Any]:
        """Queue a run and return immediately with the id to poll."""
        try:
            job = ctx.pipeline.create(payload.to_request())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ctx.pool.submit(ctx.execute, job.id)
        return {
            "job_id": job.id,
            "status": "queued",
            "poll": f"/jobs/{job.id}",
            "events": f"/jobs/{job.id}/events",
            "video": f"/jobs/{job.id}/video",
        }

    @app.get("/jobs")
    def list_jobs(limit: int = 30) -> dict[str, Any]:
        jobs = ctx.store.list_jobs(limit)
        return {"count": len(jobs), "jobs": [_summary(ctx, j) for j in jobs]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = require(job_id)
        return {**_summary(ctx, job), "artifacts": _artifacts_preview(job)}

    @app.get("/jobs/{job_id}/events")
    def get_events(job_id: str, cursor: int = Query(0, ge=0), limit: int = Query(200, le=1000)) -> dict[str, Any]:
        require(job_id)
        rows, next_cursor = ctx.store.events_after(job_id, cursor, limit)
        return {"events": rows, "cursor": next_cursor, "running": job_id in ctx.running}

    @app.get("/jobs/{job_id}/script")
    def get_script(job_id: str) -> dict[str, Any]:
        path = require(job_id).workdir / "stages" / "script" / "script.json"
        if not path.exists():
            raise HTTPException(status_code=409, detail="script not written yet")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> dict[str, Any]:
        """Re-queue a job, skipping stages that already produced artifacts."""
        job = require(job_id)
        if job_id in ctx.running:
            raise HTTPException(status_code=409, detail="job is already running")
        # Only failed stages are reset; completed ones keep their artifacts, so a
        # resume after a mid-render crash reuses the fetch and the written script.
        for run in job.stages.values():
            if run.status == Status.FAILED:
                run.status = Status.PENDING
                run.error = None
                ctx.store.save_stage(job_id, run)
        job.request.resume = True
        ctx.store.save_stage(job_id, next(iter(job.stages.values()), None) or _noop_stage())
        ctx.pool.submit(ctx.execute, job_id)
        return {"job_id": job_id, "status": "queued"}

    def job_dir(job_id: str) -> Path:
        """out/ for this job. Public helpers must not trust the id from the URL."""
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return (ctx.pipeline.settings.job_dir(job_id) / "out").resolve()

    def resolve_out(job_id: str, name: str) -> Path:
        """Join a client-supplied filename inside out/, rejecting anything that escapes.

        ``Path.resolve()`` is done on the joined path (not the parts) so `..` cannot be
        smuggled through, and the prefix test uses a trailing separator so a sibling
        directory named ``out-evil`` does not pass.
        """
        root = job_dir(job_id)
        if not name or "/" in name or "\\" in name or Path(name).is_absolute():
            raise HTTPException(status_code=400, detail="file names must be bare")
        target = (root / name).resolve()
        if target.parent != root or not target.is_file():
            raise HTTPException(status_code=404, detail=f"no file {name} for job {job_id}")
        return target

    def serve(path: Path) -> FileResponse:
        return FileResponse(path, media_type=_MIME.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/jobs/{job_id}/video")
    def get_video(job_id: str) -> FileResponse:
        require(job_id)
        return serve(resolve_out(job_id, "final.mp4"))

    @app.get("/jobs/{job_id}/subtitles")
    def get_subtitles(job_id: str) -> FileResponse:
        require(job_id)
        return serve(resolve_out(job_id, "final.srt"))

    @app.get("/jobs/{job_id}/thumbnail")
    def get_thumbnail(job_id: str) -> FileResponse:
        require(job_id)
        return serve(resolve_out(job_id, "thumb.jpg"))

    @app.get("/jobs/{job_id}/file/{name}")
    def get_file(job_id: str, name: str) -> FileResponse:
        require(job_id)
        return serve(resolve_out(job_id, name))

    return app


def _noop_stage():  # pragma: no cover - keeps resume() from needing a stage to persist
    from .models import StageRun

    return StageRun(stage=StageName.FETCH)


app = create_app()
