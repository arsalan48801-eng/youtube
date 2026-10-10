"""`ytpipe` command line.

The CLI is the real interface; the HTTP API is a thin wrapper over the same
`Pipeline.run`. Everything here resolves a Settings object and hands off.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer

from .config import build_settings
from .models import Aspect, JobRequest, StageName, Status
from .pipeline import Pipeline

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fetch a source video, learn from it, cut a new one, ship it.",
)


def _settings(workdir: Path | None = None, offline: bool = False):
    settings = build_settings()
    if workdir:
        settings.workdir = Path(workdir)
        settings.ensure_dirs()
    if offline:
        settings.offline = True
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return settings


def _pipeline(workdir: Path | None, offline: bool, verbose: bool, scale: float = 1.0) -> Pipeline:
    settings = _settings(workdir, offline)
    settings.render_scale = max(0.1, min(1.0, scale))
    if verbose:
        logging.getLogger("ytpipe").setLevel(logging.DEBUG)
    return Pipeline(settings)


@app.command()
def run(
    source: str = typer.Argument(..., help="YouTube URL, 11-char video id, or a local media file"),
    topic: str | None = typer.Option(None, help="What the new video should be about"),
    title_hint: str | None = typer.Option(None, help="Nudge the writer's title choice"),
    aspect: Aspect = typer.Option(Aspect.VERTICAL, help="Output canvas"),
    target: float = typer.Option(45.0, "--target", "--target-seconds", help="Target length in seconds"),
    wpm: float = typer.Option(152.0, help="Narration pace used for timing"),
    language: str | None = typer.Option(None, help="Preferred caption language, e.g. en, es"),
    until: StageName | None = typer.Option(None, help="Stop after this stage"),
    only: str | None = typer.Option(None, help="Comma-separated stage list to run"),
    music: Path | None = typer.Option(None, help="Music bed, ducked under the voiceover"),
    transition: str = typer.Option("dip", help="cut | dip | xfade"),
    captions: bool = typer.Option(True, help="Burn captions into the picture"),
    scale: float = typer.Option(1.0, "--scale", help="0.5 = half-res draft pass, ~4x faster"),
    subs: Path | None = typer.Option(
        None, "--subs", help="Caption track for a local clip (.vtt/.srt), any file name"
    ),
    transcript_only: bool = typer.Option(
        False, "--transcript-only", help="Read the link: metadata + captions, no media download"
    ),
    offline: bool = typer.Option(False, help="No network: deterministic writers and cards"),
    publish: bool = typer.Option(False, help="Actually upload (needs OAuth + YTPIPE_YOUTUBE_DRY_RUN=0)"),
    privacy: str = typer.Option("private", help="private | unlisted | public"),
    workdir: Path | None = typer.Option(None, help="Where job folders live"),
    job_id: str | None = typer.Option(None, help="Use a stable job id (idempotent reruns)"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Run the pipeline end to end, or up to --until <stage>."""
    from rich.console import Console
    from rich.table import Table

    pipe = _pipeline(workdir, offline, verbose, scale)
    request = JobRequest(
        source=source,
        topic=topic,
        title_hint=title_hint,
        aspect=aspect,
        target_seconds=target,
        wpm=wpm,
        language=language,
        caption_track=subs,
        transcript_only=transcript_only,
        music_path=music,
        transition=transition,
        captions=captions,
        offline=offline,
        publish=publish,
        privacy=privacy,
        until=until,
        only=[StageName(s.strip()) for s in only.split(",")] if only else None,
    )
    console = Console()
    providers = pipe.settings.describe_providers()
    console.rule("ytpipe")
    console.print(
        f"[bold]source[/] {source}\n"
        f"[bold]canvas[/] {aspect.value} {aspect.size[0]}x{aspect.size[1]} "
        f"[bold]target[/] {target:.0f}s @ {wpm:.0f} wpm"
    )
    console.print("[bold]providers[/] " + json.dumps(providers))

    started = time.monotonic()
    job = pipe.run(request, job_id=job_id)
    elapsed = time.monotonic() - started

    table = Table(title=f"job {job.id}  [{job.status.value}]  {elapsed:.1f}s")
    table.add_column("stage", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("s", justify="right")
    table.add_column("tries", justify="right")
    table.add_column("artifact", overflow="fold")
    for name in [s.value for s in StageName.ordered()]:
        run_row = job.stages.get(name)
        if run_row is None:
            continue
        colour = {"done": "green", "failed": "red", "running": "yellow"}.get(run_row.status.value, "dim")
        table.add_row(
            name,
            f"[{colour}]{run_row.status.value}[/]",
            f"{run_row.duration_s:.2f}",
            str(run_row.attempts),
            Path(run_row.artifact).name if run_row.artifact else "-",
        )
    console.print(table)

    render = None
    if (job.workdir / "out" / "final.mp4").exists():
        render = job.workdir / "out" / "final.mp4"
        size_mb = render.stat().st_size / 1e6
        console.print(f"[bold green]video[/] {render} ({size_mb:.2f} MB)")
        console.print(f"[bold dim]subs[/]  {job.workdir / 'out' / 'final.srt'}")
    if job.error:
        console.print(f"[bold red]error[/] {job.error}")
        raise typer.Exit(code=1)
    console.print(f"[dim]dir[/] {job.workdir}")


DEMO_LINES = [
    "Most people buy the wrong monitor and it costs them a year of eyes.",
    "The spec sheet hides one number that actually decides the picture.",
    "We tested four panels at 120 hertz for thirty days straight.",
    "Two locked the refresh rate and two dropped frames every ninety seconds.",
    "The fix costs eleven dollars and takes about one minute to apply.",
    "Measure the thing you care about before you spend anything on it.",
]


@app.command()
def demo(
    out: Path = typer.Option(Path("fixtures/demo"), help="Writes <out>.mp4 and <out>.vtt"),
    seconds: float = typer.Option(18.0, help="Length of the generated source clip"),
    force: bool = typer.Option(False, help="Re-encode the clip even if it exists"),
    offline: bool = typer.Option(True, help="Ignored: this command never touches the network"),
) -> None:
    """Build a local source clip with word-timed captions, for an offline first run.

    The fixture is generated instead of committed: nothing to download, nothing to keep in
    git, and the captions carry per-word timestamps, which is the part that makes the
    karaoke and the timing maths run on real data rather than on cue-level guesses.
    """
    from rich.console import Console

    from .ffmpeg import FFmpeg

    console = Console()
    settings = _settings(None, offline)
    ff = FFmpeg(settings.ffmpeg_binary)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    video, subs = out.with_suffix(".mp4"), out.with_suffix(".vtt")

    if force or not video.exists():
        ff.run(
            [
                "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={float(seconds):.2f}",
                "-f", "lavfi", "-i", f"sine=frequency=300:duration={float(seconds):.2f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
            ],
            timeout=600,
        )
    else:
        console.print(f"[dim]keeping existing {video} (use --force to re-encode)[/dim]")

    per = max(1.0, float(seconds) / len(DEMO_LINES))
    blocks = []
    for i, line in enumerate(DEMO_LINES):
        start = i * per
        words = line.split()
        step = per / max(1, len(words))
        body = " ".join(f"<{_clock(start + j * step)}>{w}" for j, w in enumerate(words))
        blocks.append(f"{_clock(start)} --> {_clock(start + per - 0.05)}\n{body}")
    subs.write_text("WEBVTT\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")

    size = video.stat().st_size / 1e6 if video.exists() else 0.0
    console.print(f"[green]wrote[/green] {video} [dim]({size:.1f} MB, {float(seconds):.0f}s)[/dim]")
    console.print(f"[green]wrote[/green] {subs} [dim]({len(DEMO_LINES)} cues, word-timed)[/dim]")
    console.print(
        f"\nnext: [bold]ytpipe run {video} --offline --target 22 --scale 0.25 "
        f'--topic "your topic"[/bold]'
    )


def _clock(seconds: float) -> str:
    ms = int(round(max(0.0, float(seconds)) * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


@app.command()
def transcript(
    source: str = typer.Argument(..., help="YouTube URL, 11-char video id, or a local media file"),
    language: str | None = typer.Option(None, help="Caption language, e.g. en, es, pt-BR"),
    srt: Path | None = typer.Option(None, help="Also write an .srt file here"),
    raw: bool = typer.Option(False, help="Print one line per caption cue instead of per sentence"),
    json_out: Path | None = typer.Option(None, "--json", help="Also write the timed segments as JSON"),
    workdir: Path | None = typer.Option(None, help="Where job folders live"),
) -> None:
    """Read a video's transcript straight from the link, with no media download.

    Metadata plus the caption track is all this needs: tens of kilobytes, and it never
    touches the streaming URLs -- the part that usually gets bot-checked. Run it before
    committing to a render to answer "is there anything here to work with?".
    """
    import json as _json

    from rich.console import Console

    pipe = _pipeline(workdir, False, False)
    job = pipe.run(
        JobRequest(
            source=source,
            offline=False,
            transcript_only=True,
            language=language,
            only=[StageName.FETCH, StageName.TRANSCRIBE],
        )
    )
    console = Console()
    if job.status is not Status.DONE:
        console.print(f"[red]error[/red]  {job.error or 'transcript failed'}")
        raise typer.Exit(1)

    track = pipe.load_artifact(job, StageName.TRANSCRIBE)
    if track is None or not track.segments:
        console.print("[yellow]no caption track on this video[/yellow] -- nothing to read")
        raise typer.Exit(2)

    last = track.segments[-1].end
    rows: list[tuple[float, str]] = []
    if raw:
        rows = [(seg.start, seg.text) for seg in track.segments]
    else:
        # An auto-caption track is one cue per *word*; printing that reads like a ransom
        # note. Sentences are what a person wants to scan when deciding on a source.
        from .stages.analyze import sentences_of

        rows = [(start, text) for start, _end, text in sentences_of(track)]
    console.print(
        f"[bold]{len(track.segments)}[/bold] cues [dim]· {track.provider} · {track.language or '?'} · "
        f"{len(rows)} lines · ends {int(last // 60):02d}:{last % 60:04.1f} · {job.id}[/dim]"
    )
    for start, text in rows:
        console.print(f"[dim]{int(start // 60):02d}:{start % 60:04.1f}[/dim] {text}")
    if srt:
        from .captions import CaptionCue, to_srt

        cues = [CaptionCue(start=s.start, end=s.end, text=s.text) for s in track.segments]
        Path(srt).write_text(to_srt(cues), encoding="utf-8")
        console.print(f"[green]wrote[/green] {srt}")
    if json_out:
        payload = [s.model_dump(mode="json") for s in track.segments]
        Path(json_out).write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {json_out}")


@app.command("doctor")
def doctor(workdir: Path | None = typer.Option(None)) -> None:
    """What is available right now: ffmpeg, filters, fonts, provider keys."""
    from rich.console import Console

    console = Console()
    settings = _settings(workdir)
    try:
        from .ffmpeg import FFmpeg

        ff = FFmpeg(settings.ffmpeg_binary)
        console.print(f"[green]ffmpeg[/green] {ff.version()}")
        console.print(f"[dim]{ff.binary}[/dim]")
        for label, flag in (
            ("libass captions", ff.has_filter("subtitles")),
            ("zoompan motion", ff.has_filter("zoompan")),
            ("xfade transitions", ff.has_filter("xfade")),
            ("drawtext (unused)", ff.has_filter("drawtext")),
        ):
            console.print(f"  {'[green]+[/green]' if flag else '[red]-[/red]'} {label}")
    except Exception as exc:
        console.print(f"[red]ffmpeg[/red] {exc}")
    for name, value in settings.describe_providers().items():
        console.print(f"[cyan]{name}[/cyan] {value}")
    console.print(f"[cyan]workdir[/cyan] {settings.workdir.resolve()}")
    console.print(
        "[yellow]quota[/yellow] videos.insert = 1600 units of 10000/day "
        "(~6 uploads/day without a quota grant)"
    )


@app.command()
def jobs(limit: int = 20, workdir: Path | None = typer.Option(None)) -> None:
    """Recent jobs with their stage progress."""
    from rich.console import Console
    from rich.table import Table

    from .state import JobStore

    settings = _settings(workdir)
    store = JobStore(settings.db_path())
    table = Table(title=f"jobs in {settings.workdir.resolve()}")
    for col in ("id", "status", "stages done", "source", "updated"):
        table.add_column(col)
    for job in store.list_jobs(limit):
        done = sum(1 for r in job.stages.values() if r.status == Status.DONE)
        table.add_row(
            job.id,
            job.status.value,
            f"{done}/{len(StageName) - (0 if job.request.publish else 1)}",
            (job.request.source or "")[:38],
            job.updated_at.strftime("%m-%d %H:%M:%S"),
        )
    Console().print(table)


@app.command()
def show(job_id: str, workdir: Path | None = typer.Option(None)) -> None:
    """Full stage detail for one job, including per-stage log lines."""
    from rich.console import Console
    from rich.syntax import Syntax

    settings = _settings(workdir)
    from .state import JobStore

    job = JobStore(settings.db_path()).get(job_id)
    if job is None:
        raise typer.BadParameter(f"no job {job_id}")
    console = Console()
    console.print(Syntax(json.dumps(job.snapshot(), indent=2), "json", theme="ansi_light"))


@app.command()
def tail(job_id: str, workdir: Path | None = typer.Option(None), interval: float = 1.0) -> None:
    """Follow a running job's event stream."""
    from .state import JobStore

    settings = _settings(workdir)
    store = JobStore(settings.db_path())
    cursor = 0
    while True:
        rows, cursor = store.events_after(job_id, cursor)
        for row in rows:
            print(f"{row['ts'][11:19]} {row['stage'] or '-':<11} {row['level']:<7} {row['message']}")
        job = store.get(job_id)
        if job and job.status in (Status.DONE, Status.FAILED) and not rows:
            print(f"-- job {job.status.value} --")
            return
        time.sleep(interval)


@app.command()
def api(
    host: str = "0.0.0.0",  # noqa: S104 - the sandbox preview needs all interfaces
    port: int = 8000,
    workdir: Path | None = typer.Option(None),
    reload: bool = False,
) -> None:
    """Serve the HTTP API + the job dashboard."""
    import uvicorn

    _settings(workdir)
    uvicorn.run("ytpipe.api:app", host=host, port=port, reload=reload, log_level="info")


def main() -> None:  # pragma: no cover - entry point shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
