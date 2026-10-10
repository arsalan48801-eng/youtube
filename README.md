# ytpipe

A full pipeline for making short-form video from a video you already have.

```
fetch → transcribe → analyze → write → generate → edit → publish
```

Give it a YouTube URL — or any local clip — and get back a vertical MP4 with narration,
timed captions, B-roll, a thumbnail, and an upload manifest.

Every networked step has a deterministic offline fallback, so the pipeline runs with no
API keys and no internet: it will mine a real transcript, write a real script, and render
a real video with ffmpeg. `--offline` forces that mode. Add an LLM key and the writer
starts producing better lines — nothing else changes.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[api,dev]"

ytpipe doctor                     # which providers, fonts, filters and keys are live
ytpipe demo                         # local 18s clip + word-timed captions, no download
ytpipe run fixtures/demo.mp4 --offline --target 22 --scale 0.25 \
    --topic "how to buy a monitor"
open work/jobs/<id>/out/final.mp4                                  # 22.6s MP4, ~0.4 MB
```

---

## What each stage actually does

| Stage | Output artifact | What it does |
|---|---|---|
| `fetch` | `source.json` | Metadata + best MP4 + caption track + thumbnail via yt-dlp. For a local file it skips the download and probes it with ffmpeg instead, then picks up a sibling `.vtt`/`.srt`. |
| `transcribe` | `transcript.json` | Parses VTT/SRT into one segment per **word** when the source has word timestamps (auto captions do) and per cue when it does not. Falls back to local `faster-whisper` if you installed `.[whisper]`. YouTube's rolling captions repeat the tail of the previous cue; that delta is stripped. |
| `analyze` | `brief.json` | Style brief from the transcript, no model needed: measured words-per-minute, how the hook opens, sentence length, Flesch–Kincaid grade, repeated phrases, and which windows of the source actually contain speech. |
| `script` | `script.json` | Voiceover, on-screen text, title, description, tags, hashtags. LLM if configured, deterministic writer otherwise, which lifts the source's own sentence shapes and the facts it states. |
| `storyboard` | `storyboard.json` | Turns lines into timed shots, picks a visual per shot (source B-roll, procedural card, generated still), and rebuilds caption cues from **measured** audio lengths. |
| `generate` | `assets.json` | Narration per shot (TTS if configured, else a timed silence) plus optional B-roll clips, in a thread pool. |
| `edit` | `render.json` | One ffmpeg pass assembles a single graph: per-shot retime, zoom, crop/pad to canvas, caption burn-in, optional music with sidechain ducking, then encode. |
| `publish` | `publish.json` | YouTube Data API v3 resumable upload. Dry-run by default: it writes the exact request body and quota cost instead of posting. |

Per-job layout (`work/jobs/<id>/`): `job.json`, `stages/<stage>/*.json`, `cache/`, `out/`.

The CLI hides the stages that cannot run — no `transcribe` when there is no audio track
and no local ASR, no `fetch` for a local file, no `publish` unless you asked for it.

`ytpipe run --help` lists everything. Two flags do most of the day-to-day work:
`--only edit,publish` re-runs a subset from cached artifacts, and `--job-id <id>` pins the
folder so a rerun reuses them. That is the re-render loop — change the caption style or the
transition, and re-cut the same job in seconds instead of re-fetching and re-writing it.

## Two decisions worth explaining

### 1. The model never decides timing

A generated script is text; a rendered video is a clock. The gap between them is where
these pipelines usually fall apart — a model returns `"start": 0.0, "end": 3.5` for a
line that takes 5.2 seconds to read aloud, and everything after it slides.

So the script model is asked for words and nothing else. Timing is computed locally and
then *corrected by measurement*:

```python
from ytpipe import timing

lines = timing.assign_timings(script.lines, wpm=158, target=44.0)
```

`assign_timings` budgets words from the target duration (pace comes from the analysed
source), then lays shots end-to-start with no gaps. A lead-in and between-line pauses are
*padding inside* shots, never gaps between them, because `concat` deletes gaps and would
silently desync every later caption from the picture it belongs to.

`retime_from_audio` then replaces those estimates with the real duration of each narration
file. If TTS is unavailable, the estimate stands. Either way the timeline is derived from
audio length, so the picture can't drift:

```text
# storyboard.json, as saved after generate measured the narration (22s target, 5 beats)
0.000 → 4.912  (speech 0.35→4.69)  "Everything said about monitor buying mistakes skips…
4.912 → 9.869  (speech 4.91→9.65)  "Here is the part of monitor buying mistakes that…
sum(shot lengths) = 20.003   rendered file = 20.27 s   (plan + tail pad)
```

Because `generate` owns the final timeline, it writes the retimed board *back* into
`stages/storyboard/storyboard.json`. Otherwise a later `--only edit` would read the
pre-measurement board and produce a subtly different video than the run that made it.
`stages/generate/storyboard.retimed.json` keeps the same copy for auditing.

Over-length is fixed in the **writer**, not the edit. `_enforce_budget` runs two passes:
words first (drop the longest body line until the script fits the read-time budget), then
beats (every line costs at least `MIN_SHOT` seconds of picture, so a 6s target cannot carry
a five-beat arc and drops to hook + CTA). Trimming seconds instead would cut a sentence off
mid-syllable; trimming words keeps the narration intact, and every dropped line is logged
with the reason so you can see what the budget ate.

### 2. One filter graph, not a concat of clips

The first version rendered every shot to its own clip and stitched them with a concat
demuxer. It was slow and fragile, and on the PyPI ffmpeg build used here it segfaulted.

Now the whole video is built as a single filter graph in one encode (`assembly.py`):

```
[0:v] trim → setpts → fps → scale → crop/pad → setsar → zoompan ─┐
[1:v] ...                                                        ├─ concat=n=N ─→ [vout]
[0:a] atrim → adelay → apad → afade ──────────────────────────── ┘        └→ [aout]
```

Consequences that mattered:

* Shots with no audio still need a stream, so a *shared* `anullsrc` is `asplit` across
  them. An ffmpeg pad can only be consumed once — feeding the same `[acat]` into both
  the sidechain key and the mixer fails with "matches no streams", so the voiceover is
  split exactly once.
* `-t` owns the duration; `-shortest` would truncate the video against the padded audio.
* Music is `-stream_loop`ed to length, `asplit` into a sidechain key and a mix leg,
  ducked with `sidechaincompress`. Loudness goes through `loudnorm` at −14 LUFS, which is
  roughly what YouTube and TikTok both normalise toward.
* Transitions are `cut`, `dip` (dip-to-black, default, 0.18 s) or `xfade` with a
  cumulative offset.

`--scale 0.25` runs the identical graph at quarter canvas for a draft pass (~16× fewer
pixels), which is how a 30-second edit is checked in about ten seconds instead of two
minutes.

### Captions

One ASS document is burned in with libass, containing a bottom karaoke layer (`\kf`
per-word highlight) and a top emphasis layer for the on-screen text of each shot.

* Cues are regrouped to 2 lines / 42 characters and their text is planned over each shot's
  **speech window**, not its full picture window — otherwise the first words of a line
  appear while the screen is still showing the previous line.
* `WrapStyle: 0` is deliberate: smart wrapping (2) clips wide overlays on a 9:16 canvas.
* The emphasis font size shrinks automatically so a four-word overlay still fits.
* `final.srt` is written alongside as the *clean* track for YouTube, because an uploaded
  SRT should not contain style tags.

`drawtext` is not in the PyPI ffmpeg build, and it does not need to be: every piece of
text goes through libass.

### Publishing

`publish` builds the `videos.insert` body from the script (title clamped to 100 chars at a
word boundary, description to 5000, tags to 500 total) and then stops:

```bash
ytpipe run "$SRC" --publish --privacy public
```

Upload happens only when all three of `--publish`, a refresh token, and
`YTPIPE_YOUTUBE_DRY_RUN=0` are present. Otherwise you get `out/publish.json` — the exact
request body plus the quota it would cost (1600 units of 10000/day, ~6 uploads), which is
the number that gets new channels suspended. It also warns when the duration is ≥180 s
(that clip will not post as a Short) and sets `containsSyntheticMedia` when the voice is
generated, per YouTube's disclosure rule.

To go live: create a Desktop-app OAuth client, run the consent flow once, put the refresh
token in `.env` (see `.env.example`), and flip the dry-run flag.

## HTTP API

```bash
ytpipe api --host 0.0.0.0 --port 8000      # dashboard at /, docs at /docs
```

```
POST /jobs                      start a pipeline (accepts the same fields as JobRequest)
GET  /jobs?limit=20             recent jobs
GET  /jobs/{id}                 status + per-stage timings
GET  /jobs/{id}/events?cursor=  incremental progress, for polling or SSE
GET  /jobs/{id}/resume          re-run only what failed, keep the rest
GET  /jobs/{id}/script          the generated script
GET  /jobs/{id}/video           final.mp4 (byte-ranged, so seeking works)
GET  /jobs/{id}/subtitles       final.srt
GET  /jobs/{id}/thumbnail       thumb.jpg
GET  /jobs/{id}/file/{name}     any other file under out/
GET  /healthz                   providers, filters, font, workdir
```

The dashboard is one HTML string in `api.py`: stage chips light up as a job advances, the
event log appends, and progress polls every 1.8 s. That is the whole front end — no build
step, no node_modules, and `GET /` works from any browser on the box.

`create_app(settings)` is a factory, so tests and embedded use build an app against a temp
workdir instead of importing module-level globals. Job ids are validated against a strict
pattern and every file download resolves through a helper that rejects separators and
parent escapes, because `/jobs/{id}/file/{name}` is otherwise a directory-traversal
endpoint with nice error messages.

## Configuration

No YAML. A single pydantic `Settings` reads `YTPIPE_*` environment variables, where the
field is the last segment (`YTPIPE_TARGET_SECONDS=45` → `target_seconds`;
`YTPIPE_LLM_MODEL=...` → `settings.llm.model`). See `.env.example` for the full annotated
list. CLI flags override environment; environment overrides defaults.

Provider resolution, in one place (`providers/__init__.py`):

| Capability | Configured | Offline fallback |
|---|---|---|
| Script writing | any OpenAI-compatible `/chat/completions` (OpenAI, Groq, Together, OpenRouter, llama.cpp, Ollama) | deterministic writer over the source brief |
| Narration | OpenAI TTS, or `edge-tts` if installed | timed silence; captions carry the video |
| Stills | `images.generate` | procedural cards (background + text via libass) |
| Music | `--music file.mp3` | none |
| Publish | YouTube OAuth refresh token | dry-run manifest |

Set `YTPIPE_OFFLINE=1` to ignore every key.

## Library use

```python
import ytpipe
from ytpipe.models import JobRequest

settings = ytpipe.build_settings()
settings.render_scale = 0.5                    # half-res draft, identical timing
pipeline = ytpipe.Pipeline(settings)

job = pipeline.run(JobRequest(
    source="https://youtu.be/dQw4w9WgXcQ",
    topic="why this argument keeps winning",
    target_seconds=48,
    publish=False,                             # render only, never upload
    until=None,                                # or StageName.SCRIPT to stop at the script
))
print(job.id, job.status, job.output("final.mp4"))
print(pipeline.load_artifact(job, ytpipe.StageName.SCRIPT).title)
```

`Pipeline(settings, stages=[...])` runs a chosen subset; a subset never wipes artifacts
cached by earlier runs, and `Pipeline.load_artifact(job, stage)` re-reads any finished
stage's model from disk — that is how `--only edit` re-renders without re-fetching.

Stages are plain objects with a `name` and `run(ctx)`, so a replacement writer is 20 lines:

```python
from ytpipe import StageName
from ytpipe.models import Script, ScriptLine


class TwoLineWriter:
    name = StageName.SCRIPT

    def run(self, ctx):
        ctx.emit("writing two lines")
        return Script(
            lines=[
                ScriptLine(index=0, role="hook", voiceover="Two lines is enough.",
                           on_screen="Two Lines", start=0.0, end=2.4),
                ScriptLine(index=1, role="cta", voiceover="Follow for more.",
                           on_screen="Follow", start=2.4, end=4.2),
            ],
        )
```

`ctx` is everything a stage gets: `ctx.job`, `ctx.settings`, `ctx.artifacts` (all upstream
results), `ctx.ffmpeg`, `ctx.out` / `ctx.path()` / `ctx.save()` / `ctx.load()`, and
`ctx.emit()` for progress. Stages never read `os.environ` and never import a provider SDK
at module scope — that is what keeps `--offline` a single switch instead of a
try/except around every call site.

## Tests

132 tests, no network, no API keys:

```bash
pytest -q                    # unit tests, including real ffmpeg renders
```

* `test_media.py` — ffmpeg probing, tone/silence encoding, ASS escaping and cue packing,
  and `ytpipe demo` producing a clip the rest of the pipeline can actually use.
* `test_fetch.py` — how a `--source` string is classified (URL, bare id, local file, junk)
  and what each failure tells you. No network: yt-dlp is never invoked.
* `test_writing.py` — the offline writer, the regexes that clean up ASR output, the
  `Brief` heuristics (pace, hook style, readability, B-roll windows), storyboard chunk
  selection, and `on_screen_for`.
* `test_assembly.py` — the filter graph as text: input ordering, shared silent audio, the
  double-consumption bug above, transition offsets, encode args.
* `test_render.py` — encodes real clips (a source clip, TTS-length retiming, music ducking
  with and without a bed, caption burn-in) and asserts on `ffmpeg -i` output.
* `test_pipeline.py` — retry/backoff on transient failures, non-retryable aborts, halt on
  failure, resume skipping completed stages, artifact round-trip, event cursor, log
  trimming, dry-run publish manifest.
* `test_api.py` — every endpoint through `TestClient`, including traversal attempts and a
  404 when a route leaks a `Path`.

`test_llm_path.py` injects fake `llm`/`tts` clients through `ctx.with_providers(...)`,
which is the same seam the real providers plug into: it proves a model's JSON is used, that
a model's *timings* are ignored, that garbage output falls back to the offline writer
instead of failing the job, and that a TTS failure degrades to silence rather than to no
video.

`conftest.py` builds a 640×360, 18 s `testsrc2` clip with a sine tone and a VTT with word
timestamps, and pins `render_scale=0.25`, `max_retries=1` and `offline=True` so a suite
run never reaches the network and never waits on real retries.

## Limits

Stated plainly, because they are the parts that decide whether this is a toy or a tool:

* The offline writer is **template-driven**. Its structure comes from the source video and
  its sentences come from facts mined out of the transcript, but it will not surprise you.
  That is the point of the fallback: it proves the plumbing, not the taste. Configure an
  LLM for real writing.
* No `drawtext` and no Headless Chrome means **no motion-graphics overlays** — text,
  boxes and bars are all libass. Real template graphics (a `.ftvec`/HTML renderer) would
  replace `VisualSpec(kind=CARD)` and nothing else.
* Music is whatever file you hand it; there is no generation and no stem-aware mixing.
  `sidechaincompress` is a single-band duck, not a broadcast limiter.
* ASR quality is exactly your caption track's quality. Auto-ASR tracks are word-timed but
  mis-capitalise and repeat; `captions.clean` fixes the mechanical parts and leaves the
  rest to the writer.
* Publishing is a single upload, not a distribution system: no scheduling, no cross-post
  to TikTok/Reels, no analytics loop back into the next script. The `Brief` → `Script`
  path is where that feedback would attach.
* Downloading other people's videos is your ToS and copyright problem, not this code's.
  Default behaviour is to reuse *your* source; `assets.json` records every reused span.
* `--only publish` on a job that never rendered is rejected with a message naming the
  missing stage, not a stack trace.
* YouTube is sometimes unreachable (datacenter IPs, age gates, this sandbox). `fetch`
  reports that as one line pointing at `--source /path/to/clip.mp4` rather than relaying
  yt-dlp's issue-tracker boilerplate — but it cannot work around a block, so the fetch
  step has to run somewhere with access.
