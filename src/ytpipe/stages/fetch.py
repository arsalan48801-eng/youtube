"""Stage 1 — get the source: metadata, media, and the caption track.

Accepts a YouTube URL, a bare video id, or a local file. Local files matter more
than they look: `--source ./clip.mp4` runs the whole pipeline without network
access, which is how the edit stages stay testable and how you can repurpose any
downloaded footage.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..errors import FetchError
from ..models import SourceVideo, StageName

#: yt-dlp writes stack-trace-shaped errors aimed at its own issue tracker. A pipeline
#: user needs the opposite: one line, plus what to do next when the cause is a blocked
#: network rather than a bad URL.
_NETWORK_MARKERS = (
    "tls/ssl",
    "unable to download api page",
    "unable to connect",
    "urlopen error",
    "timed out",
    "temporary failure in name resolution",
    "network is unreachable",
    "http error 5",
)


def explain_failure(text: str) -> str:
    """Shorten a yt-dlp failure into something actionable."""
    noise = ("please report this issue", "filling out the appropriate", "confirm you are on the latest")
    kept: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        for marker in noise:
            cut = min(cut, line.lower().find(marker) if marker in line.lower() else cut)
        line = line[:cut].strip(" ;,.")
        if line:
            kept.append(line)
    first = " ".join(kept)[:220].rstrip(" ;,.") or "yt-dlp failed"
    low = first.lower()
    if any(m in low for m in _NETWORK_MARKERS):
        return (
            f"{first}. YouTube looks unreachable from this machine -- run the fetch where "
            f"you have access, or pass a file you downloaded: ytpipe run /path/to/clip.mp4"
        )
    gated = ("private video", "sign in to confirm", "members-only", "not available in your country")
    if any(m in low for m in gated):
        return f"{first}. This video cannot be fetched anonymously; use a local file as --source."
    return first

_URL_RE = re.compile(r"^https?://", re.I)
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def resolve_source(source: str) -> tuple[str | None, str | None]:
    """Classify input into (url, local_path). Bare ids become canonical watch URLs."""
    source = source.strip()
    if _URL_RE.match(source):
        return source, None
    path = Path(source).expanduser()
    if path.exists():
        return None, str(path.resolve())
    if _ID_RE.match(source):
        return f"https://www.youtube.com/watch?v={source}", None
    raise FetchError(f"source is neither a URL nor an existing file: {source!r}")


def to_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


class FetchStage:
    name = StageName.FETCH

    def run(self, ctx: Any) -> SourceVideo:  # noqa: C901 - linear, many optional fields
        source = ctx.job.request.source
        url, local = resolve_source(source)
        if local is not None:
            return self._from_local(ctx, Path(local))
        if ctx.settings.offline:
            raise FetchError(
                "offline mode cannot fetch; pass a local file as --source or drop --offline"
            )
        return self._from_youtube(ctx, url or source)

    # -- local ----------------------------------------------------------------

    def _from_local(self, ctx: Any, path: Path) -> SourceVideo:
        ctx.emit(f"local source {path.name}")
        info = ctx.ffmpeg.probe(path)
        if not info.duration:
            raise FetchError(f"could not read duration of {path} (is it a video?)")
        media = path
        if path.parent != ctx.dir:
            media = ctx.path(path.name)
            if not media.exists():
                import shutil

                shutil.copy2(path, media)
        # A sibling .vtt/.srt next to the media is treated as its caption track, so
        # local runs get the same transcript path as a YouTube download.
        subs = None
        for suffix in (".en-orig.vtt", ".vtt", ".srt", ".ass"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                subs = candidate
                break
        if subs:
            ctx.emit(f"local caption track: {subs.name}")
        return SourceVideo(
            video_id=path.stem,
            url=None,
            title=path.stem.replace("_", " ").replace("-", " ").strip().title(),
            duration=info.duration,
            media_path=media,
            subtitle_path=subs,
            stats={
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "vcodec": info.vcodec,
                "acodec": info.acodec,
                "origin": "local",
            },
        )

    # -- youtube ---------------------------------------------------------------

    def _ydl_options(self, ctx: Any) -> dict[str, Any]:
        request = ctx.job.request
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "outtmpl": str(ctx.dir / "source.%(ext)s"),
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [request.language] if request.language else ["en.*", "en"],
            "subtitlesformat": "vtt/srt/best",
            "retries": 3,
            "socket_timeout": 30,
            "cachedir": str(ctx.dir / ".ytdlp-cache"),
            "noprogress": True,
        }
        if request.transcript_only:
            # "Just read the link": one metadata page and one subtitle file, tens of KB,
            # no media. It also avoids the streaming URLs, which is where bot-checks
            # usually fail, so this mode works in more places than a real download.
            opts["skip_download"] = True
            return opts
        # Vertical-first: 1080p max keeps renders fast; the pipeline re-times captions
        # itself so we only need the audio and picture, not 4K.
        opts.update(
            {
                "format": (
                    "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
                    "best[ext=mp4][height<=1080]/bestvideo+bestaudio/best"
                ),
                "merge_output_format": "mp4",
                "writethumbnail": True,
                "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
            }
        )
        return opts

    def _from_youtube(self, ctx: Any, url: str) -> SourceVideo:
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover
            raise FetchError("yt-dlp is not installed; `pip install yt-dlp`") from exc

        only_text = bool(ctx.job.request.transcript_only)
        ctx.emit(f"reading {url}" + (" (transcript only, no download)" if only_text else ""))
        try:
            with yt_dlp.YoutubeDL(self._ydl_options(ctx)) as ydl:
                info = ydl.extract_info(url, download=not only_text)
        except Exception as exc:
            raise FetchError(explain_failure(str(exc) or repr(exc))) from exc
        if not info:
            raise FetchError("yt-dlp returned no metadata")
        if info.get("_type") == "playlist":  # pragma: no cover - noplaylist guards this
            entries = [e for e in info.get("entries", []) if e]
            if not entries:
                raise FetchError("playlist was empty")
            info = entries[0]

        media = None if only_text else self._find_media(ctx.dir, info)
        if only_text:
            poster = None
        elif media is None:
            raise FetchError(f"download produced no playable file in {ctx.dir}")

        duration = float(info.get("duration") or 0.0) or (
            ctx.ffmpeg.duration(media) if media else 0.0
        )
        poster = None if only_text else next(iter(sorted(ctx.dir.glob("source*.jpg"))), None)
        subs = self._find_captions(ctx.dir)
        if only_text and subs is None:
            raise FetchError(
                "this video has no caption track, so there is nothing to read without "
                "downloading it; drop --transcript-only to fetch the audio for ASR, or "
                "pass a local file with a sibling .vtt/.srt"
            )
        if subs:
            ctx.emit(f"caption track: {subs.name}")
        else:
            ctx.emit("no caption track returned; transcribe will fall back to ASR", level="warning")

        return SourceVideo(
            video_id=info.get("id"),
            url=info.get("webpage_url") or url,
            title=info.get("title") or "",
            channel=info.get("uploader") or info.get("channel"),
            description=info.get("description") or "",
            duration=duration,
            publish_date=str(info.get("upload_date") or ""),
            view_count=info.get("view_count"),
            like_count=info.get("like_count"),
            language=(ctx.job.request.language or (subs.stem.split(".")[-1] if subs else None)),
            media_path=media,
            poster_path=poster,
            subtitle_path=subs,
            stats={
                "width": info.get("width"),
                "height": info.get("height"),
                "fps": info.get("fps"),
                "abr": info.get("abr"),
                "resolution": info.get("resolution"),
                "uploader_id": info.get("uploader_id"),
                "comment_count": info.get("comment_count"),
                "average_rating": info.get("average_rating"),
                "origin": "youtube",
            },
        )

    @staticmethod
    def _find_media(stage_dir: Path, info: dict[str, Any]) -> Path | None:
        requested = (info.get("requested_downloads") or [{}])[0].get("filepath")
        if requested and Path(requested).exists():
            return Path(requested)
        candidates = [
            p
            for p in sorted(stage_dir.glob("source.*"))
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4a", ".opus"}
        ]
        return candidates[0] if candidates else None

    @staticmethod
    def _find_captions(stage_dir: Path) -> Path | None:
        order = (".vtt", ".srt", ".ass")
        for suffix in order:
            found = sorted(p for p in stage_dir.glob("source*.en*"+suffix)) or sorted(
                p for p in stage_dir.glob(f"source*{suffix}")
            )
            if found:
                return found[0]
        return None
