"""ffmpeg layer.

Two constraints shaped this module:

1. The PyPI-delivered ffmpeg (``imageio-ffmpeg``) ships **without ffprobe**, so
   metadata is read by parsing ``ffmpeg -i`` stderr.
2. That build also lacks ``drawtext`` but has **libass**, so all on-screen text
   goes through ASS subtitles. That turns out to be the better choice anyway:
   one styled script controls the whole caption look.

Every helper here is pure ffmpeg I/O and works offline, which is what makes the
edit stage testable without a network.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?: Video: .*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"([\d.]+) fps")
_VCODEC_RE = re.compile(r"Video:\s*([A-Za-z0-9_.-]+)")
_ACODEC_RE = re.compile(r"Audio:\s*([A-Za-z0-9_.-]+)")


def ass_time(seconds: float) -> str:
    total = max(0.0, seconds)
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total % 60
    return f"{h}:{m:02d}:{s:05.2f}"


@dataclass
class MediaInfo:
    path: Path
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str | None = None
    acodec: str | None = None
    raw: str = field(default="", repr=False)

    @property
    def has_audio(self) -> bool:
        return bool(self.acodec)

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


class FFmpeg:
    """Small, explicit wrapper: build a filtergraph, run it, read it back."""

    def __init__(self, binary: str | None = None) -> None:
        if binary is None:
            found = shutil.which("ffmpeg")
            if found:
                binary = found
            else:
                try:
                    import imageio_ffmpeg

                    binary = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
                    ) from exc
        self.binary = str(binary)

    # -- process --------------------------------------------------------------

    def run(self, args: Sequence[str], *, timeout: int = 900, quiet: bool = True) -> str:
        cmd = [self.binary, "-hide_banner", "-nostdin", "-y" if quiet else "-n", *args]
        log.debug("ffmpeg: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
            raise RuntimeError(f"ffmpeg exited {proc.returncode}\n" + "\n".join(tail))
        return proc.stderr or ""

    def version(self) -> str:
        out = subprocess.run(
            [self.binary, "-version"], capture_output=True, text=True, timeout=30
        ).stdout
        return out.splitlines()[0] if out else "unknown"

    def has_filter(self, name: str) -> bool:
        out = subprocess.run(
            [self.binary, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=60
        ).stdout
        return re.search(rf"\s{name}\s", out) is not None

    # -- inspection -----------------------------------------------------------

    def probe(self, path: Path | str) -> MediaInfo:
        """ffprobe-free metadata: parse the ``ffmpeg -i`` banner."""
        path = Path(path)
        proc = subprocess.run(
            [self.binary, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = proc.stderr or ""
        info = MediaInfo(path=path, raw=text)
        if m := _DURATION_RE.search(text):
            h, mnt, sec = m.groups()
            info.duration = int(h) * 3600 + int(mnt) * 60 + float(sec)
        if m := _VIDEO_RE.search(text):
            info.width, info.height = int(m.group(1)), int(m.group(2))
        if m := _FPS_RE.search(text):
            fps = float(m.group(1))
            info.fps = fps if 0 < fps < 241 else 30.0
        if m := _VCODEC_RE.search(text):
            info.vcodec = m.group(1)
        if m := _ACODEC_RE.search(text):
            info.acodec = m.group(1)
        return info

    def duration(self, path: Path | str) -> float:
        return self.probe(path).duration

    # -- primitives -----------------------------------------------------------

    def silence(self, out: Path, seconds: float, rate: int = 48000) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.run(
            [
                "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=stereo",
                "-t", f"{seconds:.3f}", *_audio_codec(out), str(out),
            ]
        )
        return out

    def tone(self, out: Path, seconds: float, freq: int = 220, gain_db: float = -24) -> Path:
        """Deterministic audio bed; used as the offline narration stand-in."""
        out.parent.mkdir(parents=True, exist_ok=True)
        self.run(
            [
                "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000",
                "-af", f"volume={10 ** (gain_db / 20):.4f}", "-t", f"{seconds:.3f}",
                *_audio_codec(out), str(out),
            ]
        )
        return out

    def extract_audio(self, src: Path | str, out: Path, *, wav: bool = True) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        codec = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1"] if wav else ["-c:a", "aac"]
        self.run(["-i", str(src), "-vn", *codec, str(out)])
        return out

    def cut(self, src: Path | str, out: Path, start: float, end: float, *, copy: bool = False) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        dur = max(0.04, float(end) - float(start))
        args = ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}"]
        args += ["-c", "copy"] if copy else ["-c:v", "libx264", "-preset", "veryfast"]
        self.run([*args, str(out)])
        return out

    def frame(self, src: Path | str, out: Path, at: float) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.run(["-ss", f"{max(0.0, at):.3f}", "-i", str(src), "-frames:v", "1", str(out)])
        return out

    # -- output -------------------------------------------------------------

    def thumbnail(
        self, src: Path, out: Path, *, at: float = 1.0, size: tuple[int, int] | None = None
    ) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        vf = f"scale={size[0]}:{size[1]}" if size else "scale=-2:720"
        self.run(["-ss", f"{at:.3f}", "-i", str(src), "-frames:v", "1", "-vf", vf, str(out)])
        return out


def _audio_codec(out: Path | str) -> list[str]:
    """Codec matching the container. Hard-coding AAC here broke ``.mp3`` writes:
    ffmpeg picks the muxer from the extension and then refuses the stream."""
    suffix = Path(out).suffix.lower()
    return {
        ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
        ".wav": ["-c:a", "pcm_s16le"],
        ".ogg": ["-c:a", "libvorbis", "-b:a", "160k"],
        ".flac": ["-c:a", "flac"],
    }.get(suffix, ["-c:a", "aac", "-b:a", "160k"])


def escape_filter_path(path: Path | str) -> str:
    """Quote a path for use inside an ffmpeg filtergraph.

    ``subtitles=filename=`` resolves relative paths against the *ffmpeg process*
    cwd, never against the file's own directory, so everything here is absolute;
    ``:`` and ``'`` are filtergraph metacharacters.
    """
    text = str(Path(path).resolve()).replace("\\", "/")
    return text.replace(":", "\\:").replace("'", "\\'")
