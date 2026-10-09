"""Filtergraph assembly.

The edit stage used to render each shot to a file and stitch them with the concat
demuxer. That is the right tool when shot clips arrive from elsewhere, but for a
pipeline that owns the whole timeline it costs a second encode of every frame and
depends on the intermediate container being seek-safe -- and the static ffmpeg many
people end up with (the PyPI ``imageio-ffmpeg`` wheel) segfaults on MPEG-TS *input*,
which is precisely the failure this design avoids.

So: one command, N shot branches, ``concat`` filter, captions, loudness, one encode.
This module is pure string building -- no subprocess -- so the graph is
unit-testable without ffmpeg being present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ffmpeg import escape_filter_path


@dataclass
class ShotInput:
    """One branch of the graph: a picture, a narration track, a fixed length.

    ``still`` means "loop this image for the whole shot" -- passing a PNG straight in
    avoids pre-encoding a clip that the assembly pass would only encode again.
    ``video=None`` means no source at all: the branch is drawn from a flat background,
    which is what the offline card fallback needs.
    """

    index: int
    seconds: float
    video: str | None = None
    audio: str | None = None
    background: str = "0x0b1220"
    zoom: float = 1.0
    pan: str = "none"
    still: bool = False
    #: Seconds of silence held before the narration inside this shot (the lead-in).
    #: Padding the shot absorbs it instead of a gap between shots, which `concat`
    #: would silently delete and desynchronise every caption after it.
    audio_delay: float = 0.0


@dataclass
class AssemblySpec:
    shots: list[ShotInput]
    width: int
    height: int
    fps: int = 30
    transition: str = "dip"  # cut | dip | xfade
    xfade_seconds: float = 0.3
    ass: str | None = None
    music_path: str | None = None
    music_gain_db: float = -18.0
    loudness_i: float = -14.0
    tail_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        raw = sum(s.seconds for s in self.shots)
        if self.transition == "xfade" and len(self.shots) > 1:
            raw -= self.xfade_seconds * (len(self.shots) - 1)
        return raw + self.tail_seconds


def _pan_expr(pan: str) -> tuple[str, str]:
    """zoompan x/y expressions. ``on`` is the output frame counter."""
    centre = ("'iw/2-(iw/zoom/2)'", "'ih/2-(ih/zoom/2)'")
    return {
        "none": centre,
        "in": centre,
        "left-right": ("'(iw-iw/zoom)*on/100'", centre[1]),
        "right-left": ("'(iw-iw/zoom)*(1-on/100)'", centre[1]),
        "up-down": (centre[0], "'(ih-ih/zoom)*on/100'"),
        "down-up": (centre[0], "'(ih-ih/zoom)*(1-on/100)'"),
    }.get(pan, centre)


def _fades(seconds: float, fade: float) -> tuple[list[str], list[str]]:
    """Short dip-to-black at both ends of a shot. Keeps hard cuts from strobing."""
    if fade <= 0 or seconds <= 0:
        return [], []
    f = min(fade, seconds / 3)
    return (
        [f"fade=t=in:st=0:d={f:.3f}", f"fade=t=out:st={max(0.0, seconds - f):.3f}:d={f:.3f}"],
        [
            f"afade=t=in:st=0:d={f:.3f}",
            f"afade=t=out:st={max(0.0, seconds - f):.3f}:d={f:.3f}",
        ],
    )


def _video_chain(shot: ShotInput, spec: AssemblySpec, fade: float) -> str:
    w, h, fps = spec.width, spec.height, spec.fps
    chain = [
        f"trim=duration={shot.seconds:.3f}",
        "setpts=PTS-STARTPTS",
        f"scale={w}:{h}:force_original_aspect_ratio=increase",
        f"crop={w}:{h}",
        "setsar=1",
        f"fps={fps}",
    ]
    if abs(shot.zoom - 1.0) > 1e-3:
        step = (shot.zoom - 1.0) / max(1.0, shot.seconds * fps)
        x, y = _pan_expr(shot.pan)
        # d=1 => one output frame per input frame. Larger d *duplicates* each frame,
        # which is what a still image needs and what would stretch a video clip.
        chain.append(
            f"zoompan=z='min(zoom+{step:.6f},{shot.zoom:.3f})':d=1:x={x}:y={y}:s={w}x{h}"
        )
    vf, _ = _fades(shot.seconds, fade)
    chain += vf
    chain.append("format=yuv420p")
    return ",".join(chain)


def build_graph(spec: AssemblySpec) -> tuple[str, list[str]]:
    """Return ``(filter_complex, input_args)`` for a single-pass assembly.

    Every branch is normalised to the same canvas, frame rate, pixel format, sample
    rate and layout before ``concat``: that filter refuses mismatched inputs, so the
    normalisation is not cosmetic. Input args and stream labels are derived from one
    pass over the shots, because a label pointing at the wrong input is the classic
    silent-corruption bug in hand-written graphs.
    """
    if not spec.shots:
        raise ValueError("AssemblySpec needs at least one shot")

    fade = 0.18 if spec.transition == "dip" else 0.0
    inputs: list[str] = []
    parts: list[str] = []
    v_labels: list[str] = []
    a_labels: list[str] = []
    idx = 0  # explicit input counter; "-loop"/"-f lavfi" break any args//2 arithmetic

    silent_map: dict[int, int] = {}
    silent_pos = 0

    for pos, shot in enumerate(spec.shots):
        if shot.video and shot.still:
            inputs += ["-loop", "1", "-framerate", str(spec.fps), "-i", shot.video]
            idx += 1
        elif shot.video:
            inputs += ["-i", shot.video]
            idx += 1
        else:
            inputs += [
                "-f",
                "lavfi",
                "-i",
                f"color=c={shot.background}:s={spec.width}x{spec.height}:r={spec.fps}",
            ]
            idx += 1
        v_ref = f"{idx - 1}:v"
        if shot.audio:
            inputs += ["-i", shot.audio]
            a_ref = f"{idx}:a"
            idx += 1
        else:
            # Never fall back to the picture's own audio: b-roll cut from a source
            # clip has none (we strip it on purpose), and an image has no stream at
            # all. Silence from a single shared anullsrc is the honest answer.
            a_ref = f"sil{silent_pos}"
            silent_map[pos] = silent_pos
            silent_pos += 1

        pos = len(v_labels)
        parts.append(f"[{v_ref}]{_video_chain(shot, spec, fade)}[v{pos}]")
        _, af = _fades(shot.seconds, fade)
        delay = f"adelay={int(shot.audio_delay * 1000)}:all=1," if shot.audio_delay > 0 else ""
        parts.append(
            f"[{a_ref}]atrim=duration={shot.seconds:.3f},asetpts=PTS-STARTPTS,{delay}apad,"
            f"atrim=duration={shot.seconds:.3f},"
            "aformat=sample_rates=48000:channel_layouts=stereo"
            + ("," + ",".join(af) if af else "")
            + f"[a{pos}]"
        )
        v_labels.append(f"[v{pos}]")
        a_labels.append(f"[a{pos}]")

    if silent_map:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        parts.append(
            f"[{idx}:a]asplit={len(silent_map)}"
            + "".join(f"[sil{n}]" for n in range(len(silent_map)))
        )
        idx += 1

    music_ref = None
    if spec.music_path:
        inputs += ["-stream_loop", "-1", "-i", spec.music_path]
        music_ref = f"{idx}:a"
        idx += 1

    if spec.transition == "xfade" and len(v_labels) > 1:
        vx, ax = _xfade_chain(parts, v_labels, a_labels, spec)
    else:
        joined = "".join(f"{v}{a}" for v, a in zip(v_labels, a_labels, strict=True))
        parts.append(f"{joined}concat=n={len(v_labels)}:v=1:a=1[vcat][acat]")
        vx, ax = "[vcat]", "[acat]"

    tail: list[str] = []
    if spec.tail_seconds > 0:
        tail.append(f"tpad=stop_mode=clone:stop_duration={spec.tail_seconds:.3f}")
    if spec.ass:
        tail.append(f"subtitles=filename={escape_filter_path(spec.ass)}:charenc=UTF-8")
    tail.append(f"fps={spec.fps}")
    parts.append(f"{vx}{','.join(tail)}[vout]")

    loud = f"loudnorm=I={spec.loudness_i}:TP=-1.5:LRA=11"
    aformat = "aformat=sample_rates=48000:channel_layouts=stereo"
    if music_ref:
        # `sidechaincompress` writes its *first* input, keyed by its second, and an
        # ffmpeg pad can only be read once -- so the voiceover is split: one copy is
        # the key, the other is what the bed is mixed back against.
        gain = 10 ** (spec.music_gain_db / 20)
        parts.append(f"{ax}asplit=2[vokey][vomix]")
        parts.append(f"[{music_ref}]{aformat},volume={gain:.4f}[bed]")
        parts.append(
            "[bed][vokey]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=250[duck]"
        )
        parts.append(
            f"[vomix][duck]amix=inputs=2:duration=first:dropout_transition=0,{aformat},"
            f"{loud}[aout]"
        )
    else:
        parts.append(f"{ax}{loud},{aformat}[aout]")

    return ";\n".join(parts), inputs


def _xfade_chain(
    parts: list[str], v_labels: list[str], a_labels: list[str], spec: AssemblySpec
) -> tuple[str, str]:
    """Chain ``xfade``/``acrossfade`` with cumulative offsets.

    Each offset is measured from the *start of the current combined stream*, so it
    accumulates the previous shot's length minus the overlap; getting that bookkeeping
    wrong produces a frozen tail, which is why it is isolated and unit-tested.
    """
    offset = 0.0
    vx, ax = v_labels[0], a_labels[0]
    for i in range(1, len(spec.shots)):
        prev = spec.shots[i - 1].seconds
        offset = max(0.0, offset + prev - spec.xfade_seconds)
        parts.append(
            f"{vx}{v_labels[i]}xfade=transition=fade:duration={spec.xfade_seconds:.3f}:"
            f"offset={offset:.3f}[vx{i}]"
        )
        parts.append(f"{ax}{a_labels[i]}acrossfade=d={spec.xfade_seconds:.3f}:c1=tri:c2=tri[ax{i}]")
        vx, ax = f"[vx{i}]", f"[ax{i}]"
    return vx, ax


def encode_args(
    spec: AssemblySpec, out_path: str, *, vmap: str, amap: str, crf: int = 20, preset: str = "veryfast"
) -> list[str]:
    """The output half of the command, kept separate so tests can assert on it."""
    return [
        "-map",
        vmap,
        "-map",
        amap,
        "-t",
        f"{spec.total_seconds:.3f}",
        "-r",
        str(spec.fps),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        out_path,
    ]
