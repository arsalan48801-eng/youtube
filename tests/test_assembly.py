"""Filtergraph construction. Pure string logic, so it is checked without ffmpeg."""

from __future__ import annotations

import re

import pytest

from ytpipe.assembly import AssemblySpec, ShotInput, build_graph, encode_args


def graph(spec: AssemblySpec) -> tuple[str, list[str]]:
    return build_graph(spec)


def input_indices(inputs: list[str]) -> int:
    return inputs.count("-i")


class TestGraphStructure:
    def test_every_branch_is_normalised_before_concat(self):
        spec = AssemblySpec(
            shots=[ShotInput(index=0, seconds=2.0, video="/tmp/a.mp4", audio="/tmp/a.wav")],
            width=1080, height=1920,
        )
        g, _inputs = graph(spec)
        for token in ("scale=1080:1920", "crop=1080:1920", "setsar=1", "fps=30", "format=yuv420p",
                      "aformat=sample_rates=48000:channel_layouts=stereo"):
            assert token in g

    def test_shot_durations_are_trimmed_exactly(self):
        spec = AssemblySpec(
            shots=[ShotInput(index=i, seconds=1.5 + i, video=f"/tmp/{i}.mp4") for i in range(3)],
            width=540, height=960,
        )
        g, _inputs = graph(spec)
        assert "concat=n=3:v=1:a=1" in g
        for i in range(3):
            assert f"trim=duration={1.5 + i:.3f}" in g

    def test_total_length_is_the_sum_of_shots(self):
        """`concat` deletes gaps between shots, so padding must live inside a shot."""
        spec = AssemblySpec(
            shots=[ShotInput(index=0, seconds=2.0), ShotInput(index=1, seconds=3.0)],
            width=540, height=960, tail_seconds=0.25,
        )
        assert spec.total_seconds == pytest.approx(5.25)

    def test_xfade_shortens_the_timeline_by_the_overlaps(self):
        spec = AssemblySpec(
            shots=[ShotInput(index=i, seconds=2.0) for i in range(3)],
            width=540, height=960, transition="xfade", xfade_seconds=0.4,
        )
        assert spec.total_seconds == pytest.approx(6.0 - 0.8)
        g, _inputs = graph(spec)
        offsets = [float(m) for m in re.findall(r"offset=([\d.]+)", g)]
        assert offsets == pytest.approx([1.6, 3.2])

    def test_labels_match_input_order_after_mixed_input_kinds(self):
        """The classic silent corruption: a label pointing at the wrong input."""
        spec = AssemblySpec(
            shots=[
                ShotInput(index=0, seconds=1.0, video="/tmp/a.mp4", audio="/tmp/a.wav"),
                ShotInput(index=1, seconds=1.0),  # lavfi background, no audio file
                ShotInput(index=2, seconds=1.0, video="/tmp/i.png", still=True),
            ],
            width=540, height=960,
        )
        g, inputs = graph(spec)
        # a.mp4, a.wav, lavfi background, png, plus one shared anullsrc for the two
        # shots that have no narration file.
        assert input_indices(inputs) == 5
        assert "-loop 1" in " ".join(inputs)
        used = {int(m) for m in re.findall(r"\[(\d+):[va]\]", g)}
        assert used == {0, 1, 2, 3, 4}
        assert "[5:" not in g, "no label may point past the declared inputs"

    def test_missing_narration_uses_one_shared_silence_source(self):
        spec = AssemblySpec(
            shots=[ShotInput(index=i, seconds=1.0, video=f"/tmp/{i}.mp4") for i in range(3)],
            width=540, height=960,
        )
        g, inputs = graph(spec)
        assert "anullsrc" in " ".join(inputs)
        assert re.search(r"asplit=3", g)
        for i in range(3):
            assert f"[sil{i}]" in g

    def test_music_bed_is_ducked_under_the_voiceover(self):
        spec = AssemblySpec(
            shots=[ShotInput(index=0, seconds=1.0, audio="/tmp/n.wav", video="/tmp/v.mp4")],
            width=540, height=960, music_path="/tmp/music.mp3",
        )
        g, inputs = graph(spec)
        assert "-stream_loop" in inputs
        assert "sidechaincompress" in g and "amix=inputs=2" in g
        assert "volume=0.125" in g  # -18 dB

    def test_voiceover_is_split_once_for_the_sidechain(self):
        """An ffmpeg pad can only be read once; feeding [acat] to two filters is a
        hard graph error, so the duck path must asplit first."""
        spec = AssemblySpec(
            shots=[ShotInput(index=0, seconds=1.0, video="/tmp/v.mp4", audio="/tmp/n.wav")],
            width=540, height=960, music_path="/tmp/music.mp3",
        )
        g, _inputs = graph(spec)
        # Once to define it (concat output), once to read it -- a third would mean a
        # pad consumed twice, which ffmpeg rejects at graph-bind time.
        assert g.count("[acat]") == 2, "[acat] must be consumed exactly once"
        assert "asplit=2[vokey][vomix]" in g
        assert "[vomix][duck]amix" in g

    def test_still_images_are_looped_instead_of_pre_encoded(self):
        spec = AssemblySpec(
            shots=[ShotInput(index=0, seconds=2.5, video="/tmp/x.png", still=True)],
            width=540, height=960,
        )
        _g, inputs = graph(spec)
        assert inputs[:4] == ["-loop", "1", "-framerate", "30"]
        assert inputs[4] == "-i" and inputs[5] == "/tmp/x.png"

    def test_zoompan_uses_one_output_frame_per_input_frame(self):
        """d>1 duplicates frames: right for a still, wrong for a video clip."""
        spec = AssemblySpec(
            shots=[ShotInput(index=0, seconds=2.0, video="/tmp/a.mp4", zoom=1.2, pan="left-right")],
            width=540, height=960,
        )
        g, _inputs = graph(spec)
        assert re.search(r"zoompan=.*:d=1:", g)
        assert "on/100" in g

    def test_lead_in_delays_audio_inside_the_shot(self):
        spec = AssemblySpec(
            shots=[
                ShotInput(index=0, seconds=2.0, video="/tmp/a.mp4", audio="/tmp/a.wav", audio_delay=0.35),
                ShotInput(index=1, seconds=2.0, video="/tmp/b.mp4", audio="/tmp/b.wav"),
            ],
            width=540, height=960,
        )
        g, _inputs = graph(spec)
        assert "adelay=350:all=1" in g
        assert g.count("adelay") == 1

    def test_ass_path_is_escaped_into_the_graph(self):
        spec = AssemblySpec(
            shots=[ShotInput(index=0, seconds=1.0, video="/tmp/a.mp4")],
            width=540, height=960, ass="/tmp/we:ird/f.ass",
        )
        g, _inputs = graph(spec)
        assert "subtitles=filename=" in g
        assert "\\:" in g, "colons must be escaped inside a filtergraph"

    def test_empty_shot_list_is_rejected(self):
        with pytest.raises(ValueError):
            build_graph(AssemblySpec(shots=[], width=540, height=960))


class TestEncodeArgs:
    def test_upload_safe_output_flags(self):
        spec = AssemblySpec(shots=[ShotInput(index=0, seconds=2.0)], width=540, height=960)
        args = encode_args(spec, "/tmp/out.mp4", vmap="[vout]", amap="[aout]")
        joined = " ".join(args)
        for flag in ("-c:v libx264", "-pix_fmt yuv420p", "-c:a aac", "+faststart", "-profile:v high"):
            assert flag in joined
        assert "-shortest" not in joined, "-t and -shortest fight; -t owns the length"
        assert args[args.index("-t") + 1] == "2.000"
