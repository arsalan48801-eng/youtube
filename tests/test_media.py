"""Caption + transcript layer: the parts that decide whether the video reads well."""

from __future__ import annotations

import pytest

from ytpipe.captions import CaptionStyle, build_ass, plan_cues, regroup, to_ass, to_srt
from ytpipe.models import CaptionCue, Shot
from ytpipe.stages.transcribe import parse_srt, parse_webvtt


def cues(*specs: tuple[float, float, str]) -> list[CaptionCue]:
    return [CaptionCue(start=a, end=b, text=t) for a, b, t in specs]


class TestWebVTTParsing:
    def test_plain_cues(self):
        raw = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello there friend.\n"
        segs = parse_webvtt(raw)
        assert len(segs) == 1
        assert segs[0].start == 1.0 and segs[0].text == "Hello there friend."

    def test_word_timestamps_are_kept_as_word_segments(self):
        """Flattening these would throw away the only real caption timing there is."""
        raw = (
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n"
            "<00:00:00.200><00:00:00.800><00:00:01.400>one two three\n"
        )
        segs = parse_webvtt(raw)
        assert [s.text for s in segs] == ["one", "two", "three"]
        assert segs[1].start == pytest.approx(0.8)
        assert segs[2].end == pytest.approx(2.0)

    def test_rolling_captions_are_not_duplicated(self):
        """YouTube repeats the previous line in every cue; the transcript must not."""
        raw = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\nfirst line here\n\n"
            "00:00:02.000 --> 00:00:04.000\nfirst line here second line now\n\n"
            "00:00:04.000 --> 00:00:06.000\nsecond line now third line next\n"
        )
        segs = parse_webvtt(raw)
        text = " ".join(s.text for s in segs)
        assert text.count("first line here") == 1

    def test_tags_and_cue_settings_are_stripped(self):
        raw = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000 line:80% position:50% align:center\n<c>Styled</c> text &amp; more\n"
        segs = parse_webvtt(raw)
        assert "<" not in segs[0].text
        assert "Styled text & more" in segs[0].text

    def test_srt_fallback(self):
        raw = "1\n00:00:00,500 --> 00:00:02,000\nSentence one.\n\n2\n00:00:02,000 --> 00:00:04,000\nSentence two.\n"
        segs = parse_srt(raw)
        assert [s.text for s in segs] == ["Sentence one.", "Sentence two."]
        assert segs[0].start == pytest.approx(0.5)


class TestRegroup:
    def test_packs_words_and_splits_long_cues(self):
        out = regroup(cues((0.0, 4.0, "one two three four five six seven eight")), max_words=3, max_seconds=9)
        assert [c.text for c in out][:3] == ["one two three", "four five six", "seven eight"]

    def test_words_are_timed_by_position_not_squashed(self):
        out = regroup(cues((0.0, 2.0, "alpha beta")), max_words=4)
        assert out[0].start == pytest.approx(0.0)
        assert out[0].end - out[0].start > 0.2

    def test_never_merges_across_a_finished_sentence(self):
        """Packing the next line's first words onto "Done." reads as a typo."""
        out = regroup(cues((0.0, 1.0, "Done."), (1.0, 2.0, "Next here")), max_words=8, min_seconds=0.1)
        assert [c.text for c in out] == ["Done.", "Next here"]

    def test_timeline_is_monotonic(self):
        out = regroup(cues((0.0, 1.0, "a b c d e f"), (1.0, 1.05, "g h"), (1.05, 9.0, "i j k")), max_words=2)
        starts = [c.start for c in out]
        assert starts == sorted(starts)
        assert all(c.end > c.start for c in out)


class TestAssAndSrt:
    def test_srt_uses_comma_milliseconds(self):
        out = to_srt(cues((1.5, 2.25, "one two")))
        assert "00:00:01,500 --> 00:00:02,250" in out
        assert out.startswith("1\n")

    def test_ass_has_per_word_highlight_timing(self):
        items = cues((0.0, 2.0, "one two three"))
        out = to_ass(items, width=1080, height=1920, style=CaptionStyle())
        assert out.count("\\kf") == 3
        assert "PlayResX: 1080" in out

    def test_build_ass_adds_overlay_layer_and_wraps_style_sizes(self):
        cap = cues((0.3, 1.2, "the spec sheet"))
        overlays = cues((0.0, 2.0, "Spec"))
        out = build_ass(cap, width=540, height=960, style=CaptionStyle(), overlays=overlays)
        assert "Style: CAP," in out and "Style: POP," in out
        assert "PlayResX: 540" in out
        assert "Dialogue: 1," in out  # overlay on its own layer
        assert "WrapStyle: 0" in out  # must be able to wrap, not clip

    def test_overlay_font_shrinks_for_long_text(self):
        short = build_ass(cues((0, 1, "x")), width=1080, height=1920, style=CaptionStyle(), overlays=cues((0, 1, "No")))
        long = build_ass(cues((0, 1, "x")), width=1080, height=1920, style=CaptionStyle(), overlays=cues((0, 1, "A much longer phrase here")))
        def pop_size(doc: str) -> int:
            return int(doc.split("Style: POP,")[1].split(",")[1])

        assert pop_size(long) < pop_size(short)

    def test_braces_in_speech_cannot_break_the_ass_script(self):
        out = to_ass(cues((0.0, 1.0, "if {x} then")), width=1080, height=1920, style=CaptionStyle())
        event = out.split("Dialogue:")[1]
        # Only tag braces (\kf..) may remain; a stray one would swallow the cue.
        assert "{x}" not in event and "(" in event

    def test_plan_cues_uses_the_speech_window_not_the_shot_window(self):
        shot = Shot(index=0, start=0.0, end=5.0, speech_start=0.35, speech_end=4.0, narration="one two three four")
        out = plan_cues([shot], vertical=True)
        assert out[0].start == pytest.approx(0.35)
        assert out[-1].end <= 4.05


class TestDemoFixture:
    """`ytpipe demo` is the documented no-network entry point, so it is tested as such."""

    def test_demo_writes_a_clip_and_word_timed_captions(self, tmp_path, ffmpeg):
        from typer.testing import CliRunner

        from ytpipe.cli import app

        out = tmp_path / "fixtures" / "demo"
        result = CliRunner().invoke(app, ["demo", "--out", str(out), "--seconds", "6"])
        assert result.exit_code == 0, result.output
        assert out.with_suffix(".mp4").exists()
        vtt = out.with_suffix(".vtt").read_text(encoding="utf-8")
        assert vtt.startswith("WEBVTT")
        assert vtt.count("<00:00:") > 20, "words must carry their own timestamps"

        info = ffmpeg.probe(out.with_suffix(".mp4"))
        assert 5.0 <= info.duration <= 7.5
        assert info.has_audio, "the demo clip needs a track for transcribe to mine"
