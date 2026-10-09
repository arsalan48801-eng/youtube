"""Provider clients: JSON recovery, HTTP contracts, upload-body limits, offline gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ytpipe.config import LLMSettings, TTSSettings, YouTubeSettings
from ytpipe.providers.llm import ChatLLM, LLMError, extract_json
from ytpipe.providers.tts import SpeechProvider, estimate_speech_seconds
from ytpipe.providers.youtube import YouTubeUploader, _mime, _parse_range


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})
        self.content = self.text.encode()
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class FakeClient:
    """Stands in for httpx.Client so retries and parsing are testable offline."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("no scripted response left")
        out = self.responses.pop(0)
        return out() if callable(out) else out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestJsonRecovery:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('Here is the JSON:\n{"a": 1}\nHope that helps.', {"a": 1}),
            ('{"a": 1,}', {"a": 1}),
            ('{a: 1}', {"a": 1}),
            ('{"a": True, "b": None}//note', {"a": True, "b": None}),
            ('{"lines": [{"text": "hi"}', {"lines": [{"text": "hi"}]}),
            ('{"a": 1}\n\n{"b": 2}', {"a": 1}),
        ],
    )
    def test_recovers_common_model_sloppiness(self, raw, expected):
        assert extract_json(raw) == expected

    def test_array_payload(self):
        assert extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_prose_without_json_raises(self):
        with pytest.raises(LLMError):
            extract_json("I cannot help with that.")

    def test_truncated_json_is_repaired_or_raises(self):
        assert extract_json('{"a": [1, 2') == {"a": [1, 2]}


class TestChatLLM:
    def test_disabled_without_key(self):
        assert not ChatLLM(LLMSettings(api_key="")).available

    def test_request_shape(self):
        client = FakeClient([FakeResponse(payload={"choices": [{"message": {"content": '{"ok": true}'}}]})])
        llm = ChatLLM(LLMSettings(api_key="sk-test", model="m1"), client=client)
        assert llm.complete_json("sys", "usr") == {"ok": True}
        sent = client.calls[0]
        assert sent["url"] == "/chat/completions"
        assert sent["json"]["model"] == "m1"
        assert sent["json"]["response_format"] == {"type": "json_object"}
        assert [m["role"] for m in sent["json"]["messages"]] == ["system", "user"]

    def test_rate_limit_is_retried_and_client_error_is_not(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        retryable = FakeClient([
            FakeResponse(status_code=429, text="slow down"),
            FakeResponse(payload={"choices": [{"message": {"content": '{"a":1}'}}]}),
        ])
        llm = ChatLLM(LLMSettings(api_key="k"), client=retryable)
        assert llm.complete_json("s", "u") == {"a": 1}
        assert len(retryable.calls) == 2

        fatal = FakeClient([FakeResponse(status_code=401, text="bad key")])
        llm2 = ChatLLM(LLMSettings(api_key="k"), client=fatal)
        with pytest.raises(LLMError):
            llm2.complete_json("s", "u")
        assert len(fatal.calls) == 1, "an auth failure must not be retried"

    def test_empty_completion_is_an_error(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        llm = ChatLLM(LLMSettings(api_key="k"), client=FakeClient([FakeResponse(payload={"choices": []})] * 3))
        with pytest.raises(LLMError):
            llm.complete("s", "u")


class TestTTS:
    def test_estimate_grows_with_length(self):
        short = estimate_speech_seconds("short line")
        long = estimate_speech_seconds(" ".join(["word"] * 40))
        assert 0 < short < long

    def test_pace_setting_shortens_the_estimate(self):
        slow = estimate_speech_seconds("one two three four", wpm=120)
        fast = estimate_speech_seconds("one two three four", wpm=240)
        assert fast < slow

    def test_backend_reported_from_config(self):
        assert SpeechProvider(TTSSettings(api_key="k")).backend == "openai-tts"

    def test_empty_text_is_rejected(self):
        provider = SpeechProvider(TTSSettings(api_key="k"))
        with pytest.raises(ValueError):
            provider.synthesize("   ", Path("/tmp/nope.mp3"))


class TestYouTubeUpload:
    def test_title_is_clamped_at_a_word_boundary(self):
        yt = YouTubeUploader(YouTubeSettings())
        title = " ".join(f"word{i}" for i in range(30))
        body = yt.build_body(title=title)
        assert len(body["snippet"]["title"]) <= 100
        assert not body["snippet"]["title"].endswith(("word2", "word24")) or "…" in body["snippet"]["title"]
        assert body["snippet"]["title"].endswith("…")

    def test_tag_budget_is_respected(self):
        yt = YouTubeUploader(YouTubeSettings())
        tags = [f"tag{i}" for i in range(200)]
        body = yt.build_body(title="t", tags=tags)
        assert sum(len(t) + 1 for t in body["snippet"]["tags"]) - 1 <= 500
        assert len(body["snippet"]["tags"]) < 200

    def test_description_is_clamped(self):
        body = YouTubeUploader(YouTubeSettings()).build_body(title="t", description="x" * 9000)
        assert len(body["snippet"]["description"]) == 5000

    def test_status_flags(self):
        yt = YouTubeUploader(YouTubeSettings(privacy="unlisted", made_for_kids=False, contains_synthetic_media=True))
        body = yt.build_body(title="t")
        assert body["status"]["privacyStatus"] == "unlisted"
        assert body["status"]["containsSyntheticMedia"] is True
        assert body["status"]["selfDeclaredMadeForKids"] is False

    def test_dry_run_manifest_records_the_request(self, tmp_path):
        media = tmp_path / "final.mp4"
        media.write_bytes(b"0" * 5000)
        body = YouTubeUploader(YouTubeSettings()).build_body(title="Hello", description="d", tags=["a"])
        out = YouTubeUploader(YouTubeSettings()).manifest(body, media, tmp_path / "publish.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["quota_units"] == 1600
        assert data["media"]["bytes"] == 5000
        assert data["media"]["mime"] == "video/mp4"
        assert data["body"]["snippet"]["title"] == "Hello"
        assert "uploadType=resumable" in data["endpoint"]

    def test_unconfigured_is_not_usable(self):
        yt = YouTubeUploader(YouTubeSettings())
        assert yt.configured is False
        with pytest.raises(RuntimeError, match="OAuth"):
            yt.access_token()

    def test_configured_requires_all_three(self):
        assert YouTubeUploader(YouTubeSettings(client_id="a", client_secret="b")).configured is False
        assert YouTubeUploader(
            YouTubeSettings(client_id="a", client_secret="b", refresh_token="c")
        ).configured is True

    def test_resumable_range_parsing(self):
        assert _parse_range("bytes=0-2097151", 99) == 2097152
        assert _parse_range("", 42) == 42
        assert _parse_range("bytes=0-", 7) == 7

    def test_mime_types(self):
        assert _mime(Path("a.mp4")) == "video/mp4"
        assert _mime(Path("a.jpg")) == "image/jpeg"
        assert _mime(Path("a.bin")) == "application/octet-stream"
