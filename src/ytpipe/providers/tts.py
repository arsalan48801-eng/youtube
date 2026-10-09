"""Text-to-speech.

`synthesize` writes audio and returns nothing about its length on purpose -- the
generate stage measures real duration with ffmpeg and *then* rewrites shot timing,
because a narration that runs long is an editing decision, not a TTS one.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import httpx

from ..config import TTSSettings

log = logging.getLogger(__name__)

#: Average narration speed for short-form voiceover, words per minute.
DEFAULT_WPM = 150.0


def estimate_speech_seconds(text: str, wpm: float = DEFAULT_WPM, *, padding: float = 0.35) -> float:
    """Timing prior used before/without audio: syllable-ish length + breath room."""
    words = max(1, len(text.split()))
    minutes = words / max(60.0, wpm)
    return round(minutes * 60.0 + padding, 3)


class SpeechProvider:
    def __init__(self, settings: TTSSettings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return bool(self.settings.api_key) or importlib.util.find_spec("edge_tts") is not None

    @property
    def backend(self) -> str:
        return "openai-tts" if self.settings.api_key else "edge-tts"

    def synthesize(self, text: str, out: Path, *, speed: float | None = None) -> Path:
        """Render `text` to `out`. Raises on provider failure; caller decides fallback."""
        text = " ".join(text.split())
        if not text:
            raise ValueError("cannot synthesize empty text")
        out.parent.mkdir(parents=True, exist_ok=True)
        rate = self.settings.speed if speed is None else speed
        if self.settings.api_key:
            return self._openai(text, out, rate)
        if importlib.util.find_spec("edge_tts"):
            return self._edge(text, out, rate)
        raise RuntimeError("no TTS backend configured (set YTPIPE_TTS_API_KEY or pip install edge-tts)")

    def _openai(self, text: str, out: Path, rate: float) -> Path:
        payload = {
            "model": self.settings.model,
            "voice": self.settings.voice,
            "input": text,
            "response_format": "mp3",
            "speed": max(0.5, min(2.0, rate)),
        }
        with httpx.Client(
            base_url=self.settings.base_url.rstrip("/"),
            timeout=self.settings.timeout_s,
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
        ) as client:
            resp = client.post("/audio/speech", json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"tts HTTP {resp.status_code}: {resp.text[:200]}")
            out.write_bytes(resp.content)
        return out

    async def _edge_async(self, text: str, out: Path, rate: float) -> None:  # pragma: no cover
        edge_tts = __import__("edge_tts")
        communicate = edge_tts.Communicate(
            text, self.settings.voice, rate=f"{int((rate - 1) * 100):+d}%"
        )
        await communicate.save(str(out))

    def _edge(self, text: str, out: Path, rate: float) -> Path:  # pragma: no cover - needs network
        import asyncio

        edge_tts = __import__("edge_tts")
        communicate = edge_tts.Communicate(
            text, self.settings.voice.replace("nova", "en-US-GuyNeural"), rate=f"{int((rate - 1) * 100):+d}%"
        )
        asyncio.run(communicate.save(str(out)))
        return out
