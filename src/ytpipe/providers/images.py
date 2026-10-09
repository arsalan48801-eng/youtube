"""Still-image provider for b-roll, plus the procedural fallback.

The fallback is not a stub: an animated gradient + title card keeps the render
timeline honest so you can review pacing before spending money on generation.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from ..config import ImageSettings

log = logging.getLogger(__name__)


class ImageProvider:
    def __init__(self, settings: ImageSettings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return bool(self.settings.api_key)

    def generate(self, prompt: str, out: Path, *, size: str | None = None) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.settings.model,
            "prompt": prompt.strip(),
            "n": 1,
            "size": size or self.settings.size,
        }
        with httpx.Client(
            base_url=self.settings.base_url.rstrip("/"),
            timeout=300.0,
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
        ) as client:
            resp = client.post("/images/generations", json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"images HTTP {resp.status_code}: {resp.text[:200]}")
            item = (resp.json().get("data") or [{}])[0]
            if b64 := item.get("b64_json"):
                out.write_bytes(base64.b64decode(b64))
            elif url := item.get("url"):
                out.write_bytes(httpx.get(url, timeout=120.0).content)
            else:
                raise RuntimeError(f"image provider returned no payload: {str(item)[:200]}")
        return out
