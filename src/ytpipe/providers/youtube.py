"""YouTube Data API v3 uploader (resumable) + OAuth refresh.

Deliberately no `google-api-python-client` dependency: the whole upload flow is
one token POST plus one resumable PUT session, and skipping the client library
keeps the install small and the retry behaviour visible.

Operations note: ``videos.insert`` costs **1600 quota units** of the default
10 000/day, i.e. roughly six uploads a day per project until the quota is raised.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..config import YouTubeSettings
from ..models import utcnow

log = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
THUMB_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
SCOPES = "https://www.googleapis.com/auth/youtube.upload"

#: Resumable chunks must be a multiple of 256 KiB.
CHUNK = 8 * 256 * 1024  # 2 MiB
TITLE_MAX = 100
DESC_MAX = 5000
TAGS_MAX = 500

Progress = Callable[[int, int], None]


@dataclass
class UploadOutcome:
    video_id: str | None
    etag_url: str | None
    raw: dict[str, Any]
    dry_run: bool = False


class YouTubeUploader:
    def __init__(self, settings: YouTubeSettings, *, timeout: float = 600.0) -> None:
        self.settings = settings
        self.timeout = timeout
        self._token: tuple[str, float] | None = None

    # -- auth -----------------------------------------------------------------

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.client_id and s.client_secret and s.refresh_token)

    def access_token(self) -> str:
        """Refresh-token grant, cached in-process until ~60s before expiry."""
        now = time.time()
        if self._token and self._token[1] > now + 60:
            return self._token[0]
        if not self.configured:
            raise RuntimeError(
                "YouTube OAuth not configured: set YTPIPE_YOUTUBE_CLIENT_ID, "
                "_CLIENT_SECRET and _REFRESH_TOKEN"
            )
        resp = httpx.post(
            TOKEN_URL,
            data={
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret,
                "refresh_token": self.settings.refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"token refresh failed HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        token = data["access_token"]
        self._token = (token, now + float(data.get("expires_in", 3600)))
        return token

    # -- metadata -------------------------------------------------------------

    def build_body(
        self,
        *,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        privacy: str | None = None,
        category_id: str | None = None,
        made_for_kids: bool | None = None,
        publish_at: str | None = None,
    ) -> dict[str, Any]:
        """Clamp to the API's documented limits rather than letting the 400 bite later."""
        title = (title or "Untitled").strip()
        if len(title) > TITLE_MAX:
            cut = title[: TITLE_MAX - 1]
            title = cut[: cut.rfind(" ")].rstrip(",;:-") + "…"
        tags = [t.strip() for t in (tags or []) if t.strip()]
        budget = TAGS_MAX
        trimmed: list[str] = []
        for tag in tags:
            cost = len(tag) + (1 if trimmed else 0)
            if budget - cost < 0:
                break
            budget -= cost
            trimmed.append(tag)
        snippet: dict[str, Any] = {
            "title": title,
            "description": (description or "").strip()[:DESC_MAX],
            "categoryId": category_id or self.settings.category_id,
            "tags": trimmed,
        }
        status: dict[str, Any] = {
            "privacyStatus": privacy or self.settings.privacy,
            "selfDeclaredMadeForKids": (
                self.settings.made_for_kids if made_for_kids is None else made_for_kids
            ),
            "embeddable": True,
        }
        if publish_at:
            status["publishAt"] = publish_at
        if getattr(self.settings, "contains_synthetic_media", False):
            # AI-disclosure flag for realistic altered/synthetic content.
            status["containsSyntheticMedia"] = True
        return {"snippet": snippet, "status": status}

    # -- upload ---------------------------------------------------------------

    def upload(
        self,
        media: Path,
        body: dict[str, Any],
        *,
        progress: Progress | None = None,
    ) -> UploadOutcome:
        media = Path(media)
        size = media.stat().st_size
        headers = {
            "Authorization": f"Bearer {self.access_token()}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": _mime(media),
            "X-Upload-Content-Length": str(size),
        }
        session_uri = self._start_session(body, headers)

        sent = 0
        offset = 0
        attempts = 0
        with media.open("rb") as fh:
            while offset < size:
                fh.seek(offset)
                chunk = fh.read(CHUNK)
                end = offset + len(chunk) - 1
                last = end >= size - 1
                headers = {
                    "Authorization": f"Bearer {self.access_token()}",
                    "Content-Range": f"bytes {offset}-{end}/{size}",
                    "Content-Length": str(len(chunk)),
                }
                if last:
                    headers["Content-Type"] = _mime(media)
                while True:
                    attempts += 1
                    resp = httpx.put(session_uri, content=chunk, headers=headers, timeout=self.timeout)
                    if resp.status_code in (200, 201):
                        data = resp.json() if resp.content else {}
                        if progress:
                            progress(size, size)
                        return UploadOutcome(
                            video_id=data.get("id"),
                            etag_url=f"https://youtu.be/{data.get('id')}" if data.get("id") else None,
                            raw=data,
                        )
                    if resp.status_code == 308:
                        range_header = resp.headers.get("Range", "")
                        next_offset = _parse_range(range_header, end + 1)
                        if next_offset != offset:  # server already has more; skip forward
                            offset = next_offset
                            break
                        if attempts % 4 == 0:
                            time.sleep(min(30, 2 ** (attempts // 4)))
                        time.sleep(0.4)
                        continue
                    if resp.status_code == 404:  # session expired: restart once, resend chunk
                        log.warning("upload session expired; restarting at offset %s", offset)
                        session_uri = self._start_session(body, headers)
                        attempts = 0
                        continue
                    if resp.status_code in (429, 500, 502, 503, 504) and attempts < 12:
                        delay = min(60.0, 1.5 * (2 ** (attempts % 5)))
                        log.warning("upload HTTP %s; backing off %.1fs", resp.status_code, delay)
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"upload chunk at {offset} failed HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                sent = end + 1
                offset = sent
                if progress:
                    progress(sent, size)
        raise RuntimeError("upload loop ended without a terminal response")

    def _start_session(self, body: dict[str, Any], headers: dict[str, str]) -> str:
        query = {"uploadType": "resumable", "part": "snippet,status"}
        resp = httpx.post(
            UPLOAD_URL,
            params=query,
            headers={**headers, "Content-Length": str(len(json.dumps(body).encode()))},
            json=body,
            timeout=60.0,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"resumable start HTTP {resp.status_code}: {resp.text[:250]}")
        location = resp.headers.get("location") or resp.headers.get("X-Upload-Status")
        if not location:
            raise RuntimeError(f"no upload session URI in response headers: {dict(resp.headers)}")
        return location

    def set_thumbnail(self, video_id: str, image: Path) -> bool:
        image = Path(image)
        if not image.exists():
            return False
        with httpx.Client(timeout=self.timeout) as client:
            token = self.access_token()
            resp = client.post(
                THUMB_URL,
                params={"videoId": video_id},
                headers={"Authorization": f"Bearer {token}"},
                files={"media": (image.name, image.read_bytes(), _mime(image))},
            )
        if resp.status_code >= 400:
            log.warning("thumbnail set failed HTTP %s: %s", resp.status_code, resp.text[:200])
            return False
        return True

    def manifest(self, body: dict[str, Any], media: Path, out: Path) -> Path:
        """What we *would* send. Written in dry-run mode so the request is reviewable."""
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "endpoint": f"{UPLOAD_URL}?uploadType=resumable&part=snippet,status",
            "generated_at": utcnow().isoformat(),
            "media": {
                "path": str(media),
                "bytes": media.stat().st_size if media.exists() else 0,
                "mime": _mime(media),
            },
            "quota_units": 1600,
            "body": body,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


def _mime(path: Path) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(path.suffix.lower(), "application/octet-stream")


def _parse_range(header: str, default: int) -> int:
    """`Range: bytes=0-2097151` -> next byte to send."""
    if "=" in header:
        spec = header.split("=", 1)[1]
        if "-" in spec:
            try:
                return int(spec.rsplit("-", 1)[1]) + 1
            except ValueError:
                return default
    return default
