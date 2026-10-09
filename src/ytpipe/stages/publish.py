"""Stage 8 — publish. Default is a dry run that writes a reviewable manifest.

Upload is the one stage that cannot be undone politely, so it stays behind two
switches: credentials must exist *and* `allow_publish`/`--publish` must be set.
Otherwise we emit exactly the JSON body and file we would send, which is what you
actually want to eyeball before burning 1600 quota units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import PublishError
from ..models import PublishResult, StageName
from ..providers.youtube import YouTubeUploader

DESC_FOOTER = "\n\n---\nCut automatically by ytpipe ({provider})"


class PublishStage:
    name = StageName.PUBLISH

    def run(self, ctx: Any) -> PublishResult:
        render = ctx.artifacts.render
        script = ctx.artifacts.script
        if render is None or not Path(render.path).exists():
            raise PublishError("nothing rendered yet; run the edit stage first")
        if script is None:
            raise PublishError("no script metadata to publish")

        title, description = self._metadata(ctx, script, render)
        tags = list(script.tags or [])
        if ctx.artifacts.brief is not None:
            for kw in ctx.artifacts.brief.keywords[:6]:
                if kw not in tags:
                    tags.append(kw)

        uploader = YouTubeUploader(ctx.settings.youtube)
        body = uploader.build_body(
            title=title,
            description=description,
            tags=tags,
            privacy=ctx.job.request.privacy or ctx.settings.youtube.privacy,
        )

        manifest = uploader.manifest(body, Path(render.path), ctx.output("publish.json"))
        wants_live = bool(ctx.job.request.publish) and not ctx.settings.youtube.dry_run
        if not wants_live or not uploader.configured:
            reason = (
                "no OAuth credentials configured"
                if not uploader.configured
                else "dry run (pass --publish with YTPIPE_YOUTUBE_DRY_RUN=0 to upload)"
            )
            ctx.emit(f"publish skipped: {reason}", level="warning")
            return PublishResult(
                status="dry-run",
                title=body["snippet"]["title"],
                privacy=body["status"]["privacyStatus"],
                manifest_path=manifest,
                detail=reason,
            )

        ctx.emit(f"uploading {Path(render.path).name} ({Path(render.path).stat().st_size / 1e6:.1f} MB)")
        try:
            outcome = uploader.upload(
                Path(render.path),
                body,
                progress=_progress_factory(ctx),
            )
        except Exception as exc:
            raise PublishError(f"upload failed: {exc}") from exc

        if outcome.video_id and render.thumbnail_path:
            if uploader.set_thumbnail(outcome.video_id, Path(render.thumbnail_path)):
                ctx.emit("custom thumbnail set")
            else:
                ctx.emit("thumbnail upload failed; YouTube will pick a frame", level="warning")

        ctx.emit(f"published https://youtu.be/{outcome.video_id}", level="info")
        return PublishResult(
            status="uploaded",
            video_id=outcome.video_id,
            url=outcome.etag_url,
            title=body["snippet"]["title"],
            privacy=body["status"]["privacyStatus"],
            manifest_path=manifest,
            uploaded_at=ctx.now(),
            detail="processing on YouTube for a few minutes before it is watchable",
        )

    # -- metadata --------------------------------------------------------------

    @staticmethod
    def _metadata(ctx: Any, script: Any, render: Any) -> tuple[str, str]:
        title = (ctx.job.request.title_hint or script.title or "Untitled").strip()
        lines = [ln.voiceover for ln in script.lines if ln.voiceover]
        beats = "\n".join(f"• {ln}" for ln in lines[1:5])
        description = script.description or (lines[0] if lines else "")
        description = (
            f"{description.strip()}\n\n{beats}\n"
            f"{DESC_FOOTER.format(provider=script.provider)}".rstrip()
        )
        if render.duration >= 60 * 3:
            ctx.emit("clip is 3 minutes or longer; it will not post as a Short", level="warning")
        return title[:100], description[:5000]


def _progress_factory(ctx: Any):
    state = {"last": 0}

    def progress(sent: int, total: int) -> None:
        pct = int(sent * 100 / max(1, total))
        if pct - state["last"] >= 25 or pct == 100:
            state["last"] = pct
            ctx.emit(f"upload {pct}%")

    return progress
