"""PR feedback collection helpers: filtering, formatting, batch readiness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from database import CommentEvent


def parse_utc(ts: str) -> datetime:
    """Parse ISO timestamps from SQLite or GitHub."""
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_bot_author(login: str, user_payload: Optional[dict[str, Any]] = None) -> bool:
    name = (login or "").lower()
    if not name:
        return True
    if name.endswith("[bot]") or name.endswith("-bot"):
        return True
    if user_payload and str(user_payload.get("type", "")).lower() == "bot":
        return True
    return False


def is_trusted_human(
    login: str,
    *,
    trusted_users: frozenset[str],
    agent_login: str,
    user_payload: Optional[dict[str, Any]] = None,
) -> bool:
    """Only trusted humans may trigger feedback runs."""
    if not trusted_users:
        return False
    name = (login or "").strip().lstrip("@").lower()
    if not name:
        return False
    if name == agent_login.strip().lstrip("@").lower():
        return False
    if is_bot_author(name, user_payload):
        return False
    return name in trusted_users


@dataclass(frozen=True)
class IncomingComment:
    comment_id: str
    pr_number: int
    comment_type: str
    author: str
    body: str
    path: Optional[str] = None
    line: Optional[int] = None
    diff_hunk: Optional[str] = None


def normalize_issue_comment(pr_number: int, item: dict[str, Any]) -> IncomingComment:
    user = item.get("user") or {}
    return IncomingComment(
        comment_id=f"issue_comment:{item['id']}",
        pr_number=pr_number,
        comment_type="issue_comment",
        author=str(user.get("login") or ""),
        body=(item.get("body") or "").strip(),
    )


def normalize_review_comment(pr_number: int, item: dict[str, Any]) -> IncomingComment:
    user = item.get("user") or {}
    line = item.get("line")
    if line is None:
        line = item.get("original_line")
    return IncomingComment(
        comment_id=f"review_comment:{item['id']}",
        pr_number=pr_number,
        comment_type="review_comment",
        author=str(user.get("login") or ""),
        body=(item.get("body") or "").strip(),
        path=item.get("path"),
        line=int(line) if line is not None else None,
        diff_hunk=item.get("diff_hunk"),
    )


def normalize_review(pr_number: int, item: dict[str, Any]) -> Optional[IncomingComment]:
    """Submitted reviews — only those with a non-empty body are actionable."""
    body = (item.get("body") or "").strip()
    if not body:
        return None
    user = item.get("user") or {}
    state = (item.get("state") or "").upper()
    return IncomingComment(
        comment_id=f"review:{item['id']}",
        pr_number=pr_number,
        comment_type=f"review:{state.lower() or 'commented'}",
        author=str(user.get("login") or ""),
        body=body,
    )


def _event_anchor(event: CommentEvent) -> datetime:
    """Timestamp used for batching / retry backoff."""
    if event.status == "failed" and event.processed_at:
        return parse_utc(event.processed_at)
    return parse_utc(event.created_at)


def batch_ready(events: list[CommentEvent], batch_seconds: int, *, now: Optional[datetime] = None) -> bool:
    """
    True when events have been quiet long enough to start an agent run.

    Uses the newest anchor time so late-arriving comments extend the batch
    window, and failed retries wait FEEDBACK_BATCH_SECONDS after the last attempt.
    """
    if not events:
        return False
    if batch_seconds <= 0:
        return True
    clock = now or utc_now()
    newest = max(_event_anchor(e) for e in events)
    return (clock - newest).total_seconds() >= batch_seconds


def format_feedback_block(events: list[CommentEvent]) -> str:
    parts: list[str] = []
    for event in events:
        header = f"### {event.comment_type} by @{event.author} (id={event.comment_id})"
        location = ""
        if event.path:
            loc = event.path
            if event.line is not None:
                loc = f"{loc}:{event.line}"
            location = f"\nLocation: `{loc}`"
        hunk = ""
        if event.diff_hunk:
            hunk = f"\nCode context:\n```\n{event.diff_hunk}\n```"
        parts.append(f"{header}{location}\n\n{event.body}{hunk}")
    return "\n\n---\n\n".join(parts)
