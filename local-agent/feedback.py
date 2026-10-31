"""PR feedback collection helpers: filtering, formatting, batch readiness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    """
    Only trusted humans may trigger feedback runs.

    If the controller's gh login is also listed in TRUSTED_GITHUB_USERS (common
    for solo setups), that account's comments are allowed. Untrusted accounts
    that match the agent login are still ignored.
    """
    if not trusted_users:
        return False
    name = (login or "").strip().lstrip("@").lower()
    if not name:
        return False
    if is_bot_author(name, user_payload):
        return False
    # Trusted list is authoritative, including solo setups where `agent_login`
    # is the same GitHub user as the human reviewer.
    return name in trusted_users  # agent_login intentionally not a blocker here


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


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        cleaned = " ".join(line.split())
        if cleaned:
            return cleaned
    return ""


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: max(0, limit - 1)].rstrip() + "…"


def summarize_feedback_commit(
    events: list[CommentEvent],
    *,
    issue_number: int,
    pr_number: int,
    subject_limit: int = 64,
) -> tuple[str, str]:
    """
    Build a descriptive commit subject + body from batched feedback.

    Returns (subject_without_prefix, body). Subject is capped so that with the
    default `agent:` prefix the full first line stays near ~72 characters.
    """
    if not events:
        return (f"Address PR feedback for issue #{issue_number}", "")

    primary = events[0]
    snippet = _first_line(primary.body) or "feedback"
    where = ""
    if primary.path:
        where = Path(primary.path).name
        if primary.line is not None:
            where = f"{where}:{primary.line}"

    if where:
        subject = f"#{issue_number} feedback ({where}): {snippet}"
    elif len(events) == 1:
        subject = f"#{issue_number} feedback: {snippet}"
    else:
        subject = f"#{issue_number} feedback ({len(events)} notes): {snippet}"
    subject = _truncate(subject, subject_limit)

    body_lines = [
        f"Respond to trusted review on PR #{pr_number} for issue #{issue_number}.",
        "",
    ]
    for event in events:
        loc = event.path or ""
        if event.path and event.line is not None:
            loc = f"{event.path}:{event.line}"
        loc_bit = f" ({loc})" if loc else ""
        body_lines.append(
            f"- @{event.author}{loc_bit}: {_truncate(_first_line(event.body) or event.comment_type, 120)}"
        )
    return subject, "\n".join(body_lines).rstrip() + "\n"
