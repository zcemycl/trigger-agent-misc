from __future__ import annotations

from datetime import datetime, timedelta, timezone

from database import CommentEvent
from feedback import (
    batch_ready,
    format_feedback_block,
    is_trusted_human,
    normalize_issue_comment,
    normalize_review,
    normalize_review_comment,
)


def test_is_trusted_human_filters_bots_and_agent() -> None:
    trusted = frozenset({"alice"})
    assert is_trusted_human(
        "alice", trusted_users=trusted, agent_login="agent-bot"
    )
    assert not is_trusted_human(
        "alice", trusted_users=trusted, agent_login="alice"
    )
    assert not is_trusted_human(
        "dependabot[bot]",
        trusted_users=frozenset({"dependabot[bot]"}),
        agent_login="agent",
        user_payload={"type": "Bot"},
    )
    assert not is_trusted_human(
        "eve", trusted_users=trusted, agent_login="agent"
    )
    assert not is_trusted_human(
        "alice", trusted_users=frozenset(), agent_login="agent"
    )


def test_normalize_comments() -> None:
    issue = normalize_issue_comment(
        7,
        {"id": 11, "body": "please fix", "user": {"login": "alice"}},
    )
    assert issue.comment_id == "issue_comment:11"
    assert issue.comment_type == "issue_comment"

    review_comment = normalize_review_comment(
        7,
        {
            "id": 22,
            "body": "nits",
            "user": {"login": "alice"},
            "path": "a.py",
            "line": 10,
            "diff_hunk": "@@ -1 +1 @@",
        },
    )
    assert review_comment.path == "a.py"
    assert review_comment.line == 10

    empty_review = normalize_review(
        7, {"id": 33, "body": "  ", "user": {"login": "alice"}, "state": "APPROVED"}
    )
    assert empty_review is None

    review = normalize_review(
        7,
        {
            "id": 34,
            "body": "needs work",
            "user": {"login": "alice"},
            "state": "CHANGES_REQUESTED",
        },
    )
    assert review is not None
    assert review.comment_type == "review:changes_requested"


def test_batch_ready_waits_for_window() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    event = CommentEvent(
        comment_id="c1",
        pr_number=1,
        comment_type="issue_comment",
        author="alice",
        body="fix",
        status="pending",
        created_at=(now - timedelta(seconds=5)).isoformat(),
        processed_at=None,
        error=None,
    )
    assert batch_ready([event], 15, now=now) is False
    assert batch_ready([event], 15, now=now + timedelta(seconds=20)) is True
    assert batch_ready([event], 0, now=now) is True


def test_batch_ready_extends_when_newer_comment_arrives() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    older = CommentEvent(
        comment_id="c1",
        pr_number=1,
        comment_type="issue_comment",
        author="alice",
        body="first",
        status="pending",
        created_at=(now - timedelta(seconds=20)).isoformat(),
        processed_at=None,
        error=None,
    )
    newer = CommentEvent(
        comment_id="c2",
        pr_number=1,
        comment_type="issue_comment",
        author="alice",
        body="second",
        status="pending",
        created_at=(now - timedelta(seconds=5)).isoformat(),
        processed_at=None,
        error=None,
    )
    assert batch_ready([older, newer], 15, now=now) is False
    assert batch_ready([older, newer], 15, now=now + timedelta(seconds=10)) is True


def test_batch_ready_failed_retry_uses_processed_at() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    event = CommentEvent(
        comment_id="c1",
        pr_number=1,
        comment_type="issue_comment",
        author="alice",
        body="fix",
        status="failed",
        created_at=(now - timedelta(hours=1)).isoformat(),
        processed_at=(now - timedelta(seconds=5)).isoformat(),
        error="boom",
    )
    assert batch_ready([event], 15, now=now) is False
    assert batch_ready([event], 15, now=now + timedelta(seconds=20)) is True


def test_format_feedback_includes_inline_location() -> None:
    event = CommentEvent(
        comment_id="review_comment:1",
        pr_number=3,
        comment_type="review_comment",
        author="alice",
        body="rename this",
        status="pending",
        created_at="2026-01-01T00:00:00+00:00",
        processed_at=None,
        error=None,
        path="src/app.py",
        line=42,
        diff_hunk="- old\n+ new",
    )
    text = format_feedback_block([event])
    assert "src/app.py:42" in text
    assert "rename this" in text
    assert "- old" in text
