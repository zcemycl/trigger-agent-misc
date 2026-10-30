from __future__ import annotations

from pathlib import Path

import pytest

from database import Database


def test_create_and_update_task(tmp_path: Path) -> None:
    db = Database(tmp_path / "agent.db")
    task = db.create_task(
        issue_number=7,
        branch="agent/issue-7",
        workspace=str(tmp_path / "issue-7"),
        status="queued",
    )
    assert task.status == "queued"
    assert db.task_exists(7)

    updated = db.update_task(
        7,
        status="pr_open",
        pr_number=42,
        pr_url="https://example.com/pr/42",
    )
    assert updated.pr_number == 42
    assert updated.status == "pr_open"


def test_task_lock_is_exclusive(tmp_path: Path) -> None:
    db = Database(tmp_path / "agent.db")
    assert db.try_acquire_lock(1, "a") is True
    assert db.try_acquire_lock(1, "b") is False
    db.release_lock(1, "a")
    assert db.try_acquire_lock(1, "b") is True


def test_comment_event_dedup_and_retryable_failed(tmp_path: Path) -> None:
    db = Database(tmp_path / "agent.db")
    assert db.insert_comment_event(
        comment_id="c1",
        pr_number=9,
        comment_type="issue_comment",
        author="alice",
        body="please fix",
    )
    assert (
        db.insert_comment_event(
            comment_id="c1",
            pr_number=9,
            comment_type="issue_comment",
            author="alice",
            body="please fix",
        )
        is False
    )
    db.update_comment_event("c1", status="failed", error="boom")
    events = db.list_comment_events(statuses=["failed"])
    assert len(events) == 1
    assert events[0].error == "boom"


def test_invalid_status_rejected(tmp_path: Path) -> None:
    db = Database(tmp_path / "agent.db")
    with pytest.raises(ValueError):
        db.create_task(
            issue_number=1,
            branch="b",
            workspace="w",
            status="nope",
        )
