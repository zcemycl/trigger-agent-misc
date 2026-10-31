from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from agent_runner import CommandResult
from config import Config
from database import Database
from github_client import Issue, PullRequest
from task_manager import TaskManager
from workspace import WorktreeInfo


def _config(tmp_path: Path, *, batch_seconds: int = 0) -> Config:
    return Config(
        github_repo="acme/demo",
        owner="acme",
        repo="demo",
        base_branch="main",
        source_repo=tmp_path,
        workspace_root=tmp_path / "workspaces",
        poll_interval=30,
        max_concurrent_tasks=1,
        agent_command="agent",
        agent_timeout=60,
        test_command="true",
        trusted_github_users=frozenset({"alice"}),
        database_path=tmp_path / "agent.db",
        log_dir=tmp_path / "logs",
        feedback_batch_seconds=batch_seconds,
        max_agent_retries=2,
    )


def _ok() -> CommandResult:
    return CommandResult(
        command=["agent"],
        exit_code=0,
        stdout="ok",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
    )


def test_ingest_and_apply_feedback_without_new_pr(tmp_path: Path) -> None:
    cfg = _config(tmp_path, batch_seconds=0)
    (cfg.log_dir).mkdir(parents=True, exist_ok=True)
    db = Database(cfg.database_path)
    db.create_task(
        issue_number=11,
        branch="agent/issue-11",
        workspace=str(tmp_path / "wt"),
        status="review_pending",
    )
    db.update_task(
        11,
        pr_number=99,
        pr_url="https://example.com/pull/99",
    )

    github = MagicMock()
    github.get_authenticated_user.return_value = "agent-user"
    github.get_pull_request.return_value = PullRequest(
        number=99,
        title="PR title",
        body="PR body",
        state="open",
        url="https://example.com/pull/99",
        merged=False,
        head_ref="agent/issue-11",
        base_ref="main",
    )
    github.get_issue.return_value = Issue(
        number=11,
        title="Issue title",
        body="Issue body",
        labels=["agent:review"],
        author="alice",
        state="open",
        url="https://example.com/issues/11",
    )
    github.list_issue_comments.return_value = [
        {
            "id": 1,
            "body": "Please fix the naming",
            "user": {"login": "alice", "type": "User"},
        },
        {
            "id": 2,
            "body": "ignore me",
            "user": {"login": "mallory", "type": "User"},
        },
        {
            "id": 3,
            "body": "bot noise",
            "user": {"login": "dependabot[bot]", "type": "Bot"},
        },
    ]
    github.list_review_comments.return_value = [
        {
            "id": 10,
            "body": "Inline note",
            "user": {"login": "alice", "type": "User"},
            "path": "app.py",
            "line": 3,
            "diff_hunk": "+x",
        }
    ]
    github.list_reviews.return_value = [
        {
            "id": 20,
            "body": "",
            "user": {"login": "alice", "type": "User"},
            "state": "COMMENTED",
        }
    ]

    workspaces = MagicMock()
    workspaces.ensure_worktree.return_value = WorktreeInfo(
        issue_number=11,
        branch="agent/issue-11",
        path=tmp_path / "wt",
    )
    workspaces.get_diff.return_value = "diff --git a/app.py"
    workspaces.commit_all.return_value = True
    workspaces.rebase_onto_base.return_value = MagicMock(
        changed=False,
        already_up_to_date=True,
        base_ref="origin/main",
        conflict_in_progress=False,
        detail="",
    )

    agent = MagicMock()
    agent.build_feedback_prompt.return_value = "feedback prompt"
    agent.run_agent.return_value = _ok()
    agent.run_tests.return_value = _ok()

    mgr = TaskManager(cfg, db, github, workspaces, agent)
    mgr.process_pr_comments()

    # Worker runs in thread pool — drain it.
    mgr.shutdown(wait=True)

    events = db.list_comment_events(pr_number=99)
    assert {e.comment_id for e in events} == {
        "issue_comment:1",
        "review_comment:10",
    }
    assert all(e.status == "completed" for e in events)
    github.create_pull_request.assert_not_called()
    workspaces.push_branch.assert_called_once()
    assert db.get_task(11).status == "review_pending"


def test_process_pr_comments_noop_without_trusted_users(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg = Config(**{**cfg.__dict__, "trusted_github_users": frozenset()})
    db = Database(cfg.database_path)
    github = MagicMock()
    mgr = TaskManager(cfg, db, github, MagicMock(), MagicMock())
    mgr.process_pr_comments()
    github.list_issue_comments.assert_not_called()


def test_batch_not_enqueued_before_window(tmp_path: Path) -> None:
    cfg = _config(tmp_path, batch_seconds=15)
    db = Database(cfg.database_path)
    db.create_task(
        issue_number=11,
        branch="agent/issue-11",
        workspace=str(tmp_path / "wt"),
        status="review_pending",
    )
    db.update_task(11, pr_number=99, pr_url="https://example.com/pull/99")
    # Insert a fresh pending event directly.
    db.insert_comment_event(
        comment_id="issue_comment:1",
        pr_number=99,
        comment_type="issue_comment",
        author="alice",
        body="fix please",
    )
    github = MagicMock()
    github.get_authenticated_user.return_value = "agent-user"
    github.get_pull_request.return_value = PullRequest(
        number=99,
        title="t",
        body="b",
        state="open",
        url="u",
        merged=False,
        head_ref="agent/issue-11",
        base_ref="main",
    )
    github.list_issue_comments.return_value = []
    github.list_review_comments.return_value = []
    github.list_reviews.return_value = []

    agent = MagicMock()
    mgr = TaskManager(cfg, db, github, MagicMock(), agent)
    mgr.process_pr_comments()
    mgr.shutdown(wait=True)
    agent.run_agent.assert_not_called()
    assert db.list_comment_events(statuses=["pending"])
