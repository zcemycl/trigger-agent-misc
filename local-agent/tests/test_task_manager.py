from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent_runner import CommandResult
from config import Config
from database import Database
from github_client import Issue, PullRequest
from task_manager import TaskManager
from workspace import WorktreeInfo


def _config(tmp_path: Path) -> Config:
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
        test_command="pytest -q",
        trusted_github_users=frozenset(),
        database_path=tmp_path / "agent.db",
        log_dir=tmp_path / "logs",
        feedback_batch_seconds=15,
        max_agent_retries=2,
    )


def _issue(number: int = 11) -> Issue:
    return Issue(
        number=number,
        title="Add feature",
        body="Please implement X",
        labels=["agent:ready", "enhancement"],
        author="alice",
        state="open",
        url="https://example.com/issues/11",
    )


def _ok_result() -> CommandResult:
    return CommandResult(
        command=["agent"],
        exit_code=0,
        stdout="done",
        stderr="",
        timed_out=False,
        duration_seconds=1.0,
        log_path=None,
    )


def _fail_result() -> CommandResult:
    return CommandResult(
        command=["pytest"],
        exit_code=1,
        stdout="",
        stderr="fail",
        timed_out=False,
        duration_seconds=0.5,
        log_path=Path("/tmp/fail.log"),
    )


def test_handle_new_issue_success_creates_pr(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    db = Database(cfg.database_path)
    github = MagicMock()
    workspaces = MagicMock()
    agent = MagicMock()

    wt = WorktreeInfo(issue_number=11, branch="agent/issue-11", path=tmp_path / "wt")
    workspaces.branch_name.return_value = "agent/issue-11"
    workspaces.worktree_path.return_value = tmp_path / "wt"
    workspaces.ensure_worktree.return_value = wt
    workspaces.commit_all.return_value = True
    workspaces.rebase_onto_base.return_value = MagicMock(
        changed=False, already_up_to_date=True, base_ref="origin/main"
    )

    agent.build_implementation_prompt.return_value = "prompt"
    agent.run_agent.return_value = _ok_result()
    agent.run_tests.return_value = _ok_result()

    github.create_pull_request.return_value = PullRequest(
        number=99,
        title="Add feature",
        body="Automated implementation. Closes #11",
        state="open",
        url="https://example.com/pull/99",
        merged=False,
        head_ref="agent/issue-11",
        base_ref="main",
    )
    github.get_issue.return_value = Issue(
        number=11,
        title="Add feature",
        body="Please implement X",
        labels=["agent:running", "enhancement"],
        author="alice",
        state="open",
        url="https://example.com/issues/11",
    )

    mgr = TaskManager(cfg, db, github, workspaces, agent)
    mgr._handle_new_issue(_issue())

    task = db.get_task(11)
    assert task is not None
    assert task.status == "review_pending"
    assert task.pr_number == 99
    github.create_pull_request.assert_called_once()
    workspaces.push_branch.assert_called_once()
    # ready -> running, then running -> review
    assert github.replace_agent_label.call_count >= 2


def test_handle_new_issue_test_failure_no_pr(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    db = Database(cfg.database_path)
    github = MagicMock()
    workspaces = MagicMock()
    agent = MagicMock()

    wt = WorktreeInfo(issue_number=12, branch="agent/issue-12", path=tmp_path / "wt")
    workspaces.branch_name.return_value = "agent/issue-12"
    workspaces.worktree_path.return_value = tmp_path / "wt"
    workspaces.ensure_worktree.return_value = wt

    agent.build_implementation_prompt.return_value = "prompt"
    agent.run_agent.return_value = _ok_result()
    agent.run_tests.return_value = _fail_result()

    mgr = TaskManager(cfg, db, github, workspaces, agent)
    mgr._handle_new_issue(_issue(12))

    task = db.get_task(12)
    assert task is not None
    assert task.status == "failed"
    github.create_pull_request.assert_not_called()
    workspaces.push_branch.assert_not_called()


def test_process_ready_issues_skips_existing_task(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    db = Database(cfg.database_path)
    db.create_task(
        issue_number=11,
        branch="agent/issue-11",
        workspace=str(tmp_path / "wt"),
        status="review_pending",
    )
    github = MagicMock()
    github.list_ready_issues.return_value = [_issue(11)]
    mgr = TaskManager(cfg, db, github, MagicMock(), MagicMock())
    mgr.process_ready_issues()
    # Should clear ready label but not enqueue another worker run that creates duplicate.
    github.replace_agent_label.assert_called()
    assert db.get_task(11).status == "review_pending"
