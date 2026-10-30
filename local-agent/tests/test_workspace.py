from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from config import Config
from workspace import WorkspaceError, WorkspaceManager


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    # Fake origin as the repo itself so fetch works offline.
    subprocess.run(
        ["git", "remote", "add", "origin", str(path)],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "origin"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _config(tmp_path: Path, source: Path) -> Config:
    return Config(
        github_repo="acme/demo",
        owner="acme",
        repo="demo",
        base_branch="main",
        source_repo=source,
        workspace_root=tmp_path / "workspaces",
        poll_interval=30,
        max_concurrent_tasks=1,
        agent_command="agent",
        agent_timeout=60,
        test_command="true",
        trusted_github_users=frozenset(),
        database_path=tmp_path / "agent.db",
        log_dir=tmp_path / "logs",
        feedback_batch_seconds=15,
        max_agent_retries=2,
    )


def test_ensure_worktree_creates_branch(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    mgr = WorkspaceManager(_config(tmp_path, source))
    info = mgr.ensure_worktree(3)
    assert info.branch == "agent/issue-3"
    assert info.path.exists()
    assert (info.path / "README.md").exists()
    head = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert head.stdout.strip() == "agent/issue-3"


def test_ensure_worktree_reuses_existing(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    mgr = WorkspaceManager(_config(tmp_path, source))
    first = mgr.ensure_worktree(5)
    second = mgr.ensure_worktree(5)
    assert first.path == second.path


def test_commit_all_and_refuse_path_collision(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    mgr = WorkspaceManager(_config(tmp_path, source))
    info = mgr.ensure_worktree(8)
    (info.path / "feature.txt").write_text("x\n", encoding="utf-8")
    assert mgr.commit_all(info.path, "Implement issue #8") is True

    # Create a colliding directory without registering a worktree.
    collision = tmp_path / "workspaces" / "issue-9"
    collision.mkdir(parents=True)
    with pytest.raises(WorkspaceError, match="already exists"):
        mgr.ensure_worktree(9)
