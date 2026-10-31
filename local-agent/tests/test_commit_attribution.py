from __future__ import annotations

import subprocess
from pathlib import Path

from config import Config
from workspace import WorkspaceManager


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
        commit_message_prefix="agent:",
        git_author_name="local-agent",
        git_author_email="local-agent[bot]@users.noreply.github.com",
        commit_co_authored_by="local-agent <local-agent[bot]@users.noreply.github.com>",
    )


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "human@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Human"],
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
    subprocess.run(
        ["git", "remote", "add", "origin", str(path)],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "fetch", "origin"], cwd=path, check=True, capture_output=True)


def test_format_commit_message_prefix_and_trailer() -> None:
    cfg = Config(
        github_repo="acme/demo",
        owner="acme",
        repo="demo",
        base_branch="main",
        source_repo=Path("/tmp"),
        workspace_root=Path("/tmp"),
        poll_interval=30,
        max_concurrent_tasks=1,
        agent_command="agent",
        agent_timeout=60,
        test_command="true",
        trusted_github_users=frozenset(),
        database_path=Path("/tmp/db"),
        log_dir=Path("/tmp/logs"),
        feedback_batch_seconds=15,
        max_agent_retries=2,
    )
    msg = cfg.format_commit_message("Implement issue #3")
    assert msg.startswith("agent: Implement issue #3")
    assert "Co-authored-by: local-agent <local-agent[bot]@users.noreply.github.com>" in msg
    # Do not double-prefix.
    assert cfg.format_commit_message("agent: already").startswith("agent: already")


def test_commit_uses_agent_author(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    cfg = _config(tmp_path, source)
    mgr = WorkspaceManager(cfg)
    info = mgr.ensure_worktree(1)
    (info.path / "feat.txt").write_text("x\n", encoding="utf-8")
    message = cfg.format_commit_message("Implement issue #1")
    assert mgr.commit_all(info.path, message) is True

    show = subprocess.run(
        ["git", "log", "-1", "--format=%an%n%ae%n%B"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    body = show.stdout
    assert "local-agent" in body.splitlines()[0]
    assert "local-agent[bot]@users.noreply.github.com" in body.splitlines()[1]
    assert "agent: Implement issue #1" in body
    assert "Co-authored-by: local-agent" in body
