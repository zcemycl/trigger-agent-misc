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
    with pytest.raises(WorkspaceError, match="not usable"):
        mgr.ensure_worktree(9)


def test_commit_all_retries_when_pre_commit_rewrites(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    # Install a failing-then-pass hook simulator via PATH override is heavy;
    # exercise the rewrite detector + retry path with a fake _git instead.
    mgr = WorkspaceManager(_config(tmp_path, source))
    info = mgr.ensure_worktree(10)
    (info.path / "feature.txt").write_text("x\n", encoding="utf-8")

    calls: list[list[str]] = []
    real_git = mgr._git

    def fake_git(args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(args))
        if args[:1] == ["commit"]:
            commit_attempts = sum(1 for c in calls if c[:1] == ["commit"])
            if commit_attempts == 1:
                return subprocess.CompletedProcess(
                    args=["git", *args],
                    returncode=1,
                    stdout="",
                    stderr=(
                        "black................................................Failed\n"
                        "- hook id: black\n"
                        "- files were modified by this hook\n"
                        "reformatted feature.txt\n"
                    ),
                )
            return real_git(args, **kwargs)
        return real_git(args, **kwargs)

    mgr._git = fake_git  # type: ignore[method-assign]
    # After the fake failure, simulate hook rewrite so retry has something staged.
    (info.path / "feature.txt").write_text("x\n", encoding="utf-8")
    assert mgr.commit_all(info.path, "msg", max_hook_retries=1) is True
    assert sum(1 for c in calls if c[:1] == ["commit"]) >= 2


def test_pre_commit_rewrote_files_detector() -> None:
    assert WorkspaceManager._pre_commit_rewrote_files(
        "files were modified by this hook\nreformatted a.py\n"
    )
    assert not WorkspaceManager._pre_commit_rewrote_files("flake8 failed\nerror")


def _advance_main(source: Path, text: str) -> None:
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=source,
        check=True,
        capture_output=True,
    )
    readme = source / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + text, encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=source, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "advance main"],
        cwd=source,
        check=True,
        capture_output=True,
    )


def test_rebase_onto_base_when_parent_moves(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    mgr = WorkspaceManager(_config(tmp_path, source))
    info = mgr.ensure_worktree(1)
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    assert mgr.commit_all(info.path, "add feature") is True

    assert mgr.needs_rebase_onto_base(info.path) is False
    noop = mgr.rebase_onto_base(info.path)
    assert noop.already_up_to_date is True
    assert noop.changed is False

    _advance_main(source, "base change\n")
    mgr.fetch_origin()
    assert mgr.needs_rebase_onto_base(info.path) is True
    result = mgr.rebase_onto_base(info.path, fetch=False)
    assert result.changed is True
    assert mgr.needs_rebase_onto_base(info.path) is False
    assert "base change" in (info.path / "README.md").read_text(encoding="utf-8")
    assert (info.path / "feature.txt").exists()


def test_get_diff_omits_lockfiles_and_truncates(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    mgr = WorkspaceManager(_config(tmp_path, source))
    info = mgr.ensure_worktree(4)
    (info.path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (info.path / "uv.lock").write_text("x" * 5000 + "\n", encoding="utf-8")
    assert mgr.commit_all(info.path, "add files") is True

    diff = mgr.get_diff(info.path, max_chars=10_000)
    assert "app.py" in diff
    assert "Omitted large/binary paths" in diff
    assert "uv.lock" in diff
    # Full lockfile body should not be embedded.
    assert "x" * 5000 not in diff


def test_rebase_onto_base_aborts_on_conflict(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _init_repo(source)
    mgr = WorkspaceManager(_config(tmp_path, source))
    info = mgr.ensure_worktree(2)
    (info.path / "README.md").write_text("feature edit\n", encoding="utf-8")
    assert mgr.commit_all(info.path, "edit readme on branch") is True

    subprocess.run(
        ["git", "checkout", "main"],
        cwd=source,
        check=True,
        capture_output=True,
    )
    (source / "README.md").write_text("main edit\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=source, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "edit readme on main"],
        cwd=source,
        check=True,
        capture_output=True,
    )

    with pytest.raises(WorkspaceError, match="Rebase onto"):
        mgr.rebase_onto_base(info.path)

    # Worktree should not be left mid-rebase.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert " Ul " not in status.stdout  # not unmerged after abort
    head = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert head.stdout.strip() == "agent/issue-2"
