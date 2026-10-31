from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent_runner import CommandResult
from config import Config
from database import Database
from task_manager import TaskManager
from workspace import RebaseResult


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
        test_command="true",
        trusted_github_users=frozenset(),
        database_path=tmp_path / "agent.db",
        log_dir=tmp_path / "logs",
        feedback_batch_seconds=15,
        max_agent_retries=2,
    )


def test_rebase_branch_invokes_agent_on_conflict(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    cfg = _config(tmp_path)
    db = Database(cfg.database_path)
    workspaces = MagicMock()
    agent = MagicMock()
    github = MagicMock()

    workspaces.rebase_onto_base.return_value = RebaseResult(
        changed=False,
        already_up_to_date=False,
        base_ref="origin/main",
        conflict_in_progress=True,
        detail="CONFLICT in app.py",
    )
    # Loop: in progress -> resolve -> stage -> continue -> done
    workspaces.is_rebase_in_progress.side_effect = [True, False, False]
    workspaces.conflicted_files.side_effect = [["app.py"], []]
    workspaces.continue_rebase.return_value = MagicMock(
        returncode=0, stdout="", stderr=""
    )
    agent.build_rebase_conflict_prompt.return_value = "resolve please"
    agent.run_agent.return_value = CommandResult(
        command=["agent"],
        exit_code=0,
        stdout="ok",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
    )

    mgr = TaskManager(cfg, db, github, workspaces, agent)
    assert (
        mgr._rebase_branch(
            tmp_path,
            "agent/issue-3",
            push=False,
            issue_number=3,
        )
        is True
    )
    agent.run_agent.assert_called_once()
    workspaces.stage_paths.assert_called()
    workspaces.continue_rebase.assert_called()
    workspaces.abort_rebase.assert_not_called()


def test_conflict_prompt_lists_files(tmp_path: Path) -> None:
    from agent_runner import AgentRunner

    runner = AgentRunner(_config(tmp_path))
    prompt = runner.build_rebase_conflict_prompt(
        issue_number=3,
        base_branch="origin/main",
        conflicted_files=["src/a.py", "README.md"],
        rebase_detail="could not apply abc",
    )
    assert "src/a.py" in prompt
    assert "README.md" in prompt
    assert "Do NOT run `git rebase --continue`" in prompt
