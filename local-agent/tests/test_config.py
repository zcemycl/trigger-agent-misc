from __future__ import annotations

import os
from pathlib import Path

import pytest

from config import ConfigError, load_config


def _write_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir()


def test_load_config_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "src"
    _write_git_repo(source)
    workspace = tmp_path / "workspaces"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "GITHUB_REPO=acme/demo",
                f"SOURCE_REPO={source}",
                f"WORKSPACE_ROOT={workspace}",
                "BASE_BRANCH=develop",
                "POLL_INTERVAL=15",
                "MAX_CONCURRENT_TASKS=2",
                "AGENT_COMMAND=git",
                "AGENT_TIMEOUT=120",
                "TEST_COMMAND=pytest -q",
                "TRUSTED_GITHUB_USERS=alice, @Bob",
                f"DATABASE_PATH={tmp_path / 'agent.db'}",
                f"LOG_DIR={tmp_path / 'logs'}",
            ]
        ),
        encoding="utf-8",
    )
    # Clear conflicting env.
    for key in list(os.environ):
        if key.startswith("GITHUB_") or key in {
            "SOURCE_REPO",
            "WORKSPACE_ROOT",
            "BASE_BRANCH",
            "POLL_INTERVAL",
            "AGENT_COMMAND",
        }:
            monkeypatch.delenv(key, raising=False)

    cfg = load_config(env_file)
    assert cfg.owner == "acme"
    assert cfg.repo == "demo"
    assert cfg.base_branch == "develop"
    assert cfg.poll_interval == 15
    assert cfg.max_concurrent_tasks == 2
    assert cfg.trusted_github_users == frozenset({"alice", "bob"})
    assert workspace.exists()


def test_load_config_rejects_bad_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "src"
    _write_git_repo(source)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "GITHUB_REPO=not-a-slug",
                f"SOURCE_REPO={source}",
                f"WORKSPACE_ROOT={tmp_path / 'ws'}",
                "AGENT_COMMAND=git",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    with pytest.raises(ConfigError, match="OWNER/REPOSITORY"):
        load_config(env_file)


def test_load_config_requires_source_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "GITHUB_REPO=acme/demo",
                f"SOURCE_REPO={tmp_path / 'missing'}",
                f"WORKSPACE_ROOT={tmp_path / 'ws'}",
                "AGENT_COMMAND=git",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("SOURCE_REPO", raising=False)
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(env_file)
