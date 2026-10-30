from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import Config
from github_client import GitHubClient


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


def test_list_ready_issues_skips_pull_requests(tmp_path: Path) -> None:
    client = GitHubClient(_config(tmp_path))
    payload = [
        {
            "number": 1,
            "title": "Real issue",
            "body": "body",
            "labels": [{"name": "agent:ready"}],
            "user": {"login": "alice"},
            "state": "open",
            "html_url": "https://example.com/issues/1",
        },
        {
            "number": 2,
            "title": "Actually a PR",
            "body": "",
            "labels": [{"name": "agent:ready"}],
            "user": {"login": "bot"},
            "state": "open",
            "html_url": "https://example.com/pull/2",
            "pull_request": {"url": "https://api.example.com/pulls/2"},
        },
    ]
    with patch.object(client, "_api", return_value=payload):
        issues = client.list_ready_issues()
    assert len(issues) == 1
    assert issues[0].number == 1


def test_create_pull_request_parses_url(tmp_path: Path) -> None:
    client = GitHubClient(_config(tmp_path))
    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = "https://github.com/acme/demo/pull/55\n"
    completed.stderr = ""
    with patch.object(client, "_run", return_value=completed):
        with patch.object(
            client,
            "_api",
            return_value=[
                {
                    "number": 55,
                    "title": "t",
                    "body": "b",
                    "state": "open",
                    "html_url": "https://github.com/acme/demo/pull/55",
                    "merged": False,
                    "head": {"ref": "agent/issue-1"},
                    "base": {"ref": "main"},
                }
            ],
        ):
            pr = client.create_pull_request(
                head="agent/issue-1",
                base="main",
                title="t",
                body="b",
            )
    assert pr.number == 55


def test_replace_agent_label_posts_json(tmp_path: Path) -> None:
    client = GitHubClient(_config(tmp_path))
    with patch("github_client.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        client.replace_agent_label(
            3,
            remove="agent:ready",
            add="agent:running",
            current_labels=["agent:ready", "bug"],
        )
        args, kwargs = run.call_args
        assert kwargs["input"]
        body = json.loads(kwargs["input"])
        assert "agent:running" in body["labels"]
        assert "agent:ready" not in body["labels"]
        assert "bug" in body["labels"]
