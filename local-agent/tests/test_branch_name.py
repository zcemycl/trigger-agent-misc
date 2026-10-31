from __future__ import annotations

from pathlib import Path

from branch_name import (
    BRANCH_NAME_FILENAME,
    consume_agent_branch_name,
    sanitize_branch_name,
)


def test_sanitize_branch_name() -> None:
    assert (
        sanitize_branch_name("Add Latex Math Docs", issue_number=3)
        == "agent/add-latex-math-docs-3"
    )
    assert (
        sanitize_branch_name("agent/fix-flake8", issue_number=3)
        == "agent/fix-flake8-3"
    )
    assert (
        sanitize_branch_name("issue-3-static-readings", issue_number=3)
        == "agent/issue-3-static-readings"
    )
    assert sanitize_branch_name("@@@", issue_number=3) is None


def test_consume_agent_branch_name(tmp_path: Path) -> None:
    path = tmp_path / BRANCH_NAME_FILENAME
    path.write_text("Improve line search notebook\n", encoding="utf-8")
    assert (
        consume_agent_branch_name(tmp_path, issue_number=5)
        == "agent/improve-line-search-notebook-5"
    )
    assert not path.exists()


def test_consume_missing(tmp_path: Path) -> None:
    assert consume_agent_branch_name(tmp_path, issue_number=1) is None
