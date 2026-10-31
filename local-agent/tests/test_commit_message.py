from __future__ import annotations

from pathlib import Path

from commit_message import (
    COMMIT_MESSAGE_FILENAME,
    commit_message_instructions,
    consume_agent_commit_message,
    parse_commit_message_file,
)


def test_parse_commit_message_file() -> None:
    subject, body = parse_commit_message_file(
        "\nAdd latex equations to notebook\n\nExplain line search steps.\n"
    )
    assert subject == "Add latex equations to notebook"
    assert body == "Explain line search steps.\n"


def test_consume_agent_commit_message_reads_and_deletes(tmp_path: Path) -> None:
    path = tmp_path / COMMIT_MESSAGE_FILENAME
    path.write_text("Fix flake8 pre-commit config\n\nDetails here.\n", encoding="utf-8")
    result = consume_agent_commit_message(tmp_path)
    assert result == ("Fix flake8 pre-commit config", "Details here.\n")
    assert not path.exists()


def test_consume_missing_or_empty(tmp_path: Path) -> None:
    assert consume_agent_commit_message(tmp_path) is None
    (tmp_path / COMMIT_MESSAGE_FILENAME).write_text("\n\n", encoding="utf-8")
    assert consume_agent_commit_message(tmp_path) is None
    assert not (tmp_path / COMMIT_MESSAGE_FILENAME).exists()


def test_instructions_mention_filename() -> None:
    text = commit_message_instructions()
    assert COMMIT_MESSAGE_FILENAME in text
    assert "Do NOT include an `agent:` prefix" in text
