"""Read LLM-authored commit messages from the worktree."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Written by Cursor Agent; consumed (and deleted) by the controller before commit.
COMMIT_MESSAGE_FILENAME = ".local-agent-commit-message"


def commit_message_instructions() -> str:
    return f"""## Commit message (required)
After finishing your code changes, write a concise git commit message to this file
in the worktree root (create/overwrite):

`{COMMIT_MESSAGE_FILENAME}`

Format:
- Line 1: subject only (<= ~60 characters). Describe the change, not the process.
- Optional blank line, then a short body with details.
- Do NOT include an `agent:` prefix, `Co-authored-by`, or sign-off — the controller adds those.
- Do NOT `git commit` yourself.
- Do NOT leave secrets in the message.
"""


def parse_commit_message_file(text: str) -> tuple[str, str]:
    """Split file contents into (subject, body)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Drop leading empty lines.
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return "", ""
    subject = " ".join(lines[0].split())
    rest = lines[1:]
    while rest and not rest[0].strip():
        rest.pop(0)
    body = "\n".join(rest).strip()
    if body:
        body = body + "\n"
    return subject, body


def consume_agent_commit_message(worktree: Path) -> Optional[tuple[str, str]]:
    """
    Read and remove the agent-written commit message file.

    Returns (subject, body) or None if missing/empty. Always tries to delete the
    file so it is never included in `git add -A`.
    """
    path = worktree / COMMIT_MESSAGE_FILENAME
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        raw = ""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to remove %s: %s", path, exc)

    subject, body = parse_commit_message_file(raw)
    if not subject:
        logger.info("Agent commit message file was empty; using fallback")
        return None
    logger.info("Using LLM commit subject: %s", subject)
    return subject, body
