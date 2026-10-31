"""LLM-proposed git branch names for new issue runs."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BRANCH_NAME_FILENAME = ".local-agent-branch-name"
_DEFAULT_PREFIX = "agent/"
_MAX_LEN = 60


def branch_name_instructions(*, issue_number: int) -> str:
    return f"""## Branch name (required for new implementations)
Also write a short git branch name to this file in the worktree root (create/overwrite):

`{BRANCH_NAME_FILENAME}`

Rules:
- One line only.
- Describe the change from the issue (e.g. `add-latex-line-search-docs`).
- Use lowercase kebab-case: `a-z`, `0-9`, `/`, and `-` only.
- Do NOT include the `{_DEFAULT_PREFIX}` prefix — the controller adds `agent/`.
- Prefer including issue number `{issue_number}` somehow if natural (optional).
- Keep it under ~50 characters.
- Do NOT rename the git branch yourself; the controller will rename before push.
"""


def sanitize_branch_name(
    raw: str,
    *,
    issue_number: int,
    prefix: str = _DEFAULT_PREFIX,
) -> Optional[str]:
    """
    Normalize an LLM branch proposal into a safe name under agent/.

    Returns None if nothing usable remains (caller should keep the default).
    """
    line = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")[0]
    line = line.strip().strip("`\"'")
    if not line:
        return None

    # Drop accidental refs/heads/ or origin/ prefixes.
    for drop in ("refs/heads/", "origin/"):
        if line.startswith(drop):
            line = line[len(drop) :]

    lower = line.lower().replace("_", "-").replace(" ", "-")
    # If model included agent/ already, strip then re-apply.
    if lower.startswith(prefix.lower()):
        lower = lower[len(prefix) :]

    lower = re.sub(r"[^a-z0-9/-]+", "-", lower)
    lower = re.sub(r"-{2,}", "-", lower)
    lower = re.sub(r"/{2,}", "/", lower)
    lower = lower.strip("-/")
    if not lower:
        return None

    # Avoid nesting like agent/agent/...
    candidate = f"{prefix}{lower}"
    # Reserve room for optional -{issue} uniqueness suffix.
    max_core = _MAX_LEN
    if len(candidate) > max_core:
        candidate = candidate[:max_core].rstrip("-/")
    if not candidate.startswith(prefix) or candidate == prefix.rstrip("/"):
        return None

    # Ensure we can always disambiguate with issue number if needed.
    if f"issue-{issue_number}" not in candidate and str(issue_number) not in candidate:
        suffix = f"-{issue_number}"
        if len(candidate) + len(suffix) <= _MAX_LEN:
            candidate = candidate + suffix
        else:
            candidate = (candidate[: _MAX_LEN - len(suffix)]).rstrip("-/") + suffix
    return candidate


def consume_agent_branch_name(
    worktree: Path,
    *,
    issue_number: int,
) -> Optional[str]:
    """Read/delete the agent branch-name file and return a sanitized name."""
    path = worktree / BRANCH_NAME_FILENAME
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

    sanitized = sanitize_branch_name(raw, issue_number=issue_number)
    if not sanitized:
        logger.info("Agent branch name missing/invalid; keeping default branch")
        return None
    logger.info("Using LLM branch name: %s", sanitized)
    return sanitized
