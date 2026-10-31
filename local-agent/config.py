"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    github_repo: str
    owner: str
    repo: str
    base_branch: str
    source_repo: Path
    workspace_root: Path
    poll_interval: int
    max_concurrent_tasks: int
    agent_command: str
    agent_timeout: int
    test_command: str
    trusted_github_users: frozenset[str]
    database_path: Path
    log_dir: Path
    feedback_batch_seconds: int
    max_agent_retries: int
    ready_label: str = "agent:ready"
    running_label: str = "agent:running"
    review_label: str = "agent:review"
    failed_label: str = "agent:failed"
    agent_extra_args: tuple[str, ...] = field(default_factory=lambda: ("-p", "--force", "--output-format", "text"))
    # Commit attribution (push auth can still be the human gh account).
    commit_message_prefix: str = "agent:"
    git_author_name: str = "local-agent"
    git_author_email: str = "local-agent[bot]@users.noreply.github.com"
    git_committer_name: str = ""
    git_committer_email: str = ""
    commit_co_authored_by: str = "local-agent <local-agent[bot]@users.noreply.github.com>"

    @property
    def repo_slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    def format_commit_message(self, subject: str) -> str:
        """Build a controller commit message with optional prefix and trailer."""
        text = subject.strip()
        prefix = self.commit_message_prefix.strip()
        if prefix:
            if not text.lower().startswith(prefix.lower()):
                text = f"{prefix} {text}".strip()
        trailer = self.commit_co_authored_by.strip()
        if trailer:
            if not trailer.lower().startswith("co-authored-by:"):
                trailer = f"Co-authored-by: {trailer}"
            text = f"{text}\n\n{trailer}"
        return text

    def git_commit_env(self) -> dict[str, str]:
        """Env overrides so agent commits are attributable even with human push auth."""
        env: dict[str, str] = {}
        if self.git_author_name.strip():
            env["GIT_AUTHOR_NAME"] = self.git_author_name.strip()
            env["GIT_COMMITTER_NAME"] = (
                self.git_committer_name.strip() or self.git_author_name.strip()
            )
        if self.git_author_email.strip():
            env["GIT_AUTHOR_EMAIL"] = self.git_author_email.strip()
            env["GIT_COMMITTER_EMAIL"] = (
                self.git_committer_email.strip() or self.git_author_email.strip()
            )
        if self.git_committer_name.strip():
            env["GIT_COMMITTER_NAME"] = self.git_committer_name.strip()
        if self.git_committer_email.strip():
            env["GIT_COMMITTER_EMAIL"] = self.git_committer_email.strip()
        return env


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Required environment variable {name} is missing or empty")
    return value


def _parse_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_users(raw: str) -> frozenset[str]:
    if not raw.strip():
        return frozenset()
    return frozenset(part.strip().lstrip("@").lower() for part in raw.split(",") if part.strip())


def _parse_repo(slug: str) -> tuple[str, str]:
    parts = slug.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ConfigError(
            f"GITHUB_REPO must be OWNER/REPOSITORY, got {slug!r}"
        )
    return parts[0], parts[1]


def load_config(env_file: str | Path | None = None) -> Config:
    """Load and validate configuration from the environment."""
    if env_file is not None:
        load_dotenv(env_file)
    else:
        # Prefer local-agent/.env, then process cwd.
        local_env = Path(__file__).resolve().parent / ".env"
        if local_env.exists():
            load_dotenv(local_env)
        else:
            load_dotenv()

    github_repo = _require("GITHUB_REPO")
    owner, repo = _parse_repo(github_repo)

    source_repo = Path(_require("SOURCE_REPO")).expanduser().resolve()
    workspace_root = Path(_require("WORKSPACE_ROOT")).expanduser().resolve()

    database_path = Path(
        os.getenv(
            "DATABASE_PATH",
            str(Path(__file__).resolve().parent / ".agent_data" / "agent.db"),
        )
    ).expanduser().resolve()
    log_dir = Path(
        os.getenv(
            "LOG_DIR",
            str(Path(__file__).resolve().parent / ".agent_data" / "logs"),
        )
    ).expanduser().resolve()

    if not source_repo.exists():
        raise ConfigError(f"SOURCE_REPO does not exist: {source_repo}")
    if not (source_repo / ".git").exists() and not (source_repo / ".git").is_file():
        # .git may be a file in worktrees; for the source bare/normal repo expect dir or file.
        git_dir = source_repo / ".git"
        if not git_dir.exists():
            raise ConfigError(f"SOURCE_REPO is not a git repository: {source_repo}")

    agent_command = os.getenv("AGENT_COMMAND", "agent").strip() or "agent"
    if shutil.which(agent_command) is None and not Path(agent_command).exists():
        raise ConfigError(
            f"AGENT_COMMAND not found on PATH: {agent_command!r}. "
            "Install Cursor Agent CLI or set AGENT_COMMAND."
        )

    for required_bin in ("git", "gh"):
        if shutil.which(required_bin) is None:
            raise ConfigError(f"Required executable not found on PATH: {required_bin}")

    workspace_root.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        github_repo=github_repo,
        owner=owner,
        repo=repo,
        base_branch=os.getenv("BASE_BRANCH", "main").strip() or "main",
        source_repo=source_repo,
        workspace_root=workspace_root,
        poll_interval=_parse_int("POLL_INTERVAL", 30),
        max_concurrent_tasks=_parse_int("MAX_CONCURRENT_TASKS", 1),
        agent_command=agent_command,
        agent_timeout=_parse_int("AGENT_TIMEOUT", 1800, minimum=30),
        test_command=os.getenv("TEST_COMMAND", "pytest -q").strip() or "pytest -q",
        trusted_github_users=_parse_users(os.getenv("TRUSTED_GITHUB_USERS", "")),
        database_path=database_path,
        log_dir=log_dir,
        feedback_batch_seconds=_parse_int("FEEDBACK_BATCH_SECONDS", 15, minimum=0),
        max_agent_retries=_parse_int("MAX_AGENT_RETRIES", 2, minimum=0),
        commit_message_prefix=os.getenv("COMMIT_MESSAGE_PREFIX", "agent:").strip(),
        git_author_name=os.getenv("GIT_AUTHOR_NAME", "local-agent").strip(),
        git_author_email=os.getenv(
            "GIT_AUTHOR_EMAIL",
            "local-agent[bot]@users.noreply.github.com",
        ).strip(),
        git_committer_name=os.getenv("GIT_COMMITTER_NAME", "").strip(),
        git_committer_email=os.getenv("GIT_COMMITTER_EMAIL", "").strip(),
        commit_co_authored_by=os.getenv(
            "COMMIT_CO_AUTHORED_BY",
            "local-agent <local-agent[bot]@users.noreply.github.com>",
        ).strip(),
    )
