"""Local GitHub-to-Cursor coding agent controller."""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from agent_runner import AgentRunner
from config import Config, ConfigError, load_config
from database import Database
from github_client import GitHubClient
from task_manager import TaskManager
from workspace import WorkspaceManager

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.handlers.clear()
    root.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "controller.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._stop = False
        self.db = Database(config.database_path)
        self.github = GitHubClient(config)
        self.workspaces = WorkspaceManager(config)
        self.agent_runner = AgentRunner(config)
        self.tasks = TaskManager(
            config,
            self.db,
            self.github,
            self.workspaces,
            self.agent_runner,
        )

    def request_shutdown(self, signum: int | None = None, _frame=None) -> None:
        logger.info("Shutdown requested (signal=%s)", signum)
        self._stop = True

    def run(self) -> None:
        logger.info(
            "Controller starting for %s (base=%s, poll=%ss, max_tasks=%s)",
            self.config.repo_slug,
            self.config.base_branch,
            self.config.poll_interval,
            self.config.max_concurrent_tasks,
        )
        logger.info("Source repo: %s", self.config.source_repo)
        logger.info("Workspace root: %s", self.config.workspace_root)
        logger.warning(
            "SECURITY: Cursor Agent is invoked with --force. This is NOT a sandbox; "
            "it can run shell commands and modify files in the worktree. "
            "Issue bodies and PR comments are untrusted input."
        )

        while not self._stop:
            try:
                self.tasks.process_ready_issues()
                self.tasks.process_pr_comments()
                self.tasks.reconcile_tasks()
            except Exception:
                logger.exception("Unhandled error in polling loop")
            # Interruptible sleep.
            for _ in range(self.config.poll_interval):
                if self._stop:
                    break
                time.sleep(1)

        logger.info("Draining workers...")
        self.tasks.shutdown(wait=True)
        logger.info("Controller stopped")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    env_file = None
    if argv:
        env_file = Path(argv[0])

    try:
        config = load_config(env_file)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.log_dir)
    controller = Controller(config)

    signal.signal(signal.SIGINT, controller.request_shutdown)
    signal.signal(signal.SIGTERM, controller.request_shutdown)

    controller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
