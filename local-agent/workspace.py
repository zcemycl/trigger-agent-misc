"""Git worktree and branch management for issue tasks."""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class WorkspaceError(RuntimeError):
    """Raised when a git/worktree operation fails."""


@dataclass
class WorktreeInfo:
    issue_number: int
    branch: str
    path: Path


class WorkspaceManager:
    """Create and manage per-issue git worktrees under WORKSPACE_ROOT."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.source_repo = config.source_repo
        self.workspace_root = config.workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def branch_name(self, issue_number: int) -> str:
        return f"agent/issue-{issue_number}"

    def worktree_path(self, issue_number: int) -> Path:
        return self.workspace_root / f"issue-{issue_number}"

    def _git(
        self,
        args: list[str],
        *,
        cwd: Optional[Path] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["git", *args]
        workdir = cwd or self.source_repo
        logger.debug("git %s (cwd=%s)", " ".join(shlex.quote(a) for a in args), workdir)
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise WorkspaceError(
                f"git {' '.join(args)} failed in {workdir}: "
                f"{(result.stderr or result.stdout).strip()}"
            )
        return result

    def fetch_origin(self) -> None:
        self._git(["fetch", "origin", "--prune"])

    def remote_branch_exists(self, branch: str) -> bool:
        result = self._git(
            ["ls-remote", "--heads", "origin", branch],
            check=False,
        )
        return bool(result.stdout.strip())

    def local_branch_exists(self, branch: str) -> bool:
        result = self._git(
            ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            check=False,
        )
        return result.returncode == 0

    def list_worktrees(self) -> list[dict[str, str]]:
        result = self._git(["worktree", "list", "--porcelain"])
        entries: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                if current:
                    entries.append(current)
                    current = {}
                continue
            if line.startswith("worktree "):
                current["path"] = line[len("worktree ") :]
            elif line.startswith("HEAD "):
                current["head"] = line[len("HEAD ") :]
            elif line.startswith("branch "):
                current["branch"] = line[len("branch ") :]
            elif line == "detached":
                current["detached"] = "1"
            elif line == "bare":
                current["bare"] = "1"
        if current:
            entries.append(current)
        return entries

    def worktree_for_branch(self, branch: str) -> Optional[Path]:
        ref = f"refs/heads/{branch}"
        for entry in self.list_worktrees():
            if entry.get("branch") == ref:
                return Path(entry["path"])
        return None

    def ensure_worktree(self, issue_number: int) -> WorktreeInfo:
        """
        Fetch origin, create branch from base if needed, and create a worktree.

        Refuses to overwrite an existing worktree directory that is not already
        registered for this branch.
        """
        branch = self.branch_name(issue_number)
        path = self.worktree_path(issue_number)

        self.fetch_origin()

        existing = self.worktree_for_branch(branch)
        if existing is not None:
            if existing.resolve() != path.resolve():
                raise WorkspaceError(
                    f"Branch {branch} is already checked out at {existing}, "
                    f"expected {path}"
                )
            logger.info("Reusing existing worktree for %s at %s", branch, path)
            return WorktreeInfo(issue_number=issue_number, branch=branch, path=path)

        if path.exists():
            raise WorkspaceError(
                f"Worktree path already exists but is not registered: {path}. "
                "Remove it manually or choose another WORKSPACE_ROOT."
            )

        base_ref = f"origin/{self.config.base_branch}"
        # Verify base exists remotely/locally.
        probe = self._git(["rev-parse", "--verify", base_ref], check=False)
        if probe.returncode != 0:
            raise WorkspaceError(
                f"Base branch ref {base_ref} not found after fetch. "
                f"Check BASE_BRANCH={self.config.base_branch!r}"
            )

        if self.local_branch_exists(branch):
            # Branch exists locally but has no worktree — attach worktree to it.
            self._git(["worktree", "add", str(path), branch])
        elif self.remote_branch_exists(branch):
            # Track existing remote branch without force-resetting it.
            self._git(
                ["worktree", "add", "--track", "-b", branch, str(path), f"origin/{branch}"]
            )
        else:
            self._git(["worktree", "add", "-b", branch, str(path), base_ref])

        return WorktreeInfo(issue_number=issue_number, branch=branch, path=path)

    def get_diff(self, worktree: Path, *, against: str | None = None) -> str:
        base = against or f"origin/{self.config.base_branch}"
        result = self._git(["diff", f"{base}...HEAD"], cwd=worktree, check=False)
        return result.stdout

    def status_porcelain(self, worktree: Path) -> str:
        result = self._git(["status", "--porcelain"], cwd=worktree)
        return result.stdout

    def has_changes(self, worktree: Path) -> bool:
        return bool(self.status_porcelain(worktree).strip())

    def commit_all(self, worktree: Path, message: str) -> bool:
        """Stage all changes and commit. Returns False if there was nothing to commit."""
        self._git(["add", "-A"], cwd=worktree)
        if not self.has_changes(worktree):
            # After add, check for staged changes specifically.
            staged = self._git(["diff", "--cached", "--name-only"], cwd=worktree)
            if not staged.stdout.strip():
                logger.info("No changes to commit in %s", worktree)
                return False
        result = self._git(
            ["commit", "-m", message],
            cwd=worktree,
            check=False,
        )
        if result.returncode != 0:
            combined = f"{result.stderr}\n{result.stdout}"
            if "nothing to commit" in combined.lower():
                return False
            raise WorkspaceError(
                f"git commit failed in {worktree}: {combined.strip()}"
            )
        return True

    def push_branch(self, worktree: Path, branch: str) -> None:
        self._git(["push", "-u", "origin", branch], cwd=worktree)

    def remove_worktree(self, issue_number: int, *, force: bool = False) -> None:
        path = self.worktree_path(issue_number)
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path))
        result = self._git(args, check=False)
        if result.returncode != 0 and path.exists():
            raise WorkspaceError(
                f"Failed to remove worktree {path}: "
                f"{(result.stderr or result.stdout).strip()}"
            )
