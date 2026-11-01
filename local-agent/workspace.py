"""Git worktree and branch management for issue tasks."""

from __future__ import annotations

import logging
import os
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


@dataclass
class RebaseResult:
    """Outcome of syncing an agent branch onto the configured base."""

    changed: bool
    already_up_to_date: bool
    base_ref: str
    aborted_on_conflict: bool = False
    conflict_in_progress: bool = False
    detail: str = ""


class WorkspaceManager:
    """Create and manage per-issue git worktrees under WORKSPACE_ROOT."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.source_repo = config.source_repo
        self.workspace_root = config.workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def branch_name(self, issue_number: int) -> str:
        """Default bootstrap branch before the LLM proposes a descriptive name."""
        return f"agent/issue-{issue_number}"

    def worktree_path(self, issue_number: int) -> Path:
        return self.workspace_root / f"issue-{issue_number}"

    def current_branch(self, worktree: Path) -> str:
        result = self._git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree)
        return result.stdout.strip()

    def rename_branch(self, worktree: Path, new_branch: str) -> str:
        """Rename the branch checked out in the worktree. Returns the new name."""
        current = self.current_branch(worktree)
        if current == new_branch:
            return current
        self._git(["branch", "-m", new_branch], cwd=worktree)
        logger.info("Renamed branch %s -> %s", current, new_branch)
        return new_branch

    def pick_unique_branch(
        self,
        desired: str,
        *,
        issue_number: int,
        current: str | None = None,
    ) -> str:
        """Return desired if free (or already current); otherwise a unique variant."""
        if current and desired == current:
            return desired
        candidates = [desired]
        suffix = f"-{issue_number}"
        if not desired.endswith(suffix):
            trimmed = desired[: max(1, 60 - len(suffix))].rstrip("-/")
            candidates.append(f"{trimmed}{suffix}")
        for i in range(2, 30):
            extra = f"-{i}"
            trimmed = desired[: max(1, 60 - len(extra))].rstrip("-/")
            candidates.append(f"{trimmed}{extra}")
        fallback = self.branch_name(issue_number)
        if fallback not in candidates:
            candidates.append(fallback)

        for name in candidates:
            if current and name == current:
                return name
            if self.local_branch_exists(name) or self.remote_branch_exists(name):
                continue
            return name
        return fallback

    def _git(
        self,
        args: list[str],
        *,
        cwd: Optional[Path] = None,
        check: bool = True,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["git", *args]
        workdir = cwd or self.source_repo
        logger.debug("git %s (cwd=%s)", " ".join(shlex.quote(a) for a in args), workdir)
        run_env = os.environ.copy()
        # Never pass blank git identity vars (dotenv `GIT_COMMITTER_NAME=` breaks rebase).
        for key in (
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "GIT_COMMITTER_NAME",
            "GIT_COMMITTER_EMAIL",
        ):
            if key in run_env and not str(run_env.get(key, "")).strip():
                run_env.pop(key, None)
        if env:
            run_env.update({k: v for k, v in env.items() if v.strip()})
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
            env=run_env,
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

    def ensure_worktree(
        self,
        issue_number: int,
        *,
        branch: str | None = None,
    ) -> WorktreeInfo:
        """
        Fetch origin, create branch from base if needed, and create a worktree.

        If `branch` is provided (e.g. a previously renamed LLM branch), prefer it.
        An existing worktree directory is reused even if it was renamed away from
        the bootstrap `agent/issue-N` name.
        """
        preferred = branch or self.branch_name(issue_number)
        path = self.worktree_path(issue_number)

        self.fetch_origin()

        if path.exists():
            try:
                current = self.current_branch(path)
            except WorkspaceError as exc:
                raise WorkspaceError(
                    f"Worktree path exists but is not usable: {path}: {exc}"
                ) from exc
            logger.info(
                "Reusing existing worktree for issue #%s at %s (%s)",
                issue_number,
                path,
                current,
            )
            return WorktreeInfo(
                issue_number=issue_number,
                branch=current or preferred,
                path=path,
            )

        existing = self.worktree_for_branch(preferred)
        if existing is not None:
            if existing.resolve() != path.resolve():
                raise WorkspaceError(
                    f"Branch {preferred} is already checked out at {existing}, "
                    f"expected {path}"
                )
            logger.info("Reusing existing worktree for %s at %s", preferred, path)
            return WorktreeInfo(
                issue_number=issue_number, branch=preferred, path=path
            )

        base_ref = f"origin/{self.config.base_branch}"
        probe = self._git(["rev-parse", "--verify", base_ref], check=False)
        if probe.returncode != 0:
            raise WorkspaceError(
                f"Base branch ref {base_ref} not found after fetch. "
                f"Check BASE_BRANCH={self.config.base_branch!r}"
            )

        if self.local_branch_exists(preferred):
            self._git(["worktree", "add", str(path), preferred])
        elif self.remote_branch_exists(preferred):
            self._git(
                [
                    "worktree",
                    "add",
                    "--track",
                    "-b",
                    preferred,
                    str(path),
                    f"origin/{preferred}",
                ]
            )
        else:
            self._git(["worktree", "add", "-b", preferred, str(path), base_ref])

        return WorktreeInfo(
            issue_number=issue_number, branch=preferred, path=path
        )

    def branch_has_unpublished_commits(self, worktree: Path, branch: str) -> bool:
        """True when local branch has commits not on origin/branch (or vs base)."""
        self.fetch_origin()
        if self.remote_branch_exists(branch):
            result = self._git(
                ["rev-list", "--count", f"origin/{branch}..HEAD"],
                cwd=worktree,
                check=False,
            )
        else:
            result = self._git(
                ["rev-list", "--count", f"{self.base_ref()}..HEAD"],
                cwd=worktree,
                check=False,
            )
        try:
            return int((result.stdout or "0").strip() or "0") > 0
        except ValueError:
            return False

    def base_ref(self) -> str:
        return f"origin/{self.config.base_branch}"

    # Paths whose full patch content blows up prompts / argv limits.
    _OMIT_DIFF_BASENAMES = frozenset(
        {
            "uv.lock",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "Cargo.lock",
            "poetry.lock",
            "Gemfile.lock",
            "composer.lock",
            "Pipfile.lock",
        }
    )
    _OMIT_DIFF_SUFFIXES = (
        ".min.js",
        ".min.css",
        ".map",
        ".wasm",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".zip",
        ".gz",
        ".whl",
    )

    def get_diff(
        self,
        worktree: Path,
        *,
        against: str | None = None,
        max_chars: int = 80_000,
    ) -> str:
        """
        Return a compact diff suitable for agent prompts.

        Omits lockfile/binary-ish patches and truncates to max_chars so the
        controller never builds an argv/prompt that hits OS ARG_MAX.
        """
        base = against or self.base_ref()
        names = self._git(
            ["diff", "--name-status", f"{base}...HEAD"],
            cwd=worktree,
            check=False,
        )
        if not names.stdout.strip():
            return ""

        parts: list[str] = []
        omitted: list[str] = []
        total = 0
        truncated = False

        for line in names.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            status = cols[0]
            path = cols[-1]
            base_name = Path(path).name
            if (
                base_name in self._OMIT_DIFF_BASENAMES
                or path.endswith(self._OMIT_DIFF_SUFFIXES)
            ):
                omitted.append(f"{status} {path}")
                continue

            file_diff = self._git(
                ["diff", f"{base}...HEAD", "--", path],
                cwd=worktree,
                check=False,
            ).stdout
            if not file_diff:
                # Renames / mode-only: keep the name-status line.
                parts.append(f"{status}\t{path}")
                continue
            if total + len(file_diff) > max_chars:
                truncated = True
                remaining = max_chars - total
                if remaining > 200:
                    parts.append(file_diff[:remaining] + "\n... [diff truncated] ...\n")
                    total = max_chars
                break
            parts.append(file_diff)
            total += len(file_diff)

        summary_bits: list[str] = []
        if omitted:
            summary_bits.append(
                "Omitted large/binary paths (inspect locally if needed):\n"
                + "\n".join(f"  - {item}" for item in omitted)
            )
        if truncated:
            summary_bits.append(
                f"Diff truncated to ~{max_chars} characters for prompt size limits."
            )
        body = "\n".join(parts).strip()
        if summary_bits:
            header = "\n".join(summary_bits) + "\n\n"
            return header + body
        return body

    def status_porcelain(self, worktree: Path) -> str:
        result = self._git(["status", "--porcelain"], cwd=worktree)
        return result.stdout

    def has_changes(self, worktree: Path) -> bool:
        return bool(self.status_porcelain(worktree).strip())

    def rev_parse(self, worktree: Path, ref: str) -> str:
        result = self._git(["rev-parse", ref], cwd=worktree)
        return result.stdout.strip()

    def is_ancestor(self, worktree: Path, maybe_ancestor: str, ref: str) -> bool:
        result = self._git(
            ["merge-base", "--is-ancestor", maybe_ancestor, ref],
            cwd=worktree,
            check=False,
        )
        return result.returncode == 0

    def needs_rebase_onto_base(self, worktree: Path) -> bool:
        """True when origin/BASE_BRANCH has commits not reachable from HEAD."""
        base = self.base_ref()
        probe = self._git(["rev-parse", "--verify", base], cwd=worktree, check=False)
        if probe.returncode != 0:
            raise WorkspaceError(f"Base ref {base} not found in {worktree}")
        return not self.is_ancestor(worktree, base, "HEAD")

    def _git_path(self, worktree: Path, name: str) -> Path:
        result = self._git(["rev-parse", "--git-path", name], cwd=worktree)
        path = Path(result.stdout.strip())
        if not path.is_absolute():
            path = worktree / path
        return path

    def is_rebase_in_progress(self, worktree: Path) -> bool:
        return self._git_path(worktree, "rebase-merge").exists() or self._git_path(
            worktree, "rebase-apply"
        ).exists()

    def conflicted_files(self, worktree: Path) -> list[str]:
        result = self._git(
            ["diff", "--name-only", "--diff-filter=U"],
            cwd=worktree,
            check=False,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def abort_rebase(self, worktree: Path) -> None:
        if self.is_rebase_in_progress(worktree):
            self._git(["rebase", "--abort"], cwd=worktree, check=False)

    def continue_rebase(self, worktree: Path) -> subprocess.CompletedProcess[str]:
        """Continue an in-progress rebase without opening an editor."""
        env = {
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",
        }
        # Rebase continue creates commits; use agent commit identity when configured.
        env.update(self.config.git_commit_env())
        return self._git(
            ["rebase", "--continue"],
            cwd=worktree,
            check=False,
            env=env,
        )

    def stage_paths(self, worktree: Path, paths: list[str] | None = None) -> None:
        if paths:
            self._git(["add", "--", *paths], cwd=worktree)
        else:
            self._git(["add", "-A"], cwd=worktree)

    def rebase_onto_base(
        self,
        worktree: Path,
        *,
        fetch: bool = True,
        abort_on_conflict: bool = True,
    ) -> RebaseResult:
        """
        Rebase the worktree branch onto origin/BASE_BRANCH when the parent moved.

        When abort_on_conflict is False and git stops for conflicts, leaves the
        rebase in progress so a resolver can fix files and continue.
        """
        if fetch:
            self.fetch_origin()

        base = self.base_ref()
        if self.has_changes(worktree) and not self.is_rebase_in_progress(worktree):
            raise WorkspaceError(
                f"Cannot rebase {worktree}: working tree has uncommitted changes"
            )

        if not self.needs_rebase_onto_base(worktree):
            logger.info("%s already contains %s; no rebase needed", worktree, base)
            return RebaseResult(
                changed=False,
                already_up_to_date=True,
                base_ref=base,
            )

        logger.info("Rebasing %s onto %s", worktree, base)
        result = self._git(["rebase", base], cwd=worktree, check=False)
        if result.returncode == 0:
            logger.info("Rebase onto %s succeeded in %s", base, worktree)
            return RebaseResult(
                changed=True,
                already_up_to_date=False,
                base_ref=base,
            )

        detail = (result.stderr or result.stdout).strip()
        if self.is_rebase_in_progress(worktree) and not abort_on_conflict:
            logger.warning(
                "Rebase onto %s paused for conflicts in %s", base, worktree
            )
            return RebaseResult(
                changed=False,
                already_up_to_date=False,
                base_ref=base,
                conflict_in_progress=True,
                detail=detail,
            )

        self.abort_rebase(worktree)
        raise WorkspaceError(
            f"Rebase onto {base} failed in {worktree} (aborted): {detail}"
        )

    @staticmethod
    def _pre_commit_rewrote_files(output: str) -> bool:
        """True when pre-commit/format hooks failed after rewriting files."""
        lower = output.lower()
        markers = (
            "files were modified by this hook",
            "reformatted",
            "fixed by",
            "hook id: black",
            "hook id: prettier",
            "hook id: isort",
            "hook id: ruff",
        )
        return any(marker in lower for marker in markers)

    def commit_all(
        self,
        worktree: Path,
        message: str,
        *,
        env: Optional[dict[str, str]] = None,
        max_hook_retries: int = 2,
    ) -> bool:
        """
        Stage all changes and commit.

        Retries when local pre-commit hooks rewrite files (black/prettier/etc.)
        and reject the first commit attempt — a common autofix pattern.
        """
        commit_env = env if env is not None else self.config.git_commit_env()
        last_output = ""
        total_attempts = max_hook_retries + 1

        for attempt in range(1, total_attempts + 1):
            self._git(["add", "-A"], cwd=worktree)
            staged = self._git(["diff", "--cached", "--name-only"], cwd=worktree)
            if not staged.stdout.strip():
                logger.info("No staged changes to commit in %s", worktree)
                return False

            result = self._git(
                ["commit", "-m", message],
                cwd=worktree,
                check=False,
                env=commit_env,
            )
            if result.returncode == 0:
                if attempt > 1:
                    logger.info(
                        "Commit succeeded in %s after pre-commit autofix retry",
                        worktree,
                    )
                return True

            last_output = f"{result.stderr}\n{result.stdout}".strip()
            if "nothing to commit" in last_output.lower():
                return False

            hooks_rewrote = self._pre_commit_rewrote_files(last_output)
            dirty_after = self.has_changes(worktree)
            if attempt < total_attempts and (hooks_rewrote or dirty_after):
                logger.warning(
                    "Commit blocked by formatting hooks in %s "
                    "(attempt %s/%s); re-staging and retrying",
                    worktree,
                    attempt,
                    total_attempts,
                )
                continue

            break

        raise WorkspaceError(
            f"git commit failed in {worktree}: {last_output}"
        )

    def push_branch(
        self,
        worktree: Path,
        branch: str,
        *,
        force_with_lease: bool = False,
    ) -> None:
        args = ["push", "-u", "origin", branch]
        if force_with_lease:
            args = ["push", "--force-with-lease", "-u", "origin", branch]
        self._git(args, cwd=worktree)

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
