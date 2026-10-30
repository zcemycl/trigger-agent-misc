"""Orchestrates issue → agent → tests → commit → PR (requirements 1–4)."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from agent_runner import AgentRunner
from config import Config
from database import Database
from github_client import GitHubClient, Issue
from workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(
        self,
        config: Config,
        db: Database,
        github: GitHubClient,
        workspaces: WorkspaceManager,
        agent_runner: AgentRunner,
    ) -> None:
        self.config = config
        self.db = db
        self.github = github
        self.workspaces = workspaces
        self.agent_runner = agent_runner
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_tasks,
            thread_name_prefix="agent-worker",
        )
        self._inflight: dict[int, Future] = {}
        self._inflight_lock = threading.Lock()
        self._owner = f"task-manager-{threading.get_ident()}"

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def active_task_count(self) -> int:
        with self._inflight_lock:
            self._inflight = {
                num: fut for num, fut in self._inflight.items() if not fut.done()
            }
            return len(self._inflight)

    def process_ready_issues(self) -> None:
        """Poll for agent:ready issues and enqueue work up to MAX_CONCURRENT_TASKS."""
        try:
            issues = self.github.list_ready_issues()
        except Exception:
            logger.exception("Failed to list ready issues")
            return

        if not issues:
            return

        for issue in issues:
            if self.active_task_count() >= self.config.max_concurrent_tasks:
                logger.info(
                    "At max concurrent tasks (%s); deferring remaining issues",
                    self.config.max_concurrent_tasks,
                )
                break

            if self.db.task_exists(issue.number):
                existing = self.db.get_task(issue.number)
                # If already tracked, do not start a duplicate initial run.
                logger.info(
                    "Issue #%s already tracked (status=%s); skipping duplicate ready pickup",
                    issue.number,
                    existing.status if existing else "?",
                )
                # Still clear the ready label so we do not loop forever.
                try:
                    self.github.replace_agent_label(
                        issue.number,
                        remove=self.config.ready_label,
                        add=self.config.running_label
                        if existing and existing.status in {"queued", "running"}
                        else (
                            self.config.review_label
                            if existing and existing.status in {"pr_open", "review_pending", "updating"}
                            else self.config.failed_label
                        ),
                        current_labels=issue.labels,
                    )
                except Exception:
                    logger.exception(
                        "Failed to clear ready label on already-tracked issue #%s",
                        issue.number,
                    )
                continue

            with self._inflight_lock:
                if issue.number in self._inflight and not self._inflight[issue.number].done():
                    continue

            if not self.db.try_acquire_lock(issue.number, self._owner):
                logger.info("Issue #%s is locked by another worker; skipping", issue.number)
                continue

            try:
                future = self._executor.submit(self._handle_new_issue, issue)
                with self._inflight_lock:
                    self._inflight[issue.number] = future
                future.add_done_callback(
                    lambda fut, n=issue.number: self._on_future_done(n, fut)
                )
            except Exception:
                self.db.release_lock(issue.number, self._owner)
                logger.exception("Failed to enqueue issue #%s", issue.number)

    def _on_future_done(self, issue_number: int, fut: Future) -> None:
        try:
            fut.result()
        except Exception:
            logger.exception("Worker for issue #%s raised", issue_number)
        finally:
            self.db.release_lock(issue_number, self._owner)
            with self._inflight_lock:
                self._inflight.pop(issue_number, None)

    def _handle_new_issue(self, issue: Issue) -> None:
        logger.info("Starting implementation for issue #%s: %s", issue.number, issue.title)
        branch = self.workspaces.branch_name(issue.number)
        workspace_path = self.workspaces.worktree_path(issue.number)

        # Claim the issue on GitHub before heavy work.
        try:
            self.github.replace_agent_label(
                issue.number,
                remove=self.config.ready_label,
                add=self.config.running_label,
                current_labels=issue.labels,
            )
        except Exception:
            logger.exception("Failed to set running label on issue #%s", issue.number)
            return

        try:
            task = self.db.create_task(
                issue_number=issue.number,
                branch=branch,
                workspace=str(workspace_path),
                status="queued",
            )
        except Exception:
            logger.exception("Failed to create task record for issue #%s", issue.number)
            self._mark_failed_on_github(issue.number, issue.labels)
            return

        try:
            self.db.update_task(issue.number, status="running", last_error=None)
            wt = self.workspaces.ensure_worktree(issue.number)
            prompt = self.agent_runner.build_implementation_prompt(
                issue_number=issue.number,
                title=issue.title,
                body=issue.body,
            )
            agent_result = self.agent_runner.run_agent(
                worktree=wt.path,
                prompt=prompt,
                issue_number=issue.number,
            )
            if agent_result.timed_out or agent_result.exit_code != 0:
                self._fail_task(
                    issue.number,
                    f"Agent failed (exit={agent_result.exit_code}, "
                    f"timeout={agent_result.timed_out}). "
                    f"Log: {agent_result.log_path}",
                    labels=issue.labels,
                )
                return

            test_result = self.agent_runner.run_tests(
                worktree=wt.path,
                issue_number=issue.number,
            )
            if test_result.timed_out or test_result.exit_code != 0:
                self._fail_task(
                    issue.number,
                    f"Tests failed (exit={test_result.exit_code}, "
                    f"timeout={test_result.timed_out}). "
                    f"Log: {test_result.log_path}",
                    labels=issue.labels,
                )
                return

            committed = self.workspaces.commit_all(
                wt.path,
                f"Implement issue #{issue.number}",
            )
            if not committed:
                # Agent may have made no file changes — still treat as failure to publish.
                self._fail_task(
                    issue.number,
                    "No changes to commit after agent run",
                    labels=issue.labels,
                )
                return

            self.workspaces.push_branch(wt.path, wt.branch)
            pr = self.github.create_pull_request(
                head=wt.branch,
                base=self.config.base_branch,
                title=issue.title or f"Implement issue #{issue.number}",
                body=f"Automated implementation. Closes #{issue.number}",
            )
            self.db.update_task(
                issue.number,
                status="pr_open",
                pr_number=pr.number,
                pr_url=pr.url,
                last_error=None,
            )
            self.github.replace_agent_label(
                issue.number,
                remove=self.config.running_label,
                add=self.config.review_label,
                current_labels=self.github.get_issue(issue.number).labels,
            )
            # Move to review_pending once PR is up.
            self.db.update_task(issue.number, status="review_pending")
            logger.info(
                "Opened PR #%s for issue #%s: %s",
                pr.number,
                issue.number,
                pr.url,
            )
        except Exception as exc:
            logger.exception("Pipeline failed for issue #%s", issue.number)
            self._fail_task(issue.number, str(exc), labels=issue.labels)

    def _fail_task(
        self,
        issue_number: int,
        error: str,
        *,
        labels: Optional[list[str]] = None,
    ) -> None:
        logger.error("Issue #%s failed: %s", issue_number, error)
        try:
            self.db.update_task(issue_number, status="failed", last_error=error)
        except KeyError:
            pass
        self._mark_failed_on_github(issue_number, labels)

    def _mark_failed_on_github(
        self,
        issue_number: int,
        labels: Optional[list[str]],
    ) -> None:
        try:
            current = labels or self.github.get_issue(issue_number).labels
            self.github.replace_agent_label(
                issue_number,
                remove=self.config.running_label,
                add=self.config.failed_label,
                current_labels=current,
            )
            # Also strip ready if somehow still present.
            if self.config.ready_label in (
                self.github.get_issue(issue_number).labels
            ):
                self.github.remove_issue_label(
                    issue_number, self.config.ready_label
                )
        except Exception:
            logger.exception(
                "Failed to set failed label on issue #%s", issue_number
            )

    def process_pr_comments(self) -> None:
        """Placeholder for requirement 5+ (comment monitoring)."""
        logger.debug("process_pr_comments: not implemented yet (stage 5+)")

    def reconcile_tasks(self) -> None:
        """Placeholder for requirement 8 (PR lifecycle). Light status sync only."""
        for task in self.db.list_tasks(
            statuses=["pr_open", "review_pending", "updating"]
        ):
            if not task.pr_number:
                continue
            try:
                pr = self.github.get_pull_request(task.pr_number)
            except Exception:
                logger.exception(
                    "Failed to reconcile PR #%s for issue #%s",
                    task.pr_number,
                    task.issue_number,
                )
                continue
            if pr.merged:
                self.db.update_task(task.issue_number, status="merged", last_error=None)
                logger.info(
                    "Issue #%s PR #%s merged; marked merged",
                    task.issue_number,
                    task.pr_number,
                )
            elif pr.state == "closed":
                self.db.update_task(task.issue_number, status="closed", last_error=None)
                logger.info(
                    "Issue #%s PR #%s closed without merge; marked closed",
                    task.issue_number,
                    task.pr_number,
                )
