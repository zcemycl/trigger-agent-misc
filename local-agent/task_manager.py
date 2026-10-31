"""Orchestrates issue → agent → tests → commit → PR and PR feedback loops."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from agent_runner import AgentRunner
from config import Config
from database import CommentEvent, Database, Task, _utc_now
from branch_name import consume_agent_branch_name
from commit_message import consume_agent_commit_message
from feedback import (
    IncomingComment,
    batch_ready,
    format_feedback_block,
    is_trusted_human,
    normalize_issue_comment,
    normalize_review,
    normalize_review_comment,
    summarize_feedback_commit,
)
from github_client import GitHubClient, Issue
from workspace import WorktreeInfo, WorkspaceManager

logger = logging.getLogger(__name__)

_FEEDBACK_ELIGIBLE_STATUSES = frozenset({"pr_open", "review_pending", "failed"})


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
        self._agent_login: Optional[str] = None
        self._agent_login_lock = threading.Lock()

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _get_agent_login(self) -> str:
        with self._agent_login_lock:
            if self._agent_login is None:
                try:
                    self._agent_login = self.github.get_authenticated_user()
                except Exception:
                    logger.exception("Failed to resolve authenticated GitHub user")
                    self._agent_login = ""
            return self._agent_login

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

            # Prefer LLM branch name before first push/PR (feedback keeps existing branch).
            proposed = consume_agent_branch_name(wt.path, issue_number=issue.number)
            if proposed:
                unique = self.workspaces.pick_unique_branch(
                    proposed,
                    issue_number=issue.number,
                    current=wt.branch,
                )
                if unique != wt.branch:
                    renamed = self.workspaces.rename_branch(wt.path, unique)
                    wt = WorktreeInfo(
                        issue_number=wt.issue_number,
                        branch=renamed,
                        path=wt.path,
                    )
                    self.db.update_task(issue.number, branch=renamed)
                    branch = renamed

            llm_msg = consume_agent_commit_message(wt.path)
            if llm_msg is not None:
                subject, body = llm_msg
            else:
                subject = f"#{issue.number} implement: {issue.title}".strip()
                if len(subject) > 64:
                    subject = subject[:63].rstrip() + "…"
                body = f"Automated implementation for issue #{issue.number}.\n"
            committed = self.workspaces.commit_all(
                wt.path,
                self.config.format_commit_message(subject, body=body),
            )
            if not committed:
                # Agent may have made no file changes — still treat as failure to publish.
                self._fail_task(
                    issue.number,
                    "No changes to commit after agent run",
                    labels=issue.labels,
                )
                return

            # Parent may have moved while the agent ran.
            rebased = self._rebase_branch(
                wt.path,
                wt.branch,
                push=False,
                issue_number=issue.number,
            )
            self.workspaces.push_branch(
                wt.path,
                wt.branch,
                force_with_lease=rebased,
            )
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
        """Ingest trusted PR feedback, batch briefly, then enqueue update runs."""
        if not self.config.trusted_github_users:
            logger.debug(
                "TRUSTED_GITHUB_USERS is empty; skipping PR feedback processing"
            )
            return

        tasks = self.db.list_tasks(
            statuses=["pr_open", "review_pending", "updating", "failed"]
        )
        for task in tasks:
            if not task.pr_number:
                continue
            try:
                self._ingest_pr_feedback(task)
            except Exception:
                logger.exception(
                    "Failed to ingest feedback for PR #%s (issue #%s)",
                    task.pr_number,
                    task.issue_number,
                )

        self._enqueue_ready_feedback_batches()

    def _ingest_pr_feedback(self, task: Task) -> None:
        assert task.pr_number is not None
        pr = self.github.get_pull_request(task.pr_number)
        if pr.merged or pr.state == "closed":
            return

        agent_login = self._get_agent_login()
        incoming: list[tuple[IncomingComment, dict]] = []

        for item in self.github.list_issue_comments(task.pr_number):
            incoming.append((normalize_issue_comment(task.pr_number, item), item))
        for item in self.github.list_review_comments(task.pr_number):
            incoming.append((normalize_review_comment(task.pr_number, item), item))
        for item in self.github.list_reviews(task.pr_number):
            normalized = normalize_review(task.pr_number, item)
            if normalized is not None:
                incoming.append((normalized, item))

        for comment, raw in incoming:
            if not comment.body:
                continue
            user = raw.get("user") or {}
            if not is_trusted_human(
                comment.author,
                trusted_users=self.config.trusted_github_users,
                agent_login=agent_login,
                user_payload=user,
            ):
                continue
            inserted = self.db.insert_comment_event(
                comment_id=comment.comment_id,
                pr_number=comment.pr_number,
                comment_type=comment.comment_type,
                author=comment.author,
                body=comment.body,
                path=comment.path,
                line=comment.line,
                diff_hunk=comment.diff_hunk,
                status="pending",
            )
            if inserted:
                logger.info(
                    "Queued %s from @%s on PR #%s",
                    comment.comment_type,
                    comment.author,
                    comment.pr_number,
                )

    def _enqueue_ready_feedback_batches(self) -> None:
        # Group pending + failed (retryable) events by PR.
        events = self.db.list_comment_events(statuses=["pending", "failed"])
        by_pr: dict[int, list[CommentEvent]] = {}
        for event in events:
            by_pr.setdefault(event.pr_number, []).append(event)

        for pr_number, pr_events in by_pr.items():
            if not batch_ready(pr_events, self.config.feedback_batch_seconds):
                logger.debug(
                    "PR #%s feedback batch still collecting (%s event(s))",
                    pr_number,
                    len(pr_events),
                )
                continue

            task = self.db.get_task_by_pr(pr_number)
            if task is None:
                continue
            if task.status not in _FEEDBACK_ELIGIBLE_STATUSES:
                continue
            if self.active_task_count() >= self.config.max_concurrent_tasks:
                logger.info(
                    "At max concurrent tasks; deferring feedback for PR #%s",
                    pr_number,
                )
                break

            with self._inflight_lock:
                existing = self._inflight.get(task.issue_number)
                if existing is not None and not existing.done():
                    continue

            if not self.db.try_acquire_lock(task.issue_number, self._owner):
                continue

            event_ids = [e.comment_id for e in pr_events]
            try:
                future = self._executor.submit(
                    self._handle_feedback, task.issue_number, event_ids
                )
                with self._inflight_lock:
                    self._inflight[task.issue_number] = future
                future.add_done_callback(
                    lambda fut, n=task.issue_number: self._on_future_done(n, fut)
                )
            except Exception:
                self.db.release_lock(task.issue_number, self._owner)
                logger.exception(
                    "Failed to enqueue feedback for issue #%s", task.issue_number
                )

    def _handle_feedback(self, issue_number: int, event_ids: list[str]) -> None:
        task = self.db.get_task(issue_number)
        if task is None or not task.pr_number:
            return

        events = [
            e
            for e in self.db.list_comment_events(pr_number=task.pr_number)
            if e.comment_id in set(event_ids)
        ]
        if not events:
            return

        logger.info(
            "Applying %s feedback event(s) to issue #%s / PR #%s",
            len(events),
            issue_number,
            task.pr_number,
        )

        for event in events:
            self.db.update_comment_event(event.comment_id, status="processing", error=None)

        try:
            pr = self.github.get_pull_request(task.pr_number)
            if pr.merged or pr.state == "closed":
                for event in events:
                    self.db.update_comment_event(
                        event.comment_id,
                        status="failed",
                        error="PR closed or merged",
                        processed_at=_utc_now(),
                    )
                return

            issue = self.github.get_issue(issue_number)
            self.db.update_task(issue_number, status="updating", last_error=None)

            # Ensure worktree exists (reuse branch; never create a new PR).
            wt = self.workspaces.ensure_worktree(issue_number)
            # Keep the PR branch current with BASE_BRANCH before applying feedback.
            self._rebase_branch(
                wt.path,
                wt.branch,
                push=True,
                issue_number=issue_number,
            )
            diff_text = self.workspaces.get_diff(wt.path)
            prompt = self.agent_runner.build_feedback_prompt(
                issue_number=issue_number,
                issue_title=issue.title,
                issue_body=issue.body,
                pr_number=pr.number,
                pr_title=pr.title,
                pr_body=pr.body,
                feedback_block=format_feedback_block(events),
                diff_text=diff_text,
            )
            agent_result = self.agent_runner.run_agent(
                worktree=wt.path,
                prompt=prompt,
                issue_number=issue_number,
            )
            if agent_result.timed_out or agent_result.exit_code != 0:
                raise RuntimeError(
                    f"Feedback agent failed (exit={agent_result.exit_code}, "
                    f"timeout={agent_result.timed_out}). Log: {agent_result.log_path}"
                )

            test_result = self.agent_runner.run_tests(
                worktree=wt.path,
                issue_number=issue_number,
            )
            if test_result.timed_out or test_result.exit_code != 0:
                raise RuntimeError(
                    f"Feedback tests failed (exit={test_result.exit_code}, "
                    f"timeout={test_result.timed_out}). Log: {test_result.log_path}"
                )

            # Feedback must stay on the existing PR branch.
            consume_agent_branch_name(wt.path, issue_number=issue_number)
            llm_msg = consume_agent_commit_message(wt.path)
            if llm_msg is not None:
                subject, body = llm_msg
            else:
                subject, body = summarize_feedback_commit(
                    events,
                    issue_number=issue_number,
                    pr_number=task.pr_number,
                )
            committed = self.workspaces.commit_all(
                wt.path,
                self.config.format_commit_message(subject, body=body),
            )
            if committed:
                # Rebase again in case base moved during the agent run, then push.
                rebased = self._rebase_branch(
                    wt.path,
                    wt.branch,
                    push=False,
                    issue_number=issue_number,
                )
                self.workspaces.push_branch(
                    wt.path,
                    wt.branch,
                    force_with_lease=rebased,
                )
            else:
                logger.info(
                    "Feedback run for issue #%s produced no file changes",
                    issue_number,
                )

            now = _utc_now()
            for event in events:
                self.db.update_comment_event(
                    event.comment_id,
                    status="completed",
                    processed_at=now,
                    error=None,
                )
            self.db.update_task(
                issue_number,
                status="review_pending",
                last_error=None,
            )
            # Restore review label if we came from failed.
            try:
                labels = self.github.get_issue(issue_number).labels
                if self.config.failed_label in labels:
                    self.github.replace_agent_label(
                        issue_number,
                        remove=self.config.failed_label,
                        add=self.config.review_label,
                        current_labels=labels,
                    )
            except Exception:
                logger.exception(
                    "Failed to refresh labels after feedback on issue #%s",
                    issue_number,
                )
            logger.info(
                "Feedback applied for issue #%s on existing PR #%s",
                issue_number,
                task.pr_number,
            )
        except Exception as exc:
            logger.exception("Feedback pipeline failed for issue #%s", issue_number)
            err = str(exc)
            now = _utc_now()
            for event in events:
                self.db.update_comment_event(
                    event.comment_id,
                    status="failed",
                    error=err,
                    # Record attempt time for retry backoff; status stays failed/retryable.
                    processed_at=now,
                )
            self._fail_task(issue_number, err)

    def _rebase_branch(
        self,
        worktree_path,
        branch: str,
        *,
        push: bool,
        issue_number: int | None = None,
    ) -> bool:
        """
        Rebase onto origin/BASE_BRANCH if needed.

        On conflicts, runs Cursor Agent to resolve markers, then continues the
        rebase. Returns True when a rebase rewrote the branch. When push=True
        and the branch was rewritten, publishes with --force-with-lease.
        """
        result = self.workspaces.rebase_onto_base(
            worktree_path,
            abort_on_conflict=False,
        )
        if result.already_up_to_date:
            return False
        if result.conflict_in_progress:
            issue_no = issue_number
            if issue_no is None:
                # Best-effort parse from agent/issue-N bootstrap names.
                issue_no = 0
            self._resolve_rebase_conflicts(
                worktree_path,
                issue_number=issue_no,
                base_ref=result.base_ref,
                detail=result.detail,
            )
        elif not result.changed:
            return False

        logger.info("Branch %s rebased onto %s", branch, result.base_ref)
        if push:
            self.workspaces.push_branch(
                worktree_path,
                branch,
                force_with_lease=True,
            )
        return True

    def _resolve_rebase_conflicts(
        self,
        worktree_path,
        *,
        issue_number: int,
        base_ref: str,
        detail: str,
        max_rounds: int = 8,
    ) -> None:
        """Drive agent conflict resolution until rebase finishes or fails."""
        from pathlib import Path

        worktree = Path(worktree_path)
        last_detail = detail
        for round_idx in range(1, max_rounds + 1):
            if not self.workspaces.is_rebase_in_progress(worktree):
                logger.info(
                    "Rebase finished during conflict resolution for issue #%s",
                    issue_number,
                )
                return

            conflicted = self.workspaces.conflicted_files(worktree)
            if not conflicted:
                # Sometimes only the continue step is needed (e.g. empty commit).
                cont = self.workspaces.continue_rebase(worktree)
                if cont.returncode == 0 and not self.workspaces.is_rebase_in_progress(
                    worktree
                ):
                    return
                last_detail = (cont.stderr or cont.stdout or last_detail).strip()
                if self.workspaces.is_rebase_in_progress(worktree):
                    conflicted = self.workspaces.conflicted_files(worktree)
                if not conflicted and cont.returncode != 0:
                    self.workspaces.abort_rebase(worktree)
                    raise RuntimeError(
                        f"rebase --continue failed without conflict files: {last_detail}"
                    )
                if not conflicted:
                    continue

            logger.info(
                "Resolving rebase conflicts for issue #%s (round %s): %s",
                issue_number,
                round_idx,
                ", ".join(conflicted),
            )
            prompt = self.agent_runner.build_rebase_conflict_prompt(
                issue_number=issue_number,
                base_branch=base_ref,
                conflicted_files=conflicted,
                rebase_detail=last_detail,
            )
            agent_result = self.agent_runner.run_agent(
                worktree=worktree,
                prompt=prompt,
                issue_number=issue_number,
            )
            if agent_result.timed_out or agent_result.exit_code != 0:
                self.workspaces.abort_rebase(worktree)
                raise RuntimeError(
                    f"Conflict-resolution agent failed (exit={agent_result.exit_code}, "
                    f"timeout={agent_result.timed_out}). Log: {agent_result.log_path}"
                )

            # Ensure agent did not leave markers; still stage and continue.
            self.workspaces.stage_paths(worktree, conflicted)
            remaining = self.workspaces.conflicted_files(worktree)
            if remaining:
                self.workspaces.abort_rebase(worktree)
                raise RuntimeError(
                    "Conflict markers remain after agent resolution: "
                    + ", ".join(remaining)
                )

            cont = self.workspaces.continue_rebase(worktree)
            last_detail = (cont.stderr or cont.stdout or "").strip()
            if cont.returncode == 0 and not self.workspaces.is_rebase_in_progress(
                worktree
            ):
                logger.info(
                    "Rebase conflicts resolved for issue #%s after %s round(s)",
                    issue_number,
                    round_idx,
                )
                return
            if not self.workspaces.is_rebase_in_progress(worktree) and cont.returncode != 0:
                self.workspaces.abort_rebase(worktree)
                raise RuntimeError(f"rebase --continue failed: {last_detail}")
            # else: another commit in the rebase hit conflicts — loop

        self.workspaces.abort_rebase(worktree)
        raise RuntimeError(
            f"Exceeded {max_rounds} rebase conflict-resolution rounds "
            f"for issue #{issue_number}: {last_detail}"
        )

    def reconcile_tasks(self) -> None:
        """Detect merged/closed managed PRs and rebase open branches onto base."""
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
                continue
            if pr.state == "closed":
                self.db.update_task(task.issue_number, status="closed", last_error=None)
                logger.info(
                    "Issue #%s PR #%s closed without merge; marked closed",
                    task.issue_number,
                    task.pr_number,
                )
                continue

            # Keep open PR branches current when idle (skip if a worker holds the lock).
            with self._inflight_lock:
                fut = self._inflight.get(task.issue_number)
                if fut is not None and not fut.done():
                    continue
            if task.status == "updating":
                continue
            if not self.db.try_acquire_lock(task.issue_number, self._owner):
                continue
            try:
                # Avoid re-running expensive conflict agents every poll after a hard fail.
                if task.last_error and task.last_error.startswith(
                    "auto-rebase conflict:"
                ):
                    logger.info(
                        "Skipping rebase for issue #%s until conflict error is cleared",
                        task.issue_number,
                    )
                    continue
                wt = self.workspaces.ensure_worktree(task.issue_number)
                if self.workspaces.has_changes(wt.path):
                    logger.info(
                        "Skipping rebase for issue #%s: dirty worktree",
                        task.issue_number,
                    )
                    continue
                if not self.workspaces.needs_rebase_onto_base(wt.path):
                    continue
                self._rebase_branch(
                    wt.path,
                    wt.branch,
                    push=True,
                    issue_number=task.issue_number,
                )
                self.db.update_task(task.issue_number, last_error=None)
            except Exception as exc:
                logger.exception(
                    "Auto-rebase failed for issue #%s", task.issue_number
                )
                prefix = (
                    "auto-rebase conflict:"
                    if "conflict" in str(exc).lower()
                    or "Conflict" in str(exc)
                    else "auto-rebase failed:"
                )
                try:
                    self.db.update_task(
                        task.issue_number,
                        last_error=f"{prefix} {exc}",
                    )
                except KeyError:
                    pass
            finally:
                self.db.release_lock(task.issue_number, self._owner)
