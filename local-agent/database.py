"""SQLite persistence for tasks and comment events."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


TASK_STATUSES = (
    "queued",
    "running",
    "pr_open",
    "review_pending",
    "updating",
    "failed",
    "merged",
    "closed",
)

COMMENT_STATUSES = (
    "pending",
    "processing",
    "completed",
    "failed",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Task:
    issue_number: int
    branch: str
    workspace: str
    pr_number: Optional[int]
    pr_url: Optional[str]
    status: str
    created_at: str
    updated_at: str
    last_error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        return cls(
            issue_number=row["issue_number"],
            branch=row["branch"],
            workspace=row["workspace"],
            pr_number=row["pr_number"],
            pr_url=row["pr_url"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_error=row["last_error"],
        )


@dataclass
class CommentEvent:
    comment_id: str
    pr_number: int
    comment_type: str
    author: str
    body: str
    status: str
    created_at: str
    processed_at: Optional[str]
    error: Optional[str]
    path: Optional[str] = None
    line: Optional[int] = None
    diff_hunk: Optional[str] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CommentEvent":
        keys = set(row.keys())
        return cls(
            comment_id=row["comment_id"],
            pr_number=row["pr_number"],
            comment_type=row["comment_type"],
            author=row["author"],
            body=row["body"],
            status=row["status"],
            created_at=row["created_at"],
            processed_at=row["processed_at"],
            error=row["error"],
            path=row["path"] if "path" in keys else None,
            line=row["line"] if "line" in keys else None,
            diff_hunk=row["diff_hunk"] if "diff_hunk" in keys else None,
        )


class Database:
    """Thread-safe SQLite helper with task locking."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    issue_number INTEGER PRIMARY KEY,
                    branch TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    pr_number INTEGER,
                    pr_url TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS comment_events (
                    comment_id TEXT PRIMARY KEY,
                    pr_number INTEGER NOT NULL,
                    comment_type TEXT NOT NULL,
                    author TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    error TEXT,
                    path TEXT,
                    line INTEGER,
                    diff_hunk TEXT
                );

                CREATE TABLE IF NOT EXISTS task_locks (
                    issue_number INTEGER PRIMARY KEY,
                    locked_at TEXT NOT NULL,
                    owner TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_comment_events_status ON comment_events(status);
                CREATE INDEX IF NOT EXISTS idx_comment_events_pr ON comment_events(pr_number);
                """
            )

    def get_task(self, issue_number: int) -> Optional[Task]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE issue_number = ?",
                (issue_number,),
            ).fetchone()
            return Task.from_row(row) if row else None

    def list_tasks(self, statuses: Optional[list[str]] = None) -> list[Task]:
        with self.transaction() as conn:
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                rows = conn.execute(
                    f"SELECT * FROM tasks WHERE status IN ({placeholders}) ORDER BY issue_number",
                    tuple(statuses),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY issue_number"
                ).fetchall()
            return [Task.from_row(r) for r in rows]

    def create_task(
        self,
        *,
        issue_number: int,
        branch: str,
        workspace: str,
        status: str = "queued",
    ) -> Task:
        if status not in TASK_STATUSES:
            raise ValueError(f"Invalid task status: {status}")
        now = _utc_now()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    issue_number, branch, workspace, pr_number, pr_url,
                    status, created_at, updated_at, last_error
                ) VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, NULL)
                """,
                (issue_number, branch, workspace, status, now, now),
            )
        task = self.get_task(issue_number)
        assert task is not None
        return task

    def update_task(self, issue_number: int, **fields: Any) -> Task:
        allowed = {
            "branch",
            "workspace",
            "pr_number",
            "pr_url",
            "status",
            "last_error",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            task = self.get_task(issue_number)
            if task is None:
                raise KeyError(f"Task {issue_number} not found")
            return task
        if "status" in updates and updates["status"] not in TASK_STATUSES:
            raise ValueError(f"Invalid task status: {updates['status']}")

        updates["updated_at"] = _utc_now()
        columns = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [issue_number]
        with self.transaction() as conn:
            cur = conn.execute(
                f"UPDATE tasks SET {columns} WHERE issue_number = ?",
                values,
            )
            if cur.rowcount == 0:
                raise KeyError(f"Task {issue_number} not found")
        task = self.get_task(issue_number)
        assert task is not None
        return task

    def task_exists(self, issue_number: int) -> bool:
        return self.get_task(issue_number) is not None

    def try_acquire_lock(self, issue_number: int, owner: str) -> bool:
        now = _utc_now()
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT owner FROM task_locks WHERE issue_number = ?",
                (issue_number,),
            ).fetchone()
            if existing is not None:
                return False
            conn.execute(
                "INSERT INTO task_locks (issue_number, locked_at, owner) VALUES (?, ?, ?)",
                (issue_number, now, owner),
            )
            return True

    def release_lock(self, issue_number: int, owner: str | None = None) -> None:
        with self.transaction() as conn:
            if owner is None:
                conn.execute(
                    "DELETE FROM task_locks WHERE issue_number = ?",
                    (issue_number,),
                )
            else:
                conn.execute(
                    "DELETE FROM task_locks WHERE issue_number = ? AND owner = ?",
                    (issue_number, owner),
                )

    def insert_comment_event(
        self,
        *,
        comment_id: str,
        pr_number: int,
        comment_type: str,
        author: str,
        body: str,
        path: str | None = None,
        line: int | None = None,
        diff_hunk: str | None = None,
        status: str = "pending",
    ) -> bool:
        """Insert a comment event. Returns False if the id already exists."""
        if status not in COMMENT_STATUSES:
            raise ValueError(f"Invalid comment status: {status}")
        now = _utc_now()
        with self.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO comment_events (
                        comment_id, pr_number, comment_type, author, body,
                        status, created_at, processed_at, error, path, line, diff_hunk
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                    """,
                    (
                        comment_id,
                        pr_number,
                        comment_type,
                        author,
                        body,
                        status,
                        now,
                        path,
                        line,
                        diff_hunk,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def list_comment_events(
        self,
        *,
        statuses: Optional[list[str]] = None,
        pr_number: Optional[int] = None,
    ) -> list[CommentEvent]:
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if pr_number is not None:
            clauses.append("pr_number = ?")
            params.append(pr_number)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.transaction() as conn:
            rows = conn.execute(
                f"SELECT * FROM comment_events {where} ORDER BY created_at",
                tuple(params),
            ).fetchall()
            return [CommentEvent.from_row(r) for r in rows]

    def update_comment_event(self, comment_id: str, **fields: Any) -> CommentEvent:
        allowed = {"status", "processed_at", "error"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if "status" in updates and updates["status"] not in COMMENT_STATUSES:
            raise ValueError(f"Invalid comment status: {updates['status']}")
        if not updates:
            events = [e for e in self.list_comment_events() if e.comment_id == comment_id]
            if not events:
                raise KeyError(f"Comment event {comment_id} not found")
            return events[0]
        columns = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [comment_id]
        with self.transaction() as conn:
            cur = conn.execute(
                f"UPDATE comment_events SET {columns} WHERE comment_id = ?",
                values,
            )
            if cur.rowcount == 0:
                raise KeyError(f"Comment event {comment_id} not found")
            row = conn.execute(
                "SELECT * FROM comment_events WHERE comment_id = ?",
                (comment_id,),
            ).fetchone()
            return CommentEvent.from_row(row)
