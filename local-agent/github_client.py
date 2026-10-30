"""GitHub operations via the authenticated GitHub CLI (`gh`)."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from config import Config

logger = logging.getLogger(__name__)


class GitHubError(RuntimeError):
    """Raised when a GitHub CLI operation fails."""


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    author: str
    state: str
    url: str


@dataclass
class PullRequest:
    number: int
    title: str
    body: str
    state: str
    url: str
    merged: bool
    head_ref: str
    base_ref: str


class GitHubClient:
    """Thin wrapper around `gh` for the configured repository only."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.owner = config.owner
        self.repo = config.repo

    def _run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["gh", *args]
        logger.debug("Running: %s", " ".join(shlex.quote(a) for a in cmd))
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                input=input_text,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubError(f"gh timed out: {' '.join(args)}") from exc
        if check and result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise GitHubError(f"gh {' '.join(args)} failed: {stderr}")
        return result

    def _api(
        self,
        path: str,
        *,
        method: str = "GET",
        fields: Optional[dict[str, Any]] = None,
        paginate: bool = False,
    ) -> Any:
        args = ["api", path, "--method", method]
        if paginate:
            args.append("--paginate")
        if fields:
            for key, value in fields.items():
                if isinstance(value, (dict, list)):
                    args.extend(["-f", f"{key}={json.dumps(value)}"])
                elif value is None:
                    continue
                else:
                    args.extend(["-f", f"{key}={value}"])
        result = self._run(args)
        text = result.stdout.strip()
        if not text:
            return None
        # --paginate may concatenate multiple JSON arrays/objects.
        if paginate and text.startswith("["):
            items: list[Any] = []
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(text):
                while idx < len(text) and text[idx].isspace():
                    idx += 1
                if idx >= len(text):
                    break
                chunk, offset = decoder.raw_decode(text[idx:])
                idx += offset
                if isinstance(chunk, list):
                    items.extend(chunk)
                else:
                    items.append(chunk)
            return items
        return json.loads(text)

    def ensure_repo_match(self, full_name: str) -> None:
        if full_name.lower() != self.config.repo_slug.lower():
            raise GitHubError(
                f"Refusing to operate on {full_name}; configured for {self.config.repo_slug}"
            )

    def get_authenticated_user(self) -> str:
        data = self._api("user")
        return str(data["login"])

    def list_ready_issues(self, label: str | None = None) -> list[Issue]:
        label = label or self.config.ready_label
        path = (
            f"repos/{self.owner}/{self.repo}/issues"
            f"?state=open&labels={label}&per_page=100"
        )
        data = self._api(path, paginate=True) or []
        issues: list[Issue] = []
        for item in data:
            # Pull requests also appear in the issues API; skip them.
            if "pull_request" in item:
                continue
            issues.append(self._parse_issue(item))
        return issues

    def get_issue(self, number: int) -> Issue:
        data = self._api(f"repos/{self.owner}/{self.repo}/issues/{number}")
        return self._parse_issue(data)

    def _parse_issue(self, data: dict[str, Any]) -> Issue:
        labels = [lbl["name"] for lbl in data.get("labels", [])]
        user = data.get("user") or {}
        return Issue(
            number=int(data["number"]),
            title=data.get("title") or "",
            body=data.get("body") or "",
            labels=labels,
            author=str(user.get("login") or ""),
            state=data.get("state") or "open",
            url=data.get("html_url") or "",
        )

    def set_issue_labels(self, number: int, labels: list[str]) -> None:
        self._api(
            f"repos/{self.owner}/{self.repo}/issues/{number}/labels",
            method="PUT",
            fields={"labels": labels},
        )

    def add_issue_labels(self, number: int, labels: list[str]) -> None:
        # gh api -f does not send JSON arrays cleanly for all endpoints;
        # use raw input via subprocess for label add.
        payload = json.dumps({"labels": labels})
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{self.owner}/{self.repo}/issues/{number}/labels",
                "--method",
                "POST",
                "--input",
                "-",
            ],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            raise GitHubError(
                f"Failed to add labels {labels} to issue #{number}: "
                f"{(result.stderr or result.stdout).strip()}"
            )

    def remove_issue_label(self, number: int, label: str) -> None:
        from urllib.parse import quote

        encoded = quote(label, safe="")
        result = self._run(
            [
                "api",
                f"repos/{self.owner}/{self.repo}/issues/{number}/labels/{encoded}",
                "--method",
                "DELETE",
            ],
            check=False,
        )
        # 404 means label already absent — treat as success.
        if result.returncode != 0 and "404" not in (result.stderr or ""):
            # gh may put HTTP status in stderr differently; accept 404 via stdout too.
            combined = f"{result.stderr}\n{result.stdout}"
            if "Not Found" not in combined and "404" not in combined:
                raise GitHubError(
                    f"Failed to remove label {label!r} from issue #{number}: "
                    f"{combined.strip()}"
                )

    def replace_agent_label(
        self,
        number: int,
        *,
        remove: str,
        add: str,
        current_labels: Optional[list[str]] = None,
    ) -> None:
        """Atomically-ish swap agent labels while preserving other labels."""
        if current_labels is None:
            current_labels = self.get_issue(number).labels
        next_labels = [lbl for lbl in current_labels if lbl != remove and lbl != add]
        if add not in next_labels:
            next_labels.append(add)
        payload = json.dumps({"labels": next_labels})
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{self.owner}/{self.repo}/issues/{number}",
                "--method",
                "PATCH",
                "--input",
                "-",
            ],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            raise GitHubError(
                f"Failed to update labels on issue #{number}: "
                f"{(result.stderr or result.stdout).strip()}"
            )

    def create_pull_request(
        self,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequest:
        result = self._run(
            [
                "pr",
                "create",
                "--repo",
                self.config.repo_slug,
                "--head",
                head,
                "--base",
                base,
                "--title",
                title,
                "--body",
                body,
            ]
        )
        url = result.stdout.strip().splitlines()[-1].strip()
        # Resolve PR number from URL or API.
        pr_data = self._api(
            f"repos/{self.owner}/{self.repo}/pulls"
            f"?head={self.owner}:{head}&state=open&per_page=1"
        )
        if not pr_data:
            # Fallback: parse number from URL .../pull/123
            number = int(url.rstrip("/").split("/")[-1])
            return self.get_pull_request(number)
        return self._parse_pr(pr_data[0])

    def get_pull_request(self, number: int) -> PullRequest:
        data = self._api(f"repos/{self.owner}/{self.repo}/pulls/{number}")
        return self._parse_pr(data)

    def _parse_pr(self, data: dict[str, Any]) -> PullRequest:
        return PullRequest(
            number=int(data["number"]),
            title=data.get("title") or "",
            body=data.get("body") or "",
            state=data.get("state") or "open",
            url=data.get("html_url") or "",
            merged=bool(data.get("merged")),
            head_ref=(data.get("head") or {}).get("ref") or "",
            base_ref=(data.get("base") or {}).get("ref") or "",
        )

    # --- Comment endpoints (used from stage 5+; implemented for completeness) ---

    def list_issue_comments(self, pr_number: int) -> list[dict[str, Any]]:
        path = (
            f"repos/{self.owner}/{self.repo}/issues/{pr_number}/comments?per_page=100"
        )
        return list(self._api(path, paginate=True) or [])

    def list_review_comments(self, pr_number: int) -> list[dict[str, Any]]:
        path = (
            f"repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments?per_page=100"
        )
        return list(self._api(path, paginate=True) or [])

    def list_reviews(self, pr_number: int) -> list[dict[str, Any]]:
        path = (
            f"repos/{self.owner}/{self.repo}/pulls/{pr_number}/reviews?per_page=100"
        )
        return list(self._api(path, paginate=True) or [])
