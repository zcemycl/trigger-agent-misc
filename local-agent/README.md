# Local GitHub-to-Cursor Coding Agent

Python controller (WSL2) that turns GitHub issues labelled `agent:ready` into Cursor Agent worktrees, verified commits, and pull requests.

GitHub is the UI. WSL2 is the execution environment. Cursor Agent CLI is the coding engine.

## Status

Implemented through **functional requirements 1–4**:

1. Issue detection (`agent:ready` → `agent:running`, SQLite task, worktree)
2. Cursor Agent execution (`agent -p --force --output-format text`)
3. Independent test enforcement
4. Git commit/push + `gh pr create`, label → `agent:review`

**Not yet implemented:** PR comment feedback loop (5–6), full comment_events processing, advanced lifecycle cleanup (8 partial: merge/close detection only).

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Git
- GitHub CLI (`gh`) authenticated (`gh auth login` / `gh auth refresh`)
- Cursor Agent CLI (`agent`) on `PATH`
- A local clone of the target repository (`SOURCE_REPO`)

## Setup

```bash
cd local-agent
uv sync
cp .env.example .env
# Edit .env — set GITHUB_REPO, SOURCE_REPO, WORKSPACE_ROOT, etc.
```

Do **not** commit `.env`.

### Required labels

Create these labels in the GitHub repository:

- `agent:ready`
- `agent:running`
- `agent:review`
- `agent:failed`

### Configuration

| Variable | Purpose |
|---|---|
| `GITHUB_REPO` | `OWNER/REPOSITORY` |
| `BASE_BRANCH` | Branch to branch from (default `main`) |
| `SOURCE_REPO` | Absolute path to local clone |
| `WORKSPACE_ROOT` | Where per-issue worktrees are created |
| `POLL_INTERVAL` | Seconds between poll cycles |
| `MAX_CONCURRENT_TASKS` | Parallel agent workers |
| `AGENT_COMMAND` | Cursor CLI binary (default `agent`) |
| `AGENT_TIMEOUT` | Agent/test timeout seconds |
| `TEST_COMMAND` | Controller-run test command (default `pytest -q`) |
| `TRUSTED_GITHUB_USERS` | Comma-separated logins for future feedback loop |
| `DATABASE_PATH` | SQLite file path |
| `LOG_DIR` | Agent/test/controller logs |

Paths and binaries are validated at startup.

## Run

```bash
cd local-agent
uv run python main.py
# or: uv run python main.py /path/to/.env
```

Loop:

```text
process_ready_issues()
process_pr_comments()   # stub until stage 5
reconcile_tasks()       # merge/close detection
sleep(POLL_INTERVAL)
```

Agent work runs in a thread pool so polling continues while `agent` is busy.

## Flow (requirements 1–4)

1. Poll open issues with `agent:ready`.
2. Skip if SQLite already has the issue.
3. Swap label to `agent:running`, create task + branch `agent/issue-{n}` + worktree.
4. Run Cursor Agent with the issue title/body (treated as untrusted data).
5. Run `TEST_COMMAND` independently.
6. On success: `git add -A`, commit, push, `gh pr create` with `Closes #{n}`.
7. Store PR number/URL; label → `agent:review`.
8. On agent/test failure: status `failed`, label `agent:failed`, no PR.

## Security

- Uses existing `gh` authentication; do not paste tokens into prompts.
- Restricts operations to `GITHUB_REPO`.
- Never auto-merges PRs.
- Agent subprocess does not receive `CURSOR_API_KEY` / `GITHUB_TOKEN` / `GH_TOKEN` from the controller environment scrubber.
- Issue/comment text is wrapped as untrusted content and cannot instruct the controller to merge, create extra PRs, or change security rules.
- **`--force` is not a sandbox.** It allows Cursor Agent to run shell commands unless explicitly denied. Isolate with dedicated worktrees, least-privilege OS users, and no production secrets in the worktree.

## Tests

```bash
cd local-agent
uv run pytest -q
```

CI runs the same suite on push/PR to `main` via `.github/workflows/ci.yml` (Python 3.10–3.12).

## Layout

```text
local-agent/
├── main.py
├── config.py
├── github_client.py
├── workspace.py
├── agent_runner.py
├── database.py
├── task_manager.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── README.md
├── tests/
└── workspaces/
```
