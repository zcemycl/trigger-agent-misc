#!/usr/bin/env bash
# Run the same checks expected in CI (pre-commit on all tracked files).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv sync --frozen
uv run pre-commit run --all-files
