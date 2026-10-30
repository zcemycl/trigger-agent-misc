# trigger-agent-misc

Root Python tooling (formatting, lint, hooks) applies to code under `src/` and `tests/`. The `local-agent/` tree is a separate subproject.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest -q
```

CI runs pre-commit on all files and pytest inside `local-agent/` (see `.github/workflows/ci.yml`).
