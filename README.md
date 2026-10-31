# trigger-agent-misc

Root Python tooling (formatting, lint, hooks) applies to code under `src/` and `tests/`. The `local-agent/` tree is a separate subproject.

Layout for ML work:

- `src/trigger_agent_misc/` — library code (e.g. optimization helpers)
- `tests/` — pytest suite
- `evals/` — scripted evaluation metrics
- `notebooks/` — exploratory demos (see `notebooks/line_search_demo.ipynb`)

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest -q
uv run python -m evals.line_search_eval
```

CI runs pre-commit on all files and pytest inside `local-agent/` (see `.github/workflows/ci.yml`).
