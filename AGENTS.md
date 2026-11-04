# Coding Agent Instructions

## General

- Follow existing project conventions.
- Inspect existing implementation before changing it.
- Prefer small, maintainable changes.
- Do not introduce unnecessary dependencies.

## Development setup

- Run `uv sync` in the repository root before editing Python files.
- Run `uv run pre-commit install` once per worktree so commit hooks are available locally.
- Before finishing, run `uv run pre-commit run --all-files` (or `./scripts/ci-precommit.sh`) and apply any hook fixes.

## Testing

- Add tests for new functionality.
- Run relevant tests before finishing.
- Report test failures honestly.

## GitHub Pages app (`src/app`)

- Prefer layout (equations, charts, tables, controls) over prose.
- Avoid redundant secondary labels and description paragraphs under headings; sidebar chapter titles duplicate the header chapter switcher.
- Keep control labels minimal (symbols such as η, β, α where they match the math).

## Git

- Work only on the assigned feature branch.
- Never push to main.
- Never merge pull requests.
- You may update CI workflow files when the assigned issue or PR feedback asks for it.
- Never change GitHub secrets, permissions, environments, or repository security settings.
