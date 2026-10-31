"""Ensure pre-commit and lint tooling stay wired for the repository root."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_precommit_config_declares_required_hooks() -> None:
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for hook_id in ("black", "isort", "flake8"):
        assert f"id: {hook_id}" in text


def test_black_excludes_local_agent() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "local-agent" in pyproject


def test_flake8_hook_loads_pyproject_via_plugin() -> None:
    """Flake8-pyproject reads [tool.flake8]; do not pass invalid flake8 CLI flags."""
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "flake8-pyproject" in text.lower()
    assert "--extend-config" not in text
