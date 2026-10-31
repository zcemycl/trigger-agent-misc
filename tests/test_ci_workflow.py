"""Guardrails for root CI workflow configuration."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_local_agent_pytest_matrix_is_python_311_only() -> None:
    text = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("python-version:")
    ]
    assert len(matrix_lines) == 1
    assert matrix_lines[0] == 'python-version: ["3.11"]'
