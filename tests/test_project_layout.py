"""Repository layout expected for ML research work."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_research_directories_exist() -> None:
    assert (REPO_ROOT / "evals").is_dir()
    assert (REPO_ROOT / "notebooks").is_dir()


def test_line_search_notebook_present() -> None:
    nb = REPO_ROOT / "notebooks" / "line_search_demo.ipynb"
    assert nb.is_file()
    text = nb.read_text(encoding="utf-8")
    assert "backtracking_armijo" in text
