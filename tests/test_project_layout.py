"""Repository layout expected for ML research work."""

import json
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


def test_line_search_notebook_includes_optimization_math() -> None:
    """Notebook explains line search with LaTeX (Armijo, restriction, Rosenbrock)."""
    nb_path = REPO_ROOT / "notebooks" / "line_search_demo.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    markdown = "".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert r"\phi(\alpha)" in markdown
    assert "c_1" in markdown
    assert r"\nabla f" in markdown
    assert "(1 - x)^2 + 100" in markdown


def test_line_search_notebook_retains_viewable_outputs() -> None:
    """Notebook is checked in with outputs for GitHub/Jupyter viewers."""
    nb_path = REPO_ROOT / "notebooks" / "line_search_demo.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 4

    cells_with_outputs = [c for c in code_cells if c.get("outputs")]
    assert len(cells_with_outputs) == 3

    plot_cell = code_cells[2]
    assert any(
        out.get("output_type") == "display_data" and "image/png" in out.get("data", {})
        for out in plot_cell["outputs"]
    )
