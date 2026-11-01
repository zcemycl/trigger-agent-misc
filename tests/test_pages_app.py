"""Static GitHub Pages app under src/app (React, TypeScript, Tailwind)."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "src" / "app"


def test_app_directory_layout() -> None:
    assert APP_ROOT.is_dir()
    assert (APP_ROOT / "package.json").is_file()
    assert (APP_ROOT / "vite.config.ts").is_file()
    assert (APP_ROOT / "src" / "App.tsx").is_file()
    assert (APP_ROOT / "src" / "main.tsx").is_file()
    assert (APP_ROOT / "index.html").is_file()


def test_app_uses_react_typescript_and_tailwind() -> None:
    pkg = json.loads((APP_ROOT / "package.json").read_text(encoding="utf-8"))
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert "react" in deps
    assert "typescript" in deps
    assert "tailwindcss" in deps

    app_tsx = (APP_ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "export default function App" in app_tsx

    css = (APP_ROOT / "src" / "index.css").read_text(encoding="utf-8")
    assert "tailwindcss" in css


def test_pages_workflow_deploys_gh_pages_branch() -> None:
    workflow = (REPO_ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "working-directory: src/app" in workflow
    assert "npm run build" in workflow
    assert "publish_branch: gh-pages" in workflow
    assert "peaceiris/actions-gh-pages" in workflow
    assert "workflow_dispatch:" in workflow
    assert "branches: [main]" not in workflow
    assert '"**"' in workflow


def test_ci_workflow_runs_node_checks() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "node-app:" in workflow
    assert "working-directory: src/app" in workflow
    assert "npm run format:check" in workflow
    assert "npm run typecheck" in workflow
    assert "npm test" in workflow


def test_pre_commit_includes_prettier_for_app() -> None:
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "mirrors-prettier" in config
    assert "files: ^src/app/" in config


def test_app_has_format_and_test_scripts() -> None:
    pkg = json.loads((APP_ROOT / "package.json").read_text(encoding="utf-8"))
    scripts = pkg.get("scripts", {})
    assert scripts.get("format:check") == "prettier --check ."
    assert scripts.get("test") == "vitest run"
    assert (APP_ROOT / "src" / "App.test.tsx").is_file()


def test_app_includes_interactive_line_search_demos() -> None:
    concepts = (APP_ROOT / "src" / "components" / "LineSearchConcepts.tsx").read_text(
        encoding="utf-8"
    )
    for title in (
        "Exact line search",
        "Backtracking line search",
        "Armijo condition",
        "Wolfe conditions",
        "Strong Wolfe conditions",
        "Goldstein conditions",
    ):
        assert title in concepts
    assert (APP_ROOT / "src" / "lineSearch.ts").is_file()
    assert "MathBlock" in concepts


def test_app_uses_sidebar_for_line_search_chapters() -> None:
    concepts = (APP_ROOT / "src" / "components" / "LineSearchConcepts.tsx").read_text(
        encoding="utf-8"
    )
    assert "LINE_SEARCH_CHAPTER" in concepts
    assert "Line search" in concepts
    assert "LINE_SEARCH_PAGES" in concepts
    assert "aria-label={`${LINE_SEARCH_CHAPTER} concepts`}" in concepts


def test_app_includes_interactive_gradient_method_demos() -> None:
    concepts = (
        APP_ROOT / "src" / "components" / "GradientMethodConcepts.tsx"
    ).read_text(encoding="utf-8")
    for title in (
        "Gradient descent (GD)",
        "Stochastic gradient descent (SGD)",
        "Mini-batch SGD",
        "Momentum",
        "Nesterov accelerated gradient",
        "Heavy-ball method",
    ):
        assert title in concepts
    assert (APP_ROOT / "src" / "gradientMethod.ts").is_file()
    assert "GRADIENT_METHOD_PAGES" in concepts


def test_gradient_method_pages_include_where_explanations() -> None:
    concepts = (
        APP_ROOT / "src" / "components" / "GradientMethodConcepts.tsx"
    ).read_text(encoding="utf-8")
    assert "EquationWithWhere" in concepts
    assert (APP_ROOT / "src" / "components" / "EquationWithWhere.tsx").is_file()


def test_gradient_method_pages_include_step_by_step_tables() -> None:
    concepts = (
        APP_ROOT / "src" / "components" / "GradientMethodConcepts.tsx"
    ).read_text(encoding="utf-8")
    assert "OptimizationStepTable" in concepts
    assert concepts.count("OptimizationStepTable") >= 6
    assert (APP_ROOT / "src" / "components" / "OptimizationStepTable.tsx").is_file()
    table = (APP_ROOT / "src" / "components" / "OptimizationStepTable.tsx").read_text(
        encoding="utf-8"
    )
    assert "vector" in table.lower()
    assert "formatVector" in table


def test_agents_md_documents_pages_app_ui_guidance() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "src/app" in agents
    assert "secondary labels" in agents.lower() or "redundant" in agents.lower()


def test_app_uses_chapter_nav_for_line_search_and_gradient_methods() -> None:
    app_tsx = (APP_ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "Gradient methods" in app_tsx
    assert "Line search" in app_tsx
    assert "GradientMethodConcepts" in app_tsx


def test_app_renders_latex_equations_for_line_search() -> None:
    pkg = json.loads((APP_ROOT / "package.json").read_text(encoding="utf-8"))
    assert "katex" in pkg.get("dependencies", {})
    assert (APP_ROOT / "src" / "components" / "MathBlock.tsx").is_file()
