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
    assert "branches: [main]" in workflow
    assert "workflow_dispatch" not in workflow
    assert "refs/heads/main" in workflow
