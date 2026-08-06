"""Contracts for path-scoped CodeQL workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tools import ci_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEQL_WORKFLOWS = {
    "codeql-python.yml": [
        "**/*.py",
        "**/*.pyi",
        ".github/codeql/**",
        ".github/workflows/codeql-python.yml",
    ],
    "codeql-actions.yml": [
        "**/*",
    ],
}


def _load_workflow(filename: str) -> dict[str, Any]:
    """Load a workflow while accounting for YAML 1.1 coercing the `on` key to a boolean."""
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def test_codeql_triggers_are_language_scoped_and_refresh_default_branch_alerts() -> None:
    """Run each language only for relevant PR/main changes, plus weekly/manual scans."""
    for filename, expected_paths in CODEQL_WORKFLOWS.items():
        workflow = _load_workflow(filename)
        triggers = workflow.get(True)
        if triggers is None:
            triggers = workflow["on"]

        assert triggers["pull_request"]["branches"] == ["main"]
        assert triggers["pull_request"]["paths"] == expected_paths
        assert triggers["push"]["branches"] == ["main"]
        assert triggers["push"]["paths"] == expected_paths
        assert "schedule" in triggers
        assert "workflow_dispatch" in triggers


def test_ci_matrix_rejects_codeql_workflow_without_push_trigger(tmp_path, monkeypatch) -> None:
    """Prevent path-scoped CodeQL workflows from silently losing default-branch analysis."""
    workflow_path = tmp_path / "codeql-python.yml"
    workflow_path.write_text(
        """\
name: CodeQL Python
on:
  pull_request:
    branches: [main]
    paths:
      - "**/*.py"
      - "**/*.pyi"
      - ".github/codeql/**"
      - ".github/workflows/codeql-python.yml"
  schedule:
    - cron: "7 3 * * 0"
  workflow_dispatch:
jobs: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ci_matrix, "_workflow_files", lambda: (workflow_path,))
    monkeypatch.setattr(
        ci_matrix,
        "CODEQL_WORKFLOW_PATHS",
        {workflow_path: ci_matrix.CODEQL_WORKFLOW_PATHS[ci_matrix.CODEQL_PYTHON_WORKFLOW]},
    )

    issues = ci_matrix._check_workflow_push_triggers()

    assert f"{workflow_path} CodeQL workflow is missing 'push' trigger" in issues
