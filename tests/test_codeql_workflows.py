"""Contracts for path-scoped CodeQL workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEQL_WORKFLOWS = {
    "codeql-python.yml": [
        "**/*.py",
        "**/*.pyi",
        ".github/codeql/**",
        ".github/workflows/codeql-python.yml",
    ],
    "codeql-actions.yml": [
        ".github/**",
        "**/action.yml",
        "**/action.yaml",
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
