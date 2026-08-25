"""Contracts for the explicit Torch runtime selection in CI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from tools.ci_matrix import CI_FOUNDATION_SHA, TORCH_RUNTIME_JOBS, _check_ci_dependency_groups

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_ACTION = REPO_ROOT / ".github" / "actions" / "setup-ci" / "action.yml"


def _jobs(path: Path) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]


def test_setup_action_composes_tools_with_an_explicit_runtime_profile() -> None:
    action = yaml.safe_load(SETUP_ACTION.read_text(encoding="utf-8"))
    text = SETUP_ACTION.read_text(encoding="utf-8")

    assert action["inputs"]["runtime-profile"]["default"] == "none"
    assert "--group ci-torch-cpu" in text
    assert "cache-suffix: ${{ inputs.dependency-group }}-${{ inputs.runtime-profile }}" in text
    assert f"albumentations-team/ci-foundation/actions/setup-python-uv@{CI_FOUNDATION_SHA}" in text
    assert f"albumentations-team/ci-foundation/actions/torch-cpu@{CI_FOUNDATION_SHA}" in text
    assert "mode: verify" in text
    assert "CI_DEPENDENCY_GROUP: ${{ inputs.dependency-group }}" in text
    assert "ALBU_CI_RUNTIME_PROFILE=${CI_RUNTIME_PROFILE}" in text


def test_importing_jobs_explicitly_select_the_cpu_torch_profile() -> None:
    for workflow_path, expected_jobs in TORCH_RUNTIME_JOBS.items():
        jobs = _jobs(workflow_path)
        for job_name, dependency_group in expected_jobs.items():
            setup_step = next(
                step for step in jobs[job_name]["steps"] if step.get("uses") == "./.github/actions/setup-ci"
            )

            assert setup_step["with"]["dependency-group"] == dependency_group
            assert setup_step["with"]["runtime-profile"] == "torch-cpu"


def test_workflows_use_only_declared_groups_and_no_inline_torch_installs() -> None:
    for workflow_path in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        text = workflow_path.read_text(encoding="utf-8")
        for job in _jobs(workflow_path).values():
            for step in job.get("steps", []):
                if step.get("uses") != "./.github/actions/setup-ci":
                    continue
                assert step["with"]["dependency-group"] in {
                    "ci-benchmark",
                    "ci-package",
                    "ci-quality",
                    "ci-release",
                    "ci-security",
                    "ci-test",
                    "ci-types",
                }

        assert re.search(r"(?:uv |python -m )?pip install[^\n]*torch", text, flags=re.IGNORECASE) is None


def test_dependency_group_check_reports_broken_include_reference() -> None:
    issues = _check_ci_dependency_groups({"ci-release": [{"include-group": "ci-benhcmark"}]})

    assert "refers to non-existent group 'ci-benhcmark'" in "\n".join(issues)
