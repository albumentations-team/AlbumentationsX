"""Contracts for the greenfield pull-request workflow."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import yaml

from tools.ci_matrix import SUPPORTED_PYTHONS, TIER_1_OSES

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr.yml"


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_pr_workflow_exposes_direct_required_contexts_without_aggregate_aliases() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger = text.split("permissions:", maxsplit=1)[0]
    jobs = _workflow()["jobs"]

    assert "paths:" not in trigger
    assert "paths-ignore:" not in trigger
    assert jobs["plan"]["name"] == "PR plan"
    assert {
        job_id: jobs[job_id]["name"]
        for job_id in (
            "pre_commit_ruff",
            "pre_commit_ruff_format",
            "pre_commit_mypy",
            "pre_commit_pyrefly",
            "pre_commit_other",
        )
    } == {
        "pre_commit_ruff": "Pre-commit / Ruff",
        "pre_commit_ruff_format": "Pre-commit / Ruff format",
        "pre_commit_mypy": "Pre-commit / mypy",
        "pre_commit_pyrefly": "Pre-commit / Pyrefly",
        "pre_commit_other": "Pre-commit / Other hooks",
    }
    assert not {"fast_checks", "correctness", "security_policy"} & set(jobs)


def test_pr_plan_keeps_version_comparison_for_release_preflight() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    plan = _workflow()["jobs"]["plan"]

    assert plan["outputs"]["release_preflight"] == "${{ steps.plan.outputs.release_preflight }}"
    assert plan["outputs"]["version_change"] == "${{ steps.plan.outputs.version_change }}"
    assert 'git show "${BASE_SHA}:pyproject.toml"' in text
    assert 'git show "${BASE_SHA}:uv.lock"' in text


def test_compatibility_is_exactly_one_job_for_each_os_python_contract() -> None:
    matrix = _workflow()["jobs"]["compatibility"]["strategy"]["matrix"]

    assert matrix == {
        "operating-system": list(TIER_1_OSES),
        "python-version": list(SUPPORTED_PYTHONS),
    }
    assert len(set(product(matrix["operating-system"], matrix["python-version"]))) == 15


def test_test_handoffs_are_portable() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    compatibility = text.split("\n  compatibility:\n", maxsplit=1)[1].split("\n  targeted:\n", maxsplit=1)[0]
    targeted = text.split("\n  targeted:\n", maxsplit=1)[1].split("\n  pytorch:\n", maxsplit=1)[0]

    assert "mapfile" not in compatibility
    assert "xargs -0 python -m pytest" in compatibility
    assert "mapfile" not in targeted
    assert "xargs -0 python -m pytest" in targeted


def test_coverage_only_and_duplicate_product_paths_are_removed() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    for retired in ("codecov/", "CODECOV_TOKEN", "--cov=", "\n  coverage:\n", "\n  primary:\n", "\n  install_smoke:\n"):
        assert retired not in text
    assert "tests/test_benchmark_coverage.py" not in text
    assert "tests/test_serialization.py" not in text


def test_pre_commit_hooks_are_partitioned_without_direct_tool_bypasses() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    for command in (
        "pre-commit run ruff --all-files --show-diff-on-failure",
        "pre-commit run ruff-format --all-files --show-diff-on-failure",
        "pre-commit run mypy --all-files --show-diff-on-failure",
        "pre-commit run pyrefly-check --all-files --show-diff-on-failure",
        "pre-commit run --all-files --show-diff-on-failure",
    ):
        assert command in text
    assert "SKIP: ruff,ruff-format,mypy,pyrefly-check" in text
    assert "tools.quality_gate" not in text


def test_every_importing_pr_job_explicitly_selects_cpu_torch() -> None:
    expected_groups = {
        "pre_commit_other": "ci-quality",
        "compatibility": "ci-test",
        "targeted": "ci-test",
        "pytorch": "ci-test",
        "release_preflight": "ci-release",
    }

    for job_name, expected_group in expected_groups.items():
        job = _workflow()["jobs"][job_name]
        setup_step = next(step for step in job["steps"] if step.get("uses") == "./.github/actions/setup-ci")

        assert setup_step["with"]["dependency-group"] == expected_group
        assert setup_step["with"]["runtime-profile"] == "torch-cpu"


def test_release_preflight_retains_clean_install_and_strict_release_evidence() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["release_preflight"]
    upload_step = next(step for step in job["steps"] if step["name"] == "Upload publishable release bundle")
    run_text = "\n".join(str(step.get("run", "")) for step in job["steps"])

    assert job["if"] == "needs.plan.outputs.release_preflight == 'true'"
    assert "tools/install_contract.py prepare" in run_text
    assert "tools/install_contract.py smoke" in run_text
    assert upload_step["with"]["include-hidden-files"] is True
    assert "asv --config asv.conf.json continuous" in run_text
    assert "--profile release-core" in run_text
    assert "--require-comparison" in run_text
    assert "--fail-on-release-blockers" in run_text
    assert "asv --config asv.conf.json check" not in run_text
    assert "--allow-missing" not in run_text


def test_obsolete_unconditional_pr_workflows_are_removed() -> None:
    assert not (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists()
    assert not (REPO_ROOT / ".github" / "workflows" / "legal-integrity.yml").exists()
