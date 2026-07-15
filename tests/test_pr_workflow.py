"""Contracts for selective pull-request CI and its stable required gates."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr.yml"
SUPPORTED_PYTHONS = {"3.10", "3.11", "3.12", "3.13", "3.14"}
TIER_1_OSES = {"ubuntu-latest", "windows-latest", "macos-latest"}


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_pr_workflow_always_starts_and_exposes_only_stable_gates() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger = text.split("permissions:", maxsplit=1)[0]
    jobs = _workflow()["jobs"]

    assert "paths:" not in trigger
    assert "paths-ignore:" not in trigger
    assert jobs["plan"]["name"] == "PR plan"
    assert jobs["fast_checks"]["name"] == "Fast checks"
    assert jobs["correctness"]["name"] == "Correctness"
    assert jobs["security_policy"]["name"] == "Security and policy"
    assert jobs["fast_checks"]["if"] == "${{ always() }}"
    assert jobs["correctness"]["if"] == "${{ always() }}"
    assert jobs["security_policy"]["if"] == "${{ always() }}"


def test_pr_plan_compares_base_and_head_project_versions() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    plan = _workflow()["jobs"]["plan"]

    assert plan["outputs"]["release_preflight"] == "${{ steps.plan.outputs.release_preflight }}"
    assert plan["outputs"]["version_change"] == "${{ steps.plan.outputs.version_change }}"
    assert 'git show "${BASE_SHA}:pyproject.toml"' in text
    assert "--base-pyproject ci-plan/base-pyproject.toml" in text
    assert "--head-pyproject ci-plan/head-pyproject.toml" in text


def test_runtime_compatibility_covers_every_supported_os_python_pair() -> None:
    matrix = _workflow()["jobs"]["compatibility"]["strategy"]["matrix"]
    pairs = set(product(matrix["operating-system"], matrix["python-version"]))
    pairs.update((entry["operating-system"], entry["python-version"]) for entry in matrix["include"])

    assert set(product(TIER_1_OSES, SUPPORTED_PYTHONS)) <= pairs


def test_only_observed_windows_outliers_are_split() -> None:
    includes = _workflow()["jobs"]["compatibility"]["strategy"]["matrix"]["include"]
    shard_counts: dict[tuple[str, str], set[int]] = {}
    for entry in includes:
        key = (entry["operating-system"], entry["python-version"])
        shard_counts.setdefault(key, set()).add(entry["shard-count"])

    assert shard_counts[("windows-latest", "3.10")] == {1}
    assert shard_counts[("windows-latest", "3.14")] == {1}
    for version in ("3.11", "3.12", "3.13"):
        assert shard_counts[("windows-latest", version)] == {2}


def test_compatibility_shard_handoff_is_portable_to_macos_bash() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    compatibility = text.split("\n  compatibility:\n", maxsplit=1)[1].split("\n  coverage:\n", maxsplit=1)[0]

    assert "mapfile" not in compatibility
    assert "xargs -0 python -m pytest" in compatibility


def test_coverage_and_pytorch_are_not_duplicated_across_matrix_cells() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    compatibility = text.split("\n  compatibility:\n", maxsplit=1)[1].split("\n  coverage:\n", maxsplit=1)[0]

    assert "--cov=albumentations" not in compatibility
    assert "Install CPU-only PyTorch" not in compatibility
    assert text.count("--cov=albumentations") == 1
    assert text.count("Install CPU-only PyTorch") == 1
    assert "tests/test_benchmark_coverage.py" in text
    assert "tests/test_serialization.py" in text


def test_version_bump_preflight_builds_publishable_bundle_in_core_profile() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["release_preflight"]
    run_text = "\n".join(str(step.get("run", "")) for step in job["steps"])
    policy_run_text = "\n".join(str(step.get("run", "")) for step in workflow["jobs"]["security_policy"]["steps"])

    assert job["if"] == "needs.plan.outputs.release_preflight == 'true'"
    assert job["permissions"] == {"contents": "read"}
    assert "dependency-group: ci-release" in WORKFLOW_PATH.read_text(encoding="utf-8")
    assert 'uv build --out-dir "${RUNNER_TEMP}/release-bundle/dist"' in run_text
    assert "${{ runner.temp }}/release-bundle/" in WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "tools.release_bundle finalize" in run_text
    assert "--core-only" in run_text
    assert "--check performance_core" in run_text
    assert "release-preflight=${{ needs.release_preflight.result }}" in policy_run_text


def test_obsolete_unconditional_pr_workflows_are_removed() -> None:
    assert not (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists()
    assert not (REPO_ROOT / ".github" / "workflows" / "legal-integrity.yml").exists()
