"""Contracts for selective pull-request CI and its stable required gates."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr.yml"
SUPPORTED_PYTHONS = {"3.10", "3.11", "3.12", "3.13", "3.14"}
TIER_1_OSES = {"ubuntu-latest", "windows-latest", "macos-latest"}


def _workflow() -> dict:
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


def test_coverage_and_pytorch_are_not_duplicated_across_matrix_cells() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    compatibility = text.split("\n  compatibility:\n", maxsplit=1)[1].split("\n  coverage:\n", maxsplit=1)[0]

    assert "--cov=albumentations" not in compatibility
    assert "Install CPU-only PyTorch" not in compatibility
    assert text.count("--cov=albumentations") == 1
    assert text.count("Install CPU-only PyTorch") == 1
    assert "tests/test_benchmark_coverage.py" in text
    assert "tests/test_serialization.py" in text


def test_obsolete_unconditional_pr_workflows_are_removed() -> None:
    assert not (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists()
    assert not (REPO_ROOT / ".github" / "workflows" / "legal-integrity.yml").exists()
