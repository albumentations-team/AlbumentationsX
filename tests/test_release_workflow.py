"""Contracts for delivery-only release publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "upload_to_pypi.yml"
RELEASE_CANDIDATE_PATH = REPO_ROOT / ".github" / "workflows" / "release-candidate.yml"


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_release_workflow_resolves_bundle_by_tag_version_and_source_digest() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["resolve_release_bundle"]
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert job["permissions"] == {"actions": "read", "contents": "read"}
    assert "tools.release_bundle metadata" in text
    assert "--expected-tag" in text
    assert "tools.release_bundle resolve" in text
    assert "steps.metadata.outputs.artifact_name" in text
    assert "steps.resolve.outputs.artifact_id" in text
    assert "steps.resolve.outputs.run_id" in text
    assert "tools.release_bundle verify" in text
    assert "steps.metadata.outputs.source_digest" in text


def test_release_workflow_contains_only_identity_and_delivery_steps() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    forbidden_commands = (
        "uv build",
        "twine check",
        "verify_legal_integrity",
        "pytest",
        "benchmark_coverage",
        "performance_budget",
        "pip-audit",
        "zizmor",
        "cyclonedx-py",
        "generate_correctness_report",
    )

    for command in forbidden_commands:
        assert command not in text


def test_release_workflow_delivers_one_verified_bundle_to_both_destinations() -> None:
    jobs = _workflow()["jobs"]
    attach_job = jobs["attach_release_assets"]
    pypi_job = jobs["publish_to_pypi"]

    assert attach_job["needs"] == "resolve_release_bundle"
    assert attach_job["permissions"] == {"actions": "read", "contents": "write"}
    assert pypi_job["needs"] == "attach_release_assets"
    assert pypi_job["permissions"] == {"actions": "read", "contents": "read", "id-token": "write"}
    assert "verified-release-bundle" in WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "packages-dir: release-bundle/dist" in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_only_pypi_job_receives_oidc_permission() -> None:
    jobs = _workflow()["jobs"]

    assert jobs["resolve_release_bundle"]["permissions"].get("id-token") is None
    assert jobs["attach_release_assets"]["permissions"].get("id-token") is None
    assert jobs["publish_to_pypi"]["permissions"]["id-token"] == "write"


def test_release_candidate_core_profiles_do_not_require_optional_pytorch_coverage() -> None:
    text = RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8")

    assert text.count("tools/performance_budget.py summarize") == 2
    assert text.count("--core-only") == 2
