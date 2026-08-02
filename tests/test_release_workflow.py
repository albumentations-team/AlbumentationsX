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


def test_release_workflow_can_recover_an_existing_tag_from_main() -> None:
    workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True, {}))
    recovery_input = triggers["workflow_dispatch"]["inputs"]["release_tag"]
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert recovery_input["required"] is True
    assert recovery_input["type"] == "string"
    assert "github.event.release.tag_name || inputs.release_tag" in text
    assert "path: release-automation" in text
    assert "path: released-source" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert '"refs/heads/${DEFAULT_BRANCH}"' in text
    assert '--source-root "${GITHUB_WORKSPACE}/released-source"' in text
    assert "tag_name: ${{ env.RELEASE_TAG }}" in text


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
    resolve_job = jobs["resolve_release_bundle"]
    attach_job = jobs["attach_release_assets"]
    pypi_job = jobs["publish_to_pypi"]
    handoff_step = next(
        step for step in resolve_job["steps"] if step["name"] == "Hand verified bundle to delivery jobs"
    )

    assert attach_job["needs"] == "resolve_release_bundle"
    assert attach_job["permissions"] == {"actions": "read", "contents": "write"}
    assert pypi_job["needs"] == "attach_release_assets"
    assert pypi_job["permissions"] == {"actions": "read", "contents": "read", "id-token": "write"}
    assert handoff_step["with"]["include-hidden-files"] is True
    assert "verified-release-bundle" in WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "packages-dir: release-bundle/dist" in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_only_pypi_job_receives_oidc_permission() -> None:
    jobs = _workflow()["jobs"]

    assert jobs["resolve_release_bundle"]["permissions"].get("id-token") is None
    assert jobs["attach_release_assets"]["permissions"].get("id-token") is None
    assert jobs["publish_to_pypi"]["permissions"]["id-token"] == "write"


def test_release_candidate_core_profiles_do_not_require_dedicated_tensor_coverage() -> None:
    text = RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8")

    assert text.count("python -m tools.performance_budget summarize") == 2
    assert "python tools/performance_budget.py" not in text
    assert text.count("--core-only") == 2
