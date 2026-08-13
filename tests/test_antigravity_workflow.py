"""Contracts for the thin local Antigravity reusable-workflow caller."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 uses the locked backport.
    import tomli as tomllib

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "antigravity-pr-checks.yml"
POLICY = Path(__file__).resolve().parents[1] / ".github" / "ci-foundation" / "antigravity.toml"
INSTRUCTIONS = Path(__file__).resolve().parents[1] / ".github" / "ci-foundation" / "antigravity-review.md"
FOUNDATION_SHA = "6b9045dbea58026a1e8f96b0392c411934a27199"


def test_local_caller_has_only_trigger_security_and_project_configuration() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Antigravity PR Checks" in workflow
    assert "on: # zizmor: ignore[dangerous-triggers]" in workflow
    assert "pull_request_target:" in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "github.event.pull_request.draft == false" in workflow
    assert "pull-requests: write" in workflow
    assert f"albumentations-team/ci-foundation/.github/workflows/antigravity-review.yml@{FOUNDATION_SHA}" in workflow
    assert "policy-path: .github/ci-foundation/antigravity.toml" in workflow
    assert "gh pr diff" not in workflow
    assert "run-gemini-cli" not in workflow
    assert "gh pr review" not in workflow


def test_local_policy_is_data_only_and_points_to_trusted_instructions() -> None:
    policy = tomllib.loads(POLICY.read_text(encoding="utf-8"))

    assert policy["paths"]["include"]
    assert policy["review"]["instructions"] == ".github/ci-foundation/antigravity-review.md"
    assert INSTRUCTIONS.is_file()
