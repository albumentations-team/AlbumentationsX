"""Tests for stable fail-closed CI gate aggregation."""

from __future__ import annotations

from tools.ci_gate import _parse_results, validate_gate


def test_gate_passes_when_every_selected_job_succeeds() -> None:
    plan = {"gates": {"fast": ["lint", "mypy"]}}
    results = {"lint": "success", "mypy": "success", "markdown": "skipped"}

    assert validate_gate(plan, "fast", results) == []


def test_gate_fails_when_selected_job_is_skipped() -> None:
    plan = {"gates": {"correctness": ["compatibility"]}}

    assert validate_gate(plan, "correctness", {"compatibility": "skipped"}) == [
        "Selected job 'compatibility' finished with 'skipped'; expected 'success'.",
    ]


def test_gate_fails_when_selected_job_is_missing() -> None:
    plan = {"gates": {"policy": ["legal"]}}

    assert validate_gate(plan, "policy", {}) == [
        "Selected job 'legal' finished with 'missing'; expected 'success'.",
    ]


def test_gate_fails_on_unselected_failure() -> None:
    plan = {"gates": {"fast": []}}

    assert validate_gate(plan, "fast", {"lint": "failure"}) == [
        "Unselected job 'lint' unexpectedly finished with 'failure'.",
    ]


def test_gate_rejects_invalid_plan_shape() -> None:
    assert validate_gate({}, "fast", {}) == ["CI plan does not contain a gates mapping."]


def test_parse_results_accepts_github_result_pairs() -> None:
    assert _parse_results(["lint=success", "mypy=skipped"]) == {"lint": "success", "mypy": "skipped"}
