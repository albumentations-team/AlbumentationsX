"""Contracts for the stable pull-request routing gate."""

from tools.ci_gate import ALWAYS_RUN_CHECKS, ROUTED_CHECKS, _read_environment, validate_results
from tools.ci_plan import CHECK_NAMES


def test_gate_check_names_match_the_router() -> None:
    assert ROUTED_CHECKS == CHECK_NAMES


def _results() -> dict[str, str]:
    return {
        **dict.fromkeys(ALWAYS_RUN_CHECKS, "success"),
        **dict.fromkeys(ROUTED_CHECKS, "skipped"),
    }


def test_version_only_release_accepts_only_release_preflight() -> None:
    selected = dict.fromkeys(ROUTED_CHECKS, False)
    selected["release_preflight"] = True
    results = _results()
    results["release_preflight"] = "success"

    assert validate_results(plan_result="success", selected=selected, results=results) == ()


def test_runtime_plan_requires_selected_matrix_and_pytorch_jobs() -> None:
    selected = dict.fromkeys(ROUTED_CHECKS, False)
    selected["compatibility"] = True
    selected["pytorch"] = True
    results = _results()
    results["compatibility"] = "success"
    results["pytorch"] = "success"

    assert validate_results(plan_result="success", selected=selected, results=results) == ()


def test_selected_failure_is_reported() -> None:
    selected = dict.fromkeys(ROUTED_CHECKS, False)
    selected["compatibility"] = True
    results = _results()
    results["compatibility"] = "failure"

    assert validate_results(plan_result="success", selected=selected, results=results) == (
        "compatibility: expected success, got 'failure'",
    )


def test_unselected_job_must_not_run() -> None:
    selected = dict.fromkeys(ROUTED_CHECKS, False)
    results = _results()
    results["compatibility"] = "success"

    assert validate_results(plan_result="success", selected=selected, results=results) == (
        "compatibility: expected skipped, got 'success'",
    )


def test_failed_plan_is_not_masked_by_skipped_jobs() -> None:
    selected = dict.fromkeys(ROUTED_CHECKS, False)

    assert validate_results(plan_result="failure", selected=selected, results=_results()) == (
        "PR plan finished with 'failure'; routed checks are not trustworthy",
    )


def test_failed_plan_does_not_parse_missing_outputs(monkeypatch) -> None:
    monkeypatch.setenv("PLAN_RESULT", "failure")

    plan_result, selected, results = _read_environment()

    assert plan_result == "failure"
    assert selected == dict.fromkeys(ROUTED_CHECKS, False)
    assert results == {}
