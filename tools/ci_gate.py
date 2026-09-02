"""Validate that routed pull-request jobs match the CI plan."""

from __future__ import annotations

import os
from collections.abc import Mapping

ROUTED_CHECKS = (
    "compatibility",
    "targeted",
    "pytorch",
    "dependency_audit",
    "workflow_audit",
    "legal",
    "package",
    "release_preflight",
)
ALWAYS_RUN_CHECKS = (
    "pre_commit_ruff",
    "pre_commit_ruff_format",
    "pre_commit_mypy",
    "pre_commit_pyrefly",
    "pre_commit_other",
)


def _parse_selection(value: str, check_name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Plan output for {check_name} must be true or false, got {value!r}")


def validate_results(
    *,
    plan_result: str,
    selected: Mapping[str, bool],
    results: Mapping[str, str],
) -> tuple[str, ...]:
    """Return routing errors for one pull-request run."""
    if plan_result != "success":
        return (f"PR plan finished with {plan_result!r}; routed checks are not trustworthy",)

    errors: list[str] = []
    for check_name in (*ALWAYS_RUN_CHECKS, *ROUTED_CHECKS):
        result = results.get(check_name)
        expected = "success" if check_name in ALWAYS_RUN_CHECKS else "success" if selected[check_name] else "skipped"
        if result != expected:
            errors.append(f"{check_name}: expected {expected}, got {result!r}")
    return tuple(errors)


def _read_environment() -> tuple[str, dict[str, bool], dict[str, str]]:
    plan_result = os.environ.get("PLAN_RESULT", "")
    if plan_result != "success":
        return plan_result, dict.fromkeys(ROUTED_CHECKS, False), {}

    selected = {
        check_name: _parse_selection(
            os.environ.get(f"PLAN_{check_name.upper()}", ""),
            check_name,
        )
        for check_name in ROUTED_CHECKS
    }
    results = {
        check_name: os.environ.get(f"RESULT_{check_name.upper()}", "")
        for check_name in (*ALWAYS_RUN_CHECKS, *ROUTED_CHECKS)
    }
    return plan_result, selected, results


def main() -> int:
    """Validate the GitHub Actions job results exposed through the environment."""
    try:
        plan_result, selected, results = _read_environment()
    except ValueError as error:
        print(f"::error::{error}")
        return 1

    errors = validate_results(plan_result=plan_result, selected=selected, results=results)
    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1

    print("PR gate passed: every selected check succeeded and every unselected check was skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
