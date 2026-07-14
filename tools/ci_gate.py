"""Validate a stable CI gate against a router plan and GitHub job results."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

VALID_GATES = ("fast", "correctness", "policy")
PASSING_RESULT = "success"
IGNORED_RESULTS = {"skipped", ""}


def validate_gate(plan: dict[str, Any], gate: str, results: dict[str, str]) -> list[str]:
    """Return gate violations; an empty list means the gate passes."""
    gates = plan.get("gates")
    if not isinstance(gates, dict):
        return ["CI plan does not contain a gates mapping."]
    expected = gates.get(gate)
    if not isinstance(expected, list):
        return [f"CI plan does not contain a valid {gate!r} gate."]

    issues: list[str] = []
    expected_jobs = {str(job) for job in expected}
    for job in sorted(expected_jobs):
        result = results.get(job)
        if result != PASSING_RESULT:
            issues.append(f"Selected job {job!r} finished with {result or 'missing'!r}; expected 'success'.")

    for job, result in sorted(results.items()):
        if job not in expected_jobs and result not in {*IGNORED_RESULTS, PASSING_RESULT}:
            issues.append(f"Unselected job {job!r} unexpectedly finished with {result!r}.")
    return issues


def _load_json(value: str, label: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as error:
        msg = f"{label} is not valid JSON: {error}"
        raise ValueError(msg) from error
    if not isinstance(data, dict):
        msg = f"{label} must be a JSON object"
        raise TypeError(msg)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=VALID_GATES, required=True)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--results-json", help="JSON object mapping job names to GitHub results.")
    parser.add_argument("--result", action="append", default=[], help="One NAME=RESULT pair; may be repeated.")
    return parser.parse_args()


def _parse_results(values: list[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    for value in values:
        name, separator, result = value.partition("=")
        if not separator or not name:
            msg = f"Invalid result pair: {value!r}"
            raise ValueError(msg)
        results[name] = result
    return results


def main() -> int:
    args = parse_args()
    try:
        plan = _load_json(args.plan_json, "plan")
        raw_results = _load_json(args.results_json, "results") if args.results_json else {}
        raw_results.update(_parse_results(args.result))
    except (TypeError, ValueError) as error:
        print(f"CI gate failed: {error}", file=sys.stderr)
        return 1

    results = {str(job): str(result) for job, result in raw_results.items()}
    issues = validate_gate(plan, args.gate, results)
    if issues:
        print(f"{args.gate.capitalize()} gate failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    selected = plan["gates"][args.gate]
    print(f"{args.gate.capitalize()} gate passed for {len(selected)} selected job(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
