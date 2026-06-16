"""Verify golden regression vectors against current transform behavior."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

try:
    from tools.generate_regression_vectors import REGRESSION_ROOT, VECTOR_DIR, _array_metadata, _run_contract
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from generate_regression_vectors import REGRESSION_ROOT, VECTOR_DIR, _array_metadata, _run_contract
from tests.regression.transform_contracts import contract_by_name, registered_transform_names

MANIFEST_PATH = REGRESSION_ROOT / "manifest.json"


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        msg = f"Regression manifest is missing: {MANIFEST_PATH}"
        raise FileNotFoundError(msg)
    return json.loads(MANIFEST_PATH.read_text())


def _selected_cases(transform_name: str | None) -> list[dict[str, Any]]:
    cases = _load_manifest().get("cases", [])
    if transform_name is None:
        return list(cases)
    return [case for case in cases if case.get("transform") == transform_name]


def _manifest_contract_issues(cases: list[dict[str, Any]], transform_name: str | None) -> list[str]:
    manifest_names = {str(case.get("transform")) for case in cases}
    if transform_name is not None:
        return []

    registered_names = registered_transform_names()
    missing = sorted(registered_names - manifest_names)
    extra = sorted(manifest_names - registered_names)
    issues = [f"Registered regression contract has no manifest case: {name}" for name in missing]
    issues.extend(f"Manifest case has no registered regression contract: {name}" for name in extra)
    return issues


def _compare_array(
    case_id: str,
    target: str,
    expected: np.ndarray,
    actual: np.ndarray,
    stability: str,
    tolerance: float,
) -> list[str]:
    issues: list[str] = []
    if expected.shape != actual.shape:
        issues.append(f"{case_id}:{target} shape mismatch: {actual.shape} != {expected.shape}")
    if expected.dtype != actual.dtype:
        issues.append(f"{case_id}:{target} dtype mismatch: {actual.dtype} != {expected.dtype}")
    if issues:
        return issues

    if stability in {"exact", "digest"} or target.endswith("_labels") or not np.issubdtype(expected.dtype, np.number):
        try:
            np.testing.assert_array_equal(actual, expected)
        except AssertionError as error:
            issues.append(f"{case_id}:{target} values differ: {error}")
    elif stability == "tolerance":
        try:
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
        except AssertionError as error:
            issues.append(f"{case_id}:{target} values differ outside tolerance {tolerance}: {error}")
    elif stability == "structural":
        return issues
    else:
        issues.append(f"{case_id}:{target} has unsupported stability mode {stability!r}")
    return issues


def verify_case(case: dict[str, Any]) -> list[str]:
    case_id = str(case["id"])
    vector_path = REGRESSION_ROOT / str(case["vector_file"])
    if not vector_path.exists():
        return [f"{case_id} vector file is missing: {vector_path}"]

    contract = contract_by_name(str(case["transform"]))
    actual_outputs = _run_contract(contract)
    issues: list[str] = []
    stability = str(case.get("stability", contract.stability))
    tolerance = float(case.get("tolerance", contract.tolerance))

    with np.load(vector_path) as expected_outputs:
        for target in case["targets"]:
            expected = expected_outputs[target]
            actual = actual_outputs[target]
            issues.extend(_compare_array(case_id, target, expected, actual, stability, tolerance))
            metadata = _array_metadata(actual)
            expected_metadata = case.get("outputs", {}).get(target, {})
            if stability in {"exact", "digest"} and metadata.get("sha256") != expected_metadata.get("sha256"):
                issues.append(f"{case_id}:{target} digest mismatch")

    return issues


def verify_cases(transform_name: str | None = None) -> list[str]:
    if not VECTOR_DIR.exists():
        return [f"Regression vector directory is missing: {VECTOR_DIR}"]

    issues: list[str] = []
    cases = _selected_cases(transform_name)
    issues.extend(_manifest_contract_issues(cases, transform_name))
    if not cases:
        return [f"No regression cases selected for transform={transform_name!r}"]

    for case in cases:
        issues.extend(verify_case(case))
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", help="Single transform name to verify.")
    parser.add_argument("--all", action="store_true", help="Verify all vectors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transform_name = None if args.all else args.transform
    issues = verify_cases(transform_name)
    if issues:
        print("Regression vector verification failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Regression vector verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
