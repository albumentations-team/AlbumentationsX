"""Validate the documented AlbumentationsX compatibility matrix."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
OLDEST_PYTHON = "3.10"
LATEST_PYTHON = "3.14"
PYTHON_PROBE = "3.15-dev"

TIER_1_OSES = ("ubuntu-latest", "windows-latest", "macos-latest")
DEPENDENCY_SETS = (
    "locked-latest",
    "declared-minimum",
    "optional-extras",
    "pre-release-probe",
)
TEST_GROUPS = (
    "unit-fast",
    "unit-full",
    "property-fast",
    "property-nightly",
    "regression-golden",
    "benchmark-smoke",
    "security",
    "release-smoke",
)

SUPPORT_POLICY = REPO_ROOT / "docs" / "maintaining" / "support-policy.md"
REPORT_TEMPLATE = REPO_ROOT / "docs" / "maintaining" / "correctness-report-template.md"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load_pyproject() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as workflow_file:
        data = yaml.safe_load(workflow_file)
    if not isinstance(data, dict):
        msg = f"{path} does not contain a YAML mapping"
        raise TypeError(msg)
    return data


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _python_classifiers(pyproject: dict[str, Any]) -> set[str]:
    project = pyproject.get("project", {})
    classifiers = project.get("classifiers", [])
    if not isinstance(classifiers, list):
        return set()

    prefix = "Programming Language :: Python :: "
    versions = set()
    for classifier in classifiers:
        if not isinstance(classifier, str) or not classifier.startswith(prefix):
            continue
        suffix = classifier.removeprefix(prefix)
        if re.fullmatch(r"3\.\d+", suffix):
            versions.add(suffix)
    return versions


def _ci_python_versions(workflow: dict[str, Any]) -> set[str]:
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return set()

    versions: set[str] = set()
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        strategy = job.get("strategy", {})
        matrix = strategy.get("matrix", {}) if isinstance(strategy, dict) else {}
        values = matrix.get("python-version", []) if isinstance(matrix, dict) else []
        if isinstance(values, str):
            versions.add(values)
        elif isinstance(values, list):
            versions.update(str(value) for value in values)
    return versions


def _ci_operating_systems(workflow: dict[str, Any]) -> set[str]:
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return set()

    operating_systems: set[str] = set()
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        strategy = job.get("strategy", {})
        matrix = strategy.get("matrix", {}) if isinstance(strategy, dict) else {}
        values = matrix.get("operating-system", []) if isinstance(matrix, dict) else []
        if isinstance(values, str):
            operating_systems.add(values)
        elif isinstance(values, list):
            operating_systems.update(str(value) for value in values)
    return operating_systems


def _check_pyproject() -> list[str]:
    pyproject = _load_pyproject()
    project = pyproject.get("project", {})
    issues: list[str] = []

    requires_python = project.get("requires-python")
    if requires_python != f">={OLDEST_PYTHON}":
        issues.append(f"pyproject.toml requires-python is {requires_python!r}, expected '>={OLDEST_PYTHON}'")

    classifiers = _python_classifiers(pyproject)
    expected = set(SUPPORTED_PYTHONS)
    if classifiers != expected:
        issues.append(
            f"pyproject.toml Python classifiers are {sorted(classifiers)!r}, expected {sorted(expected)!r}",
        )

    return issues


def _check_ci_workflow() -> list[str]:
    workflow = _load_yaml(CI_WORKFLOW)
    issues: list[str] = []

    ci_versions = _ci_python_versions(workflow)
    missing_versions = set(SUPPORTED_PYTHONS) - ci_versions
    if missing_versions:
        issues.append(f"{CI_WORKFLOW} does not test Python versions {sorted(missing_versions)!r}")

    ci_oses = _ci_operating_systems(workflow)
    missing_oses = set(TIER_1_OSES) - ci_oses
    if missing_oses:
        issues.append(f"{CI_WORKFLOW} does not test operating systems {sorted(missing_oses)!r}")

    return issues


def _check_python_mentions(text: str, path: Path) -> list[str]:
    return [f"{path} does not mention Python {version}" for version in SUPPORTED_PYTHONS if version not in text]


def _check_required_mentions(text: str, path: Path, values: tuple[str, ...], label: str) -> list[str]:
    return [f"{path} does not mention {label} {value}" for value in values if value not in text]


def _check_docs() -> list[str]:
    support_policy = _read_text(SUPPORT_POLICY)
    report_template = _read_text(REPORT_TEMPLATE)
    issues: list[str] = []

    if not support_policy:
        issues.append(f"{SUPPORT_POLICY} is missing or empty")
    if not report_template:
        issues.append(f"{REPORT_TEMPLATE} is missing or empty")

    issues.extend(_check_python_mentions(support_policy, SUPPORT_POLICY))
    issues.extend(_check_python_mentions(report_template, REPORT_TEMPLATE))
    issues.extend(_check_required_mentions(support_policy, SUPPORT_POLICY, TIER_1_OSES, "operating system"))
    issues.extend(_check_required_mentions(support_policy, SUPPORT_POLICY, DEPENDENCY_SETS, "dependency set"))
    issues.extend(_check_required_mentions(report_template, REPORT_TEMPLATE, DEPENDENCY_SETS, "dependency set"))

    return issues


def check() -> int:
    issues = [*_check_pyproject(), *_check_ci_workflow(), *_check_docs()]
    if issues:
        print("CI matrix validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("CI matrix validation passed.")
    print(f"Supported Python versions: {', '.join(SUPPORTED_PYTHONS)}")
    print(f"Tier 1 operating systems: {', '.join(TIER_1_OSES)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",), help="Action to run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        return check()
    return 1


if __name__ == "__main__":
    sys.exit(main())
