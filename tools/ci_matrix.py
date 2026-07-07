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
NIGHTLY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
PERFORMANCE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "performance.yml"
PYTORCH_PERFORMANCE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pytorch-performance.yml"
RELEASE_CANDIDATE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-candidate.yml"
SECURITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "upload_to_pypi.yml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOWS = (
    CI_WORKFLOW,
    NIGHTLY_WORKFLOW,
    PERFORMANCE_WORKFLOW,
    PYTORCH_PERFORMANCE_WORKFLOW,
    RELEASE_CANDIDATE_WORKFLOW,
    SECURITY_WORKFLOW,
    RELEASE_WORKFLOW,
)
MAX_WORKFLOW_TIMEOUT_MINUTES = 90

LOWER_BOUND_REQUIREMENTS = (
    "numpy==2.2.6",
    "scipy==1.15.3",
    "pydantic==2.12.4",
    "albucore==0.2.4",
    "opencv-python-headless==5.0.0.93",
)

SUPPORT_POLICY_TABLE_ROWS = (
    "| `ubuntu-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Required PR gate |",
    "| `windows-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Required PR gate |",
    "| `macos-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Required PR gate |",
    ("| `locked-latest` | Tests the repository lockfile and normal contributor environment. | Required PR gate |"),
    (
        "| `declared-minimum` | Tests the declared lower runtime bounds on Ubuntu and Python 3.10. | "
        "Nightly and release gate |"
    ),
    (
        "| `optional-extras` | Smoke-tests extras such as `pillow`, `pytorch`, `text`, `hub`, "
        "and OpenCV variants. | Advisory until stable |"
    ),
    (
        "| `pre-release-probe` | Probes future Python or dependency releases when wheels are available. | "
        "Scheduled advisory |"
    ),
)

REPORT_TEMPLATE_ROWS = (
    "| `ubuntu-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | `locked-latest` | `<result>` |",
    "| `windows-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | `locked-latest` | `<result>` |",
    "| `macos-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | `locked-latest` | `<result>` |",
    "| `ubuntu-latest` | 3.10 | `declared-minimum` | `<result>` |",
    "| `ubuntu-latest` | 3.14 | `optional-extras` | `<result>` |",
    "| `ubuntu-latest` | `3.15-dev` | `pre-release-probe` | `<result>` |",
)


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


def _workflow_files() -> tuple[Path, ...]:
    return tuple(sorted(WORKFLOW_DIR.glob("*.yml")))


def _workflow_yaml_issue(path: Path) -> str | None:
    try:
        _load_yaml(path)
    except (OSError, TypeError, yaml.YAMLError) as error:
        return f"{path} is not valid workflow YAML: {error}"
    return None


def _workflow_jobs(path: Path) -> dict[str, Any]:
    workflow = _load_yaml(path)
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return {}
    return jobs


def _walk_values_for_key(data: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(data, dict):
        for candidate_key, value in data.items():
            if candidate_key == key:
                values.append(value)
            values.extend(_walk_values_for_key(value, key))
    elif isinstance(data, list):
        for value in data:
            values.extend(_walk_values_for_key(value, key))
    return values


def _string_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _workflow_python_versions(workflow: dict[str, Any]) -> set[str]:
    versions: set[str] = set()
    for value in _walk_values_for_key(workflow, "python-version"):
        versions.update(item for item in _string_values(value) if re.fullmatch(r"3\.\d+|3\.\d+-dev", item))
    return versions


def _workflow_matrix_values(workflow: dict[str, Any], matrix_key: str) -> set[str]:
    values: set[str] = set()
    for matrix in _walk_values_for_key(workflow, "matrix"):
        if isinstance(matrix, dict):
            values.update(_string_values(matrix.get(matrix_key, [])))
    return values


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
    return _workflow_matrix_values(workflow, "python-version")


def _ci_operating_systems(workflow: dict[str, Any]) -> set[str]:
    return _workflow_matrix_values(workflow, "operating-system")


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

    issues.extend(
        _check_text_mentions(
            CI_WORKFLOW,
            (
                "--hypothesis-profile=ci-fast",
                "tools/pytest_summary.py",
                "--allow-incomplete",
            ),
            "CI evidence gate",
        ),
    )

    return issues


def _check_workflow_inventory() -> list[str]:
    issues: list[str] = []
    present = set(_workflow_files())
    expected = set(WORKFLOWS)
    missing = sorted(expected - present)
    if missing:
        issues.append("Missing expected workflow file(s): " + ", ".join(str(path) for path in missing))

    for path in _workflow_files():
        issue = _workflow_yaml_issue(path)
        if issue is not None:
            issues.append(issue)
    return issues


def _check_workflow_python_versions() -> list[str]:
    allowed_versions = {*SUPPORTED_PYTHONS, PYTHON_PROBE}
    issues: list[str] = []
    for path in _workflow_files():
        versions = _workflow_python_versions(_load_yaml(path))
        unsupported = sorted(versions - allowed_versions)
        if unsupported:
            issues.append(f"{path} references unsupported Python version(s): {unsupported!r}")
    return issues


def _check_workflow_job_timeouts() -> list[str]:
    issues: list[str] = []
    for path in _workflow_files():
        for job_name, job in _workflow_jobs(path).items():
            if not isinstance(job, dict):
                issues.append(f"{path} job {job_name!r} is not a YAML mapping")
                continue
            timeout = job.get("timeout-minutes")
            if not isinstance(timeout, int):
                issues.append(f"{path} job {job_name!r} is missing integer timeout-minutes")
                continue
            if timeout <= 0 or timeout > MAX_WORKFLOW_TIMEOUT_MINUTES:
                issues.append(
                    f"{path} job {job_name!r} timeout-minutes is {timeout}, expected 1..{MAX_WORKFLOW_TIMEOUT_MINUTES}",
                )
    return issues


def _check_workflow_push_triggers() -> list[str]:
    issues: list[str] = []
    for path in _workflow_files():
        workflow_header = _read_text(path).split("jobs:", maxsplit=1)[0]
        if re.search(r"(?m)^  push:\s*$", workflow_header):
            issues.append(f"{path} must not run from a push trigger; use PR, schedule, manual, or release events")
    return issues


def _check_full_matrix_workflow(path: Path) -> list[str]:
    workflow = _load_yaml(path)
    issues: list[str] = []
    versions = _workflow_matrix_values(workflow, "python-version")
    operating_systems = _workflow_matrix_values(workflow, "operating-system")
    missing_versions = set(SUPPORTED_PYTHONS) - versions
    missing_oses = set(TIER_1_OSES) - operating_systems
    if missing_versions:
        issues.append(f"{path} full matrix is missing Python version(s): {sorted(missing_versions)!r}")
    if missing_oses:
        issues.append(f"{path} full matrix is missing operating system(s): {sorted(missing_oses)!r}")
    return issues


def _check_text_mentions(path: Path, required: tuple[str, ...], context: str) -> list[str]:
    text = _read_text(path)
    return [f"{path} does not include required {context}: {item}" for item in required if item not in text]


def _check_nightly_workflow() -> list[str]:
    issues = _check_text_mentions(NIGHTLY_WORKFLOW, LOWER_BOUND_REQUIREMENTS, "lower-bound dependency")
    issues.extend(
        _check_text_mentions(
            NIGHTLY_WORKFLOW,
            (
                'python-version: "3.10"',
                "--hypothesis-profile=ci-nightly",
                "tools/verify_regression_vectors.py --all",
                "tools/pytest_summary.py",
                "--allow-incomplete",
                "environment-lower-bound.json",
                "environment-property-regression.json",
                "environment-optional-extras.json",
            ),
            "nightly evidence gate",
        ),
    )
    return issues


def _check_release_candidate_workflow() -> list[str]:
    issues = _check_full_matrix_workflow(RELEASE_CANDIDATE_WORKFLOW)
    issues.extend(
        _check_text_mentions(
            RELEASE_CANDIDATE_WORKFLOW,
            (
                "--hypothesis-profile=release",
                "tools/verify_regression_vectors.py --all",
                "tools.benchmark_coverage summary",
                "tools.benchmark_coverage details",
                "tools/pytest_summary.py",
                "--allow-incomplete",
                "tools/performance_budget.py",
                "benchmark-performance-budget-",
            ),
            "release-candidate evidence gate",
        ),
    )
    return issues


def _check_performance_workflow() -> list[str]:
    return _check_text_mentions(
        PERFORMANCE_WORKFLOW,
        (
            "continue-on-error: true",
            "tools.benchmark_coverage summary",
            "tools.benchmark_coverage details",
            "asv --config asv.conf.json check --verbose",
            "tools/asv_summary.py",
            "tools/performance_budget.py",
            "tools/select_benchmark_filters.py",
            "benchmark-coverage-detail.json",
            "benchmark-performance-budget.json",
            "benchmark-evidence/",
            "benchmark-filter.txt",
            "changed-files.txt",
        ),
        "performance evidence gate",
    )


def _check_pytorch_performance_workflow() -> list[str]:
    return _check_text_mentions(
        PYTORCH_PERFORMANCE_WORKFLOW,
        (
            "schedule:",
            "workflow_dispatch:",
            "continue-on-error: true",
            "uv sync --locked --group dev --inexact --no-install-package torch --no-install-package torchvision",
            "asv --config asv-pytorch.conf.json check --verbose",
            "asv --config asv-pytorch.conf.json run",
            "pytorch-benchmark-evidence/",
        ),
        "PyTorch tensor performance evidence gate",
    )


def _check_security_workflow() -> list[str]:
    return _check_text_mentions(
        SECURITY_WORKFLOW,
        (
            "pip-audit",
            "zizmor",
            "ossf/scorecard-action",
            "workflow_dispatch",
            "security-evidence/",
            "security-pip-audit.json",
            "security-zizmor.json",
            "results_file: scorecard-results.json",
            "results_format: json",
            "results_file: scorecard-results.sarif",
            "results_format: sarif",
            "publish_results: true",
            "github.ref == 'refs/heads/main'",
        ),
        "security gate",
    )


def _check_release_workflow() -> list[str]:
    return _check_text_mentions(
        RELEASE_WORKFLOW,
        (
            "uv build",
            "twine check",
            "cyclonedx-py",
            "--no-install-package torch",
            "--no-install-package torchvision",
            "tools/collect_test_environment.py",
            "tools/verify_regression_vectors.py --all",
            "tests/regression tests/property",
            "--hypothesis-profile=ci-fast",
            "tools/pytest_summary.py",
            "pytest-summary-release.json",
            "tools.benchmark_coverage summary",
            "tools.benchmark_coverage details",
            "tools/performance_budget.py",
            "tools/generate_correctness_report.py",
            "benchmark-performance-budget-release.json",
            "SHA256SUMS.txt",
            "softprops/action-gh-release",
            "pypa/gh-action-pypi-publish",
        ),
        "release artifact gate",
    )


def _check_workflows() -> list[str]:
    return [
        *_check_workflow_inventory(),
        *_check_workflow_python_versions(),
        *_check_workflow_job_timeouts(),
        *_check_workflow_push_triggers(),
        *_check_ci_workflow(),
        *_check_full_matrix_workflow(RELEASE_CANDIDATE_WORKFLOW),
        *_check_nightly_workflow(),
        *_check_release_candidate_workflow(),
        *_check_performance_workflow(),
        *_check_pytorch_performance_workflow(),
        *_check_security_workflow(),
        *_check_release_workflow(),
    ]


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
    issues.extend(_check_required_mentions(support_policy, SUPPORT_POLICY, SUPPORT_POLICY_TABLE_ROWS, "table row"))
    issues.extend(_check_required_mentions(report_template, REPORT_TEMPLATE, REPORT_TEMPLATE_ROWS, "table row"))

    return issues


def check() -> int:
    issues = [*_check_pyproject(), *_check_workflows(), *_check_docs()]
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
