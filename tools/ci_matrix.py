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
PR_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr.yml"
ANTIGRAVITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "antigravity-pr-checks.yml"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
PERFORMANCE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "performance.yml"
PYTORCH_PERFORMANCE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pytorch-performance.yml"
RELEASE_CANDIDATE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-candidate.yml"
SECURITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "upload_to_pypi.yml"
SETUP_CI_ACTION = REPO_ROOT / ".github" / "actions" / "setup-ci" / "action.yml"
CI_FOUNDATION_SHA = "6b9045dbea58026a1e8f96b0392c411934a27199"
RETIRED_ASV_RUN_PATTERN = re.compile(r"asv --config asv\.conf\.json\s+run\b")
RETIRED_REVISION_SELECTOR_PATTERN = re.compile(r"HEAD\^!")
CONDA_RECIPE = REPO_ROOT / "conda.recipe" / "meta.yaml"
DEVELOPMENT_REQUIREMENTS = REPO_ROOT / "requirements-dev.txt"
CODEQL_ACTIONS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "codeql-actions.yml"
CODEQL_PYTHON_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "codeql-python.yml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOWS = (
    PR_WORKFLOW,
    ANTIGRAVITY_WORKFLOW,
    NIGHTLY_WORKFLOW,
    PERFORMANCE_WORKFLOW,
    PYTORCH_PERFORMANCE_WORKFLOW,
    RELEASE_CANDIDATE_WORKFLOW,
    SECURITY_WORKFLOW,
    RELEASE_WORKFLOW,
    CODEQL_ACTIONS_WORKFLOW,
    CODEQL_PYTHON_WORKFLOW,
)
CODEQL_WORKFLOW_PATHS = {
    CODEQL_ACTIONS_WORKFLOW: ("**/*",),
    CODEQL_PYTHON_WORKFLOW: (
        "**/*.py",
        "**/*.pyi",
        ".github/codeql/**",
        ".github/workflows/codeql-python.yml",
    ),
}
FORBIDDEN_WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "ci.yml",
    REPO_ROOT / ".github" / "workflows" / "legal-integrity.yml",
)
MAX_WORKFLOW_TIMEOUT_MINUTES = 90

LOWER_BOUND_REQUIREMENTS = (
    "numpy==2.2.6",
    "scipy==1.15.3",
    "pydantic==2.12.4",
    "albucore==0.2.14",
    "opencv-python-headless==5.0.0.93",
)
CI_DEPENDENCY_GROUPS = {
    "ci-benchmark": {"asv", "opencv-python-headless"},
    "ci-package": {"pytest", "twine"},
    "ci-quality": {
        "defusedxml",
        "google-docstring-parser",
        "hypothesis",
        "opencv-python-headless",
        "packaging",
        "pre-commit",
        "pytest",
        "ruff",
    },
    "ci-release": {"asv", "cyclonedx-bom"},
    "ci-security": {"pip-audit", "zizmor"},
    "ci-test": {"defusedxml", "opencv-python-headless", "pytest", "pytest-cov", "pytest-xdist"},
    "ci-torch-cpu": {"torch"},
    "ci-types": {"mypy", "opencv-python-headless", "pyrefly"},
}
CI_RUNTIME_PROFILES = frozenset({"none", "torch-cpu"})
TORCH_RUNTIME_JOBS = {
    PR_WORKFLOW: {
        "markdown": "ci-quality",
        "contracts": "ci-quality",
        "compatibility": "ci-test",
        "coverage": "ci-test",
        "primary": "ci-test",
        "targeted": "ci-test",
        "pytorch": "ci-test",
        "release_preflight": "ci-release",
    },
    NIGHTLY_WORKFLOW: {
        "compatibility_matrix": "ci-test",
        "pytorch": "ci-test",
        "property_and_regression": "ci-test",
        "optional_extras": "ci-test",
    },
    RELEASE_CANDIDATE_WORKFLOW: {
        "release_candidate": "ci-test",
        "release_candidate_pytorch": "ci-test",
        "release_candidate_performance": "ci-benchmark",
    },
    PERFORMANCE_WORKFLOW: {
        "benchmark_evidence": "ci-benchmark",
        "pr_core_comparison": "ci-benchmark",
        "asv_comparison": "ci-benchmark",
        "scheduled_core_comparison": "ci-benchmark",
    },
    PYTORCH_PERFORMANCE_WORKFLOW: {"pytorch_tensor_asv": "ci-benchmark"},
}

SUPPORT_POLICY_TABLE_ROWS = (
    ("| `ubuntu-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Runtime-change PR gate and nightly |"),
    ("| `windows-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Runtime-change PR gate and nightly |"),
    ("| `macos-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Runtime-change PR gate and nightly |"),
    (
        "| `locked-latest` | Tests the repository lockfile and normal contributor environment. | "
        "Selected PR gates and full nightly/release |"
    ),
    (
        "| `declared-minimum` | Tests the declared lower runtime bounds on Ubuntu and Python 3.10. | "
        "Nightly and release gate |"
    ),
    (
        "| `optional-extras` | Smoke-tests extras such as `pillow`, `text`, `hub`, "
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


def _workflow_triggers(path: Path) -> dict[str, Any]:
    with path.open() as workflow_file:
        workflow = yaml.safe_load(workflow_file)
    if not isinstance(workflow, dict):
        return {}
    triggers = workflow.get(True, workflow.get("on", {}))
    return triggers if isinstance(triggers, dict) else {}


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
            includes = matrix.get("include", [])
            if isinstance(includes, list):
                for include in includes:
                    if isinstance(include, dict) and matrix_key in include:
                        values.add(str(include[matrix_key]))
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


def _direct_dependency_names(entries: list[Any]) -> set[str]:
    return {re.split(r"[<>=!~\[]", entry, maxsplit=1)[0].casefold() for entry in entries if isinstance(entry, str)}


def _dependency_group_packages(
    group_name: str,
    dependency_groups: dict[str, Any],
    active_groups: frozenset[str] = frozenset(),
) -> set[str]:
    """Return direct package names, including recursively included groups."""
    if group_name in active_groups:
        return set()
    entries = dependency_groups.get(group_name, [])
    if not isinstance(entries, list):
        return set()
    packages = _direct_dependency_names(entries)
    next_active_groups = active_groups | {group_name}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("include-group"), str):
            packages.update(_dependency_group_packages(entry["include-group"], dependency_groups, next_active_groups))
    return packages


def _dependency_group_reference_issues(dependency_groups: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for group, entries in dependency_groups.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            included_group = entry.get("include-group") if isinstance(entry, dict) else None
            if isinstance(included_group, str) and included_group not in dependency_groups:
                issues.append(
                    f"pyproject.toml dependency group {group!r} refers to non-existent group {included_group!r}",
                )
    return issues


def _check_ci_dependency_groups(dependency_groups: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    unexpected_groups = set(dependency_groups) - {*CI_DEPENDENCY_GROUPS, "dev"}
    if unexpected_groups:
        issues.append(f"pyproject.toml defines unsupported dependency groups {sorted(unexpected_groups)!r}")
    issues.extend(_dependency_group_reference_issues(dependency_groups))

    for group, required_packages in sorted(CI_DEPENDENCY_GROUPS.items()):
        entries = dependency_groups.get(group)
        if not isinstance(entries, list):
            issues.append(f"pyproject.toml is missing CI dependency group {group!r}")
            continue
        packages = _dependency_group_packages(group, dependency_groups)
        missing_packages = required_packages - packages
        if missing_packages:
            issues.append(f"pyproject.toml group {group!r} is missing {sorted(missing_packages)!r}")
    for group, entries in dependency_groups.items():
        if not isinstance(entries, list):
            continue
        packages = _direct_dependency_names(entries)
        if "torchvision" in packages:
            issues.append(f"pyproject.toml group {group!r} must not install torchvision")
        if "torch" in packages and group != "ci-torch-cpu":
            issues.append(f"pyproject.toml group {group!r} must not install Torch directly")

    return issues


def _check_project_torch_metadata(project: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    project_dependencies = project.get("dependencies")
    if not isinstance(project_dependencies, list):
        issues.append("pyproject.toml project.dependencies must be a list")
    elif {"torch", "torchvision"} & _direct_dependency_names(project_dependencies):
        issues.append("pyproject.toml project.dependencies must not select Torch or TorchVision")

    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict):
        issues.append("pyproject.toml project.optional-dependencies must be a table")
    else:
        for extra, entries in optional_dependencies.items():
            if isinstance(entries, list) and {"torch", "torchvision"} & _direct_dependency_names(entries):
                issues.append(f"pyproject.toml extra {extra!r} must not select Torch or TorchVision")
    return issues


def _check_torch_free_install_surfaces() -> list[str]:
    issues: list[str] = []
    development_requirements = _read_text(DEVELOPMENT_REQUIREMENTS)
    if re.search(r"(?im)^\s*(?:torch|torchvision)\b", development_requirements):
        issues.append("requirements-dev.txt must not select Torch or TorchVision")
    if re.search(r"(?im)^\s*--(?:extra-)?index-url\b", development_requirements):
        issues.append("requirements-dev.txt must not select a package index")

    conda_metadata = _read_text(CONDA_RECIPE)
    run_dependencies = re.search(r"(?ms)^  run:\s*\n(.*?)(?=^\S|\Z)", conda_metadata)
    if run_dependencies is None:
        issues.append("conda.recipe/meta.yaml must define a run dependency section")
    elif re.search(r"(?im)^\s*-\s*(?:pytorch|torchvision)\b", run_dependencies.group(1)):
        issues.append("conda.recipe/meta.yaml run dependencies must not select Torch or TorchVision")
    return issues


def _check_dev_group(dependency_groups: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    dev_entries = dependency_groups.get("dev", [])
    if not isinstance(dev_entries, list):
        return ["pyproject.toml dependency group 'dev' must be a list"]

    included_groups = {
        entry.get("include-group")
        for entry in dev_entries
        if isinstance(entry, dict) and isinstance(entry.get("include-group"), str)
    }
    expected_dev_groups = set(CI_DEPENDENCY_GROUPS) - {"ci-torch-cpu"}
    missing_includes = expected_dev_groups - included_groups
    if missing_includes:
        issues.append(f"pyproject.toml dev group does not include {sorted(missing_includes)!r}")
    if "ci-torch-cpu" in included_groups:
        issues.append("pyproject.toml dev group must not install the CI CPU Torch profile implicitly")
    return issues


def _check_pyproject() -> list[str]:
    pyproject = _load_pyproject()
    project = pyproject.get("project", {})
    issues: list[str] = []

    if not isinstance(project, dict):
        return ["pyproject.toml project must be a table"]

    requires_python = project.get("requires-python")
    if requires_python != f">={OLDEST_PYTHON}":
        issues.append(f"pyproject.toml requires-python is {requires_python!r}, expected '>={OLDEST_PYTHON}'")

    classifiers = _python_classifiers(pyproject)
    expected = set(SUPPORTED_PYTHONS)
    if classifiers != expected:
        issues.append(
            f"pyproject.toml Python classifiers are {sorted(classifiers)!r}, expected {sorted(expected)!r}",
        )

    issues.extend(_check_project_torch_metadata(project))
    issues.extend(_check_torch_free_install_surfaces())

    dependency_groups = pyproject.get("dependency-groups", {})
    if not isinstance(dependency_groups, dict):
        return [*issues, "pyproject.toml dependency-groups must be a table"]

    issues.extend(_check_ci_dependency_groups(dependency_groups))
    issues.extend(_check_dev_group(dependency_groups))

    uv = pyproject.get("tool", {}).get("uv", {})
    if not isinstance(uv, dict) or uv.get("sources", {}).get("torch") != {"index": "pytorch-cpu"}:
        issues.append("pyproject.toml must route Torch through the explicit pytorch-cpu index")

    return issues


def _setup_ci_step(job: dict[str, Any]) -> dict[str, Any] | None:
    steps = job.get("steps", [])
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and step.get("uses") == "./.github/actions/setup-ci":
            return step
    return None


def _setup_ci_inputs(job: dict[str, Any]) -> dict[str, Any] | None:
    step = _setup_ci_step(job)
    if step is None:
        return None
    inputs = step.get("with", {})
    return inputs if isinstance(inputs, dict) else None


def _check_setup_ci_action() -> list[str]:
    action = _load_yaml(SETUP_CI_ACTION)
    inputs = action.get("inputs", {})
    runs = action.get("runs", {})
    issues: list[str] = []

    if not isinstance(inputs, dict):
        return [f"{SETUP_CI_ACTION} inputs must be a mapping"]
    runtime_input = inputs.get("runtime-profile")
    if not isinstance(runtime_input, dict) or runtime_input.get("default") != "none":
        issues.append(f"{SETUP_CI_ACTION} must define runtime-profile with default 'none'")

    if not isinstance(runs, dict) or not isinstance(runs.get("steps"), list):
        return [*issues, f"{SETUP_CI_ACTION} runs.steps must be a list"]
    action_text = _read_text(SETUP_CI_ACTION)
    required_contracts = (
        f"albumentations-team/ci-foundation/actions/setup-python-uv@{CI_FOUNDATION_SHA}",
        f"albumentations-team/ci-foundation/actions/torch-cpu@{CI_FOUNDATION_SHA}",
        "cache-suffix: ${{ inputs.dependency-group }}-${{ inputs.runtime-profile }}",
        "ci-benchmark|ci-package|ci-quality|ci-release|ci-security|ci-test|ci-types",
        "torch-cpu) runtime_group=(--group ci-torch-cpu)",
        "Unknown CI runtime profile",
        "mode: verify",
        "ALBU_CI_RUNTIME_PROFILE=${CI_RUNTIME_PROFILE}",
    )
    issues.extend(
        f"{SETUP_CI_ACTION} is missing runtime-profile contract: {required}"
        for required in required_contracts
        if required not in action_text
    )
    return issues


def _expected_torch_runtime_jobs() -> dict[tuple[Path, str], str]:
    return {(path, job_id): group for path, jobs in TORCH_RUNTIME_JOBS.items() for job_id, group in jobs.items()}


def _check_expected_torch_runtime_jobs(expected_jobs: dict[tuple[Path, str], str]) -> list[str]:
    issues: list[str] = []
    for (path, job_id), expected_group in expected_jobs.items():
        job = _workflow_jobs(path).get(job_id)
        if not isinstance(job, dict):
            issues.append(f"{path} is missing Torch runtime job {job_id!r}")
            continue
        inputs = _setup_ci_inputs(job)
        if inputs is None:
            issues.append(f"{path} job {job_id!r} must use the local dependency-profile action")
            continue
        if inputs.get("dependency-group") != expected_group:
            issues.append(
                f"{path} job {job_id!r} must use dependency group {expected_group!r}, "
                f"found {inputs.get('dependency-group')!r}",
            )
        if inputs.get("runtime-profile") != "torch-cpu":
            issues.append(f"{path} job {job_id!r} must explicitly use runtime-profile 'torch-cpu'")
    return issues


def _check_declared_workflow_runtime_profiles(expected_jobs: dict[tuple[Path, str], str]) -> list[str]:
    issues: list[str] = []
    for path in _workflow_files():
        for job_id, job in _workflow_jobs(path).items():
            if not isinstance(job, dict):
                continue
            inputs = _setup_ci_inputs(job)
            if inputs is None:
                continue
            dependency_group = inputs.get("dependency-group")
            allowed_groups = set(CI_DEPENDENCY_GROUPS) - {"ci-torch-cpu"}
            if dependency_group not in allowed_groups:
                issues.append(f"{path} job {job_id!r} uses unsupported dependency group {dependency_group!r}")
            runtime_profile = inputs.get("runtime-profile", "none")
            if runtime_profile not in CI_RUNTIME_PROFILES:
                issues.append(f"{path} job {job_id!r} uses unknown runtime profile {runtime_profile!r}")
            if runtime_profile == "torch-cpu" and (path, job_id) not in expected_jobs:
                issues.append(f"{path} job {job_id!r} requests an undocumented CPU Torch runtime")
    return issues


def _check_lower_bound_torch_runtime() -> list[str]:
    issues: list[str] = []
    nightly_lower_bound = _workflow_jobs(NIGHTLY_WORKFLOW).get("lower_bound_dependencies")
    if not isinstance(nightly_lower_bound, dict):
        issues.append(f"{NIGHTLY_WORKFLOW} is missing lower_bound_dependencies")
    elif f"albumentations-team/ci-foundation/actions/torch-cpu@{CI_FOUNDATION_SHA}" not in _read_text(
        NIGHTLY_WORKFLOW,
    ):
        issues.append(f"{NIGHTLY_WORKFLOW} lower_bound_dependencies must install the shared CPU Torch runtime")
    return issues


def _check_workflow_torch_cleanup() -> list[str]:
    issues: list[str] = []
    for path in _workflow_files():
        workflow_text = _read_text(path)
        if re.search(r"(?:uv |python -m )?pip install[^\n]*torch", workflow_text, flags=re.IGNORECASE):
            issues.append(f"{path} must use the ci-foundation Torch action instead of installing Torch inline")
    return issues


def _check_torch_runtime_jobs() -> list[str]:
    expected_jobs = _expected_torch_runtime_jobs()
    return [
        *_check_expected_torch_runtime_jobs(expected_jobs),
        *_check_declared_workflow_runtime_profiles(expected_jobs),
        *_check_lower_bound_torch_runtime(),
        *_check_workflow_torch_cleanup(),
    ]


def _check_pr_workflow() -> list[str]:
    workflow = _load_yaml(PR_WORKFLOW)
    issues: list[str] = []

    ci_versions = _ci_python_versions(workflow)
    missing_versions = set(SUPPORTED_PYTHONS) - ci_versions
    if missing_versions:
        issues.append(f"{PR_WORKFLOW} does not test Python versions {sorted(missing_versions)!r}")

    ci_oses = _ci_operating_systems(workflow)
    missing_oses = set(TIER_1_OSES) - ci_oses
    if missing_oses:
        issues.append(f"{PR_WORKFLOW} does not test operating systems {sorted(missing_oses)!r}")

    workflow_header = _read_text(PR_WORKFLOW).split("permissions:", maxsplit=1)[0]
    if "paths:" in workflow_header or "paths-ignore:" in workflow_header:
        issues.append(f"{PR_WORKFLOW} must always start and route changed paths inside the plan job")

    jobs = _workflow_jobs(PR_WORKFLOW)
    expected_stable_jobs = {
        "plan": "PR plan",
        "fast_checks": "Fast checks",
        "correctness": "Correctness",
        "security_policy": "Security and policy",
    }
    for job_id, expected_name in expected_stable_jobs.items():
        job = jobs.get(job_id)
        if not isinstance(job, dict) or job.get("name") != expected_name:
            issues.append(f"{PR_WORKFLOW} must define stable job {job_id!r} named {expected_name!r}")

    issues.extend(
        _check_text_mentions(
            PR_WORKFLOW,
            (
                "python -m tools.ci_plan",
                "python -m tools.ci_gate",
                "dependency-group: ci-test",
                "dependency-group: ci-quality",
                "dependency-group: ci-types",
                "dependency-group: ci-security",
                "dependency-group: ci-package",
                "runtime-profile: torch-cpu",
                "python -m tools.ci_shard select",
                "--dist=worksteal",
                '-m "not pytorch"',
                "--hypothesis-profile=ci-fast",
                "tools/pytest_summary.py",
                "--allow-incomplete",
                "tools.release_bundle finalize",
                "dependency-group: ci-release",
                "--core-only",
                "asv --config asv.conf.json continuous",
                "--profile stf-core",
                "benchmark-asv-summary.json",
                "--require-comparison",
                "benchmark-baseline-sha.txt",
                "benchmark-candidate-sha.txt",
                "release-preflight=${{ needs.release_preflight.result }}",
            ),
            "selective PR gate",
        ),
    )

    release_job = jobs.get("release_preflight")
    if isinstance(release_job, dict):
        release_text = _workflow_job_run_text(release_job)
        if RETIRED_ASV_RUN_PATTERN.search(release_text) or RETIRED_REVISION_SELECTOR_PATTERN.search(release_text):
            issues.append(f"{PR_WORKFLOW} release_preflight uses a retired single-revision ASV path")
        if "--fail-on-release-blockers" not in release_text:
            issues.append(f"{PR_WORKFLOW} release_preflight must fail on release performance blockers")

    return issues


def _check_workflow_inventory() -> list[str]:
    issues: list[str] = []
    present = set(_workflow_files())
    expected = set(WORKFLOWS)
    missing = sorted(expected - present)
    if missing:
        issues.append("Missing expected workflow file(s): " + ", ".join(str(path) for path in missing))

    forbidden = sorted(path for path in FORBIDDEN_WORKFLOWS if path in present)
    if forbidden:
        issues.append("Obsolete workflow file(s) must be removed: " + ", ".join(str(path) for path in forbidden))

    for path in _workflow_files():
        issue = _workflow_yaml_issue(path)
        if issue is not None:
            issues.append(issue)
        if "--group dev" in _read_text(path):
            issues.append(f"{path} must use a purpose-specific CI dependency group instead of dev")
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
            if "uses" in job:
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
        expected_paths = CODEQL_WORKFLOW_PATHS.get(path)
        if expected_paths is None:
            workflow_header = _read_text(path).split("jobs:", maxsplit=1)[0]
            if re.search(r"(?m)^  push:\s*$", workflow_header):
                issues.append(f"{path} must not run from a push trigger; use PR, schedule, manual, or release events")
            continue

        triggers = _workflow_triggers(path)
        for event_name in ("pull_request", "push"):
            event = triggers.get(event_name)
            if not isinstance(event, dict):
                issues.append(f"{path} CodeQL workflow is missing {event_name!r} trigger")
                continue
            if event.get("branches") != ["main"]:
                issues.append(f"{path} CodeQL {event_name} trigger must be limited to main")
            if event.get("paths") != list(expected_paths):
                issues.append(f"{path} CodeQL {event_name} trigger must use its documented path filter")

        issues.extend(
            f"{path} CodeQL workflow is missing {event_name!r} trigger"
            for event_name in ("schedule", "workflow_dispatch")
            if event_name not in triggers
        )
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


def _workflow_job_run_text(job: dict[str, Any]) -> str:
    steps = job.get("steps", [])
    if not isinstance(steps, list):
        return ""
    return "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))


def _check_nightly_workflow() -> list[str]:
    issues = _check_full_matrix_workflow(NIGHTLY_WORKFLOW)
    issues.extend(_check_text_mentions(NIGHTLY_WORKFLOW, LOWER_BOUND_REQUIREMENTS, "lower-bound dependency"))
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
                "dependency-group: ci-test",
                "runtime-profile: torch-cpu",
                f"albumentations-team/ci-foundation/actions/torch-cpu@{CI_FOUNDATION_SHA}",
                "python -m tools.ci_shard select",
                '-m "not pytorch"',
                "-m pytorch",
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
                "python -m tools.performance_budget summarize",
                "benchmark-performance-budget-",
                "dependency-group: ci-test",
                "dependency-group: ci-benchmark",
                "runtime-profile: torch-cpu",
                "python -m tools.ci_shard select",
                '-m "not pytorch"',
                "-m pytorch",
            ),
            "release-candidate evidence gate",
        ),
    )
    return issues


def _check_benchmark_evidence_job(job: Any) -> list[str]:
    if not isinstance(job, dict):
        return [f"{PERFORMANCE_WORKFLOW} is missing benchmark_evidence job"]
    issues: list[str] = []
    if job.get("timeout-minutes") != 10:
        issues.append(f"{PERFORMANCE_WORKFLOW} benchmark_evidence job must keep timeout-minutes at 10")
    if job.get("if") != "github.event_name == 'pull_request'":
        issues.append(f"{PERFORMANCE_WORKFLOW} benchmark_evidence must be PR-only")
    run_text = _workflow_job_run_text(job)
    issues.extend(
        f"{PERFORMANCE_WORKFLOW} benchmark_evidence job must not run timing command: {command}"
        for command, is_present in (
            ("asv --config asv.conf.json continuous", "asv --config asv.conf.json continuous" in run_text),
            ("single-revision ASV run", RETIRED_ASV_RUN_PATTERN.search(run_text) is not None),
        )
        if is_present
    )
    return issues


def _check_pr_core_comparison_job(job: Any) -> list[str]:
    if not isinstance(job, dict):
        return [f"{PERFORMANCE_WORKFLOW} is missing pr_core_comparison job"]
    issues: list[str] = []
    if job.get("timeout-minutes") != 10:
        issues.append(f"{PERFORMANCE_WORKFLOW} pr_core_comparison must keep timeout-minutes at 10")
    if "benchmark_evidence" not in str(job.get("needs", "")):
        issues.append(f"{PERFORMANCE_WORKFLOW} pr_core_comparison must consume benchmark_evidence")
    run_text = _workflow_job_run_text(job)
    if "--profile pr-core" not in run_text:
        issues.append(f"{PERFORMANCE_WORKFLOW} pr_core_comparison must select pr-core")
    resolve_step = next(
        (
            step
            for step in job.get("steps", [])
            if isinstance(step, dict) and step.get("name") == "Resolve PR core comparison"
        ),
        None,
    )
    if not isinstance(resolve_step, dict) or resolve_step.get("env", {}).get("PR_CANDIDATE_SHA") != "${{ github.sha }}":
        issues.append(f"{PERFORMANCE_WORKFLOW} pr_core_comparison must compare the checked-out merge commit")
    return issues


def _check_targeted_comparison_job(job: Any) -> list[str]:
    if not isinstance(job, dict):
        return [f"{PERFORMANCE_WORKFLOW} is missing asv_comparison job"]
    issues: list[str] = []
    if "run-performance" not in str(job.get("if", "")):
        issues.append(f"{PERFORMANCE_WORKFLOW} asv_comparison job must be PR label gated")
    run_text = _workflow_job_run_text(job)
    if "asv --config asv.conf.json continuous" not in run_text:
        issues.append(f"{PERFORMANCE_WORKFLOW} asv_comparison job is missing timing command")
    if "git describe --tags --abbrev=0 --match '[0-9]*' \"$CANDIDATE_REF^\"" not in run_text:
        issues.append(f"{PERFORMANCE_WORKFLOW} asv_comparison must derive its default baseline from the candidate ref")
    if RETIRED_ASV_RUN_PATTERN.search(run_text):
        issues.append(f"{PERFORMANCE_WORKFLOW} asv_comparison job must not use the retired ASV run command")
    return issues


def _check_scheduled_comparison_job(job: Any) -> list[str]:
    if not isinstance(job, dict):
        return [f"{PERFORMANCE_WORKFLOW} is missing scheduled_core_comparison job"]
    issues: list[str] = []
    if job.get("timeout-minutes") != 20:
        issues.append(f"{PERFORMANCE_WORKFLOW} scheduled_core_comparison must keep timeout-minutes at 20")
    if job.get("if") != "github.event_name == 'schedule'":
        issues.append(f"{PERFORMANCE_WORKFLOW} scheduled_core_comparison must be schedule-only")
    run_text = _workflow_job_run_text(job)
    issues.extend(
        f"{PERFORMANCE_WORKFLOW} scheduled_core_comparison is missing {required}"
        for required in ("--profile stf-core", "git describe --tags --abbrev=0 --match '[0-9]*' HEAD^")
        if required not in run_text
    )
    return issues


def _check_performance_workflow() -> list[str]:
    issues = _check_text_mentions(
        PERFORMANCE_WORKFLOW,
        (
            "continue-on-error: true",
            "tools.benchmark_coverage summary",
            "tools.benchmark_coverage details",
            "asv --config asv.conf.json check --verbose",
            "asv --config asv.conf.json continuous",
            "tools/asv_summary.py",
            "python -m tools.performance_budget summarize",
            "tools/select_benchmark_filters.py",
            "--profile pr-core",
            "--profile stf-core",
            "--profile changed",
            "pr-core-performance-evidence/",
            "targeted-performance-evidence/",
            "scheduled-core-performance-evidence/",
            "benchmark-evidence/",
            "benchmark-filter.txt",
            "changed-files.txt",
            "run-performance",
            "git describe --tags --abbrev=0 --match '[0-9]*' HEAD^",
        ),
        "performance evidence gate",
    )
    workflow_text = _read_text(PERFORMANCE_WORKFLOW)
    if RETIRED_ASV_RUN_PATTERN.search(workflow_text):
        issues.append(f"{PERFORMANCE_WORKFLOW} must not use the retired ASV run command")
    if RETIRED_REVISION_SELECTOR_PATTERN.search(workflow_text):
        issues.append(f"{PERFORMANCE_WORKFLOW} must not use the retired single-revision selector")
    jobs = _workflow_jobs(PERFORMANCE_WORKFLOW)
    issues.extend(_check_benchmark_evidence_job(jobs.get("benchmark_evidence")))
    issues.extend(_check_pr_core_comparison_job(jobs.get("pr_core_comparison")))
    issues.extend(_check_targeted_comparison_job(jobs.get("asv_comparison")))
    issues.extend(_check_scheduled_comparison_job(jobs.get("scheduled_core_comparison")))
    return issues


def _check_pytorch_performance_workflow() -> list[str]:
    return _check_text_mentions(
        PYTORCH_PERFORMANCE_WORKFLOW,
        (
            "schedule:",
            "workflow_dispatch:",
            "continue-on-error: true",
            "dependency-group: ci-benchmark",
            "runtime-profile: torch-cpu",
            "asv --config asv-pytorch.conf.json check --verbose",
            "asv --config asv-pytorch.conf.json continuous",
            "git describe --tags --abbrev=0 --match '[0-9]*' \"$CANDIDATE_REF^\"",
            "benchmark-baseline-sha.txt",
            "benchmark-candidate-sha.txt",
            "benchmark-asv-summary.json",
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
    issues = _check_text_mentions(
        RELEASE_WORKFLOW,
        (
            "workflow_dispatch",
            "github.event.release.tag_name || inputs.release_tag",
            "path: release-automation",
            "path: released-source",
            "tools.release_bundle metadata",
            "--expected-tag",
            "tools.release_bundle resolve",
            "steps.metadata.outputs.artifact_name",
            "steps.resolve.outputs.artifact_id",
            "steps.resolve.outputs.run_id",
            "tools.release_bundle verify",
            "steps.metadata.outputs.source_digest",
            "verified-release-bundle",
            "release-bundle/public/SHA256SUMS.txt",
            "softprops/action-gh-release",
            "tag_name: ${{ env.RELEASE_TAG }}",
            "pypa/gh-action-pypi-publish",
            "packages-dir: release-bundle/dist",
        ),
        "delivery-only release contract",
    )
    release_text = _read_text(RELEASE_WORKFLOW)
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
    issues.extend(
        f"{RELEASE_WORKFLOW} delivery-only workflow must not run {command!r}"
        for command in forbidden_commands
        if command in release_text
    )
    return issues


def _check_workflows() -> list[str]:
    return [
        *_check_workflow_inventory(),
        *_check_workflow_python_versions(),
        *_check_workflow_job_timeouts(),
        *_check_workflow_push_triggers(),
        *_check_setup_ci_action(),
        *_check_torch_runtime_jobs(),
        *_check_pr_workflow(),
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
