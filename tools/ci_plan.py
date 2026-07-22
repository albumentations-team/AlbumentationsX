"""Build the fail-closed pull-request CI plan from changed repository paths."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

SCHEMA_VERSION = 2

CHECK_NAMES = (
    "markdown",
    "lint",
    "mypy",
    "pyrefly",
    "contracts",
    "compatibility",
    "coverage",
    "primary",
    "targeted",
    "pytorch",
    "dependency_audit",
    "workflow_audit",
    "legal",
    "package",
    "install_smoke",
    "release_preflight",
)

FAST_CHECKS = ("markdown", "lint", "mypy", "pyrefly", "contracts")
CORRECTNESS_CHECKS = ("compatibility", "coverage", "primary", "targeted", "pytorch")
POLICY_CHECKS = (
    "dependency_audit",
    "workflow_audit",
    "legal",
    "package",
    "install_smoke",
    "release_preflight",
)

LEGAL_PATHS = {
    "CLA.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE_HISTORY.md",
    "LICENSING.md",
    "MANIFEST.in",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "conda.recipe/meta.yaml",
    "docs/maintaining/license-provenance.md",
    "pyproject.toml",
    "tests/test_legal_integrity.py",
    "tools/verify_legal_integrity.py",
}
PACKAGING_PATHS = {
    "MANIFEST.in",
    "README.md",
    "conda.recipe/meta.yaml",
    "pyproject.toml",
}
DEPENDENCY_PATHS = {
    "conda.recipe/meta.yaml",
    "pyproject.toml",
    "requirements-dev.txt",
    "uv.lock",
}
INSTALL_SMOKE_PATHS = {
    ".github/workflows/upload_to_pypi.yml",
    "MANIFEST.in",
    "conda.recipe/meta.yaml",
    "pyproject.toml",
    "uv.lock",
}
QUALITY_CONFIG_PATHS = {
    ".pre-commit-config.yaml",
    "pyproject.toml",
}
CI_POLICY_PATHS = {
    "docs/maintaining/ci-greenfield-plan.md",
    "docs/maintaining/ci-policy.md",
    "docs/maintaining/correctness-and-compatibility-report.md",
    "docs/maintaining/correctness-report-template.md",
    "docs/maintaining/release-process.md",
    "docs/maintaining/support-policy.md",
}
SELF_CI_PATHS = {
    ".github/workflows/pr.yml",
    "tools/ci_gate.py",
    "tools/ci_plan.py",
    "tools/ci_shard.py",
    "tools/quality_gate.py",
}
SHARED_TEST_PATHS = {
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/utils.py",
}
BENCHMARK_TOOL_PATHS = {
    "tools/asv_summary.py",
    "tools/benchmark_coverage.py",
    "tools/performance_budget.py",
    "tools/select_benchmark_filters.py",
}
PYTORCH_TEST_PATHS = {
    "tests/test_additional_targets.py",
    "tests/test_benchmark_coverage.py",
    "tests/test_flip_masks_comprehensive.py",
    "tests/test_per_worker_seed.py",
    "tests/test_pytorch.py",
    "tests/test_serialization.py",
    "tests/transforms3d/test_pytorch.py",
}
PYTORCH_ONLY_TEST_PATHS = {
    "tests/test_pytorch.py",
    "tests/transforms3d/test_pytorch.py",
}
KNOWN_REPOSITORY_FILES = {
    ".git-blame-ignore-revs",
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "codecov.yaml",
}


@dataclass(frozen=True)
class CIPlan:
    """Serializable CI selection for one pull-request revision."""

    schema_version: int
    base_version: str | None
    head_version: str | None
    version_change: str
    changed_files: tuple[str, ...]
    domains: tuple[str, ...]
    checks: dict[str, bool]
    gates: dict[str, tuple[str, ...]]
    pytest_targets: tuple[str, ...]
    advisory_asv: bool
    antigravity: bool
    draft: bool
    forced_full: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Return compact deterministic JSON for GitHub job outputs."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def _normalise_path(raw_path: str) -> str:
    path = raw_path.removeprefix("./").replace("\\", "/")
    candidate = PurePosixPath(path)
    has_control_character = any(ord(character) < 32 or ord(character) == 127 for character in path)
    if not path or has_control_character or candidate.is_absolute() or ".." in candidate.parts:
        return ""
    return candidate.as_posix()


def _is_test_module(path: str) -> bool:
    candidate = PurePosixPath(path)
    return path.startswith("tests/") and candidate.name.startswith("test_") and candidate.suffix == ".py"


def _is_shared_test_path(path: str) -> bool:
    return path in SHARED_TEST_PATHS or path.startswith(("tests/helpers/", "tests/property/"))


def _is_documentation(path: str) -> bool:
    return path.endswith(".md") or path.startswith(("docs/", ".codex/")) or path in {"AGENTS.md", "CONTRIBUTING.md"}


def _is_ci_tooling(path: str) -> bool:
    named_ci_tools = {
        "tools/ci_matrix.py",
        "tools/collect_test_environment.py",
        "tools/pytest_summary.py",
    }
    generic_tool = path.startswith("tools/") and path.endswith(".py")
    specialised_tool = path in BENCHMARK_TOOL_PATHS or path == "tools/verify_legal_integrity.py"
    return path in SELF_CI_PATHS or path in named_ci_tools or (generic_tool and not specialised_tool)


def _is_packaging(path: str) -> bool:
    return path in PACKAGING_PATHS or path == ".github/workflows/upload_to_pypi.yml"


def _is_workflow(path: str) -> bool:
    return path.startswith((".github/workflows/", ".github/actions/", ".github/"))


DOMAIN_RULES = (
    ("docs", _is_documentation),
    ("ci_policy", CI_POLICY_PATHS.__contains__),
    ("runtime", lambda path: path.startswith("albumentations/")),
    ("pytorch", lambda path: path.startswith("albumentations/pytorch/") or path in PYTORCH_TEST_PATHS),
    ("tests", lambda path: path.startswith("tests/")),
    ("shared_tests", _is_shared_test_path),
    ("benchmarks", lambda path: path.startswith("benchmark/") or path in BENCHMARK_TOOL_PATHS),
    ("legal", lambda path: path in LEGAL_PATHS or path.startswith(("legal/", "THIRD_PARTY_LICENSES/"))),
    ("packaging", _is_packaging),
    ("dependencies", DEPENDENCY_PATHS.__contains__),
    ("quality_config", QUALITY_CONFIG_PATHS.__contains__),
    ("workflows", _is_workflow),
    ("ci_tooling", _is_ci_tooling),
    ("self_ci", lambda path: path in SELF_CI_PATHS or path.startswith(".github/actions/setup-ci/")),
    ("repository_config", KNOWN_REPOSITORY_FILES.__contains__),
)


def classify_path(raw_path: str) -> frozenset[str]:
    """Classify a repository path into additive CI risk domains."""
    path = _normalise_path(raw_path)
    if not path:
        return frozenset({"unknown"})

    domains = {domain for domain, matches in DOMAIN_RULES if matches(path)}

    return frozenset(domains or {"unknown"})


def _new_checks() -> dict[str, bool]:
    return dict.fromkeys(CHECK_NAMES, False)


def _select_fast_checks(checks: dict[str, bool], domains: set[str]) -> None:
    checks["markdown"] = "docs" in domains
    checks["lint"] = bool(
        domains & {"runtime", "tests", "benchmarks", "ci_tooling", "quality_config", "self_ci", "unknown"}
    )
    checks["mypy"] = bool(domains & {"runtime", "quality_config", "unknown"})
    checks["pyrefly"] = checks["mypy"]
    checks["contracts"] = bool(
        domains
        & {
            "runtime",
            "tests",
            "benchmarks",
            "ci_tooling",
            "ci_policy",
            "dependencies",
            "packaging",
            "workflows",
            "legal",
            "self_ci",
            "quality_config",
            "repository_config",
            "unknown",
        },
    )


def _select_correctness_checks(checks: dict[str, bool], domains: set[str], changed_files: tuple[str, ...]) -> None:
    full_matrix = bool(domains & {"runtime", "shared_tests", "self_ci", "unknown"})
    primary = bool(domains & {"dependencies", "ci_tooling"})
    isolated_tests = tuple(
        path
        for path in changed_files
        if _is_test_module(path) and not _is_shared_test_path(path) and path not in PYTORCH_ONLY_TEST_PATHS
    )

    checks["compatibility"] = full_matrix
    checks["coverage"] = full_matrix
    checks["primary"] = primary and not full_matrix
    checks["targeted"] = bool(isolated_tests) and not full_matrix and not primary
    checks["pytorch"] = bool(domains & {"pytorch", "dependencies", "benchmarks", "unknown"}) or any(
        path.startswith("albumentations/core/") for path in changed_files
    )


def _select_policy_checks(checks: dict[str, bool], domains: set[str], changed_files: tuple[str, ...]) -> None:
    checks["dependency_audit"] = bool(domains & {"dependencies", "packaging", "unknown"})
    checks["workflow_audit"] = bool(domains & {"workflows", "unknown"})
    checks["legal"] = bool(domains & {"legal", "dependencies", "packaging", "unknown"})
    checks["package"] = bool(domains & {"legal", "packaging", "unknown"})
    checks["install_smoke"] = "unknown" in domains or any(path in INSTALL_SMOKE_PATHS for path in changed_files)


def _select_everything(checks: dict[str, bool]) -> None:
    checks.update(dict.fromkeys(CHECK_NAMES, True))
    checks["primary"] = False
    checks["targeted"] = False
    checks["release_preflight"] = False


def _disable_heavy_checks(checks: dict[str, bool]) -> None:
    for name in (*CORRECTNESS_CHECKS, *POLICY_CHECKS):
        checks[name] = False


def _gate_jobs(checks: dict[str, bool], names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name.replace("_", "-") for name in names if checks[name])


def _version_change(base_version: str | None, head_version: str | None) -> str:
    if base_version is None and head_version is None:
        return "unspecified"
    if base_version is None or head_version is None:
        msg = "Both base_version and head_version are required when comparing project versions"
        raise ValueError(msg)

    try:
        parsed_base = Version(base_version)
        parsed_head = Version(head_version)
    except InvalidVersion as error:
        msg = f"Invalid PEP 440 version: {error}"
        raise ValueError(msg) from error

    if parsed_head > parsed_base:
        return "increase"
    if parsed_head < parsed_base:
        return "decrease"
    return "unchanged"


def build_plan(
    paths: list[str] | tuple[str, ...],
    *,
    draft: bool = False,
    force_full: bool = False,
    base_version: str | None = None,
    head_version: str | None = None,
) -> CIPlan:
    """Build the final CI plan for the supplied changed paths."""
    changed_files = tuple(sorted({_normalise_path(path) or "<invalid-path>" for path in paths}))
    classified = [classify_path(path) for path in changed_files]
    domains = set().union(*classified) if classified else {"unknown"}
    checks = _new_checks()
    version_change = _version_change(base_version, head_version)

    _select_fast_checks(checks, domains)
    _select_correctness_checks(checks, domains, changed_files)
    _select_policy_checks(checks, domains, changed_files)
    if force_full or "unknown" in domains or version_change == "increase":
        _select_everything(checks)
    checks["release_preflight"] = version_change == "increase"
    if draft and not force_full:
        _disable_heavy_checks(checks)

    pytest_targets = tuple(
        path
        for path in changed_files
        if _is_test_module(path) and not _is_shared_test_path(path) and path not in PYTORCH_ONLY_TEST_PATHS
    )
    gates = {
        "fast": _gate_jobs(checks, FAST_CHECKS),
        "correctness": _gate_jobs(checks, CORRECTNESS_CHECKS),
        "policy": _gate_jobs(checks, POLICY_CHECKS),
    }
    advisory_asv = not draft and bool(domains & {"runtime", "benchmarks", "unknown"})
    antigravity = not draft and bool(domains & {"runtime", "tests", "workflows", "legal", "self_ci", "unknown"})
    reasons = (
        f"Classified {len(changed_files)} changed path(s).",
        f"Selected domains: {', '.join(sorted(domains))}.",
        f"Project version change: {version_change}.",
        (
            "Unknown paths select the complete conservative profile."
            if "unknown" in domains
            else "All paths matched known domains."
        ),
    )

    return CIPlan(
        schema_version=SCHEMA_VERSION,
        base_version=base_version,
        head_version=head_version,
        version_change=version_change,
        changed_files=changed_files,
        domains=tuple(sorted(domains)),
        checks=checks,
        gates=gates,
        pytest_targets=pytest_targets,
        advisory_asv=advisory_asv,
        antigravity=antigravity,
        draft=draft,
        forced_full=force_full,
        reasons=reasons,
    )


def _parse_bool(value: str) -> bool:
    lowered = value.casefold()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no", ""}:
        return False
    msg = f"Expected a boolean value, received {value!r}"
    raise argparse.ArgumentTypeError(msg)


def _read_paths(path: Path, *, null_delimited: bool) -> list[str]:
    data = path.read_bytes()
    separator = b"\0" if null_delimited else b"\n"
    return [item.decode("utf-8") for item in data.split(separator) if item]


def _read_github_files(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    paths: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            filename = value.get("filename")
            if isinstance(filename, str):
                paths.append(filename)
            else:
                for item in value.values():
                    visit(item)

    visit(data)
    return paths


def _write_github_outputs(path: Path, plan: CIPlan) -> None:
    lines = [f"plan={plan.to_json()}"]
    lines.extend(f"{name}={str(selected).lower()}" for name, selected in sorted(plan.checks.items()))
    lines.extend(
        (
            f"advisory_asv={str(plan.advisory_asv).lower()}",
            f"antigravity={str(plan.antigravity).lower()}",
            f"base_version={plan.base_version or ''}",
            f"head_version={plan.head_version or ''}",
            f"version_change={plan.version_change}",
            "pytest_targets=" + json.dumps(plan.pytest_targets, separators=(",", ":")),
        ),
    )
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def _write_summary(path: Path, plan: CIPlan) -> None:
    selected = [name for name, enabled in sorted(plan.checks.items()) if enabled]
    lines = [
        "## Pull-request CI plan",
        "",
        f"Changed paths: {len(plan.changed_files)}",
        f"Domains: {', '.join(plan.domains)}",
        f"Selected checks: {', '.join(selected) if selected else 'none'}",
        f"Advisory ASV: {'yes' if plan.advisory_asv else 'no'}",
        f"Antigravity review: {'yes' if plan.antigravity else 'no'}",
        f"Version: {plan.base_version or 'unspecified'} → {plan.head_version or 'unspecified'} ({plan.version_change})",
        "",
        *[f"- {reason}" for reason in plan.reasons],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[], help="Changed path; may be repeated.")
    parser.add_argument("--paths-file", type=Path, help="File containing changed paths.")
    parser.add_argument("--github-files-json", type=Path, help="Paginated GitHub pull-request files JSON.")
    parser.add_argument("--null", action="store_true", help="Read the paths file as NUL-delimited data.")
    parser.add_argument("--draft", type=_parse_bool, default=False, help="Whether the pull request is a draft.")
    parser.add_argument("--force-full", type=_parse_bool, default=False, help="Select the complete CI profile.")
    parser.add_argument("--base-pyproject", type=Path, help="Base revision pyproject.toml.")
    parser.add_argument("--head-pyproject", type=Path, help="Head revision pyproject.toml.")
    parser.add_argument("--github-output", type=Path, help="Append scalar outputs for GitHub Actions.")
    parser.add_argument("--summary", type=Path, help="Write a human-readable job summary.")
    return parser.parse_args()


def _read_project_version(path: Path) -> str:
    with path.open("rb") as pyproject_file:
        data = tomllib.load(pyproject_file)
    project = data.get("project")
    if not isinstance(project, dict):
        msg = f"{path} is missing a [project] table"
        raise TypeError(msg)
    version = project.get("version")
    if not isinstance(version, str) or not version:
        msg = f"{path} is missing a non-empty project.version"
        raise ValueError(msg)
    return version


def main() -> int:
    args = parse_args()
    try:
        paths = list(args.path)
        if args.paths_file is not None:
            paths.extend(_read_paths(args.paths_file, null_delimited=args.null))
        if args.github_files_json is not None:
            paths.extend(_read_github_files(args.github_files_json))
        base_version = _read_project_version(args.base_pyproject) if args.base_pyproject is not None else None
        head_version = _read_project_version(args.head_pyproject) if args.head_pyproject is not None else None
        plan = build_plan(
            paths,
            draft=args.draft,
            force_full=args.force_full,
            base_version=base_version,
            head_version=head_version,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"CI plan failed: {error}", file=sys.stderr)
        return 1
    if args.github_output is not None:
        _write_github_outputs(args.github_output, plan)
    if args.summary is not None:
        _write_summary(args.summary, plan)
    print(plan.to_json())
    if plan.version_change == "decrease":
        print(
            f"CI plan failed: project.version decreased from {plan.base_version} to {plan.head_version}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
