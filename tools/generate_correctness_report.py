"""Generate the public Correctness & Compatibility Report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from tools.ci_matrix import DEPENDENCY_SETS, SUPPORTED_PYTHONS, TIER_1_OSES
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from ci_matrix import DEPENDENCY_SETS, SUPPORTED_PYTHONS, TIER_1_OSES

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


def _load_pyproject() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _read_json_files(directory: Path | None, pattern: str) -> list[dict[str, Any]]:
    if directory is None or not directory.exists():
        return []

    return [_read_json_file(path) for path in sorted(directory.glob(pattern))]


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        msg = f"Could not parse {path}: {error}"
        raise ValueError(msg) from error


def _project_version() -> str:
    project = _load_pyproject().get("project", {})
    version = project.get("version")
    if not isinstance(version, str):
        msg = "pyproject.toml is missing project.version"
        raise TypeError(msg)
    return version


def _format_test_summary(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "- Test summary artifacts: not provided in this evidence bundle"

    lines = []
    for index, summary in enumerate(summaries, start=1):
        totals = summary.get("totals", {})
        lines.append(
            "- Test summary "
            f"{index}: {totals.get('passed', 0)} passed, {totals.get('failures', 0)} failed, "
            f"{totals.get('errors', 0)} errors, {totals.get('skipped', 0)} skipped",
        )
    return "\n".join(lines)


def _format_environment_summary(environments: list[dict[str, Any]]) -> str:
    if not environments:
        return "- Environment artifacts: not provided in this evidence bundle"

    lines = []
    for index, environment in enumerate(environments, start=1):
        python = environment.get("python", {})
        os_info = environment.get("os", {})
        packages = environment.get("packages", {})
        lines.append(
            "- Environment "
            f"{index}: Python {python.get('version', 'unknown')} on {os_info.get('platform', 'unknown')}; "
            f"NumPy {packages.get('numpy')}; OpenCV {environment.get('opencv_runtime_version')}",
        )
    return "\n".join(lines)


def _evidence_status(items: list[dict[str, Any]], name: str) -> str:
    if items:
        return f"{name}: provided ({len(items)} artifact(s))"
    return f"{name}: not provided in this evidence bundle"


def generate_report(evidence_dir: Path | None = None, allow_missing_evidence: bool = False) -> str:
    environments = _read_json_files(evidence_dir, "environment*.json")
    pytest_summaries = _read_json_files(evidence_dir, "pytest-summary*.json")
    benchmark_summaries = _read_json_files(evidence_dir, "benchmark*.json")
    security_summaries = _read_json_files(evidence_dir, "security*.json")

    if not allow_missing_evidence and not environments:
        msg = "At least one environment*.json evidence artifact is required."
        raise ValueError(msg)

    version = _project_version()
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    python_versions = ", ".join(SUPPORTED_PYTHONS)
    operating_systems = ", ".join(TIER_1_OSES)
    dependency_sets = ", ".join(DEPENDENCY_SETS)

    return f"""# AlbumentationsX Correctness & Compatibility Report: {version}

Generated: {generated_at}

## Compatibility

Supported Python versions: {python_versions}

Tier 1 operating systems: {operating_systems}

Dependency sets tracked by the support policy: {dependency_sets}

| OS | Python | Dependency Set | Result |
| --- | --- | --- | --- |
| ubuntu-latest | 3.10, 3.11, 3.12, 3.13, 3.14 | locked-latest | see CI evidence |
| windows-latest | 3.10, 3.11, 3.12, 3.13, 3.14 | locked-latest | see CI evidence |
| macos-latest | 3.10, 3.11, 3.12, 3.13, 3.14 | locked-latest | see CI evidence |
| ubuntu-latest | 3.10 | declared-minimum | see lower-bound CI evidence |

## Correctness Coverage

{_format_test_summary(pytest_summaries)}

- Golden regression vectors: covered by `tests/regression`
- Property-based invariant tests: covered by `tests/property`
- Serialization and ReplayCompose checks: covered by regression and property suites
- Bbox/keypoint/OBB checks: covered by existing tests plus regression/property suites
- Volumetric checks: covered by existing 3D tests plus property suites

## Guaranteed Contracts

- Fixed-seed Compose pipelines are deterministic for tested transforms.
- ReplayCompose reproduces tested transform parameters and outputs.
- Image outputs preserve documented dtype and channel semantics.
- Bbox and keypoint label fields remain aligned with surviving annotations.
- Compose-level compatibility checks fail at pipeline creation where applicable.

## Known Limitations

- Exact pixel values for selected OpenCV-backed interpolation paths may vary across dependency versions.
- Optional extras are smoke-tested, not exhaustively cross-product tested.
- Performance artifacts from shared CI runners are treated as advisory unless release notes say otherwise.

## Performance

- {_evidence_status(benchmark_summaries, "Benchmark summary")}

## Security And Release Integrity

- Runtime dependency audit: see security workflow evidence
- GitHub Actions hardening audit: see security workflow evidence
- OpenSSF Scorecard: see security workflow evidence
- CodeQL: managed through GitHub default setup where enabled
- SBOM and SHA256 checksums: attached to the GitHub Release
- PyPI provenance: provided through trusted publishing attestations
- {_evidence_status(security_summaries, "Security summary")}

## Reproducibility

{_format_environment_summary(environments)}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Markdown report to write.")
    parser.add_argument("--evidence-dir", type=Path, help="Directory containing JSON evidence artifacts.")
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="Allow local dry runs without environment evidence artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = generate_report(args.evidence_dir, args.allow_missing_evidence)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote correctness report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
