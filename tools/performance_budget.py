"""Classify benchmark coverage and ASV comparison evidence against release policy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools import benchmark_coverage as _benchmark_coverage
except ImportError:  # pragma: no cover - direct script execution fallback
    import benchmark_coverage as _benchmark_coverage  # type: ignore[no-redef]

WARNING_REGRESSION_RATIO = 1.05
BLOCKING_REGRESSION_RATIO = 1.10

REQUIRED_COVERAGE_LAYERS = (
    "catalog_smoke",
    "family_matrix",
    "annotation_scaling",
    "batch_matrix",
    "reference_data",
    "target_matrix",
    "volumetric_matrix",
    "direct_kernel",
    "memory",
    "parameter_sensitivity",
    "pytorch_tensor",
)


@dataclass(frozen=True)
class BenchmarkPolicy:
    """Release policy for one ASV benchmark class."""

    pattern: str
    benchmark_class: str
    stability: str
    release_blocking: bool
    reason: str


BENCHMARK_POLICIES = (
    BenchmarkPolicy(
        "PeakMemory",
        "memory",
        "advisory",
        False,
        "peak-memory measurements on shared runners are triage evidence until enough history exists",
    ),
    BenchmarkPolicy(
        "BatchMatrix",
        "batch_matrix",
        "stable",
        True,
        "batch target routes are user-facing performance paths for images, masks, and masks3d",
    ),
    BenchmarkPolicy(
        "TimeParameterSensitivity",
        "parameter_sensitivity",
        "advisory",
        False,
        "parameter stress scenarios are required evidence but need scheduled history before release blocking",
    ),
    BenchmarkPolicy(
        "TimeCatalogTransformSmoke",
        "catalog_smoke",
        "advisory",
        False,
        "catalog smoke detects public-route drift but is too broad for direct release blocking",
    ),
    BenchmarkPolicy(
        "TimeCorePipeline",
        "core_pipeline",
        "stable",
        True,
        "Compose, ReplayCompose, and batch dispatch costs are core user-facing performance paths",
    ),
    BenchmarkPolicy(
        "TimeGeometryFullMatrix",
        "geometry_matrix",
        "stable",
        True,
        "geometry transform matrices cover normal image-size, channel, and dtype performance",
    ),
    BenchmarkPolicy(
        "TimePixelFullMatrix",
        "pixel_matrix",
        "stable",
        True,
        "pixel transform matrices cover normal image-size, channel, and dtype performance",
    ),
    BenchmarkPolicy(
        "TimeVolumetricFullMatrix",
        "volumetric_matrix",
        "stable",
        True,
        "3D transform matrices cover public volumetric performance paths",
    ),
    BenchmarkPolicy(
        "TimeAnnotationTargets",
        "annotation_scaling",
        "stable",
        True,
        "bbox, OBB, keypoint, and label scaling are release-critical target paths",
    ),
    BenchmarkPolicy(
        "TimeSpecialTargetMatrix",
        "special_targets",
        "stable",
        True,
        "bbox-safe crop, mask, and constrained-dropout paths are release-critical target paths",
    ),
    BenchmarkPolicy(
        "TimeFunctional",
        "direct_kernel",
        "stable",
        True,
        "direct functional kernels isolate shared hot-path changes from Compose overhead",
    ),
    BenchmarkPolicy(
        "TimeReferenceDataFullMatrix",
        "reference_data",
        "advisory",
        False,
        "metadata and reference-data paths are required evidence but can be runner-sensitive",
    ),
    BenchmarkPolicy(
        "TimeGeometricTransforms",
        "legacy_geometry_representative",
        "advisory",
        False,
        "legacy representative geometry benchmarks remain useful but are superseded by family matrices",
    ),
    BenchmarkPolicy(
        "TimePixelTransforms",
        "legacy_pixel_representative",
        "advisory",
        False,
        "legacy representative pixel benchmarks remain useful but are superseded by family matrices",
    ),
    BenchmarkPolicy(
        "TimeMixingTransforms",
        "legacy_mixing_representative",
        "advisory",
        False,
        "legacy representative mixing benchmarks remain useful but are superseded by reference-data matrices",
    ),
    BenchmarkPolicy(
        "TimeVolumetricTransforms",
        "legacy_volumetric_representative",
        "advisory",
        False,
        "legacy representative volumetric benchmarks remain useful but are superseded by volumetric matrices",
    ),
    BenchmarkPolicy(
        "TimeToTensor",
        "pytorch_tensor",
        "advisory",
        False,
        "optional PyTorch tensor transforms are tracked in a separate dependency lane",
    ),
)

UNKNOWN_BENCHMARK_POLICY = BenchmarkPolicy(
    "",
    "unknown",
    "advisory",
    False,
    "unclassified benchmark changes require maintainer triage before release claims",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def classify_benchmark(benchmark_name: str) -> BenchmarkPolicy:
    """Return the performance policy class for an ASV benchmark name."""
    for policy in BENCHMARK_POLICIES:
        if policy.pattern in benchmark_name:
            return policy
    return UNKNOWN_BENCHMARK_POLICY


def _coverage_issue_summary(
    coverage_detail: dict[str, Any],
    *,
    require_optional: bool,
) -> tuple[list[str], dict[str, Any]]:
    detail_summary = coverage_detail.get("summary", {})
    layer_counts = coverage_detail.get("layer_counts", {})
    issues: list[str] = []

    if detail_summary.get("contract_failures", 0) != 0:
        issues.append(f"{detail_summary.get('contract_failures')} benchmark coverage contract failure(s)")
    if detail_summary.get("smoke_only_transforms", 0) != 0:
        issues.append(f"{detail_summary.get('smoke_only_transforms')} runnable smoke-only transform(s)")

    required_layers = REQUIRED_COVERAGE_LAYERS if require_optional else REQUIRED_COVERAGE_LAYERS[:-1]
    missing_layers = [layer for layer in required_layers if layer_counts.get(layer, 0) <= 0]
    if missing_layers:
        issues.append("missing required benchmark coverage layer(s): " + ", ".join(missing_layers))

    return issues, {
        "contract_failures": detail_summary.get("contract_failures", 0),
        "deep_coverage_transforms": detail_summary.get("deep_coverage_transforms", 0),
        "layer_counts": {layer: layer_counts.get(layer, 0) for layer in REQUIRED_COVERAGE_LAYERS},
        "optional_transforms": detail_summary.get("optional_transforms", 0),
        "public_transforms": coverage_detail.get("public_transforms", 0),
        "smoke_only_transforms": detail_summary.get("smoke_only_transforms", 0),
    }


def _coverage_summary_issues(
    coverage_summary: dict[str, Any],
    coverage_detail: dict[str, Any],
    *,
    require_optional: bool,
) -> list[str]:
    issues: list[str] = []
    summary_public = coverage_summary.get("public_transforms")
    detail_public = coverage_detail.get("public_transforms")
    if summary_public != detail_public:
        issues.append(
            f"benchmark coverage summary/detail public transform count mismatch: {summary_public} != {detail_public}",
        )

    coverage_depth = coverage_summary.get("coverage_depth", {})
    if coverage_depth.get("contract_failures", 0) != 0:
        issues.append("benchmark coverage summary reports contract failures")
    if coverage_summary.get("memory_benchmarks", 0) <= 0:
        issues.append("benchmark coverage summary reports no memory benchmarks")
    if require_optional and coverage_summary.get("pytorch_tensor_benchmark_cases", 0) <= 0:
        issues.append("benchmark coverage summary reports no optional PyTorch tensor benchmark cases")
    return issues


def _regression_severity(policy: BenchmarkPolicy, ratio: float | None) -> str:
    if ratio is None:
        return "triage_required"
    if policy.release_blocking and ratio >= BLOCKING_REGRESSION_RATIO:
        return "release_blocker"
    if ratio >= WARNING_REGRESSION_RATIO:
        return "triage_required"
    return "changed_below_budget"


def _classify_regressions(asv_summary: dict[str, Any] | None) -> dict[str, Any]:
    if asv_summary is None:
        return {
            "provided": False,
            "asv_exit_code": None,
            "release_blockers": [],
            "triage_items": [],
            "changed_below_budget": [],
            "unknown_benchmarks": [],
            "totals": {"regressions": 0, "improvements": 0, "changed": 0},
        }

    release_blockers: list[dict[str, Any]] = []
    triage_items: list[dict[str, Any]] = []
    changed_below_budget: list[dict[str, Any]] = []
    unknown_benchmarks: list[str] = []

    for row in asv_summary.get("regressions", []):
        benchmark_name = str(row.get("benchmark", ""))
        ratio = row.get("ratio")
        policy = classify_benchmark(benchmark_name)
        item = {
            **row,
            "benchmark_class": policy.benchmark_class,
            "policy_reason": policy.reason,
            "release_blocking_class": policy.release_blocking,
            "stability": policy.stability,
        }
        severity = _regression_severity(policy, ratio if isinstance(ratio, float) else None)
        item["severity"] = severity
        if policy.benchmark_class == "unknown":
            unknown_benchmarks.append(benchmark_name)
        if severity == "release_blocker":
            release_blockers.append(item)
        elif severity == "triage_required":
            triage_items.append(item)
        else:
            changed_below_budget.append(item)

    return {
        "provided": True,
        "asv_exit_code": asv_summary.get("asv_exit_code"),
        "release_blockers": release_blockers,
        "triage_items": triage_items,
        "changed_below_budget": changed_below_budget,
        "unknown_benchmarks": sorted(set(unknown_benchmarks)),
        "totals": asv_summary.get("totals", {"regressions": 0, "improvements": 0, "changed": 0}),
    }


def _budget_status(
    coverage_issues: list[str],
    comparison: dict[str, Any],
    *,
    require_comparison: bool,
) -> str:
    if coverage_issues:
        return "release_blocked"
    if require_comparison and not comparison["provided"]:
        return "missing_comparison"
    if comparison["release_blockers"]:
        return "release_blocked"
    if comparison["triage_items"] or comparison["unknown_benchmarks"]:
        return "triage_required"
    return "ok"


def build_budget(
    coverage_summary: dict[str, Any],
    coverage_detail: dict[str, Any],
    asv_summary: dict[str, Any] | None = None,
    *,
    require_comparison: bool = False,
    require_optional: bool = True,
) -> dict[str, Any]:
    """Build machine-readable performance budget evidence."""
    detail_issues, coverage = _coverage_issue_summary(coverage_detail, require_optional=require_optional)
    coverage_issues = [
        *detail_issues,
        *_coverage_summary_issues(coverage_summary, coverage_detail, require_optional=require_optional),
    ]
    comparison = _classify_regressions(asv_summary)
    status = _budget_status(coverage_issues, comparison, require_comparison=require_comparison)
    issues = list(coverage_issues)
    if require_comparison and not comparison["provided"]:
        issues.append("ASV comparison summary is required but was not provided")
    issues.extend(
        f"unclassified ASV benchmark regression requires triage: {benchmark}"
        for benchmark in comparison["unknown_benchmarks"]
    )

    return {
        "schema_version": 1,
        "kind": "performance-budget",
        "status": status,
        "thresholds": {
            "blocking_regression_ratio": BLOCKING_REGRESSION_RATIO,
            "warning_regression_ratio": WARNING_REGRESSION_RATIO,
        },
        "coverage": coverage,
        "comparison": comparison,
        "issues": issues,
    }


def _write_budget(path: Path, budget: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")


def _summarize(args: argparse.Namespace) -> int:
    asv_summary = _read_json(args.asv_summary) if args.asv_summary is not None and args.asv_summary.exists() else None
    budget = build_budget(
        _read_json(args.coverage_summary),
        _read_json(args.coverage_detail),
        asv_summary,
        require_comparison=args.require_comparison,
        require_optional=not args.core_only,
    )
    _write_budget(args.output, budget)
    print(f"Wrote performance budget to {args.output}: {budget['status']}")
    for issue in budget["issues"]:
        print(f"- {issue}")
    if args.fail_on_release_blockers and budget["status"] in {"missing_comparison", "release_blocked"}:
        return 1
    return 0


def _check_current() -> int:
    detail = _benchmark_coverage.coverage_details()
    summary = {
        "coverage_depth": detail["summary"],
        "memory_benchmarks": detail["layer_counts"].get("memory", 0),
        "public_transforms": detail["public_transforms"],
        "pytorch_tensor_benchmark_cases": detail["layer_counts"].get("pytorch_tensor", 0),
    }
    budget = build_budget(
        summary,
        detail,
        require_optional=not _benchmark_coverage.unavailable_optional_transform_names(),
    )
    if budget["status"] != "ok":
        print("Performance budget validation failed:")
        for issue in budget["issues"]:
            print(f"- {issue}")
        return 1

    print("Performance budget validation passed.")
    print(
        "Required benchmark coverage layers: "
        + ", ".join(f"{layer}={budget['coverage']['layer_counts'][layer]}" for layer in REQUIRED_COVERAGE_LAYERS),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Validate current benchmark coverage against policy.")
    check_parser.set_defaults(func=lambda _args: _check_current())

    summarize_parser = subparsers.add_parser("summarize", help="Write a performance budget JSON artifact.")
    summarize_parser.add_argument("--coverage-summary", required=True, type=Path)
    summarize_parser.add_argument("--coverage-detail", required=True, type=Path)
    summarize_parser.add_argument("--asv-summary", type=Path)
    summarize_parser.add_argument("--output", required=True, type=Path)
    summarize_parser.add_argument(
        "--core-only",
        action="store_true",
        help="Validate the core benchmark lane without requiring optional PyTorch evidence.",
    )
    summarize_parser.add_argument(
        "--require-comparison",
        action="store_true",
        help="Treat a missing ASV comparison summary as a policy failure.",
    )
    summarize_parser.add_argument(
        "--fail-on-release-blockers",
        action="store_true",
        help="Return a non-zero exit code for release-blocked or missing-comparison status.",
    )
    summarize_parser.set_defaults(func=_summarize)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
