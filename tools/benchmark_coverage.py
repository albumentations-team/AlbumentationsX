"""Validate benchmark coverage for the public transform catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmarks.catalog import (  # noqa: E402
    OPTIONAL_BENCHMARK_TRANSFORMS,
    asv_case_ids,
    benchmark_specs,
    instantiate_transform,
    make_compose,
    make_data,
    public_transform_names,
)
from benchmarks.test_family_matrix import (  # noqa: E402
    ANNOTATION_CASES,
    GEOMETRY_CASES,
    PIXEL_CASES,
    REFERENCE_CASES,
    VOLUME_CASES,
)
from benchmarks.test_functional_kernels import (  # noqa: E402
    FUNCTIONAL_3D_CASES,
    FUNCTIONAL_BLUR_CASES,
    FUNCTIONAL_GEOMETRY_ANNOTATION_CASES,
    FUNCTIONAL_GEOMETRY_IMAGE_CASES,
    FUNCTIONAL_PIXEL_CASES,
)

MEMORY_BENCHMARKS = (
    "peakmem_affine_large_rgb",
    "peakmem_batch_pipeline_medium_rgb",
    "peakmem_copy_paste_small_rgb",
    "peakmem_mosaic_small_rgb",
    "peakmem_normalize_large_rgb",
    "peakmem_resize_large_rgb",
    "peakmem_volume_pad_medium",
)


def _spec_summary() -> dict[str, Any]:
    specs = benchmark_specs()
    route_counts = Counter(spec.route for spec in specs.values())
    runnable = [spec.name for spec in specs.values() if spec.benchmark]
    optional = {spec.name: spec.reason for spec in specs.values() if not spec.benchmark}
    return {
        "annotation_matrix_cases": len(ANNOTATION_CASES),
        "public_transforms": len(public_transform_names()),
        "full_matrix_cases": {
            "annotations": len(ANNOTATION_CASES),
            "geometry": len(GEOMETRY_CASES),
            "pixel": len(PIXEL_CASES),
            "reference_data": len(REFERENCE_CASES),
            "volumetric": len(VOLUME_CASES),
        },
        "direct_kernel_cases": {
            "blur": len(FUNCTIONAL_BLUR_CASES),
            "geometry_annotations": len(FUNCTIONAL_GEOMETRY_ANNOTATION_CASES),
            "geometry_images": len(FUNCTIONAL_GEOMETRY_IMAGE_CASES),
            "pixel": len(FUNCTIONAL_PIXEL_CASES),
            "volumetric": len(FUNCTIONAL_3D_CASES),
        },
        "memory_benchmarks": len(MEMORY_BENCHMARKS),
        "registered_specs": len(specs),
        "asv_cases": len(asv_case_ids()),
        "optional_cases": optional,
        "route_counts": dict(sorted(route_counts.items())),
        "runnable_transforms": runnable,
    }


def _validate_public_registry(public_names: set[str], spec_names: set[str]) -> list[str]:
    errors: list[str] = []
    missing = sorted(public_names - spec_names)
    unexpected = sorted(spec_names - public_names)
    if missing:
        errors.append("Missing benchmark specs: " + ", ".join(missing))
    if unexpected:
        errors.append("Benchmark specs reference unknown transforms: " + ", ".join(unexpected))
    unknown_optional = sorted(set(OPTIONAL_BENCHMARK_TRANSFORMS) - public_names)
    if unknown_optional:
        errors.append("Optional benchmark transform is not public: " + ", ".join(unknown_optional))
    return errors


def _validate_transform_construction(specs: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for spec in specs.values():
        if spec.route == "optional":
            continue
        try:
            instantiate_transform(spec)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.name}: cannot instantiate benchmark transform: {exc}")
    return errors


def _validate_case_groups(groups: Mapping[str, tuple[str, ...]], message: str) -> list[str]:
    errors: list[str] = []
    for group, cases in groups.items():
        if not cases:
            errors.append(message.format(group=group))
    return errors


def _validate_registry() -> list[str]:
    specs = benchmark_specs()
    errors = _validate_public_registry(set(public_transform_names()), set(specs))
    errors.extend(_validate_transform_construction(specs))
    errors.extend(
        _validate_case_groups(
            {
                "annotation": ANNOTATION_CASES,
                "geometry": GEOMETRY_CASES,
                "pixel": PIXEL_CASES,
                "reference_data": REFERENCE_CASES,
                "volumetric": VOLUME_CASES,
            },
            "Missing full-matrix benchmark cases for {group}",
        ),
    )
    required_direct_groups = {
        "blur": FUNCTIONAL_BLUR_CASES,
        "geometry_annotations": FUNCTIONAL_GEOMETRY_ANNOTATION_CASES,
        "geometry_images": FUNCTIONAL_GEOMETRY_IMAGE_CASES,
        "pixel": FUNCTIONAL_PIXEL_CASES,
        "volumetric": FUNCTIONAL_3D_CASES,
    }
    errors.extend(
        _validate_case_groups(
            required_direct_groups,
            "Missing direct functional-kernel benchmark cases for {group}",
        ),
    )
    return errors


def _validate_smoke() -> list[str]:
    errors: list[str] = []
    for spec in benchmark_specs().values():
        if not spec.benchmark:
            continue
        try:
            transform = make_compose(spec)
            data = make_data(spec)
            transform(**data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.name}: smoke route failed: {exc}")
    return errors


def check(*, run_smoke: bool, output: Path | None) -> int:
    """Run benchmark coverage validation."""
    errors = _validate_registry()
    if run_smoke:
        errors.extend(_validate_smoke())

    summary = _spec_summary()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Benchmark coverage failed: {len(errors)} issue(s) across "
            f"{summary['public_transforms']} public transforms.",
            file=sys.stderr,
        )
        return 1

    print(
        "Benchmark coverage ok: "
        f"{summary['asv_cases']} ASV cases, "
        f"{len(summary['optional_cases'])} optional cases, "
        f"{summary['public_transforms']} public transforms accounted.",
    )
    return 0


def summary(output: Path | None) -> int:
    """Write or print benchmark coverage summary."""
    data = _spec_summary()
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check", help="validate benchmark catalog coverage")
    check_parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="validate registry shape without executing transform smoke routes",
    )
    check_parser.add_argument(
        "--output",
        type=Path,
        help="write a JSON coverage summary",
    )
    summary_parser = subparsers.add_parser("summary", help="emit benchmark coverage JSON summary")
    summary_parser.add_argument(
        "--output",
        type=Path,
        help="write summary JSON to this path instead of stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        return check(run_smoke=not args.no_smoke, output=args.output)
    if args.command == "summary":
        return summary(output=args.output)
    return 2


if __name__ == "__main__":
    sys.exit(main())
