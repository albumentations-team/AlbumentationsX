"""Validate benchmark coverage for the public transform catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def _spec_summary() -> dict[str, Any]:
    specs = benchmark_specs()
    route_counts = Counter(spec.route for spec in specs.values())
    runnable = [spec.name for spec in specs.values() if spec.benchmark]
    optional = {spec.name: spec.reason for spec in specs.values() if not spec.benchmark}
    return {
        "public_transforms": len(public_transform_names()),
        "registered_specs": len(specs),
        "asv_cases": len(asv_case_ids()),
        "optional_cases": optional,
        "route_counts": dict(sorted(route_counts.items())),
        "runnable_transforms": runnable,
    }


def _validate_registry() -> list[str]:
    errors: list[str] = []
    public_names = set(public_transform_names())
    specs = benchmark_specs()
    spec_names = set(specs)

    missing = sorted(public_names - spec_names)
    unexpected = sorted(spec_names - public_names)
    if missing:
        errors.append("Missing benchmark specs: " + ", ".join(missing))
    if unexpected:
        errors.append("Benchmark specs reference unknown transforms: " + ", ".join(unexpected))

    unknown_optional = sorted(set(OPTIONAL_BENCHMARK_TRANSFORMS) - public_names)
    if unknown_optional:
        errors.append("Optional benchmark transform is not public: " + ", ".join(unknown_optional))

    for spec in specs.values():
        if spec.route == "optional":
            continue
        try:
            instantiate_transform(spec)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.name}: cannot instantiate benchmark transform: {exc}")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        return check(run_smoke=not args.no_smoke, output=args.output)
    return 2


if __name__ == "__main__":
    sys.exit(main())
