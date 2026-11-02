"""Enforce bounded Linux Compose RSS growth and per-call working-memory budgets."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2
RSS_CASES = ("varying", "explicit", "fixed", "image_mask", "hwc1", "direct")
PEAK_OUTPUTS = {
    "blur": ((1024, 1024), "float32", 4),
    "blur_float16": ((1024, 1024), "float16", 2),
    "normalize": ((1024, 1024, 3), "float32", 4),
    "batch_normalize": ((4, 512, 512, 3), "float32", 4),
    "tensor_normalize": ((3, 1024, 1024), "torch.float32", 4),
    "volume_pad": ((24, 288, 288, 1), "float32", 4),
}


def expected_output(case: str) -> dict[str, Any]:
    shape, dtype, itemsize = PEAK_OUTPUTS[case]
    return {"output_shape": list(shape), "output_dtype": dtype, "output_bytes": math.prod(shape) * itemsize}


def classify_samples(samples: list[dict[str, Any]], mode: str, repeats: int) -> dict[str, Any]:
    """Reject incomplete evidence, bound post-GC growth, and cap native working peaks."""
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_counts = dict.fromkeys(selected_cases(mode), repeats)
    counts = Counter((sample["mode"], sample["case"]) for sample in samples)
    if counts != expected_counts:
        failures.append("Memory samples must cover every selected case with the requested repeat count")
    rss = {
        case: [sample["growth_bytes"] for sample in samples if sample["mode"] == "rss" and sample["case"] == case]
        for case in RSS_CASES
    }
    control = max([0, *(statistics.median(rss[case]) for case in ("hwc1", "direct") if rss[case])])
    for case, values in rss.items():
        if not values:
            continue
        growth = statistics.median(values)
        budget = 32 * MIB if case == "image_mask" else 16 * MIB
        excess = max(0, growth - control)
        passed = growth <= budget
        cases.append(
            {
                "mode": "rss",
                "case": case,
                "growth_bytes": growth,
                "excess_bytes": excess,
                "budget_bytes": budget,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"rss/{case}: median growth {growth:.0f} exceeds {budget} bytes")
    for case in PEAK_OUTPUTS:
        layout = expected_output(case)
        output_bytes = layout["output_bytes"]
        records = [sample for sample in samples if sample["mode"] == "peak" and sample["case"] == case]
        if not records:
            continue
        peak = statistics.median(sample["working_peak_bytes"] for sample in records)
        budget = 8 * MIB + 2 * output_bytes
        matches_output = all(all(sample[key] == value for key, value in layout.items()) for sample in records)
        passed = peak <= budget and matches_output
        cases.append(
            {"mode": "peak", "case": case, "working_peak_bytes": peak, "budget_bytes": budget, "passed": passed}
        )
        if not passed:
            failures.append(f"peak/{case}: working peak exceeds its budget or output shape/dtype/storage differs")
    return {"status": "failed" if failures else "passed", "cases": cases, "failures": failures}


def selected_cases(mode: str) -> list[tuple[str, str]]:
    cases = [("rss", case) for case in RSS_CASES] if mode != "peak" else []
    if mode != "rss":
        cases.extend(("peak", case) for case in PEAK_OUTPUTS)
    return cases


def run_probes(mode: str, repeats: int) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), environment.get("PYTHONPATH"))))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    samples = []
    for repeat in range(repeats):
        for measurement_mode, case in selected_cases(mode):
            command = [sys.executable, str(REPO_ROOT / "tools" / "_memory_probe.py"), measurement_mode, case]
            try:
                process = subprocess.run(  # noqa: S603 - fixed worker path, interpreter, and registered case names.
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=120,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"{measurement_mode}/{case} probe failed:\n{error.stderr}") from error
            sample = json.loads(process.stdout)
            sample["repeat"] = repeat + 1
            samples.append(sample)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rss", "peak", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux /proc memory accounting is required")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    samples = run_probes(args.mode, args.repeats)
    report = {"schema_version": 1, **classify_samples(samples, args.mode, args.repeats), "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(f"{case['mode']}/{case['case']}: {'passed' if case['passed'] else 'FAILED'}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
