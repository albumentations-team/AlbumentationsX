"""Memory evidence must reject retained buffers, excess workspace, and incomplete runs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tools.check_memory import MIB, PEAK_OUTPUT_BYTES, RSS_CASES, classify_samples
from tools.ci_plan import classify_path


def _samples() -> list[dict]:
    return [{"mode": "rss", "case": case, "growth_bytes": MIB} for case in RSS_CASES for _ in range(3)] + [
        {"mode": "peak", "case": case, "working_peak_bytes": output_bytes, "output_bytes": output_bytes}
        for case, output_bytes in PEAK_OUTPUT_BYTES.items()
        for _ in range(3)
    ]


@pytest.mark.parametrize("case", RSS_CASES)
def test_memory_budget_rejects_retained_full_image_buffers(case: str) -> None:
    samples = _samples()
    for sample in samples:
        if sample["mode"] == "rss" and sample["case"] == case:
            sample["growth_bytes"] = 64 * MIB

    report = classify_samples(samples, "all", 3)

    assert report["status"] == "failed"
    assert len(report["failures"]) == 1
    assert f"rss/{case}" in report["failures"][0]


@pytest.mark.parametrize("case", PEAK_OUTPUT_BYTES)
def test_memory_budget_rejects_excess_native_workspace(case: str) -> None:
    samples = _samples()
    for sample in samples:
        if sample["mode"] == "peak" and sample["case"] == case:
            sample["working_peak_bytes"] = 9 * MIB + 2 * PEAK_OUTPUT_BYTES[case]

    report = classify_samples(samples, "all", 3)

    assert report["status"] == "failed"
    assert len(report["failures"]) == 1
    assert f"peak/{case}" in report["failures"][0]


@pytest.mark.parametrize("mode", ["rss", "peak", "all"])
def test_memory_budget_requires_every_requested_repeat(mode: str) -> None:
    samples = [sample for sample in _samples() if mode == "all" or sample["mode"] == mode]
    assert classify_samples(samples, mode, 3)["status"] == "passed"

    samples.pop()
    assert classify_samples(samples, mode, 3)["status"] == "failed"
    assert classify_samples([], mode, 3)["status"] == "failed"


def test_memory_budget_preserves_raw_samples_and_handles_allocator_noise() -> None:
    samples = _samples()
    samples[0]["growth_bytes"] = 128 * MIB
    samples[1]["growth_bytes"] = -MIB
    expected = copy.deepcopy(samples)

    report = classify_samples(samples, "all", 3)

    assert report["status"] == "passed"
    assert samples == expected


def test_memory_budget_rejects_wrong_output_storage() -> None:
    samples = _samples()
    next(sample for sample in samples if sample["mode"] == "peak")["output_bytes"] = 0

    report = classify_samples(samples, "all", 3)

    assert report["status"] == "failed"
    assert "peak/blur" in report["failures"][0]


@pytest.mark.parametrize("path", ["tools/check_memory.py", "tools/_memory_probe.py"])
def test_memory_tool_changes_require_product_ci(path: str) -> None:
    assert classify_path(path) == frozenset({"ci_tooling"})


@pytest.mark.parametrize(
    ("workflow", "job"),
    [
        ("pr.yml", "compatibility"),
        ("pr.yml", "release_preflight"),
        ("performance.yml", "scheduled_core_comparison"),
        ("release-candidate.yml", "release_candidate"),
    ],
)
def test_memory_budgets_are_required_in_ci(workflow: str, job: str) -> None:
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / workflow
    steps = yaml.safe_load(path.read_text())["jobs"][job]["steps"]
    memory_step = next(step for step in steps if "tools/check_memory.py" in step.get("run", ""))

    assert "--output" in memory_step["run"]
    assert not memory_step.get("continue-on-error", False)
    if job in {"compatibility", "release_candidate"}:
        assert memory_step["if"] == "runner.os == 'Linux' && matrix.python-version == '3.11'"
