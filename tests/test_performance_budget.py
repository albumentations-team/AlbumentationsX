"""Tests for benchmark performance-budget policy evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import performance_budget
from tools.performance_budget import build_budget, classify_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_BUDGET_SCRIPT = REPO_ROOT / "tools" / "performance_budget.py"


def _coverage_detail() -> dict:
    return {
        "kind": "benchmark-coverage-detail",
        "layer_counts": {
            "annotation_scaling": 7,
            "batch_matrix": 11,
            "catalog_smoke": 134,
            "direct_kernel": 28,
            "family_matrix": 109,
            "memory": 9,
            "parameter_sensitivity": 6,
            "pytorch_tensor": 2,
            "reference_data": 7,
            "target_matrix": 6,
            "volumetric_matrix": 7,
        },
        "public_transforms": 136,
        "summary": {
            "contract_failures": 0,
            "deep_coverage_transforms": 134,
            "optional_transforms": 2,
            "smoke_only_transforms": 0,
        },
    }


def _coverage_summary() -> dict:
    return {
        "coverage_depth": {
            "contract_failures": 0,
            "deep_coverage_transforms": 134,
            "optional_transforms": 2,
            "smoke_only_transforms": 0,
        },
        "memory_benchmarks": 7,
        "public_transforms": 136,
        "pytorch_tensor_benchmark_cases": 2,
    }


def test_classify_benchmark_maps_stable_transform_matrix() -> None:
    policy = classify_benchmark("benchmarks.test_family_matrix.TimePixelFullMatrix.time_transform('blur|small|3')")

    assert policy.benchmark_class == "pixel_matrix"
    assert policy.release_blocking is True


def test_classify_benchmark_maps_batch_matrix_as_release_blocking() -> None:
    policy = classify_benchmark("benchmarks.test_batch_matrix.TimeImageBatchMatrix.time_transform('resize|images')")

    assert policy.benchmark_class == "batch_matrix"
    assert policy.release_blocking is True


def test_classify_benchmark_maps_parameter_sensitivity_as_required_advisory() -> None:
    policy = classify_benchmark(
        "benchmarks.test_parameter_sensitivity.TimeParameterSensitivity.time_transform('blur_kernel_15')",
    )

    assert policy.benchmark_class == "parameter_sensitivity"
    assert policy.release_blocking is False


def test_build_budget_marks_large_stable_regression_as_release_blocker() -> None:
    budget = build_budget(
        _coverage_summary(),
        _coverage_detail(),
        {
            "asv_exit_code": 1,
            "regressions": [
                {
                    "after": "13.2ms",
                    "before": "10.0ms",
                    "benchmark": "benchmarks.test_core_pipeline.TimeCorePipeline.time_single_transform_compose",
                    "change": "+",
                    "ratio": 1.32,
                },
            ],
            "totals": {"changed": 1, "improvements": 0, "regressions": 1},
        },
    )

    assert budget["status"] == "release_blocked"
    assert budget["comparison"]["release_blockers"][0]["benchmark_class"] == "core_pipeline"


def test_build_budget_keeps_memory_regression_as_triage_item() -> None:
    budget = build_budget(
        _coverage_summary(),
        _coverage_detail(),
        {
            "asv_exit_code": 1,
            "regressions": [
                {
                    "after": "820MiB",
                    "before": "700MiB",
                    "benchmark": "benchmarks.test_family_matrix.PeakMemoryHotPaths.peakmem_resize_large_rgb",
                    "change": "+",
                    "ratio": 1.17,
                },
            ],
            "totals": {"changed": 1, "improvements": 0, "regressions": 1},
        },
    )

    assert budget["status"] == "triage_required"
    assert budget["comparison"]["triage_items"][0]["benchmark_class"] == "memory"
    assert budget["comparison"]["release_blockers"] == []


def test_build_budget_requires_comparison_when_requested() -> None:
    budget = build_budget(_coverage_summary(), _coverage_detail(), require_comparison=True)

    assert budget["status"] == "missing_comparison"
    assert "ASV comparison summary is required" in budget["issues"][0]


def test_build_budget_rejects_missing_required_coverage_layers() -> None:
    coverage_detail = _coverage_detail()
    coverage_detail["layer_counts"] = {**coverage_detail["layer_counts"], "family_matrix": 0}

    budget = build_budget(_coverage_summary(), coverage_detail)

    assert budget["status"] == "release_blocked"
    assert any("family_matrix" in issue for issue in budget["issues"])


def test_build_budget_allows_optional_layer_only_in_core_mode() -> None:
    coverage_detail = _coverage_detail()
    coverage_detail["layer_counts"] = {**coverage_detail["layer_counts"], "pytorch_tensor": 0}
    coverage_detail["public_transforms"] = 134
    coverage_detail["summary"] = {**coverage_detail["summary"], "optional_transforms": 0}
    coverage_summary = _coverage_summary()
    coverage_summary["public_transforms"] = 134
    coverage_summary["pytorch_tensor_benchmark_cases"] = 0

    assert build_budget(coverage_summary, coverage_detail, require_optional=False)["status"] == "ok"
    assert build_budget(coverage_summary, coverage_detail, require_optional=True)["status"] == "release_blocked"


def test_summarize_prints_concrete_release_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    coverage_detail = _coverage_detail()
    coverage_detail["layer_counts"] = {**coverage_detail["layer_counts"], "family_matrix": 0}
    summary_path = tmp_path / "summary.json"
    detail_path = tmp_path / "detail.json"
    output_path = tmp_path / "budget.json"
    summary_path.write_text(json.dumps(_coverage_summary()), encoding="utf-8")
    detail_path.write_text(json.dumps(coverage_detail), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "performance_budget.py",
            "summarize",
            "--coverage-summary",
            str(summary_path),
            "--coverage-detail",
            str(detail_path),
            "--output",
            str(output_path),
            "--fail-on-release-blockers",
        ],
    )

    assert performance_budget.main() == 1
    assert "missing required benchmark coverage layer(s): family_matrix" in capsys.readouterr().out


def test_direct_script_ignores_unrelated_tools_package(tmp_path: Path) -> None:
    shadow_package = tmp_path / "tools"
    shadow_package.mkdir()
    (shadow_package / "__init__.py").write_text('"""Unrelated package."""\n', encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(PERFORMANCE_BUDGET_SCRIPT), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Classify benchmark coverage and ASV comparison evidence" in result.stdout
