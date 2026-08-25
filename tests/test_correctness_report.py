"""Tests for correctness report evidence handling."""

from __future__ import annotations

import json

import pytest

from tools.generate_correctness_report import _validate_required_evidence, generate_report


def _write_json(path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n")


def test_generate_report_requires_release_evidence_by_default(tmp_path) -> None:
    _write_json(tmp_path / "environment-local.json", {})

    with pytest.raises(ValueError, match="benchmark coverage summary JSON"):
        generate_report(tmp_path)


def test_generate_report_accepts_required_release_evidence(tmp_path) -> None:
    _write_json(tmp_path / "environment-local.json", {})
    _write_json(
        tmp_path / "pytest-summary-local.json",
        {
            "kind": "pytest-summary",
            "missing": False,
            "source": "junit.xml",
            "status": "ok",
            "totals": {"errors": 0, "failures": 0, "passed": 10, "skipped": 1, "tests": 11, "time": 1.0},
        },
    )
    _write_json(
        tmp_path / "benchmark-coverage-local.json",
        {
            "asv_cases": 1,
            "coverage_depth": {"contract_failures": 0, "deep_coverage_transforms": 1},
            "direct_kernel_cases": {},
            "full_matrix_cases": {},
            "memory_benchmarks": 0,
            "public_transforms": 1,
            "pytorch_tensor_benchmark_cases": 0,
        },
    )
    _write_json(
        tmp_path / "benchmark-coverage-detail-local.json",
        {
            "kind": "benchmark-coverage-detail",
            "layer_counts": {},
            "summary": {
                "contract_failures": 0,
                "deep_coverage_transforms": 1,
                "dedicated_tensor_transforms": 0,
                "performance_contract_status_counts": {
                    "batch": {"covered": 1, "tracked_without_dedicated_matrix": 2},
                    "memory": {"covered_advisory": 1},
                },
                "smoke_only_transforms": 0,
            },
        },
    )
    _write_json(
        tmp_path / "benchmark-coverage-diff-local.json",
        {
            "kind": "benchmark-coverage-diff",
            "schema_version": 1,
            "summary": {
                "added_transforms": 0,
                "changed_transforms": 0,
                "removed_transforms": 0,
                "status": "ok",
            },
        },
    )
    _write_json(
        tmp_path / "benchmark-performance-budget-local.json",
        {
            "coverage": {
                "contract_failures": 0,
                "public_transforms": 1,
                "smoke_only_transforms": 0,
            },
            "comparison": {
                "provided": True,
                "release_blockers": [],
                "triage_items": [],
            },
            "kind": "performance-budget",
            "status": "ok",
        },
    )
    _write_json(
        tmp_path / "benchmark-asv-summary-local.json",
        {
            "kind": "asv-comparison",
            "missing": False,
            "asv_exit_code": 0,
            "status": {"performance_decreased": False},
            "totals": {"changed": 0, "improvements": 0, "regressions": 0},
        },
    )
    (tmp_path / "benchmark-baseline-ref.txt").write_text("2.4.2\n")
    (tmp_path / "benchmark-candidate-ref.txt").write_text("release-head\n")
    _write_json(tmp_path / "security-local.json", {})

    report = generate_report(tmp_path)

    assert "Benchmark coverage detail" in report
    assert "Benchmark coverage diff" in report
    assert "Performance budget" in report
    assert "10 passed, 0 failed, 0 errors, 1 skipped" in report
    assert "batch: covered=1, tracked_without_dedicated_matrix=2" in report
    assert "memory: covered_advisory=1" in report
    assert "0 coverage contract failure(s)" in report
    assert "Security summary: provided" in report
    assert "CodeQL: managed through path-scoped advanced workflows" in report
    assert "ASV comparison refs: `2.4.2` -> `release-head`" in report


@pytest.mark.parametrize(
    ("summary", "match"),
    [
        (
            {
                "kind": "pytest-summary",
                "missing": True,
                "source": "junit.xml",
                "status": "missing",
                "totals": {"errors": 1, "failures": 0, "passed": 0, "skipped": 0, "tests": 0, "time": 0.0},
            },
            "junit.xml is missing",
        ),
        (
            {
                "kind": "pytest-summary",
                "missing": False,
                "source": "junit.xml",
                "status": "ok",
                "totals": {"errors": 0, "failures": 1, "passed": 9, "skipped": 0, "tests": 10, "time": 1.0},
            },
            "junit.xml has 1 failure",
        ),
    ],
)
def test_generate_report_rejects_incomplete_or_failing_pytest_evidence(tmp_path, summary: dict, match: str) -> None:
    _write_json(tmp_path / "environment-local.json", {})
    _write_json(tmp_path / "pytest-summary-local.json", summary)
    _write_json(
        tmp_path / "benchmark-coverage-local.json",
        {
            "asv_cases": 1,
            "coverage_depth": {"contract_failures": 0, "deep_coverage_transforms": 1},
            "direct_kernel_cases": {},
            "full_matrix_cases": {},
            "memory_benchmarks": 0,
            "public_transforms": 1,
            "pytorch_tensor_benchmark_cases": 0,
        },
    )
    _write_json(
        tmp_path / "benchmark-coverage-detail-local.json",
        {
            "kind": "benchmark-coverage-detail",
            "layer_counts": {},
            "summary": {"contract_failures": 0, "deep_coverage_transforms": 1, "dedicated_tensor_transforms": 0},
        },
    )
    _write_json(
        tmp_path / "benchmark-performance-budget-local.json",
        {
            "coverage": {"contract_failures": 0, "public_transforms": 1, "smoke_only_transforms": 0},
            "comparison": {"provided": True, "release_blockers": [], "triage_items": []},
            "kind": "performance-budget",
            "status": "ok",
        },
    )
    _write_json(
        tmp_path / "benchmark-asv-summary-local.json",
        {
            "kind": "asv-comparison",
            "missing": False,
            "asv_exit_code": 0,
            "status": {"performance_decreased": False},
            "totals": {"changed": 0, "improvements": 0, "regressions": 0},
        },
    )
    _write_json(tmp_path / "security-local.json", {})

    with pytest.raises(ValueError, match=match):
        generate_report(tmp_path)


def test_generate_report_rejects_missing_asv_comparison_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="ASV comparison summary JSON"):
        _validate_required_evidence(
            [{}],
            [{"missing": False, "status": "ok", "totals": {}}],
            [
                {"asv_cases": 1, "coverage_depth": {}, "memory_benchmarks": 1},
                {"kind": "benchmark-coverage-detail", "summary": {}, "layer_counts": {}},
                {"kind": "performance-budget", "comparison": {"provided": True}},
            ],
            [{}],
            allow_missing_evidence=False,
        )


@pytest.mark.parametrize("comparison", [{"provided": False}, None])
def test_generate_report_rejects_budget_without_provided_comparison(comparison: dict | None) -> None:
    with pytest.raises(ValueError, match=r"comparison\.provided=true"):
        _validate_required_evidence(
            [{}],
            [{"missing": False, "status": "ok", "totals": {}}],
            [
                {"asv_cases": 1, "coverage_depth": {}, "memory_benchmarks": 1},
                {"kind": "benchmark-coverage-detail", "summary": {}, "layer_counts": {}},
                {"kind": "asv-comparison", "missing": False},
                {"kind": "performance-budget", "comparison": comparison},
            ],
            [{}],
            allow_missing_evidence=False,
        )
