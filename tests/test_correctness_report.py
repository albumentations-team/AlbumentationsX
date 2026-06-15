"""Tests for correctness report evidence handling."""

from __future__ import annotations

import json

import pytest

from tools.generate_correctness_report import generate_report


def _write_json(path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n")


def test_generate_report_requires_release_evidence_by_default(tmp_path) -> None:
    _write_json(tmp_path / "environment-local.json", {})

    with pytest.raises(ValueError, match="benchmark coverage summary JSON"):
        generate_report(tmp_path)


def test_generate_report_accepts_required_release_evidence(tmp_path) -> None:
    _write_json(tmp_path / "environment-local.json", {})
    _write_json(
        tmp_path / "benchmark-coverage-local.json",
        {
            "asv_cases": 1,
            "coverage_depth": {"deep_coverage_transforms": 1},
            "direct_kernel_cases": {},
            "full_matrix_cases": {},
            "memory_benchmarks": 0,
            "public_transforms": 1,
        },
    )
    _write_json(
        tmp_path / "benchmark-coverage-detail-local.json",
        {
            "kind": "benchmark-coverage-detail",
            "layer_counts": {},
            "summary": {
                "deep_coverage_transforms": 1,
                "optional_transforms": 0,
                "smoke_only_transforms": 0,
            },
        },
    )
    _write_json(tmp_path / "security-local.json", {})

    report = generate_report(tmp_path)

    assert "Benchmark coverage detail" in report
    assert "Security summary: provided" in report
