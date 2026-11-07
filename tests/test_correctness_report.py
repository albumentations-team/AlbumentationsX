"""Tests for correctness report evidence handling."""

from __future__ import annotations

import json

import pytest
from packaging.version import Version

from tools.check_behavior_changes import NOTES_PATH, REPO_ROOT, read_inventory
from tools.generate_correctness_report import _validate_inventory_revision, _validate_required_evidence, generate_report
from tools.release_bundle import _git


def _write_json(path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n")


def test_generate_report_requires_release_evidence_by_default(tmp_path) -> None:
    _write_json(tmp_path / "environment-local.json", {})

    with pytest.raises(ValueError, match="benchmark coverage summary JSON"):
        generate_report(tmp_path)


@pytest.fixture
def release_evidence(tmp_path, monkeypatch):
    """Provide passing artifacts and a candidate version newer than the inventory baseline."""
    baseline = Version(read_inventory(REPO_ROOT)[0])
    version = f"{baseline.major}.{baseline.minor + 1}.0"
    monkeypatch.setattr("tools.generate_correctness_report._project_version", lambda: version)
    revision = "a" * 40
    recorded_notes = (REPO_ROOT / NOTES_PATH).read_bytes()

    def git_snapshot(root, *args):
        """Keep unit evidence independent of uncommitted edits to the working inventory."""
        if args == ("show", f"{revision}:{NOTES_PATH.as_posix()}"):
            return recorded_notes
        return _git(root, *args)

    monkeypatch.setattr("tools.generate_correctness_report._git", git_snapshot)
    _write_json(
        tmp_path / "environment-local.json",
        {"git_commit": revision, "github": {"run_id": "137"}},
    )
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
    validation_path = tmp_path / "validation-contract.json"
    validation_summary = {
        "source": "junit-validation-contract.xml",
        "status": "ok",
        "totals": {"passed": 35, "failures": 0, "errors": 0, "skipped": 0},
    }
    _write_json(validation_path, validation_summary)
    _write_json(
        tmp_path / "geometry-contract.json",
        {
            "source": "junit-geometry-contract.xml",
            "status": "ok",
            "totals": {"passed": 44, "failures": 0, "errors": 0, "skipped": 0},
        },
    )
    _write_json(
        tmp_path / "determinism-contract.json",
        {
            "source": "junit-determinism-contract.xml",
            "status": "ok",
            "totals": {"passed": 137, "failures": 0, "errors": 0, "skipped": 0},
        },
    )
    return tmp_path


def test_generate_report_accepts_required_release_evidence(release_evidence) -> None:
    report = generate_report(release_evidence)

    assert "Benchmark coverage detail" in report
    assert "Benchmark coverage diff" in report
    assert "Performance budget" in report
    assert "10 passed, 0 failed, 0 errors, 1 skipped" in report
    assert "Validation contract: 35 passed, 0 failed, 0 errors, 0 skipped" in report
    revision = json.loads((release_evidence / "environment-local.json").read_text())["git_commit"]
    docs_path = f"https://github.com/albumentations-team/AlbumentationsX/blob/{revision}/docs"
    assert f"[2D geometry contract]({docs_path}/geometry.md): 44 passed, 0 failed, 0 errors, 0 skipped" in report
    assert (
        f"[Determinism and replay contract]({docs_path}/determinism.md): 137 passed, 0 failed, 0 errors, 0 skipped"
        in report
    )
    assert f"[Composition audit and permanent tests]({docs_path}/maintaining/determinism-audit.md)" in report
    assert f"[release notes]({docs_path}/release-notes/unreleased.md)" in report
    assert "[CI run 137](https://github.com/albumentations-team/AlbumentationsX/actions/runs/137)" in report
    assert "batch: covered=1, tracked_without_dedicated_matrix=2" in report
    assert "memory: covered_advisory=1" in report
    assert "0 coverage contract failure(s)" in report
    assert "Security summary: provided" in report
    assert "CodeQL: managed through path-scoped advanced workflows" in report
    assert "ASV comparison refs: `2.4.2` -> `release-head`" in report

    validation_path = release_evidence / "validation-contract.json"
    validation_summary = json.loads(validation_path.read_text())
    validation_summary["totals"]["skipped"] = 1
    _write_json(validation_path, validation_summary)
    with pytest.raises(ValueError, match="validation contract evidence must show a nonempty suite with no skipped"):
        generate_report(release_evidence)

    validation_path.unlink()
    with pytest.raises(ValueError, match=r"Missing required release evidence artifact: validation-contract\.json"):
        generate_report(release_evidence)


def test_generate_report_rejects_unavailable_inventory_revision(release_evidence) -> None:
    _write_json(release_evidence / "environment-local.json", {"git_commit": "b" * 40})

    with pytest.raises(ValueError, match="Cannot verify behavior inventory at tested revision"):
        generate_report(release_evidence)


@pytest.mark.parametrize("revision", [None, "main"])
def test_release_report_requires_an_immutable_source_revision(release_evidence, revision) -> None:
    _write_json(release_evidence / "environment-local.json", {"git_commit": revision})

    with pytest.raises(ValueError, match="must identify"):
        generate_report(release_evidence)


def test_release_report_rejects_mixed_environment_revisions(release_evidence) -> None:
    _write_json(release_evidence / "environment-other.json", {"git_commit": "b" * 40})

    with pytest.raises(ValueError, match="same tested source revision"):
        generate_report(release_evidence)


def test_inventory_revision_check_rejects_post_test_edits(tmp_path) -> None:
    """A release must not attribute later migration edits to an earlier test run."""
    notes = tmp_path / NOTES_PATH
    notes.parent.mkdir(parents=True)
    notes.write_text("Recorded inventory\n", encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@example.com")
    _git(tmp_path, "config", "user.name", "CI")
    _git(tmp_path, "add", NOTES_PATH.as_posix())
    _git(tmp_path, "commit", "-m", "Record migration inventory")
    revision = _git(tmp_path, "rev-parse", "HEAD").decode().strip()

    notes.write_bytes(b"Recorded inventory\r\n")
    _validate_inventory_revision(tmp_path, revision)

    notes.write_text("Edited after evidence collection\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Behavior inventory differs from tested revision"):
        _validate_inventory_revision(tmp_path, revision)


@pytest.mark.parametrize("contract", ["geometry", "determinism"])
@pytest.mark.parametrize(
    ("totals", "match"),
    [
        pytest.param(None, r"Missing required release evidence artifact: {contract}-contract\.json", id="missing"),
        pytest.param(
            {"passed": 0, "failures": 0, "errors": 0, "skipped": 0},
            "{contract} contract evidence must show a nonempty suite",
            id="empty",
        ),
        pytest.param(
            {"passed": 43, "failures": 1, "errors": 0, "skipped": 0},
            "{contract} contract evidence is incomplete or failing",
            id="failing",
        ),
        pytest.param(
            {"passed": 43, "failures": 0, "errors": 0, "skipped": 1},
            "{contract} contract evidence must show a nonempty suite with no skipped tests",
            id="skipped",
        ),
    ],
)
def test_generate_report_requires_passing_contract_evidence(release_evidence, contract, totals, match) -> None:
    path = release_evidence / f"{contract}-contract.json"
    if totals is None:
        path.unlink()
    else:
        _write_json(path, {"source": f"junit-{contract}-contract.xml", "status": "ok", "totals": totals})

    with pytest.raises(ValueError, match=match.format(contract=contract)):
        generate_report(release_evidence)


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
