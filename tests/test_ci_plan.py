"""Tests for pull-request CI path classification and selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci_plan import _read_github_files, build_plan, classify_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("docs/maintaining/ci-policy.md", {"docs", "ci_policy"}),
        ("albumentations/core/composition.py", {"runtime"}),
        ("albumentations/pytorch/transforms.py", {"runtime", "pytorch"}),
        ("tests/conftest.py", {"tests", "shared_tests"}),
        ("benchmark/benchmarks/augmentations.py", {"benchmarks"}),
        ("LICENSE", {"legal"}),
        ("LICENSE_HISTORY.md", {"docs", "legal"}),
        ("LICENSING.md", {"docs", "legal"}),
        ("THIRD_PARTY_LICENSES/OFL-1.1.txt", {"legal"}),
        ("pyproject.toml", {"dependencies", "legal", "packaging", "quality_config"}),
        (".github/workflows/pr.yml", {"workflows", "ci_tooling", "self_ci"}),
        ("unclassified/new-area/data.bin", {"unknown"}),
    ],
)
def test_classify_path(path: str, expected: set[str]) -> None:
    assert set(classify_path(path)) == expected


def test_markdown_only_plan_never_selects_python_tests() -> None:
    plan = build_plan(["docs/design/mosaic.md"])

    assert plan.checks["markdown"]
    assert plan.gates["correctness"] == ()
    assert plan.gates["policy"] == ()
    assert not plan.advisory_asv
    assert not plan.antigravity


def test_readme_runs_package_policy_without_product_tests() -> None:
    plan = build_plan(["README.md"])

    assert plan.checks["markdown"]
    assert plan.checks["package"]
    assert not plan.checks["install_smoke"]
    assert plan.gates["correctness"] == ()


def test_forbidden_license_history_path_runs_legal_policy() -> None:
    plan = build_plan(["LICENSE_HISTORY.md"])

    assert plan.checks["legal"]


def test_runtime_plan_keeps_full_compatibility_and_performance_evidence() -> None:
    plan = build_plan(["albumentations/core/composition.py"])

    assert plan.checks["compatibility"]
    assert plan.checks["coverage"]
    assert plan.checks["lint"]
    assert plan.checks["mypy"]
    assert plan.checks["pyrefly"]
    assert plan.checks["contracts"]
    assert not plan.checks["primary"]
    assert plan.advisory_asv
    assert plan.antigravity


def test_core_runtime_change_also_selects_dedicated_pytorch_coverage() -> None:
    plan = build_plan(["albumentations/core/composition.py"])

    assert plan.checks["pytorch"]


def test_benchmark_tooling_selects_dedicated_pytorch_policy() -> None:
    plan = build_plan(["tools/benchmark_coverage.py"])

    assert plan.checks["pytorch"]


def test_isolated_test_change_selects_targeted_compatibility() -> None:
    plan = build_plan(["tests/test_bbox.py"])

    assert plan.checks["targeted"]
    assert not plan.checks["compatibility"]
    assert plan.pytest_targets == ("tests/test_bbox.py",)


def test_shared_test_change_falls_back_to_full_matrix() -> None:
    plan = build_plan(["tests/helpers/data.py"])

    assert plan.checks["compatibility"]
    assert not plan.checks["targeted"]


def test_dependency_change_selects_primary_install_and_security_checks() -> None:
    plan = build_plan(["uv.lock"])

    assert plan.checks["primary"]
    assert plan.checks["pytorch"]
    assert plan.checks["install_smoke"]
    assert plan.checks["dependency_audit"]
    assert plan.checks["legal"]
    assert not plan.checks["compatibility"]


def test_workflow_change_runs_security_contracts_without_product_matrix() -> None:
    plan = build_plan([".github/workflows/performance.yml"])

    assert plan.checks["workflow_audit"]
    assert plan.checks["contracts"]
    assert not plan.checks["compatibility"]
    assert plan.antigravity


def test_quality_configuration_runs_tools_without_product_tests() -> None:
    plan = build_plan([".pre-commit-config.yaml"])

    assert plan.checks["lint"]
    assert plan.checks["mypy"]
    assert plan.checks["pyrefly"]
    assert plan.checks["contracts"]
    assert plan.gates["correctness"] == ()


def test_self_ci_change_runs_complete_compatibility_matrix() -> None:
    plan = build_plan(["tools/ci_plan.py"])

    assert plan.checks["compatibility"]
    assert plan.checks["contracts"]


def test_unknown_path_fails_closed_to_complete_profile() -> None:
    plan = build_plan(["new-root/file.dat"])

    assert plan.checks["compatibility"]
    assert plan.checks["pytorch"]
    assert plan.checks["workflow_audit"]
    assert plan.checks["package"]
    assert "unknown" in plan.domains


def test_draft_disables_heavy_checks_but_preserves_fast_feedback() -> None:
    plan = build_plan(["albumentations/core/composition.py"], draft=True)

    assert plan.gates["fast"]
    assert plan.gates["correctness"] == ()
    assert plan.gates["policy"] == ()
    assert not plan.advisory_asv
    assert not plan.antigravity


def test_force_full_overrides_draft_and_selects_conservative_profile() -> None:
    plan = build_plan(["docs/design/mosaic.md"], draft=True, force_full=True)

    assert plan.checks["compatibility"]
    assert plan.checks["pytorch"]
    assert plan.checks["package"]
    assert not plan.checks["primary"]
    assert not plan.checks["targeted"]


def test_torch_only_module_uses_dedicated_profile_without_targeted_base_job() -> None:
    plan = build_plan(["tests/test_pytorch.py"])

    assert plan.checks["pytorch"]
    assert not plan.checks["targeted"]
    assert plan.pytest_targets == ()


def test_github_file_reader_preserves_untrusted_filename_boundaries(tmp_path: Path) -> None:
    files_json = tmp_path / "files.json"
    files_json.write_text(
        '[[{"filename":"docs/normal.md"}], [{"filename":"docs/line\\nbreak.md"}]]',
        encoding="utf-8",
    )

    assert _read_github_files(files_json) == ["docs/normal.md", "docs/line\nbreak.md"]
    assert "unknown" in build_plan(_read_github_files(files_json)).domains


def test_version_increase_selects_complete_release_preflight() -> None:
    plan = build_plan(["pyproject.toml"], base_version="2.3.2", head_version="2.3.3")

    assert plan.base_version == "2.3.2"
    assert plan.head_version == "2.3.3"
    assert plan.version_change == "increase"
    assert plan.checks["release_preflight"]
    assert plan.checks["compatibility"]
    assert plan.checks["pytorch"]
    assert "release-preflight" in plan.gates["policy"]


def test_prerelease_version_increase_selects_release_preflight() -> None:
    plan = build_plan(["pyproject.toml"], base_version="2.3.2", head_version="2.3.3rc1")

    assert plan.version_change == "increase"
    assert plan.checks["release_preflight"]


def test_unchanged_version_does_not_select_release_preflight() -> None:
    plan = build_plan(["pyproject.toml"], base_version="2.3.2", head_version="2.3.2")

    assert plan.version_change == "unchanged"
    assert not plan.checks["release_preflight"]


def test_draft_version_bump_defers_release_preflight() -> None:
    plan = build_plan(["pyproject.toml"], base_version="2.3.2", head_version="2.3.3", draft=True)

    assert plan.version_change == "increase"
    assert not plan.checks["release_preflight"]
    assert plan.gates["policy"] == ()


def test_version_decrease_is_recorded_as_invalid_release_direction() -> None:
    plan = build_plan(["pyproject.toml"], base_version="2.4.0", head_version="2.3.3")

    assert plan.version_change == "decrease"
    assert not plan.checks["release_preflight"]


@pytest.mark.parametrize("version", ["", "release-2.3.3", "2.3.3+"])
def test_invalid_project_version_fails_closed(version: str) -> None:
    with pytest.raises(ValueError, match="Invalid PEP 440 version"):
        build_plan(["pyproject.toml"], base_version="2.3.2", head_version=version)
