"""Contracts for pull-request CI path classification and selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci_plan import CHECK_NAMES, _read_github_files, build_plan, classify_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("docs/maintaining/ci-policy.md", {"docs", "ci_policy"}),
        ("albumentations/core/composition.py", {"runtime"}),
        ("albumentations/pytorch/transforms.py", {"runtime", "pytorch"}),
        ("tests/conftest.py", {"tests", "shared_tests"}),
        ("benchmark/benchmarks/augmentations.py", {"benchmarks"}),
        ("LICENSE", {"legal"}),
        ("pyproject.toml", {"dependencies", "legal", "packaging", "quality_config"}),
        (".github/workflows/pr.yml", {"workflows", "ci_tooling", "self_ci"}),
        ("unclassified/new-area/data.bin", {"unknown"}),
    ],
)
def test_classify_path(path: str, expected: set[str]) -> None:
    assert set(classify_path(path)) == expected


def test_docs_only_plan_does_not_invent_product_work() -> None:
    plan = build_plan(["docs/design/mosaic.md"])

    assert not any(plan.checks.values())


def test_readme_runs_package_and_legal_checks_without_product_tests() -> None:
    plan = build_plan(["README.md"])

    assert plan.checks["package"]
    assert plan.checks["legal"]
    assert not plan.checks["compatibility"]


def test_legal_change_runs_package_and_legal_checks() -> None:
    plan = build_plan(["LICENSE"])

    assert plan.checks["legal"]
    assert plan.checks["package"]
    assert not plan.checks["compatibility"]


def test_runtime_plan_keeps_matrix_and_core_tensor_coverage() -> None:
    plan = build_plan(["albumentations/core/composition.py"])

    assert plan.checks["compatibility"]
    assert plan.checks["pytorch"]
    assert not plan.checks["targeted"]


def test_benchmark_tooling_does_not_start_product_or_tensor_tests() -> None:
    plan = build_plan(["tools/benchmark_coverage.py"])

    assert not plan.checks["compatibility"]
    assert not plan.checks["pytorch"]


def test_isolated_test_change_selects_targeted_tests() -> None:
    plan = build_plan(["tests/test_bbox.py"])

    assert plan.checks["targeted"]
    assert not plan.checks["compatibility"]
    assert plan.pytest_targets == ("tests/test_bbox.py",)


def test_shared_test_change_selects_full_matrix() -> None:
    plan = build_plan(["tests/helpers/data.py"])

    assert plan.checks["compatibility"]
    assert not plan.checks["targeted"]


def test_dependency_change_selects_matrix_and_policy_without_duplicate_primary_suite() -> None:
    plan = build_plan(["uv.lock"])

    assert plan.checks["compatibility"]
    assert plan.checks["pytorch"]
    assert plan.checks["dependency_audit"]
    assert plan.checks["legal"]
    assert plan.checks["package"]


def test_workflow_change_runs_workflow_audit_without_product_matrix() -> None:
    plan = build_plan([".github/workflows/performance.yml"])

    assert plan.checks["workflow_audit"]
    assert not plan.checks["compatibility"]


def test_self_ci_change_runs_full_matrix() -> None:
    plan = build_plan(["tools/ci_plan.py"])

    assert plan.checks["compatibility"]


def test_unknown_path_fails_closed() -> None:
    plan = build_plan(["new-root/file.dat"])

    assert all(plan.checks[name] for name in CHECK_NAMES if name not in {"targeted", "release_preflight"})
    assert not plan.checks["targeted"]


def test_draft_defers_routed_jobs_but_not_always_run_pre_commit() -> None:
    plan = build_plan(["albumentations/core/composition.py"], draft=True)

    assert not any(plan.checks.values())


def test_force_full_overrides_draft() -> None:
    plan = build_plan(["docs/design/mosaic.md"], draft=True, force_full=True)

    assert plan.checks["compatibility"]
    assert plan.checks["pytorch"]
    assert plan.checks["package"]


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


def test_version_increase_selects_release_preflight() -> None:
    plan = build_plan(["pyproject.toml"], base_version="2.3.2", head_version="2.3.3")

    assert plan.version_change == "increase"
    assert plan.checks["release_preflight"]
    assert plan.checks["compatibility"]


def test_version_decrease_fails_closed() -> None:
    with pytest.raises(ValueError, match="Project version must not decrease"):
        build_plan(["pyproject.toml"], base_version="2.3.3", head_version="2.3.2")


def test_version_only_release_selects_only_preflight(tmp_path: Path) -> None:
    base_pyproject = tmp_path / "base-pyproject.toml"
    head_pyproject = tmp_path / "head-pyproject.toml"
    base_lock = tmp_path / "base-uv.lock"
    head_lock = tmp_path / "head-uv.lock"
    base_pyproject.write_text('[project]\nname = "albumentationsx"\nversion = "2.3.2"\n', encoding="utf-8")
    head_pyproject.write_text('[project]\nname = "albumentationsx"\nversion = "2.3.3"\n', encoding="utf-8")
    base_lock.write_text(
        'version = 1\n[[package]]\nname = "albumentationsx"\nversion = "2.3.2"\nsource = { editable = "." }\n',
        encoding="utf-8",
    )
    head_lock.write_text(
        'version = 1\n[[package]]\nname = "albumentationsx"\nversion = "2.3.3"\nsource = { editable = "." }\n',
        encoding="utf-8",
    )

    plan = build_plan(
        ["pyproject.toml", "uv.lock"],
        base_version="2.3.2",
        head_version="2.3.3",
        base_pyproject=base_pyproject,
        head_pyproject=head_pyproject,
        base_lock=base_lock,
        head_lock=head_lock,
    )

    assert {name for name, selected in plan.checks.items() if selected} == {"release_preflight"}


@pytest.mark.parametrize("version", ["", "release-2.3.3", "2.3.3+"])
def test_invalid_project_version_fails_closed(version: str) -> None:
    with pytest.raises(ValueError, match="Invalid PEP 440 version"):
        build_plan(["pyproject.toml"], base_version="2.3.2", head_version=version)
