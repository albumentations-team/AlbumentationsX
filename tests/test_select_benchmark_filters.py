from __future__ import annotations

import pytest

from tools.select_benchmark_filters import (
    main,
    select_benchmark_patterns,
    select_benchmark_regex,
    select_profile_patterns,
    select_profile_regex,
)


def test_select_benchmark_filters_pr_core_is_bounded() -> None:
    assert select_profile_patterns("pr-core") == ("TimeCorePipeline",)
    assert select_profile_regex("pr-core") == "TimeCorePipeline"


def test_select_benchmark_filters_stf_core_contains_runtime_and_memory_sentinels() -> None:
    patterns = select_profile_patterns("stf-core")

    assert patterns == (
        "TimeCatalogTransformSmoke",
        "TimeCorePipeline",
        "peakmem_resize_large_rgb",
        "peakmem_normalize_large_rgb",
        "peakmem_batch_pipeline_medium_rgb",
        "peakmem_mosaic_small_rgb",
        "peakmem_copy_paste_small_rgb",
        "peakmem_volume_pad_medium",
    )


def test_select_benchmark_filters_changed_profile_routes_changed_paths() -> None:
    assert select_profile_patterns("changed", ["albumentations/core/composition.py"]) == (
        "TimeBatch",
        "TimeCatalogTransformSmoke",
        "TimeCorePipeline",
    )


def test_select_benchmark_filters_keep_baseline_for_docs_only_changes() -> None:
    assert select_benchmark_patterns(["docs/maintaining/performance-coverage.md"]) == (
        "TimeBatch",
        "TimeCatalogTransformSmoke",
        "TimeCorePipeline",
    )


def test_select_benchmark_filters_add_pixel_family_matrix_for_pixel_changes() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/pixel/functional.py"])

    assert "TimePixelFullMatrix" in patterns
    assert "TimeFunctionalPixelKernels" in patterns
    assert "TimeBatch" in patterns
    assert "TimeParameterSensitivity" in patterns
    assert "TimeCatalogTransformSmoke" in patterns


def test_select_benchmark_filters_add_memory_for_blur_changes() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/blur/transforms.py"])

    assert "PeakMemory" in patterns


def test_select_benchmark_filters_add_geometry_and_annotation_paths_for_geometric_changes() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/geometric/functional.py"])

    assert "TimeGeometryFullMatrix" in patterns
    assert "TimeAnnotationTargets" in patterns
    assert "TimeSpecialTargetMatrix" in patterns
    assert "TimeFunctionalGeometry" in patterns
    assert "TimeParameterSensitivity" in patterns


def test_select_benchmark_filters_add_memory_and_volume_paths_for_3d_changes() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/transforms3d/functional.py"])

    assert "PeakMemory" in patterns
    assert "TimeFunctional3DKernels" in patterns
    assert "TimeVolumetricFullMatrix" in patterns


def test_select_benchmark_filters_keeps_benchmark_infrastructure_changes_bounded() -> None:
    assert select_benchmark_patterns(["benchmark/benchmarks/test_family_matrix.py"]) == (
        "TimeBatch",
        "TimeCatalogTransformSmoke",
        "TimeCorePipeline",
    )


def test_select_benchmark_filters_returns_asv_regex() -> None:
    regex = select_benchmark_regex(["albumentations/augmentations/blur/transforms.py"])

    assert regex == (
        "PeakMemory|TimeBatch|TimeCatalogTransformSmoke|TimeCorePipeline|TimeParameterSensitivity|"
        "TimePixelFullMatrix|TimeFunctionalPixelKernels|TimeFunctionalBlurKernels"
    )


def test_select_benchmark_filters_do_not_use_broad_functional_class_by_default() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/geometric/functional.py"])

    assert "TimeFunctional" not in patterns
    assert "TimeFunctionalGeometry" in patterns


def test_select_benchmark_filters_reject_invalid_profile_inputs() -> None:
    with pytest.raises(ValueError, match="does not accept changed paths"):
        select_profile_patterns("pr-core", ["albumentations/core/composition.py"])
    with pytest.raises(ValueError, match="requires changed paths"):
        select_profile_patterns("changed")
    with pytest.raises(ValueError, match="unknown benchmark profile"):
        select_profile_patterns("everything")


def test_select_benchmark_filters_cli_requires_profile_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["select_benchmark_filters.py", "--profile", "changed"])
    with pytest.raises(SystemExit, match="requires --changed-files"):
        main()

    monkeypatch.setattr(
        "sys.argv",
        ["select_benchmark_filters.py", "--profile", "pr-core", "--changed-files", "paths.txt"],
    )
    with pytest.raises(SystemExit, match="does not accept --changed-files"):
        main()
