from __future__ import annotations

from tools.select_benchmark_filters import select_benchmark_patterns, select_benchmark_regex


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


def test_select_benchmark_filters_add_median_blur_routes_and_memory_for_blur_changes() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/blur/transforms.py"])

    assert "PeakMemory" in patterns
    assert "TimeMedianBlurDirectBatch" in patterns
    assert "TimeMedianBlurTargetRoutes" in patterns


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
        "PeakMemory|TimeBatch|TimeMedianBlurDirectBatch|TimeMedianBlurTargetRoutes|TimeCatalogTransformSmoke|"
        "TimeCorePipeline|TimeParameterSensitivity|TimePixelFullMatrix|TimeFunctionalPixelKernels|"
        "TimeFunctionalBlurKernels"
    )


def test_select_benchmark_filters_do_not_use_broad_functional_class_by_default() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/geometric/functional.py"])

    assert "TimeFunctional" not in patterns
    assert "TimeFunctionalGeometry" in patterns
