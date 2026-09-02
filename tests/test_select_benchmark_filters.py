from __future__ import annotations

import re

import pytest

from tools.select_benchmark_filters import (
    main,
    select_benchmark_patterns,
    select_benchmark_regex,
    select_profile_patterns,
    select_profile_regex,
)


def test_select_benchmark_filters_changed_profile_routes_changed_paths() -> None:
    assert select_profile_patterns("changed", ["albumentations/core/composition.py"]) == (
        "TimeBatch",
        "TimeCatalogTransformSmoke",
        "TimeComposeFullMatrix",
        "TimeCorePipeline",
    )


def test_select_benchmark_filters_do_not_invent_evidence_for_docs_only_changes() -> None:
    assert select_benchmark_patterns(["docs/maintaining/performance-coverage.md"]) == ()


def test_select_benchmark_filters_add_pixel_family_matrix_for_pixel_changes() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/pixel/functional.py"])

    assert "TimePixelFullMatrix" in patterns
    assert "TimeFunctionalPixelKernels" in patterns
    assert "TimeBatch" in patterns
    assert "TimeParameterSensitivity" in patterns


@pytest.mark.parametrize(
    "benchmark_name",
    [
        "benchmarks.test_batch_matrix.TimeBatchPlasmaBrightnessContrastDirectMatrix.time_apply_to_images",
        "benchmarks.test_batch_matrix.TimeBatchPlasmaBrightnessContrastImageMatrix.time_transform",
        "benchmarks.test_batch_matrix.TimeBatchPlasmaBrightnessContrastVolumeMatrix.time_transform",
    ],
)
def test_select_benchmark_filters_changed_pixel_profile_selects_plasma_batch_matrix(benchmark_name: str) -> None:
    regex = select_profile_regex("changed", ["albumentations/augmentations/pixel/_functional_illumination.py"])

    assert re.search(regex, benchmark_name) is not None


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


def test_select_benchmark_filters_requires_an_explicit_filter_for_benchmark_infrastructure_changes() -> None:
    assert select_benchmark_patterns(["benchmark/benchmarks/test_family_matrix.py"]) == ()


def test_select_benchmark_filters_returns_asv_regex() -> None:
    regex = select_benchmark_regex(["albumentations/augmentations/blur/transforms.py"])

    assert regex == (
        "PeakMemory|TimeBatch|TimeParameterSensitivity|"
        "TimePixelFullMatrix|TimeFunctionalPixelKernels|TimeFunctionalBlurKernels"
    )


def test_select_benchmark_filters_do_not_use_broad_functional_class_by_default() -> None:
    patterns = select_benchmark_patterns(["albumentations/augmentations/geometric/functional.py"])

    assert "TimeFunctional" not in patterns
    assert "TimeFunctionalGeometry" in patterns


def test_select_benchmark_filters_reject_invalid_profile_inputs() -> None:
    with pytest.raises(ValueError, match="does not accept changed paths"):
        select_profile_patterns("release-core", ["albumentations/core/composition.py"])
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
        ["select_benchmark_filters.py", "--profile", "release-core", "--changed-files", "paths.txt"],
    )
    with pytest.raises(SystemExit, match="does not accept --changed-files"):
        main()
