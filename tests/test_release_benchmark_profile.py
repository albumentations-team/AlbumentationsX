"""Executable contract for the fixed release ASV profile."""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tools.select_benchmark_filters import select_profile_patterns

BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "benchmark"
RELEASE_CORE_CASES = (
    (
        "benchmarks.test_core_pipeline",
        "TimeCorePipeline",
        ("small", 3),
        (
            "time_single_transform_compose",
            "time_skip_transform_compose",
            "time_noop_compose",
            "time_noop_probability_compose",
            "time_multi_transform_compose",
        ),
    ),
    (
        "benchmarks.test_core_pipeline",
        "TimeCorePipelineTargetProcessors",
        (10,),
        ("time_bbox_keypoint_processor_roundtrip",),
    ),
    (
        "benchmarks.test_geometric",
        "TimeGeometricTransforms",
        ("small", 3),
        ("time_horizontal_flip", "time_resize", "time_pad_if_needed", "time_affine", "peakmem_affine"),
    ),
    (
        "benchmarks.test_pixel",
        "TimePixelTransforms",
        ("small", 3),
        ("time_random_brightness_contrast", "time_gaussian_blur", "time_normalize", "peakmem_normalize"),
    ),
    ("benchmarks.test_mixing", "TimeMixingTransforms", (), ("time_mosaic", "peakmem_mosaic")),
    (
        "benchmarks.test_volumetric",
        "TimeVolumetricTransforms",
        (),
        (
            "time_center_crop3d",
            "time_pad_if_needed3d",
            "peakmem_center_crop3d",
            "peakmem_pad_if_needed3d",
        ),
    ),
)


def _profile_case_names() -> tuple[str, ...]:
    return tuple(
        f"{class_name}.{method_name}" for _, class_name, _, methods in RELEASE_CORE_CASES for method_name in methods
    )


def test_volumetric_benchmark_discovery_without_medical_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARK_ROOT))
    for cached_name in tuple(sys.modules):
        if cached_name.startswith("albumentations.augmentations.medical"):
            monkeypatch.delitem(sys.modules, cached_name)
    monkeypatch.setitem(sys.modules, "albumentations", types.ModuleType("albumentations"))

    original_import = builtins.__import__

    def import_without_medical(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("albumentations.augmentations.medical"):
            raise ModuleNotFoundError("medical package is unavailable", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_medical)
    module_name = "benchmarks.test_volumetric"
    previous_module = sys.modules.pop(module_name, None)
    try:
        benchmark_module = importlib.import_module(module_name)
        assert Path(benchmark_module.__file__).resolve() == BENCHMARK_ROOT / "benchmarks" / "test_volumetric.py"
        assert hasattr(benchmark_module, "TimeVolumetricTransforms")
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_release_core_profile_has_exactly_the_runnable_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARK_ROOT))

    assert select_profile_patterns("release-core") == _profile_case_names()

    for module_name, class_name, setup_args, methods in RELEASE_CORE_CASES:
        benchmark_class: type[Any] = getattr(importlib.import_module(module_name), class_name)
        benchmark = benchmark_class()
        benchmark.setup(*setup_args)
        for method_name in methods:
            getattr(benchmark, method_name)(*setup_args)
