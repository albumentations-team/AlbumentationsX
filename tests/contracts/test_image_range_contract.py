"""Range invariants for public NumPy image and volume transforms."""

from __future__ import annotations

import copy
from typing import Any, get_args

import cv2
import numpy as np
import pytest

import albumentations as A
from tests.helpers.transform_cases import TRANSFORM_CONTRACT_CASES, TransformContractCase

_IMAGE_TARGETS = ("image", "images", "volume")
_FLOAT32_FILL_PARAMETERS = frozenset({"fill", "drop_value", "mask_drop_value"})


def _range_data(case: TransformContractCase, dtype: np.dtype[Any]) -> dict[str, Any]:
    data = case.make_data(np.random.default_rng(137))
    for target in _IMAGE_TARGETS:
        if target in data:
            data[target] = _range_extrema(data[target], dtype)
    for metadata_key in case.metadata_keys:
        if case.transform_cls is A.GuidedCoarseDropout:
            data[metadata_key] = np.ones(data["image"].shape[:2], dtype=np.uint8)
        else:
            data[metadata_key] = _range_reference_images(data[metadata_key], dtype)
    return data


def _range_extrema(image: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.empty_like(image, dtype=dtype)
    result.fill(0)
    width_axis = -2 if result.ndim >= 3 else -1
    edge = [slice(None)] * result.ndim
    edge[width_axis] = slice(result.shape[width_axis] // 2, None)
    result[tuple(edge)] = np.iinfo(dtype).max if dtype == np.uint8 else 1.0
    return result


def _range_reference_images(value: Any, dtype: np.dtype[Any]) -> Any:
    if isinstance(value, np.ndarray):
        return _range_extrema(value, dtype)
    if isinstance(value, list):
        return [_range_reference_images(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_range_reference_images(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _range_reference_images(item, dtype) if key == "image" else item for key, item in value.items()}
    return value


def _range_init_kwargs(case: TransformContractCase, dtype: np.dtype[Any]) -> dict[str, Any]:
    init_kwargs = copy.deepcopy(dict(case.init_kwargs))
    if dtype == np.float32:
        for parameter_name in _FLOAT32_FILL_PARAMETERS & init_kwargs.keys():
            init_kwargs[parameter_name] = 0
    interpolation_field = case.transform_cls.InitSchema.model_fields.get("interpolation")
    interpolation_values = get_args(interpolation_field.annotation) if interpolation_field is not None else ()
    if interpolation_field is not None and (not interpolation_values or cv2.INTER_CUBIC in interpolation_values):
        init_kwargs["interpolation"] = cv2.INTER_CUBIC
        if "area_for_downscale" in init_kwargs:
            init_kwargs["area_for_downscale"] = None
    return init_kwargs


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("case", TRANSFORM_CONTRACT_CASES, ids=lambda case: case.case_id)
def test_range_preserving_transforms_keep_normalized_image_targets_in_range(
    case: TransformContractCase,
    dtype: np.dtype[Any],
) -> None:
    if not case.transform_cls._preserves_input_image_range:
        pytest.skip("The transform explicitly changes the image value range.")

    source = _range_data(case, dtype)
    pipeline = A.Compose(
        [case.transform_cls(**_range_init_kwargs(case, dtype), p=1.0)],
        strict=True,
        telemetry=False,
        seed=case.seeds[0],
        **copy.deepcopy(dict(case.primary_compose_kwargs)),
    )

    result = pipeline(**source)

    for target in _IMAGE_TARGETS:
        if target not in result:
            continue
        image = result[target]
        max_value = np.iinfo(image.dtype).max if image.dtype == np.uint8 else 1.0
        range_tolerance = np.finfo(np.float32).eps if image.dtype == np.float32 else 0
        assert image.dtype in (np.uint8, np.float32), f"{case.case_id}: {target} returned {image.dtype}"
        assert np.isfinite(image).all(), f"{case.case_id}: {target} contains non-finite values"
        assert image.min() >= -range_tolerance, f"{case.case_id}: {target} has values below zero"
        assert image.max() <= max_value + range_tolerance, f"{case.case_id}: {target} exceeds {max_value}"
