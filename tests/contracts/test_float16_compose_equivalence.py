"""Compare registered modes and Lambda through NumPy and Tensor Compose at float16 and float32."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from tests.helpers.contract_assertions import assert_contract_values_equal
from tests.helpers.target_contracts import (
    TENSOR_TARGET_CONTRACT_PAIRS,
    _assert_image_dtype,
    _convert_contract_data,
    _numpy_target_to_tensor,
    _supports_profile,
)
from tests.helpers.target_profiles import IMAGE_ONLY_TENSOR_TARGET_PROFILES
from tests.helpers.transform_cases import (
    TRANSFORM_CONTRACT_CASES,
    TransformContractCase,
)

_IMAGE_TARGETS = ("image", "images", "volume", "volumes")
_REFERENCE_IMAGE_TRANSFORMS = {A.FDA, A.HistogramMatching, A.PixelDistributionAdaptation}
_DONOR_IMAGE_TRANSFORMS = {A.Mosaic, A.CopyAndPaste, A.OverlayElements}
_UINT8_ROUNDTRIP_ATOL = {A.FromFloat: 1}
_FLOAT16_REFERENCE_MAX_ULP = {A.ColorJitter: 4, A.PhotoMetricDistort: 2}
_FLOAT16_REFERENCE_ATOL = {A.RandomBrightnessContrast: 1 / 2048}


def _identity_callback(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    return image


_LAMBDA_CASE = TransformContractCase(
    case_id="lambda-identity",
    transform_cls=A.Lambda,
    init_kwargs={"image": _identity_callback},
)
_FLOAT16_NUMPY_CASES = (*TRANSFORM_CONTRACT_CASES, _LAMBDA_CASE)
_TENSOR_CASE_IDS = {pair.case.case_id for pair in TENSOR_TARGET_CONTRACT_PAIRS} | {
    case.case_id
    for case in TRANSFORM_CONTRACT_CASES
    if any(_supports_profile(case, profile, A.BasicTransform) for profile in IMAGE_ONLY_TENSOR_TARGET_PROFILES)
}
_FLOAT16_TENSOR_CASES = tuple(case for case in TRANSFORM_CONTRACT_CASES if case.case_id in _TENSOR_CASE_IDS) + (
    _LAMBDA_CASE,
)
_FLOAT16_COMPOSE_CASES = [
    *(pytest.param(case, False, id=f"{case.case_id}-numpy") for case in _FLOAT16_NUMPY_CASES),
    *(pytest.param(case, True, id=f"{case.case_id}-tensor") for case in _FLOAT16_TENSOR_CASES),
]


def _float32_half_image_pair(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logical = source.astype(np.float32) / np.float32(255) if source.dtype == np.uint8 else source.astype(np.float32)
    half = logical.astype(np.float16)
    return half.astype(np.float32), half


def _map_reference_images(value: Any, convert: Callable[[np.ndarray], np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        return convert(value)
    if isinstance(value, list):
        return [_map_reference_images(item, convert) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_reference_images(item, convert) for item in value)
    if isinstance(value, dict):
        return {key: _map_reference_images(item, convert) for key, item in value.items()}
    return value


def _map_donor_images(value: Any, convert: Callable[[np.ndarray], np.ndarray]) -> Any:
    if isinstance(value, list):
        return [_map_donor_images(item, convert) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_donor_images(item, convert) for item in value)
    if isinstance(value, dict):
        return {
            key: convert(item) if key == "image" and isinstance(item, np.ndarray) else _map_donor_images(item, convert)
            for key, item in value.items()
        }
    return value


def _float32_half_pair(case: TransformContractCase, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    float32_data = copy.deepcopy(data)
    half_data = copy.deepcopy(data)
    for target in _IMAGE_TARGETS:
        source = data.get(target)
        if not isinstance(source, np.ndarray):
            continue
        float32_image, half_image = _float32_half_image_pair(source)
        float32_data[target] = float32_image
        half_data[target] = half_image

    for metadata_key in case.metadata_keys:
        source = data[metadata_key]
        if case.transform_cls in _REFERENCE_IMAGE_TRANSFORMS:
            float32_data[metadata_key] = _map_reference_images(source, lambda image: _float32_half_image_pair(image)[0])
            half_data[metadata_key] = _map_reference_images(source, lambda image: _float32_half_image_pair(image)[1])
        elif case.transform_cls in _DONOR_IMAGE_TRANSFORMS:
            float32_data[metadata_key] = _map_donor_images(source, lambda image: _float32_half_image_pair(image)[0])
            half_data[metadata_key] = _map_donor_images(source, lambda image: _float32_half_image_pair(image)[1])
    return float32_data, half_data


def _numpy_image(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _assert_float16_matches_float32(
    case: TransformContractCase,
    target: str,
    half_output: Any,
    float32_output: Any,
) -> None:
    half = _numpy_image(half_output)
    float32 = _numpy_image(float32_output)
    assert half.shape == float32.shape, f"{case.case_id}: {target} shape differs by input dtype"
    assert half.dtype == float32.dtype or (half.dtype == np.float16 and float32.dtype == np.float32), (
        f"{case.case_id}: {target} dtypes differ unexpectedly: {half.dtype} vs {float32.dtype}"
    )

    if np.issubdtype(float32.dtype, np.integer):
        assert half.dtype == float32.dtype, f"{case.case_id}: {target} integer output dtype differs"
        delta = np.abs(half.astype(np.int64) - float32.astype(np.int64))
        allowance = _UINT8_ROUNDTRIP_ATOL.get(case.transform_cls, 0)
        assert delta.max(initial=0) <= allowance, f"{case.case_id}: {target} differs by more than {allowance}"
        return

    expected = float32.astype(np.float16)
    actual = half.astype(np.float16, copy=False)
    for classify in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(classify(actual), classify(expected), err_msg=f"{case.case_id}: {target} class")
    finite = np.isfinite(actual) & np.isfinite(expected)
    if finite.any():
        if case.transform_cls in _FLOAT16_REFERENCE_ATOL:
            np.testing.assert_allclose(
                actual[finite],
                expected[finite],
                rtol=0,
                atol=_FLOAT16_REFERENCE_ATOL[case.transform_cls],
            )
        else:
            np.testing.assert_array_max_ulp(
                actual[finite],
                expected[finite],
                maxulp=_FLOAT16_REFERENCE_MAX_ULP.get(case.transform_cls, 1),
            )


def test_every_registered_mode_has_numpy_and_tensor_compose_coverage() -> None:
    numpy_case_ids = {case.case_id for case in _FLOAT16_NUMPY_CASES}
    tensor_case_ids = {case.case_id for case in _FLOAT16_TENSOR_CASES}

    assert tensor_case_ids == numpy_case_ids


@pytest.mark.parametrize(("case", "tensor_inputs"), _FLOAT16_COMPOSE_CASES)
def test_every_transform_mode_matches_float32_through_compose(
    case: TransformContractCase,
    tensor_inputs: bool,
) -> None:
    seed = case.seeds[0]
    source = case.make_data(np.random.default_rng(seed))
    float32_data, half_data = _float32_half_pair(case, source)
    if tensor_inputs:
        transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1.0)
        float32_data = _convert_contract_data(float32_data, transform, _numpy_target_to_tensor)
        half_data = _convert_contract_data(half_data, transform, _numpy_target_to_tensor)
    float32_snapshot = copy.deepcopy(float32_data)
    half_snapshot = copy.deepcopy(half_data)
    compose_kwargs = copy.deepcopy(dict(case.primary_compose_kwargs))

    float32_pipeline = A.Compose(
        [case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1.0)],
        save_applied_params=True,
        seed=seed,
        strict=True,
        telemetry=False,
        **compose_kwargs,
    )
    float32_result = float32_pipeline(**float32_data)
    half_pipeline = A.Compose(
        [case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1.0)],
        save_applied_params=True,
        seed=seed,
        strict=True,
        telemetry=False,
        **copy.deepcopy(compose_kwargs),
    )
    half_result = half_pipeline(**half_data)

    assert_contract_values_equal(float32_data, float32_snapshot, f"{case.case_id}.float32_input")
    assert_contract_values_equal(half_data, half_snapshot, f"{case.case_id}.float16_input")
    assert_contract_values_equal(
        half_result["applied_transforms"],
        float32_result["applied_transforms"],
        f"{case.case_id}.sampled_parameters",
    )
    _assert_image_dtype(case, float32_snapshot, float32_result, np.dtype(np.float32))
    _assert_image_dtype(case, half_snapshot, half_result, np.dtype(np.float16))

    image_targets = set(_IMAGE_TARGETS) & set(source)
    assert image_targets, f"{case.case_id}: registered case has no image input"
    for target in image_targets:
        _assert_float16_matches_float32(case, target, half_result[target], float32_result[target])

    ignored = image_targets | set(case.metadata_keys) | {"applied_transforms", "replay"}
    for key in float32_result.keys() - ignored:
        assert_contract_values_equal(half_result[key], float32_result[key], f"{case.case_id}.{key}")
