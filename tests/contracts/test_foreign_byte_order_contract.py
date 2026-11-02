from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from tests.helpers.contract_assertions import assert_contract_values_equal
from tests.helpers.target_contracts import (
    CORE_TARGET_CONTRACT_PAIRS,
    EXTENDED_TARGET_CONTRACT_PAIRS,
    IMAGE_TARGET_NAMES,
    TENSOR_IMAGE_ONLY_TARGET_CONTRACT_PAIRS,
    VOLUME_ONLY_TARGET_CONTRACT_PAIRS,
    TargetContractPair,
    make_target_contract_data,
)


def _selected_pairs(pairs: Iterable[TargetContractPair]) -> tuple[TargetContractPair, ...]:
    cases: dict[str, list[TargetContractPair]] = defaultdict(list)
    for pair in pairs:
        if pair.profile.required_targets & IMAGE_TARGET_NAMES:
            cases[pair.case.case_id].append(pair)

    priority = ("image", "image-mask", "images-batch", "volume-mask3d", "volumes-masks3d", "image-only")
    return tuple(
        min(
            case_pairs,
            key=lambda pair: (
                priority.index(pair.profile.profile_id) if pair.profile.profile_id in priority else len(priority)
            ),
        )
        for case_pairs in cases.values()
    )


def _to_foreign_float(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.type in (np.float16, np.float32):
        foreign_dtype = value.dtype.newbyteorder("S")
        if value.flags.c_contiguous or value.flags.f_contiguous:
            converted = value.astype(foreign_dtype, order="K", copy=True)
        else:
            backing = np.empty((*value.shape, 2), dtype=foreign_dtype)
            converted = backing[..., 0]
            np.copyto(converted, value)
        if not value.flags.writeable:
            converted.flags.writeable = False
        return converted
    if isinstance(value, list):
        return [_to_foreign_float(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_foreign_float(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_foreign_float(item) for key, item in value.items()}
    return value


def _to_float32(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.type is np.float16:
        if value.flags.c_contiguous or value.flags.f_contiguous:
            converted = value.astype(np.float32, order="K", copy=True)
        else:
            backing = np.empty((*value.shape, 2), dtype=np.float32)
            converted = backing[..., 0]
            np.copyto(converted, value)
        if not value.flags.writeable:
            converted.flags.writeable = False
        return converted
    if isinstance(value, list):
        return [_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_float32(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_float32(item) for key, item in value.items()}
    return value


_FOREIGN_BYTE_ORDER_PAIRS = _selected_pairs(
    (
        *CORE_TARGET_CONTRACT_PAIRS,
        *EXTENDED_TARGET_CONTRACT_PAIRS,
        *TENSOR_IMAGE_ONLY_TARGET_CONTRACT_PAIRS,
        *VOLUME_ONLY_TARGET_CONTRACT_PAIRS,
    ),
)


@pytest.mark.parametrize("pair", _FOREIGN_BYTE_ORDER_PAIRS, ids=lambda pair: pair.pair_id)
@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float32)], ids=["float16", "float32"])
def test_compose_foreign_float_byte_order_matches_native(
    pair: TargetContractPair,
    dtype: np.dtype[np.float16] | np.dtype[np.float32],
) -> None:
    native_data = make_target_contract_data(
        pair.case,
        pair.profile,
        np.random.default_rng(137),
        image_dtype=np.dtype(np.float16),
    )
    if dtype.type is np.float32:
        native_data = {
            key: _to_float32(value) if key in IMAGE_TARGET_NAMES else value for key, value in native_data.items()
        }
    foreign_data = copy.deepcopy(native_data)
    for target in IMAGE_TARGET_NAMES & foreign_data.keys():
        foreign_data[target] = _to_foreign_float(foreign_data[target])
    foreign_snapshot = copy.deepcopy(foreign_data)
    compose_kwargs = copy.deepcopy(dict(pair.profile.compose_kwargs))

    native_result = A.Compose(
        [pair.case.transform_cls(**copy.deepcopy(dict(pair.case.init_kwargs)), p=1)],
        strict=True,
        telemetry=False,
        seed=137,
        **compose_kwargs,
    )(**native_data)
    foreign_result = A.Compose(
        [pair.case.transform_cls(**copy.deepcopy(dict(pair.case.init_kwargs)), p=1)],
        strict=True,
        telemetry=False,
        seed=137,
        **compose_kwargs,
    )(**foreign_data)

    for target in foreign_snapshot:
        assert_contract_values_equal(foreign_data[target], foreign_snapshot[target], f"{pair.pair_id}:{target}")
    pair.profile.assert_result(pair.case, foreign_snapshot, foreign_result)
    for target in IMAGE_TARGET_NAMES & native_result.keys():
        expected = native_result[target]
        actual = foreign_result[target]
        if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            assert actual.dtype.type is expected.dtype.type, f"{pair.pair_id}:{target} changed dtype"
            assert actual.shape == expected.shape, f"{pair.pair_id}:{target} changed shape"
            np.testing.assert_array_equal(actual, expected)
            source = foreign_snapshot[target]
            if (
                isinstance(source, np.ndarray)
                and expected.dtype.type in (np.float16, np.float32)
                and expected.dtype.type is source.dtype.type
            ):
                assert actual.dtype == source.dtype, f"{pair.pair_id}:{target} did not retain byte order"
        elif isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert_contract_values_equal(actual, expected, f"{pair.pair_id}:{target}")
