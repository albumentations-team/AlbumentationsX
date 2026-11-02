"""Storage ownership at the NumPy-to-Tensor fallback boundary."""

import warnings

import numpy as np
import pytest
import torch

from albumentations.core.tensor import numpy_to_tensor_annotation, numpy_to_tensor_spatial

SPATIAL_CASES = [
    ("image", (5, 7), (2, 0, 1)),
    ("images", (2, 5, 7), (0, 3, 1, 2)),
    ("volume", (2, 5, 7), (3, 0, 1, 2)),
    ("volumes", (2, 2, 5, 7), (0, 4, 1, 2, 3)),
    ("mask", (5, 7), (2, 0, 1)),
    ("masks", (2, 5, 7), (0, 3, 1, 2)),
    ("mask3d", (2, 5, 7), (3, 0, 1, 2)),
    ("masks3d", (2, 2, 5, 7), (0, 4, 1, 2, 3)),
]


def _storage_view(value: np.ndarray, storage: str) -> np.ndarray:
    if storage == "fortran":
        return np.asfortranarray(value)
    if storage == "bytes":
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if storage in {"positive", "readonly_view"}:
        value = value[..., ::2, :]
    elif storage in {"zero", "readonly_zero"}:
        value = np.broadcast_to(value[..., :1], value.shape)
        if storage == "zero":
            value.setflags(write=True)
    elif storage in {"negative", "readonly_negative"}:
        value = value[..., ::-1]
    if storage.startswith("readonly"):
        value.setflags(write=False)
    return value


@pytest.mark.parametrize(
    ("target", "shape", "axes", "dtype"),
    [
        (*case, dtype)
        for case in SPATIAL_CASES
        for dtype in (np.uint8, np.int16 if case[0].startswith("mask") else np.float16, np.float32)
    ],
)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize(
    ("storage", "shares_storage"),
    [
        ("c", True),
        ("fortran", True),
        ("positive", True),
        ("zero", True),
        ("bytes", False),
        ("readonly_view", False),
        ("readonly_zero", False),
        ("negative", False),
        ("readonly_negative", False),
    ],
)
def test_numpy_to_tensor_spatial_storage(
    target: str,
    shape: tuple[int, ...],
    axes: tuple[int, ...],
    dtype: type[np.generic],
    channels: int,
    storage: str,
    shares_storage: bool,
) -> None:
    shape = (*shape, channels)
    value = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if dtype == np.int16:
        value -= 1000
    value = _storage_view(value, storage)
    source = value.copy()
    expected = value.transpose(axes).copy()
    if storage in {"negative", "readonly_negative"} and channels == 1:
        assert value.flags.c_contiguous
        assert value.strides[-1] < 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = numpy_to_tensor_spatial(value, target)

    assert not caught
    assert np.shares_memory(value, result.numpy()) == shares_storage
    assert result.numpy().dtype == value.dtype
    np.testing.assert_array_equal(result.numpy(), expected)
    result.fill_(137)
    if shares_storage:
        np.testing.assert_array_equal(value, np.full_like(value, 137))
    else:
        np.testing.assert_array_equal(value, source)


@pytest.mark.parametrize("target", ["bboxes", "keypoints"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("storage", ["c", "bytes", "negative"])
def test_numpy_to_tensor_annotation_storage(target: str, dtype: type[np.generic], storage: str) -> None:
    value = _storage_view(np.arange(12, dtype=dtype).reshape(3, 4), storage)
    source = value.copy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = numpy_to_tensor_annotation(value, target)

    assert not caught
    shares_storage = storage == "c" and dtype == np.float32
    assert np.shares_memory(value, result.numpy()) == shares_storage
    assert result.dtype == torch.float32
    np.testing.assert_array_equal(result.numpy(), source.astype(np.float32))
    result.fill_(137)
    if shares_storage:
        np.testing.assert_array_equal(value, np.full_like(value, 137))
    else:
        np.testing.assert_array_equal(value, source)


@pytest.mark.parametrize(("target", "shape", "axes"), SPATIAL_CASES)
def test_numpy_to_tensor_spatial_empty_readonly(target: str, shape: tuple[int, ...], axes: tuple[int, ...]) -> None:
    value = np.frombuffer(b"", dtype=np.float32).reshape(0, *shape[1:], 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = numpy_to_tensor_spatial(value, target)

    assert not caught
    assert result.dtype == torch.float32
    assert result.shape == value.transpose(axes).shape
