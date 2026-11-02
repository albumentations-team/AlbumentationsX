"""Tensor outputs safely expose NumPy fallback storage."""

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.tensor import numpy_to_tensor_spatial
from albumentations.core.transforms_interface import ImageOnlyTransform


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize(
    ("target", "shape", "permutation"),
    [
        ("image", (5, 7, 3), (2, 0, 1)),
        ("images", (2, 5, 7, 3), (0, 3, 1, 2)),
        ("volume", (2, 5, 7, 3), (3, 0, 1, 2)),
        ("volumes", (2, 3, 5, 7, 3), (0, 4, 1, 2, 3)),
    ],
)
@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("storage", ["contiguous", "fortran", "negative"])
def test_numpy_tensor_storage_ownership(dtype, target, shape, permutation, readonly, storage):
    original = np.arange(np.prod(shape), dtype=np.float32).astype(dtype).reshape(shape)
    source = np.frombuffer(original.tobytes(), dtype=dtype).reshape(shape) if readonly else original.copy()

    if storage == "fortran":
        source = np.asfortranarray(source)
    elif storage == "negative":
        source = source[::-1]
    if readonly:
        source.flags.writeable = False
    before = source.copy()

    result = numpy_to_tensor_spatial(source, target)

    np.testing.assert_array_equal(result.numpy(), before.transpose(permutation))
    copy_required = readonly or storage == "negative"
    assert np.shares_memory(source, result.numpy()) is not copy_required
    if copy_required:
        result.zero_()
        np.testing.assert_array_equal(source, before)


class _ReadOnlyNumpyProbe(ImageOnlyTransform):
    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.result: np.ndarray | None = None

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        result = image + 1
        self.result = np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)
        return self.result


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_compose_copies_readonly_fallback_result(dtype):
    image = torch.arange(3 * 5 * 7, dtype=dtype).reshape(3, 5, 7)
    original = image.clone()
    transform = _ReadOnlyNumpyProbe()

    result = A.Compose([transform], strict=True)(image=image)["image"]

    torch.testing.assert_close(result, original + 1, rtol=0, atol=0)
    assert transform.result is not None
    assert not np.shares_memory(result.numpy(), transform.result)
    expected = transform.result.copy()
    result.zero_()
    np.testing.assert_array_equal(transform.result, expected)
    torch.testing.assert_close(image, original, rtol=0, atol=0)
