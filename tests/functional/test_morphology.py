"""Half morphology storage and precision contracts."""

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from albumentations.augmentations.geometric import functional as fgeometric


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float16).newbyteorder("S")])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("storage", ["contiguous", "fortran", "strided", "negative", "readonly"])
@pytest.mark.parametrize("operation,reference", [(fgeometric.erode, cv2.erode), (fgeometric.dilate, cv2.dilate)])
def test_half_morphology_matches_float32_and_preserves_source(
    dtype: np.dtype[np.float16],
    channels: int,
    storage: str,
    operation: Callable[..., np.ndarray],
    reference: Callable[..., np.ndarray],
) -> None:
    rng = np.random.default_rng(137)
    source = rng.uniform(-2, 2, size=(13, 17, channels)).astype(dtype)
    if storage == "fortran":
        source = np.asfortranarray(source)
    elif storage == "strided":
        source = np.repeat(source, 2, axis=1)[:, ::2]
    elif storage == "negative":
        source = source[:, ::-1]
    elif storage == "readonly":
        source.setflags(write=False)
    original = source.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
    expected = reference(source.astype(np.float32), kernel).reshape(source.shape).astype(dtype)

    result = operation(source, kernel)

    assert result.dtype == source.dtype
    assert result.shape == source.shape
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, original)
