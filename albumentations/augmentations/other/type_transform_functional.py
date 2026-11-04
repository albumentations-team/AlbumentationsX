"""Array policies of the public dtype conversion transforms."""

from __future__ import annotations

from typing import cast

import numpy as np
from albucore import apply_uint8_lut, clip, convert_dtype
from numpy.typing import NDArray

_FLOAT_DTYPES = (np.float16, np.float32)


def to_float(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], max_value: float | None
) -> NDArray[np.float16] | NDArray[np.float32]:
    """Normalize byte inputs to float32 while preserving floating inputs and omitted channels."""
    source_dtype = img.dtype.type
    if source_dtype in _FLOAT_DTYPES:
        return cast("NDArray[np.float16] | NDArray[np.float32]", img)
    if source_dtype is not np.uint8:
        raise TypeError(f"ToFloat does not support dtype {img.dtype}; expected uint8, float16 or float32.")
    is_channel_less = img.ndim == 2
    working = img[..., None] if is_channel_less else img
    result = convert_dtype(working, np.float32, max_value=max_value)
    return result[..., 0] if is_channel_less else result


def from_float(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], max_value: float
) -> NDArray[np.uint8]:
    """Quantize float images and retain the byte-input scalar-promotion contract."""
    is_channel_less = img.ndim == 2
    if is_channel_less:
        img = img[..., None]
    if img.dtype == np.uint8:
        # Widening before this expression changes byte-input overflow and rounding.
        domain = np.arange(256, dtype=np.uint8) if img.shape[-1] <= 4 else np.arange(256, dtype=np.float32)
        table = clip(np.rint(domain * max_value), np.uint8)
        result = apply_uint8_lut(cast("NDArray[np.uint8]", img), table, out=np.empty(img.shape, dtype=np.uint8))
    else:
        result = convert_dtype(img, np.uint8, max_value=max_value)
    return result[..., 0] if is_channel_less else result
