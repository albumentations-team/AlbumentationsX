"""Array policies of the public dtype conversion transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from albucore import apply_uint8_lut, clip
from albucore import from_float as core_from_float
from numpy.typing import NDArray

if TYPE_CHECKING:
    from albucore import ImageType


def from_float(img: ImageType, max_value: float) -> NDArray[np.uint8]:
    """Quantize float images and retain the byte-input scalar-promotion contract."""
    if img.dtype == np.uint8:
        # Widening before this expression changes byte-input overflow and rounding.
        domain = np.arange(256, dtype=np.uint8) if img.shape[-1] <= 4 else np.arange(256, dtype=np.float32)
        table = clip(np.rint(domain * max_value), np.uint8)
        return apply_uint8_lut(cast("NDArray[np.uint8]", img), table, out=np.empty(img.shape, dtype=np.uint8))
    return core_from_float(cast("NDArray[np.float16] | NDArray[np.float32]", img), np.uint8, max_value)
