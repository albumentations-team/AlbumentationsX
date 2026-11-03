"""Shared imports for split modules."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal, cast
from warnings import warn

import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    add,
    add_weighted,
    apply_uint8_lut,
    clip,
    convert_dtype,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    maybe_process_in_chunks,
    mean,
    multiply,
    multiply_add,
    normalize_per_image,
    power,
    reduce_sum,
    remap,
    std,
)

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.utils import (
    PCA,
    non_rgb_error,
)
from albumentations.core.type_definitions import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    ImageFloat32,
    ImageType,
    ImageUInt8,
)

MULTICHANNEL_LUT_MEDIUM_IMAGE_PIXELS = 512 * 512
MULTICHANNEL_LUT_LARGE_IMAGE_PIXELS = 1024 * 1024


def to_uint8_working_image(img: ImageType) -> tuple[ImageUInt8, np.dtype]:
    input_dtype = img.dtype
    if input_dtype.type is np.uint8:
        return cast("ImageUInt8", img), input_dtype
    return convert_dtype(img, np.uint8), input_dtype


def restore_uint8_working_image(img: ImageUInt8, input_dtype: np.dtype) -> ImageType:
    if input_dtype.type is np.uint8:
        return img
    restored = convert_dtype(img, np.float32)
    return convert_dtype(restored, np.float16) if input_dtype.type is np.float16 else restored


__all__ = [
    "MAX_VALUES_BY_DTYPE",
    "MONO_CHANNEL_DIMENSIONS",
    "MULTICHANNEL_LUT_LARGE_IMAGE_PIXELS",
    "MULTICHANNEL_LUT_MEDIUM_IMAGE_PIXELS",
    "NUM_MULTI_CHANNEL_DIMENSIONS",
    "NUM_RGB_CHANNELS",
    "PCA",
    "Any",
    "ImageFloat32",
    "ImageType",
    "ImageUInt8",
    "Literal",
    "Sequence",
    "add",
    "add_weighted",
    "apply_uint8_lut",
    "clip",
    "convert_dtype",
    "cv2",
    "fgeometric",
    "get_num_channels",
    "is_grayscale_image",
    "is_rgb_image",
    "lru_cache",
    "math",
    "maybe_process_in_chunks",
    "mean",
    "multiply",
    "multiply_add",
    "non_rgb_error",
    "normalize_per_image",
    "np",
    "power",
    "random",
    "reduce_sum",
    "remap",
    "restore_uint8_working_image",
    "std",
    "to_uint8_working_image",
    "warn",
]
