"""Shared imports for split modules."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal
from warnings import warn

import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    add,
    add_array,
    add_constant,
    add_vector,
    add_weighted,
    apply_uint8_lut,
    clip,
    clipped,
    float32_io,
    from_float,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    maybe_process_in_chunks,
    mean,
    multiply,
    multiply_add,
    multiply_by_array,
    multiply_by_constant,
    normalize_per_image,
    power,
    preserve_channel_dim,
    reduce_sum,
    remap,
    reshape_xhwc_channel,
    restore_xhwc_channel,
    std,
    sz_lut,
    to_float,
    uint8_io,
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
    "add_array",
    "add_constant",
    "add_vector",
    "add_weighted",
    "apply_uint8_lut",
    "clip",
    "clipped",
    "cv2",
    "fgeometric",
    "float32_io",
    "from_float",
    "get_num_channels",
    "is_grayscale_image",
    "is_rgb_image",
    "lru_cache",
    "math",
    "maybe_process_in_chunks",
    "mean",
    "multiply",
    "multiply_add",
    "multiply_by_array",
    "multiply_by_constant",
    "non_rgb_error",
    "normalize_per_image",
    "np",
    "power",
    "preserve_channel_dim",
    "random",
    "reduce_sum",
    "remap",
    "reshape_xhwc_channel",
    "restore_xhwc_channel",
    "std",
    "sz_lut",
    "to_float",
    "uint8_io",
    "warn",
]
