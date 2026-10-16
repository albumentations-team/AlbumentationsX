"""Shared imports for split modules."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal, cast
from warnings import warn

import cv2
import numpy as np
from albucore import (
    clip,
    from_float,
    preserve_channel_dim,
    reduce_sum,
    remap,
    to_float,
    warp_perspective,
)
from albucore import (
    copy_make_border as albucore_copy_make_border,
)
from albucore import (
    hflip as albucore_hflip,
)
from albucore import (
    resize as albucore_resize,
)
from albucore import (
    vflip as albucore_vflip,
)

from albumentations.augmentations.utils import angle_2pi_range, handle_empty_array
from albumentations.core.bbox_utils import (
    BBOX_OBB_MIN_COLUMNS,
    bboxes_from_masks,
    bboxes_to_mask,
    denormalize_bboxes,
    mask_to_bboxes,
    masks_from_bboxes,
    normalize_bboxes,
    obb_to_polygons,
    polygons_to_obb,
)
from albumentations.core.type_definitions import (
    NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS,
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    REFLECT_BORDER_MODES,
    ImageType,
)

__all__ = [
    "BBOX_OBB_MIN_COLUMNS",
    "NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS",
    "NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS",
    "NUM_MULTI_CHANNEL_DIMENSIONS",
    "REFLECT_BORDER_MODES",
    "Any",
    "ImageType",
    "Literal",
    "Mapping",
    "Sequence",
    "albucore_copy_make_border",
    "albucore_resize",
    "angle_2pi_range",
    "bboxes_from_masks",
    "bboxes_to_mask",
    "cast",
    "clip_if_interpolation_can_overshoot",
    "cv2",
    "defaultdict",
    "denormalize_bboxes",
    "from_float",
    "handle_empty_array",
    "hflip",
    "lru_cache",
    "mask_to_bboxes",
    "masks_from_bboxes",
    "math",
    "normalize_bboxes",
    "np",
    "obb_to_polygons",
    "os",
    "polygons_to_obb",
    "preserve_channel_dim",
    "reduce_sum",
    "remap",
    "to_float",
    "vflip",
    "warn",
    "warp_perspective",
]


_RANGE_OVERSHOOT_INTERPOLATIONS = frozenset({cv2.INTER_CUBIC, cv2.INTER_LANCZOS4})


def hflip(img: ImageType) -> ImageType:
    """Mirror image or mask columns, preserving an omitted channel axis."""
    if img.ndim == 2:
        return albucore_hflip(img[..., None])[..., 0]
    return albucore_hflip(img)


def vflip(img: ImageType) -> ImageType:
    """Reverse image or mask rows, preserving an omitted channel axis."""
    if img.ndim == 2:
        return albucore_vflip(img[..., None])[..., 0]
    return albucore_vflip(img)


def clip_if_interpolation_can_overshoot(
    image: ImageType,
    interpolation: int,
    *,
    source: ImageType | None = None,
) -> ImageType:
    """Clip new floating samples after cubic or Lanczos interpolation to [0, 1].
    Pass the original source to leave unresampled identity and shared-storage views untouched.
    Use it for images, not masks.
    """
    is_half = image.dtype.type is np.float16
    if interpolation not in _RANGE_OVERSHOOT_INTERPOLATIONS or not (is_half or image.dtype == np.float32):
        return image
    if image is source or (source is not None and np.shares_memory(image, source)):
        return image
    if not image.flags.writeable:
        image = image.copy()
    if is_half:
        return np.clip(image, 0.0, 1.0, out=image, dtype=np.float32, casting="unsafe")
    return clip(image, image.dtype, inplace=True)
