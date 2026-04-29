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
    copy_make_border as albucore_copy_make_border,
)
from albucore import (
    from_float,
    hflip,
    preserve_channel_dim,
    reduce_sum,
    remap,
    to_float,
    vflip,
    warp_perspective,
)
from albucore import (
    resize as albucore_resize,
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
