"""Compatibility facade for crop transforms."""

from .base import *
from .basic import *
from .basic import (
    CenterCrop,
    Crop,
    CropAndPad,
    RandomCrop,
)
from .bbox_safe import *
from .bbox_safe import (
    AtLeastOneBBoxRandomCrop,
    BBoxSafeRandomCrop,
    RandomSizedBBoxSafeCrop,
)
from .sized import *
from .sized import (
    RandomResizedCrop,
    RandomSizedCrop,
)
from .special import *
from .special import (
    CropNonEmptyMaskIfExists,
    RandomCropFromBorders,
    RandomCropNearBBox,
)

__all__ = [
    "AtLeastOneBBoxRandomCrop",
    "BBoxSafeRandomCrop",
    "CenterCrop",
    "Crop",
    "CropAndPad",
    "CropNonEmptyMaskIfExists",
    "RandomCrop",
    "RandomCropFromBorders",
    "RandomCropNearBBox",
    "RandomResizedCrop",
    "RandomSizedBBoxSafeCrop",
    "RandomSizedCrop",
]

_obj: object | None = None
for _name in __all__:
    _obj = globals().get(_name)
    if isinstance(_obj, type):
        _obj.__module__ = __name__

del _name, _obj
