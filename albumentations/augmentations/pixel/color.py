"""Compatibility facade for color, tone, and brightness transforms."""

from .color_advanced import *
from .color_advanced import (
    ChromaticAberration,
    ColorJitter,
    HEStain,
    PhotoMetricDistort,
    PlanckianJitter,
    RGBShift,
)
from .color_basic import *
from .color_basic import (
    CLAHE,
    AutoContrast,
    Equalize,
    HueSaturationValue,
    Posterize,
    RandomBrightnessContrast,
    RandomGamma,
    RandomToneCurve,
    Solarize,
)
from .color_gray import *
from .color_gray import (
    Colorize,
    FancyPCA,
    ToGray,
    ToRGB,
    ToSepia,
)
from .color_lighting import *
from .color_lighting import (
    Illumination,
    PlasmaBrightnessContrast,
    PlasmaShadow,
    Vignetting,
)

__all__ = [
    "CLAHE",
    "AutoContrast",
    "ChromaticAberration",
    "ColorJitter",
    "Colorize",
    "Equalize",
    "FancyPCA",
    "HEStain",
    "HueSaturationValue",
    "Illumination",
    "PhotoMetricDistort",
    "PlanckianJitter",
    "PlasmaBrightnessContrast",
    "PlasmaShadow",
    "Posterize",
    "RGBShift",
    "RandomBrightnessContrast",
    "RandomGamma",
    "RandomToneCurve",
    "Solarize",
    "ToGray",
    "ToRGB",
    "ToSepia",
    "Vignetting",
]

_obj: object | None = None
for _name in __all__:
    _obj = globals().get(_name)
    if isinstance(_obj, type):
        _obj.__module__ = __name__

del _name, _obj
