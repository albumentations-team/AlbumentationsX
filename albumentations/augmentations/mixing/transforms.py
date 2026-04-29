"""Compatibility facade for mixing transforms."""

from .copy_paste import *
from .copy_paste import __all__ as _copy_paste_all
from .mosaic import *
from .mosaic import __all__ as _mosaic_all
from .overlay import *
from .overlay import __all__ as _overlay_all

__all__ = list(
    _overlay_all + _copy_paste_all + _mosaic_all,
)

for _name in __all__:
    _obj = globals().get(_name)
    if isinstance(_obj, type):
        _obj.__module__ = __name__

del _name, _obj
