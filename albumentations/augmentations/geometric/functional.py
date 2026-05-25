"""Compatibility facade for geometric functional helpers."""

from typing import TYPE_CHECKING

from ._functional_bboxes import *
from ._functional_bboxes import __all__ as __functional_bboxes_all
from ._functional_distortion import *
from ._functional_distortion import __all__ as __functional_distortion_all
from ._functional_grid import *
from ._functional_grid import __all__ as __functional_grid_all
from ._functional_images import *
from ._functional_images import __all__ as __functional_images_all
from ._functional_keypoints import *
from ._functional_keypoints import __all__ as __functional_keypoints_all
from ._functional_shared import *
from ._functional_shared import __all__ as __functional_shared_all

if TYPE_CHECKING:
    __all__: list[str]
else:
    __all__ = list(
        __functional_shared_all
        + __functional_images_all
        + __functional_bboxes_all
        + __functional_keypoints_all
        + __functional_distortion_all
        + __functional_grid_all,
    )
