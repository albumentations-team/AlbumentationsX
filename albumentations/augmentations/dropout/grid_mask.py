"""Implementation of GridMask augmentation.

GridMask drops grid-line regions (thin stripes in both directions), unlike GridDropout
which drops rectangular cells within a grid. Based on the GridMask paper.

Reference: https://arxiv.org/abs/2001.04086
"""

from typing import Annotated, Any, Literal

from pydantic import AfterValidator

from albumentations.augmentations.dropout import functional as fdropout
from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import check_range_bounds, nondecreasing

__all__ = ["GridMask"]


class GridMask(BaseDropout):
    """Apply GridMask augmentation by dropping grid-line regions.

    Unlike GridDropout which drops rectangular cells, GridMask drops the grid lines
    themselves — continuous horizontal and vertical stripes forming a grid pattern.
    The grid can optionally be rotated.

    Args:
        num_grid_range (tuple[int, int]): Range for number of grid divisions along
            the shorter image side. Default: (3, 7).
        line_width_range (tuple[float, float]): Range for line width as a fraction
            of grid cell size. Default: (0.2, 0.5).
        rotation_range (tuple[float, float]): Range for grid rotation in radians.
            Default: (0, 0) (no rotation).
        fill (float | tuple | str): Fill value for dropped pixels. Default: 0.
        fill_mask (float | tuple | None): Fill value for mask. Default: None.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        Any

    Supported bboxes:
        hbb

    Note:
        GridMask was shown to outperform AutoAugment while being less computationally
        expensive. It achieves +1.4% on ImageNet (ResNet50), +1.8% on COCO detection
        (FasterRCNN-50-FPN), and +0.8% on Cityscapes segmentation (PSPNet50).

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.GridMask(num_grid_range=(3, 5), line_width_range=(0.2, 0.4), p=1.0)
        >>> result = transform(image=image)["image"]

    References:
        GridMask paper: https://arxiv.org/abs/2001.04086

    """

    class InitSchema(BaseDropout.InitSchema):
        num_grid_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(2, None)),
            AfterValidator(nondecreasing),
        ]
        line_width_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        rotation_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        num_grid_range: tuple[int, int] = (3, 7),
        line_width_range: tuple[float, float] = (0.2, 0.5),
        rotation_range: tuple[float, float] = (0, 0),
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.num_grid_range = num_grid_range
        self.line_width_range = line_width_range
        self.rotation_range = rotation_range

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        num_grid = self.py_random.randint(*self.num_grid_range)
        line_width_ratio = self.py_random.uniform(*self.line_width_range)
        rotation = self.py_random.uniform(*self.rotation_range)

        holes = fdropout.generate_grid_mask_holes(
            image_shape,
            num_grid,
            line_width_ratio,
            rotation,
            self.random_generator,
        )

        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}
