"""Implementation of GridMask augmentation.

GridMask drops grid-line regions (thin stripes in both directions), unlike GridDropout
which drops rectangular cells within a grid. Based on the GridMask paper.

Reference: https://arxiv.org/abs/2001.04086
"""

from typing import Annotated, Any, Literal

import cv2
import numpy as np
from albucore import warp_affine
from pydantic import AfterValidator

from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import check_range_bounds, nondecreasing

__all__ = ["GridMask"]


def _mask_to_rects(mask: np.ndarray) -> np.ndarray:
    """Decompose a binary mask's zero-regions into axis-aligned rectangles.

    Scans row-by-row and merges contiguous zero-runs vertically into rectangles.
    This preserves the rotated grid-line structure when passed to BaseDropout.cutout,
    rather than collapsing everything into a single bounding box.

    Args:
        mask: 2D uint8 mask where 0 indicates a dropped region.

    Returns:
        Array of shape (N, 4) with [x1, y1, x2, y2] rectangles, or empty (0, 4) array.

    """
    height = mask.shape[0]
    rects: list[list[int]] = []
    open_rects: dict[tuple[int, int], list[int]] = {}

    for y in range(height):
        row = mask[y] == 0
        if not np.any(row):
            rects.extend(open_rects.values())
            open_rects.clear()
            continue

        row_int = row.astype(np.int8)
        diff = np.diff(np.concatenate(([0], row_int, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        current_keys = {(int(x1), int(x2)) for x1, x2 in zip(starts, ends, strict=True)}

        closed = [k for k in open_rects if k not in current_keys]
        rects.extend(open_rects.pop(k) for k in closed)

        for x1, x2 in zip(starts, ends, strict=True):
            key = (int(x1), int(x2))
            if key in open_rects:
                open_rects[key][3] = y + 1
            else:
                open_rects[key] = [int(x1), y, int(x2), y + 1]

    rects.extend(open_rects.values())
    return np.array(rects, dtype=np.int32) if rects else np.empty((0, 4), dtype=np.int32)


def _generate_grid_mask_holes(
    image_shape: tuple[int, int],
    num_grid: int,
    line_width_ratio: float,
    rotation: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate grid-line shaped holes for GridMask.

    Args:
        image_shape: (height, width) of the image.
        num_grid: Number of grid divisions along the shorter side.
        line_width_ratio: Width of masked lines as fraction of grid cell size.
        rotation: Rotation angle in radians.
        random_generator: NumPy random generator.

    Returns:
        Array of holes as (N, 4) with [x1, y1, x2, y2] format.

    """
    height, width = image_shape
    shorter = min(height, width)
    cell_size = max(2, shorter // num_grid)
    line_width = max(1, int(cell_size * line_width_ratio))

    if abs(rotation) < 1e-6:
        holes = []
        offset_x = int(random_generator.integers(0, cell_size))
        offset_y = int(random_generator.integers(0, cell_size))

        x = offset_x
        while x < width:
            x_end = min(x + line_width, width)
            holes.append([x, 0, x_end, height])
            x += cell_size

        y = offset_y
        while y < height:
            y_end = min(y + line_width, height)
            holes.append([0, y, width, y_end])
            y += cell_size

        return np.array(holes, dtype=np.int32) if holes else np.empty((0, 4), dtype=np.int32)

    mask = np.ones((height, width), dtype=np.uint8)
    diag = int(np.sqrt(height**2 + width**2)) + cell_size * 2
    grid_mask = np.ones((diag, diag), dtype=np.uint8)

    offset = int(random_generator.integers(0, cell_size))
    pos = offset
    while pos < diag:
        grid_mask[pos : pos + line_width, :] = 0
        grid_mask[:, pos : pos + line_width] = 0
        pos += cell_size

    center = (diag // 2, diag // 2)
    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(rotation), 1.0)
    grid_3d = grid_mask[:, :, np.newaxis]
    rotated_3d = warp_affine(grid_3d, rot_mat, (diag, diag), border_value=1)
    rotated = rotated_3d[:, :, 0]

    start_y = (diag - height) // 2
    start_x = (diag - width) // 2
    crop = rotated[start_y : start_y + height, start_x : start_x + width]

    mask *= crop

    return _mask_to_rects(mask)


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

        holes = _generate_grid_mask_holes(
            image_shape,
            num_grid,
            line_width_ratio,
            rotation,
            self.random_generator,
        )

        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}
