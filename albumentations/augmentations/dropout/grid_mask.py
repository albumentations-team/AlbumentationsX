"""Implementation of GridMask augmentation.

GridMask drops grid lines or repeated cell regions with one shared random phase.
Based on the GridMask paper and competition cell-pattern variants.

Reference: https://arxiv.org/abs/2001.04086
"""

from typing import Annotated, Any, Literal

from pydantic import AfterValidator

from albumentations.augmentations.dropout import functional as fdropout
from albumentations.augmentations.dropout.transforms import BaseDropout, BaseDropoutInitSchema, DropoutFillValue
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transform_params import SampledParams, TargetSet

__all__ = ["GridMask"]


class GridMask(BaseDropout):
    """Occlude images with repeated grid lines or cell patterns to train models against structured
    missing regions and partial object visibility.

    Args:
        num_grid_range (tuple[int, int]): Range for number of grid divisions along the shorter image side
            for "lines", or along each axis for cell patterns. Cell boundaries are rounded down to pixels.
            Default: (3, 7).
        line_width_range (tuple[float, float]): Range for line width as a fraction
            of grid cell size. Default: (0.2, 0.5).
        rotation_range (tuple[float, float]): Range for grid rotation in radians.
            Default: (0, 0) (no rotation).
        fill (float | tuple | str): Fill value for dropped pixels. Use "grayscale" to convert dropped regions
            to grayscale while preserving channel count. Default: 0.
        fill_mask (float | tuple | None): Fill value for mask. Must be None when fill="grayscale".
            Default: None.
        p (float): Probability of applying the transform. Default: 0.5.
        pattern (Literal["lines", "top_left", "top_left_inverse", "diagonal"]):
            Grid pattern. "lines" retains the original grid-line algorithm.
            Cell patterns divide each image axis into num_grid cells independently.
            "top_left" drops each cell's top-left region; "top_left_inverse" keeps only that region;
            "diagonal" also drops the bottom-right quarter. Default: "lines".
        cell_mask_ratio_range (tuple[float, float]): Uniformly sampled side fraction of the top-left
            region, strictly between 0 and 1. Ignored for "lines". Default: (0.5, 0.5).
        shift_xy (tuple[int, int] | None): Cell-pattern crop offset in pixels, wrapped by the integer
            cell width and height. None samples one shared X/Y offset for the entire grid.
            Ignored for "lines". Default: None.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        Any

    Supported bboxes:
        hbb

    Note:
        - Historical competition modes 0, 1, and 2 map to "top_left", "top_left_inverse", and "diagonal".
          A side ratio of 0.5 reproduces the quarter-cell patterns. For the DFDC keep_ratio parameter,
          use a side ratio of sqrt(1 - keep_ratio); the diagonal bottom-right quarter remains fixed.
        - Cell-pattern rotation uses nearest-neighbor interpolation and reflected borders on a binary grid,
          then crops the shared offset window. It does not reproduce historical interpolation at rotated boundaries.
        - line_width_range applies only to "lines". The default algorithm and its random draw order are unchanged.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.ones((100, 120, 3), dtype=np.uint8)
        >>> mask = np.ones((100, 120), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.GridMask(pattern="top_left", cell_mask_ratio_range=(0.3, 0.6), fill_mask=0, p=1),
        ... ], bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        ...    keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]), seed=137)
        >>> result = transform(image=image, mask=mask, bboxes=[[10, 10, 50, 50]], bbox_labels=[1],
        ...                    keypoints=[[20, 20]], keypoint_labels=["point"])
        >>> augmented_image, augmented_mask = result["image"], result["mask"]

    References:
        GridMask paper: https://arxiv.org/abs/2001.04086

    """

    class InitSchema(BaseDropoutInitSchema):
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
        pattern: Literal["lines", "top_left", "top_left_inverse", "diagonal"]
        cell_mask_ratio_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1, min_inclusive=False, max_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        shift_xy: Annotated[tuple[int, int], AfterValidator(check_range_bounds(0, None))] | None

    def __init__(
        self,
        num_grid_range: tuple[int, int] = (3, 7),
        line_width_range: tuple[float, float] = (0.2, 0.5),
        rotation_range: tuple[float, float] = (0, 0),
        fill: DropoutFillValue = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
        *,
        pattern: Literal["lines", "top_left", "top_left_inverse", "diagonal"] = "lines",
        cell_mask_ratio_range: tuple[float, float] = (0.5, 0.5),
        shift_xy: tuple[int, int] | None = None,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.num_grid_range = num_grid_range
        self.line_width_range = line_width_range
        self.rotation_range = rotation_range
        self.pattern = pattern
        self.cell_mask_ratio_range = cell_mask_ratio_range
        self.shift_xy = shift_xy

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        image_shape = targets.require_aligned_spatial_shape(2)
        if self.pattern != "lines":
            return self._sample_cell_parameters(image_shape, sampling)
        num_grid = sampling.py_random.randint(*self.num_grid_range)
        line_width_ratio = sampling.py_random.uniform(*self.line_width_range)
        rotation = sampling.py_random.uniform(*self.rotation_range)

        holes = fdropout.generate_grid_mask_holes(
            image_shape,
            num_grid,
            line_width_ratio,
            rotation,
            sampling.random_generator,
        )

        sampling.applied_overrides.update(
            {
                "num_grid_range": num_grid,
                "line_width_range": line_width_ratio,
                "rotation_range": rotation,
            },
        )
        return SampledParams(params={"holes": holes, "seed": sampling.random_generator.integers(0, 2**32 - 1)})

    def _sample_cell_parameters(
        self,
        image_shape: tuple[int, int],
        sampling: SamplingContext,
    ) -> SampledParams:
        num_grid = sampling.py_random.randint(*self.num_grid_range)
        ratio = sampling.py_random.uniform(*self.cell_mask_ratio_range)
        rotation = sampling.py_random.uniform(*self.rotation_range)
        shift_xy = self.shift_xy
        if shift_xy is None:
            height, width = image_shape
            shift_xy = (
                int(sampling.random_generator.integers(max(1, width // num_grid))),
                int(sampling.random_generator.integers(max(1, height // num_grid))),
            )
        holes = fdropout.generate_grid_cell_holes(image_shape, num_grid, ratio, self.pattern, rotation, shift_xy)
        sampling.applied_overrides.update(
            {
                "num_grid_range": num_grid,
                "cell_mask_ratio_range": ratio,
                "rotation_range": rotation,
                "shift_xy": shift_xy,
            },
        )
        return SampledParams(params={"holes": holes, "seed": sampling.random_generator.integers(0, 2**32 - 1)})
