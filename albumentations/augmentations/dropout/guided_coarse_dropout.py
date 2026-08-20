"""Implementation of GuidedCoarseDropout augmentation.

This module provides the GuidedCoarseDropout transform, which applies coarse dropout
guided by an external binary mask while protecting selected bounding box regions.
The guidance mask is passed as metadata through user_data, and protected bboxes are
dilated before rasterization to guarantee strict spatial protection.

Ref: https://github.com/albumentations-team/AlbumentationsX/issues/338
"""

from typing import Annotated, Any, cast

import cv2
import numpy as np
from pydantic import AfterValidator

import albumentations.augmentations.dropout.functional as fdropout
from albumentations.augmentations.dropout.transforms import BaseDropoutInitSchema, DropoutFillValue
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.invocation import SamplingContext
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.type_definitions import ALL_TARGETS, ImageType

__all__ = ["GuidedCoarseDropout"]


class GuidedCoarseDropout(DualTransform):
    """Apply coarse dropout guided by an external binary mask with optional bbox protection.

    This transform reads a binary guidance mask from ``user_data[region_key]`` and samples
    rectangular dropout holes whose centers lie on eligible pixels.  Protected bounding
    boxes (selected by label) are dilated by a relative margin, rasterized, and subtracted
    from the guidance mask to form the eligible region.  Sampled rectangles are intersected
    with the eligible mask before filling, which guarantees:

    * ``dropout_mask ⊆ guidance_mask``
    * ``dropout_mask ∩ protected_mask = ∅``

    The guidance mask is **not** transformed by geometric augmentations — place this
    transform **before** any geometric transforms when the mask uses original-image
    coordinates.

    Args:
        region_key (str): Key inside ``user_data`` that holds the binary guidance mask
            (2-D ``np.ndarray``).  ``True`` / non-zero means dropout is **allowed**.
            Default: ``"guidance_mask"``.
        protected_bbox_labels (list[str | int | float] | None): Bbox class labels whose
            regions must remain unchanged.  String labels are resolved through the
            existing label encoder.  ``None`` means no protection.  Default: ``None``.
        protection_margin (float | tuple[float, float]): Relative dilation applied to
            each protected bbox before rasterization.  A scalar ``m`` expands each side
            by ``m * box_width`` (horizontally) and ``m * box_height`` (vertically).
            A tuple ``(m_h, m_w)`` uses separate factors for height and width.
            Dilated boxes are clipped to image bounds.  Default: ``0.0``.
        num_holes_range (tuple[int, int]): Inclusive range ``(min, max)`` for the number
            of rectangular holes sampled **per image**.  Default: ``(1, 1)``.
        hole_height_range (tuple[float, float]): Range ``(min, max)`` for hole height
            as a **fraction of image height**.  Default: ``(0.05, 0.20)``.
        hole_width_range (tuple[float, float]): Range ``(min, max)`` for hole width
            as a **fraction of image width**.  Default: ``(0.05, 0.20)``.
        fill (float | tuple[float, ...] | str): Value for dropped pixels.  Accepts
            numeric constants, ``"random"``, ``"random_uniform"``, ``"inpaint_telea"``,
            ``"inpaint_ns"``, or ``"grayscale"``.  Default: ``0``.
        fill_mask (float | None): Value for dropped pixels in segmentation masks.
            ``None`` leaves masks unchanged.  Default: ``None``.
        p (float): Probability of applying the transform.  Default: ``0.5``.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb

    Note:
        - The guidance mask must have the same height and width as the image.
        - If the eligible region is empty (guidance fully protected or all-zero),
          the transform is a no-op.
        - Empty bboxes, missing labels, and missing guidance keys are all no-ops.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> saliency = np.zeros((100, 100), dtype=np.uint8)
        >>> saliency[20:80, 20:80] = 1  # allow dropout in the center
        >>> bboxes = [[40, 40, 60, 60]]
        >>> labels = ["cat"]
        >>> transform = A.Compose(
        ...     [
        ...         A.GuidedCoarseDropout(
        ...             region_key="saliency_mask",
        ...             protected_bbox_labels=["cat"],
        ...             protection_margin=0.10,
        ...             num_holes_range=(1, 3),
        ...             hole_height_range=(0.05, 0.20),
        ...             hole_width_range=(0.05, 0.20),
        ...             fill="inpaint_telea",
        ...             p=1.0,
        ...         ),
        ...     ],
        ...     bbox_params=A.BboxParams(
        ...         coord_coord_format="pascal_voc",
        ...         label_fields=["labels"],
        ...     ),
        ... )
        >>> result = transform(
        ...     image=image,
        ...     user_data={"saliency_mask": saliency},
        ...     bboxes=bboxes,
        ...     labels=labels,
        ... )

    References:
        Issue: https://github.com/albumentations-team/AlbumentationsX/issues/338

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseDropoutInitSchema):
        region_key: str
        protected_bbox_labels: list[str | int | float] | None = None
        protection_margin: float | tuple[float, float]
        num_holes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        hole_height_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        hole_width_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

    def __init__(
        self,
        region_key: str = "guidance_mask",
        protected_bbox_labels: list[str | int | float] | None = None,
        protection_margin: float | tuple[float, float] = 0.0,
        num_holes_range: tuple[int, int] = (1, 1),
        hole_height_range: tuple[float, float] = (0.05, 0.20),
        hole_width_range: tuple[float, float] = (0.05, 0.20),
        fill: DropoutFillValue = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.region_key = region_key
        self.protected_bbox_labels = protected_bbox_labels
        self.protection_margin = protection_margin
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.fill = fill  # type: ignore[assignment]
        self.fill_mask = fill_mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_guidance_mask(self, data: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray | None:
        """Read the binary guidance mask from user_data[region_key].

        Returns a boolean 2-D array or None when the key is absent.
        """
        user_data = data.get("user_data")
        if user_data is None or not isinstance(user_data, dict):
            return None

        mask = user_data.get(self.region_key)
        if mask is None:
            return None

        # Squeeze channel dim if present (e.g. (H, W, 1))
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)
            elif mask.shape[0] == 1:
                mask = np.squeeze(mask, axis=0)

        if mask.ndim != 2:
            msg = (
                f"Guidance mask must be 2-D (H, W), got {mask.ndim}-D with shape {mask.shape}. "
                f"Key: user_data['{self.region_key}']"
            )
            raise ValueError(msg)

        if mask.shape != image_shape:
            msg = (
                f"Guidance mask shape {mask.shape} does not match image shape {image_shape}. "
                f"Key: user_data['{self.region_key}']"
            )
            raise ValueError(msg)

        return mask.astype(bool, copy=False)

    def _get_protected_bboxes(self, data: dict[str, Any]) -> np.ndarray | None:
        """Filter bboxes by protected_bbox_labels.

        Re-uses the same label-encoding pattern as ConstrainedCoarseDropout.
        Returns denormalized bboxes (pixel coords) or None.
        """
        if self.protected_bbox_labels is None:
            return None

        bboxes = data.get("bboxes")
        if bboxes is None or len(bboxes) == 0:
            return None

        bbox_processor = self.get_processor("bboxes")
        if bbox_processor is None:
            return None

        # Resolve string labels via the label encoder
        if not all(isinstance(lbl, (int, float)) for lbl in self.protected_bbox_labels):
            label_fields = bbox_processor.params.label_fields
            if label_fields is None:
                raise ValueError(
                    "BboxParams.label_fields must be specified when using string labels in protected_bbox_labels",
                )
            first_class_label = label_fields[0]
            metadata = bbox_processor.label_manager.metadata["bboxes"][first_class_label]
            if metadata.encoder is None:
                raise ValueError(f"No encoder found for label field {first_class_label}")
            try:
                target_labels = metadata.encoder.transform(self.protected_bbox_labels)
            except KeyError:
                return None
        else:
            target_labels = np.array(self.protected_bbox_labels)

        # Column 4 holds the encoded label
        mask = np.isin(bboxes[:, 4], target_labels)
        filtered = bboxes[mask, :4]
        if len(filtered) == 0:
            return None

        image_shape = data["image"].shape[:2]
        return denormalize_bboxes(filtered, image_shape)

    def _build_protected_mask(
        self,
        bboxes: np.ndarray | None,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Rasterize protected bboxes (dilated + clipped) into a boolean mask."""
        height, width = image_shape
        protected = np.zeros(image_shape, dtype=bool)

        if bboxes is None or len(bboxes) == 0:
            return protected

        margin_h, margin_w = (
            (self.protection_margin, self.protection_margin)
            if isinstance(self.protection_margin, (int, float))
            else self.protection_margin
        )

        for x1, y1, x2, y2 in bboxes:
            box_w = x2 - x1
            box_h = y2 - y1
            dx = margin_w * box_w
            dy = margin_h * box_h

            rx1 = max(0, int(x1 - dx))
            ry1 = max(0, int(y1 - dy))
            rx2 = min(width, int(x2 + dx + 0.5))
            ry2 = min(height, int(y2 + dy + 0.5))

            protected[ry1:ry2, rx1:rx2] = True

        return protected

    @staticmethod
    def _sample_centers_uniform(
        eligible_mask: np.ndarray,
        num_holes: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Sample *num_holes* (x, y) centers uniformly from eligible pixels.

        Returns an (N, 2) int32 array of [x, y] pairs.
        """
        ys, xs = np.nonzero(eligible_mask)
        if len(ys) == 0:
            return np.empty((0, 2), dtype=np.int32)

        indices = random_generator.integers(0, len(ys), size=num_holes)
        return np.stack([xs[indices], ys[indices]], axis=-1).astype(np.int32)

    @staticmethod
    def _centers_to_holes(
        centers: np.ndarray,
        hole_heights: np.ndarray,
        hole_widths: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Convert (x, y) centers + dimensions into clipped [x1, y1, x2, y2] holes."""
        height, width = image_shape
        half_h = hole_heights // 2
        half_w = hole_widths // 2

        x1 = np.clip(centers[:, 0] - half_w, 0, width)
        y1 = np.clip(centers[:, 1] - half_h, 0, height)
        x2 = np.clip(centers[:, 0] + half_w, 0, width)
        y2 = np.clip(centers[:, 1] + half_h, 0, height)

        return np.stack([x1, y1, x2, y2], axis=-1).astype(np.int32)

    @staticmethod
    def _rasterize_holes(holes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        """Rasterize rectangle holes into a boolean mask."""
        mask = np.zeros(image_shape, dtype=bool)
        for x1, y1, x2, y2 in holes:
            mask[y1:y2, x1:x2] = True
        return mask

    # ------------------------------------------------------------------
    # Core pipeline methods
    # ------------------------------------------------------------------

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        """Compute the pixel-level dropout mask.

        Steps:
        1. Read guidance mask from user_data via region_key.
        2. Select protected bboxes by label; dilate + clip; rasterize.
        3. eligible_mask = guidance_mask & ~protected_mask.
        4. If eligible_mask empty → no-op.
        5. Sample N hole centers uniformly across eligible pixels.
        6. Create rectangles sized relative to image dimensions.
        7. Rasterize rectangles; intersect with eligible_mask → dropout_mask.
        """
        image_shape = params["shape"][:2]
        height, width = image_shape

        # 1. Guidance mask
        guidance_mask = self._get_guidance_mask(data, image_shape)
        if guidance_mask is None:
            return {"dropout_mask": None}

        # 2. Protected bboxes → protected mask
        protected_bboxes = self._get_protected_bboxes(data)
        protected_mask = self._build_protected_mask(protected_bboxes, image_shape)

        # 3. Eligible mask
        eligible_mask = guidance_mask & ~protected_mask

        # 4. Early exit if nothing is eligible
        if not eligible_mask.any():
            return {"dropout_mask": None}

        # 5. Sample number of holes and centers
        num_holes = sampling.py_random.randint(*self.num_holes_range)
        centers = self._sample_centers_uniform(eligible_mask, num_holes, sampling.random_generator)

        if len(centers) == 0:
            return {"dropout_mask": None}

        # 6. Hole dimensions — relative to full image
        hole_heights = (
            height
            * sampling.random_generator.uniform(
                self.hole_height_range[0],
                self.hole_height_range[1],
                size=num_holes,
            )
        ).astype(np.int32)
        hole_widths = (
            width
            * sampling.random_generator.uniform(
                self.hole_width_range[0],
                self.hole_width_range[1],
                size=num_holes,
            )
        ).astype(np.int32)

        # Ensure minimum 1px
        hole_heights = np.maximum(hole_heights, 1)
        hole_widths = np.maximum(hole_widths, 1)

        # 7. Build rectangles, rasterize, intersect with eligible
        holes = self._centers_to_holes(centers, hole_heights, hole_widths, image_shape)
        rect_mask = self._rasterize_holes(holes, image_shape)
        dropout_mask = rect_mask & eligible_mask

        if not dropout_mask.any():
            return {"dropout_mask": None}

        sampling.applied_overrides.update(
            {
                "num_holes_range": num_holes,
                "hole_height_range": (int(hole_heights.min()), int(hole_heights.max())),
                "hole_width_range": (int(hole_widths.min()), int(hole_widths.max())),
            },
        )

        return {
            "dropout_mask": dropout_mask,
            "seed": sampling.random_generator.integers(0, 2**32 - 1),
        }

    # ------------------------------------------------------------------
    # Apply methods (pixel-level mask, similar to MaskDropout)
    # ------------------------------------------------------------------

    def apply(self, img: ImageType, dropout_mask: np.ndarray | None = None, seed: int = 0, **params: Any) -> ImageType:
        if dropout_mask is None:
            return img

        if self.fill in {"inpaint_telea", "inpaint_ns"}:
            mask_uint8 = dropout_mask.astype(np.uint8)
            _, _, w, h = cv2.boundingRect(mask_uint8)
            radius = max(1, min(3, max(w, h) // 2))
            method = cv2.INPAINT_TELEA if self.fill == "inpaint_telea" else cv2.INPAINT_NS
            return cast("ImageType", cv2.inpaint(img, mask_uint8, radius, method))

        if self.fill == "grayscale":
            return fdropout.fill_mask_with_grayscale(img, dropout_mask)

        img = img.copy()

        if self.fill == "random":
            random_gen = np.random.Generator(np.random.PCG64(seed))
            random_fill = fdropout.generate_random_fill(img.dtype, img[dropout_mask].shape, random_gen)
            img[dropout_mask] = random_fill
            return img

        if self.fill == "random_uniform":
            random_gen = np.random.Generator(np.random.PCG64(seed))
            fill_shape = (1, img.shape[2]) if img.ndim == 3 else (1,)
            random_fill = fdropout.generate_random_fill(img.dtype, fill_shape, random_gen)
            img[dropout_mask] = random_fill
            return img

        # Numeric fill
        if isinstance(self.fill, (int, float)):
            img[dropout_mask] = self.fill
        else:
            fill_array = np.array(self.fill, dtype=img.dtype)
            img[dropout_mask] = fill_array

        return img

    def apply_to_mask(
        self,
        mask: ImageType,
        dropout_mask: np.ndarray | None = None,
        **params: Any,
    ) -> ImageType:
        if dropout_mask is None or self.fill_mask is None:
            return mask
        mask = mask.copy()
        mask[dropout_mask] = self.fill_mask
        return mask

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        dropout_mask: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        if dropout_mask is None or len(bboxes) == 0:
            return bboxes

        processor = cast("BboxProcessor", self.get_processor("bboxes"))
        if processor is None:
            return bboxes

        image_shape = params["shape"][:2]
        denorm = denormalize_bboxes(bboxes, image_shape)
        result = fdropout.mask_dropout_bboxes(
            denorm,
            dropout_mask,
            image_shape,
            processor.params.min_area,
            processor.params.min_visibility,
        )
        return normalize_bboxes(result, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        dropout_mask: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        if dropout_mask is None or len(keypoints) == 0:
            return keypoints

        processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
        if processor is None or not processor.params.remove_invisible:
            return keypoints

        return fdropout.mask_dropout_keypoints(keypoints, dropout_mask)
