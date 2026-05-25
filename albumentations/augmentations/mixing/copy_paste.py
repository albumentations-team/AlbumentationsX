"""CopyAndPaste mixing transform."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

from ._transforms_shared import (
    _BBOX_INSTANCE_ID,
    _KP_INSTANCE_ID,
    AfterValidator,
    BaseTransformInitSchema,
    BboxProcessor,
    ClassVar,
    DualTransform,
    Field,
    ImageType,
    KeypointsProcessor,
    StackedMasks4D,
    Targets,
    check_range_bounds,
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
    convert_keypoints_from_albumentations,
    convert_keypoints_to_albumentations,
    cv2,
    fgeometric,
    fmixing,
    nondecreasing,
    np,
    warnings,
)


class CopyAndPaste(DualTransform):
    """Paste object instances onto the primary image, updating all annotations (instance masks,
    bboxes, keypoints). Designed for instance segmentation training.

    Each donor object is tight-cropped to its `mask` (or `bbox` rect for bbox-only donors,
    optionally expanded to include `keypoints`), shrunk to fit the target image with aspect
    preserved (no upscaling), optionally jittered by `scale_range`, and stamped at a uniformly
    random location inside the target. Existing instances that become sufficiently occluded by
    pasted objects are removed from annotations.

    All per-object **content** augmentation (rotation, flip, color jitter, scale-up beyond fit)
    is the user's responsibility — the transform only does crop -> shrink-fit -> optional scale
    jitter -> uniform random placement -> stamp.

    Note:
        Most Copy-Paste implementations (e.g. detectron2) accept a single donor image with all
        its instance masks and internally sample a random subset of instances to paste, coupling
        donor selection, instance sampling, and pasting into one opaque step. This implementation
        separates those concerns: donor selection and instance selection are done by the user
        externally, and the transform pastes every object in the provided list. The metadata
        format is `list[dict]` (one dict per object), consistent with `Mosaic`.

    Args:
        min_visibility_after_paste (float): Minimum mask area ratio (area_after / area_before) for
            an existing instance to survive after occlusion by pasted objects. Instances whose
            remaining visible area falls below this threshold are removed from masks and bboxes.
            Default: 0.05.
        blend_mode (Literal["hard", "gaussian"]): How to blend pasted pixels. "hard" does direct
            pixel copy (paper default). "gaussian" applies gaussian blur to the alpha mask for
            soft edges at instance boundaries. Default: "hard".
        blend_sigma_range (tuple[float, float]): Sigma range for gaussian blur when
            blend_mode="gaussian". Ignored when blend_mode="hard". Default: (1.0, 3.0).
        scale_range (tuple[float, float]): Multiplicative scale jitter applied on top of the
            shrink-to-fit scale. Sampled uniformly from this range and capped at the fit scale,
            so the result can shrink the donor further but never exceed fit-to-target.
            Default: `(1.0, 1.0)` (pure shrink-to-fit, no jitter).
        min_paste_area (int): Minimum scaled paste footprint area (pixels). Donors whose final
            scaled `H*W` falls below this value are silently dropped — useful to avoid pasting
            tiny blob-noise from huge donors onto small targets. Default: 1.
        metadata_key (str): Key in the Compose call data dict containing the list of object
            dictionaries to paste. Default: "copy_paste_metadata".
        p (float): Probability of applying the transform. Default: 0.5.

    Metadata Format:
        The value at `metadata_key` must be a list of dicts. Each dict describes one donor object;
        donor image dimensions can differ from the target image dimensions and from each other.
        Coordinates for `bbox` / `keypoints` MUST be in the same `coord_format` declared in the
        pipeline's `BboxParams` / `KeypointParams`, normalized (where applicable) to the
        **donor** image dimensions — exactly as you would provide them if the donor were the
        primary image. The transform handles internal coordinate conversions.
            - image (np.ndarray): Donor image (Hd, Wd, C) containing the object. Required.
            - mask (np.ndarray): Binary **instance** mask (Hd, Wd) defining the paste footprint.
              Optional when `bbox` is provided. Empty masks (no positive pixels) are dropped.
            - bbox (np.ndarray | list): Horizontal bounding box of the object in
              `BboxParams.coord_format` on donor dims. Required for bbox-only donors; optional
              otherwise (a tight box is derived from `mask` if absent).
            - semantic_mask (np.ndarray): Optional semantic label map (Hd, Wd), same dims as
              `image`. When provided AND the pipeline passes a `mask` target, the donor's class
              ids replace the primary semantic mask inside the paste footprint. When the pipeline
              has a `mask` target but no donor supplies `semantic_mask`, a `UserWarning` fires
              once.
            - keypoints (np.ndarray): Keypoints in `KeypointParams.coord_format` on donor dims.
              Optional. Keypoints outside the mask/bbox tight crop expand the crop bounds so they
              are preserved into the target.
            - bbox_labels (dict[str, Any]): Label values for this donor's bbox, keyed by the
              names declared in `BboxParams.label_fields`. E.g. `{"class_id": 3, "is_crowd": 0}`.
            - keypoint_labels (dict[str, Any]): Label values for this donor's keypoints, keyed by
              the names declared in `KeypointParams.label_fields`. A list value is accepted when
              the object has multiple keypoints.

    Targets:
        image, mask, bboxes, keypoints

    Keypoints vs instance masks:
        When the pipeline supplies instance masks as `masks` (N, H, W) and
        `paste_surviving_indices` is computed from them, primary keypoints are filtered only if
        `keypoints.shape[0]` equals N (one row per instance, same order as stacked masks).
        Otherwise existing keypoints are left unchanged and pasted keypoints are still appended.

    Image types:
        uint8, float32

    Supported bboxes:
        hbb

    Reference:
        Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation: https://arxiv.org/abs/2012.07177

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Primary data (target image is 100x100)
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> instance_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        >>> instance_masks[0, 10:30, 10:30] = 1
        >>> bboxes = np.array([[10, 10, 30, 30]], dtype=np.float32)
        >>> class_labels = [1]
        >>>
        >>> # Donor 1: tight 40x40 mask-based donor (any donor dims work).
        >>> donor1_image = np.full((40, 40, 3), 200, dtype=np.uint8)
        >>> donor1_mask = np.ones((40, 40), dtype=np.uint8)
        >>>
        >>> # Donor 2: bbox-only donor on a 60x80 image (rectangle paste footprint).
        >>> donor2_image = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)
        >>>
        >>> transform = A.Compose([
        ...     A.CopyAndPaste(
        ...         min_visibility_after_paste=0.05,
        ...         scale_range=(0.5, 1.0),  # randomly shrink donors to 50%-100% of fit
        ...         p=1.0,
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['class_labels']))
        >>>
        >>> result = transform(
        ...     image=image,
        ...     masks=instance_masks,
        ...     bboxes=bboxes,
        ...     class_labels=class_labels,
        ...     copy_paste_metadata=[
        ...         {
        ...             'image': donor1_image,
        ...             'mask': donor1_mask,
        ...             'bbox_labels': {'class_labels': 2},
        ...         },
        ...         {
        ...             'image': donor2_image,
        ...             'bbox': [10, 5, 70, 55],  # pascal_voc on 60x80 donor dims
        ...             'bbox_labels': {'class_labels': 3},
        ...         },
        ...     ],
        ... )
        >>> result_image = result['image']
        >>> result_masks = result['masks']         # (N_surviving + K, H, W)
        >>> result_bboxes = result['bboxes']       # Updated bboxes (in pascal_voc, target dims)
        >>> result_labels = result['class_labels'] # Updated labels

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        min_visibility_after_paste: float
        blend_mode: Literal["hard", "gaussian"]
        blend_sigma_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        scale_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        min_paste_area: Annotated[int, Field(ge=1)]
        metadata_key: str

    def __init__(
        self,
        min_visibility_after_paste: float = 0.05,
        blend_mode: Literal["hard", "gaussian"] = "hard",
        blend_sigma_range: tuple[float, float] = (1.0, 3.0),
        scale_range: tuple[float, float] = (1.0, 1.0),
        min_paste_area: int = 1,
        metadata_key: str = "copy_paste_metadata",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.min_visibility_after_paste = min_visibility_after_paste
        self.blend_mode = blend_mode
        self.blend_sigma_range = blend_sigma_range
        self.scale_range = scale_range
        self.min_paste_area = min_paste_area
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key]

    @staticmethod
    def _instance_masks_to_3d(masks: Any) -> np.ndarray | None:
        """Normalize masks to (N, H, W) for CopyAndPaste visibility from a stacked ndarray (4D ok) or a sequence of
        per-instance (H, W) arrays.
        """
        if masks is None:
            return None
        if isinstance(masks, np.ndarray):
            if masks.size == 0:
                return None
            return masks.squeeze(-1) if masks.ndim == 4 else masks
        if isinstance(masks, Sequence) and not isinstance(masks, (str, bytes, np.ndarray)):
            if len(masks) == 0:
                return None
            return np.stack([np.asarray(m) for m in masks], axis=0)
        return None

    def _compute_surviving_indices(
        self,
        data: dict[str, Any],
        paste_union_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, int | None]:
        """Return surviving indices and n_instances from stacked masks vs paste visibility,
        or (None, None) without instance masks.

        Compares each instance mask to the opaque paste footprint for visibility ratios.
        n_instances is the stacked `masks` axis length and matches keypoint row count when filtering survivors.
        """
        masks_3d = self._instance_masks_to_3d(data.get("masks"))
        if masks_3d is None:
            return None, None

        n_instances = int(masks_3d.shape[0])
        visibility = fmixing.compute_instance_visibility(masks_3d, paste_union_mask)
        surviving = np.where(visibility >= self.min_visibility_after_paste)[0]
        return surviving, n_instances

    def _bbox_instance_id_column(self, data: dict[str, Any]) -> np.ndarray | None:
        """Return the per-row `_bbox_instance_id` integer values from `data["bboxes"]`
        when instance binding is active, otherwise return `None`.

        Drives ID-based filtering in `apply_to_bboxes` and keeps `apply_to_masks` row-aligned with the
        surviving bboxes so `Compose._repack_instances` can take its row-aligned fast path instead of the
        sparse fallback that crashes when positions and IDs disagree.
        """
        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
        if bbox_processor is None:
            return None
        label_fields = bbox_processor.params.label_fields or []
        if _BBOX_INSTANCE_ID not in label_fields:
            return None
        bboxes = data.get("bboxes")
        if bboxes is None:
            return np.empty((0,), dtype=np.int64)
        if not isinstance(bboxes, np.ndarray):
            raise TypeError(
                f"CopyAndPaste expects data['bboxes'] to be a numpy.ndarray when instance binding is active, "
                f"got {type(bboxes).__name__}",
            )
        if bboxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        n_lf = len(label_fields)
        id_col_idx = bboxes.shape[1] - n_lf + label_fields.index(_BBOX_INSTANCE_ID)
        return bboxes[:, id_col_idx].astype(np.int64, copy=False)

    def _resolve_paste_surviving_ids(
        self,
        data: dict[str, Any],
        surviving_indices: np.ndarray | None,
        n_instances: int | None,
    ) -> tuple[list[int] | None, int]:
        """Compute the ordered list of surviving original instance IDs and the next
        paste ID for ID-driven survival selection in CopyAndPaste.

        Returns `(paste_surviving_ids_ordered, next_paste_instance_id)` where:
        - `paste_surviving_ids_ordered` is in current bbox order, intersected with visibility-surviving
          mask positions (treated as IDs since the binding contract keeps `data["masks"]` positionally
          indexed by `_bbox_instance_id`). Empty mask rows are excluded so we never resurrect
          zero-area instances that an upstream transform happened to leave behind. Returns `None`
          when binding is not active.
        - `next_paste_instance_id` is `max(all_existing_ids) + 1` so pasted IDs cannot collide with
          existing ones (prevents the bug where positions != IDs after upstream bbox filtering).
        """
        bbox_id_col = self._bbox_instance_id_column(data)
        if bbox_id_col is None:
            if surviving_indices is not None and surviving_indices.size > 0:
                next_paste = int(np.max(surviving_indices)) + 1
            else:
                next_paste = 0
            return None, next_paste

        masks_3d = self._instance_masks_to_3d(data.get("masks"))
        ordered = self._select_alive_ids(bbox_id_col, masks_3d, surviving_indices)

        candidates: list[int] = []
        if n_instances is not None and n_instances > 0:
            candidates.append(n_instances - 1)
        if bbox_id_col.size > 0:
            candidates.append(int(bbox_id_col.max()))
        next_paste = (max(candidates) + 1) if candidates else 0
        return ordered, next_paste

    @staticmethod
    def _select_alive_ids(
        bbox_id_col: np.ndarray,
        masks_3d: np.ndarray | None,
        surviving_indices: np.ndarray | None,
    ) -> list[int]:
        """Vectorized survivor selection: keep `_bbox_instance_id` values whose mask row is both
        non-empty and visible after paste, preserving current bbox order.

        Filters `bbox_id_col` (per-row `_bbox_instance_id`) to the IDs that point at a non-empty mask row
        in `masks_3d` AND lie in `surviving_indices` (the visibility-survivor set from
        `_compute_surviving_indices`). Guards against upstream transforms (e.g. Crop without explicit
        `min_area`) leaving zero-area mask rows behind, since `compute_instance_visibility` returns 1.0
        for empty masks and would otherwise resurrect dead instances.
        """
        if masks_3d is None or bbox_id_col.size == 0:
            return bbox_id_col.astype(int).tolist()
        n_rows = masks_3d.shape[0]
        in_range = (bbox_id_col >= 0) & (bbox_id_col < n_rows)
        alive = in_range.copy()
        valid_ids = bbox_id_col[in_range]
        nonempty_mask = np.any(masks_3d > 0, axis=(1, 2))
        alive[in_range] &= nonempty_mask[valid_ids]
        if surviving_indices is not None:
            visibility_mask = np.zeros(n_rows, dtype=bool)
            if surviving_indices.size > 0:
                visibility_mask[surviving_indices.astype(np.int64)] = True
            alive[in_range] &= visibility_mask[valid_ids]
        return bbox_id_col[alive].astype(int).tolist()

    @staticmethod
    def _resize_mask_to_target(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]:
            return fgeometric.resize(mask, target_shape, cv2.INTER_NEAREST)
        return mask

    # ------------------------------------------------------------------
    # Donor-side / target-side coord_format <-> pixel conversion helpers.
    # The user passes donor `bbox` / `keypoints` in the same coord_format declared in
    # BboxParams / KeypointParams (today's contract). The transform converts them into
    # donor pixel coords for the geometric remap, then back to the pipeline coord_format
    # on target dims before handing to _prepare_pasted_bboxes / _prepare_pasted_keypoints.
    # ------------------------------------------------------------------

    @staticmethod
    def _donor_bbox_to_pascal_px(
        item: dict[str, Any],
        bbox_processor: BboxProcessor | None,
        donor_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Convert one user-provided donor bbox into pascal_voc pixel coordinates on donor dims,
        round-tripping through the albumentations normalized representation.

        Returns a 1D array (4 cols for HBB, 5 cols for OBB) preserving the angle column for OBB.
        Returns None when the item has no bbox or no bbox processor is wired in.
        """
        if bbox_processor is None or "bbox" not in item:
            return None
        coord_format = bbox_processor.params.coord_format
        bbox_type = bbox_processor.params.bbox_type
        bbox_2d = np.asarray(item["bbox"], dtype=np.float32).reshape(1, -1)
        if coord_format == "albumentations":
            alb = bbox_2d
        else:
            alb = convert_bboxes_to_albumentations(
                bbox_2d,
                coord_format,
                donor_shape,
                bbox_type,
                check_validity=False,
            )
        pascal_px = convert_bboxes_from_albumentations(
            alb,
            "pascal_voc",
            donor_shape,
            bbox_type,
            check_validity=False,
        )
        return pascal_px.reshape(-1)

    @staticmethod
    def _donor_keypoints_to_pixels(
        item: dict[str, Any],
        kp_processor: KeypointsProcessor | None,
    ) -> np.ndarray | None:
        """Convert one user-provided donor keypoint array into the internal albumentations layout
        with donor pixel positions in cols 0/1 and z/angle/scale preserved.

        Returns an (N, 5+) array with pixel x in column 0, pixel y in column 1, plus z/angle/scale
        and any extras preserved. Coordinate values are donor pixel positions (the alb keypoint
        format is *not* normalized). Returns None when the item has no keypoints.
        """
        if kp_processor is None or "keypoints" not in item:
            return None
        raw = np.asarray(item["keypoints"], dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[np.newaxis]
        if raw.size == 0:
            return None
        coord_format = kp_processor.params.coord_format
        angle_in_degrees = kp_processor.params.angle_in_degrees
        # Shape arg only matters for check_validity (which we skip), so any (h, w) works.
        return convert_keypoints_to_albumentations(
            raw,
            coord_format,
            (1, 1),
            check_validity=False,
            angle_in_degrees=angle_in_degrees,
        )

    @staticmethod
    def _target_pascal_px_to_coord_format(
        pascal_px: np.ndarray,
        target_shape: tuple[int, int],
        bbox_processor: BboxProcessor,
    ) -> np.ndarray:
        """Convert one bbox in pascal_voc pixel coords on target dims back to the pipeline
        coord_format declared in BboxParams, returning a 1D array.
        """
        bbox_type = bbox_processor.params.bbox_type
        coord_format = bbox_processor.params.coord_format
        bbox_2d = pascal_px.reshape(1, -1)
        alb = convert_bboxes_to_albumentations(
            bbox_2d,
            "pascal_voc",
            target_shape,
            bbox_type,
            check_validity=False,
        )
        if coord_format == "albumentations":
            return alb.reshape(-1)
        return convert_bboxes_from_albumentations(
            alb,
            coord_format,
            target_shape,
            bbox_type,
            check_validity=False,
        ).reshape(-1)

    @staticmethod
    def _target_alb_keypoints_to_coord_format(
        alb_kps: np.ndarray,
        kp_processor: KeypointsProcessor,
    ) -> np.ndarray:
        """Convert (N, 5+) albumentations-format keypoints (with target-pixel x,y in cols 0,1) back to
        the pipeline coord_format. Returns an (N, 2+) array.
        """
        coord_format = kp_processor.params.coord_format
        angle_in_degrees = kp_processor.params.angle_in_degrees
        return convert_keypoints_from_albumentations(
            alb_kps,
            coord_format,
            (1, 1),
            check_validity=False,
            angle_in_degrees=angle_in_degrees,
        )

    # ------------------------------------------------------------------
    # Per-donor geometry helpers: tight crop -> fit-to-target shrink + jitter -> random placement -> stamp.
    # All work in pixel coords; coord_format <-> pixel translation is handled by the helpers above.
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_crop_bounds(
        item: dict[str, Any],
        bbox_px: np.ndarray | None,
        kp_alb: np.ndarray | None,
        donor_shape: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        """Resolve (y0, y1, x0, x1) tight crop bounds in donor pixel coords from mask / bbox /
        keypoints, returning None when there is no usable footprint.

        Mask path takes the mask tight bbox; bbox-only path uses floor/ceil of the user bbox rect.
        Keypoints (when present) expand the bounds so every keypoint stays inside the crop.
        Returns None when the donor has neither mask nor bbox or when the resulting crop is empty.
        """
        height, width = donor_shape
        bounds: tuple[int, int, int, int] | None = None

        mask = item.get("mask")
        if mask is not None and np.any(mask > 0):
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            row_idx = np.where(rows)[0]
            col_idx = np.where(cols)[0]
            bounds = (int(row_idx[0]), int(row_idx[-1]) + 1, int(col_idx[0]), int(col_idx[-1]) + 1)
        elif bbox_px is not None:
            x0_f, y0_f, x1_f, y1_f = (float(v) for v in bbox_px[:4])
            bounds = (
                max(0, int(np.floor(y0_f))),
                min(height, int(np.ceil(y1_f))),
                max(0, int(np.floor(x0_f))),
                min(width, int(np.ceil(x1_f))),
            )

        if bounds is None:
            return None

        if kp_alb is not None and kp_alb.size > 0:
            xs = kp_alb[:, 0]
            ys = kp_alb[:, 1]
            bounds = (
                min(bounds[0], int(np.floor(ys.min()))),
                max(bounds[1], int(np.ceil(ys.max())) + 1),
                min(bounds[2], int(np.floor(xs.min()))),
                max(bounds[3], int(np.ceil(xs.max())) + 1),
            )

        y0, y1, x0, x1 = bounds
        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(height, y1)
        x1 = min(width, x1)
        if y1 <= y0 or x1 <= x0:
            return None
        return y0, y1, x0, x1

    @staticmethod
    def _tight_crop(
        item: dict[str, Any],
        bounds: tuple[int, int, int, int],
        bbox_px: np.ndarray | None,
        kp_alb: np.ndarray | None,
    ) -> dict[str, Any]:
        """Slice donor arrays to the resolved bounds and translate bbox/keypoints into crop-local
        pixels, synthesizing a footprint mask when the donor is bbox-only.

        For bbox-only donors (no mask in item) the helper synthesizes an all-ones uint8 mask covering
        the crop so the downstream paste pipeline has a single uniform "footprint" representation.
        """
        y0, y1, x0, x1 = bounds
        crop_h, crop_w = y1 - y0, x1 - x0

        image = item["image"][y0:y1, x0:x1]
        if "mask" in item and item["mask"] is not None:
            mask = item["mask"][y0:y1, x0:x1]
        else:
            mask = np.ones((crop_h, crop_w), dtype=np.uint8)
        semantic_mask = item["semantic_mask"][y0:y1, x0:x1] if item.get("semantic_mask") is not None else None

        local_bbox: np.ndarray | None = None
        if bbox_px is not None:
            local_bbox = bbox_px.copy()
            local_bbox[0] -= x0
            local_bbox[1] -= y0
            local_bbox[2] -= x0
            local_bbox[3] -= y0

        local_kp: np.ndarray | None = None
        if kp_alb is not None and kp_alb.size > 0:
            local_kp = kp_alb.copy()
            local_kp[:, 0] -= x0
            local_kp[:, 1] -= y0

        return {
            "image": image,
            "mask": mask,
            "semantic_mask": semantic_mask,
            "bbox_px": local_bbox,
            "kp_alb": local_kp,
        }

    @staticmethod
    def _fit_and_jitter_scale(
        crop_h: int,
        crop_w: int,
        target_shape: tuple[int, int],
        scale_jitter: float,
    ) -> float:
        """Compute the per-donor scale combining the shrink-to-fit cap with a uniform `scale_range`
        jitter, capped so the scaled donor still fits inside the target image.

        `s_fit = min(1, target_h/crop_h, target_w/crop_w)` ensures the scaled crop fits inside the
        target without upscaling. The final scale is `min(s_fit, s_fit * scale_jitter)` so users
        opting into `scale_range > 1.0` cannot upscale beyond what fits.
        """
        height, width = target_shape
        s_fit = min(1.0, height / max(crop_h, 1), width / max(crop_w, 1))
        return min(s_fit, s_fit * scale_jitter)

    @staticmethod
    def _resize_cropped(cropped: dict[str, Any], scale: float) -> dict[str, Any]:
        """Resize the cropped donor data by `scale`: INTER_AREA for the image, INTER_NEAREST for masks,
        and bbox / keypoint pixel coords multiplied by the same factor.

        Bbox / keypoint pixel coords are multiplied by the same factor. When `scale == 1.0` the
        arrays are returned as-is to skip the resize round trip.
        """
        image = cropped["image"]
        mask = cropped["mask"]
        crop_h, crop_w = mask.shape[:2]

        if abs(scale - 1.0) < 1e-6:
            new_h, new_w = crop_h, crop_w
            scaled_image = image
            scaled_mask = mask
            scaled_semantic = cropped["semantic_mask"]
        else:
            new_h = max(1, round(crop_h * scale))
            new_w = max(1, round(crop_w * scale))
            scaled_image = fgeometric.resize(image, (new_h, new_w), cv2.INTER_AREA)
            scaled_mask = fgeometric.resize(mask, (new_h, new_w), cv2.INTER_NEAREST)
            scaled_semantic = (
                fgeometric.resize(cropped["semantic_mask"], (new_h, new_w), cv2.INTER_NEAREST)
                if cropped["semantic_mask"] is not None
                else None
            )

        local_bbox = cropped["bbox_px"]
        if local_bbox is not None:
            local_bbox = local_bbox.copy()
            local_bbox[:4] = local_bbox[:4] * scale

        local_kp = cropped["kp_alb"]
        if local_kp is not None:
            local_kp = local_kp.copy()
            local_kp[:, :2] = local_kp[:, :2] * scale

        return {
            "image": scaled_image,
            "mask": scaled_mask,
            "semantic_mask": scaled_semantic,
            "bbox_px": local_bbox,
            "kp_alb": local_kp,
            "shape": (new_h, new_w),
        }

    def _sample_placement(self, scaled_h: int, scaled_w: int, target_shape: tuple[int, int]) -> tuple[int, int]:
        """Sample a top-left (y0, x0) for the scaled donor uniformly inside the target image,
        relying on the shrink-to-fit cap to guarantee non-negative placement bounds.

        Bounds are inclusive and guaranteed non-negative thanks to the shrink-to-fit cap in
        `_fit_and_jitter_scale`.
        """
        height, width = target_shape
        y0 = self.py_random.randint(0, max(0, height - scaled_h))
        x0 = self.py_random.randint(0, max(0, width - scaled_w))
        return y0, x0

    @staticmethod
    def _stamp(
        scaled: dict[str, Any],
        y0: int,
        x0: int,
        target_shape: tuple[int, int],
        composite_image: np.ndarray,
        semantic_canvas: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Stamp the scaled donor onto the target-shaped canvases (image + optional semantic mask)
        and shift the donor bbox / keypoints into target pixel coordinates.

        Returns `(stamped_mask, bbox_target_px, kp_target_alb)` where `stamped_mask` is the
        target-shaped binary footprint (uint8). `bbox_target_px` is in pascal_voc pixel coords on
        target dims (None if no donor bbox). `kp_target_alb` keeps the (N, 5+) albumentations
        layout with target-pixel x,y in cols 0,1 (None if no donor keypoints).
        """
        h_s, w_s = scaled["shape"]
        stamped_mask = np.zeros(target_shape, dtype=np.uint8)
        donor_mask = scaled["mask"]
        stamped_mask[y0 : y0 + h_s, x0 : x0 + w_s] = (donor_mask > 0).astype(np.uint8)

        donor_image = scaled["image"]
        # Use offset slicing on composite to avoid building a target-shaped copy of the donor image.
        local_bool = donor_mask > 0
        comp_view = composite_image[y0 : y0 + h_s, x0 : x0 + w_s]
        comp_view[local_bool] = donor_image[local_bool]

        if semantic_canvas is not None and scaled["semantic_mask"] is not None:
            sem_view = semantic_canvas[y0 : y0 + h_s, x0 : x0 + w_s]
            sem_view[local_bool] = scaled["semantic_mask"][local_bool]

        bbox_target_px = None
        if scaled["bbox_px"] is not None:
            bbox_target_px = scaled["bbox_px"].copy()
            bbox_target_px[0] += x0
            bbox_target_px[1] += y0
            bbox_target_px[2] += x0
            bbox_target_px[3] += y0

        kp_target_alb = None
        if scaled["kp_alb"] is not None:
            kp_target_alb = scaled["kp_alb"].copy()
            kp_target_alb[:, 0] += x0
            kp_target_alb[:, 1] += y0

        return stamped_mask, bbox_target_px, kp_target_alb

    @classmethod
    def _derive_bbox_from_mask(cls, mask: np.ndarray, bbox_processor: BboxProcessor) -> np.ndarray:
        """Derive a tight HBB from a binary mask in BboxParams.coord_format via internal Pascal VOC pixels; OBB appends
        angle zero.
        """
        rows = np.any(mask > 0, axis=1)
        height, width = mask.shape[:2]
        if not np.any(rows):
            pascal_px = np.zeros((1, 4), dtype=np.float32)
        else:
            cols = np.any(mask > 0, axis=0)
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            pascal_px = np.array(
                [
                    [
                        float(col_indices[0]),
                        float(row_indices[0]),
                        float(col_indices[-1] + 1),
                        float(row_indices[-1] + 1),
                    ],
                ],
                dtype=np.float32,
            )

        if bbox_processor.params.bbox_type == "obb":
            pascal_px = np.column_stack([pascal_px, np.zeros(1, dtype=np.float32)])

        return cls._target_pascal_px_to_coord_format(pascal_px.reshape(-1), (height, width), bbox_processor)

    @staticmethod
    def _keypoint_label_values_for_item(
        val: Any,
        num_keypoints: int,
        field: str,
        item_idx: int,
    ) -> list[Any]:
        if isinstance(val, np.ndarray):
            field_values = np.asarray(val).reshape(-1).tolist()
        elif isinstance(val, list):
            field_values = val
        else:
            field_values = [val] * num_keypoints
        if len(field_values) != num_keypoints:
            raise ValueError(
                f"CopyAndPaste: keypoint label field '{field}' must have one value per keypoint "
                f"for pasted object at index {item_idx}; got {len(field_values)} for "
                f"{num_keypoints} keypoints.",
            )
        return field_values

    def _prepare_pasted_bboxes(
        self,
        items: list[dict[str, Any]],
        pasted_masks: np.ndarray,
        target_image: np.ndarray,
        instance_ids: list[int],
    ) -> np.ndarray | None:
        """Build a preprocessed bounding-box array from all pasted object items, encoding extra
        label fields through the bbox processor.

        Label values are read from `bbox_labels` in each item — a dict mapping
        label field name to the scalar value for that object, e.g.
        `{"class_id": 3, "is_crowd": 0}`. With instance binding, internal `_ibl_bbox_*` fields
        map to user keys; `_bbox_instance_id` is taken from `instance_ids`.
        """
        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
        label_fields = (bbox_processor.params.label_fields or []) if bbox_processor else []

        all_bboxes: list[np.ndarray] = []
        all_labels: dict[str, list[Any]] = {field: [] for field in label_fields}

        for idx, item in enumerate(items):
            bbox = (
                np.asarray(item["bbox"], dtype=np.float32).ravel()
                if "bbox" in item
                else self._derive_bbox_from_mask(pasted_masks[idx], bbox_processor)
            )
            all_bboxes.append(bbox)

            item_labels: dict[str, Any] = item.get("bbox_labels", {})
            for field in label_fields:
                if field == _BBOX_INSTANCE_ID:
                    all_labels[field].append(instance_ids[idx])
                    continue
                if field.startswith("_ibl_bbox_"):
                    user_key = field.removeprefix("_ibl_bbox_")
                    if user_key not in item_labels:
                        raise ValueError(
                            f"CopyAndPaste: missing bbox label field '{user_key}' for pasted object at index {idx}. "
                            "Provide `bbox_labels` with every field declared in BboxParams.label_fields.",
                        )
                    all_labels[field].append(item_labels[user_key])
                    continue
                if field not in item_labels:
                    raise ValueError(
                        f"CopyAndPaste: missing bbox label field '{field}' for pasted object at index {idx}. "
                        "Provide `bbox_labels` with every field declared in BboxParams.label_fields.",
                    )
                all_labels[field].append(item_labels[field])

        num_bboxes = len(all_bboxes)
        for field_name, field_values in all_labels.items():
            if len(field_values) != num_bboxes:
                raise ValueError(
                    f"CopyAndPaste: label field '{field_name}' has {len(field_values)} values for "
                    f"{num_bboxes} pasted bboxes; expected one label per bbox.",
                )

        donor_item: dict[str, Any] = {
            "image": target_image,
            "bboxes": np.array(all_bboxes, dtype=np.float32),
        }
        for field in label_fields:
            donor_item[field] = all_labels[field]

        return fmixing.preprocess_copy_paste_annotations(donor_item, bbox_processor, "bboxes")

    def _collect_labels_for_one_pasted_keypoint_item(
        self,
        item_idx: int,
        item: dict[str, Any],
        kp_label_fields: Sequence[str],
        instance_ids: list[int],
        num_keypoints: int,
        all_labels: dict[str, list[Any]],
    ) -> None:
        item_labels: dict[str, Any] = item.get("keypoint_labels", {})
        for field in kp_label_fields:
            if field == _KP_INSTANCE_ID:
                all_labels[field].extend([instance_ids[item_idx]] * num_keypoints)
                continue
            if field.startswith("_ibl_kp_"):
                user_key = field.removeprefix("_ibl_kp_")
                if user_key not in item_labels:
                    raise ValueError(
                        f"CopyAndPaste: missing keypoint label field '{user_key}' for pasted object at "
                        f"index {item_idx}. Provide `keypoint_labels` with every field declared in "
                        "KeypointParams.label_fields.",
                    )
                val = item_labels[user_key]
                field_values = self._keypoint_label_values_for_item(val, num_keypoints, user_key, item_idx)
                all_labels[field].extend(field_values)
                continue
            if field not in item_labels:
                raise ValueError(
                    f"CopyAndPaste: missing keypoint label field '{field}' for pasted object at "
                    f"index {item_idx}. Provide `keypoint_labels` with every field declared in "
                    "KeypointParams.label_fields.",
                )
            val = item_labels[field]
            field_values = self._keypoint_label_values_for_item(val, num_keypoints, field, item_idx)
            all_labels[field].extend(field_values)

    def _prepare_pasted_keypoints(
        self,
        items: list[dict[str, Any]],
        target_image: np.ndarray,
        instance_ids: list[int],
    ) -> np.ndarray | None:
        """Build a preprocessed keypoints array from all pasted object items, encoding label
        fields through the keypoint processor.

        Label values are read from `keypoint_labels` in each item — a dict mapping
        label field name to scalar or list of values for that object, e.g.
        `{"joint_name": "left_eye"}` or `{"visibility": [2, 2]}`. With instance binding,
        `_kp_instance_id` is replicated per keypoint row from `instance_ids`.
        """
        keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
        kp_label_fields = (keypoint_processor.params.label_fields or []) if keypoint_processor else []

        all_kps: list[np.ndarray] = []
        all_labels: dict[str, list[Any]] = {field: [] for field in kp_label_fields}

        for item_idx, item in enumerate(items):
            if "keypoints" not in item:
                continue
            raw = np.asarray(item["keypoints"], dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[np.newaxis]
            num_keypoints = raw.shape[0]
            all_kps.append(raw)
            self._collect_labels_for_one_pasted_keypoint_item(
                item_idx,
                item,
                kp_label_fields,
                instance_ids,
                num_keypoints,
                all_labels,
            )

        if not all_kps:
            return None

        concatenated_keypoints = np.concatenate(all_kps, axis=0)
        total_keypoints = concatenated_keypoints.shape[0]

        for field in kp_label_fields:
            if len(all_labels[field]) != total_keypoints:
                raise ValueError(
                    f"CopyAndPaste: keypoint label field '{field}' has {len(all_labels[field])} values "
                    f"for {total_keypoints} concatenated keypoints.",
                )

        donor_item: dict[str, Any] = {
            "image": target_image,
            "keypoints": concatenated_keypoints,
        }
        for field in kp_label_fields:
            donor_item[field] = all_labels[field]

        return fmixing.preprocess_copy_paste_annotations(donor_item, keypoint_processor, "keypoints")

    # Sentinel for `_process_one_donor` to signal "donor lacked a usable footprint" without
    # conflating it with the silent-drop None return.
    _NO_FOOTPRINT: ClassVar[object] = object()

    def _process_one_donor(
        self,
        item: dict[str, Any],
        target_shape: tuple[int, int],
        composite_image: np.ndarray,
        semantic_canvas_holder: list[np.ndarray | None],
        bbox_processor: BboxProcessor | None,
        kp_processor: KeypointsProcessor | None,
        scale_jitter: float,
    ) -> tuple[dict[str, Any], np.ndarray] | object | None:
        """Run the per-donor pipeline (convert, crop, scale, place, stamp, back-convert) returning
        the stamped item + mask, `_NO_FOOTPRINT` sentinel, or None on drop.

        Returns the stamped item + binary mask on success, the `_NO_FOOTPRINT` sentinel when the
        donor has neither a usable mask nor a bbox (caller counts it for the UserWarning), or
        `None` when the donor is silently dropped (invalid shape, below `min_paste_area`, etc.).
        """
        if not isinstance(item, dict) or "image" not in item:
            return None

        donor_image = item["image"]
        donor_shape = (donor_image.shape[0], donor_image.shape[1])

        has_mask = item.get("mask") is not None and np.any(item["mask"] > 0)
        has_bbox = "bbox" in item and bbox_processor is not None
        if not has_mask and not has_bbox:
            return self._NO_FOOTPRINT

        bbox_px = self._donor_bbox_to_pascal_px(item, bbox_processor, donor_shape) if has_bbox else None
        kp_alb = self._donor_keypoints_to_pixels(item, kp_processor)

        bounds = self._resolve_crop_bounds(item, bbox_px, kp_alb, donor_shape)
        if bounds is None:
            return self._NO_FOOTPRINT

        cropped = self._tight_crop(item, bounds, bbox_px, kp_alb)
        crop_h, crop_w = cropped["mask"].shape[:2]
        scale = self._fit_and_jitter_scale(crop_h, crop_w, target_shape, scale_jitter)
        scaled = self._resize_cropped(cropped, scale)

        h_s, w_s = scaled["shape"]
        if h_s * w_s < self.min_paste_area:
            return None

        y0, x0 = self._sample_placement(h_s, w_s, target_shape)

        if scaled["semantic_mask"] is not None and semantic_canvas_holder[0] is None:
            semantic_canvas_holder[0] = np.zeros(target_shape, dtype=scaled["semantic_mask"].dtype)

        stamped_mask, bbox_target_px, kp_target_alb = self._stamp(
            scaled,
            y0,
            x0,
            target_shape,
            composite_image,
            semantic_canvas_holder[0],
        )
        if not np.any(stamped_mask > 0):
            return None

        stamped_item = self._build_stamped_item(
            item,
            bbox_target_px,
            kp_target_alb,
            target_shape,
            bbox_processor,
            kp_processor,
        )
        return stamped_item, stamped_mask

    def _build_stamped_item(
        self,
        item: dict[str, Any],
        bbox_target_px: np.ndarray | None,
        kp_target_alb: np.ndarray | None,
        target_shape: tuple[int, int],
        bbox_processor: BboxProcessor | None,
        kp_processor: KeypointsProcessor | None,
    ) -> dict[str, Any]:
        """Assemble the per-donor output dict, converting bbox / keypoints back to the pipeline
        coord_format on target dims while forwarding labels unchanged.
        """
        stamped: dict[str, Any] = {}
        if bbox_target_px is not None and bbox_processor is not None:
            stamped["bbox"] = self._target_pascal_px_to_coord_format(bbox_target_px, target_shape, bbox_processor)
        if "bbox_labels" in item:
            stamped["bbox_labels"] = item["bbox_labels"]
        if kp_target_alb is not None and kp_target_alb.size > 0 and kp_processor is not None:
            stamped["keypoints"] = self._target_alb_keypoints_to_coord_format(kp_target_alb, kp_processor)
        if "keypoint_labels" in item:
            stamped["keypoint_labels"] = item["keypoint_labels"]
        return stamped

    def _gather_valid_copy_paste_items(
        self,
        data: dict[str, Any],
        target_shape: tuple[int, int],
        scale_jitter: float,
    ) -> tuple[list[dict[str, Any]], list[np.ndarray], np.ndarray, np.ndarray | None] | None:
        """Iterate donors through `_process_one_donor` to collect stamped items and masks, and
        emit the no-footprint UserWarning if any donors lacked both mask and bbox.

        Returns `(stamped_items, pasted_masks_list, composite_image, semantic_canvas)` or
        `None` when no usable donor produced a stamp. `semantic_canvas` is None when no donor
        provided a `semantic_mask`.
        """
        metadata = data.get(self.metadata_key)
        if not isinstance(metadata, list) or not metadata:
            return None

        bbox_processor = cast("BboxProcessor | None", self.get_processor("bboxes"))
        kp_processor = cast("KeypointsProcessor | None", self.get_processor("keypoints"))

        composite_image = data["image"].copy()
        # Wrapped in a 1-element list so `_process_one_donor` can lazily allocate it on first use
        # without forcing a separate plumbing path.
        semantic_canvas_holder: list[np.ndarray | None] = [None]
        stamped_items: list[dict[str, Any]] = []
        pasted_masks_list: list[np.ndarray] = []
        dropped_no_footprint = 0

        for item in metadata:
            outcome = self._process_one_donor(
                item,
                target_shape,
                composite_image,
                semantic_canvas_holder,
                bbox_processor,
                kp_processor,
                scale_jitter,
            )
            if outcome is None:
                continue
            if outcome is self._NO_FOOTPRINT:
                dropped_no_footprint += 1
                continue
            stamped_item, stamped_mask = cast("tuple[dict[str, Any], np.ndarray]", outcome)
            stamped_items.append(stamped_item)
            pasted_masks_list.append(stamped_mask)

        if dropped_no_footprint > 0:
            warnings.warn(
                f"CopyAndPaste dropped {dropped_no_footprint} donor item(s) with neither a usable "
                "`mask` (non-empty) nor a usable `bbox` (requires `bbox_params` on the pipeline). "
                "Each donor must provide at least one to define the paste footprint.",
                UserWarning,
                stacklevel=3,
            )

        if not stamped_items:
            return None
        return stamped_items, pasted_masks_list, composite_image, semantic_canvas_holder[0]

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        # Sample blend_sigma + scale_jitter upfront so applied_config records them even when the
        # no-op path is taken (e.g. no valid paste items provided). scale_jitter is shared across
        # all donors in a single call so the recorded value matches what was actually applied.
        blend_sigma = self.py_random.uniform(*self.blend_sigma_range)
        self.applied_config["blend_sigma_range"] = blend_sigma
        scale_jitter = self.py_random.uniform(*self.scale_range)
        self.applied_config["scale_range"] = scale_jitter

        target_shape = params["shape"][:2]
        gathered = self._gather_valid_copy_paste_items(data, target_shape, scale_jitter)
        if gathered is None:
            return self._no_op_params()

        valid_items, pasted_masks_list, composite_image, donor_mask = gathered
        pasted_masks = np.stack(pasted_masks_list, axis=0)
        paste_union_mask = np.any(pasted_masks > 0, axis=0)

        alpha = fmixing.create_copy_paste_alpha(pasted_masks, self.blend_mode, blend_sigma)

        surviving_indices, paste_primary_instance_count = self._compute_surviving_indices(data, paste_union_mask)

        paste_surviving_ids_ordered, next_paste_instance_id = self._resolve_paste_surviving_ids(
            data,
            surviving_indices,
            paste_primary_instance_count,
        )
        paste_instance_ids = [next_paste_instance_id + k for k in range(len(valid_items))]

        pasted_bboxes = (
            self._prepare_pasted_bboxes(valid_items, pasted_masks, composite_image, paste_instance_ids)
            if "bboxes" in data
            else None
        )

        pasted_keypoints = (
            self._prepare_pasted_keypoints(valid_items, composite_image, paste_instance_ids)
            if "keypoints" in data
            else None
        )

        if donor_mask is None and "mask" in data:
            warnings.warn(
                "CopyAndPaste received a `mask` target but no donor item provided `semantic_mask`; "
                "the primary mask will be returned unchanged. Add a `semantic_mask` (H, W) entry to "
                "each donor dict to update the semantic mask under the paste footprint, or drop the "
                "`mask` target from the pipeline if this is intentional.",
                UserWarning,
                stacklevel=2,
            )

        return {
            "paste_donor_image": composite_image,
            "paste_alpha": alpha,
            "paste_instance_masks": pasted_masks,
            "paste_surviving_indices": surviving_indices,
            "paste_surviving_ids_ordered": paste_surviving_ids_ordered,
            "paste_primary_instance_count": paste_primary_instance_count,
            "paste_bboxes": pasted_bboxes,
            "paste_keypoints": pasted_keypoints,
            "paste_donor_mask": donor_mask,
            "paste_instance_ids": paste_instance_ids,
        }

    @staticmethod
    def _no_op_params() -> dict[str, Any]:
        return {
            "paste_donor_image": None,
            "paste_alpha": None,
            "paste_instance_masks": None,
            "paste_surviving_indices": None,
            "paste_surviving_ids_ordered": None,
            "paste_primary_instance_count": None,
            "paste_bboxes": None,
            "paste_keypoints": None,
            "paste_donor_mask": None,
            "paste_instance_ids": None,
        }

    def apply(
        self,
        img: ImageType,
        paste_donor_image: np.ndarray | None,
        paste_alpha: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if paste_donor_image is None or paste_alpha is None:
            return img
        return fmixing.blend_images_using_alpha(img, paste_donor_image, paste_alpha)

    def apply_to_mask(
        self,
        mask: ImageType,
        paste_alpha: np.ndarray | None,
        paste_donor_mask: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if paste_alpha is None:
            return mask
        if paste_donor_mask is not None:
            result = mask.copy()
            paste_instance_masks = params.get("paste_instance_masks")
            if paste_instance_masks is not None:
                paste_region = np.any(paste_instance_masks > 0, axis=0)
            else:
                paste_region = paste_alpha > 0
            donor = paste_donor_mask
            if result.ndim > donor.ndim:
                donor = donor[..., np.newaxis]
            if result.ndim > paste_region.ndim:
                paste_region = paste_region[..., np.newaxis]
            result[paste_region] = donor[paste_region]
            return result
        return mask

    @staticmethod
    def _normalize_existing_masks(
        masks: ImageType,
        paste_instance_masks: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Coerce existing masks and the pasted instance-masks scratch array into the canonical
        `(N, H, W, C)` 4-D shape that the rest of the paste path assumes.

        Existing masks under instance binding arrive 4-D from `Compose._add_grayscale_channels`;
        list/tuple input from non-binding callers gets stacked then expanded if needed. The pasted
        scratch array is built internally as `(N, H, W)` and always grows a trailing singleton here
        so downstream concat / `_zero_out_paste_region` can stay branchless.
        """
        if isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                return None, paste_instance_masks[..., np.newaxis]
            masks = np.stack([np.asarray(m) for m in masks], axis=0)
        if masks.ndim == 3:
            masks = masks[..., np.newaxis]
        pasted = paste_instance_masks[..., np.newaxis]
        return masks, pasted

    @staticmethod
    def _select_surviving_masks(
        masks: np.ndarray,
        paste_surviving_ids_ordered: list[int] | None,
        paste_surviving_indices: np.ndarray | None,
    ) -> np.ndarray:
        """Select the subset of mask rows that survive after the paste using either
        ID-driven or legacy positional indexing depending on whether binding is active.

        When `paste_surviving_ids_ordered` is not `None` (instance binding active) it indexes by surviving
        `_bbox_instance_id` values; otherwise the legacy positional `paste_surviving_indices` path is used.
        """
        if paste_surviving_ids_ordered is not None:
            if len(paste_surviving_ids_ordered) > 0:
                return masks[paste_surviving_ids_ordered].copy()
            return np.empty((0, *masks.shape[1:]), dtype=masks.dtype)
        if paste_surviving_indices is not None:
            return masks[paste_surviving_indices].copy()
        return masks.copy()

    @staticmethod
    def _zero_out_paste_region(surviving: np.ndarray, paste_region: np.ndarray) -> None:
        region = paste_region[np.newaxis, :, :, np.newaxis]
        np.putmask(surviving, np.broadcast_to(region, surviving.shape), 0)

    def apply_to_masks(
        self,
        masks: StackedMasks4D,
        paste_alpha: np.ndarray | None,
        paste_instance_masks: np.ndarray | None,
        paste_surviving_indices: np.ndarray | None,
        paste_surviving_ids_ordered: list[int] | None,
        **params: Any,
    ) -> StackedMasks4D:
        if paste_alpha is None or paste_instance_masks is None:
            return masks

        existing, pasted = self._normalize_existing_masks(masks, paste_instance_masks)
        if existing is None or existing.size == 0:
            return StackedMasks4D(pasted)

        # ID-driven path (instance binding active): index masks by surviving _bbox_instance_id values
        # so the output stays row-aligned with `apply_to_bboxes`. Without this, upstream bbox-only
        # filtering (e.g. Crop with min_area) leaves position != ID and mask rows attach to wrong ids.
        surviving = self._select_surviving_masks(existing, paste_surviving_ids_ordered, paste_surviving_indices)
        if surviving.size == 0:
            return StackedMasks4D(pasted)

        paste_region = np.any(paste_instance_masks > 0, axis=0)
        self._zero_out_paste_region(surviving, paste_region)
        return StackedMasks4D(np.concatenate([surviving, pasted], axis=0))

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        paste_surviving_indices: np.ndarray | None,
        paste_surviving_ids_ordered: list[int] | None,
        paste_bboxes: np.ndarray | None,
        paste_alpha: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        if paste_alpha is None:
            return bboxes

        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
        bbox_label_fields = bbox_processor.params.label_fields or []

        if paste_surviving_indices is not None and bboxes.size > 0:
            if _BBOX_INSTANCE_ID in bbox_label_fields:
                # Filter by _bbox_instance_id values against the resolved surviving ID set.
                # Mixing IDs with positions (the old `paste_surviving_indices` path) silently
                # mis-attached bboxes whenever upstream filtering broke position == ID.
                surviving_ids = paste_surviving_ids_ordered if paste_surviving_ids_ordered is not None else []
                n_lf = len(bbox_label_fields)
                id_col = bboxes.shape[1] - n_lf + bbox_label_fields.index(_BBOX_INSTANCE_ID)
                inst_col = bboxes[:, id_col].astype(np.int64, copy=False)
                keep = np.isin(inst_col, np.asarray(surviving_ids, dtype=np.int64))
                surviving_bboxes = bboxes[keep]
            else:
                surviving_bboxes = bboxes[paste_surviving_indices]
        else:
            surviving_bboxes = bboxes

        if paste_bboxes is not None and paste_bboxes.size > 0:
            combined = (
                paste_bboxes
                if surviving_bboxes.size == 0
                else np.concatenate(
                    [surviving_bboxes, paste_bboxes],
                    axis=0,
                )
            )
        else:
            combined = surviving_bboxes

        # Re-stamp `_bbox_instance_id` to dense `arange(N_out)` so the output is positionally
        # equal to its id index. Without this, the output had sparse ids (`[0,1,2,3,donor_id]`)
        # and any subsequent bbox processor drop broke the implicit `masks[id]` assumption that
        # the resync used. Phase 2b of the rewrite moves this re-stamp into the transform
        # itself so `_resync_instance_ids` becomes a pure assertion + keypoint-rebase.
        if _BBOX_INSTANCE_ID in bbox_label_fields and combined.size > 0:
            n_lf = len(bbox_label_fields)
            id_col = combined.shape[1] - n_lf + bbox_label_fields.index(_BBOX_INSTANCE_ID)
            combined[:, id_col] = np.arange(combined.shape[0], dtype=combined.dtype)
        return combined

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        paste_alpha: np.ndarray | None,
        paste_keypoints: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        if paste_alpha is None:
            return keypoints

        paste_surviving_indices = params.get("paste_surviving_indices")
        paste_surviving_ids_ordered = params.get("paste_surviving_ids_ordered")
        paste_primary_instance_count = params.get("paste_primary_instance_count")
        keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
        kp_label_fields = keypoint_processor.params.label_fields or []

        surviving_keypoints = keypoints
        if paste_surviving_indices is not None and keypoints.size > 0:
            if _KP_INSTANCE_ID in kp_label_fields:
                # Filter keypoints by _kp_instance_id against the resolved surviving ID set
                # (same set that drives bbox/mask survival), not raw mask positions.
                surviving_ids = paste_surviving_ids_ordered if paste_surviving_ids_ordered is not None else []
                n_kf = len(kp_label_fields)
                id_col = keypoints.shape[1] - n_kf + kp_label_fields.index(_KP_INSTANCE_ID)
                inst_col = keypoints[:, id_col].astype(np.int64, copy=False)
                keep = np.isin(inst_col, np.asarray(surviving_ids, dtype=np.int64))
                surviving_keypoints = keypoints[keep]
            else:
                aligned = (
                    paste_primary_instance_count is not None and keypoints.shape[0] == paste_primary_instance_count
                )
                if aligned:
                    survivor_idx = np.asarray(paste_surviving_indices)
                    if survivor_idx.size == 0:
                        surviving_keypoints = keypoints[:0]
                    elif int(survivor_idx.max()) < keypoints.shape[0] and int(survivor_idx.min()) >= 0:
                        surviving_keypoints = keypoints[survivor_idx]

        if paste_keypoints is not None and paste_keypoints.size > 0:
            combined = (
                paste_keypoints
                if surviving_keypoints.size == 0
                else np.concatenate(
                    [surviving_keypoints, paste_keypoints],
                    axis=0,
                )
            )
        else:
            combined = surviving_keypoints

        # Re-stamp `_kp_instance_id` to refer to the new dense bbox positions emitted by
        # `apply_to_bboxes` (Phase 2b). The mapping mirrors the bbox concatenation order:
        # surviving first (indexed by `paste_surviving_ids_ordered`) then pasted (indexed by
        # `paste_instance_ids`). After this, no `_resync_instance_ids` keypoint rebase is
        # needed for the CopyAndPaste boundary.
        if _KP_INSTANCE_ID in kp_label_fields and combined.size > 0:
            self._restamp_keypoint_ids(
                combined,
                kp_label_fields,
                paste_surviving_ids_ordered,
                params.get("paste_instance_ids"),
            )
        return combined

    @staticmethod
    def _restamp_keypoint_ids(
        keypoints: np.ndarray,
        kp_label_fields: Sequence[str],
        paste_surviving_ids_ordered: list[int] | None,
        paste_instance_ids: list[int] | None,
    ) -> None:
        n_kf = len(kp_label_fields)
        id_col = keypoints.shape[1] - n_kf + kp_label_fields.index(_KP_INSTANCE_ID)

        old_to_new: dict[int, int] = {}
        new_idx = 0
        if paste_surviving_ids_ordered:
            for old in paste_surviving_ids_ordered:
                old_to_new[old] = new_idx
                new_idx += 1
        if paste_instance_ids:
            for old in paste_instance_ids:
                old_to_new[old] = new_idx
                new_idx += 1

        kp_old = keypoints[:, id_col].astype(np.int64, copy=False)
        keypoints[:, id_col] = np.array(
            [old_to_new.get(int(k), int(k)) for k in kp_old],
            dtype=keypoints.dtype,
        )


__all__ = [
    "CopyAndPaste",
]
