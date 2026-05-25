"""Mosaic mixing transform."""

from typing import Annotated, Any, Literal, cast

from typing_extensions import Self

from ._transforms_shared import (
    _BBOX_INSTANCE_ID,
    _KP_INSTANCE_ID,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    AfterValidator,
    BaseTransformInitSchema,
    BboxProcessor,
    DualTransform,
    FullInterpolationType,
    ImageType,
    KeypointsProcessor,
    StackedMasks4D,
    Targets,
    check_range_bounds,
    deepcopy,
    filter_bboxes_with_mask,
    fmixing,
    model_validator,
    nondecreasing,
    np,
)


def _as_full_interpolation_type(interpolation: int) -> FullInterpolationType:
    return cast("FullInterpolationType", interpolation)


class Mosaic(DualTransform):
    """Combine multiple images and annotations into one image via a mosaic grid. Uses metadata
    for additional images; common in object detection training.

    Mosaic creates a grid of images by placing the primary image and additional images from metadata
    into cells of a larger canvas, then crops a region to produce the final output. This is commonly
    used in object detection training to increase data diversity and help models learn to detect
    objects at different scales and contexts.

    The transform takes a primary input image (and its annotations) and combines it with
    additional images/annotations provided via metadata. It calculates the geometry for
    a mosaic grid, selects additional items, preprocesses annotations consistently
    (handling label encoding updates), applies geometric transformations, and assembles
    the final output.

    Args:
        grid_yx (tuple[int, int]): The number of rows (y) and columns (x) in the mosaic grid.
            Determines the maximum number of images involved (grid_yx[0] * grid_yx[1]).
            Default: (2, 2).
        target_size (tuple[int, int]): The desired output (height, width) for the final mosaic image.
            after cropping the mosaic grid.
        cell_shape (tuple[int, int]): cell shape of each cell in the mosaic grid.
        fit_mode (Literal['cover', 'contain']): How to fit images into mosaic cells.
            - "cover": Scale image to fill the entire cell, potentially cropping parts.
            - "contain": Scale image to fit entirely within the cell, potentially adding padding.
            Default: "cover".
        metadata_key (str): Key in the input dictionary specifying the list of additional data dictionaries
            for the mosaic. Each dictionary in the list should represent one potential additional item.
            Expected keys: 'image' (required, np.ndarray), and optionally 'mask' (np.ndarray),
            'masks' (np.ndarray, stacked instance masks), 'bboxes' (np.ndarray), 'keypoints' (np.ndarray),
            and label fields supplied via the `bbox_labels` and `keypoint_labels` wrapper dicts
            (see Metadata Format below). Default: "mosaic_metadata".
        center_range (tuple[float, float]): Range [0.0-1.0] to sample the center point of the mosaic view
            relative to the valid central region of the conceptual large grid. This affects which parts
            of the assembled grid are visible in the final crop. Default: (0.3, 0.7).
        interpolation (int): OpenCV interpolation flag used for resizing images during geometric processing.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (int): OpenCV interpolation flag used for resizing masks during geometric processing.
            Default: cv2.INTER_NEAREST.
        fill (tuple[float, ...] | float): Value used for padding images if needed during geometric processing.
            Default: 0.
        fill_mask (tuple[float, ...] | float): Value used for padding masks if needed during geometric processing.
            Default: 0.
        p (float): Probability of applying the transform. Default: 0.5.

    Workflow (`get_params_dependent_on_data`):
        1. Calculate Geometry & Visible Cells: Determine which grid cells are visible in the final
           `target_size` crop and their placement coordinates on the output canvas.
        2. Validate Raw Additional Metadata: Filter the list provided via `metadata_key`,
           keeping only valid items (dicts with an 'image' key).
        3. Select Subset of Raw Additional Metadata: Choose a subset of the valid raw items based
           on the number of visible cells requiring additional data.
        4. Preprocess Selected Raw Additional Items: Preprocess bboxes/keypoints for the *selected*
           additional items *only*. This uses shared processors from `Compose`, updating their
           internal state (e.g., `LabelEncoder`) based on labels in these selected items.
        5. Prepare Primary Data: Extract preprocessed primary data fields from the input `data` dictionary
            into a `primary` dictionary.
        6. Determine & Perform Replication: If fewer additional items were selected than needed,
           replicate the preprocessed primary data as required.
        7. Combine Final Items: Create the list of all preprocessed items (primary, selected additional,
           replicated primary) that will be used.
        8. Assign Items to VISIBLE Grid Cells
        9. Process Geometry & Shift Coordinates: For each assigned item:
            a. Apply geometric transforms to image/mask based on `fit_mode`:
               - "cover": Resize to smallest dimension covering the cell, then crop to cell size
               - "contain": Resize to largest dimension fitting in the cell, then pad to cell size
            b. Apply geometric shift to the *preprocessed* bboxes/keypoints based on cell placement.
       10. Return Parameters: Return the processed cell data (image, mask, shifted bboxes, shifted kps)
           keyed by placement coordinates.

    Label Handling:
        - The transform relies on `bbox_processor` and `keypoint_processor` provided by `Compose`.
        - `Compose.preprocess` initially fits the processors' `LabelEncoder` on the primary data.
        - This transform (`Mosaic`) preprocesses the *selected* additional raw items using the same
          processors. If new labels are found, the shared `LabelEncoder` state is updated via its
          `update` method.
        - `Compose.postprocess` uses the final updated encoder state to decode all labels present
          in the mosaic output for the current `Compose` call.
        - The encoder state is transient per `Compose` call.

    Note:
        If fewer additional images are provided than needed to fill the grid, the primary image
        will be replicated to fill the remaining cells. For example, with a 2x2 grid, if only
        one additional image is provided, the mosaic will contain the primary image in two cells
        and the additional image in one cell, with one visible cell selected from these three.
        Stacked instance masks on the `masks` key (N, H, W) are transformed via `apply_to_masks` like
        other DualTransforms; `_targets` only lists `Targets` enum values (no `Targets.MASKS`).

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Reference:
        YOLOv4: Optimal Speed and Accuracy of Object Detection: https://arxiv.org/pdf/2004.10934

    Metadata Format:
        Each dict in the metadata list represents one additional image and must contain:
            - image (np.ndarray): Additional image. Required.
            - mask (np.ndarray): Semantic mask for the additional image. Optional.
            - masks (np.ndarray): Stacked instance masks (N, H, W) for the additional image.
              Optional; same geometry as image. Use with instance_binding / pipeline masks target.
            - bboxes (np.ndarray): Bounding boxes in the **same coordinate format** as
              `BboxParams.coord_format` declared in `Compose`. Optional.
            - keypoints (np.ndarray): Keypoints in the **same format** as
              `KeypointParams.coord_format` declared in `Compose`. Optional.
            - bbox_labels (dict[str, Any]): Label lists for bboxes, keyed by label field name
              as declared in `BboxParams.label_fields`. Each value must be a list with one
              entry per bbox. E.g. `{"class_id": [3, 7], "is_crowd": [0, 1]}`.
            - keypoint_labels (dict[str, Any]): Label lists for keypoints, keyed by label
              field name as declared in `KeypointParams.label_fields`. Each value must be a
              list with one entry per keypoint. E.g. `{"joint_name": ["left_eye", "nose"]}`.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare primary data
        >>> primary_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> primary_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> primary_bboxes = np.array([[10, 10, 40, 40], [50, 50, 90, 90]], dtype=np.float32)
        >>> primary_bbox_classes = [1, 2]
        >>> primary_keypoints = np.array([[25, 25], [75, 75]], dtype=np.float32)
        >>> primary_keypoint_classes = ['eye', 'nose']
        >>>
        >>> # Prepare additional images for mosaic.
        >>> # bbox_labels and keypoint_labels are dicts mapping field name -> list of values.
        >>> mosaic_metadata = [
        ...     {
        ...         'image': np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ...         'mask': np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        ...         'bboxes': np.array([[20, 20, 60, 60]], dtype=np.float32),
        ...         'bbox_labels': {'bbox_classes': [3]},
        ...         'keypoints': np.array([[40, 40]], dtype=np.float32),
        ...         'keypoint_labels': {'keypoint_classes': ['mouth']},
        ...     },
        ...     {
        ...         'image': np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ...         'mask': np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        ...         'bboxes': np.array([[30, 30, 70, 70]], dtype=np.float32),
        ...         'bbox_labels': {'bbox_classes': [4]},
        ...         'keypoints': np.array([[50, 50], [65, 65]], dtype=np.float32),
        ...         'keypoint_labels': {'keypoint_classes': ['eye', 'eye']},
        ...     },
        ... ]
        >>>
        >>> transform = A.Compose([
        ...     A.Mosaic(
        ...         grid_yx=(2, 2),
        ...         target_size=(200, 200),
        ...         cell_shape=(120, 120),
        ...         center_range=(0.4, 0.6),
        ...         fit_mode="cover",
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_classes']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_classes']))
        >>>
        >>> transformed = transform(
        ...     image=primary_image,
        ...     mask=primary_mask,
        ...     bboxes=primary_bboxes,
        ...     bbox_classes=primary_bbox_classes,
        ...     keypoints=primary_keypoints,
        ...     keypoint_classes=primary_keypoint_classes,
        ...     mosaic_metadata=mosaic_metadata,
        ... )
        >>>
        >>> mosaic_image = transformed['image']
        >>> mosaic_bboxes = transformed['bboxes']
        >>> mosaic_bbox_classes = transformed['bbox_classes']
        >>> mosaic_keypoint_classes = transformed['keypoint_classes']

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        grid_yx: tuple[int, int]
        target_size: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
        ]
        cell_shape: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
        ]
        metadata_key: str
        center_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        fit_mode: Literal["cover", "contain"]

        @model_validator(mode="after")
        def _check_cell_shape(self) -> Self:
            if (
                self.cell_shape[0] * self.grid_yx[0] < self.target_size[0]
                or self.cell_shape[1] * self.grid_yx[1] < self.target_size[1]
            ):
                raise ValueError("Target size should be smaller than cell_shape * grid_yx")
            return self

    def __init__(
        self,
        grid_yx: tuple[int, int] = (2, 2),
        target_size: tuple[int, int] = (512, 512),
        cell_shape: tuple[int, int] = (512, 512),
        center_range: tuple[float, float] = (0.3, 0.7),
        fit_mode: Literal["cover", "contain"] = "cover",
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        metadata_key: str = "mosaic_metadata",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.grid_yx = grid_yx
        self.target_size = target_size

        self.metadata_key = metadata_key
        self.center_range = center_range
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.fill = fill
        self.fill_mask = fill_mask
        self.fit_mode = fit_mode
        self.cell_shape = cell_shape

    @property
    def targets_as_params(self) -> list[str]:
        """Return list of target keys passed as params (e.g. to get_params_dependent_on_data).
        For Mosaic/FMix: metadata key for preprocessed mosaic/mix.

        Returns:
            list[str]: List containing the metadata key name

        """
        return [self.metadata_key]

    def _calculate_geometry(self, data: dict[str, Any]) -> list[tuple[int, int, int, int]]:
        # Step 1: Calculate Geometry & Cell Placements
        center_xy = fmixing.calculate_mosaic_center_point(
            grid_yx=self.grid_yx,
            cell_shape=self.cell_shape,
            target_size=self.target_size,
            center_range=self.center_range,
            py_random=self.py_random,
        )

        self.applied_config = {
            "center_range": center_xy,
        }

        return fmixing.calculate_cell_placements(
            grid_yx=self.grid_yx,
            cell_shape=self.cell_shape,
            target_size=self.target_size,
            center_xy=center_xy,
        )

    def _select_additional_items(self, data: dict[str, Any], num_additional_needed: int) -> list[dict[str, Any]]:
        valid_items = fmixing.filter_valid_metadata(data.get(self.metadata_key), self.metadata_key, data)
        if len(valid_items) > num_additional_needed:
            return self.py_random.sample(valid_items, num_additional_needed)
        return valid_items

    def _preprocess_additional_items(
        self,
        additional_items: list[dict[str, Any]],
        data: dict[str, Any],
    ) -> list[fmixing.ProcessedMosaicItem]:
        if "bboxes" in data or "keypoints" in data:
            bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
            keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
            return fmixing.preprocess_selected_mosaic_items(additional_items, bbox_processor, keypoint_processor)
        if "masks" in data:
            out: list[fmixing.ProcessedMosaicItem] = []
            for item in additional_items:
                if not isinstance(item, dict) or "image" not in item:
                    continue
                flat_item = fmixing.unpack_label_wrappers(item)
                entry: fmixing.ProcessedMosaicItem = {"image": item["image"]}
                if flat_item.get("mask") is not None:
                    entry["mask"] = flat_item["mask"]
                if flat_item.get("masks") is not None:
                    entry["masks"] = np.copy(np.asarray(flat_item["masks"]))
                out.append(entry)
            return out
        return cast("list[fmixing.ProcessedMosaicItem]", list(additional_items))

    def _prepare_final_items(
        self,
        primary: fmixing.ProcessedMosaicItem,
        additional_items: list[fmixing.ProcessedMosaicItem],
        num_needed: int,
    ) -> list[fmixing.ProcessedMosaicItem]:
        num_replications = max(0, num_needed - len(additional_items))
        replicated = [deepcopy(primary) for _ in range(num_replications)]
        return [primary, *additional_items, *replicated]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        cell_placements = self._calculate_geometry(data)

        num_cells = len(cell_placements)
        num_additional_needed = max(0, num_cells - 1)

        additional_items = self._select_additional_items(data, num_additional_needed)

        preprocessed_additional = self._preprocess_additional_items(additional_items, data)

        primary = self.get_primary_data(data)
        final_items = self._prepare_final_items(primary, preprocessed_additional, num_additional_needed)

        placement_to_item_index = fmixing.assign_items_to_grid_cells(
            num_items=len(final_items),
            cell_placements=cell_placements,
            py_random=self.py_random,
        )

        processed_cells = fmixing.process_all_mosaic_geometries(
            canvas_shape=self.target_size,
            cell_shape=self.cell_shape,
            placement_to_item_index=placement_to_item_index,
            final_items_for_grid=final_items,
            fill=self.fill,
            fill_mask=self.fill_mask if self.fill_mask is not None else self.fill,
            fit_mode=self.fit_mode,
            interpolation=_as_full_interpolation_type(self.interpolation),
            mask_interpolation=_as_full_interpolation_type(self.mask_interpolation),
        )

        if "bboxes" in data or "keypoints" in data or "masks" in data:
            processed_cells = fmixing.shift_all_coordinates(processed_cells, canvas_shape=self.target_size)
            bbox_proc = self.get_processor("bboxes")
            kp_proc = self.get_processor("keypoints")
            # Per-cell mask realignment MUST run BEFORE `remap_mosaic_instance_label_ids`:
            # at this point each cell's surviving bbox `_bbox_instance_id` column still holds
            # the LOCAL position of the input mask stack (id k == row k in `cell_data["masks"]`).
            # Once remap rewrites those to globally-unique ids, the local-position handle is gone.
            # Without this pass, the cell-level Crop drops some bboxes but leaves all input mask
            # rows intact, breaking positional alignment the moment we concatenate cells.
            if "masks" in data and isinstance(bbox_proc, BboxProcessor):
                processed_cells = self._filter_cell_masks_to_surviving_bboxes(processed_cells, bbox_proc)
            processed_cells = fmixing.remap_mosaic_instance_label_ids(
                processed_cells,
                bbox_proc if isinstance(bbox_proc, BboxProcessor) else None,
                kp_proc if isinstance(kp_proc, KeypointsProcessor) else None,
            )

        result: dict[str, Any] = {
            "processed_cells": processed_cells,
            "target_shape": self._get_target_shape(data["image"].shape),
        }
        if "mask" in data:
            result["target_mask_shape"] = self._get_target_shape(data["mask"].shape)
        if "masks" in data:
            ms = data["masks"].shape
            # Stacked instance masks are (N, H, W); do not treat N as spatial dim.
            if len(ms) >= 3:
                result["target_masks_shape"] = (int(ms[0]), self.target_size[0], self.target_size[1])
            else:
                result["target_masks_shape"] = tuple(self._get_target_shape(ms))

        # Compute the survival decision ONCE here so apply_to_{bboxes,masks,keypoints} share it.
        # Without this, bboxes were filtered inside apply_to_bboxes and masks/keypoints had no
        # way to mirror the survival, breaking positional alignment on the way to the next
        # transform (the root cause of the Mosaic+Perspective+CopyAndPaste IndexError).
        result.update(self._compute_mosaic_survival(processed_cells, data))
        return result

    @staticmethod
    def _filter_cell_masks_to_surviving_bboxes(
        processed_cells: dict[tuple[int, int, int, int], fmixing.ProcessedMosaicItem],
        bbox_proc: BboxProcessor,
    ) -> dict[tuple[int, int, int, int], fmixing.ProcessedMosaicItem]:
        """Per-cell, slice `cell_data["masks"]` down to rows whose corresponding bbox survived
        the cell-level crop in `process_cell_geometry`.

        The cell pipeline (PadIfNeeded + Crop) drops bboxes but leaves the input mask stack
        untouched, so per-cell `len(masks) > len(bboxes)` is the rule, not the exception.
        Surviving bboxes still carry their LOCAL `_bbox_instance_id` (== input mask row
        position) in their last label column at this point — `remap_mosaic_instance_label_ids`
        hasn't run yet. We use that handle to fancy-index the per-cell mask stack so cells
        come out of this pass with `len(masks) == len(bboxes)`, which is what the global
        concat in `_compute_mosaic_survival` and `assemble_mosaic_instance_masks_stack` rely
        on for keep-mask sharing.

        Cells without bboxes (e.g. metadata items that brought masks-only) keep their masks
        unchanged because there is no bbox-driven survival decision to mirror.
        """
        bbox_fields = bbox_proc.params.label_fields or []
        if _BBOX_INSTANCE_ID not in bbox_fields:
            return processed_cells
        n_bf = len(bbox_fields)
        id_offset_from_end = n_bf - bbox_fields.index(_BBOX_INSTANCE_ID)
        out: dict[tuple[int, int, int, int], fmixing.ProcessedMosaicItem] = {}
        for placement, cell in processed_cells.items():
            masks = cell.get("masks")
            if masks is None or not isinstance(masks, np.ndarray) or masks.size == 0:
                out[placement] = cell
                continue
            bboxes = cell.get("bboxes")
            new_cell = cast("fmixing.ProcessedMosaicItem", dict(cell))
            if bboxes is None or not isinstance(bboxes, np.ndarray) or bboxes.size == 0:
                # No surviving bboxes in this cell → drop ALL its mask rows so the cell does
                # not contribute orphan mask layers to the global stack. Keeping them would
                # restore the per-cell desync this pass exists to eliminate.
                new_cell["masks"] = masks[:0]
            else:
                id_col = bboxes.shape[1] - id_offset_from_end
                local_ids = bboxes[:, id_col].astype(np.int64, copy=False)
                valid = local_ids[(local_ids >= 0) & (local_ids < masks.shape[0])]
                new_cell["masks"] = masks[valid] if valid.shape[0] > 0 else masks[:0]
            out[placement] = new_cell
        return out

    def _compute_mosaic_survival(
        self,
        processed_cells: dict[tuple[int, int, int, int], fmixing.ProcessedMosaicItem],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Concatenate per-cell bboxes once, run `filter_bboxes_with_mask`, and stash the result
        so all three apply methods share one survival decision per call.

        `keep_mask` is positional over `combined_bboxes`. `surviving_instance_ids` is the
        `_INSTANCE_ID` set extracted from `combined_bboxes[keep_mask]` when instance binding
        is active, otherwise `None` (legacy in-bounds-only filtering kicks in).
        """
        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
        if bbox_processor is None:
            return {"mosaic_survival": None}

        all_shifted: list[np.ndarray] = []
        for cell_data in processed_cells.values():
            shifted = cell_data.get("bboxes")
            if shifted is not None and np.asarray(shifted).size > 0:
                all_shifted.append(shifted)

        if not all_shifted:
            return {"mosaic_survival": None}

        combined_bboxes = np.concatenate(all_shifted, axis=0)
        filtered_bboxes, keep_mask = filter_bboxes_with_mask(
            combined_bboxes,
            self.target_size,
            bbox_processor.params.bbox_type,
            min_area=bbox_processor.params.min_area,
            min_visibility=bbox_processor.params.min_visibility,
            min_width=bbox_processor.params.min_width,
            min_height=bbox_processor.params.min_height,
            max_accept_ratio=bbox_processor.params.max_accept_ratio,
            clip_after_transform=bbox_processor.params.clip_after_transform,
        )

        label_fields = bbox_processor.params.label_fields or []
        surviving_instance_ids: set[int] | None = None
        if _BBOX_INSTANCE_ID in label_fields and combined_bboxes.size > 0:
            n_lf = len(label_fields)
            id_col = combined_bboxes.shape[1] - n_lf + label_fields.index(_BBOX_INSTANCE_ID)
            surviving_instance_ids = set(combined_bboxes[keep_mask, id_col].astype(np.int64).tolist())

        return {
            "mosaic_survival": {
                "combined_bboxes": combined_bboxes,
                "filtered_bboxes": filtered_bboxes,
                "keep_mask": keep_mask,
                "surviving_instance_ids": surviving_instance_ids,
            },
        }

    @staticmethod
    def get_primary_data(data: dict[str, Any]) -> fmixing.ProcessedMosaicItem:
        """Return a copy of the primary item from data so the original is not mutated. Call from
        Mosaic/FMix to build composed image from primary plus patches.

        Args:
            data (dict[str, Any]): Dictionary containing the primary data.

        Returns:
            fmixing.ProcessedMosaicItem: A copy of the primary data.

        """
        mask = data.get("mask")
        if mask is not None:
            mask = mask.copy()
        bboxes = data.get("bboxes")
        if bboxes is not None:
            bboxes = bboxes.copy()
        keypoints = data.get("keypoints")
        if keypoints is not None:
            keypoints = keypoints.copy()
        masks = data.get("masks")
        if masks is not None:
            masks = masks.copy()
        primary: fmixing.ProcessedMosaicItem = {
            "image": data["image"],
            "mask": mask,
            "bboxes": bboxes,
            "keypoints": keypoints,
        }
        if masks is not None:
            primary["masks"] = masks
        return primary

    def _get_target_shape(self, np_shape: tuple[int, ...]) -> list[int]:
        target_shape = list(np_shape)
        target_shape[0] = self.target_size[0]
        target_shape[1] = self.target_size[1]
        return target_shape

    def apply(
        self,
        img: ImageType,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_shape: tuple[int, int],
        **params: Any,
    ) -> ImageType:
        return fmixing.assemble_mosaic_from_processed_cells(
            processed_cells=processed_cells,
            target_shape=target_shape,
            dtype=img.dtype,
            data_key="image",
            fill=self.fill,
        )

    def apply_to_mask(
        self,
        mask: ImageType,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_mask_shape: tuple[int, int],
        **params: Any,
    ) -> ImageType:
        return fmixing.assemble_mosaic_from_processed_cells(
            processed_cells=processed_cells,
            target_shape=target_mask_shape,
            dtype=mask.dtype,
            data_key="mask",
            fill=self.fill_mask,
        )

    def apply_to_masks(
        self,
        masks: StackedMasks4D,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_masks_shape: tuple[int, ...],
        mosaic_survival: dict[str, Any] | None,
        **params: Any,
    ) -> StackedMasks4D:
        canvas_hw = (target_masks_shape[1], target_masks_shape[2])
        assembled = fmixing.assemble_mosaic_instance_masks_stack(
            processed_cells=processed_cells,
            canvas_hw=canvas_hw,
            dtype=masks.dtype,
            fill=self.fill_mask,
        )
        # Mirror the bbox survival decision onto masks so `len(masks) == len(bboxes)` going
        # into the next transform. Only kicks in under instance binding; without binding the
        # legacy "one mask per per-cell input mask" semantics are preserved.
        if mosaic_survival is not None and mosaic_survival.get("surviving_instance_ids") is not None:
            keep_mask = mosaic_survival["keep_mask"]
            if assembled.shape[0] == keep_mask.shape[0]:
                assembled = assembled[keep_mask]
        return StackedMasks4D(assembled)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,  # Original bboxes - ignored
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        mosaic_survival: dict[str, Any] | None,
        **params: Any,
    ) -> np.ndarray:
        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))

        if mosaic_survival is not None:
            return mosaic_survival["filtered_bboxes"]

        # Empty / no-bbox-processor fallback (mosaic_survival is None when there were no
        # input bboxes anywhere in the grid).
        if bbox_processor is None or bbox_processor.params.bbox_type == "obb":
            num_cols = max(bboxes.shape[1] if bboxes.ndim > 1 else 5, 5)
        else:
            num_cols = max(bboxes.shape[1] if bboxes.ndim > 1 else 4, 4)
        return np.empty((0, num_cols), dtype=bboxes.dtype)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,  # Original keypoints - ignored
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        mosaic_survival: dict[str, Any] | None,
        **params: Any,
    ) -> np.ndarray:
        all_shifted_keypoints = []

        for cell_data in processed_cells.values():
            shifted_keypoints = cell_data.get("keypoints")
            if shifted_keypoints is not None and np.asarray(shifted_keypoints).size > 0:
                all_shifted_keypoints.append(shifted_keypoints)

        if not all_shifted_keypoints:
            return np.empty((0, keypoints.shape[1]), dtype=keypoints.dtype)

        combined_keypoints = np.concatenate(all_shifted_keypoints, axis=0)

        keypoint_processor = self.get_processor("keypoints")
        kp_fields = (
            keypoint_processor.params.label_fields
            if isinstance(keypoint_processor, KeypointsProcessor) and keypoint_processor.params.label_fields
            else []
        )

        if _KP_INSTANCE_ID in kp_fields:
            # Under binding: drop keypoints whose instance was filtered from bboxes so the
            # post-transform `_resync_instance_ids` invariant (no orphan keypoints) holds.
            if mosaic_survival is not None and mosaic_survival.get("surviving_instance_ids") is not None:
                surviving_ids = mosaic_survival["surviving_instance_ids"]
                n_kf = len(kp_fields)
                id_col = combined_keypoints.shape[1] - n_kf + kp_fields.index(_KP_INSTANCE_ID)
                kp_inst = combined_keypoints[:, id_col].astype(np.int64, copy=False)
                in_surviving = np.fromiter((int(k) in surviving_ids for k in kp_inst), dtype=bool, count=kp_inst.size)
                return combined_keypoints[in_surviving]
            return combined_keypoints

        target_h, target_w = self.target_size
        valid_indices = (
            (combined_keypoints[:, 0] >= 0)
            & (combined_keypoints[:, 0] < target_w)
            & (combined_keypoints[:, 1] >= 0)
            & (combined_keypoints[:, 1] < target_h)
        )

        return combined_keypoints[valid_indices]


__all__ = [
    "Mosaic",
]
