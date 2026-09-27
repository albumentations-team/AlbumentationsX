"""Transforms for temporal image sequences."""

from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
import torch
from pydantic import Field

from albumentations.augmentations.other import temporal_functional as ftemporal
from albumentations.core.binding import TargetResult
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetSet
from albumentations.core.transforms_interface import BaseTransformInitSchema, BasicTransform
from albumentations.core.type_definitions import ImageType, StackedMasks4D, Targets

__all__ = ["UniformTemporalSubsample"]


class UniformTemporalSubsample(BasicTransform):
    """Select uniformly spaced frames to give videos a fixed sequence length for training or inference,
    repeating frames from shorter clips.

    Args:
        num_frames (int): Number of output frames. Must be positive. Requests longer than the input
            repeat frames. A request for one frame selects the first frame.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        images, masks, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The `images` target is the temporal sequence. Supply NumPy videos as (T, H, W, C)
          or (T, H, W), and CPU Tensors as (T, C, H, W).
        - With at least two output frames, both endpoints are included. Intermediate evenly spaced
          positions are rounded down to frame indices.
        - Use Compose(frame_binding=["images", "masks"]) for one semantic mask per frame, or
          frame_binding=["images", "frame_annotations"] for per-frame dictionaries containing
          masks, bboxes, keypoints, or instances. Bound annotations follow the selected frames.

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> video = np.zeros((24, 64, 96, 3), dtype=np.uint8)
        >>> masks = np.zeros((24, 64, 96), dtype=np.uint8)
        >>> transform = A.Compose(
        ...     [A.UniformTemporalSubsample(num_frames=8)],
        ...     frame_binding=["images", "masks"],
        ... )
        >>> result = transform(images=video, masks=masks)
        >>> result["images"].shape, result["masks"].shape
        ((8, 64, 96, 3), (8, 64, 96))

        Preprocess a video for classification with temporal selection, spatial augmentation,
        and normalization:

        >>> preprocess = A.Compose(
        ...     [
        ...         A.UniformTemporalSubsample(num_frames=8),
        ...         A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        ...         A.HorizontalFlip(p=0.5),
        ...         A.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        ...     ],
        ...     seed=137,
        ... )
        >>> preprocess(images=video)["images"].shape
        (8, 224, 224, 3)

    """

    _targets = (Targets.IMAGES, Targets.MASKS, Targets.BBOXES, Targets.KEYPOINTS)
    _runtime_generated_params = frozenset(
        {"frame_ids", "instance_ids", "instance_frame_ids", "frame_id_column", "source_frame_ids"}
    )

    class InitSchema(BaseTransformInitSchema):
        num_frames: int = Field(gt=0)

    def __init__(self, num_frames: int, p: float = 1.0):
        super().__init__(p=p)
        self.num_frames = num_frames

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        last_index = len(targets.primary_image_like().value) - 1
        if last_index < 0:
            raise ValueError("Cannot sample frame_indices from an empty video")
        denominator = max(self.num_frames - 1, 1)
        return SampledParams(
            params={"frame_indices": tuple(index * last_index // denominator for index in range(self.num_frames))},
        )

    def apply_to_images(
        self,
        images: Annotated[ImageType, torch.Tensor],
        frame_indices: Sequence[int],
        source_frame_ids: np.ndarray | None,
        **params: Any,
    ) -> ImageType | TargetResult[ImageType]:
        values = ftemporal.uniform_temporal_subsample(images, frame_indices)
        return (
            values
            if source_frame_ids is None
            else TargetResult(values, source_frame_ids=source_frame_ids[list(frame_indices)])
        )

    def apply_to_masks(
        self,
        masks: Annotated[StackedMasks4D, torch.Tensor],
        frame_indices: Sequence[int],
        frame_ids: np.ndarray | None,
        instance_ids: np.ndarray | None,
        instance_frame_ids: np.ndarray | None,
        **params: Any,
    ) -> TargetResult[StackedMasks4D]:
        if frame_ids is None:
            raise ValueError("Mask frame selection requires Compose(frame_binding=...)")
        rows, frames, instances = ftemporal.select_annotation_rows(
            frame_ids,
            frame_indices,
            instance_ids,
            instance_frame_ids,
        )
        return TargetResult(ftemporal.uniform_temporal_subsample(masks, rows), frames, instances)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        frame_indices: Sequence[int],
        frame_id_column: int | None,
        instance_frame_ids: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        return self._select_coordinates(
            bboxes,
            frame_indices,
            frame_id_column,
            instance_frame_ids,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        frame_indices: Sequence[int],
        frame_id_column: int | None,
        instance_frame_ids: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        return self._select_coordinates(
            keypoints,
            frame_indices,
            frame_id_column,
            instance_frame_ids,
        )

    @staticmethod
    def _select_coordinates(
        values: np.ndarray,
        frame_indices: Sequence[int],
        frame_id_column: int | None,
        instance_frame_ids: np.ndarray | None,
    ) -> np.ndarray:
        if frame_id_column is None:
            raise ValueError("Annotation frame selection requires Compose(frame_binding=...)")
        ids = values[:, -1].astype(np.intp) if instance_frame_ids is not None else None
        rows, frames, instances = ftemporal.select_annotation_rows(
            values[:, frame_id_column],
            frame_indices,
            ids,
            instance_frame_ids,
        )
        result = values[rows]
        result[:, frame_id_column] = frames
        if instances is not None:
            result[:, -1] = instances
        return result
