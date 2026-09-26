"""Transforms for temporal image sequences."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from pydantic import Field

from albumentations.augmentations.other import temporal_functional as ftemporal
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetParams, TargetSet, requirements_for_views
from albumentations.core.transforms_interface import BaseTransformInitSchema, BasicTransform
from albumentations.core.type_definitions import ImageType, Targets

__all__ = ["UniformTemporalSubsample"]


class UniformTemporalSubsample(BasicTransform):
    """Select a fixed number of uniformly spaced frames from a video sequence, preserving endpoints and repeating
    nearest frames when upsampling.

    Both the first and last frames are selected when `num_frames` is at least two and
    the input contains more than one frame. When `num_frames` exceeds the input
    length, indices are repeated to produce exactly the requested number of frames.

    Args:
        num_frames (int): Number of frames in the output sequence. Must be positive.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Input layouts:
        - NumPy: (T, H, W, C)
        - PyTorch: (T, C, H, W)

    Image types:
        uint8, float32

    Note:
        Supply the video sequence through the `images` target. The singular `image`
        target is not supported.

        Additional targets mapped to `images` receive the same frame indices and
        must have the same number of input frames. Spatial masks, bounding boxes,
        keypoints, volumes, and single images are rejected because their temporal
        alignment is ambiguous.

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> import torch
        >>> video = np.random.randint(0, 256, (24, 256, 320, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.UniformTemporalSubsample(num_frames=8),
        ...     A.RandomResizedCrop(size=(224, 224)),
        ...     A.HorizontalFlip(p=0.5),
        ...     A.Normalize(),
        ... ])
        >>> sampled = transform(images=video)["images"]
        >>> model_input = torch.from_numpy(sampled).permute(0, 3, 1, 2)
        >>> model_input.shape
        torch.Size([8, 3, 224, 224])

    """

    _targets = Targets.IMAGE

    class InitSchema(BaseTransformInitSchema):
        num_frames: int = Field(gt=0)

    def __init__(self, num_frames: int, p: float = 1.0):
        super().__init__(p=p)
        self.num_frames = num_frames

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "user_data": self.apply_to_user_data,
        }

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        del params, data, sampling
        image_views = targets.by_canonical_type("images")
        if not image_views:
            raise ValueError("UniformTemporalSubsample supports only `images` targets")

        frame_counts = {int(view.value.shape[0]) for view in image_views}
        if 0 in frame_counts:
            raise ValueError("UniformTemporalSubsample requires at least one frame")
        if len(frame_counts) != 1:
            raise ValueError("All `images` targets must have the same number of frames")

        input_frames = frame_counts.pop()
        frame_indices = tuple(np.linspace(0, input_frames - 1, self.num_frames, dtype=np.int64).tolist())
        return SampledParams(
            params={},
            target_params=(
                TargetParams(
                    targets=tuple(view.name for view in image_views),
                    params={"frame_indices": frame_indices},
                    requirements=requirements_for_views(image_views, shape=True, sampling_topology=True),
                ),
            ),
        )

    def apply(self, img: ImageType | torch.Tensor, **params: Any) -> ImageType | torch.Tensor:
        raise ValueError("UniformTemporalSubsample supports only `images` targets")

    def apply_to_images(
        self,
        images: ImageType | torch.Tensor,
        frame_indices: tuple[int, ...],
        **params: Any,
    ) -> ImageType | torch.Tensor:
        return ftemporal.uniform_temporal_subsample(images, frame_indices)

    def apply_with_params(
        self,
        sampled_params: SampledParams,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Reject unsupported spatial targets before dispatching the sampled indices."""
        unsupported_targets = {
            key
            for key, value in kwargs.items()
            if value is not None
            and self._additional_targets.get(key, key)
            in {"image", "mask", "masks", "bboxes", "keypoints", "volume", "mask3d"}
        }
        if unsupported_targets:
            names = ", ".join(sorted(unsupported_targets))
            raise ValueError(
                f"UniformTemporalSubsample supports only `images` targets; received unsupported target(s): {names}",
            )
        return super().apply_with_params(sampled_params, *args, **kwargs)
