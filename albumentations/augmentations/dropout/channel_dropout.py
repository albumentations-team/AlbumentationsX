"""Drop a fraction of eligible image channels."""

from decimal import Decimal
from typing import Annotated, Any

from pydantic import Field, field_validator

from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import (
    SampledParams,
    TargetParams,
    TargetSet,
    requirements_for_views,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.type_definitions import ImageType

from .functional import channel_dropout

__all__ = ["ChannelDropout"]


class ChannelDropout(ImageOnlyTransform):
    """Replace a fraction of eligible channels with a fill value to model missing spectral bands
    or unavailable sensors while retaining at least one input channel.

    Args:
        channels_to_drop (tuple[int, ...] | None): Unique, non-negative channel indices eligible for
            dropout. None makes every input channel eligible. An explicit sequence must be
            non-empty, and every index must exist in each image-like target. Default: None.
        dropout_ratio (float): Fraction of eligible channels to replace, in [0, 1]. The count
            is rounded to the nearest integer using decimal half-up rounding, then capped at the
            total number of input channels minus one. Default: 0.5.
        fill (float): Value assigned to the selected channels. Use a value in the input
            dtype's image range, such as 0 for missing signal. Default: 0.
        p (float): Probability of applying the transform, in [0, 1]. Default: 0.5.

    Targets:
        image, images, volume, volumes

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Channels are selected uniformly without replacement from the eligible set.
        - The fraction is measured against eligible channels, not all input channels.
          With four eligible channels, dropout_ratio=0.5 selects two of them.
        - At least one input channel is retained. A single-channel input is unchanged.
        - A zero ratio or a count rounded to zero leaves the input unchanged.
        - Targets with equal channel counts share the selected indices, including all
          frames in an image batch and all slices in a volume. Different channel counts
          receive separate selections.
        - Retained channel values are unchanged; no dropout rescaling is applied.

    Raises:
        ValueError: If channels_to_drop is empty, contains duplicate or negative indices, or contains
            an index outside an input target's channel range; or if dropout_ratio is outside [0, 1].

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.ones((64, 96, 10), dtype=np.float32)
        >>> transform = A.ChannelDropout(
        ...     channels_to_drop=(0, 2, 4, 6), dropout_ratio=0.5, p=1.0,
        ... )
        >>> result = transform(image=image)["image"]
        >>> assert np.count_nonzero(np.all(result == 0, axis=(0, 1))) == 2
        >>> np.testing.assert_array_equal(result[..., (1, 3, 5, 7, 8, 9)], 1)

    See Also:
        - PixelDropout: Replace individual pixels instead of entire channels.
        - CoarseDropout: Replace rectangular spatial regions.

    """

    class InitSchema(BaseTransformInitSchema):
        channels_to_drop: (
            Annotated[
                tuple[Annotated[int, Field(strict=True, ge=0)], ...],
                Field(min_length=1),
            ]
            | None
        )
        dropout_ratio: float = Field(ge=0, le=1)
        fill: float

        @field_validator("channels_to_drop")
        @classmethod
        def _validate_channels_to_drop(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
            if value is not None and len(value) != len(set(value)):
                raise ValueError("channels_to_drop must not contain duplicate indices")
            return value

    def __init__(
        self,
        channels_to_drop: tuple[int, ...] | None = None,
        dropout_ratio: float = 0.5,
        fill: float = 0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.channels_to_drop = channels_to_drop
        self.dropout_ratio = dropout_ratio
        self.fill = fill

    def apply(self, img: ImageType, selected_channels: tuple[int, ...], **params: Any) -> ImageType:
        return channel_dropout(img, selected_channels, self.fill)

    def apply_to_images(self, images: ImageType, selected_channels: tuple[int, ...], **params: Any) -> ImageType:
        return channel_dropout(images, selected_channels, self.fill)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        ratio_numerator, ratio_denominator = Decimal(str(self.dropout_ratio)).as_integer_ratio()
        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(lambda view: view.descriptor.channels):
            num_channels = views[0].descriptor.channels or 1
            if self.channels_to_drop is not None and max(self.channels_to_drop) >= num_channels:
                raise ValueError(f"channels_to_drop contains an index outside the input's {num_channels} channels")
            eligible_channels = range(num_channels) if self.channels_to_drop is None else self.channels_to_drop
            num_drop_channels = min(
                (2 * len(eligible_channels) * ratio_numerator + ratio_denominator) // (2 * ratio_denominator),
                num_channels - 1,
            )
            selected_channels = tuple(sampling.py_random.sample(eligible_channels, k=num_drop_channels))
            groups.append(
                TargetParams(
                    targets=tuple(view.name for view in views),
                    params={"selected_channels": selected_channels},
                    requirements=requirements_for_views(views, channels=True),
                ),
            )
        return SampledParams(params={}, target_params=tuple(groups))
