"""Channel manipulation transforms.

Transforms that reorder or permute image channels.
"""

import warnings
from typing import Any, ClassVar

from pydantic import field_validator

from albumentations.augmentations.pixel import functional as fpixel
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import ImageType, VolumeType

__all__ = [
    "ChannelShuffle",
    "ChannelSwap",
]


class _ChannelShuffleInitSchema(BaseTransformInitSchema):
    channel_order: tuple[int, ...] | None

    @field_validator("channel_order")
    @classmethod
    def validate_channel_order(
        cls,
        v: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        """Validate that channel_order is a valid permutation of consecutive integers starting
        from zero (i.e., a permutation of range(len(channel_order))).
        """
        if v is None:
            return v
        if len(v) < 2:
            msg = "channel_order must have at least 2 elements."
            raise ValueError(msg)
        if sorted(v) != list(range(len(v))):
            msg = f"channel_order must be a permutation of range({len(v)}), got {v}"
            raise ValueError(msg)
        return v


class ChannelShuffle(ImageOnlyTransform):
    """Permute image channels. By default the permutation is random (uniform over all
    orderings); set `channel_order` to pin a fixed reordering.

    Args:
        channel_order (tuple[int, ...] | None): Fixed permutation of channel indices.
            When `None` (default), a random permutation is sampled each call.
            When a tuple, that exact order is applied every time (length must match
            the number of image channels).
            Default: None.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - When `channel_order` is `None`, the permutation is chosen uniformly
          over all channel orderings; the same image can get different orderings on
          different calls.
        - When `channel_order` is set, the transform behaves deterministically
          (same as `ChannelSwap`).

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> # Random shuffle (default)
        >>> transform = A.ChannelShuffle(p=1.0)
        >>> result = transform(image=image)["image"]
        >>>
        >>> # Fixed reorder: RGB → BGR
        >>> transform = A.ChannelShuffle(channel_order=(2, 1, 0), p=1.0)
        >>> result = transform(image=image)["image"]

    See Also:
        - ChannelSwap: Convenience alias with `channel_order` required.

    """

    InitSchema: ClassVar[type[BaseTransformInitSchema]] = _ChannelShuffleInitSchema

    def __init__(
        self,
        channel_order: tuple[int, ...] | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.channel_order = channel_order

    def apply(
        self,
        img: ImageType,
        channels_shuffled: list[int] | None,
        **params: Any,
    ) -> ImageType:
        if channels_shuffled is None:
            return img
        return fpixel.channel_shuffle(img, channels_shuffled)

    def apply_to_images(
        self,
        images: ImageType,
        channels_shuffled: list[int] | None,
        **params: Any,
    ) -> ImageType:
        if channels_shuffled is None:
            return images
        return fpixel.volume_channel_shuffle(images, channels_shuffled)

    def apply_to_volumes(
        self,
        volumes: VolumeType,
        channels_shuffled: list[int] | None,
        **params: Any,
    ) -> VolumeType:
        if channels_shuffled is None:
            return volumes
        return fpixel.volumes_channel_shuffle(volumes, channels_shuffled)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        shape = params["shape"]
        num_channels = 1 if len(shape) == 2 else shape[-1]

        if self.channel_order is not None:
            if num_channels != len(self.channel_order):
                warnings.warn(
                    f"channel_order has {len(self.channel_order)} elements "
                    f"but data has {num_channels} channel(s); "
                    f"returning data unchanged.",
                    UserWarning,
                    stacklevel=2,
                )
                return {"channels_shuffled": None}
            return {"channels_shuffled": list(self.channel_order)}

        if num_channels <= 1:
            return {"channels_shuffled": None}

        ch_arr = list(range(num_channels))
        self.py_random.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}


class ChannelSwap(ChannelShuffle):
    """Fixed channel reordering (e.g. RGB->BGR). Convenience subclass of `ChannelShuffle` with a
    required `channel_order` argument.

    Args:
        channel_order (tuple[int, ...]): Permutation of channel indices. Length must
            match image channels. For 3-channel, (2, 1, 0) swaps R and B (RGB->BGR).
            Default: (2, 1, 0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any (channel_order length must match)

    Note:
        - channel_order must be a permutation of 0..C-1 for C channels.
        - (2, 1, 0) gives RGB->BGR; (0, 2, 1) swaps G and B.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> # Swap R and B (RGB -> BGR)
        >>> transform = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        >>> result = transform(image=image)["image"]
        >>> np.testing.assert_array_equal(result[:, :, 0], image[:, :, 2])

    See Also:
        - ChannelShuffle: Random permutation each call; use for invariance
          to channel order.

    """

    class InitSchema(BaseTransformInitSchema):
        channel_order: tuple[int, ...]

    def __init__(
        self,
        channel_order: tuple[int, ...] = (2, 1, 0),
        p: float = 0.5,
    ):
        super().__init__(channel_order=channel_order, p=p)
