"""Compression and downscale transforms.

Transforms that simulate image compression artifacts and resolution loss.
"""

from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic.functional_validators import AfterValidator

from albumentations.augmentations.pixel import functional as fpixel
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transform_params import SampledParams, TargetSet
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import CV2_INTER_NEAREST, FullInterpolationType

__all__ = [
    "Downscale",
    "ImageCompression",
]


class ImageCompression(ImageOnlyTransform):
    """Re-encode the image as lossy JPEG or WebP and decode it, adding the blocking, ringing, and
    color-smearing artifacts of photos saved at reduced quality.

    Use it to make a model robust to compression it meets in real data, sampling a range of qualities
    during training, or to measure how accuracy falls as quality drops by fixing one quality at a time
    during evaluation. Encoding and decoding happen in memory; nothing is written to disk.

    Args:
        quality_range (tuple[int, int]): Inclusive range of integer qualities in [1, 100]. One quality is
            sampled uniformly each time the transform is applied. Lower values discard more detail; 1 is the
            strongest compression and 100 the weakest. Set both endpoints equal for a fixed quality.
            Default: (99, 100).
        compression_type (Literal['jpeg', 'webp']): Codec used for the round trip. Default: "jpeg".
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, images, volume, volumes

    Image types:
        uint8, float16, float32

    Number of channels:
        Any

    Note:
        - Every channel is image data and receives artifacts. The first three channels are compressed together
          as a color image. Each further channel is compressed separately as a grayscale plane. One channel is
          compressed as grayscale, and two channels are padded with a zero third channel for encoding.
        - A fourth channel is an ordinary channel, not an alpha mask. Pass masks and labels through `mask`
          or `masks`, which this transform does not change.
        - Floating-point images pass through uint8 during encoding and are restored to the input dtype, so
          values are quantized to 256 levels.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.default_rng(137).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ImageCompression(quality_range=(30, 70), compression_type="webp", p=1.0)
        >>> compressed = transform(image=image)["image"]
        >>> compressed.shape == image.shape
        True

    See Also:
        Downscale: Lose fine detail by shrinking and enlarging the image instead of re-encoding it.
        Posterize: Reduce intensity levels per channel to produce visible banding.

    References:
        - JPEG compression: https://en.wikipedia.org/wiki/JPEG
        - WebP compression: https://developers.google.com/speed/webp

    """

    class InitSchema(BaseTransformInitSchema):
        quality_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, 100)),
            AfterValidator(nondecreasing),
        ]

        compression_type: Literal["jpeg", "webp"]

    def __init__(
        self,
        compression_type: Literal["jpeg", "webp"] = "jpeg",
        quality_range: tuple[int, int] = (99, 100),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.quality_range = quality_range
        self.compression_type = compression_type

    def apply(
        self,
        img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
        quality: int,
        image_type: Literal[".jpg", ".webp"],
        **params: Any,
    ) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
        return fpixel.image_compression(img, quality, image_type)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        image_type = ".jpg" if self.compression_type == "jpeg" else ".webp"

        quality = sampling.py_random.randint(*self.quality_range)

        sampling.applied_overrides["quality_range"] = quality

        return SampledParams(
            params={
                "quality": quality,
                "image_type": image_type,
            }
        )


class Downscale(ImageOnlyTransform):
    """Reduce quality by downscale then upscale. scale_min and scale_max control factor.
    Simulates resolution or compression loss.

    This transform simulates the effect of a low-resolution image by first downscaling
    the image to a lower resolution and then upscaling it back to its original size.
    This process introduces loss of detail and can be used to simulate low-quality
    images or to test the robustness of models to different image resolutions.

    Args:
        scale_range (tuple[float, float]): Range for the downscaling factor.
            Should be two float values between 0 and 1, where the first value is less than or equal to the second.
            The actual downscaling factor will be randomly chosen from this range for each image.
            Lower values result in more aggressive downscaling.
            Default: (0.25, 0.25)

        interpolation_pair (dict[Literal['downscale', 'upscale'], int]): A dictionary specifying
            the interpolation methods to use for downscaling and upscaling.
            Should contain two keys:
            - 'downscale': Interpolation method for downscaling
            - 'upscale': Interpolation method for upscaling
            Values should be OpenCV interpolation flags (e.g., cv2.INTER_NEAREST, cv2.INTER_LINEAR, etc.)
            Default: {'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_NEAREST}

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image, images, volume, volumes

    Image types:
        uint8, float16, float32

    Note:
        - The actual downscaling factor is randomly chosen for each image from the range
          specified in scale_range.
        - Using different interpolation methods for downscaling and upscaling can produce
          various effects. For example, using INTER_NEAREST for both can create a pixelated look,
          while using INTER_LINEAR or INTER_CUBIC can produce smoother results.
        - This transform can be useful for data augmentation, especially when training models
          that need to be robust to variations in image quality or resolution.

    See Also:
        ImageCompression: Add JPEG or WebP encoding artifacts instead of resampling blur.

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> transform = A.Downscale(
        ...     scale_range=(0.5, 0.75),
        ...     interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR},
        ...     p=0.5
        ... )
        >>> transformed = transform(image=image)
        >>> downscaled_image = transformed['image']

    """

    class InitSchema(BaseTransformInitSchema):
        interpolation_pair: dict[
            Literal["downscale", "upscale"],
            FullInterpolationType,
        ]
        scale_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.25, 0.25),
        interpolation_pair: dict[
            Literal["downscale", "upscale"],
            FullInterpolationType,
        ] = {"upscale": CV2_INTER_NEAREST, "downscale": CV2_INTER_NEAREST},
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale_range = scale_range
        self.interpolation_pair = interpolation_pair

    def apply(
        self, img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], scale: float, **params: Any
    ) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
        return fpixel.downscale(
            img,
            scale=scale,
            down_interpolation=self.interpolation_pair["downscale"],
            up_interpolation=self.interpolation_pair["upscale"],
        )

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        scale = sampling.py_random.uniform(*self.scale_range)

        sampling.applied_overrides["scale_range"] = scale

        return SampledParams(params={"scale": scale})
