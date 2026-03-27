"""Compression and downscale transforms.

Transforms that simulate image compression artifacts and resolution loss.
"""

from typing import Annotated, Any, Literal

import cv2
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator

from albumentations.augmentations.pixel import functional as fpixel
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import ImageType

__all__ = [
    "Downscale",
    "ImageCompression",
]


class ImageCompression(ImageOnlyTransform):
    """Reduce image quality via JPEG or WebP compression. quality_range and compression_type
    control strength and format. Simulates real-world compression artifacts.

    This transform simulates the effect of saving an image with lower quality settings,
    which can introduce compression artifacts. It's useful for data augmentation and
    for testing model robustness against varying image qualities.

    Args:
        quality_range (tuple[int, int]): Range for the compression quality.
            The values should be in [1, 100] range, where:
            - 1 is the lowest quality (maximum compression)
            - 100 is the highest quality (minimum compression)
            Default: (99, 100)

        compression_type (Literal['jpeg', 'webp']): Type of compression to apply.
            - "jpeg": JPEG compression
            - "webp": WebP compression
            Default: "jpeg"

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This transform expects images with 1, 3, or 4 channels.
        - For JPEG compression, alpha channels (4th channel) will be ignored.
        - WebP compression supports transparency (4 channels).
        - The actual file is not saved to disk; the compression is simulated in memory.
        - Lower quality values result in smaller file sizes but may introduce visible artifacts.
        - This transform can be useful for:
          * Data augmentation to improve model robustness
          * Testing how models perform on images of varying quality
          * Simulating images transmitted over low-bandwidth connections

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ImageCompression(quality_range=(50, 90), compression_type=0, p=1.0)
        >>> result = transform(image=image)
        >>> compressed_image = result["image"]

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
        img: ImageType,
        quality: int,
        image_type: Literal[".jpg", ".webp"],
        **params: Any,
    ) -> ImageType:
        return fpixel.image_compression(img, quality, image_type)

    def get_params(self) -> dict[str, int | str]:
        image_type = ".jpg" if self.compression_type == "jpeg" else ".webp"

        quality = self.py_random.randint(*self.quality_range)

        self.applied_config = {"quality_range": quality}

        return {
            "quality": quality,
            "image_type": image_type,
        }


class InterpolationPydantic(BaseModel):
    upscale: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ]

    downscale: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ]


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
        image, volume

    Image types:
        uint8, float32

    Note:
        - The actual downscaling factor is randomly chosen for each image from the range
          specified in scale_range.
        - Using different interpolation methods for downscaling and upscaling can produce
          various effects. For example, using INTER_NEAREST for both can create a pixelated look,
          while using INTER_LINEAR or INTER_CUBIC can produce smoother results.
        - This transform can be useful for data augmentation, especially when training models
          that need to be robust to variations in image quality or resolution.

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
            Literal[
                cv2.INTER_NEAREST,
                cv2.INTER_NEAREST_EXACT,
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC,
                cv2.INTER_AREA,
                cv2.INTER_LANCZOS4,
                cv2.INTER_LINEAR_EXACT,
            ],
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
            Literal[
                cv2.INTER_NEAREST,
                cv2.INTER_NEAREST_EXACT,
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC,
                cv2.INTER_AREA,
                cv2.INTER_LANCZOS4,
                cv2.INTER_LINEAR_EXACT,
            ],
        ] = {"upscale": cv2.INTER_NEAREST, "downscale": cv2.INTER_NEAREST},
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale_range = scale_range
        self.interpolation_pair = interpolation_pair

    def apply(self, img: ImageType, scale: float, **params: Any) -> ImageType:
        return fpixel.downscale(
            img,
            scale=scale,
            down_interpolation=self.interpolation_pair["downscale"],
            up_interpolation=self.interpolation_pair["upscale"],
        )

    def get_params(self) -> dict[str, Any]:
        scale = self.py_random.uniform(*self.scale_range)

        self.applied_config = {"scale_range": scale}

        return {"scale": scale}
