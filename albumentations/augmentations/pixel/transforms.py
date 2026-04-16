"""Pixel-level transformations for image augmentation.

This module contains transforms that modify pixel values without changing the geometry of the image.
Includes transforms for normalization, sharpening, embossing, superpixels, ringing, unsharp mask,
dithering, halftone, and lens flare effects.
"""

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

import cv2
import numpy as np
from albucore import (
    normalize,
    normalize_per_image,
    reduce_sum,
)
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from scipy import special
from typing_extensions import Self

from albumentations.augmentations.blur import functional as fblur
from albumentations.augmentations.blur.transforms import BlurInitSchema
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.utils import non_rgb_error
from albumentations.core.pydantic import (
    check_range_bounds,
    convert_to_0plus_range,
    convert_to_1plus_int_range,
    nondecreasing,
    process_non_negative_range,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import ImageType, VolumeType

__all__ = [
    "Dithering",
    "Emboss",
    "Enhance",
    "Halftone",
    "InvertImg",
    "LensFlare",
    "Normalize",
    "RingingOvershoot",
    "Sharpen",
    "Superpixels",
    "UnsharpMask",
]


class Normalize(ImageOnlyTransform):
    """Applies various normalization techniques to an image. The specific normalization technique can be selected
        with the `normalization` parameter.

    Standard normalization is applied using the formula:
        `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`.
        Other normalization techniques adjust the image based on global or per-channel statistics,
        or scale pixel values to a specified range.

    Args:
        mean (tuple[float, float] | float | None): Mean values for standard normalization.
            For "standard" normalization, the default values are ImageNet mean values: (0.485, 0.456, 0.406).
        std (tuple[float, float] | float | None): Standard deviation values for standard normalization.
            For "standard" normalization, the default values are ImageNet standard deviation :(0.229, 0.224, 0.225).
        max_pixel_value (float | None): Maximum possible pixel value, used for scaling in standard normalization.
            Defaults to 255.0.
        normalization (Literal['standard', 'image', 'image_per_channel', 'min_max', 'min_max_per_channel']):
            Specifies the normalization technique to apply. Defaults to "standard".
            - "standard": Applies the formula `(img - mean * max_pixel_value) / (std * max_pixel_value)`.
                The default mean and std are based on ImageNet. You can use mean and std values of (0.5, 0.5, 0.5)
                for inception normalization. And mean values of (0, 0, 0) and std values of (1, 1, 1) for YOLO.
            - "image": Normalizes the whole image based on its global mean and standard deviation.
            - "image_per_channel": Normalizes the image per channel based on each channel's mean and standard deviation.
            - "min_max": Scales the image pixel values to a [0, 1] range based on the global
                minimum and maximum pixel values.
            - "min_max_per_channel": Scales each channel of the image pixel values to a [0, 1]
                range based on the per-channel minimum and maximum pixel values.

        p (float): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - For "standard" normalization, `mean`, `std`, and `max_pixel_value` must be provided.
        - For other normalization types, these parameters are ignored.
        - For inception normalization, use mean values of (0.5, 0.5, 0.5).
        - For YOLO normalization, use mean values of (0, 0, 0) and std values of (1, 1, 1).
        - This transform is often used as a final step in image preprocessing pipelines to
          prepare images for neural network input.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Standard ImageNet normalization
        >>> transform = A.Normalize(
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225),
        ...     max_pixel_value=255.0,
        ...     p=1.0
        ... )
        >>> normalized_image = transform(image=image)["image"]
        >>>
        >>> # Min-max normalization
        >>> transform_minmax = A.Normalize(normalization="min_max", p=1.0)
        >>> normalized_image_minmax = transform_minmax(image=image)["image"]

    References:
        - ImageNet mean and std: https://pytorch.org/vision/stable/models.html
        - Inception preprocessing: https://keras.io/api/applications/inceptionv3/

    """

    class InitSchema(BaseTransformInitSchema):
        mean: tuple[float, ...] | float | None
        std: tuple[float, ...] | float | None
        max_pixel_value: float | None
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ]

        @model_validator(mode="after")
        def _validate_normalization(self) -> Self:
            if (
                self.mean is None
                or self.std is None
                or (self.max_pixel_value is None and self.normalization == "standard")
            ):
                raise ValueError(
                    "mean, std, and max_pixel_value must be provided for standard normalization.",
                )
            return self

    def __init__(
        self,
        mean: tuple[float, ...] | float | None = (0.485, 0.456, 0.406),
        std: tuple[float, ...] | float | None = (0.229, 0.224, 0.225),
        max_pixel_value: float | None = 255.0,
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ] = "standard",
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.mean = mean
        self.mean_np = np.array(mean, dtype=np.float32) * max_pixel_value
        self.std = std
        self.denominator = np.reciprocal(
            np.array(std, dtype=np.float32) * max_pixel_value,
        )
        self.max_pixel_value = max_pixel_value
        self.normalization = normalization

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        if self.normalization == "standard":
            return normalize(
                img,
                self.mean_np,
                self.denominator,
            )
        return normalize_per_image(img, self.normalization)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volume(self, volume: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volume, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


class InvertImg(ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from max values of the image types,
    i.e., 255 for uint8 and 1.0 for float32.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image with different elements
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> cv2.circle(image, (30, 30), 20, (255, 255, 255), -1)  # White circle
        >>> cv2.rectangle(image, (60, 60), (90, 90), (128, 128, 128), -1)  # Gray rectangle
        >>>
        >>> # Apply InvertImg transform
        >>> transform = A.InvertImg(p=1.0)
        >>> result = transform(image=image)
        >>> inverted_image = result['image']
        >>>
        >>> # Result:
        >>> # - Black background becomes white (0 → 255)
        >>> # - White circle becomes black (255 → 0)
        >>> # - Gray rectangle is inverted (128 → 127)
        >>> # The same approach works for float32 images (0-1 range) and grayscale images

    """

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return fpixel.invert(img)

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        return self.apply(images, *args, **params)

    def apply_to_volumes(self, volumes: VolumeType, *args: Any, **params: Any) -> VolumeType:
        return self.apply(volumes, *args, **params)


class Sharpen(ImageOnlyTransform):
    """Sharpen the image via kernel or Gaussian unsharp method. alpha and lightness control
    strength. Enhances edges; useful for document or detail-sensitive tasks.

    Implements two different approaches to image sharpening:
    1. Traditional kernel-based method using Laplacian operator
    2. Gaussian interpolation method (similar to Kornia's approach)

    Args:
        alpha (tuple[float, float]): Range for the visibility of sharpening effect.
            At 0, only the original image is visible, at 1.0 only its processed version is visible.
            Values should be in the range [0, 1].
            Used in both methods. Default: (0.2, 0.5).

        lightness (tuple[float, float]): Range for the lightness of the sharpened image.
            Only used in 'kernel' method. Larger values create higher contrast.
            Values should be greater than 0. Default: (0.5, 1.0).

        method (Literal['kernel', 'gaussian']): Sharpening algorithm to use:
            - 'kernel': Traditional kernel-based sharpening using Laplacian operator
            - 'gaussian': Interpolation between Gaussian blurred and original image
            Default: 'kernel'

        kernel_size (int): Size of the Gaussian blur kernel for 'gaussian' method.
            Must be odd. Default: 5

        sigma (float): Standard deviation for Gaussian kernel in 'gaussian' method.
            Default: 1.0

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Mathematical Formulation:
        1. Kernel Method:
           The sharpening operation is based on the Laplacian operator L:
           L = [[-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]]

           The final kernel K is a weighted sum:
           K = (1 - a)I + a(L + λI)

           where:
           - a is the alpha value
           - λ is the lightness value
           - I is the identity kernel

           The output image O is computed as:
           O = K * I  (convolution)

        2. Gaussian Method:
           Based on the unsharp mask principle:
           O = aI + (1-a)G

           where:
           - I is the input image
           - G is the Gaussian blurred version of I
           - a is the alpha value (sharpness)

           The Gaussian kernel G(x,y) is defined as:
           G(x,y) = (1/(2πs²))exp(-(x²+y²)/(2s²))

    Note:
        - Kernel sizes must be odd to maintain spatial alignment
        - Methods produce different visual results:
          * Kernel method: More pronounced edges, possible artifacts
          * Gaussian method: More natural look, limited to original sharpness

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Traditional kernel sharpening
        >>> transform = A.Sharpen(
        ...     alpha=(0.2, 0.5),
        ...     lightness=(0.5, 1.0),
        ...     method='kernel',
        ...     p=1.0
        ... )

        # Gaussian interpolation sharpening
        >>> transform = A.Sharpen(
        ...     alpha=(0.5, 1.0),
        ...     method='gaussian',
        ...     kernel_size=5,
        ...     sigma=1.0,
        ...     p=1.0
        ... )

    References:
        - R. C. Gonzalez and R. E. Woods, "Digital Image Processing (4th Edition),": Chapter 3:
            Intensity Transformations and Spatial Filtering.
        - J. C. Russ, "The Image Processing Handbook (7th Edition),": Chapter 4: Image Enhancement.
        - T. Acharya and A. K. Ray, "Image Processing: Principles and Applications,": Chapter 5: Image Enhancement.
        - Unsharp masking: https://en.wikipedia.org/wiki/Unsharp_masking
        - Laplacian operator: https://en.wikipedia.org/wiki/Laplace_operator
        - Gaussian blur: https://en.wikipedia.org/wiki/Gaussian_blur

    See Also:
        - Enhance: Compact Pillow-inspired preset family (edge/detail) for milder, more
          targeted local enhancement than broad Sharpen.
        - UnsharpMask: Alternative sharpening method.
        - Blur: For Gaussian blurring.
        - RandomBrightnessContrast: For adjusting image contrast.

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        lightness: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, None))]
        method: Literal["kernel", "gaussian"]
        kernel_size: int = Field(ge=3)
        sigma: float = Field(gt=0)

    @field_validator("kernel_size")
    @classmethod
    def _check_kernel_size(cls, value: int) -> int:
        return value + 1 if value % 2 == 0 else value

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        lightness: tuple[float, float] = (0.5, 1.0),
        method: Literal["kernel", "gaussian"] = "kernel",
        kernel_size: int = 5,
        sigma: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.alpha = alpha
        self.lightness = lightness
        self.method = method
        self.kernel_size = kernel_size
        self.sigma = sigma

    @staticmethod
    def __generate_sharpening_matrix(
        alpha: np.ndarray,
        lightness: np.ndarray,
    ) -> np.ndarray:
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]],
            dtype=np.float32,
        )

        return (1 - alpha) * matrix_nochange + alpha * matrix_effect

    def get_params(self) -> dict[str, Any]:
        alpha = self.py_random.uniform(*self.alpha)

        if self.method == "kernel":
            lightness = self.py_random.uniform(*self.lightness)
            self.applied_config = {"alpha": alpha, "lightness": lightness}
            return {
                "alpha": alpha,
                "sharpening_matrix": self.__generate_sharpening_matrix(
                    alpha,
                    lightness,
                ),
            }

        self.applied_config = {"alpha": alpha}
        return {"alpha": alpha, "sharpening_matrix": None}

    def apply(
        self,
        img: ImageType,
        alpha: float,
        sharpening_matrix: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if self.method == "kernel":
            return fpixel.convolve(img, sharpening_matrix)
        return fpixel.sharpen_gaussian(img, alpha, self.kernel_size, self.sigma)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))


class Emboss(ImageOnlyTransform):
    """Apply emboss effect (directional highlight and shadow). strength_range controls
    intensity. Pseudo-3D look; for texture or style augmentation.

    This transform creates an emboss effect by highlighting edges and creating a 3D-like texture
    in the image. It works by applying a specific convolution kernel to the image that emphasizes
    differences in adjacent pixel values.

    Args:
        alpha (tuple[float, float]): Range to choose the visibility of the embossed image.
            At 0, only the original image is visible, at 1.0 only its embossed version is visible.
            Values should be in the range [0, 1].
            Alpha will be randomly selected from this range for each image.
            Default: (0.2, 0.5)

        strength (tuple[float, float]): Range to choose the strength of the embossing effect.
            Higher values create a more pronounced 3D effect.
            Values should be non-negative.
            Strength will be randomly selected from this range for each image.
            Default: (0.2, 0.7)

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - The emboss effect is created using a 3x3 convolution kernel.
        - The 'alpha' parameter controls the blend between the original image and the embossed version.
          A higher alpha value will result in a more pronounced emboss effect.
        - The 'strength' parameter affects the intensity of the embossing. Higher strength values
          will create more contrast in the embossed areas, resulting in a stronger 3D-like effect.
        - This transform can be useful for creating artistic effects or for data augmentation
          in tasks where edge information is important.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5)
        >>> result = transform(image=image)
        >>> embossed_image = result['image']

    References:
        - Image Embossing: https://en.wikipedia.org/wiki/Image_embossing
        - Application of Emboss Filtering in Image Processing: https://www.researchgate.net/publication/303412455_Application_of_Emboss_Filtering_in_Image_Processing

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        strength: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, None))]

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        strength: tuple[float, float] = (0.2, 0.7),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.alpha = alpha
        self.strength = strength

    @staticmethod
    def __generate_emboss_matrix(
        alpha_sample: np.ndarray,
        strength_sample: np.ndarray,
    ) -> np.ndarray:
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [
                [-1 - strength_sample, 0 - strength_sample, 0],
                [0 - strength_sample, 1, 0 + strength_sample],
                [0, 0 + strength_sample, 1 + strength_sample],
            ],
            dtype=np.float32,
        )
        return (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect

    def get_params(self) -> dict[str, np.ndarray]:
        alpha = self.py_random.uniform(*self.alpha)
        strength = self.py_random.uniform(*self.strength)
        emboss_matrix = self.__generate_emboss_matrix(
            alpha_sample=alpha,
            strength_sample=strength,
        )
        self.applied_config = {"alpha": alpha, "strength": strength}
        return {"emboss_matrix": emboss_matrix}

    def apply(
        self,
        img: ImageType,
        emboss_matrix: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.convolve(img, emboss_matrix)


class Enhance(ImageOnlyTransform):
    """Mild Pillow-inspired local enhancement (edge or detail) blended with the original via
    alpha. Use for subtle contour or detail boost milder than Sharpen.

    A native Albumentations implementation of the Pillow `EDGE_ENHANCE` / `EDGE_ENHANCE_MORE`
    and `DETAIL` filter family. The enhanced image is computed via a small 3x3 convolution and
    then blended with the original:

        output = (1 - alpha) * image + alpha * enhanced_image

    Equivalently, the convolution is applied with the precomputed kernel
    `K(alpha) = (1 - alpha) * I + alpha * E` where `E` is the mode-specific kernel.
    Because each `E` sums to 1, `K(alpha)` also sums to 1 and brightness is preserved.

    For `mode="edge"`, `alpha=1` reproduces Pillow's `EDGE_ENHANCE` and `alpha=2`
    reproduces `EDGE_ENHANCE_MORE`. For `mode="detail"`, `alpha=1` reproduces Pillow's
    `DETAIL`. Values of `alpha` between 0 and 1 give milder presets; values above 1
    overshoot for stronger effects.

    Args:
        mode (Literal['edge', 'detail']): Which native enhancement operator to use.
            - "edge": crispens contours and boundaries (Pillow EDGE_ENHANCE family).
            - "detail": mild local detail / fine-structure boost (Pillow DETAIL).
            Default: "edge".
        alpha_range (tuple[float, float]): Range from which the blend strength `alpha` is
            sampled uniformly per call. `alpha=0` is no-op, `alpha=1` is the full Pillow
            preset, `alpha>1` overshoots into a stronger variant. Must be non-decreasing
            with non-negative values. Default: (0.5, 1.0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Edge handling follows the rest of Albumentations (cv2 `BORDER_REFLECT_101`),
          which differs slightly from Pillow's clamping at borders.
        - For uint8 inputs the output saturates to `[0, 255]`; for float32 it is clipped
          to `[0, 1]` by `convolve`.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> # Edge enhancement, mild to Pillow-EDGE_ENHANCE strength
        >>> transform = A.Compose([A.Enhance(mode="edge", alpha_range=(0.5, 1.0), p=1.0)])
        >>> result = transform(image=image)["image"]
        >>>
        >>> # Stronger edge variant (alpha=2 matches Pillow EDGE_ENHANCE_MORE)
        >>> transform = A.Compose([A.Enhance(mode="edge", alpha_range=(1.0, 2.0), p=1.0)])
        >>> result = transform(image=image)["image"]
        >>>
        >>> # Subtle detail / fine-structure boost
        >>> transform = A.Compose([A.Enhance(mode="detail", alpha_range=(0.5, 1.0), p=1.0)])
        >>> result = transform(image=image)["image"]

    References:
        Pillow ImageFilter (EDGE_ENHANCE, EDGE_ENHANCE_MORE, DETAIL):
            https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html

    See Also:
        - Sharpen: Broader high-frequency sharpening (kernel-Laplacian or unsharp-mask).
          Use when you need a continuous, configurable sharpening primitive rather than
          a compact preset family.
        - UnsharpMask: Classic unsharp-mask sharpening with explicit blur radius.
        - Emboss: Directional edge highlight for stylization rather than enhancement.

    """

    class InitSchema(BaseTransformInitSchema):
        mode: Literal["edge", "detail"]
        alpha_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        mode: Literal["edge", "detail"] = "edge",
        alpha_range: tuple[float, float] = (0.5, 1.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.mode = mode
        self.alpha_range = alpha_range

    def get_params(self) -> dict[str, Any]:
        alpha = self.py_random.uniform(*self.alpha_range)
        # Record the resolved scalar (not the range) for replay/debug, per the
        # applied_config contract documented on get_applied_config.
        self.applied_config = {"alpha_range": alpha}
        return {"enhance_matrix": fpixel.generate_enhance_matrix(self.mode, alpha)}

    def apply(
        self,
        img: ImageType,
        enhance_matrix: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.convolve(img, enhance_matrix)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))


class Superpixels(ImageOnlyTransform):
    """Replace image with superpixel segmentation (SLIC). p_replace, n_segments, max_size
    control fraction and segment count. Reduces fine texture.

    Args:
        p_replace (tuple[float, float] | float): Defines for any segment the probability that the pixels within that
            segment are replaced by their average color (otherwise, the pixels are not changed).


            * A probability of `0.0` would mean, that the pixels in no
                segment are replaced by their average color (image is not
                changed at all).
            * A probability of `0.5` would mean, that around half of all
                segments are replaced by their average color.
            * A probability of `1.0` would mean, that all segments are
                replaced by their average color (resulting in a voronoi
                image).

            Behavior based on chosen data types for this parameter:
            * If a `float`, then that `float` will always be used.
            * If `tuple` `(a, b)`, then a random probability will be
            sampled from the interval `[a, b]` per image.
            Default: (0.1, 0.3)

        n_segments (tuple[int, int] | int): Rough target number of how many superpixels to generate.
            The algorithm may deviate from this number.
            Lower value will lead to coarser superpixels.
            Higher values are computationally more intensive and will hence lead to a slowdown.
            If tuple `(a, b)`, then a value from the discrete interval `[a..b]` will be sampled per image.
            Default: (15, 120)

        max_size (int | None): Maximum image size at which the augmentation is performed.
            If the width or height of an image exceeds this value, it will be
            downscaled before the augmentation so that the longest side matches `max_size`.
            This is done to speed up the process. The final output image has the same size as the input image.
            Note that in case `p_replace` is below `1.0`,
            the down-/upscaling will affect the not-replaced pixels too.
            Use `None` to apply no down-/upscaling.
            Default: 128

        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This transform can significantly change the visual appearance of the image.
        - The transform makes use of a superpixel algorithm, which tends to be slow.
        If performance is a concern, consider using `max_size` to limit the image size.
        - The effect of this transform can vary greatly depending on the `p_replace` and `n_segments` parameters.
        - When `p_replace` is high, the image can become highly abstracted, resembling a voronoi diagram.
        - The transform preserves the original image type (uint8 or float32).

    Mathematical Formulation:
        1. The image is segmented into approximately `n_segments` superpixels using the SLIC algorithm.
        2. For each superpixel:
        - With probability `p_replace`, all pixels in the superpixel are replaced with their mean color.
        - With probability `1 - p_replace`, the superpixel is left unchanged.
        3. If the image was resized due to `max_size`, it is resized back to its original dimensions.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Apply superpixels with default parameters
        >>> transform = A.Superpixels(p=1.0)
        >>> augmented_image = transform(image=image)['image']

        # Apply superpixels with custom parameters
        >>> transform = A.Superpixels(
        ...     p_replace=(0.5, 0.7),
        ...     n_segments=(50, 100),
        ...     max_size=None,
        ...     interpolation=cv2.INTER_NEAREST,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)['image']

    """

    class InitSchema(BaseTransformInitSchema):
        p_replace: Annotated[
            tuple[float, float] | float,
            AfterValidator(convert_to_0plus_range),
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        n_segments: Annotated[
            tuple[int, int] | int,
            AfterValidator(convert_to_1plus_int_range),
            AfterValidator(check_range_bounds(1, None)),
        ]
        max_size: int | None = Field(ge=1)
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

    def __init__(
        self,
        p_replace: tuple[float, float] | float = (0, 0.1),
        n_segments: tuple[int, int] | int = (100, 100),
        max_size: int | None = 128,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.p_replace = cast("tuple[float, float]", p_replace)
        self.n_segments = cast("tuple[int, int]", n_segments)
        self.max_size = max_size
        self.interpolation = interpolation

    def get_params(self) -> dict[str, Any]:
        n_segments = self.py_random.randint(*self.n_segments)
        p = self.py_random.uniform(*self.p_replace)
        self.applied_config = {"n_segments": n_segments, "p_replace": p}
        return {
            "replace_samples": self.random_generator.random(n_segments) < p,
            "n_segments": n_segments,
        }

    def apply(
        self,
        img: ImageType,
        replace_samples: Sequence[bool],
        n_segments: int,
        **kwargs: Any,
    ) -> ImageType:
        return fpixel.superpixels(
            img,
            n_segments,
            replace_samples,
            self.max_size,
            self.interpolation,
        )


class RingingOvershoot(ImageOnlyTransform):
    """Create ringing or overshoot artifacts via 2D sinc convolution. blur_limit and
    cutoff control strength. Simulates sharpening or compression artifacts.

    This transform simulates the ringing artifacts that can occur in digital image processing,
    particularly after sharpening or edge enhancement operations. It creates oscillations
    or overshoots near sharp transitions in the image.

    Args:
        blur_limit (tuple[int, int] | int): Maximum kernel size for the sinc filter.
            Must be an odd number in the range [3, inf).
            If a single int is provided, the kernel size will be randomly chosen
            from the range (3, blur_limit). If a tuple (min, max) is provided,
            the kernel size will be randomly chosen from the range (min, max).
            Default: (7, 15).
        cutoff (tuple[float, float]): Range to choose the cutoff frequency in radians.
            Values should be in the range (0, π). A lower cutoff frequency will
            result in more pronounced ringing effects.
            Default: (π/4, π/2).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Ringing artifacts are oscillations of the image intensity function in the neighborhood
          of sharp transitions, such as edges or object boundaries.
        - This transform uses a 2D sinc filter (also known as a 2D cardinal sine function)
          to introduce these artifacts.
        - The severity of the ringing effect is controlled by both the kernel size (blur_limit)
          and the cutoff frequency.
        - Larger kernel sizes and lower cutoff frequencies will generally produce more
          noticeable ringing effects.
        - This transform can be useful for:
          * Simulating imperfections in image processing or transmission systems
          * Testing the robustness of computer vision models to ringing artifacts
          * Creating artistic effects that emphasize edges and transitions in images

    Mathematical Formulation:
        The 2D sinc filter kernel is defined as:

        K(x, y) = cutoff * J₁(cutoff * √(x² + y²)) / (2π * √(x² + y²))

        where:
        - J₁ is the Bessel function of the first kind of order 1
        - cutoff is the chosen cutoff frequency
        - x and y are the distances from the kernel center

        The filtered image I' is obtained by convolving the input image I with the kernel K:

        I'(x, y) = ∑∑ I(x-u, y-v) * K(u, v)

        The convolution operation introduces the ringing artifacts near sharp transitions.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Apply ringing effect with default parameters
        >>> transform = A.RingingOvershoot(p=1.0)
        >>> ringing_image = transform(image=image)['image']

        # Apply ringing effect with custom parameters
        >>> transform = A.RingingOvershoot(
        ...     blur_limit=(9, 17),
        ...     cutoff=(np.pi/6, np.pi/3),
        ...     p=1.0
        ... )
        >>> ringing_image = transform(image=image)['image']

    References:
        - Ringing artifacts: https://en.wikipedia.org/wiki/Ringing_artifacts
        - Sinc filter: https://en.wikipedia.org/wiki/Sinc_filter
        - Digital Image Processing: Rafael C. Gonzalez and Richard E. Woods, 4th Edition

    """

    class InitSchema(BlurInitSchema):
        blur_limit: tuple[int, int] | int
        cutoff: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, np.pi)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (7, 15),
        cutoff: tuple[float, float] = (np.pi / 4, np.pi / 2),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.cutoff = cutoff

    def get_params(self) -> dict[str, np.ndarray]:
        ksize = self.py_random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        if ksize % 2 == 0:
            ksize += 1

        cutoff = self.py_random.uniform(*self.cutoff)

        # From dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = np.fromfunction(
                lambda x, y: (
                    cutoff
                    * special.j1(
                        cutoff * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2),
                    )
                    / (2 * np.pi * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2))
                ),
                [ksize, ksize],
            )
        kernel[(ksize - 1) // 2, (ksize - 1) // 2] = cutoff**2 / (4 * np.pi)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / reduce_sum(kernel)

        self.applied_config = {"blur_limit": ksize, "cutoff": cutoff}
        return {"kernel": kernel}

    def apply(self, img: ImageType, kernel: np.ndarray, **params: Any) -> ImageType:
        return fpixel.convolve(img, kernel)


class UnsharpMask(ImageOnlyTransform):
    """Sharpen via unsharp masking: blur, subtract, add back. blur_limit, sigma_limit, alpha
    control strength. Luminance unchanged; edges enhanced.

    Unsharp masking is a technique that enhances edge contrast in an image, creating the illusion of increased
        sharpness.
    This transform applies Gaussian blur to create a blurred version of the image, then uses this to create a mask
    which is combined with the original image to enhance edges and fine details.

    Args:
        blur_limit (tuple[int, int] | int): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (tuple[float, float] | float): Gaussian kernel standard deviation. Must be more or equal to 0.
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha (tuple[float, float]): range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold (int): Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 255]. Default: 10.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - The algorithm creates a mask M = (I - G) * alpha, where I is the original image and G is the Gaussian
            blurred version.
        - The final image is computed as: output = I + M if |I - G| > threshold, else I.
        - Higher alpha values increase the strength of the sharpening effect.
        - Higher threshold values limit the sharpening effect to areas with more significant edges or details.
        - The blur_limit and sigma_limit parameters control the Gaussian blur used to create the mask.

    References:
        Unsharp Masking: https://en.wikipedia.org/wiki/Unsharp_masking

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        # Apply UnsharpMask with default parameters
        >>> transform = A.UnsharpMask(p=1.0)
        >>> sharpened_image = transform(image=image)['image']
        >>>
        # Apply UnsharpMask with custom parameters
        >>> transform = A.UnsharpMask(
        ...     blur_limit=(3, 7),
        ...     sigma_limit=(0.1, 0.5),
        ...     alpha=(0.2, 0.7),
        ...     threshold=15,
        ...     p=1.0
        ... )
        >>> sharpened_image = transform(image=image)['image']

    """

    class InitSchema(BaseTransformInitSchema):
        sigma_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(process_non_negative_range),
            AfterValidator(nondecreasing),
        ]
        alpha: Annotated[
            tuple[float, float] | float,
            AfterValidator(convert_to_0plus_range),
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        threshold: int = Field(ge=0, le=255)
        blur_limit: tuple[int, int] | int

        @field_validator("blur_limit")
        @classmethod
        def _process_blur(
            cls,
            value: tuple[int, int] | int,
            info: ValidationInfo,
        ) -> tuple[int, int]:
            return fblur.process_blur_limit(value, info, min_value=3)

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (3, 7),
        sigma_limit: tuple[float, float] | float = 0.0,
        alpha: tuple[float, float] | float = (0.2, 0.5),
        threshold: int = 10,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.sigma_limit = cast("tuple[float, float]", sigma_limit)
        self.alpha = cast("tuple[float, float]", alpha)
        self.threshold = threshold

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        ksize = self.py_random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        sigma = self.py_random.uniform(*self.sigma_limit)
        alpha = self.py_random.uniform(*self.alpha)
        self.applied_config = {"blur_limit": ksize, "sigma_limit": sigma, "alpha": alpha}
        return {"ksize": ksize, "sigma": sigma, "alpha": alpha}

    def apply(
        self,
        img: ImageType,
        ksize: int,
        sigma: int,
        alpha: float,
        **params: Any,
    ) -> ImageType:
        return fpixel.unsharp_mask(
            img,
            ksize,
            sigma=sigma,
            alpha=alpha,
            threshold=self.threshold,
        )

    def apply_to_images(
        self,
        images: ImageType,
        ksize: int,
        sigma: int,
        alpha: float,
        **params: Any,
    ) -> ImageType:
        return fpixel.unsharp_mask_images(
            images,
            ksize,
            sigma=sigma,
            alpha=alpha,
            threshold=self.threshold,
        )


class Dithering(ImageOnlyTransform):
    """Reduce colors via dithering: ordered Bayer, error diffusion, or random. num_levels, method.
    Good for retro look or limited-color output.

    Dithering is like creating a newspaper photo - it uses patterns of dots to create the illusion
    of more colors than are actually present. When you have a limited color palette (like only
    black and white), dithering arranges these limited colors in patterns that trick your eye
    into seeing intermediate shades.

    Think of it like pointillist paintings - up close you see individual dots, but from a distance
    they blend together to create smooth gradients and subtle color variations.

    This transform works with ANY number of channels - it processes each channel independently,
    whether you have a standard RGB image (3 channels), RGBA with transparency (4 channels),
    multispectral satellite imagery (dozens of channels), or even single-channel grayscale images.

    Args:
        method(str): Which dithering algorithm to use. Each has different characteristics:
            - "random": Adds random noise before quantization. Creates a grainy, film-like texture.
                       Good for artistic effects or simulating old photographs.
            - "ordered": Uses a repeating pattern (Bayer matrix) to decide which pixels to darken.
                        Creates distinctive crosshatch patterns. Fast and predictable.
                        Common in old computer graphics and newspaper printing.
            - "error_diffusion": Most sophisticated method. When a pixel is made darker or lighter
                                than it should be, the "error" is spread to neighboring pixels.
                                Creates the most natural-looking results. Like using a fine brush.
            Default: "error_diffusion"

        n_colors(int): How many different color levels to keep per channel. Must be between 2 and 256.
            - 2 = only black and white (or min/max values for each channel)
            - 4 = 4 levels of gray (or 4 levels per color channel)
            - 16 = 16 shades, creating a retro computer graphics look
            - 256 = full range, no reduction (but patterns still visible from dithering process)
            Lower values create more dramatic effects. Default: 2

        color_mode(str): How to handle color channels:
            - "per_channel": Each color channel (R, G, B, etc.) is dithered separately.
                           Maintains color relationships but each channel gets its own pattern.
                           Works with any number of channels.
            - "grayscale": First converts the image to grayscale (using standard luminance weights),
                          then applies dithering, then expands back to the original number of channels.
                          All color information is lost, but the dithering pattern is consistent across channels.
            Default: "grayscale"

        error_diffusion_algorithm(str): Used only in "error_diffusion" method. Which specific algorithm:
            - "floyd_steinberg": The classic, invented in 1976. Spreads error to 4 neighbors.
                               Good balance of quality and speed. Industry standard.
            - "jarvis": Jarvis-Judice-Ninke algorithm. Spreads error to 12 neighbors.
                       Higher quality but 3x slower than Floyd-Steinberg.
            - "stucki": Similar to Jarvis but with different weights. Also 12 neighbors.
            - "atkinson": Created by Bill Atkinson for original Macintosh. Only spreads 75% of
                         error, creating lighter images with more contrast.
            - "burkes": Spreads to 7 neighbors. Faster than Jarvis, better than Floyd-Steinberg.
            - "sierra": Spreads to 10 neighbors. Good quality, moderate speed.
            - "sierra_2row": Simplified Sierra using only 2 rows. Faster.
            - "sierra_lite": Minimal Sierra using only 3 neighbors. Very fast.
            Default: "floyd_steinberg"

        bayer_matrix_size(int): Used only in "ordered" method. The size of the repeating pattern (2, 4, 8, or 16).
            - 2x2: Very visible checkerboard pattern
            - 4x4: Standard, good balance
            - 8x8: Finer pattern, less visible
            - 16x16: Very fine pattern, almost noise-like
            Default: 4

        serpentine(bool): Used only in "error_diffusion" method. Whether to process rows in alternating directions
                   (left-to-right, then right-to-left). This can reduce visible "worm" artifacts
                   that sometimes appear as diagonal lines. Slightly slower. Default: False

        noise_range(tuple[float, float]): Used only in "random" method. How much random noise to add before
                    quantization. Larger range = more variation in the dithering pattern.
                    Range: (-1.0, 1.0). Default: (-0.5, 0.5)

        p(float): Probability of applying this transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> # Black and white dithering with Floyd-Steinberg
        >>> transform = A.Compose([
        ...     A.Dithering(
        ...         method="error_diffusion",
        ...         n_colors=2,
        ...         error_diffusion_algorithm="floyd_steinberg",
        ...         color_mode="grayscale",
        ...         p=1.0
        ...     )
        ... ])
        >>> transformed = transform(image=image)
        >>> dithered_image = transformed['image']  # Black and white dithered image
        >>>
        >>> # Ordered dithering with 16 colors per channel
        >>> transform = A.Compose([
        ...     A.Dithering(
        ...         method="ordered",
        ...         n_colors=16,
        ...         bayer_matrix_size=8,
        ...         color_mode="per_channel",
        ...         p=1.0
        ...     )
        ... ])
        >>> transformed = transform(image=image)
        >>> dithered_image = transformed['image']  # Reduced color depth with Bayer pattern
        >>>
        >>> # Random dithering
        >>> transform = A.Compose([
        ...     A.Dithering(
        ...         method="random",
        ...         n_colors=4,
        ...         noise_range=(-0.3, 0.3),
        ...         p=1.0
        ...     )
        ... ])
        >>> transformed = transform(image=image)
        >>> dithered_image = transformed['image']  # Noisy dithered appearance

    References:
        - Wikipedia: https://en.wikipedia.org/wiki/Dither
        - Floyd-Steinberg dithering: https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
        - Ordered dithering: https://en.wikipedia.org/wiki/Ordered_dithering
        - Error diffusion dithering: https://en.wikipedia.org/wiki/Error_diffusion

    """

    class InitSchema(BaseTransformInitSchema):
        method: Literal["random", "ordered", "error_diffusion"]
        n_colors: int = Field(ge=2, le=256)
        color_mode: Literal["grayscale", "per_channel"]
        error_diffusion_algorithm: Literal[
            "floyd_steinberg",
            "jarvis",
            "stucki",
            "atkinson",
            "burkes",
            "sierra",
            "sierra_2row",
            "sierra_lite",
        ]
        bayer_matrix_size: Literal[2, 4, 8, 16]
        serpentine: bool
        noise_range: Annotated[tuple[float, float], AfterValidator(check_range_bounds(-1, 1))]

    def __init__(
        self,
        method: Literal["random", "ordered", "error_diffusion"] = "error_diffusion",
        n_colors: int = 2,
        color_mode: Literal["grayscale", "per_channel"] = "grayscale",
        error_diffusion_algorithm: Literal[
            "floyd_steinberg",
            "jarvis",
            "stucki",
            "atkinson",
            "burkes",
            "sierra",
            "sierra_2row",
            "sierra_lite",
        ] = "floyd_steinberg",
        bayer_matrix_size: Literal[2, 4, 8, 16] = 4,
        serpentine: bool = False,
        noise_range: tuple[float, float] = (-0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.method = method
        self.n_colors = n_colors
        self.color_mode = color_mode
        self.error_diffusion_algorithm = error_diffusion_algorithm
        self.bayer_matrix_size = bayer_matrix_size
        self.serpentine = serpentine
        self.noise_range = noise_range

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        from albumentations.augmentations.pixel import dithering_functional as fdither

        return fdither.apply_dithering(
            img=img,
            method=self.method,
            n_colors=self.n_colors,
            color_mode=self.color_mode,
            error_diffusion_algorithm=self.error_diffusion_algorithm,
            matrix_size=self.bayer_matrix_size,
            serpentine=self.serpentine,
            noise_range=self.noise_range,
            random_generator=self.random_generator,
        )

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))


class Halftone(ImageOnlyTransform):
    """Halftone dot pattern (printing-style). Continuous tones become dots of varying size.
    Use for vintage or print-aesthetic augmentation.

    Simulates halftone printing: a grid of cells, each drawn as a filled circle whose
    size is proportional to mean luminance in that cell. Larger dots = brighter, smaller = darker.
    Optional blend with the original image controls strength.

    Args:
        dot_size_range (tuple[int, int]): Range for grid cell size in pixels. Larger =
            coarser pattern. Default: (4, 10).
        blend_range (tuple[float, float]): Blend with original: 0 = pure halftone, 1 = original.
            Default: (0.0, 0.5).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Mean luminance per grid cell drives dot radius; cell color from original image.
        - Dot size is proportional to luminance (bright → large dot, dark → small dot).

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> transform = A.Halftone(dot_size_range=(4, 8), blend_range=(0.0, 0.3), p=1.0)
        >>> result = transform(image=image)["image"]

    See Also:
        - FilmGrain: Luminance-dependent film grain for vintage texture.
        - Vignetting: Darkened edges for period or stylistic effect.

    """

    class InitSchema(BaseTransformInitSchema):
        dot_size_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(2, None)),
            AfterValidator(nondecreasing),
        ]
        blend_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        dot_size_range: tuple[int, int] = (4, 10),
        blend_range: tuple[float, float] = (0.0, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.dot_size_range = dot_size_range
        self.blend_range = blend_range

    def apply(
        self,
        img: ImageType,
        dot_size: int,
        blend: float,
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_halftone(img, dot_size, blend)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))

    def get_params(self) -> dict[str, Any]:
        dot_size = self.py_random.randint(*self.dot_size_range)
        blend = self.py_random.uniform(*self.blend_range)
        self.applied_config = {"dot_size_range": dot_size, "blend_range": blend}
        return {
            "dot_size": dot_size,
            "blend": blend,
        }


class LensFlare(ImageOnlyTransform):
    """Add lens flare: starburst rays and ghost reflections from a bright source.
    Use for outdoor or backlit robustness and optical-artifact simulation.

    A flare center is chosen in a configurable region; starburst rays and mirrored
    ghost circles are drawn toward the image center. Strength and blur are configurable.

    Args:
        flare_roi (tuple[float, float, float, float]): Region of interest for flare
            source placement as (x_min, y_min, x_max, y_max) in normalized [0, 1] coords.
            Default: (0, 0, 1, 0.5).
        num_ghosts_range (tuple[int, int]): Range for number of ghost reflections.
            Default: (3, 7).
        intensity_range (tuple[float, float]): Range for overall flare brightness.
            Default: (0.3, 0.7).
        num_rays_range (tuple[int, int]): Range for number of starburst rays.
            Default: (4, 8).
        bloom_range (tuple[float, float]): Range for bloom blur radius as fraction
            of image diagonal. Default: (0.01, 0.05).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - Ghost reflections lie along the line from flare source to image center.
        - Size decreases and color shifts with distance from the source.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.LensFlare(intensity_range=(0.3, 0.6), p=1.0)
        >>> result = transform(image=image)["image"]

    See Also:
        - AtmosphericFog: Depth-dependent fog via scattering.
        - RandomFog: Patch-based fog without depth.
        - RandomRain: Rain streaks for rainy conditions.
        - RandomSnow: Snow overlay for winter conditions.

    """

    class InitSchema(BaseTransformInitSchema):
        flare_roi: Annotated[
            tuple[float, float, float, float],
            AfterValidator(check_range_bounds(0, 1)),
        ]

        @field_validator("flare_roi")
        @classmethod
        def validate_flare_roi(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            """Ensure flare ROI (x_min, y_min, x_max, y_max) has valid bounds: x_min < x_max and
            y_min < y_max. Validation fails if bounds are invalid.
            """
            x_min, y_min, x_max, y_max = v
            if x_min >= x_max:
                msg = f"flare_roi x_min ({x_min}) must be less than x_max ({x_max})"
                raise ValueError(msg)
            if y_min >= y_max:
                msg = f"flare_roi y_min ({y_min}) must be less than y_max ({y_max})"
                raise ValueError(msg)
            return v

        num_ghosts_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        num_rays_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(2, None)),
            AfterValidator(nondecreasing),
        ]
        bloom_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5),
        num_ghosts_range: tuple[int, int] = (3, 7),
        intensity_range: tuple[float, float] = (0.3, 0.7),
        num_rays_range: tuple[int, int] = (4, 8),
        bloom_range: tuple[float, float] = (0.01, 0.05),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.flare_roi = flare_roi
        self.num_ghosts_range = num_ghosts_range
        self.intensity_range = intensity_range
        self.num_rays_range = num_rays_range
        self.bloom_range = bloom_range

    def apply(
        self,
        img: ImageType,
        flare_center: tuple[int, int],
        ghosts: list[tuple[int, int, int, float]],
        starburst_angles: np.ndarray,
        starburst_intensity: float,
        bloom_radius: int,
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)
        return fpixel.apply_lens_flare(
            img,
            flare_center,
            ghosts,
            starburst_angles,
            starburst_intensity,
            bloom_radius,
        )

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        x_min, y_min, x_max, y_max = self.flare_roi

        fx_lo = min(int(x_min * width), width - 1)
        fx_hi = min(int(x_max * width), width - 1)
        fx = self.py_random.randint(fx_lo, max(fx_lo, fx_hi))
        fy_lo = min(int(y_min * height), height - 1)
        fy_hi = min(int(y_max * height), height - 1)
        fy = self.py_random.randint(fy_lo, max(fy_lo, fy_hi))

        intensity = self.py_random.uniform(*self.intensity_range)
        num_rays = self.py_random.randint(*self.num_rays_range)
        num_ghosts = self.py_random.randint(*self.num_ghosts_range)

        base_angle = self.py_random.uniform(0, math.pi / num_rays) if num_rays > 0 else 0
        starburst_angles = np.array(
            [base_angle + i * math.pi / num_rays for i in range(num_rays * 2)],
            dtype=np.float32,
        )

        cx, cy = width // 2, height // 2
        ghosts = []
        for i in range(num_ghosts):
            t = (i + 1) / (num_ghosts + 1)
            gx = int(fx + (cx - fx) * 2 * t)
            gy = int(fy + (cy - fy) * 2 * t)
            gradius = max(2, int(min(height, width) * 0.02 * (1.0 - t * 0.5)))
            galpha = intensity * (1.0 - t * 0.6)
            ghosts.append((gx, gy, gradius, galpha))

        diag = math.sqrt(height**2 + width**2)
        bloom_frac = self.py_random.uniform(*self.bloom_range)
        bloom_radius = max(1, int(diag * bloom_frac)) | 1

        self.applied_config = {
            "intensity_range": intensity,
            "num_rays_range": num_rays,
            "num_ghosts_range": num_ghosts,
            "bloom_range": bloom_frac,
        }
        return {
            "flare_center": (fx, fy),
            "ghosts": ghosts,
            "starburst_angles": starburst_angles,
            "starburst_intensity": intensity,
            "bloom_radius": bloom_radius,
        }
