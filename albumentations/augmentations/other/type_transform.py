"""Transforms for type conversion between float and other data types.

This module provides transform classes for converting image data types, primarily
for converting between floating point and integer representations. These transforms
are useful for preprocessing before neural network input (ToFloat) and for converting
network outputs back to standard image formats (FromFloat).
"""

from typing import Any, Literal, Self, cast, overload

import numpy as np
from albucore import get_max_value
from numpy.typing import NDArray
from pydantic import (
    model_validator,
)

from albumentations.augmentations.other import type_transform_functional as ftype
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import ImageType

__all__ = [
    "FromFloat",
    "ToFloat",
]


class ToFloat(ImageOnlyTransform):
    """Convert byte intensities to floating values using a chosen maximum, while preserving the values and precision
    of floating inputs.

    Args:
        max_value (float | None): The maximum possible input value. If None, the transform
            will try to infer the maximum value by inspecting the data type of the input image:
            - uint8: 255
            - float16, float32: 1.0 (floating inputs are preserved)
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, images, volume, volumes

    Image types:
        uint8, float16, float32

    Returns:
        np.ndarray | torch.Tensor: Converted image in the input container.

    Note:
        - Floating inputs retain their dtype and values, including NumPy float16.
        - Uint8 values are divided by max_value into float32 without clipping.
        - This transform is often used as a preprocessing step before applying other transformations
          or feeding the image into a neural network.

    Raises:
        TypeError: If the input image data type is not supported.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Convert uint8 image to float
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ToFloat(max_value=None)
        >>> float_image = transform(image=image)['image']
        >>> assert float_image.dtype == np.float32
        >>> assert 0 <= float_image.min() <= float_image.max() <= 1.0
        >>>
    See Also:
        FromFloat: The inverse operation, converting from float back to the original data type.

    """

    class InitSchema(BaseTransformInitSchema):
        max_value: float | None

    def __init__(
        self,
        max_value: float | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.max_value = max_value

    @overload
    def apply(self, img: NDArray[np.float16], **params: Any) -> NDArray[np.float16]: ...

    @overload
    def apply(self, img: NDArray[np.uint8] | NDArray[np.float32], **params: Any) -> NDArray[np.float32]: ...

    def apply(self, img: ImageType, **params: Any) -> NDArray[np.float16] | NDArray[np.float32]:
        return ftype.to_float(img, self.max_value)

    @overload
    def apply_to_images(self, images: NDArray[np.float16], **params: Any) -> NDArray[np.float16]: ...

    @overload
    def apply_to_images(
        self, images: NDArray[np.uint8] | NDArray[np.float32], **params: Any
    ) -> NDArray[np.float32]: ...

    def apply_to_images(self, images: ImageType, **params: Any) -> NDArray[np.float16] | NDArray[np.float32]:
        return ftype.to_float(images, self.max_value)


class FromFloat(ImageOnlyTransform):
    """Convert image from float [0, 1] to discrete type (e.g. uint8 [0, 255]). Inverse of
    ToFloat. max_value and dtype control scaling and output type.

    This transform is designed to convert images from a normalized floating-point representation
    (typically with values in the range [0, 1]) to other data types, scaling the values appropriately.

    Args:
        dtype (str): The desired output data type. Only 'uint8' is supported. Default: 'uint8'.
        max_value (float | None): The maximum value for the output dtype. If None, 255 for uint8.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, images, volume, volumes

    Image types:
        float16, float32

    Note:
        - This is the inverse transform for ToFloat.
        - Input images are expected to be in floating point format with values in the range [0, 1].
        - Output is uint8 with values scaled to [0, 255].

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> transform = A.FromFloat(dtype='uint8', max_value=None, p=1.0)
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)  # Float image in [0, 1] range
        >>> result = transform(image=image)
        >>> uint8_image = result['image']
        >>> assert uint8_image.dtype == np.uint8
        >>> assert uint8_image.min() >= 0 and uint8_image.max() <= 255

    """

    class InitSchema(BaseTransformInitSchema):
        dtype: Literal["uint8"]
        max_value: float | None

        @model_validator(mode="after")
        def _update_max_value(self) -> Self:
            if self.max_value is None:
                self.max_value = get_max_value(np.dtype(self.dtype))

            return self

    def __init__(
        self,
        dtype: Literal["uint8"] = "uint8",
        max_value: float | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.dtype = dtype
        self.max_value = max_value

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return ftype.from_float(img, cast("float", self.max_value))

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)
