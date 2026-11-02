from typing import Any

import numpy as np
import torch
from albucore import power
from numpy.typing import NDArray

from albumentations.core.transforms_interface import BasicTransform, NoOp
from albumentations.core.type_definitions import ImageType
from albumentations.pytorch.transforms import ToTensor3D, ToTensorV2

image: NDArray[np.float32] = np.zeros((3, 5, 1), dtype=np.float32)
half: NDArray[np.float16] = np.zeros((3, 5, 1), dtype=np.float16)
half_out = np.empty_like(half)
tensor_image: torch.Tensor = ToTensorV2().apply(image)
tensor_volume: torch.Tensor = ToTensor3D().apply_to_volume(half[None, ...])
half_result: NDArray[np.float16] = power(half, 2, out=half_out)
tensor_result: torch.Tensor = power(tensor_image, 2, layout="CHW")
same_tensor: torch.Tensor = NoOp().apply_to_volumes(tensor_image[None, None, ...])
same_half: NDArray[np.float16] = NoOp().apply_to_volumes(half[None, None, ...])
transform: BasicTransform = ToTensorV2()
base_result: ImageType | torch.Tensor = transform.apply(image)


class FloatArrayOutput(BasicTransform):
    def apply(self, img: ImageType, *args: Any, **params: Any) -> NDArray[np.float32]:
        return img.astype(np.float32)
