from typing import Any

import numpy as np
import torch
from albucore import power
from numpy.typing import NDArray

from albumentations.core.transforms_interface import BasicTransform, NoOp
from albumentations.pytorch.transforms import ToTensorV2

image: NDArray[np.float32] = np.zeros((3, 5, 1), dtype=np.float32)
half: NDArray[np.float16] = np.zeros((3, 5, 1), dtype=np.float16)
tensor = torch.zeros((1, 3, 5))
invalid_numpy_output: NDArray[np.float32] = ToTensorV2().apply(image)  # expected-error
invalid_tensor_input = ToTensorV2().apply(tensor)  # expected-error
invalid_core_output = power(half, 2, out=image)  # expected-error
invalid_layout = power(tensor, 2)  # expected-error
invalid_noop_container: NDArray[np.float32] = NoOp().apply_to_volumes(tensor[None, None, ...])  # expected-error
invalid_noop_dtype: NDArray[np.float32] = NoOp().apply_to_volumes(half[None, None, ...])  # expected-error


class ScalarOutput(BasicTransform):
    def apply(  # expected-error
        self, img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], *args: Any, **params: Any
    ) -> int:
        return 137
