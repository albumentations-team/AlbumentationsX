"""PyTorch tensor transform benchmarks.

These benchmarks live outside the default ASV benchmark directory so Tensor
conversion evidence remains separately reviewable from the general catalog.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

import albumentations

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

benchmark_common = importlib.import_module("benchmarks.common")
CHANNELS = benchmark_common.CHANNELS
DTYPES = benchmark_common.DTYPES
SIZES = benchmark_common.SIZES
VOLUME_SIZES = benchmark_common.VOLUME_SIZES
dtype_from_name = benchmark_common.dtype_from_name
make_batch = benchmark_common.make_batch
make_image = benchmark_common.make_image
make_mask = benchmark_common.make_mask
make_mask3d = benchmark_common.make_mask3d
make_masks = benchmark_common.make_masks
make_volume = benchmark_common.make_volume

IMAGE_CASES = tuple(
    f"{size_name}|{channels}|{dtype_name}" for size_name in SIZES for channels in CHANNELS for dtype_name in DTYPES
)
VOLUME_CASES = tuple(
    f"{size_name}|{channels}|{dtype_name}" for size_name in VOLUME_SIZES for channels in (1, 3) for dtype_name in DTYPES
)
TENSOR_NATIVE_IMAGE_CASES = tuple(
    f"{size_name}|{channels}|{dtype_name}" for size_name in SIZES for channels in (1, 3) for dtype_name in DTYPES
)
TENSOR_NATIVE_VOLUME_CASES = tuple(
    f"{size_name}|{channels}|{dtype_name}"
    for size_name in VOLUME_SIZES
    for channels in (1, 3, 5)
    for dtype_name in DTYPES
)


def _parse_image_case(case_id: str) -> tuple[str, int, str]:
    size_name, channels, dtype_name = case_id.split("|")
    return size_name, int(channels), dtype_name


class TimeToTensorV2:
    """Benchmark 2D PyTorch tensor conversion over image, batch, and mask paths."""

    params = (IMAGE_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        size_name, channels, dtype_name = _parse_image_case(case_id)
        dtype = dtype_from_name(dtype_name)
        self.image = make_image(size_name, channels, dtype)
        self.images = make_batch(size_name, channels, batch_size=8).astype(dtype, copy=False)
        self.mask = make_mask(size_name)
        self.masks = make_masks(size_name, count=8)
        self.transform = albumentations.Compose([albumentations.ToTensorV2(p=1.0)], strict=True)
        self.transpose_mask = albumentations.Compose(
            [albumentations.ToTensorV2(transpose_mask=True, p=1.0)],
            strict=True,
        )

    def time_image(self, case_id: str) -> None:
        self.transform(image=self.image)

    def time_images(self, case_id: str) -> None:
        self.transform(images=self.images)

    def time_image_and_mask(self, case_id: str) -> None:
        self.transform(image=self.image, mask=self.mask)

    def time_images_and_masks(self, case_id: str) -> None:
        self.transpose_mask(images=self.images, masks=self.masks)

    def peakmem_images(self, case_id: str) -> None:
        self.transform(images=self.images)


class TimeToTensor3D:
    """Benchmark 3D PyTorch tensor conversion over volume and mask3d paths."""

    params = (VOLUME_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        size_name, channels, dtype_name = _parse_image_case(case_id)
        self.volume = make_volume(size_name, channels, dtype_from_name(dtype_name))
        self.mask3d = make_mask3d(size_name)
        self.transform = albumentations.Compose([albumentations.ToTensor3D(p=1.0)], strict=True)

    def time_volume(self, case_id: str) -> None:
        self.transform(volume=self.volume)

    def time_volume_and_mask3d(self, case_id: str) -> None:
        self.transform(volume=self.volume, mask3d=self.mask3d)

    def peakmem_volume(self, case_id: str) -> None:
        self.transform(volume=self.volume)


class TimeTensorNativeTranspose:
    """Benchmark the accepted Tensor `Transpose(image=...)` capability.

    The route is intentionally bounded to the accepted C=1 and C=3 image
    contract. `images` and `volume` Tensor targets remain unsupported.
    """

    params = (TENSOR_NATIVE_IMAGE_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        size_name, channels, dtype_name = _parse_image_case(case_id)
        self.image = make_image(size_name, channels, dtype_from_name(dtype_name))
        self.tensor = torch.from_numpy(np.ascontiguousarray(self.image.transpose(2, 0, 1)))
        self.numpy_direct = albumentations.Compose([albumentations.Transpose(p=1.0)], strict=True)
        self.numpy_model_ready = albumentations.Compose(
            [albumentations.Transpose(p=1.0), albumentations.ToTensorV2(p=1.0)],
            strict=True,
        )
        self.tensor_direct = albumentations.Compose([albumentations.Transpose(p=1.0)], strict=True)

    def time_numpy_direct(self, case_id: str) -> None:
        self.numpy_direct(image=self.image)

    def time_numpy_model_ready(self, case_id: str) -> None:
        self.numpy_model_ready(image=self.image)

    def time_tensor_direct(self, case_id: str) -> None:
        self.tensor_direct(image=self.tensor)

    def peakmem_tensor_direct(self, case_id: str) -> None:
        self.tensor_direct(image=self.tensor)


class TimeTensorNativeAnisotropy3D:
    """Benchmark the accepted Tensor `Anisotropy3D(volume=...)` capability.

    The direct Tensor route uses C,D,H,W volumes. Its NumPy comparison paths
    use the matching D,H,W,C volume and the normal `ToTensor3D` handoff.
    """

    params = (TENSOR_NATIVE_VOLUME_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        size_name, channels, dtype_name = _parse_image_case(case_id)
        self.volume = make_volume(size_name, channels, dtype_from_name(dtype_name))
        self.mask3d = make_mask3d(size_name)
        self.tensor = torch.from_numpy(np.ascontiguousarray(self.volume.transpose(3, 0, 1, 2)))
        self.tensor_mask3d = torch.from_numpy(self.mask3d)
        anisotropy = albumentations.Anisotropy3D(
            axes=(0, 2),
            num_axes_range=(2, 2),
            downscale_factor_range=(2.0, 2.0),
            p=1.0,
        )
        self.numpy_direct = albumentations.Compose([anisotropy], seed=137, strict=True)
        self.numpy_model_ready = albumentations.Compose(
            [anisotropy, albumentations.ToTensor3D(p=1.0)],
            seed=137,
            strict=True,
        )
        self.tensor_direct = albumentations.Compose(
            [
                albumentations.Anisotropy3D(
                    axes=(0, 2),
                    num_axes_range=(2, 2),
                    downscale_factor_range=(2.0, 2.0),
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )

    def time_numpy_direct(self, case_id: str) -> None:
        self.numpy_direct(volume=self.volume)

    def time_numpy_model_ready(self, case_id: str) -> None:
        self.numpy_model_ready(volume=self.volume, mask3d=self.mask3d)

    def time_tensor_direct(self, case_id: str) -> None:
        self.tensor_direct(volume=self.tensor)

    def time_tensor_and_mask3d(self, case_id: str) -> None:
        self.tensor_direct(volume=self.tensor, mask3d=self.tensor_mask3d)

    def peakmem_tensor_direct(self, case_id: str) -> None:
        self.tensor_direct(volume=self.tensor)
