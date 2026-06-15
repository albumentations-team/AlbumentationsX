"""PyTorch tensor transform benchmarks.

These benchmarks live outside the default ASV benchmark directory so the
headless benchmark suite remains importable without the optional PyTorch
dependency.
"""

from __future__ import annotations

from benchmarks.common import (
    CHANNELS,
    DTYPES,
    SIZES,
    VOLUME_SIZES,
    dtype_from_name,
    make_batch,
    make_image,
    make_mask,
    make_mask3d,
    make_masks,
    make_volume,
)

import albumentations

IMAGE_CASES = tuple(
    f"{size_name}|{channels}|{dtype_name}" for size_name in SIZES for channels in CHANNELS for dtype_name in DTYPES
)
VOLUME_CASES = tuple(
    f"{size_name}|{channels}|{dtype_name}" for size_name in VOLUME_SIZES for channels in (1, 3) for dtype_name in DTYPES
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
