"""Geometric transform benchmarks."""

from __future__ import annotations

import albumentations
from benchmarks.common import IMAGE_PARAMS, make_image


class TimeGeometricTransforms:
    """Benchmark representative geometric transforms."""

    params = IMAGE_PARAMS
    param_names = ("size_name", "channels")

    def setup(self, size_name: str, channels: int) -> None:
        self.image = make_image(size_name, channels)
        self.transforms = {
            "flip": albumentations.Compose([albumentations.HorizontalFlip(p=1.0)], strict=True),
            "resize": albumentations.Compose([albumentations.Resize(height=128, width=128, p=1.0)], strict=True),
            "pad": albumentations.Compose(
                [albumentations.PadIfNeeded(min_height=1200, min_width=1200, p=1.0)],
                strict=True,
            ),
            "affine": albumentations.Compose(
                [albumentations.Affine(scale=(1.05, 1.05), rotate=(3, 3), p=1.0)],
                strict=True,
            ),
        }

    def time_horizontal_flip(self, size_name: str, channels: int) -> None:
        self.transforms["flip"](image=self.image)

    def time_resize(self, size_name: str, channels: int) -> None:
        self.transforms["resize"](image=self.image)

    def time_pad_if_needed(self, size_name: str, channels: int) -> None:
        self.transforms["pad"](image=self.image)

    def time_affine(self, size_name: str, channels: int) -> None:
        self.transforms["affine"](image=self.image)
