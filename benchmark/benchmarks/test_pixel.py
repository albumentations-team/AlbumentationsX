"""Pixel transform benchmarks."""

from __future__ import annotations

import albumentations
from benchmarks.common import IMAGE_PARAMS, make_image


class TimePixelTransforms:
    """Benchmark representative pixel transforms."""

    params = IMAGE_PARAMS
    param_names = ("size_name", "channels")

    def setup(self, size_name: str, channels: int) -> None:
        self.image = make_image(size_name, channels)
        self.transforms = {
            "brightness": albumentations.Compose(
                [albumentations.RandomBrightnessContrast(p=1.0)],
                seed=137,
                strict=True,
            ),
            "blur": albumentations.Compose([albumentations.GaussianBlur(blur_range=(3, 3), p=1.0)], strict=True),
            "normalize": albumentations.Compose([albumentations.Normalize(p=1.0)], strict=True),
        }

    def time_random_brightness_contrast(self, size_name: str, channels: int) -> None:
        self.transforms["brightness"](image=self.image)

    def time_gaussian_blur(self, size_name: str, channels: int) -> None:
        self.transforms["blur"](image=self.image)

    def time_normalize(self, size_name: str, channels: int) -> None:
        self.transforms["normalize"](image=self.image)
