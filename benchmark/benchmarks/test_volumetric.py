"""Volumetric transform benchmarks."""

from __future__ import annotations

import albumentations
from benchmarks.common import DTYPES, VOLUME_SIZES, make_volume


class TimeVolumetricTransforms:
    """Benchmark representative volumetric transforms."""

    def setup(self) -> None:
        self.volume = make_volume()
        self.center_crop = albumentations.Compose([albumentations.CenterCrop3D(size=(4, 48, 48), p=1.0)], strict=True)
        self.pad = albumentations.Compose([albumentations.PadIfNeeded3D(min_zyx=(10, 72, 72), p=1.0)], strict=True)

    def time_center_crop3d(self) -> None:
        self.center_crop(volume=self.volume)

    def time_pad_if_needed3d(self) -> None:
        self.pad(volume=self.volume)

    def peakmem_center_crop3d(self) -> None:
        self.center_crop(volume=self.volume)


class TimeGaussianBlur3D:
    """Benchmark true-3D Gaussian blur through its public Compose route."""

    params = (tuple(VOLUME_SIZES), (1, 3, 5), tuple(DTYPES))
    param_names = ("size", "channels", "dtype")

    def setup(self, size: str, channels: int, dtype: str) -> None:
        self.volume = make_volume(size, channels, DTYPES[dtype])
        self.gaussian_blur = albumentations.Compose(
            [
                albumentations.GaussianBlur(
                    blur_range=(0, 0),
                    sigma_range=(1.25, 1.25),
                    volume_mode="3d",
                    sigma_z_range=(0.75, 0.75),
                    p=1.0,
                ),
            ],
            strict=True,
        )

    def time_gaussian_blur3d(self, size: str, channels: int, dtype: str) -> None:
        self.gaussian_blur(volume=self.volume)

    def peakmem_gaussian_blur3d(self, size: str, channels: int, dtype: str) -> None:
        self.gaussian_blur(volume=self.volume)
