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

    def peakmem_pad_if_needed3d(self) -> None:
        self.pad(volume=self.volume)


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


class TimeAffine3D:
    """Benchmark true-3D affine resampling through its public Compose route."""

    params = (tuple(VOLUME_SIZES), (1, 3, 5), tuple(DTYPES))
    param_names = ("size", "channels", "dtype")

    def setup(self, size: str, channels: int, dtype: str) -> None:
        self.volume = make_volume(size, channels, DTYPES[dtype])
        transform_kwargs = {
            "rotate_range": {"x": (3.0, 3.0), "y": (-2.0, -2.0), "z": (5.0, 5.0)},
            "scale_range": {"x": (1.05, 1.05), "y": (0.95, 0.95), "z": (1.0, 1.0)},
            "translate_percent_range": {"x": (0.02, 0.02), "y": (-0.02, -0.02), "z": (0.0, 0.0)},
            "p": 1.0,
        }
        self.affine = albumentations.Compose(
            [albumentations.Affine3D(**transform_kwargs)],
            strict=True,
        )

    def time_affine3d(self, size: str, channels: int, dtype: str) -> None:
        self.affine(volume=self.volume)

    def peakmem_affine3d(self, size: str, channels: int, dtype: str) -> None:
        self.affine(volume=self.volume)
