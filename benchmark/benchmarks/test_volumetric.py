"""Volumetric transform benchmarks."""

from __future__ import annotations

import numpy as np

import albumentations
from albumentations.augmentations.pixel import functional as fpixel
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


class TimeAdditiveNoise3D:
    """Benchmark all target-aware volumetric AdditiveNoise distributions through the public Compose route for
    per-pixel and channel-shared modes.
    """

    params = (
        ("gaussian", "uniform", "laplace", "beta"),
        ("per_pixel", "shared"),
        tuple(VOLUME_SIZES),
        (1, 3, 5),
        tuple(DTYPES),
    )
    param_names = ("noise_type", "mode", "size", "channels", "dtype")

    def setup(self, noise_type: str, mode: str, size: str, channels: int, dtype: str) -> None:
        self.volume = make_volume(size, channels, DTYPES[dtype])
        noise_params = {
            "gaussian": {"mean_range": (0.0, 0.0), "std_range": (0.05, 0.05)},
            "uniform": {"ranges": [(-0.05, 0.05)]},
            "laplace": {"mean_range": (0.0, 0.0), "scale_range": (0.05, 0.05)},
            "beta": {"alpha_range": (0.5, 1.5), "beta_range": (0.5, 1.5), "scale_range": (0.1, 0.3)},
        }[noise_type]
        self.additive_noise = albumentations.Compose(
            [
                albumentations.AdditiveNoise(
                    noise_type=noise_type,
                    spatial_mode=mode,
                    noise_params=noise_params,
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )

    def time_additive_noise3d(self, noise_type: str, mode: str, size: str, channels: int, dtype: str) -> None:
        self.additive_noise(volume=self.volume)

    def peakmem_additive_noise3d(self, noise_type: str, mode: str, size: str, channels: int, dtype: str) -> None:
        self.additive_noise(volume=self.volume)


class TimeAdditiveNoisePerChannelUniform3D:
    """Benchmark the per-channel uniform-range OpenCV path for target-aware volumetric AdditiveNoise through public
    Compose execution.
    """

    params = (tuple(VOLUME_SIZES), (3, 5), tuple(DTYPES))
    param_names = ("size", "channels", "dtype")

    def setup(self, size: str, channels: int, dtype: str) -> None:
        self.volume = make_volume(size, channels, DTYPES[dtype])
        ranges = [(-0.05 + 0.01 * channel, 0.05 + 0.01 * channel) for channel in range(channels)]
        self.additive_noise = albumentations.Compose(
            [
                albumentations.AdditiveNoise(
                    noise_type="uniform",
                    spatial_mode="per_pixel",
                    noise_params={"ranges": ranges},
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )

    def time_additive_noise_per_channel_uniform3d(self, size: str, channels: int, dtype: str) -> None:
        self.additive_noise(volume=self.volume)

    def peakmem_additive_noise_per_channel_uniform3d(self, size: str, channels: int, dtype: str) -> None:
        self.additive_noise(volume=self.volume)


class TimeGenerateVolumetricNoise:
    """Benchmark direct seeded volumetric noise generation for Gaussian, scalar uniform, and per-channel uniform
    ranges across the public volume size and channel matrix.
    """

    params = (("gaussian", "uniform", "uniform_per_channel"), tuple(VOLUME_SIZES), (1, 3, 5))
    param_names = ("noise_case", "size", "channels")

    def setup(self, noise_case: str, size: str, channels: int) -> None:
        self.shape = (*VOLUME_SIZES[size], channels)
        self.noise_type = "uniform" if noise_case.startswith("uniform") else "gaussian"
        self.noise_params = (
            {"mean_range": (0.0, 0.0), "std_range": (0.05, 0.05)}
            if noise_case == "gaussian"
            else {
                "ranges": (
                    [(-0.05, 0.05)]
                    if noise_case == "uniform"
                    else [(-0.05 + 0.01 * channel, 0.05 + 0.01 * channel) for channel in range(channels)]
                ),
            }
        )

    def time_generate_volumetric_noise(self, noise_case: str, size: str, channels: int) -> None:
        fpixel.generate_volumetric_noise(
            noise_type=self.noise_type,
            shape=self.shape,
            params=self.noise_params,
            max_value=255.0,
            random_generator=np.random.default_rng(137),
        )


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
