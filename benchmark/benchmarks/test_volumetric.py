"""Volumetric transform benchmarks."""

from __future__ import annotations

import albumentations
from albumentations.augmentations.transforms3d import functional as f3d
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


class TimeMotionArtifact:
    """Measure MRI motion reconstruction across channels, dtype, volume size, and event count."""

    params = (tuple(VOLUME_SIZES), (1, 3, 5), tuple(DTYPES), (1, 4))
    param_names = ("size", "channels", "dtype", "events")

    def setup(self, size: str, channels: int, dtype: str, events: int) -> None:
        self.volume = make_volume(size, channels, DTYPES[dtype])
        self.motion = albumentations.Compose(
            [albumentations.MotionArtifact(num_events_range=(events, events), p=1)],
            seed=137,
            strict=True,
        )
        capture = albumentations.ReplayCompose(
            [albumentations.MotionArtifact(num_events_range=(events, events), p=1)],
        )
        capture.set_random_seed(137)
        self.motion_params = capture(volume=self.volume)["replay"]["transforms"][0]["params"]["params"]

    def time_motion_artifact(self, size: str, channels: int, dtype: str, events: int) -> None:
        self.motion(volume=self.volume)

    def time_motion_kernel(self, size: str, channels: int, dtype: str, events: int) -> None:
        f3d.motion_artifact(self.volume, self.motion_params["matrices"], self.motion_params["boundaries"], 2, 1)

    def peakmem_motion_artifact(self, size: str, channels: int, dtype: str, events: int) -> None:
        self.motion(volume=self.volume)


class TimeGhostingArtifact:
    """Measure the selected-axis spectral filter and public volume route."""

    params = (tuple(VOLUME_SIZES), (1, 3, 5), tuple(DTYPES), (0, 1, 2))
    param_names = ("size", "channels", "dtype", "axis")

    def setup(self, size: str, channels: int, dtype: str, axis: int) -> None:
        self.volume = make_volume(size, channels, DTYPES[dtype])
        self.pipeline = albumentations.Compose(
            [
                albumentations.GhostingArtifact(
                    num_ghosts_range=(2, 2), intensity_range=(0.4, 0.4), axis=axis, restore_range=(0.02, 0.02), p=1
                ),
            ],
            seed=137,
            strict=True,
        )

    def time_kernel(self, size: str, channels: int, dtype: str, axis: int) -> None:
        f3d.ghosting_artifact(self.volume, 2, 0.4, axis, 0.02)

    def time_compose(self, size: str, channels: int, dtype: str, axis: int) -> None:
        self.pipeline(volume=self.volume)

    def peakmem_compose(self, size: str, channels: int, dtype: str, axis: int) -> None:
        self.pipeline(volume=self.volume)
