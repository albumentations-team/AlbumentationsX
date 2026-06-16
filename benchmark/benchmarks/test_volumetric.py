"""Volumetric transform benchmarks."""

from __future__ import annotations

import albumentations
from benchmarks.common import make_volume


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
