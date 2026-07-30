"""Batch-route benchmarks for image, mask, and volume targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

import albumentations
from benchmarks.common import (
    CHANNELS,
    DTYPES,
    VOLUME_SIZES,
    dtype_from_name,
    make_image,
    make_mask3d,
    make_masks,
    make_volume,
)

Factory = Callable[[], object]

BATCH_SIZES = (4, 8)
VOLUME_BATCH_SIZES = (2, 4)
SPATTER_BATCH_SIZES = (2, 4, 8, 16)
SPATTER_SIZES = ("small", "medium", "large")


@dataclass(frozen=True)
class BatchSpec:
    """Batch benchmark case metadata."""

    factory: Factory
    channels: tuple[int, ...] = CHANNELS
    dtypes: tuple[str, ...] = tuple(DTYPES)
    sizes: tuple[str, ...] = ("small", "medium")
    batch_sizes: tuple[int, ...] = BATCH_SIZES


IMAGE_BATCH_TRANSFORMS: Mapping[str, BatchSpec] = {
    "channel_dropout": BatchSpec(lambda: albumentations.ChannelDropout(p=1.0), channels=(3, 5)),
    "coarse_dropout": BatchSpec(lambda: albumentations.CoarseDropout(p=1.0)),
    "gauss_noise": BatchSpec(lambda: albumentations.GaussNoise(p=1.0)),
    "horizontal_flip": BatchSpec(lambda: albumentations.HorizontalFlip(p=1.0)),
    "normalize": BatchSpec(lambda: albumentations.Normalize(p=1.0)),
    "random_brightness_contrast": BatchSpec(lambda: albumentations.RandomBrightnessContrast(p=1.0)),
    "resize": BatchSpec(lambda: albumentations.Resize(height=128, width=128, p=1.0)),
    "spatter_mud": BatchSpec(
        lambda: albumentations.Spatter(mode="mud", p=1.0),
        channels=(3,),
        sizes=SPATTER_SIZES,
        batch_sizes=SPATTER_BATCH_SIZES,
    ),
    "spatter_rain": BatchSpec(
        lambda: albumentations.Spatter(mode="rain", p=1.0),
        channels=(3,),
        sizes=SPATTER_SIZES,
        batch_sizes=SPATTER_BATCH_SIZES,
    ),
}

MASK_BATCH_TRANSFORMS: Mapping[str, BatchSpec] = {
    "coarse_dropout": BatchSpec(lambda: albumentations.CoarseDropout(p=1.0)),
    "horizontal_flip": BatchSpec(lambda: albumentations.HorizontalFlip(p=1.0)),
    "resize": BatchSpec(lambda: albumentations.Resize(height=128, width=128, p=1.0)),
}

VOLUME_BATCH_TRANSFORMS: Mapping[str, BatchSpec] = {
    "d4": BatchSpec(
        lambda: albumentations.D4(p=1.0),
        channels=(1, 3),
        sizes=tuple(VOLUME_SIZES),
        batch_sizes=VOLUME_BATCH_SIZES,
    ),
    "horizontal_flip": BatchSpec(
        lambda: albumentations.HorizontalFlip(p=1.0),
        channels=(1, 3),
        sizes=tuple(VOLUME_SIZES),
        batch_sizes=VOLUME_BATCH_SIZES,
    ),
    "random_rotate90": BatchSpec(
        lambda: albumentations.RandomRotate90(p=1.0),
        channels=(1, 3),
        sizes=tuple(VOLUME_SIZES),
        batch_sizes=VOLUME_BATCH_SIZES,
    ),
    "transpose": BatchSpec(
        lambda: albumentations.Transpose(p=1.0),
        channels=(1, 3),
        sizes=tuple(VOLUME_SIZES),
        batch_sizes=VOLUME_BATCH_SIZES,
    ),
    "vertical_flip": BatchSpec(
        lambda: albumentations.VerticalFlip(p=1.0),
        channels=(1, 3),
        sizes=tuple(VOLUME_SIZES),
        batch_sizes=VOLUME_BATCH_SIZES,
    ),
}


def _cases(transforms: Mapping[str, BatchSpec], target_route: str) -> tuple[str, ...]:
    return tuple(
        f"{name}|{target_route}|{size_name}|{channel_count}|{dtype_name}|{batch_size}"
        for name, spec in transforms.items()
        for size_name in spec.sizes
        for channel_count in spec.channels
        for dtype_name in spec.dtypes
        for batch_size in spec.batch_sizes
    )


IMAGE_BATCH_CASES = _cases(IMAGE_BATCH_TRANSFORMS, "images")
SPATTER_DIRECT_CASES = tuple(
    case_id.replace("|images|", "|direct_images|", 1)
    for case_id in IMAGE_BATCH_CASES
    if case_id.startswith(("spatter_mud|", "spatter_rain|"))
)
MASK_BATCH_CASES = _cases(MASK_BATCH_TRANSFORMS, "images_and_masks")
VOLUME_BATCH_CASES = _cases(VOLUME_BATCH_TRANSFORMS, "volumes_and_masks3d")


def _parse_batch_case(case_id: str) -> tuple[str, str, str, int, str, int]:
    name, target_route, size_name, channels, dtype_name, batch_size = case_id.split("|")
    return name, target_route, size_name, int(channels), dtype_name, int(batch_size)


def _make_image_batch(
    size_name: str,
    channels: int,
    dtype: type[np.generic],
    batch_size: int,
) -> np.ndarray:
    return np.stack([make_image(size_name, channels, dtype) for _ in range(batch_size)], axis=0)


def _make_volume_batch(
    size_name: str,
    channels: int,
    dtype: type[np.generic],
    batch_size: int,
) -> np.ndarray:
    return np.stack([make_volume(size_name, channels, dtype) for _ in range(batch_size)], axis=0)


def _make_masks3d_batch(size_name: str, batch_size: int) -> np.ndarray:
    return np.stack([make_mask3d(size_name) for _ in range(batch_size)], axis=0)


class TimeImageBatchMatrix:
    """Benchmark public `images` batch routes over size, channel, dtype, and batch-size variants."""

    params = (IMAGE_BATCH_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        self.transform = albumentations.Compose([IMAGE_BATCH_TRANSFORMS[name].factory()], seed=137, strict=True)
        self.data = {"images": _make_image_batch(size_name, channels, dtype_from_name(dtype_name), batch_size)}

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)


class TimeSpatterDirectBatchMatrix:
    """Benchmark Spatter's direct `apply_to_images` route over the issue #40 matrix."""

    params = (SPATTER_DIRECT_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        mode = "mud" if name == "spatter_mud" else "rain"
        self.transform = albumentations.Spatter(mode=mode, p=1.0)
        self.transform.set_random_seed(137)
        self.images = _make_image_batch(size_name, channels, dtype_from_name(dtype_name), batch_size)
        self.transform(image=self.images[0])
        self.params = self.transform.get_applied_params()

    def time_apply_to_images(self, case_id: str) -> None:
        self.transform.apply_to_images(self.images, **self.params)


class PeakMemorySpatterBatchMatrix:
    """Measure the largest Spatter batch working set for direct and Compose routes."""

    params = (
        tuple(
            f"{route}|{mode}|{dtype_name}"
            for route in ("direct", "compose")
            for mode in ("rain", "mud")
            for dtype_name in DTYPES
        ),
    )
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        route, mode, dtype_name = case_id.split("|")
        self.route = route
        self.images = _make_image_batch("large", 3, dtype_from_name(dtype_name), 16)
        self.transform = albumentations.Spatter(mode=mode, p=1.0)
        if route == "compose":
            self.compose = albumentations.Compose([self.transform], seed=137, strict=True)
        else:
            self.transform.set_random_seed(137)
            self.transform(image=self.images[0])
            self.params = self.transform.get_applied_params()

    def peakmem_spatter_batch_large_rgb(self, case_id: str) -> None:
        if self.route == "compose":
            self.compose(images=self.images)
        else:
            self.transform.apply_to_images(self.images, **self.params)


class TimeMaskBatchMatrix:
    """Benchmark public `images` plus `masks` batch routes."""

    params = (MASK_BATCH_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        self.transform = albumentations.Compose([MASK_BATCH_TRANSFORMS[name].factory()], seed=137, strict=True)
        self.data = {
            "images": _make_image_batch(size_name, channels, dtype_from_name(dtype_name), batch_size),
            "masks": make_masks(size_name, count=batch_size),
        }

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)


class TimeVolumeBatchMatrix:
    """Benchmark public `volumes` plus `masks3d` batch routes."""

    params = (VOLUME_BATCH_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        self.transform = albumentations.Compose([VOLUME_BATCH_TRANSFORMS[name].factory()], seed=137, strict=True)
        self.data = {
            "masks3d": _make_masks3d_batch(size_name, batch_size),
            "volumes": _make_volume_batch(size_name, channels, dtype_from_name(dtype_name), batch_size),
        }

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)
