"""Batch-route benchmarks for image and mask targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np

import albumentations
from benchmarks.common import (
    CHANNELS,
    DTYPES,
    MEDIAN_BLUR_CASES,
    dtype_from_name,
    make_image,
    make_masks,
)

Factory = Callable[[], object]

BATCH_SIZES = (4, 8)
ILLUMINATION_BATCH_SIZES = (2, 4, 8, 16)
ILLUMINATION_MODES = ("linear", "corner", "gaussian")
ILLUMINATION_NAMES = tuple(f"illumination_{mode}" for mode in ILLUMINATION_MODES)
RANDOM_SHADOW_BATCH_SIZES = (2, 4, 8, 16)
RANDOM_SHADOW_SIZES = ("small", "medium", "large")
SPATTER_BATCH_SIZES = (2, 4, 8, 16)
SPATTER_SIZES = ("small", "medium", "large")
RANDOM_TONE_CURVE_NAMES = ("random_tone_curve", "random_tone_curve_per_channel")
RANDOM_TONE_CURVE_CASE_PREFIXES = tuple(f"{name}|" for name in RANDOM_TONE_CURVE_NAMES)


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
    "exposure_matching": BatchSpec(
        lambda: albumentations.ExposureMatching(p=1.0),
        sizes=("small", "medium", "large"),
    ),
    "gauss_noise": BatchSpec(lambda: albumentations.GaussNoise(p=1.0)),
    "horizontal_flip": BatchSpec(lambda: albumentations.HorizontalFlip(p=1.0)),
    **{
        f"illumination_{mode}": BatchSpec(
            partial(albumentations.Illumination, mode=mode, p=1.0),
            sizes=("small", "medium", "large"),
            batch_sizes=ILLUMINATION_BATCH_SIZES,
        )
        for mode in ILLUMINATION_MODES
    },
    **{
        name: BatchSpec(
            partial(albumentations.MedianBlur, blur_range=(kernel_size, kernel_size), p=1.0),
        )
        for name, kernel_size in MEDIAN_BLUR_CASES
    },
    "normalize": BatchSpec(lambda: albumentations.Normalize(p=1.0)),
    "random_brightness_contrast": BatchSpec(lambda: albumentations.RandomBrightnessContrast(p=1.0)),
    "random_shadow": BatchSpec(
        lambda: albumentations.RandomShadow(num_shadows_range=(2, 2), p=1.0),
        sizes=RANDOM_SHADOW_SIZES,
        batch_sizes=RANDOM_SHADOW_BATCH_SIZES,
    ),
    "random_tone_curve": BatchSpec(lambda: albumentations.RandomToneCurve(p=1.0)),
    "random_tone_curve_per_channel": BatchSpec(
        lambda: albumentations.RandomToneCurve(per_channel=True, p=1.0),
    ),
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
ILLUMINATION_DIRECT_CASES = tuple(
    case_id.replace("|images|", "|direct_images|", 1)
    for case_id in IMAGE_BATCH_CASES
    if case_id.startswith(tuple(f"{name}|" for name in ILLUMINATION_NAMES))
)
RANDOM_SHADOW_DIRECT_CASES = tuple(
    case_id.replace("|images|", "|direct_images|", 1)
    for case_id in IMAGE_BATCH_CASES
    if case_id.startswith("random_shadow|")
)
RANDOM_SHADOW_VOLUME_CASES = tuple(
    case_id.replace("|images|", "|volume|", 1) for case_id in IMAGE_BATCH_CASES if case_id.startswith("random_shadow|")
)
RANDOM_TONE_CURVE_DIRECT_IMAGE_CASES = tuple(
    case_id.replace("|images|", "|direct_images|", 1)
    for case_id in IMAGE_BATCH_CASES
    if case_id.startswith(RANDOM_TONE_CURVE_CASE_PREFIXES)
)
SPATTER_DIRECT_CASES = tuple(
    case_id.replace("|images|", "|direct_images|", 1)
    for case_id in IMAGE_BATCH_CASES
    if case_id.startswith(("spatter_mud|", "spatter_rain|"))
)
MASK_BATCH_CASES = _cases(MASK_BATCH_TRANSFORMS, "images_and_masks")


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


class TimeIlluminationDirectBatchMatrix:
    """Benchmark Illumination's direct batch route over the issue #53 matrix."""

    params = (ILLUMINATION_DIRECT_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        mode = name.removeprefix("illumination_")
        self.transform = albumentations.Illumination(mode=mode, p=1.0)
        self.transform.set_random_seed(137)
        self.images = _make_image_batch(size_name, channels, dtype_from_name(dtype_name), batch_size)
        self.transform(image=self.images[0])
        self.params = self.transform.get_applied_params()

    def time_apply_to_images(self, case_id: str) -> None:
        self.transform.apply_to_images(self.images, **self.params)


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


class TimeRandomShadowDirectBatchMatrix:
    """Benchmark RandomShadow's direct `apply_to_images` route over the issue #46 matrix."""

    params = (RANDOM_SHADOW_DIRECT_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        _, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        self.transform = albumentations.RandomShadow(num_shadows_range=(2, 2), p=1.0)
        self.transform.set_random_seed(137)
        self.images = _make_image_batch(size_name, channels, dtype_from_name(dtype_name), batch_size)
        self.transform(image=self.images[0])
        self.params = self.transform.get_applied_params()

    def time_apply_to_images(self, case_id: str) -> None:
        self.transform.apply_to_images(self.images, **self.params)


class TimeRandomShadowVolumeBatchMatrix:
    """Benchmark RandomShadow's public Compose `volume` route over the issue #46 matrix."""

    params = (RANDOM_SHADOW_VOLUME_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        self.transform = albumentations.Compose([IMAGE_BATCH_TRANSFORMS[name].factory()], seed=137, strict=True)
        self.data = {"volume": _make_image_batch(size_name, channels, dtype_from_name(dtype_name), batch_size)}

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)


class TimeRandomToneCurveDirectImageBatchMatrix:
    """Benchmark RandomToneCurve's direct `apply_to_images` route for shared and per-channel curves."""

    params = (RANDOM_TONE_CURVE_DIRECT_IMAGE_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name, batch_size = _parse_batch_case(case_id)
        self.transform = albumentations.RandomToneCurve(
            per_channel=name == "random_tone_curve_per_channel",
            p=1.0,
        )
        self.transform.set_random_seed(137)
        self.images = _make_image_batch(size_name, channels, dtype_from_name(dtype_name), batch_size)
        self.transform(image=self.images[0])
        self.applied_params = self.transform.get_applied_params()

    def time_apply_to_images(self, case_id: str) -> None:
        self.transform.apply_to_images(self.images, **self.applied_params)


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


class PeakMemoryRandomShadowBatchMatrix:
    """Measure the largest RandomShadow batch working set for direct and Compose routes."""

    params = (tuple(f"{route}|{dtype_name}" for route in ("direct", "compose") for dtype_name in DTYPES),)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        route, dtype_name = case_id.split("|")
        self.route = route
        self.images = _make_image_batch("large", 5, dtype_from_name(dtype_name), 16)
        self.transform = albumentations.RandomShadow(num_shadows_range=(2, 2), p=1.0)
        if route == "compose":
            self.compose = albumentations.Compose([self.transform], seed=137, strict=True)
        else:
            self.transform.set_random_seed(137)
            self.transform(image=self.images[0])
            self.params = self.transform.get_applied_params()

    def peakmem_random_shadow_batch_large_multichannel(self, case_id: str) -> None:
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
