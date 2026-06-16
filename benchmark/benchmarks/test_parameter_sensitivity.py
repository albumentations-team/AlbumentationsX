"""Parameter-sensitivity benchmarks for transforms with scale-dependent runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import albumentations
from benchmarks.common import dtype_from_name, make_image

Factory = Callable[[], object]


@dataclass(frozen=True)
class ParameterSensitivitySpec:
    """Parameter-sensitivity benchmark case metadata."""

    factory: Factory
    public_transform: str
    parameter_scenario: str
    channels: tuple[int, ...] = (3,)
    dtypes: tuple[str, ...] = ("uint8", "float32")
    sizes: tuple[str, ...] = ("small", "medium")


PARAMETER_SENSITIVITY_TRANSFORMS: Mapping[str, ParameterSensitivitySpec] = {
    "blur_kernel_3": ParameterSensitivitySpec(
        factory=lambda: albumentations.Blur(blur_range=(3, 3), p=1.0),
        public_transform="Blur",
        parameter_scenario="kernel_3",
    ),
    "blur_kernel_15": ParameterSensitivitySpec(
        factory=lambda: albumentations.Blur(blur_range=(15, 15), p=1.0),
        public_transform="Blur",
        parameter_scenario="kernel_15",
    ),
    "coarse_dropout_2_small_holes": ParameterSensitivitySpec(
        factory=lambda: albumentations.CoarseDropout(
            num_holes_range=(2, 2),
            hole_height_range=(8, 8),
            hole_width_range=(8, 8),
            p=1.0,
        ),
        public_transform="CoarseDropout",
        parameter_scenario="2_small_holes",
    ),
    "coarse_dropout_32_small_holes": ParameterSensitivitySpec(
        factory=lambda: albumentations.CoarseDropout(
            num_holes_range=(32, 32),
            hole_height_range=(8, 8),
            hole_width_range=(8, 8),
            p=1.0,
        ),
        public_transform="CoarseDropout",
        parameter_scenario="32_small_holes",
    ),
    "gaussian_blur_kernel_3": ParameterSensitivitySpec(
        factory=lambda: albumentations.GaussianBlur(blur_range=(3, 3), sigma_range=(0.8, 0.8), p=1.0),
        public_transform="GaussianBlur",
        parameter_scenario="kernel_3",
    ),
    "gaussian_blur_kernel_15": ParameterSensitivitySpec(
        factory=lambda: albumentations.GaussianBlur(blur_range=(15, 15), sigma_range=(2.0, 2.0), p=1.0),
        public_transform="GaussianBlur",
        parameter_scenario="kernel_15",
    ),
    "grid_distortion_full_resolution": ParameterSensitivitySpec(
        factory=lambda: albumentations.GridDistortion(
            distort_range=(0.1, 0.1),
            map_resolution_range=(1.0, 1.0),
            num_steps=5,
            p=1.0,
        ),
        public_transform="GridDistortion",
        parameter_scenario="full_resolution_map",
    ),
    "grid_distortion_half_resolution": ParameterSensitivitySpec(
        factory=lambda: albumentations.GridDistortion(
            distort_range=(0.1, 0.1),
            map_resolution_range=(0.5, 0.5),
            num_steps=5,
            p=1.0,
        ),
        public_transform="GridDistortion",
        parameter_scenario="half_resolution_map",
    ),
    "image_compression_jpeg_50": ParameterSensitivitySpec(
        factory=lambda: albumentations.ImageCompression(quality_range=(50, 50), p=1.0),
        public_transform="ImageCompression",
        parameter_scenario="jpeg_quality_50",
        dtypes=("uint8",),
    ),
    "image_compression_jpeg_95": ParameterSensitivitySpec(
        factory=lambda: albumentations.ImageCompression(quality_range=(95, 95), p=1.0),
        public_transform="ImageCompression",
        parameter_scenario="jpeg_quality_95",
        dtypes=("uint8",),
    ),
    "superpixels_segments_32": ParameterSensitivitySpec(
        factory=lambda: albumentations.Superpixels(n_segments_range=(32, 32), max_size=128, p=1.0),
        public_transform="Superpixels",
        parameter_scenario="segments_32",
        dtypes=("uint8",),
    ),
    "superpixels_segments_128": ParameterSensitivitySpec(
        factory=lambda: albumentations.Superpixels(n_segments_range=(128, 128), max_size=128, p=1.0),
        public_transform="Superpixels",
        parameter_scenario="segments_128",
        dtypes=("uint8",),
    ),
}

PARAMETER_SENSITIVITY_CASES = tuple(
    f"{name}|{spec.parameter_scenario}|{size_name}|{channel_count}|{dtype_name}"
    for name, spec in PARAMETER_SENSITIVITY_TRANSFORMS.items()
    for size_name in spec.sizes
    for channel_count in spec.channels
    for dtype_name in spec.dtypes
)


def _parse_parameter_case(case_id: str) -> tuple[str, str, str, int, str]:
    name, parameter_scenario, size_name, channels, dtype_name = case_id.split("|")
    return name, parameter_scenario, size_name, int(channels), dtype_name


class TimeParameterSensitivity:
    """Benchmark representative parameter scenarios that materially affect runtime."""

    params = (PARAMETER_SENSITIVITY_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, _, size_name, channels, dtype_name = _parse_parameter_case(case_id)
        self.image = make_image(size_name, channels, dtype_from_name(dtype_name))
        self.transform = albumentations.Compose(
            [PARAMETER_SENSITIVITY_TRANSFORMS[name].factory()],
            seed=137,
            strict=True,
        )

    def time_transform(self, case_id: str) -> None:
        self.transform(image=self.image)
