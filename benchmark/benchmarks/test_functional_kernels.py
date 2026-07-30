"""Direct functional-kernel benchmarks for shared hot paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from albumentations.augmentations.blur import functional as fblur
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.transforms3d import functional as f3d
from benchmarks.common import (
    ANNOTATION_COUNTS,
    CHANNELS,
    DTYPES,
    SIZES,
    VOLUME_SIZES,
    dtype_from_name,
    make_hbb_bboxes,
    make_image,
    make_keypoints,
    make_volume,
)

ImageKernelCall = Callable[[Any], np.ndarray]

FUNCTIONAL_GEOMETRY_IMAGE_KERNELS = (
    "hflip",
    "vflip",
    "transpose",
    "d4",
    "resize",
    "pad_with_params",
    "warp_perspective",
    "remap",
)
FUNCTIONAL_GEOMETRY_ANNOTATION_KERNELS = (
    "bboxes_affine",
    "bboxes_rot90",
    "bboxes_d4",
    "resize_bboxes",
    "keypoints_affine",
    "keypoints_rot90",
    "keypoints_d4",
)
FUNCTIONAL_GEOMETRY_ANNOTATION_COUNTS: Mapping[str, tuple[int, ...]] = {
    "bboxes_affine": (10, 100),
    "keypoints_affine": (10, 100),
}
FUNCTIONAL_3D_KERNELS = (
    "crop3d",
    "pad_3d_with_params",
    "cutout3d",
    "transform_cube",
    "swap_tiles_on_volume",
)


@dataclass(frozen=True)
class KernelSupport:
    """Supported matrix entries for one direct functional kernel."""

    channels: tuple[int, ...] = CHANNELS
    dtypes: tuple[str, ...] = tuple(DTYPES)
    sizes: tuple[str, ...] = tuple(SIZES)


FUNCTIONAL_PIXEL_KERNELS: Mapping[str, KernelSupport] = {
    "exposure_match": KernelSupport(),
    "gamma_transform": KernelSupport(),
    "multiply_add": KernelSupport(),
    "add_weighted": KernelSupport(),
    "normalize_per_image": KernelSupport(),
    "equalize": KernelSupport(),
    "auto_contrast": KernelSupport(),
    "to_gray": KernelSupport(channels=(3,)),
    "shift_hsv": KernelSupport(channels=(3,)),
    "linear_transformation_rgb": KernelSupport(channels=(3,)),
    "pixel_dropout": KernelSupport(),
    "channel_shuffle": KernelSupport(channels=(3, 5)),
    "image_compression": KernelSupport(dtypes=("uint8",)),
    "solarize": KernelSupport(),
    "posterize": KernelSupport(),
}

FUNCTIONAL_BLUR_KERNELS: Mapping[str, KernelSupport] = {
    "box_blur": KernelSupport(),
    "convolve": KernelSupport(),
    "defocus": KernelSupport(),
    "zoom_blur": KernelSupport(),
    "mode_filter": KernelSupport(),
    "glass_blur": KernelSupport(channels=(1, 3), sizes=("small", "medium")),
}


def _image_cases(
    names: tuple[str, ...],
    *,
    size_names: tuple[str, ...] = tuple(SIZES),
    channels: tuple[int, ...] = CHANNELS,
    dtype_names: tuple[str, ...] = tuple(DTYPES),
) -> tuple[str, ...]:
    return tuple(
        f"{name}|{size_name}|{channel_count}|{dtype_name}"
        for name in names
        for size_name in size_names
        for channel_count in channels
        for dtype_name in dtype_names
    )


def _support_cases(specs: Mapping[str, KernelSupport]) -> tuple[str, ...]:
    cases: list[str] = []
    for name, spec in specs.items():
        cases.extend(
            _image_cases(
                (name,),
                size_names=spec.sizes,
                channels=spec.channels,
                dtype_names=spec.dtypes,
            ),
        )
    return tuple(cases)


def _parse_image_case(case_id: str) -> tuple[str, str, int, str]:
    name, size_name, channels, dtype_name = case_id.split("|")
    return name, size_name, int(channels), dtype_name


def _normalized_hbb_bboxes(size_name: str, count: int) -> np.ndarray:
    height, width = SIZES[size_name]
    return fgeometric.normalize_bboxes(make_hbb_bboxes(size_name, count), (height, width))


def _internal_keypoints(size_name: str, count: int) -> np.ndarray:
    xy = make_keypoints(size_name, count)
    angles = np.full((count, 1), 0.25, dtype=np.float32)
    scales = np.ones((count, 1), dtype=np.float32)
    labels = np.arange(count, dtype=np.float32).reshape(count, 1)
    return np.concatenate([xy, angles, scales, labels], axis=1)


FUNCTIONAL_GEOMETRY_IMAGE_CASES = _image_cases(FUNCTIONAL_GEOMETRY_IMAGE_KERNELS)
FUNCTIONAL_GEOMETRY_ANNOTATION_CASES = tuple(
    f"{name}|{count}"
    for name in FUNCTIONAL_GEOMETRY_ANNOTATION_KERNELS
    for count in FUNCTIONAL_GEOMETRY_ANNOTATION_COUNTS.get(name, ANNOTATION_COUNTS)
)
FUNCTIONAL_PIXEL_CASES = _support_cases(FUNCTIONAL_PIXEL_KERNELS)
FUNCTIONAL_BLUR_CASES = _support_cases(FUNCTIONAL_BLUR_KERNELS)
FUNCTIONAL_3D_CASES = tuple(
    f"{name}|{size_name}|{dtype_name}"
    for name in FUNCTIONAL_3D_KERNELS
    for size_name in VOLUME_SIZES
    for dtype_name in DTYPES
)


def _call_hflip(benchmark: Any) -> np.ndarray:
    return fgeometric.hflip(benchmark.image)


def _call_vflip(benchmark: Any) -> np.ndarray:
    return fgeometric.vflip(benchmark.image)


def _call_transpose(benchmark: Any) -> np.ndarray:
    return fgeometric.transpose(benchmark.image)


def _call_d4(benchmark: Any) -> np.ndarray:
    return fgeometric.d4(benchmark.image, "r90")


def _call_resize(benchmark: Any) -> np.ndarray:
    return fgeometric.resize(benchmark.image, benchmark.resize_shape, cv2.INTER_LINEAR)


def _call_pad_with_params(benchmark: Any) -> np.ndarray:
    return fgeometric.pad_with_params(benchmark.image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, 0)


def _call_warp_perspective(benchmark: Any) -> np.ndarray:
    return fgeometric.warp_perspective(
        benchmark.image,
        benchmark.perspective_matrix,
        benchmark.output_size,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
        0,
    )


def _call_remap(benchmark: Any) -> np.ndarray:
    return fgeometric.remap(
        benchmark.image,
        benchmark.map_x,
        benchmark.map_y,
        cv2.INTER_LINEAR,
        cv2.BORDER_REFLECT_101,
        0,
    )


GEOMETRY_IMAGE_CALLS: Mapping[str, ImageKernelCall] = {
    "hflip": _call_hflip,
    "vflip": _call_vflip,
    "transpose": _call_transpose,
    "d4": _call_d4,
    "resize": _call_resize,
    "pad_with_params": _call_pad_with_params,
    "warp_perspective": _call_warp_perspective,
    "remap": _call_remap,
}


def _call_bboxes_affine(benchmark: Any) -> np.ndarray:
    return fgeometric.bboxes_affine(
        benchmark.bboxes,
        benchmark.affine_matrix,
        "largest_box",
        benchmark.image_shape,
        cv2.BORDER_CONSTANT,
        benchmark.image_shape,
        "hbb",
    )


def _call_bboxes_rot90(benchmark: Any) -> np.ndarray:
    return fgeometric.bboxes_rot90(benchmark.bboxes, "r90", "hbb")


def _call_bboxes_d4(benchmark: Any) -> np.ndarray:
    return fgeometric.bboxes_d4(benchmark.bboxes, "h", "hbb")


def _call_resize_bboxes(benchmark: Any) -> np.ndarray:
    output_shape = (benchmark.image_shape[0] // 2, benchmark.image_shape[1] // 2)
    return fgeometric.resize_bboxes(benchmark.bboxes, benchmark.image_shape, output_shape, "hbb")


def _call_keypoints_affine(benchmark: Any) -> np.ndarray:
    return fgeometric.keypoints_affine(
        benchmark.keypoints,
        benchmark.affine_matrix,
        benchmark.image_shape,
        {"x": 1.0, "y": 1.0},
        cv2.BORDER_CONSTANT,
    )


def _call_keypoints_rot90(benchmark: Any) -> np.ndarray:
    return fgeometric.keypoints_rot90(benchmark.keypoints, "r90", benchmark.image_shape)


def _call_keypoints_d4(benchmark: Any) -> np.ndarray:
    return fgeometric.keypoints_d4(benchmark.keypoints, "h", benchmark.image_shape)


GEOMETRY_ANNOTATION_CALLS: Mapping[str, ImageKernelCall] = {
    "bboxes_affine": _call_bboxes_affine,
    "bboxes_rot90": _call_bboxes_rot90,
    "bboxes_d4": _call_bboxes_d4,
    "resize_bboxes": _call_resize_bboxes,
    "keypoints_affine": _call_keypoints_affine,
    "keypoints_rot90": _call_keypoints_rot90,
    "keypoints_d4": _call_keypoints_d4,
}


def _call_gamma_transform(benchmark: Any) -> np.ndarray:
    return fpixel.gamma_transform(benchmark.image, 1.2)


def _call_exposure_match(benchmark: Any) -> np.ndarray:
    gain = fpixel.get_exposure_gain(benchmark.image, 0.4, None)
    return fpixel.exposure_match(benchmark.image, gain)


def _call_multiply_add(benchmark: Any) -> np.ndarray:
    return fpixel.multiply_add(benchmark.image, 1.08, benchmark.add_value)


def _call_add_weighted(benchmark: Any) -> np.ndarray:
    return fpixel.add_weighted(benchmark.image, 0.7, benchmark.image_b, 0.3)


def _call_normalize_per_image(benchmark: Any) -> np.ndarray:
    return fpixel.normalize_per_image(benchmark.image, "image_per_channel")


def _call_equalize(benchmark: Any) -> np.ndarray:
    return fpixel.equalize(benchmark.image)


def _call_auto_contrast(benchmark: Any) -> np.ndarray:
    return fpixel.auto_contrast(benchmark.image, 1.0, None, "cdf")


def _call_to_gray(benchmark: Any) -> np.ndarray:
    return fpixel.to_gray(benchmark.image, 3, "weighted_average")


def _call_shift_hsv(benchmark: Any) -> np.ndarray:
    return fpixel.shift_hsv(benchmark.image, 3, 5, 7)


def _call_linear_transformation_rgb(benchmark: Any) -> np.ndarray:
    return fpixel.linear_transformation_rgb(benchmark.image, benchmark.rgb_matrix)


def _call_pixel_dropout(benchmark: Any) -> np.ndarray:
    return fpixel.pixel_dropout(benchmark.image, benchmark.drop_mask, benchmark.drop_values)


def _call_channel_shuffle(benchmark: Any) -> np.ndarray:
    return fpixel.channel_shuffle(benchmark.image, benchmark.channels_shuffled)


def _call_image_compression(benchmark: Any) -> np.ndarray:
    return fpixel.image_compression(benchmark.image, 90, ".jpg")


def _call_solarize(benchmark: Any) -> np.ndarray:
    return fpixel.solarize(benchmark.image, benchmark.solarize_threshold)


def _call_posterize(benchmark: Any) -> np.ndarray:
    return fpixel.posterize(benchmark.image, 4)


PIXEL_CALLS: Mapping[str, ImageKernelCall] = {
    "exposure_match": _call_exposure_match,
    "gamma_transform": _call_gamma_transform,
    "multiply_add": _call_multiply_add,
    "add_weighted": _call_add_weighted,
    "normalize_per_image": _call_normalize_per_image,
    "equalize": _call_equalize,
    "auto_contrast": _call_auto_contrast,
    "to_gray": _call_to_gray,
    "shift_hsv": _call_shift_hsv,
    "linear_transformation_rgb": _call_linear_transformation_rgb,
    "pixel_dropout": _call_pixel_dropout,
    "channel_shuffle": _call_channel_shuffle,
    "image_compression": _call_image_compression,
    "solarize": _call_solarize,
    "posterize": _call_posterize,
}


def _call_box_blur(benchmark: Any) -> np.ndarray:
    return fblur.box_blur(benchmark.image, 3)


def _call_convolve(benchmark: Any) -> np.ndarray:
    return fblur.convolve(benchmark.image, benchmark.kernel)


def _call_defocus(benchmark: Any) -> np.ndarray:
    return fblur.defocus(benchmark.image, 3, 0.2)


def _call_zoom_blur(benchmark: Any) -> np.ndarray:
    return fblur.zoom_blur(benchmark.image, benchmark.zoom_factors)


def _call_mode_filter(benchmark: Any) -> np.ndarray:
    return fblur.mode_filter(benchmark.image, 3)


def _call_glass_blur(benchmark: Any) -> np.ndarray:
    return fblur.glass_blur(benchmark.image, 0.7, 2, benchmark.glass_iterations, benchmark.dxy, "fast")


BLUR_CALLS: Mapping[str, ImageKernelCall] = {
    "box_blur": _call_box_blur,
    "convolve": _call_convolve,
    "defocus": _call_defocus,
    "zoom_blur": _call_zoom_blur,
    "mode_filter": _call_mode_filter,
    "glass_blur": _call_glass_blur,
}


def _call_crop3d(benchmark: Any) -> np.ndarray:
    return f3d.crop3d(benchmark.volume, benchmark.crop_coords)


def _call_pad_3d_with_params(benchmark: Any) -> np.ndarray:
    return f3d.pad_3d_with_params(benchmark.volume, (1, 1, 2, 2, 2, 2), 0)


def _call_cutout3d(benchmark: Any) -> np.ndarray:
    return f3d.cutout3d(benchmark.volume, benchmark.holes, 0)


def _call_transform_cube(benchmark: Any) -> np.ndarray:
    return f3d.transform_cube(benchmark.volume, 7)


def _call_swap_tiles_on_volume(benchmark: Any) -> np.ndarray:
    return f3d.swap_tiles_on_volume(benchmark.volume, benchmark.tiles, benchmark.mapping)


FUNCTIONAL_3D_CALLS: Mapping[str, ImageKernelCall] = {
    "crop3d": _call_crop3d,
    "pad_3d_with_params": _call_pad_3d_with_params,
    "cutout3d": _call_cutout3d,
    "transform_cube": _call_transform_cube,
    "swap_tiles_on_volume": _call_swap_tiles_on_volume,
}


class TimeFunctionalGeometryImageKernels:
    """Benchmark direct 2D geometric image kernels over the standard image matrix."""

    params = (FUNCTIONAL_GEOMETRY_IMAGE_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, channels, dtype_name = _parse_image_case(case_id)
        height, width = SIZES[size_name]
        self.name = name
        self.image = make_image(size_name, channels, dtype_from_name(dtype_name))
        self.resize_shape = (max(height // 2, 1), max(width // 2, 1))
        self.output_size = (width, height)
        self.perspective_matrix = np.array(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        self.map_x = grid_x + np.float32(0.75)
        self.map_y = grid_y + np.float32(0.5)

    def time_kernel(self, case_id: str) -> None:
        GEOMETRY_IMAGE_CALLS[self.name](self)


class TimeFunctionalGeometryAnnotationKernels:
    """Benchmark direct bbox and keypoint kernels over increasing annotation counts."""

    params = (FUNCTIONAL_GEOMETRY_ANNOTATION_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, count_text = case_id.split("|")
        count = int(count_text)
        size_name = "large" if count >= 1000 else "medium" if count >= 100 else "small"
        height, width = SIZES[size_name]
        self.name = name
        self.image_shape = (height, width)
        self.bboxes = _normalized_hbb_bboxes(size_name, count)
        self.keypoints = _internal_keypoints(size_name, count)
        self.affine_matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def time_kernel(self, case_id: str) -> None:
        GEOMETRY_ANNOTATION_CALLS[self.name](self)


class TimeFunctionalPixelKernels:
    """Benchmark direct pixel kernels over their valid image matrix."""

    params = (FUNCTIONAL_PIXEL_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, channels, dtype_name = _parse_image_case(case_id)
        self.name = name
        self.image = make_image(size_name, channels, dtype_from_name(dtype_name))
        self.image_b = np.flipud(self.image).copy()
        self.add_value = 3 if self.image.dtype == np.uint8 else 0.05
        self.rgb_matrix = np.array(
            [[1.02, -0.01, 0.0], [0.0, 1.01, -0.01], [-0.01, 0.0, 1.02]],
            dtype=np.float32,
        )
        self.drop_mask = np.zeros(self.image.shape[:2], dtype=bool)
        self.drop_mask[::8, ::8] = True
        self.drop_values = np.zeros((channels,), dtype=self.image.dtype)
        self.channels_shuffled = list(reversed(range(channels)))
        self.solarize_threshold = 128 if self.image.dtype == np.uint8 else 0.5

    def time_kernel(self, case_id: str) -> None:
        PIXEL_CALLS[self.name](self)


class TimeFunctionalBlurKernels:
    """Benchmark direct blur and filter kernels over their valid image matrix."""

    params = (FUNCTIONAL_BLUR_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, channels, dtype_name = _parse_image_case(case_id)
        height, width = SIZES[size_name]
        self.name = name
        self.image = make_image(size_name, channels, dtype_from_name(dtype_name))
        self.kernel = np.ones((3, 3), dtype=np.float32) / 9
        self.zoom_factors = np.array([1.0, 1.02, 1.04], dtype=np.float32)
        self.glass_iterations = 1
        max_delta = 2
        total_pixels = (height - max_delta * 2) * (width - max_delta * 2)
        rng = np.random.default_rng(137 + height + width + channels)
        self.dxy = rng.integers(-max_delta, max_delta, size=(total_pixels, self.glass_iterations, 2))

    def time_kernel(self, case_id: str) -> None:
        BLUR_CALLS[self.name](self)


class TimeFunctional3DKernels:
    """Benchmark direct 3D kernels over volume size and dtype variants."""

    params = (FUNCTIONAL_3D_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, dtype_name = case_id.split("|")
        depth, height, width = VOLUME_SIZES[size_name]
        self.name = name
        self.volume = make_volume(size_name, 1, dtype_from_name(dtype_name))
        self.crop_coords = (1, depth - 1, height // 4, height * 3 // 4, width // 4, width * 3 // 4)
        self.holes = np.array(
            [[1, height // 4, width // 4, min(depth - 1, 4), height // 2, width // 2]],
            dtype=np.int32,
        )
        rng = np.random.default_rng(137 + depth + height + width)
        self.tiles = f3d.split_uniform_grid_3d((depth, height, width), (2, 2, 2), rng)
        shape_groups = f3d.create_shape_groups_3d(self.tiles)
        self.mapping = f3d.shuffle_tiles_within_shape_groups_3d(shape_groups, rng)

    def time_kernel(self, case_id: str) -> None:
        FUNCTIONAL_3D_CALLS[self.name](self)
