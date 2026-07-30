"""Full-matrix family benchmarks for hot transform paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import albumentations
from benchmarks.catalog import benchmark_specs
from benchmarks.catalog import make_compose as make_catalog_compose
from benchmarks.catalog import make_data as make_catalog_data
from benchmarks.common import (
    ANNOTATION_COUNTS,
    CHANNELS,
    DTYPES,
    SIZES,
    VOLUME_SIZES,
    dtype_from_name,
    make_batch,
    make_hbb_bboxes,
    make_image,
    make_keypoints,
    make_labels,
    make_mask,
    make_mask3d,
    make_obb_bboxes,
    make_volume,
)

Factory = Callable[[], object]

GEOMETRY_TRANSFORMS: Mapping[str, Factory] = {
    "horizontal_flip": lambda: albumentations.HorizontalFlip(p=1.0),
    "vertical_flip": lambda: albumentations.VerticalFlip(p=1.0),
    "transpose": lambda: albumentations.Transpose(p=1.0),
    "d4": lambda: albumentations.D4(p=1.0),
    "resize": lambda: albumentations.Resize(height=128, width=128, p=1.0),
    "random_scale": lambda: albumentations.RandomScale(scale_range=(0.1, 0.1), p=1.0),
    "smallest_max_size": lambda: albumentations.SmallestMaxSize(max_size=160, p=1.0),
    "longest_max_size": lambda: albumentations.LongestMaxSize(max_size=160, p=1.0),
    "letterbox": lambda: albumentations.LetterBox(size=(160, 160), p=1.0),
    "center_crop": lambda: albumentations.CenterCrop(height=128, width=128, p=1.0),
    "crop": lambda: albumentations.Crop(x_max=128, y_max=128, p=1.0),
    "crop_and_pad": lambda: albumentations.CropAndPad(px=8, p=1.0),
    "random_crop": lambda: albumentations.RandomCrop(height=128, width=128, p=1.0),
    "random_crop_from_borders": lambda: albumentations.RandomCropFromBorders(p=1.0),
    "random_resized_crop": lambda: albumentations.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), p=1.0),
    "random_rotate90": lambda: albumentations.RandomRotate90(p=1.0),
    "random_sized_crop": lambda: albumentations.RandomSizedCrop(
        min_max_height=(128, 192),
        size=(128, 128),
        p=1.0,
    ),
    "square_symmetry": lambda: albumentations.SquareSymmetry(p=1.0),
    "pad": lambda: albumentations.Pad(padding=16, p=1.0),
    "pad_if_needed": lambda: albumentations.PadIfNeeded(min_height=1200, min_width=1200, p=1.0),
    "affine": lambda: albumentations.Affine(scale=(1.05, 1.05), rotate=(3, 3), p=1.0),
    "rotate": lambda: albumentations.Rotate(angle_range=(7, 7), p=1.0),
    "safe_rotate": lambda: albumentations.SafeRotate(angle_range=(7, 7), p=1.0),
    "perspective": lambda: albumentations.Perspective(scale=(0.05, 0.05), p=1.0),
    "elastic": lambda: albumentations.ElasticTransform(alpha=1, sigma=30, p=1.0),
    "grid_distortion": lambda: albumentations.GridDistortion(distort_range=(0.1, 0.1), p=1.0),
    "optical_distortion": lambda: albumentations.OpticalDistortion(distort_range=(0.03, 0.03), p=1.0),
    "grid_elastic": lambda: albumentations.GridElasticDeform(num_grid_xy=(4, 4), magnitude=4, p=1.0),
    "random_grid_shuffle": lambda: albumentations.RandomGridShuffle(p=1.0),
    "thin_plate_spline": lambda: albumentations.ThinPlateSpline(scale_range=(0.2, 0.2), p=1.0),
    "piecewise_affine": lambda: albumentations.PiecewiseAffine(p=1.0),
    "pixel_spread": lambda: albumentations.PixelSpread(p=1.0),
    "morphological": lambda: albumentations.Morphological(p=1.0),
    "water_refraction": lambda: albumentations.WaterRefraction(
        amplitude_range=(0.02, 0.02),
        num_waves_range=(4, 4),
        p=1.0,
    ),
}

GEOMETRY_CHANNELS: Mapping[str, tuple[int, ...]] = {
    "grid_elastic": (1, 3),
}


@dataclass(frozen=True)
class PixelSpec:
    """Pixel benchmark case metadata."""

    factory: Factory
    channels: tuple[int, ...] = CHANNELS
    dtypes: tuple[str, ...] = tuple(DTYPES)
    sizes: tuple[str, ...] = tuple(SIZES)


PIXEL_TRANSFORMS: Mapping[str, PixelSpec] = {
    "advanced_blur": PixelSpec(lambda: albumentations.AdvancedBlur(p=1.0)),
    "annotation_artifacts": PixelSpec(lambda: albumentations.AnnotationArtifacts(p=1.0)),
    "atmospheric_fog": PixelSpec(lambda: albumentations.AtmosphericFog(p=1.0)),
    "random_brightness_contrast": PixelSpec(lambda: albumentations.RandomBrightnessContrast(p=1.0)),
    "random_gamma": PixelSpec(lambda: albumentations.RandomGamma(p=1.0)),
    "auto_contrast": PixelSpec(lambda: albumentations.AutoContrast(p=1.0), dtypes=("uint8",)),
    "exposure_matching": PixelSpec(lambda: albumentations.ExposureMatching(p=1.0)),
    "equalize": PixelSpec(lambda: albumentations.Equalize(p=1.0), channels=(1, 3), dtypes=("uint8",)),
    "clahe": PixelSpec(lambda: albumentations.CLAHE(p=1.0), channels=(1, 3), dtypes=("uint8",)),
    "chromatic_aberration": PixelSpec(lambda: albumentations.ChromaticAberration(p=1.0), channels=(3,)),
    "channel_dropout": PixelSpec(lambda: albumentations.ChannelDropout(p=1.0), channels=(3, 5)),
    "channel_shuffle": PixelSpec(lambda: albumentations.ChannelShuffle(p=1.0), channels=(3, 5)),
    "channel_swap": PixelSpec(lambda: albumentations.ChannelSwap(p=1.0), channels=(3,)),
    "color_jitter": PixelSpec(lambda: albumentations.ColorJitter(p=1.0), channels=(1, 3)),
    "colorize": PixelSpec(lambda: albumentations.Colorize(p=1.0), channels=(1,)),
    "coarse_dropout": PixelSpec(lambda: albumentations.CoarseDropout(p=1.0)),
    "dithering": PixelSpec(lambda: albumentations.Dithering(p=1.0)),
    "downscale": PixelSpec(lambda: albumentations.Downscale(p=1.0)),
    "emboss": PixelSpec(lambda: albumentations.Emboss(p=1.0)),
    "enhance": PixelSpec(lambda: albumentations.Enhance(p=1.0)),
    "erasing": PixelSpec(lambda: albumentations.Erasing(p=1.0)),
    "fancy_pca": PixelSpec(lambda: albumentations.FancyPCA(p=1.0), channels=(3,)),
    "film_grain": PixelSpec(lambda: albumentations.FilmGrain(p=1.0)),
    "grid_dropout": PixelSpec(lambda: albumentations.GridDropout(p=1.0)),
    "grid_mask": PixelSpec(lambda: albumentations.GridMask(p=1.0)),
    "hue_saturation_value": PixelSpec(lambda: albumentations.HueSaturationValue(p=1.0), channels=(3,)),
    "rgb_shift": PixelSpec(lambda: albumentations.RGBShift(p=1.0), channels=(3,)),
    "he_stain": PixelSpec(lambda: albumentations.HEStain(p=1.0), channels=(3,)),
    "halftone": PixelSpec(lambda: albumentations.Halftone(p=1.0)),
    "blur": PixelSpec(lambda: albumentations.Blur(blur_range=(3, 3), p=1.0)),
    "defocus": PixelSpec(
        lambda: albumentations.Defocus(radius_range=(3, 3), alias_blur_range=(0.2, 0.2), p=1.0),
    ),
    "gaussian_blur": PixelSpec(lambda: albumentations.GaussianBlur(blur_range=(3, 3), p=1.0)),
    "glass_blur": PixelSpec(lambda: albumentations.GlassBlur(sigma=0.7, max_delta=2, iterations=1, p=1.0)),
    "median_blur": PixelSpec(lambda: albumentations.MedianBlur(blur_range=(3, 3), p=1.0), dtypes=("uint8",)),
    "mode_filter": PixelSpec(lambda: albumentations.ModeFilter(kernel_range=(3, 3), p=1.0)),
    "motion_blur": PixelSpec(lambda: albumentations.MotionBlur(blur_range=(5, 5), p=1.0), dtypes=("uint8",)),
    "zoom_blur": PixelSpec(
        lambda: albumentations.ZoomBlur(max_factor_range=(1.1, 1.1), step_factor_range=(0.03, 0.03), p=1.0),
    ),
    "iso_noise": PixelSpec(lambda: albumentations.ISONoise(p=1.0), channels=(3,)),
    "lambda": PixelSpec(lambda: albumentations.Lambda(p=1.0)),
    "lens_flare": PixelSpec(lambda: albumentations.LensFlare(p=1.0), channels=(3,)),
    "illumination": PixelSpec(lambda: albumentations.Illumination(p=1.0)),
    "invert_img": PixelSpec(lambda: albumentations.InvertImg(p=1.0)),
    "gauss_noise": PixelSpec(lambda: albumentations.GaussNoise(p=1.0)),
    "additive_noise": PixelSpec(lambda: albumentations.AdditiveNoise(p=1.0)),
    "multiplicative_noise": PixelSpec(lambda: albumentations.MultiplicativeNoise(p=1.0)),
    "noop": PixelSpec(lambda: albumentations.NoOp(p=1.0)),
    "shot_noise": PixelSpec(lambda: albumentations.ShotNoise(p=1.0), dtypes=("uint8",)),
    "normalize": PixelSpec(lambda: albumentations.Normalize(p=1.0)),
    "photometric_distort": PixelSpec(lambda: albumentations.PhotoMetricDistort(p=1.0), channels=(1, 3)),
    "pixel_dropout": PixelSpec(lambda: albumentations.PixelDropout(dropout_prob=0.05, p=1.0)),
    "planckian_jitter": PixelSpec(lambda: albumentations.PlanckianJitter(p=1.0), channels=(3,)),
    "plasma_brightness_contrast": PixelSpec(lambda: albumentations.PlasmaBrightnessContrast(p=1.0)),
    "plasma_shadow": PixelSpec(lambda: albumentations.PlasmaShadow(p=1.0)),
    "random_fog": PixelSpec(lambda: albumentations.RandomFog(p=1.0), channels=(3,)),
    "random_gravel": PixelSpec(lambda: albumentations.RandomGravel(p=1.0), channels=(3,)),
    "random_rain": PixelSpec(lambda: albumentations.RandomRain(p=1.0), channels=(3,)),
    "random_shadow": PixelSpec(lambda: albumentations.RandomShadow(p=1.0)),
    "random_snow": PixelSpec(lambda: albumentations.RandomSnow(p=1.0), channels=(3,)),
    "random_sun_flare": PixelSpec(lambda: albumentations.RandomSunFlare(p=1.0), channels=(3,)),
    "random_tone_curve": PixelSpec(lambda: albumentations.RandomToneCurve(p=1.0)),
    "ringing_overshoot": PixelSpec(lambda: albumentations.RingingOvershoot(p=1.0)),
    "salt_and_pepper": PixelSpec(lambda: albumentations.SaltAndPepper(p=1.0)),
    "sharpen": PixelSpec(lambda: albumentations.Sharpen(p=1.0)),
    "spatter": PixelSpec(lambda: albumentations.Spatter(p=1.0), channels=(3,)),
    "posterize": PixelSpec(lambda: albumentations.Posterize(num_bits=(4, 4), p=1.0)),
    "solarize": PixelSpec(lambda: albumentations.Solarize(threshold_range=(0.5, 0.5), p=1.0)),
    "to_float": PixelSpec(lambda: albumentations.ToFloat(p=1.0), dtypes=("uint8",)),
    "from_float": PixelSpec(lambda: albumentations.FromFloat(p=1.0), dtypes=("float32",)),
    "to_gray": PixelSpec(lambda: albumentations.ToGray(p=1.0), channels=(3,)),
    "to_rgb": PixelSpec(lambda: albumentations.ToRGB(p=1.0), channels=(1,), dtypes=("uint8",)),
    "to_sepia": PixelSpec(lambda: albumentations.ToSepia(p=1.0), channels=(3,), dtypes=("uint8",)),
    "image_compression": PixelSpec(lambda: albumentations.ImageCompression(p=1.0), channels=(3,), dtypes=("uint8",)),
    "superpixels": PixelSpec(
        lambda: albumentations.Superpixels(n_segments_range=(32, 32), max_size=128, p=1.0),
        channels=(3,),
        dtypes=("uint8",),
    ),
    "unsharp_mask": PixelSpec(lambda: albumentations.UnsharpMask(p=1.0)),
    "vignetting": PixelSpec(lambda: albumentations.Vignetting(p=1.0)),
    "xy_masking": PixelSpec(
        lambda: albumentations.XYMasking(
            mask_x_length_range=(12, 12),
            mask_y_length_range=(12, 12),
            num_masks_x_range=(1, 1),
            num_masks_y_range=(1, 1),
            p=1.0,
        ),
    ),
}

ANNOTATION_TRANSFORMS: Mapping[str, Factory] = {
    "hbb_horizontal_flip": lambda: albumentations.HorizontalFlip(p=1.0),
    "hbb_affine": lambda: albumentations.Affine(scale=(1.05, 1.05), rotate=(3, 3), p=1.0),
    "hbb_perspective": lambda: albumentations.Perspective(scale=(0.05, 0.05), p=1.0),
    "hbb_safe_crop": lambda: albumentations.RandomSizedBBoxSafeCrop(height=192, width=192, p=1.0),
    "obb_horizontal_flip": lambda: albumentations.HorizontalFlip(p=1.0),
    "obb_resize": lambda: albumentations.Resize(height=384, width=512, p=1.0),
    "obb_random_scale": lambda: albumentations.RandomScale(scale_range=(0.1, 0.1), p=1.0),
}
SPECIAL_TARGET_TRANSFORMS: Mapping[str, Factory] = {
    "at_least_one_bbox_random_crop": lambda: albumentations.Compose(
        [albumentations.AtLeastOneBBoxRandomCrop(height=96, width=96, p=1.0)],
        bbox_params=albumentations.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        seed=137,
        strict=True,
    ),
    "bbox_safe_random_crop": lambda: albumentations.Compose(
        [albumentations.BBoxSafeRandomCrop(p=1.0)],
        bbox_params=albumentations.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        seed=137,
        strict=True,
    ),
    "constrained_coarse_dropout": lambda: albumentations.Compose(
        [albumentations.ConstrainedCoarseDropout(mask_indices=[1], p=1.0)],
        seed=137,
        strict=True,
    ),
    "crop_non_empty_mask_if_exists": lambda: albumentations.Compose(
        [albumentations.CropNonEmptyMaskIfExists(height=96, width=96, p=1.0)],
        seed=137,
        strict=True,
    ),
    "mask_dropout": lambda: albumentations.Compose(
        [albumentations.MaskDropout(p=1.0)],
        seed=137,
        strict=True,
    ),
    "random_crop_near_bbox": lambda: albumentations.Compose(
        [albumentations.RandomCropNearBBox(p=1.0)],
        bbox_params=albumentations.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        seed=137,
        strict=True,
    ),
}
BBOX_SPECIAL_TARGET_TRANSFORMS = frozenset(
    {
        "at_least_one_bbox_random_crop",
        "bbox_safe_random_crop",
        "random_crop_near_bbox",
    },
)
ANNOTATION_COUNTS_BY_TRANSFORM = {
    "hbb_affine": (10, 100),
    "hbb_perspective": (10, 100),
}
HBB_KEYPOINT_TRANSFORMS = frozenset(
    {
        "hbb_affine",
        "hbb_horizontal_flip",
        "hbb_perspective",
    },
)

VOLUME_TRANSFORMS: Mapping[str, Factory] = {
    "center_crop3d": lambda: albumentations.CenterCrop3D(size=(4, 48, 48), p=1.0),
    "random_crop3d": lambda: albumentations.RandomCrop3D(size=(4, 48, 48), p=1.0),
    "pad3d": lambda: albumentations.Pad3D(padding=(1, 2, 2), p=1.0),
    "pad_if_needed3d": lambda: albumentations.PadIfNeeded3D(min_zyx=(18, 144, 144), p=1.0),
    "coarse_dropout3d": lambda: albumentations.CoarseDropout3D(p=1.0),
    "grid_shuffle3d": lambda: albumentations.GridShuffle3D(grid_zyx=(2, 2, 2), p=1.0),
    "cubic_symmetry": lambda: albumentations.CubicSymmetry(p=1.0),
}

REFERENCE_TRANSFORMS = (
    "CopyAndPaste",
    "FDA",
    "HistogramMatching",
    "Mosaic",
    "OverlayElements",
    "PixelDistributionAdaptation",
    "TextImage",
)


def _matrix_cases(
    names: tuple[str, ...],
    size_names: tuple[str, ...],
    channels: tuple[int, ...],
    dtype_names: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{name}|{size_name}|{channel_count}|{dtype_name}"
        for name in names
        for size_name in size_names
        for channel_count in channels
        for dtype_name in dtype_names
    )


def _parse_image_case(case_id: str) -> tuple[str, str, int, str]:
    name, size_name, channels, dtype_name = case_id.split("|")
    return name, size_name, int(channels), dtype_name


def _pixel_cases() -> tuple[str, ...]:
    cases: list[str] = []
    for name, spec in PIXEL_TRANSFORMS.items():
        cases.extend(_matrix_cases((name,), spec.sizes, spec.channels, spec.dtypes))
    return tuple(cases)


GEOMETRY_CASES = tuple(
    case
    for name in GEOMETRY_TRANSFORMS
    for case in _matrix_cases(
        (name,),
        tuple(SIZES),
        GEOMETRY_CHANNELS.get(name, CHANNELS),
        tuple(DTYPES),
    )
)
PIXEL_CASES = _pixel_cases()
SPECIAL_TARGET_CASES = _matrix_cases(
    tuple(SPECIAL_TARGET_TRANSFORMS),
    tuple(SIZES),
    CHANNELS,
    tuple(DTYPES),
)
ANNOTATION_CASES = tuple(
    f"{name}|{count}"
    for name in ANNOTATION_TRANSFORMS
    for count in ANNOTATION_COUNTS_BY_TRANSFORM.get(name, ANNOTATION_COUNTS)
)
VOLUME_CASES = tuple(
    f"{name}|{size_name}|{dtype_name}"
    for name in VOLUME_TRANSFORMS
    for size_name in VOLUME_SIZES
    for dtype_name in DTYPES
)
REFERENCE_CASES = tuple(
    f"{name}|{size_name}"
    for name in REFERENCE_TRANSFORMS
    for size_name in ("small", "medium")
    if not (name == "TextImage" and size_name == "medium")
)


class TimeGeometryFullMatrix:
    """Benchmark hot geometric transforms over size, channel, and dtype matrices."""

    params = (GEOMETRY_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, channels, dtype_name = _parse_image_case(case_id)
        self.image = make_image(size_name, channels, dtype_from_name(dtype_name))
        self.transform = albumentations.Compose([GEOMETRY_TRANSFORMS[name]()], seed=137, strict=True)

    def time_transform(self, case_id: str) -> None:
        self.transform(image=self.image)


class TimePixelFullMatrix:
    """Benchmark hot pixel transforms over valid size, channel, and dtype matrices."""

    params = (PIXEL_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, channels, dtype_name = _parse_image_case(case_id)
        self.image = make_image(size_name, channels, dtype_from_name(dtype_name))
        self.transform = albumentations.Compose([PIXEL_TRANSFORMS[name].factory()], seed=137, strict=True)

    def time_transform(self, case_id: str) -> None:
        self.transform(image=self.image)


class TimeAnnotationTargets:
    """Benchmark annotation routing and scaling for HBB, OBB, and keypoints."""

    params = (ANNOTATION_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, count_text = case_id.split("|")
        count = int(count_text)
        size_name = "large" if count >= 1000 else "medium" if count >= 100 else "small"
        self.image = make_image(size_name, 3)
        self.transform = self._make_transform(name)
        if name.startswith("obb_"):
            self.data = {
                "bbox_labels": make_labels(count),
                "bboxes": make_obb_bboxes(count),
                "image": self.image,
            }
        else:
            self.data = {
                "bbox_labels": make_labels(count),
                "bboxes": make_hbb_bboxes(size_name, count),
                "image": self.image,
                "mask": make_mask(size_name),
            }
            if name in HBB_KEYPOINT_TRANSFORMS:
                self.data["keypoint_labels"] = make_labels(count)
                self.data["keypoints"] = make_keypoints(size_name, count)

    def _make_transform(self, name: str) -> albumentations.Compose:
        kwargs = {"seed": 137, "strict": True}
        if name.startswith("obb_"):
            kwargs["bbox_params"] = albumentations.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            )
        else:
            kwargs["bbox_params"] = albumentations.BboxParams(
                coord_format="pascal_voc",
                label_fields=["bbox_labels"],
            )
            if name in HBB_KEYPOINT_TRANSFORMS:
                kwargs["keypoint_params"] = albumentations.KeypointParams(
                    coord_format="xy",
                    label_fields=["keypoint_labels"],
                    label_mapping={},
                    remove_invisible=False,
                )
        return albumentations.Compose([ANNOTATION_TRANSFORMS[name]()], **kwargs)

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)


class TimeSpecialTargetMatrix:
    """Benchmark transforms that require bbox, mask, or crop metadata targets."""

    params = (SPECIAL_TARGET_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, channels, dtype_name = _parse_image_case(case_id)
        self.transform = SPECIAL_TARGET_TRANSFORMS[name]()
        self.data = self._make_data(name, size_name, channels, dtype_name)

    def _make_data(self, name: str, size_name: str, channels: int, dtype_name: str) -> dict[str, object]:
        data: dict[str, object] = {
            "image": make_image(size_name, channels, dtype_from_name(dtype_name)),
            "mask": make_mask(size_name),
        }
        if name in BBOX_SPECIAL_TARGET_TRANSFORMS:
            data["bboxes"] = make_hbb_bboxes(size_name, 10)
            data["bbox_labels"] = make_labels(10)
        if name == "random_crop_near_bbox":
            height, width = SIZES[size_name]
            data["cropping_bbox"] = [width // 5, height // 5, width * 4 // 5, height * 4 // 5]
        return data

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)


class TimeReferenceDataFullMatrix:
    """Benchmark metadata/reference-data transforms beyond the smoke path."""

    params = (REFERENCE_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name = case_id.split("|")
        spec = replace(benchmark_specs()[name], size_name=size_name, channels=3)
        self.transform = make_catalog_compose(spec)
        self.data = make_catalog_data(spec)

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)


class TimeVolumetricFullMatrix:
    """Benchmark all public 3D transforms over volume size and dtype."""

    params = (VOLUME_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        name, size_name, dtype_name = case_id.split("|")
        self.transform = albumentations.Compose([VOLUME_TRANSFORMS[name]()], seed=137, strict=True)
        self.data = {
            "mask3d": make_mask3d(size_name),
            "volume": make_volume(size_name, 1, dtype_from_name(dtype_name)),
        }

    def time_transform(self, case_id: str) -> None:
        self.transform(**self.data)


class PeakMemoryHotPaths:
    """Memory checks for allocation-heavy representative paths."""

    def setup(self) -> None:
        self.large_rgb = make_image("large", 3)
        self.medium_rgb = make_image("medium", 3)
        self.medium_batch = make_batch("medium", 3, batch_size=8)
        self.resize = albumentations.Compose([albumentations.Resize(height=512, width=512, p=1.0)], strict=True)
        self.affine = albumentations.Compose([albumentations.Affine(scale=(1.05, 1.05), p=1.0)], strict=True)
        self.normalize = albumentations.Compose([albumentations.Normalize(p=1.0)], strict=True)
        self.batch_pipeline = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
                albumentations.GaussianBlur(blur_range=(3, 3), p=1.0),
            ],
            seed=137,
            strict=True,
        )
        mosaic_spec = benchmark_specs()["Mosaic"]
        self.mosaic = make_catalog_compose(mosaic_spec)
        self.mosaic_data = make_catalog_data(mosaic_spec)
        copy_paste_spec = benchmark_specs()["CopyAndPaste"]
        self.copy_paste = make_catalog_compose(copy_paste_spec)
        self.copy_paste_data = make_catalog_data(copy_paste_spec)
        self.volume_pad = albumentations.Compose(
            [albumentations.PadIfNeeded3D(min_zyx=(18, 144, 144), p=1.0)],
            strict=True,
        )
        self.volume_data = {"mask3d": make_mask3d("medium"), "volume": make_volume("medium")}

    def peakmem_resize_large_rgb(self) -> None:
        self.resize(image=self.large_rgb)

    def peakmem_affine_large_rgb(self) -> None:
        self.affine(image=self.large_rgb)

    def peakmem_normalize_large_rgb(self) -> None:
        self.normalize(image=self.large_rgb)

    def peakmem_batch_pipeline_medium_rgb(self) -> None:
        self.batch_pipeline(images=self.medium_batch)

    def peakmem_mosaic_small_rgb(self) -> None:
        self.mosaic(**self.mosaic_data)

    def peakmem_copy_paste_small_rgb(self) -> None:
        self.copy_paste(**self.copy_paste_data)

    def peakmem_volume_pad_medium(self) -> None:
        self.volume_pad(**self.volume_data)
