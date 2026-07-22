"""Typed source of truth for shared transform contract cases."""

from __future__ import annotations

import copy
import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

import albumentations as A
from tests.helpers.applied_config import ReplayProfile
from tests.helpers.contract_data import (
    ContractDataFactory,
    make_copy_and_paste_data,
    make_crop_near_bbox_data,
    make_float_image_data,
    make_grayscale_image_data,
    make_hbb_data,
    make_image_batch_data,
    make_image_data,
    make_keypoint_data,
    make_mask_data,
    make_mosaic_data,
    make_obb_data,
    make_overlay_data,
    make_reference_data,
    make_text_data,
    make_volume_batch_data,
    make_volume_data,
    remap_data_key,
)


@dataclass(frozen=True)
class TransformContractCase:
    """One public constructor mode and the data needed to replay it."""

    case_id: str
    transform_cls: type[A.BasicTransform]
    init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    data_factory: ContractDataFactory = make_image_data
    compose_kwargs: Mapping[str, Any] = field(default_factory=dict)
    replay_profile: ReplayProfile = ReplayProfile.RUNNABLE
    metadata_keys: frozenset[str] = frozenset()
    seeds: tuple[int, ...] = (137,)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.case_id):
            raise ValueError(f"Invalid transform contract case_id: {self.case_id!r}")
        forbidden = {"p", "strict"} & set(self.init_kwargs)
        if forbidden:
            raise ValueError(f"{self.case_id}: harness-owned arguments are forbidden: {sorted(forbidden)}")
        signature = inspect.signature(self.transform_cls.__init__)
        public_parameters = {
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        unknown = set(self.init_kwargs) - public_parameters
        if unknown:
            raise ValueError(f"{self.case_id}: unknown public constructor arguments: {sorted(unknown)}")
        if not self.seeds:
            raise ValueError(f"{self.case_id}: at least one deterministic seed is required")
        object.__setattr__(self, "init_kwargs", MappingProxyType(copy.deepcopy(dict(self.init_kwargs))))
        object.__setattr__(self, "compose_kwargs", MappingProxyType(copy.deepcopy(dict(self.compose_kwargs))))


_BASE_CASE_SPECS: list[list[Any]] = [
    [
        A.ImageCompression,
        {
            "quality_range": (10, 80),
            "compression_type": "webp",
        },
    ],
    [
        A.HueSaturationValue,
        {"hue_shift_range": (-70, 70), "sat_shift_range": (-95, 95), "val_shift_range": (-55, 55)},
    ],
    [A.RGBShift, {"r_shift_range": (-70, 70), "g_shift_range": (-80, 80), "b_shift_range": (-40, 40)}],
    [A.RandomBrightnessContrast, {"brightness_range": (-0.5, 0.5), "contrast_range": (-0.8, 0.8)}],
    [A.Blur, {"blur_range": (3, 5)}],
    [A.MotionBlur, {"blur_range": (3, 5)}],
    [A.MedianBlur, {"blur_range": (3, 5)}],
    [A.ModeFilter, {"kernel_range": (3, 5)}],
    [A.GaussianBlur, {"blur_range": (3, 5)}],
    [
        A.GaussNoise,
        {"std_range": (0.2, 0.44), "mean_range": (0.0, 0.0), "per_channel": False},
    ],
    [A.CLAHE, {"clip_range": (1, 2), "tile_grid_size": (12, 12)}],
    [A.RandomGamma, {"gamma_range": (10, 90)}],
    [
        A.CoarseDropout,
        [
            {
                "num_holes_range": (2, 5),
                "hole_height_range": (3, 4),
                "hole_width_range": (4, 6),
            },
            {
                "num_holes_range": (2, 5),
                "hole_height_range": (0.1, 0.2),
                "hole_width_range": (0.2, 0.3),
            },
        ],
    ],
    [
        A.RandomSnow,
        {"snow_point_range": (0.2, 0.4), "brightness_coeff": 4},
    ],
    [
        A.RandomRain,
        {
            "slant_range": (-5, 5),
            "drop_length": 15,
            "drop_width": 2,
            "drop_color": (100, 100, 100),
            "blur_value": 3,
            "brightness_coefficient": 0.5,
            "rain_type": "heavy",
        },
    ],
    [A.RandomFog, {"fog_coef_range": (0.2, 0.8), "alpha_coef": 0.11}],
    [
        A.RandomSunFlare,
        {
            "flare_roi": (0.1, 0.1, 0.9, 0.6),
            "angle_range": (0.1, 0.95),
            "num_flare_circles_range": (7, 11),
            "src_radius": 300,
            "src_color": (200, 200, 200),
        },
    ],
    [
        A.RandomGravel,
        {
            "gravel_roi": (0.1, 0.4, 0.9, 0.9),
            "number_of_patches": 2,
        },
    ],
    [
        A.RandomShadow,
        {
            "shadow_roi": (0.1, 0.4, 0.9, 0.9),
            "num_shadows_range": (2, 4),
            "shadow_dimension": 8,
        },
    ],
    [
        A.PadIfNeeded,
        {
            "min_height": 512,
            "min_width": 512,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": (10, 10, 10),
        },
    ],
    [
        A.Rotate,
        {
            "angle_range": (-120, 120),
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": (10, 10, 10),
            "crop_border": False,
        },
    ],
    [
        A.SafeRotate,
        {
            "angle_range": (-120, 120),
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": 10,
        },
    ],
    [
        A.ShiftScaleRotate,
        [
            {
                "shift_range": (-0.2, 0.2),
                "scale_range": (-0.2, 0.2),
                "rotate_range": (-70, 70),
                "interpolation": cv2.INTER_CUBIC,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 10,
            },
            {
                "shift_range_x": (-0.3, 0.3),
                "shift_range_y": (-0.4, 0.4),
                "scale_range": (-0.2, 0.2),
                "rotate_range": (-70, 70),
                "interpolation": cv2.INTER_CUBIC,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 10,
            },
        ],
    ],
    [
        A.OpticalDistortion,
        {
            "distort_range": (-0.2, 0.2),
            "interpolation": cv2.INTER_AREA,
        },
    ],
    [
        A.GridDistortion,
        {
            "num_steps": 10,
            "distort_range": (-0.5, 0.5),
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [
        A.ElasticTransform,
        {
            "alpha": 2,
            "sigma": 25,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [A.PixelSpread, {"radius": 3}],
    [A.CenterCrop, {"height": 90, "width": 95}],
    [A.RandomCrop, {"height": 90, "width": 95}],
    [A.AtLeastOneBBoxRandomCrop, {"height": 90, "width": 95}],
    [A.CropNonEmptyMaskIfExists, {"height": 10, "width": 10}],
    [A.RandomSizedCrop, {"min_max_height": (90, 100), "size": (90, 90)}],
    [A.Crop, {"x_max": 64, "y_max": 64}],
    [A.ToFloat, {"max_value": 16536}],
    [
        A.Normalize,
        {
            "mean": (0.385, 0.356, 0.306),
            "std": (0.129, 0.124, 0.125),
            "max_pixel_value": 100.0,
        },
    ],
    [A.RandomScale, {"scale_range": (-0.2, 0.2), "interpolation": cv2.INTER_CUBIC}],
    [A.Resize, {"height": 64, "width": 64}],
    [A.SmallestMaxSize, {"max_size": 64, "interpolation": cv2.INTER_AREA}],
    [A.LongestMaxSize, [{"max_size": 128}, {"max_size_hw": (127, 126)}]],
    [
        A.LetterBox,
        [
            {"size": (128, 128)},
            {"size": (128, 64), "position": "top_left"},
            {"size": (64, 128), "position": "bottom_right"},
        ],
    ],
    [A.RandomGridShuffle, {"grid": (4, 4)}],
    [A.Solarize, {"threshold_range": [0.5, 0.5]}],
    [A.Posterize, {"num_bits": (3, 5)}],
    [A.Equalize, {"mode": "pil", "by_channels": False}],
    [
        A.MultiplicativeNoise,
        {"multiplier": (0.7, 2.3), "per_channel": True, "elementwise": True},
    ],
    [
        A.ColorJitter,
        {
            "brightness_range": [0.2, 0.3],
            "contrast_range": [0.7, 0.9],
            "saturation_range": [1.2, 1.7],
            "hue_range": [-0.2, 0.1],
        },
    ],
    [
        A.PhotoMetricDistort,
        {
            "brightness_range": (0.875, 1.125),
            "contrast_range": (0.5, 1.5),
            "saturation_range": (0.5, 1.5),
            "hue_range": (-0.05, 0.05),
            "distort_p": 0.5,
        },
    ],
    [
        A.Perspective,
        {
            "scale": (0.05, 0.5),
            "keep_size": True,
            "border_mode": cv2.BORDER_REFLECT_101,
            "fill": 10,
            "fill_mask": 100,
            "fit_output": True,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [A.Sharpen, {"alpha_range": [0.2, 0.5], "lightness_range": [0.5, 1.0]}],
    [A.Emboss, {"alpha_range": [0.2, 0.5], "strength_range": [0.5, 1.0]}],
    [A.Enhance, {"mode": "edge", "alpha_range": (0.5, 1.0)}],
    [A.Enhance, {"mode": "detail", "alpha_range": (0.5, 1.0)}],
    [A.RandomToneCurve, {"scale": 0.2, "per_channel": False}],
    [A.RandomToneCurve, {"scale": 0.3, "per_channel": True}],
    [
        A.CropAndPad,
        {
            "px": 10,
            "keep_size": False,
            "sample_independently": False,
            "interpolation": cv2.INTER_CUBIC,
            "fill_mask": [10, 20, 30],
            "fill": [11, 12, 13],
            "border_mode": cv2.BORDER_REFLECT101,
        },
    ],
    [
        A.Superpixels,
        {
            "p_replace_range": (0.5, 0.7),
            "n_segments_range": (20, 30),
            "max_size": 25,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [
        A.Affine,
        [
            {
                "scale": (0.5, 0.5),
                "translate_percent": (0.1, 0.1),
                "translate_px": None,
                "rotate": (33, 33),
                "shear": (21, 21),
                "interpolation": cv2.INTER_CUBIC,
                "fill": 25,
                "fill_mask": 0,
                "border_mode": cv2.BORDER_CONSTANT,
                "fit_output": False,
            },
            {
                "scale": {"x": [0.3, 0.5], "y": [0.1, 0.2]},
                "translate_percent": None,
                "translate_px": {"x": [10, 20], "y": [5, 10]},
                "rotate": [333, 360],
                "shear": {"x": [31, 38], "y": [41, 48]},
                "interpolation": 3,
                "fill": [10, 20, 30],
                "fill_mask": 1,
                "border_mode": cv2.BORDER_REFLECT,
                "fit_output": False,
                "keep_ratio": False,  # Explicitly set to False since x and y scale ranges are different
            },
        ],
    ],
    [
        A.PiecewiseAffine,
        {
            "scale_range": (0.33, 0.33),
            "nb_rows_range": (10, 20),
            "nb_cols_range": (33, 33),
            "interpolation": cv2.INTER_AREA,
            "mask_interpolation": cv2.INTER_NEAREST,
            "absolute_scale": True,
        },
    ],
    [A.ChannelDropout, dict(channel_drop_range=(1, 2), fill=1)],
    [A.ChannelShuffle, {}],
    [
        A.Downscale,
        dict(
            scale_range=[0.5, 0.75],
            interpolation_pair={
                "downscale": cv2.INTER_LINEAR,
                "upscale": cv2.INTER_LINEAR,
            },
        ),
    ],
    [A.FromFloat, dict(dtype="uint8", max_value=1)],
    [A.HorizontalFlip, {}],
    [A.ISONoise, dict(color_shift_range=(0.2, 0.3), intensity_range=(0.7, 0.9))],
    [A.InvertImg, {}],
    [A.MaskDropout, dict(max_objects_range=(2, 2), fill=0, fill_mask=0)],
    [A.NoOp, {}],
    [
        A.RandomResizedCrop,
        dict(size=(20, 30), scale=(0.5, 0.6), ratio=(0.8, 0.9)),
    ],
    [A.FancyPCA, dict(alpha=0.3)],
    [A.RandomRotate90, {}],
    [A.ToGray, {"method": "pca"}],
    [A.ToRGB, {}],
    [A.ToSepia, {}],
    [
        A.Colorize,
        dict(
            black_range=((0, 0, 200), (10, 10, 255)),
            mid_range=((100, 0, 100), (160, 50, 160)),
            white_range=((220, 200, 0), (255, 255, 50)),
            mid_value_range=(100, 160),
        ),
    ],
    [A.Transpose, {}],
    [A.VerticalFlip, {}],
    [A.RingingOvershoot, dict(blur_range=(7, 15), cutoff_range=(np.pi / 5, np.pi / 2))],
    [
        A.UnsharpMask,
        {
            "blur_range": (3, 7),  # Allow for stronger blur
            "sigma_range": (0.5, 2.0),  # Increase sigma range
            "alpha_range": (0.5, 1.0),  # Allow for stronger sharpening
            "threshold": 5,  # Lower threshold to allow more changes
        },
    ],
    [A.AdvancedBlur, dict(blur_range=(3, 5), rotate_range=(60, 90))],
    [
        A.PixelDropout,
        [
            {"dropout_prob": 0.1, "per_channel": True, "drop_value": None},
            {
                "dropout_prob": 0.1,
                "per_channel": False,
                "drop_value": 2,
                "mask_drop_value": 15,
            },
        ],
    ],
    [
        A.RandomCropFromBorders,
        dict(crop_left=0.2, crop_right=0.3, crop_top=0.05, crop_bottom=0.5),
    ],
    [
        A.Spatter,
        [
            dict(
                mode="rain",
                mean_range=(0.65, 0.65),
                std_range=(0.3, 0.3),
                gauss_sigma_range=(2, 2),
                cutout_threshold_range=(0.68, 0.68),
                intensity_range=(0.6, 0.6),
            ),
            dict(
                mode="mud",
                mean_range=(0.65, 0.65),
                std_range=(0.3, 0.3),
                gauss_sigma_range=(2, 2),
                cutout_threshold_range=(0.68, 0.68),
                intensity_range=(0.6, 0.6),
            ),
        ],
    ],
    [
        A.ChromaticAberration,
        dict(
            primary_distortion_range=(-0.02, 0.02),
            secondary_distortion_range=(-0.05, 0.05),
            mode="green_purple",
            interpolation=cv2.INTER_LINEAR,
        ),
    ],
    [A.Defocus, {"radius_range": (5, 7), "alias_blur_range": (0.2, 0.6)}],
    [A.ZoomBlur, {"max_factor_range": (1.56, 1.7), "step_factor_range": (0.02, 0.04)}],
    [
        A.XYMasking,
        {
            "num_masks_x_range": (1, 3),
            "num_masks_y_range": (3, 3),
            "mask_x_length_range": (10, 20),
            "mask_y_length_range": (10, 10),
            "fill_mask": 1,
            "fill": 0,
        },
    ],
    [
        A.PadIfNeeded,
        {
            "min_height": 512,
            "min_width": 512,
            "border_mode": 0,
            "fill": [124, 116, 104],
            "position": "top_left",
        },
    ],
    [A.GlassBlur, dict(sigma=0.8, max_delta=5, iterations=3, mode="exact")],
    [
        A.GridDropout,
        dict(
            ratio=0.75,
            holes_number_xy=(2, 10),
            shift_xy=(10, 20),
            random_offset=True,
            fill=10,
            fill_mask=20,
        ),
    ],
    [A.Morphological, {}],
    [A.D4, {}],
    [A.SquareSymmetry, {}],
    [A.AnnotationArtifacts, {}],
    [A.PlanckianJitter, {}],
    [A.OverlayElements, {}],
    [A.CopyAndPaste, {}],
    [A.RandomCropNearBBox, {}],
    [
        A.TextImage,
        dict(
            font_path="./tests/files/LiberationSerif-Bold.ttf",
            font_size_fraction_range=(0.8, 0.9),
            font_color=(255, 0, 0),  # red in RGB
            stopwords=(
                "a",
                "the",
                "is",
                "of",
                "it",
                "and",
                "to",
                "in",
                "on",
                "with",
                "for",
                "at",
                "by",
            ),
        ),
    ],
    [A.GridElasticDeform, {"num_grid_xy": (10, 10), "magnitude": 10}],
    [A.ShotNoise, {"scale_range": (0.1, 0.3)}],
    [A.TimeReverse, {}],
    [A.TimeMasking, {"time_mask_param": 10}],
    [A.FrequencyMasking, {"freq_mask_param": 30}],
    [A.Pad, {"padding": 10}],
    [A.Erasing, {}],
    [A.AdditiveNoise, {}],
    [A.SaltAndPepper, {"amount_range": (0.5, 0.5), "salt_vs_pepper_range": (0.5, 0.5)}],
    [A.PlasmaBrightnessContrast, {"brightness_range": (0.5, 0.5), "contrast_range": (0.5, 0.5)}],
    [A.PlasmaShadow, {}],
    [A.Illumination, {}],
    [A.ThinPlateSpline, {}],
    [
        A.AutoContrast,
        [
            {"cutoff": 0, "ignore": None, "method": "cdf"},
            {"cutoff": 0, "ignore": None, "method": "pil"},
        ],
    ],
    [
        A.PadIfNeeded3D,
        {
            "min_zyx": (300, 200, 400),
            "pad_divisor_zyx": (10, 10, 10),
            "position": "center",
            "fill": 10,
            "fill_mask": 20,
        },
    ],
    [A.Pad3D, {"padding": 10}],
    [A.CenterCrop3D, {"size": (2, 30, 30)}],
    [A.RandomCrop3D, {"size": (2, 30, 30)}],
    [
        A.CoarseDropout3D,
        {
            "num_holes_range": (1, 3),
            "hole_depth_range": (0.1, 0.2),
            "hole_height_range": (0.1, 0.2),
            "hole_width_range": (0.1, 0.2),
            "fill": 0,
            "fill_mask": None,
        },
    ],
    [A.CubicSymmetry, {}],
    [A.AtLeastOneBBoxRandomCrop, {"height": 80, "width": 80, "erosion_factor": 0.2}],
    [
        A.ConstrainedCoarseDropout,
        {
            "num_holes_range": (1, 3),
            "hole_height_range": (0.1, 0.2),
            "hole_width_range": (0.1, 0.2),
            "fill": 0,
            "fill_mask": 0,
            "mask_indices": [1],
        },
    ],
    [A.RandomSizedBBoxSafeCrop, {"height": 80, "width": 80, "erosion_rate": 0.2}],
    [A.BBoxSafeRandomCrop, {"erosion_rate": 0.2}],
    [
        A.HEStain,
        [
            {
                "method": "vahadane",
                "intensity_scale_range": (0.5, 1.5),
                "intensity_shift_range": (-0.1, 0.1),
                "augment_background": False,
            },
            {
                "method": "macenko",
                "intensity_scale_range": (0.5, 1.5),
                "intensity_shift_range": (-0.1, 0.1),
                "augment_background": True,
            },
            {
                "method": "random_preset",
                "intensity_scale_range": (0.5, 1.5),
                "intensity_shift_range": (-0.1, 0.1),
                "augment_background": True,
            },
        ],
    ],
    [A.FDA, {"beta_range": (0.1, 0.3), "metadata_key": "fda_metadata"}],
    [A.HistogramMatching, {"blend_ratio": (0.5, 1.0), "metadata_key": "hm_metadata"}],
    [
        A.PixelDistributionAdaptation,
        {"blend_ratio": (0.25, 1.0), "transform_type": "pca", "metadata_key": "pda_metadata"},
    ],
    [
        A.Mosaic,
        {"grid_yx": (2, 2), "target_size": (256, 256), "cell_shape": (128, 128), "metadata_key": "mosaic_metadata"},
    ],
    [A.Dithering, {"method": "error_diffusion", "n_colors": 2}],
    [A.GridShuffle3D, {"grid_zyx": (2, 2, 2)}],
    [A.Vignetting, {"intensity_range": (0.3, 0.6), "center_range": (0.4, 0.6)}],
    [A.ChannelSwap, {"channel_order": (1, 2, 0)}],
    [A.FilmGrain, {"intensity_range": (0.1, 0.3), "grain_size_range": (1, 3)}],
    [A.Halftone, {"dot_size_range": (4, 8), "blend_range": (0.0, 0.3)}],
    [A.GridMask, {"num_grid_range": (3, 5), "line_width_range": (0.2, 0.4)}],
    [A.LensFlare, {"intensity_range": (0.3, 0.6), "num_ghosts_range": (3, 5)}],
    [
        A.WaterRefraction,
        {"amplitude_range": (0.002, 0.005), "wavelength_range": (0.1, 0.15), "keypoint_remapping_method": "direct"},
    ],
    [A.AtmosphericFog, {"density_range": (1.0, 2.0), "depth_mode": "linear"}],
]

_PARAMETER_MODE_SPECS: list[tuple[str, type[A.BasicTransform], dict[str, Any]]] = [
    (
        "gaussian-shared",
        A.AdditiveNoise,
        {
            "noise_type": "gaussian",
            "spatial_mode": "shared",
            "noise_params": {"mean_range": (-0.1, 0.1), "std_range": (0.05, 0.15)},
        },
    ),
    (
        "alternate-ranges",
        A.AdvancedBlur,
        {"sigma_x_range": (0.3, 0.8), "sigma_y_range": (0.4, 0.9), "beta_range": (0.7, 1.3), "noise_range": (0.8, 1.2)},
    ),
    (
        "output-and-box-modes",
        A.Affine,
        {
            "mask_interpolation": cv2.INTER_LINEAR,
            "fit_output": True,
            "rotate_method": "ellipse",
            "balanced_scale": True,
            "scale": (0.8, 1.2),
        },
    ),
    (
        "mixed-elements",
        A.AnnotationArtifacts,
        {
            "element_types": ("text", "arrow"),
            "element_probabilities": (0.6, 0.4),
            "count_range": (2, 4),
            "text_length_range": (2, 6),
            "font_scale_range": (0.4, 1.0),
            "thickness_range": (2, 4),
            "size_ratio_range": (0.15, 0.3),
            "line_length_ratio_range": (0.2, 0.6),
            "tip_length_range": (0.1, 0.3),
            "corner_prob": 0.2,
            "black_white_prob": 0.3,
        },
    ),
    ("diagonal-colored", A.AtmosphericFog, {"fog_color": (180, 190, 200), "depth_mode": "diagonal"}),
    ("cutoff-ignore", A.AutoContrast, {"cutoff": 5, "ignore": 0}),
    (
        "padded-top-left",
        A.CenterCrop,
        {
            "pad_if_needed": True,
            "pad_position": "top_left",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
        },
    ),
    ("padded", A.CenterCrop3D, {"pad_if_needed": True, "fill": 7, "fill_mask": 3}),
    ("fixed-order", A.ChannelShuffle, {"channel_order": (2, 0, 1)}),
    (
        "red-blue-cubic",
        A.ChromaticAberration,
        {
            "primary_distortion_range": (-0.01, 0.01),
            "secondary_distortion_range": (-0.03, 0.03),
            "mode": "red_blue",
            "interpolation": cv2.INTER_CUBIC,
        },
    ),
    ("fixed-fill", A.CoarseDropout, {"fill": 17, "fill_mask": 3}),
    (
        "alternate-ranges-and-fill",
        A.CoarseDropout3D,
        {
            "hole_depth_range": (0.2, 0.3),
            "hole_height_range": (0.2, 0.3),
            "hole_width_range": (0.2, 0.3),
            "fill": 7,
            "fill_mask": 3,
        },
    ),
    ("bbox-label-selection", A.ConstrainedCoarseDropout, {"fill": 7, "bbox_labels": [1], "mask_indices": None}),
    (
        "gaussian-scaled-donor",
        A.CopyAndPaste,
        {
            "min_visibility_after_paste": 0.2,
            "blend_mode": "gaussian",
            "blend_sigma_range": (0.5, 1.5),
            "scale_range": (0.8, 1.2),
            "min_paste_area": 4,
            "metadata_key": "donors",
        },
    ),
    (
        "padded-offset",
        A.Crop,
        {
            "x_min": 5,
            "y_min": 7,
            "pad_if_needed": True,
            "pad_position": "bottom_right",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
        },
    ),
    ("percent-linear-mask", A.CropAndPad, {"px": None, "percent": (-0.1, 0.1), "mask_interpolation": cv2.INTER_LINEAR}),
    ("ignored-mask-values", A.CropNonEmptyMaskIfExists, {"ignore_values": [0], "ignore_channels": []}),
    ("fixed-element", A.D4, {"group_element": "r90"}),
    (
        "ordered-per-channel",
        A.Dithering,
        {
            "method": "ordered",
            "n_colors": 4,
            "color_mode": "per_channel",
            "error_diffusion_algorithm": "jarvis",
            "bayer_matrix_size": 8,
            "serpentine": True,
            "noise_range": (-0.2, 0.2),
        },
    ),
    (
        "uniform-direct-low-resolution",
        A.ElasticTransform,
        {
            "approximate": True,
            "same_dxdy": True,
            "mask_interpolation": cv2.INTER_LINEAR,
            "noise_distribution": "uniform",
            "keypoint_remapping_method": "direct",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
            "map_resolution_range": (0.5, 0.8),
        },
    ),
    ("wide-alpha", A.Enhance, {"alpha_range": (0.2, 0.8)}),
    ("explicit-mask", A.Equalize, {"mask": np.ones((128, 128, 1), dtype=np.uint8), "mask_params": ("mask",)}),
    ("fixed-fill-and-shape", A.Erasing, {"scale": (0.1, 0.2), "ratio": (0.8, 1.2), "fill": 11, "fill_mask": 2}),
    ("custom-metadata-key", A.FDA, {"metadata_key": "references"}),
    ("fine-grain", A.FilmGrain, {"intensity_range": (0.05, 0.2), "grain_size_range": (2, 4)}),
    ("short-mask", A.FrequencyMasking, {"freq_mask_param": 20}),
    (
        "per-channel-shifted-mean",
        A.GaussNoise,
        {"std_range": (0.1, 0.3), "mean_range": (-0.05, 0.05), "per_channel": True},
    ),
    ("wide-sigma", A.GaussianBlur, {"sigma_range": (0.2, 1.5)}),
    (
        "unnormalized-direct",
        A.GridDistortion,
        {
            "normalized": False,
            "mask_interpolation": cv2.INTER_LINEAR,
            "keypoint_remapping_method": "direct",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
            "map_resolution_range": (0.5, 0.8),
        },
    ),
    (
        "fixed-offset-unit-size",
        A.GridDropout,
        {"holes_number_xy": None, "unit_size_range": (16, 24), "random_offset": False, "shift_xy": (1, 1)},
    ),
    (
        "cubic-linear-mask",
        A.GridElasticDeform,
        {"interpolation": cv2.INTER_CUBIC, "mask_interpolation": cv2.INTER_LINEAR},
    ),
    ("rotated-fixed-fill", A.GridMask, {"rotation_range": (10, 20), "fill": 7, "fill_mask": 3}),
    ("asymmetric-grid", A.GridShuffle3D, {"grid_zyx": (1, 2, 2)}),
    ("explicit-preset", A.HEStain, {"method": "preset", "preset": "dark"}),
    ("custom-reference", A.HistogramMatching, {"blend_ratio": (0.2, 0.4), "metadata_key": "references"}),
    (
        "gaussian-darken",
        A.Illumination,
        {
            "mode": "gaussian",
            "intensity_range": (0.05, 0.1),
            "effect_type": "darken",
            "angle_range": (30, 60),
            "center_range": (0.3, 0.6),
            "sigma_range": (0.3, 0.7),
        },
    ),
    (
        "wide-roi-rays",
        A.LensFlare,
        {"flare_roi": (0.1, 0.1, 0.8, 0.6), "num_rays_range": (6, 10), "bloom_range": (0.02, 0.08)},
    ),
    (
        "custom-rendering",
        A.LetterBox,
        {"interpolation": cv2.INTER_CUBIC, "mask_interpolation": cv2.INTER_LINEAR, "fill": 7, "fill_mask": 3},
    ),
    (
        "downscale-aware",
        A.LongestMaxSize,
        {"interpolation": cv2.INTER_CUBIC, "mask_interpolation": cv2.INTER_LINEAR, "area_for_downscale": "image_mask"},
    ),
    ("fixed-fill", A.MaskDropout, {"fill": 7, "fill_mask": 3}),
    ("erosion-large-kernel", A.Morphological, {"scale": (4, 5), "operation": "erosion"}),
    (
        "wide-contain",
        A.Mosaic,
        {
            "grid_yx": (2, 3),
            "center_range": (0.2, 0.4),
            "fit_mode": "contain",
            "interpolation": cv2.INTER_CUBIC,
            "mask_interpolation": cv2.INTER_LINEAR,
            "fill": 7,
            "fill_mask": 3,
            "metadata_key": "tiles",
        },
    ),
    (
        "centered-direction",
        A.MotionBlur,
        {"allow_shifted": False, "angle_range": (30, 60), "direction_range": (-0.5, 0.5)},
    ),
    ("per-image", A.Normalize, {"normalization": "image"}),
    (
        "fisheye-direct",
        A.OpticalDistortion,
        {
            "mask_interpolation": cv2.INTER_LINEAR,
            "mode": "fisheye",
            "keypoint_remapping_method": "direct",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
            "map_resolution_range": (0.5, 0.8),
        },
    ),
    ("custom-metadata-key", A.OverlayElements, {"metadata_key": "overlays"}),
    ("reflect-fixed-fill", A.Pad, {"fill": 7, "fill_mask": 3, "border_mode": cv2.BORDER_REFLECT}),
    ("fixed-fill", A.Pad3D, {"fill": 7, "fill_mask": 3}),
    (
        "divisor-mode",
        A.PadIfNeeded,
        {
            "min_height": None,
            "min_width": None,
            "pad_height_divisor": 16,
            "pad_width_divisor": 16,
            "border_mode": cv2.BORDER_REFLECT,
            "fill_mask": 3,
        },
    ),
    ("random-position", A.PadIfNeeded3D, {"position": "random"}),
    ("variable-output-linear-mask", A.Perspective, {"keep_size": False, "mask_interpolation": cv2.INTER_LINEAR}),
    (
        "wide-ranges-high-probability",
        A.PhotoMetricDistort,
        {
            "brightness_range": (0.8, 1.2),
            "contrast_range": (0.7, 1.3),
            "saturation_range": (0.6, 1.4),
            "hue_range": (-0.1, 0.1),
            "distort_p": 0.8,
        },
    ),
    (
        "direct-low-resolution",
        A.PiecewiseAffine,
        {
            "mask_interpolation": cv2.INTER_LINEAR,
            "keypoint_remapping_method": "direct",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
            "map_resolution_range": (0.5, 0.8),
        },
    ),
    (
        "standard-custom-reference",
        A.PixelDistributionAdaptation,
        {"blend_ratio": (0.1, 0.5), "transform_type": "standard", "metadata_key": "references"},
    ),
    (
        "linear-direct",
        A.PixelSpread,
        {
            "interpolation": cv2.INTER_LINEAR,
            "mask_interpolation": cv2.INTER_LINEAR,
            "keypoint_remapping_method": "direct",
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": 7,
            "fill_mask": 3,
            "map_resolution_range": (0.5, 0.8),
        },
    ),
    (
        "cied-gaussian",
        A.PlanckianJitter,
        {"mode": "cied", "temperature_range": (4000, 12000), "sampling_method": "gaussian"},
    ),
    ("small-plasma", A.PlasmaBrightnessContrast, {"plasma_size": 64, "roughness": 2.0}),
    ("small-plasma", A.PlasmaShadow, {"shadow_intensity_range": (0.2, 0.5), "plasma_size": 64, "roughness": 2.0}),
    ("safe-output", A.RandomBrightnessContrast, {"brightness_by_max": False, "ensure_safe_output": True}),
    (
        "padded-bottom-right",
        A.RandomCrop,
        {
            "pad_if_needed": True,
            "pad_position": "bottom_right",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
        },
    ),
    ("padded", A.RandomCrop3D, {"pad_if_needed": True, "fill": 7, "fill_mask": 3}),
    ("custom-reference-box", A.RandomCropNearBBox, {"max_part_shift": (0.1, 0.2), "cropping_bbox_key": "crop_box"}),
    ("upper-roi", A.RandomGravel, {"gravel_roi": (0.2, 0.2, 0.8, 0.8), "number_of_patches": 3}),
    (
        "downscale-aware",
        A.RandomResizedCrop,
        {"interpolation": cv2.INTER_CUBIC, "mask_interpolation": cv2.INTER_LINEAR, "area_for_downscale": "image_mask"},
    ),
    ("fixed-element", A.RandomRotate90, {"group_element": "r90"}),
    ("downscale-aware", A.RandomScale, {"mask_interpolation": cv2.INTER_LINEAR, "area_for_downscale": "image_mask"}),
    ("variable-intensity", A.RandomShadow, {"shadow_intensity_range": (0.2, 0.7)}),
    (
        "cubic-linear-mask",
        A.RandomSizedBBoxSafeCrop,
        {"interpolation": cv2.INTER_CUBIC, "mask_interpolation": cv2.INTER_LINEAR},
    ),
    (
        "wide-ratio-downscale",
        A.RandomSizedCrop,
        {
            "w2h_ratio": 1.2,
            "interpolation": cv2.INTER_CUBIC,
            "mask_interpolation": cv2.INTER_LINEAR,
            "area_for_downscale": "image_mask",
        },
    ),
    ("texture", A.RandomSnow, {"method": "texture"}),
    ("physics-based", A.RandomSunFlare, {"method": "physics_based"}),
    (
        "downscale-aware",
        A.Resize,
        {"interpolation": cv2.INTER_CUBIC, "mask_interpolation": cv2.INTER_LINEAR, "area_for_downscale": "image_mask"},
    ),
    ("wide-kernel", A.RingingOvershoot, {"blur_range": (9, 17)}),
    (
        "ellipse-cropped",
        A.Rotate,
        {
            "border_mode": cv2.BORDER_REFLECT,
            "rotate_method": "ellipse",
            "crop_border": True,
            "mask_interpolation": cv2.INTER_LINEAR,
            "fill_mask": 3,
        },
    ),
    (
        "ellipse-linear-mask",
        A.SafeRotate,
        {
            "border_mode": cv2.BORDER_REFLECT,
            "rotate_method": "ellipse",
            "mask_interpolation": cv2.INTER_LINEAR,
            "fill_mask": 3,
        },
    ),
    ("gaussian-kernel", A.Sharpen, {"method": "gaussian", "kernel_size": 7, "sigma": 1.5}),
    (
        "ellipse-linear-mask",
        A.ShiftScaleRotate,
        {
            "border_mode": cv2.BORDER_REFLECT,
            "rotate_method": "ellipse",
            "mask_interpolation": cv2.INTER_LINEAR,
            "fill_mask": 3,
        },
    ),
    ("stronger", A.ShotNoise, {"scale_range": (0.2, 0.4)}),
    (
        "max-size-hw-downscale",
        A.SmallestMaxSize,
        {
            "max_size": None,
            "max_size_hw": (80, 90),
            "mask_interpolation": cv2.INTER_LINEAR,
            "area_for_downscale": "image_mask",
        },
    ),
    (
        "custom-rain",
        A.Spatter,
        {
            "mean_range": (0.5, 0.6),
            "std_range": (0.2, 0.25),
            "gauss_sigma_range": (1, 3),
            "cutout_threshold_range": (0.6, 0.7),
            "intensity_range": (0.4, 0.7),
            "color": (200, 200, 200),
        },
    ),
    ("fixed-element", A.SquareSymmetry, {"group_element": "r90"}),
    (
        "augmented-custom-key",
        A.TextImage,
        {
            "augmentations": ("insertion",),
            "fraction_range": (0.5, 0.8),
            "font_size_fraction_range": (0.5, 0.7),
            "clear_bg": True,
            "metadata_key": "text_blocks",
        },
    ),
    (
        "direct-low-resolution",
        A.ThinPlateSpline,
        {
            "scale_range": (0.1, 0.3),
            "num_control_points": 5,
            "interpolation": cv2.INTER_CUBIC,
            "mask_interpolation": cv2.INTER_LINEAR,
            "keypoint_remapping_method": "direct",
            "border_mode": cv2.BORDER_REFLECT,
            "fill": 7,
            "fill_mask": 3,
            "map_resolution_range": (0.5, 0.8),
        },
    ),
    ("single-channel", A.ToGray, {"num_output_channels": 1}),
    ("rgba", A.ToRGB, {"num_output_channels": 4}),
    ("wide-kernel", A.UnsharpMask, {"blur_range": (5, 9)}),
    (
        "direct-low-resolution",
        A.WaterRefraction,
        {
            "num_waves_range": (4, 8),
            "interpolation": cv2.INTER_CUBIC,
            "mask_interpolation": cv2.INTER_LINEAR,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": 7,
            "fill_mask": 3,
            "map_resolution_range": (0.5, 0.8),
        },
    ),
    ("fixed-fill", A.XYMasking, {"fill": 7}),
]


_VARIANT_NAMES: dict[type[A.BasicTransform], tuple[str, ...]] = {
    A.Affine: ("percent-translation", "pixel-translation"),
    A.AtLeastOneBBoxRandomCrop: ("default-erosion", "nonzero-erosion"),
    A.AutoContrast: ("cdf", "pil"),
    A.CoarseDropout: ("pixel-holes", "relative-holes"),
    A.Enhance: ("edge", "detail"),
    A.HEStain: ("vahadane", "macenko", "random-preset"),
    A.LetterBox: ("square-center", "wide-top-left", "tall-bottom-right"),
    A.LongestMaxSize: ("max-size", "max-size-hw"),
    A.PadIfNeeded: ("center", "top-left"),
    A.PixelDropout: ("random-fill", "fixed-fill"),
    A.RandomToneCurve: ("shared", "per-channel"),
    A.ShiftScaleRotate: ("shared-shift", "axis-shift"),
    A.Spatter: ("rain", "mud"),
}

_BBOX_TRANSFORMS = {
    A.AtLeastOneBBoxRandomCrop,
    A.BBoxSafeRandomCrop,
    A.RandomCropNearBBox,
    A.RandomSizedBBoxSafeCrop,
}
_MASK_TRANSFORMS = {
    A.ConstrainedCoarseDropout,
    A.CropNonEmptyMaskIfExists,
    A.MaskDropout,
}
_REFERENCE_METADATA_KEYS = {
    A.FDA: "fda_metadata",
    A.HistogramMatching: "hm_metadata",
    A.PixelDistributionAdaptation: "pda_metadata",
}
_EXACT_TRANSFORMS = {
    A.Blur,
    A.HorizontalFlip,
    A.NoOp,
    A.Pad,
    A.RandomRotate90,
    A.Resize,
    A.Transpose,
    A.VerticalFlip,
}


def _case_data(
    transform_cls: type[A.BasicTransform],
    init_kwargs: Mapping[str, Any],
) -> tuple[ContractDataFactory, dict[str, Any], frozenset[str]]:
    if issubclass(transform_cls, A.Transform3D):
        return make_volume_data, {}, frozenset()
    if transform_cls in _BBOX_TRANSFORMS:
        compose_kwargs = {
            "bbox_params": A.BboxParams(
                coord_format="albumentations",
                label_fields=["bbox_labels"],
            ),
        }
        if transform_cls is A.RandomCropNearBBox:
            key = init_kwargs.get("cropping_bbox_key", "cropping_bbox")
            factory = remap_data_key(make_crop_near_bbox_data, "cropping_bbox", key)
            return factory, compose_kwargs, frozenset({key})
        return make_hbb_data, compose_kwargs, frozenset()
    if transform_cls in _MASK_TRANSFORMS:
        if transform_cls is A.ConstrainedCoarseDropout and init_kwargs.get("bbox_labels") is not None:
            compose_kwargs = {
                "bbox_params": A.BboxParams(
                    coord_format="albumentations",
                    label_fields=["bbox_labels"],
                ),
            }
            return make_hbb_data, compose_kwargs, frozenset()
        return make_mask_data, {}, frozenset()
    if transform_cls in _REFERENCE_METADATA_KEYS:
        key = init_kwargs.get("metadata_key", _REFERENCE_METADATA_KEYS[transform_cls])
        return make_reference_data(key), {}, frozenset({key})
    if transform_cls is A.Mosaic:
        key = init_kwargs.get("metadata_key", "mosaic_metadata")
        return remap_data_key(make_mosaic_data, "mosaic_metadata", key), {}, frozenset({key})
    if transform_cls is A.CopyAndPaste:
        key = init_kwargs.get("metadata_key", "copy_paste_metadata")
        return remap_data_key(make_copy_and_paste_data, "copy_paste_metadata", key), {}, frozenset({key})
    if transform_cls is A.OverlayElements:
        key = init_kwargs.get("metadata_key", "overlay_metadata")
        return remap_data_key(make_overlay_data, "overlay_metadata", key), {}, frozenset({key})
    if transform_cls is A.TextImage:
        key = init_kwargs.get("metadata_key", "textimage_metadata")
        return remap_data_key(make_text_data, "textimage_metadata", key), {}, frozenset({key})
    if transform_cls in {A.Colorize, A.ToRGB}:
        return make_grayscale_image_data, {}, frozenset()
    if transform_cls is A.FromFloat:
        return make_float_image_data, {}, frozenset()
    if transform_cls is A.Equalize and init_kwargs.get("mask_params"):
        return make_mask_data, {}, frozenset()
    return make_image_data, {}, frozenset()


def _slugify(name: str) -> str:
    words = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", words).replace("_", "-").lower()


def _iter_base_cases() -> list[TransformContractCase]:
    cases: list[TransformContractCase] = []
    seen_counts: dict[type[A.BasicTransform], int] = {}
    for transform_cls, raw_params in _BASE_CASE_SPECS:
        parameter_sets = raw_params if isinstance(raw_params, list) else [raw_params]
        variants = _VARIANT_NAMES.get(transform_cls)
        for params in parameter_sets:
            data_factory, compose_kwargs, metadata_keys = _case_data(transform_cls, params)
            index = seen_counts.get(transform_cls, 0)
            seen_counts[transform_cls] = index + 1
            if variants is not None and index >= len(variants):
                raise ValueError(f"{transform_cls.__name__}: missing a name for variant {index}")
            variant = variants[index] if variants is not None else "default"
            case_id = f"{_slugify(transform_cls.__name__)}-{variant}"
            cases.append(
                TransformContractCase(
                    case_id=case_id,
                    transform_cls=transform_cls,
                    init_kwargs=params,
                    data_factory=data_factory,
                    compose_kwargs=compose_kwargs,
                    replay_profile=(
                        ReplayProfile.EXACT if transform_cls in _EXACT_TRANSFORMS else ReplayProfile.RUNNABLE
                    ),
                    metadata_keys=metadata_keys,
                ),
            )
    for transform_cls, variants in _VARIANT_NAMES.items():
        if seen_counts.get(transform_cls) != len(variants):
            raise ValueError(
                f"{transform_cls.__name__}: expected {len(variants)} variants, got {seen_counts.get(transform_cls, 0)}",
            )
    return cases


def _first_base_kwargs(transform_cls: type[A.BasicTransform]) -> dict[str, Any]:
    for candidate_cls, raw_params in _BASE_CASE_SPECS:
        if candidate_cls is not transform_cls:
            continue
        if isinstance(raw_params, list):
            return copy.deepcopy(raw_params[0])
        return copy.deepcopy(raw_params)
    raise ValueError(f"{transform_cls.__name__} has no base transform contract case")


def _iter_parameter_mode_cases() -> list[TransformContractCase]:
    cases = []
    for variant, transform_cls, overrides in _PARAMETER_MODE_SPECS:
        params = _first_base_kwargs(transform_cls)
        params.update(copy.deepcopy(overrides))
        data_factory, compose_kwargs, metadata_keys = _case_data(transform_cls, params)
        cases.append(
            TransformContractCase(
                case_id=f"{_slugify(transform_cls.__name__)}-{variant}",
                transform_cls=transform_cls,
                init_kwargs=params,
                data_factory=data_factory,
                compose_kwargs=compose_kwargs,
                metadata_keys=metadata_keys,
            ),
        )
    return cases


TRANSFORM_CONTRACT_CASES = (
    *_iter_base_cases(),
    *_iter_parameter_mode_cases(),
    TransformContractCase(
        case_id="random-rotate90-subset",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        replay_profile=ReplayProfile.EXACT,
    ),
    TransformContractCase(
        case_id="random-rotate90-subset-mask",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        data_factory=make_mask_data,
        replay_profile=ReplayProfile.EXACT,
    ),
    TransformContractCase(
        case_id="random-rotate90-subset-hbb",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        data_factory=make_hbb_data,
        compose_kwargs={
            "bbox_params": A.BboxParams(
                coord_format="albumentations",
                label_fields=["bbox_labels"],
            ),
        },
        replay_profile=ReplayProfile.EXACT,
    ),
    TransformContractCase(
        case_id="random-rotate90-subset-obb",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        data_factory=make_obb_data,
        compose_kwargs={
            "bbox_params": A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        },
        replay_profile=ReplayProfile.EXACT,
    ),
    TransformContractCase(
        case_id="random-rotate90-subset-keypoints",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        data_factory=make_keypoint_data,
        compose_kwargs={
            "keypoint_params": A.KeypointParams(
                coord_format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={},
            ),
        },
        replay_profile=ReplayProfile.EXACT,
    ),
    TransformContractCase(
        case_id="random-rotate90-subset-volume",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        data_factory=make_volume_data,
        replay_profile=ReplayProfile.EXACT,
    ),
    TransformContractCase(
        case_id="random-rotate90-subset-image-batch",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        data_factory=make_image_batch_data,
        replay_profile=ReplayProfile.EXACT,
    ),
    TransformContractCase(
        case_id="random-rotate90-subset-volume-batch",
        transform_cls=A.RandomRotate90,
        init_kwargs={"group_elements": ("r90", "r270")},
        data_factory=make_volume_batch_data,
        replay_profile=ReplayProfile.EXACT,
    ),
)

TRANSFORM_CASES_BY_CLASS: dict[type[A.BasicTransform], tuple[TransformContractCase, ...]] = {
    transform_cls: tuple(case for case in TRANSFORM_CONTRACT_CASES if case.transform_cls is transform_cls)
    for transform_cls in {case.transform_cls for case in TRANSFORM_CONTRACT_CASES}
}

PRIMARY_TRANSFORM_CONTRACT_CASES = tuple(
    TRANSFORM_CASES_BY_CLASS[transform_cls][0]
    for transform_cls in dict.fromkeys(case.transform_cls for case in TRANSFORM_CONTRACT_CASES)
)

PRIMARY_TRANSFORM_CASE_BY_CLASS = {case.transform_cls: case for case in PRIMARY_TRANSFORM_CONTRACT_CASES}
