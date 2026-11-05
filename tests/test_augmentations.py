import cv2
import numpy as np
import pytest
from albucore import convert_dtype

import albumentations as A
from albumentations.augmentations.pixel import _functional_weather as fweather
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.pixel import weather
from albumentations.core.invocation import SamplingContext
from albumentations.core.tensor import numpy_to_tensor_spatial, tensor_to_numpy_spatial
from albumentations.core.transforms_interface import ImageOnlyTransform
from tests.conftest import (
    IMAGES,
    RECTANGULAR_UINT8_IMAGE,
    SQUARE_FLOAT_IMAGE,
    SQUARE_MULTI_FLOAT_IMAGE,
    SQUARE_MULTI_UINT8_IMAGE,
    SQUARE_UINT8_IMAGE,
)
from tests.helpers import TransformTestHelper

from .utils import (
    get_primary_2d_transform_params,
    get_primary_dual_transform_params,
    get_primary_image_only_transform_params,
    get_resolved_applied_params,
    make_sampling_args,
    set_seed,
)


def _add_guided_dropout_region(data, augmentation_cls) -> None:
    if augmentation_cls is A.GuidedCoarseDropout:
        data["dropout_region"] = np.ones(data["image"].shape[:2], dtype=np.uint8)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_image_only_transform_params(
        except_augmentations={
            A.FromFloat,
            A.Normalize,
            A.ToFloat,
        },
    ),
)
def test_image_only_augmentations_mask_persists(augmentation_cls, params):
    image = SQUARE_UINT8_IMAGE
    mask = image.copy()

    data = TransformTestHelper.prepare_test_data(augmentation_cls, image, mask=mask)

    aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)

    data = aug(**data)

    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype
    np.testing.assert_array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_image_only_transform_params(
        except_augmentations={
            A.FromFloat,
        },
    ),
)
def test_image_only_augmentations(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE
    mask = image[:, :, 0].copy().astype(np.uint8)

    data = {
        "image": image,
        "mask": mask,
    }
    if augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": SQUARE_FLOAT_IMAGE,
                "mask": mask,
            },
        ]
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data[TransformTestHelper.METADATA_KEYS[augmentation_cls]] = [image]
        aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)
        data = aug(**data)
    else:
        aug = augmentation_cls(p=1, **params)
        data = aug(**data)

    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype
    np.testing.assert_array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_dual_transform_params(
        custom_arguments={},
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_dual_augmentations(augmentation_cls, params):
    image = SQUARE_UINT8_IMAGE
    mask = np.expand_dims(image[:, :, 0].copy(), axis=-1)

    # Use helper to prepare data with metadata
    data = TransformTestHelper.prepare_test_data(augmentation_cls, image, mask=mask)

    # Handle special case for RandomCropNearBBox
    if augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]

    aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)
    data = aug(**data)

    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_dual_transform_params(
        custom_arguments={},
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_dual_augmentations_with_float_values(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE
    mask = np.expand_dims(image.copy()[:, :, 0].astype(np.uint8), axis=-1)

    # Use helper to prepare data with metadata
    data = TransformTestHelper.prepare_test_data(augmentation_cls, image, mask=mask)

    # Handle special case for RandomCropNearBBox
    if augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]

    aug = augmentation_cls(p=1, **params)
    data = aug(**data)

    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_augmentations_wont_change_input(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE if augmentation_cls == A.FromFloat else SQUARE_UINT8_IMAGE
    mask = np.expand_dims(image[:, :, 0].copy(), axis=-1)
    image_copy = image.copy()
    mask_copy = mask.copy()
    aug = augmentation_cls(p=1, **params)

    data = {"image": image, "mask": mask}

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.CopyAndPaste:
        data["copy_paste_metadata"] = []
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": SQUARE_UINT8_IMAGE,
                "mask": mask,
            },
        ]
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data[TransformTestHelper.METADATA_KEYS[augmentation_cls]] = [image]

    _add_guided_dropout_region(data, augmentation_cls)
    aug(**data)

    np.testing.assert_array_equal(image, image_copy)
    np.testing.assert_array_equal(mask, mask_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_augmentations_wont_change_float_input(augmentation_cls, params, image_float32):
    float_image_copy = image_float32.copy()

    aug = augmentation_cls(p=1, **params)

    data = {"image": image_float32}

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.CopyAndPaste:
        data["copy_paste_metadata"] = []
    elif augmentation_cls in (A.MaskDropout, A.ConstrainedCoarseDropout):
        mask = np.zeros((image_float32.shape[0], image_float32.shape[1], 1), dtype=np.uint8)
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image_float32,
            },
        ]
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data[TransformTestHelper.METADATA_KEYS[augmentation_cls]] = [image_float32]

    _add_guided_dropout_region(data, augmentation_cls)
    aug(**data)

    np.testing.assert_array_equal(image_float32, float_image_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CenterCrop,
            A.Crop,
            A.CropNonEmptyMaskIfExists,
            A.RandomCrop,
            A.AtLeastOneBBoxRandomCrop,
            A.RandomResizedCrop,
            A.RandomSizedCrop,
            A.CropAndPad,
            A.Resize,
            A.LongestMaxSize,
            A.LetterBox,
            A.SmallestMaxSize,
            A.PadIfNeeded,
            A.RandomScale,
            A.RandomCropFromBorders,
            A.ConstrainedCoarseDropout,
            A.Pad,
            A.Mosaic,
            A.MaskDropout,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_augmentations_wont_change_shape_rgb(augmentation_cls, params):
    image_3ch = SQUARE_UINT8_IMAGE
    mask_3ch = np.zeros_like(image_3ch)

    aug = augmentation_cls(p=1, **params)

    if augmentation_cls == A.OverlayElements:
        data = {
            "image": image_3ch,
            "overlay_metadata": [],
            "mask": mask_3ch,
        }
    elif augmentation_cls == A.CopyAndPaste:
        data = {
            "image": image_3ch,
            "copy_paste_metadata": [],
            "mask": mask_3ch,
        }
    elif augmentation_cls == A.FromFloat:
        data = {
            "image": SQUARE_FLOAT_IMAGE,
            "mask": mask_3ch,
        }
    elif augmentation_cls == A.Mosaic:
        data = {
            "image": image_3ch,
            "mask": mask_3ch,
            "mosaic_metadata": [
                {
                    "image": image_3ch,
                    "mask": mask_3ch,
                },
            ],
        }
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data = {
            "image": image_3ch,
            "mask": mask_3ch,
            TransformTestHelper.METADATA_KEYS[augmentation_cls]: [image_3ch],
        }
    else:
        data = {
            "image": image_3ch,
            "mask": mask_3ch,
        }
    _add_guided_dropout_region(data, augmentation_cls)
    result = aug(**data)

    np.testing.assert_array_equal(image_3ch.shape, result["image"].shape)
    np.testing.assert_array_equal(mask_3ch.shape, result["mask"].shape)


@pytest.mark.parametrize(["augmentation_cls", "params"], [[A.RandomCropNearBBox, {"max_part_shift": (0.15, 0.15)}]])
@pytest.mark.parametrize("image", IMAGES)
def test_image_only_crop_around_bbox_augmentation(augmentation_cls, params, image):
    aug = augmentation_cls(p=1, **params)
    annotations = {"image": image, "cropping_bbox": [-59, 77, 177, 231]}
    data = aug(**annotations)
    assert data["image"].dtype == image.dtype


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [
            A.PadIfNeeded,
            {"min_height": 514, "min_width": 514, "border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1},
        ],
        [A.Rotate, {"border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1}],
        [A.SafeRotate, {"border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1}],
        [A.ShiftScaleRotate, {"border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1}],
        [A.Affine, {"border_mode": cv2.BORDER_CONSTANT, "fill_mask": 1, "fill": 100}],
    ],
)
def test_mask_fill_value(augmentation_cls, params):
    set_seed(137)
    aug = augmentation_cls(p=1, **params)
    input = {"image": np.zeros((512, 512, 1), dtype=np.uint8) + 100, "mask": np.ones((512, 512, 1))}
    output = aug(**input)
    assert (output["image"] == 100).all()
    assert (output["mask"] == 1).all()


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        custom_arguments={
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.Normalize,
            A.PhotoMetricDistort,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToFloat,
            A.ToRGB,
            A.ToSepia,
            A.Colorize,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomSunFlare,
            A.LensFlare,
            A.RandomFog,
            A.Equalize,
            A.GridElasticDeform,
            A.HEStain,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_multichannel_image_augmentations(augmentation_cls, params):
    image = SQUARE_MULTI_UINT8_IMAGE
    aug = augmentation_cls(p=1, **params)

    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.CopyAndPaste:
        data["copy_paste_metadata"] = []
    elif augmentation_cls in (A.MaskDropout, A.ConstrainedCoarseDropout):
        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            },
        ]
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data[TransformTestHelper.METADATA_KEYS[augmentation_cls]] = [image]

    _add_guided_dropout_region(data, augmentation_cls)
    data = aug(**data)
    assert data["image"].dtype == np.uint8
    assert data["image"].shape[2] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        custom_arguments={
            A.Normalize: {
                "mean": (0.5, 0.5, 0.5, 0.5, 0.5),
                "std": (0.25, 0.25, 0.25, 0.25, 0.25),
            },
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.PhotoMetricDistort,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToRGB,
            A.ToSepia,
            A.Colorize,
            A.Equalize,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomSunFlare,
            A.LensFlare,
            A.RandomFog,
            A.GridElasticDeform,
            A.HEStain,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_float_multichannel_image_augmentations(augmentation_cls, params):
    image = SQUARE_MULTI_FLOAT_IMAGE
    aug = augmentation_cls(p=1, **params)
    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.CopyAndPaste:
        data["copy_paste_metadata"] = []
    elif augmentation_cls in (A.MaskDropout, A.ConstrainedCoarseDropout):
        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            },
        ]
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data[TransformTestHelper.METADATA_KEYS[augmentation_cls]] = [image]

    _add_guided_dropout_region(data, augmentation_cls)
    data = aug(**data)

    assert data["image"].dtype == np.float32
    assert data["image"].shape[-1] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        custom_arguments={
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.Normalize,
            A.PhotoMetricDistort,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToFloat,
            A.ToRGB,
            A.ToSepia,
            A.Colorize,
            A.FancyPCA,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomSunFlare,
            A.LensFlare,
            A.RandomFog,
            A.Equalize,
            A.GridElasticDeform,
            A.HEStain,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    image = SQUARE_MULTI_UINT8_IMAGE

    aug = augmentation_cls(p=1, **params)

    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.CopyAndPaste:
        data["copy_paste_metadata"] = []
    elif augmentation_cls in (A.MaskDropout, A.ConstrainedCoarseDropout):
        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            },
        ]
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data[TransformTestHelper.METADATA_KEYS[augmentation_cls]] = [image]

    _add_guided_dropout_region(data, augmentation_cls)
    data = aug(**data)

    assert data["image"].dtype == np.uint8
    assert data["image"].shape[-1] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        custom_arguments={
            A.Normalize: {
                "mean": (0.5, 0.5, 0.5, 0.5, 0.5),
                "std": (0.25, 0.25, 0.25, 0.25, 0.25),
            },
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
            },
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.PhotoMetricDistort,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToRGB,
            A.ToSepia,
            A.Colorize,
            A.Equalize,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomSunFlare,
            A.LensFlare,
            A.RandomFog,
            A.GridElasticDeform,
            A.HEStain,
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_float_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    image = SQUARE_MULTI_FLOAT_IMAGE
    aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)

    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.CopyAndPaste:
        data["copy_paste_metadata"] = []
    elif augmentation_cls in (A.MaskDropout, A.ConstrainedCoarseDropout):
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            },
        ]
    elif augmentation_cls in TransformTestHelper.METADATA_KEYS:
        data[TransformTestHelper.METADATA_KEYS[augmentation_cls]] = [image]

    _add_guided_dropout_region(data, augmentation_cls)
    data = aug(**data)

    assert data["image"].dtype == np.float32
    assert data["image"].shape[2] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params", "image_shape"],
    [
        [A.PadIfNeeded, {"min_height": 514, "min_width": 516}, (300, 200, 1)],
        [A.PadIfNeeded, {"min_height": 514, "min_width": 516}, (512, 516, 1)],
        [A.PadIfNeeded, {"min_height": 514, "min_width": 516}, (600, 600, 1)],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 128, "pad_width_divisor": 128},
            (300, 200, 1),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (72, 128, 1),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (15, 15, 1),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (144, 256, 1),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (200, 300, 1),
        ],
        [A.PadIfNeeded, {"min_height": 512, "min_width": None, "pad_width_divisor": 128}, (300, 200, 1)],
        [A.PadIfNeeded, {"min_height": None, "min_width": 512, "pad_height_divisor": 128}, (300, 200, 1)],
    ],
)
def test_pad_if_needed(augmentation_cls: type[A.PadIfNeeded], params: dict, image_shape: tuple[int, int]):
    image = np.zeros(image_shape)
    pad = augmentation_cls(**params)

    image_padded = pad(image=image)["image"]

    if pad.min_width is not None:
        assert image_padded.shape[1] >= pad.min_width

    if pad.min_height is not None:
        assert image_padded.shape[0] >= pad.min_height

    if pad.pad_width_divisor is not None:
        assert image_padded.shape[1] % pad.pad_width_divisor == 0
        assert image_padded.shape[1] >= image.shape[1]
        assert image_padded.shape[1] - image.shape[1] <= pad.pad_width_divisor

    if pad.pad_height_divisor is not None:
        assert image_padded.shape[0] % pad.pad_height_divisor == 0
        assert image_padded.shape[0] >= image.shape[0]
        assert image_padded.shape[0] - image.shape[0] <= pad.pad_height_divisor


@pytest.mark.parametrize(
    ["params", "image_shape"],
    [
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "fill": 1, "position": "center"},
            (5, 6, 1),
        ],
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "fill": 1, "position": "top_left"},
            (5, 6, 1),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 1,
                "position": "top_right",
            },
            (5, 6, 1),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 1,
                "position": "bottom_left",
            },
            (5, 6, 1),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 1,
                "position": "bottom_right",
            },
            (5, 6, 1),
        ],
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "fill": 1, "position": "random"},
            (5, 6, 1),
        ],
    ],
)
def test_pad_if_needed_position(params, image_shape):
    image = np.zeros(image_shape)
    pad = A.PadIfNeeded(**params)
    pad.set_random_seed(0)

    transformed = pad(image=image)
    image_padded = transformed["image"]

    true_result = np.ones((max(image_shape[0], params["min_height"]), max(image_shape[1], params["min_width"]), 1))

    if params["position"] == "center":
        x_start = image_shape[0] // 2
        y_start = image_shape[1] // 2
        true_result[x_start : x_start + image_shape[0], y_start : y_start + image_shape[1]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "top_left":
        true_result[: image_shape[0], : image_shape[1]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "top_right":
        true_result[: image_shape[0], -image_shape[1] :] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "bottom_left":
        true_result[-image_shape[0] :, : image_shape[1]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "bottom_right":
        true_result[-image_shape[0] :, -image_shape[1] :] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "random":
        # Find where the original image was placed (where pixels are 0)
        zero_mask = image_padded == 0

        # Get the bounds of the zero region
        zero_rows = np.where(zero_mask.any(axis=1))[0]
        zero_cols = np.where(zero_mask.any(axis=0))[0]

        # Check that the zero region is contiguous and of correct size
        assert len(zero_rows) == image_shape[0], "Height of placed image incorrect"
        assert len(zero_cols) == image_shape[1], "Width of placed image incorrect"
        np.testing.assert_array_equal(np.diff(zero_rows), 1, err_msg="Image placement not contiguous in height")
        np.testing.assert_array_equal(np.diff(zero_cols), 1, err_msg="Image placement not contiguous in width")

        # Verify the rest of the image is filled with ones
        padded_mask = np.ones_like(true_result)
        padded_mask[zero_rows[0] : zero_rows[-1] + 1, zero_cols[0] : zero_cols[-1] + 1] = 0
        np.testing.assert_array_equal(image_padded[padded_mask == 1], 1, err_msg="Padding value incorrect")


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_primary_2d_transform_params(
        custom_arguments={
            A.ShiftScaleRotate: {
                "fill": 0,
                "interpolation": cv2.INTER_NEAREST,
            },
            A.SafeRotate: {
                "interpolation": cv2.INTER_NEAREST,
                "fill": 0,
            },
            A.Rotate: {
                "interpolation": cv2.INTER_NEAREST,
                "fill": 0,
            },
            A.RandomScale: {
                "scale_range": (-0.2, 0.2),
                "interpolation": cv2.INTER_NEAREST,
            },
            A.Affine: {
                "interpolation": cv2.INTER_NEAREST,
                "fill": 0,
            },
            A.PixelDropout: {
                "drop_value": 0,
            },
            A.PadIfNeeded: {
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 0,
            },
            A.ChannelDropout: {
                "fill": 0,
            },
            A.PiecewiseAffine: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.Perspective: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.GridDropout: {
                "fill": 0,
            },
            A.GridDistortion: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.ElasticTransform: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.Pad: {
                "fill": 0,
            },
            A.LetterBox: {
                "size": (128, 128),
                "fill": 0,
                "fill_mask": 0,
            },
            A.Resize: {
                "interpolation": cv2.INTER_NEAREST,
                "height": 50,
                "width": 50,
            },
            A.CropAndPad: {
                "fill": 0,
                "px": 10,
            },
            A.OpticalDistortion: {
                "interpolation": cv2.INTER_NEAREST,
            },
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.FromFloat,
            A.ToFloat,
            A.Normalize,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.OverlayElements,
            A.RGBShift,
            A.HueSaturationValue,
            A.ColorJitter,
            A.PhotoMetricDistort,
            A.Mosaic,
            A.Dithering,  # Error diffusion is sensitive to floating-point precision
            A.RandomSnow,  # OpenCV HLS/HSV quantization differences
            A.BBoxSubsetSafeRandomCrop,
        },
    ),
)
def test_augmentations_match_uint8_float32(augmentation_cls, params):
    image_uint8 = RECTANGULAR_UINT8_IMAGE
    image_float32 = convert_dtype(image_uint8, np.float32)

    transform = A.Compose([augmentation_cls(p=1, **params)], seed=137, strict=True)

    data = {"image": image_uint8}
    if augmentation_cls in (A.MaskDropout, A.ConstrainedCoarseDropout):
        mask = np.zeros_like(image_uint8)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [12, 77, 177, 231]
    elif augmentation_cls == A.CopyAndPaste:
        data["copy_paste_metadata"] = []

    _add_guided_dropout_region(data, augmentation_cls)
    transformed_uint8 = transform(**data)["image"]

    data["image"] = image_float32

    transform.set_random_seed(137)
    transformed_float32 = transform(**data)["image"]

    np.testing.assert_array_almost_equal(convert_dtype(transformed_uint8, np.float32), transformed_float32, decimal=2)


def test_solarize_threshold():
    image = SQUARE_UINT8_IMAGE
    image[20:40, 20:40] = 255
    transform = A.Solarize(threshold_range=(0.5, 0.5), p=1)
    transformed_image = transform(image=image)["image"]
    assert (transformed_image[20:40, 20:40] == 0).all()

    transform = A.Solarize(threshold_range=(0.5, 0.5), p=1)

    float_image = SQUARE_FLOAT_IMAGE
    float_image[20:40, 20:40] = 1
    transformed_image = transform(image=float_image)["image"]
    assert (transformed_image[20:40, 20:40] == 0).all()


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.float32],
)
def test_solarize_apply_to_images(dtype):
    if dtype == np.uint8:
        images = np.random.RandomState(137).randint(0, 256, (2, 100, 100, 3), dtype=np.uint8)
    else:
        images = np.random.RandomState(137).random((2, 100, 100, 3)).astype(np.float32)

    threshold = 0.5
    transform = A.Solarize(threshold_range=(threshold, threshold), p=1.0)

    transformed = transform(images=images)["images"]

    assert transformed.shape == images.shape
    assert transformed.dtype == images.dtype

    expected = np.stack([transform(image=images[i])["image"] for i in range(images.shape[0])])
    np.testing.assert_array_equal(transformed, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_zoom_blur_apply_to_images(dtype):
    if dtype == np.uint8:
        images = np.random.RandomState(137).randint(0, 256, (2, 100, 100, 3), dtype=np.uint8)
    else:
        images = np.random.RandomState(137).random((2, 100, 100, 3)).astype(dtype)

    # Use fixed parameters so parameter generation produces deterministic zoom_factors
    transform = A.ZoomBlur(max_factor_range=(1.1, 1.1), step_factor_range=(0.01, 0.01), p=1.0)

    transformed = transform(images=images)["images"]

    assert transformed.shape == images.shape
    assert transformed.dtype == images.dtype

    expected = np.stack([transform(image=images[i])["image"] for i in range(images.shape[0])])
    np.testing.assert_array_equal(transformed, expected)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_zoom_blur_float16_matches_float32_reference(channels, container):
    half_image = np.random.default_rng(137).random((32, 40, channels), dtype=np.float32).astype(np.float16)
    float32_image = half_image.astype(np.float32)
    half_source = numpy_to_tensor_spatial(half_image.copy(), "image") if container == "tensor" else half_image.copy()
    float32_source = (
        numpy_to_tensor_spatial(float32_image.copy(), "image") if container == "tensor" else float32_image.copy()
    )
    if container == "numpy":
        half_source.flags.writeable = False
        float32_source.flags.writeable = False

    def make_pipeline():
        return A.Compose(
            [A.ZoomBlur(max_factor_range=(1.1, 1.1), step_factor_range=(0.01, 0.01), p=1.0)],
            seed=137,
            strict=True,
        )

    half_result = make_pipeline()(image=half_source)["image"]
    float32_result = make_pipeline()(image=float32_source)["image"]
    if container == "tensor":
        half_result = tensor_to_numpy_spatial(half_result, "image")
        float32_result = tensor_to_numpy_spatial(float32_result, "image")
        half_source = tensor_to_numpy_spatial(half_source, "image")
        float32_source = tensor_to_numpy_spatial(float32_source, "image")

    assert half_result.shape == half_image.shape
    assert half_result.dtype == np.float16
    np.testing.assert_array_equal(half_result, float32_result.astype(np.float16))
    np.testing.assert_array_equal(half_source, half_image)
    np.testing.assert_array_equal(float32_source, float32_image)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_planckian_jitter_float16_compose_clips_and_matches_float32(container):
    image = np.broadcast_to(np.array([0.05, 0.95, 0.5], dtype=np.float16), (16, 17, 3)).copy()
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    def make_pipeline():
        return A.Compose(
            [A.PlanckianJitter(mode="blackbody", temperature_range=(3000, 6500), sampling_method="uniform", p=1.0)],
            seed=137,
            strict=True,
        )

    half_result = make_pipeline()(image=source)["image"]
    actual = tensor_to_numpy_spatial(half_result, "image") if container == "tensor" else half_result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    float32_image = image.astype(np.float32)
    float32_source = numpy_to_tensor_spatial(float32_image.copy(), "image") if container == "tensor" else float32_image
    float32_result = make_pipeline()(image=float32_source)["image"]
    if container == "tensor":
        float32_result = tensor_to_numpy_spatial(float32_result, "image")

    assert actual.shape == image.shape
    assert actual.dtype == np.float16
    assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_allclose(actual, float32_result.astype(np.float16), rtol=0, atol=1 / 2048)
    np.testing.assert_array_equal(preserved, original)


@pytest.mark.parametrize(
    ("transform_cls", "kwargs"),
    [
        (A.ZoomBlur, {"max_factor_range": (1.1, 1.1), "step_factor_range": (0.01, 0.01)}),
        (A.UnsharpMask, {"blur_range": (3, 3), "sigma_range": (0.5, 0.5), "alpha_range": (0.3, 0.3), "threshold": 10}),
        (
            A.Sharpen,
            {
                "alpha_range": (0.3, 0.3),
                "lightness_range": (0.7, 0.7),
                "method": "kernel",
                "kernel_size": 5,
                "sigma": 1.0,
            },
        ),
        (
            A.Sharpen,
            {
                "alpha_range": (0.3, 0.3),
                "lightness_range": (0.7, 0.7),
                "method": "gaussian",
                "kernel_size": 5,
                "sigma": 1.0,
            },
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_image_only_transforms_apply_to_volume(transform_cls, kwargs, dtype):
    shape = (4, 32, 32, 3)
    if dtype == np.uint8:
        volume = np.random.RandomState(137).randint(0, 256, shape, dtype=np.uint8)
    else:
        volume = np.random.RandomState(137).random(shape).astype(np.float32)

    actual = A.Compose([transform_cls(**kwargs, p=1.0)], strict=True)(volume=volume)["volume"]
    expected = A.Compose([transform_cls(**kwargs, p=1.0)], strict=True)(images=volume)["images"]

    np.testing.assert_array_equal(actual, expected)


def test_constrained_coarse_dropout_with_mask():
    """Test ConstrainedCoarseDropout with segmentation mask."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100, 1), dtype=np.uint8)

    mask[10:30, 10:30] = 1  # First object (class 1)
    mask[40:60, 40:60] = 2  # Second object (class 2)
    mask[70:90, 70:90] = 2  # Third object (class 2)

    transform = A.ConstrainedCoarseDropout(
        num_holes_range=(2, 2),  # Fixed 2 holes per object
        hole_height_range=(0.3, 0.3),  # Fixed 30% of object height
        hole_width_range=(0.3, 0.3),  # Fixed 30% of object width
        mask_indices=[1, 2],
        p=1.0,
    )
    transform.set_random_seed(137)

    _ = transform(image=image, mask=mask)

    # Get holes
    params = transform.sample_parameters(
        *make_sampling_args(transform, {"image": image, "mask": mask}),
        SamplingContext.from_owner(transform, {}),
    ).params
    holes = params["holes"]

    # Verify number of holes (2 per object, 3 objects)
    assert len(holes) == 6, f"Expected 6 holes (2 per object), got {len(holes)}"

    # Verify holes are within image bounds
    for hole in holes:
        x1, y1, x2, y2 = hole
        assert 0 <= x1 < x2 <= 100, f"Invalid hole x coordinates: {x1}, {x2}"
        assert 0 <= y1 < y2 <= 100, f"Invalid hole y coordinates: {y1}, {y2}"


@pytest.mark.parametrize(
    ["bbox_labels", "bboxes", "expected_num_objects"],
    [
        # Case 1: String labels
        (
            ["Billy The Cat", "dog"],
            [
                [10, 10, 20, 20, "Billy The Cat"],
                [30, 30, 40, 40, "dog"],
                [50, 50, 60, 60, "bird"],  # Should be ignored
            ],
            2,  # 2 objects: one cat, one dog
        ),
        # Case 2: Numeric labels
        (
            [1, 2],
            [
                [10, 10, 20, 20, 1],
                [30, 30, 40, 40, 2],
                [50, 50, 60, 60, 3],  # Should be ignored
            ],
            2,  # 2 objects: class 1 and class 2
        ),
    ],
)
def test_constrained_coarse_dropout_with_bboxes(bbox_labels, bboxes, expected_num_objects):
    """Test ConstrainedCoarseDropout with bounding boxes."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    ccd = A.ConstrainedCoarseDropout(
        num_holes_range=(2, 2),  # Fixed 2 holes per object
        hole_height_range=(0.3, 0.3),  # Fixed 30% of object height
        hole_width_range=(0.3, 0.3),  # Fixed 30% of object width
        bbox_labels=bbox_labels,
        p=1.0,
    )
    transform = A.Compose(
        [ccd],
        strict=True,
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_labels"]),
        save_applied_params=True,
        seed=137,
    )

    # Extract labels for bbox_params
    labels = [bbox[4] for bbox in bboxes]
    bboxes_without_labels = [bbox[:4] for bbox in bboxes]

    transform(image=image, bboxes=bboxes_without_labels, class_labels=labels)

    holes = get_resolved_applied_params(ccd)["holes"]

    # Verify number of holes (2 per object)
    assert len(holes) == expected_num_objects * 2, (
        f"Expected {expected_num_objects * 2} holes (2 per object), got {len(holes)}"
    )

    # Verify holes are within image bounds
    for hole in holes:
        x1, y1, x2, y2 = hole
        assert 0 <= x1 < x2 <= 100, f"Invalid hole x coordinates: {x1}, {x2}"
        assert 0 <= y1 < y2 <= 100, f"Invalid hole y coordinates: {y1}, {y2}"

    # Verify holes overlap with target boxes
    target_boxes = [bbox[:4] for bbox, label in zip(bboxes, labels, strict=False) if label in bbox_labels]

    for hole in holes:
        overlaps_any = False
        for box in target_boxes:
            # Check for overlap
            if not (
                hole[2] <= box[0]  # hole right < box left
                or hole[0] >= box[2]  # hole left > box right
                or hole[3] <= box[1]  # hole bottom < box top
                or hole[1] >= box[3]
            ):  # hole top > box bottom
                overlaps_any = True
                break
        assert overlaps_any, f"Hole {hole} doesn't overlap with any target box"


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("compression_type", ["jpeg", "webp"])
def test_image_compression_compose_preserves_singleton_channel(dtype, compression_type):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (33, 29, 1), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((33, 29, 1), dtype=np.float32).astype(dtype)
    )
    original = image.copy()
    image_type = ".jpg" if compression_type == "jpeg" else ".webp"
    transform = A.Compose(
        [A.ImageCompression(quality_range=(75, 75), compression_type=compression_type, p=1.0)],
        seed=137,
    )

    result = transform(image=image)["image"]
    expected = fpixel.image_compression(image, 75, image_type)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("compression_type", ["jpeg", "webp"])
def test_image_compression_float16_matches_float32_reference(target, container, channels, compression_type):
    shapes = {
        "image": (33, 29, channels),
        "images": (2, 33, 29, channels),
        "volume": (3, 33, 29, channels),
        "volumes": (2, 3, 33, 29, channels),
    }
    half_image = np.random.default_rng(137).random(shapes[target], dtype=np.float32).astype(np.float16)
    original = half_image.copy()
    params = {"quality_range": (75, 75), "compression_type": compression_type, "p": 1.0}

    def make_source(image):
        source = image.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(image):
        source = make_source(image)
        pipeline = A.Compose([A.ImageCompression(**params)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    half_result, half_source = run(half_image)
    float32_result, _ = run(half_image.astype(np.float32))
    if container == "tensor":
        half_result = tensor_to_numpy_spatial(half_result, target)
        float32_result = tensor_to_numpy_spatial(float32_result, target)
        half_source = tensor_to_numpy_spatial(half_source, target)

    assert half_result.shape == half_image.shape
    assert half_result.dtype == np.float16
    np.testing.assert_array_equal(half_result, float32_result.astype(np.float16))
    np.testing.assert_array_equal(half_source, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_superpixels_compose_preserves_singleton_channel(dtype):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (32, 29, 1), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((32, 29, 1), dtype=np.float32).astype(dtype)
    )
    original = image.copy()
    transform = A.Compose(
        [
            A.Superpixels(
                p_replace_range=(1.0, 1.0),
                n_segments_range=(8, 8),
                max_size=24,
                interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_superpixels_compose_supports_all_targets_and_dtypes(target, container, channels, dtype):
    shapes = {
        "image": (32, 29, channels),
        "images": (2, 32, 29, channels),
        "volume": (3, 32, 29, channels),
        "volumes": (2, 3, 32, 29, channels),
    }
    rng = np.random.default_rng(137)
    if dtype is np.uint8:
        image = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    else:
        half = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
        image = half if dtype is np.float16 else half.astype(np.float32)
    original = image.copy()
    params = {"p_replace_range": (1.0, 1.0), "n_segments_range": (8, 8), "max_size": 24, "p": 1.0}

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose([A.Superpixels(**params)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source
    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(preserved, original)

    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_fancy_pca_compose_preserves_singleton_channel(dtype):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (31, 29, 1), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((31, 29, 1), dtype=np.float32).astype(dtype)
    )
    original = image.copy()
    result = A.Compose([A.FancyPCA(alpha=0.1, p=1.0)], seed=137)(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_fancy_pca_compose_supports_all_targets_and_dtypes(target, container, channels, dtype):
    shapes = {
        "image": (31, 29, channels),
        "images": (2, 31, 29, channels),
        "volume": (3, 31, 29, channels),
        "volumes": (2, 3, 31, 29, channels),
    }
    rng = np.random.default_rng(137)
    if dtype is np.uint8:
        image = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    else:
        half_image = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
        image = half_image if dtype is np.float16 else half_image.astype(np.float32)
    original = image.copy()

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose([A.FancyPCA(alpha=0.1, p=1.0)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source

    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.isfinite(actual).all()
    if dtype is np.uint8:
        assert np.all((actual >= 0) & (actual <= 255))
    else:
        assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_array_equal(preserved, original)

    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_error_diffusion_dither_compose_supports_all_targets_and_dtypes(target, container, channels, dtype):
    shapes = {
        "image": (19, 23, channels),
        "images": (2, 19, 23, channels),
        "volume": (3, 19, 23, channels),
        "volumes": (2, 3, 19, 23, channels),
    }
    rng = np.random.default_rng(137)
    if dtype is np.uint8:
        image = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    else:
        half_image = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
        image = half_image if dtype is np.float16 else half_image.astype(np.float32)
    original = image.copy()

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose(
            [
                A.Dithering(
                    method="error_diffusion",
                    n_colors=4,
                    color_mode="per_channel",
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source

    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.isfinite(actual).all()
    if dtype is np.uint8:
        assert np.all((actual >= 0) & (actual <= 255))
    else:
        assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_array_equal(preserved, original)

    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_hue_saturation_value_compose_preserves_grayscale_channel(dtype):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (31, 29, 1), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((31, 29, 1), dtype=np.float32).astype(dtype)
    )
    original = image.copy()
    transform = A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_range=(0, 0),
                sat_shift_range=(0, 0),
                val_shift_range=(12, 12),
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]
    expected = fpixel.shift_hsv(image, hue_shift=0, sat_shift=0, val_shift=12)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("val_shift", [0, 12])
def test_hue_saturation_value_float16_matches_float32_reference(target, container, channels, val_shift):
    shapes = {
        "image": (31, 29, channels),
        "images": (2, 31, 29, channels),
        "volume": (3, 31, 29, channels),
        "volumes": (2, 3, 31, 29, channels),
    }
    half_image = np.random.default_rng(137).random(shapes[target], dtype=np.float32).astype(np.float16)
    original = half_image.copy()
    params = {
        "hue_shift_range": (0, 0),
        "sat_shift_range": (0, 0),
        "val_shift_range": (val_shift, val_shift),
        "p": 1.0,
    }

    def run(image):
        source = image.copy()
        if container == "tensor":
            source = numpy_to_tensor_spatial(source, target)
        else:
            source.flags.writeable = False
        pipeline = A.Compose([A.HueSaturationValue(**params)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    half_result, half_source = run(half_image)
    float_result, _ = run(half_image.astype(np.float32))
    if container == "tensor":
        half_result = tensor_to_numpy_spatial(half_result, target)
        float_result = tensor_to_numpy_spatial(float_result, target)
        half_source = tensor_to_numpy_spatial(half_source, target)

    assert half_result.shape == half_image.shape
    assert half_result.dtype == np.float16
    np.testing.assert_array_equal(half_result, float_result.astype(np.float16))
    np.testing.assert_array_equal(half_source, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_to_gray_from_lab_float16_compose_matches_float32(target, container):
    shapes = {
        "image": (31, 29, 3),
        "images": (2, 31, 29, 3),
        "volume": (3, 31, 29, 3),
        "volumes": (2, 3, 31, 29, 3),
    }
    half_image = np.random.default_rng(137).random(shapes[target], dtype=np.float32).astype(np.float16)
    original = half_image.copy()

    def make_source(image):
        source = image.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(image):
        source = make_source(image)
        pipeline = A.Compose([A.ToGray(method="from_lab", num_output_channels=1, p=1.0)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    half_result, half_source = run(half_image)
    float32_result, _ = run(half_image.astype(np.float32))
    if container == "tensor":
        half_result = tensor_to_numpy_spatial(half_result, target)
        float32_result = tensor_to_numpy_spatial(float32_result, target)
        half_source = tensor_to_numpy_spatial(half_source, target)

    assert half_result.shape == (*half_image.shape[:-1], 1)
    assert half_result.dtype == np.float16
    np.testing.assert_array_equal(half_result, float32_result.astype(np.float16))
    np.testing.assert_array_equal(half_source, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("operation", ["equalize_cv", "equalize_pil", "clahe"])
def test_equalize_and_clahe_compose_preserve_singleton_channel(dtype, operation):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (31, 29, 1), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((31, 29, 1), dtype=np.float32).astype(dtype)
    )
    original = image.copy()
    if operation == "clahe":
        transform = A.CLAHE(clip_range=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)
        expected = fpixel.clahe(image, 2.0, (8, 8))
    else:
        mode = operation.removeprefix("equalize_")
        transform = A.Equalize(mode=mode, p=1.0)
        expected = fpixel.equalize(image, mode=mode)

    result = A.Compose([transform], seed=137)(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("operation", ["equalize_cv", "equalize_pil", "clahe"])
def test_equalize_and_clahe_float16_match_float32_reference(operation):
    half_image = np.random.default_rng(137).random((31, 29, 1), dtype=np.float32).astype(np.float16)
    float_image = half_image.astype(np.float32)
    if operation == "clahe":
        half_transform = A.CLAHE(clip_range=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)
        float_transform = A.CLAHE(clip_range=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)
    else:
        mode = operation.removeprefix("equalize_")
        half_transform = A.Equalize(mode=mode, p=1.0)
        float_transform = A.Equalize(mode=mode, p=1.0)

    half_result = A.Compose([half_transform], seed=137)(image=half_image)["image"]
    float_result = A.Compose([float_transform], seed=137)(image=float_image)["image"]

    assert half_result.shape == half_image.shape
    assert half_result.dtype == np.float16
    np.testing.assert_array_equal(half_result, float_result.astype(np.float16))


@pytest.mark.parametrize(
    ("dtype", "drop_value", "source_value"),
    [
        (np.uint8, 137, 201),
        (np.float16, 0.125, 0.75),
        (np.float32, 0.125, 0.75),
    ],
)
def test_pixel_dropout_compose_preserves_singleton_channel(dtype, drop_value, source_value):
    image = np.full((3, 4, 1), source_value, dtype=dtype)
    original = image.copy()
    transform = A.Compose([A.PixelDropout(dropout_prob=1.0, drop_value=drop_value, p=1.0)])

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, np.full_like(image, drop_value))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    ["drop_value", "expected_values"],
    [
        (None, None),  # Random values will be generated
        (0, 0),  # Single value
        ((1, 2, 3), np.array([1, 2, 3])),  # Sequence of values
    ],
)
def test_pixel_dropout_drop_values(drop_value, expected_values):
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    transform = A.PixelDropout(dropout_prob=1.0, drop_value=drop_value, p=1.0)

    result = transform(image=image)["image"]

    if drop_value is None:
        # For None, we just verify values are within valid range
        assert result.dtype == np.uint8
        assert np.all((result >= 0) & (result <= 255))
    elif isinstance(drop_value, (int, float)):
        # For single value, all channels should have same value
        np.testing.assert_array_equal(result, expected_values)
    else:
        # For sequence, each channel should have corresponding value
        for channel_idx, expected_value in enumerate(expected_values):
            np.testing.assert_array_equal(result[:, :, channel_idx], expected_value)


def test_pixel_dropout_per_channel():
    """Test that per_channel=True works correctly with different drop_values"""
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    # Test with single value
    transform = A.PixelDropout(
        dropout_prob=0.5,
        drop_value=0,
        per_channel=True,
        p=1.0,
    )
    result = transform(image=image)["image"]
    assert np.any(result == 0)  # Should have some dropped pixels

    # Test with sequence
    transform = A.PixelDropout(
        dropout_prob=0.5,
        drop_value=(1, 2, 3),
        per_channel=True,
        p=1.0,
    )
    result = transform(image=image)["image"]
    # Each channel should only contain original values or its drop value
    for channel_idx, drop_val in enumerate((1, 2, 3)):
        unique_values = np.unique(result[:, :, channel_idx])
        assert len(unique_values) == 2  # Should only have original value and drop value
        assert drop_val in unique_values
        assert 255 in unique_values


def test_pixel_dropout_multiple_images():
    # I actually don't care about these params, I just need a transform instance.
    transform = A.PixelDropout(
        dropout_prob=0.5,
        drop_value=0,
        per_channel=True,
        p=1.0,
    )

    # Prepare inputs:
    images = np.ones((2, 10, 10, 3), dtype=np.uint8) * 255
    rng = np.random.default_rng(42)
    drop_mask = fpixel.get_drop_mask(images.shape, True, 0.5, rng)
    drop_values = fpixel.prepare_drop_values(images, 0, rng)

    result = transform.apply_to_images(images, drop_mask, drop_values)
    assert result.shape == images.shape  # Check that original shape is preserved
    assert np.all([np.any(image == 0) for image in result])  # Each image should have some dropped pixels


@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("images", (0, 8, 12, 3)),
        ("masks", (0, 8, 12)),
        ("volume", (0, 8, 12, 1)),
        ("mask3d", (0, 8, 12)),
    ],
)
def test_pixel_dropout_preserves_empty_targets(target: str, shape: tuple[int, ...]) -> None:
    value = np.empty(shape, dtype=np.uint8)
    pipeline = A.Compose(
        [A.PixelDropout(dropout_prob=1.0, drop_value=0, mask_drop_value=1, p=1.0)],
    )

    result = pipeline(**{target: value})

    assert result[target].shape == shape
    assert result[target].dtype == value.dtype


@pytest.mark.parametrize(
    ["drop_value", "channels", "expected_values"],
    [
        # Matching dimensions
        ((255, 0, 127), 3, [255, 0, 127]),
        # Fewer values than channels - should cycle
        ((255, 0), 3, [255, 0, 255]),
        ((255,), 3, [255, 255, 255]),
        # More values than channels - should use first N
        ((255, 0, 127, 64), 3, [255, 0, 127]),
        ((255, 0, 127, 64, 32), 3, [255, 0, 127]),
        # 4-channel RGBA
        ((255, 0), 4, [255, 0, 255, 0]),
        ((255, 0, 127, 64), 4, [255, 0, 127, 64]),
    ],
)
def test_pixel_dropout_mismatched_tuple_dimensions(drop_value, channels, expected_values):
    """Test that PixelDropout handles mismatched tuple dimensions correctly."""
    shape = (10, 10, channels) if channels > 1 else (10, 10)
    image = np.ones(shape, dtype=np.uint8) * 128

    # Apply transform with 100% dropout to test all pixels
    transform = A.PixelDropout(dropout_prob=1.0, drop_value=drop_value, p=1.0)
    result = transform(image=image)["image"]

    assert result.shape == image.shape

    # Check values are as expected
    if channels == 1:
        # Grayscale - should use first value
        np.testing.assert_array_equal(result, expected_values[0])
    else:
        # Multi-channel - check each channel
        for channel_idx in range(channels):
            np.testing.assert_array_equal(result[:, :, channel_idx], expected_values[channel_idx])


BASE_DROPOUT_GRAYSCALE_CASES = [
    (
        A.CoarseDropout,
        {
            "num_holes_range": (1, 1),
            "hole_height_range": (8, 8),
            "hole_width_range": (8, 8),
        },
        False,
    ),
    (
        A.Erasing,
        {
            "scale": (0.2, 0.2),
            "ratio": (1.0, 1.0),
        },
        False,
    ),
    (
        A.GridDropout,
        {
            "ratio": 0.5,
            "unit_size_range": (8, 8),
            "random_offset": False,
        },
        False,
    ),
    (
        A.GridMask,
        {
            "num_grid_range": (4, 4),
            "line_width_range": (0.5, 0.5),
            "rotation_range": (0, 0),
        },
        False,
    ),
    (
        A.XYMasking,
        {
            "num_masks_x_range": (1, 1),
            "mask_x_length_range": (8, 8),
        },
        False,
    ),
    (
        A.ConstrainedCoarseDropout,
        {
            "num_holes_range": (1, 1),
            "hole_height_range": (0.5, 0.5),
            "hole_width_range": (0.5, 0.5),
            "mask_indices": [1],
        },
        True,
    ),
]


def _assert_grayscale_fill_behavior(image: np.ndarray, result: np.ndarray) -> np.ndarray:
    assert result.shape == image.shape
    assert result.dtype == image.dtype

    if image.shape[-1] == 1:
        np.testing.assert_array_equal(result, image)
        return np.zeros(image.shape[:-1], dtype=bool)

    changed_mask = np.any(result != image, axis=-1)
    assert changed_mask.any()
    assert (~changed_mask).any()

    changed_pixels = result[changed_mask]
    np.testing.assert_array_equal(changed_pixels, np.repeat(changed_pixels[:, :1], image.shape[-1], axis=-1))

    original_changed_pixels = image[changed_mask]
    assert np.any(original_changed_pixels != np.repeat(original_changed_pixels[:, :1], image.shape[-1], axis=-1))

    return changed_mask


def _create_grayscale_fill_test_array(shape: tuple[int, ...], dtype: type) -> np.ndarray:
    channels = shape[-1]
    base = np.arange(np.prod(shape[:-1]), dtype=np.float32).reshape(shape[:-1])

    if dtype == np.uint8:
        return np.stack([((base + 29 * channel) % 251) for channel in range(channels)], axis=-1).astype(np.uint8)

    normalized = base / max(1, base.size - 1)
    return np.stack(
        [np.mod(normalized + channel / (channels + 1), 1.0) for channel in range(channels)],
        axis=-1,
    ).astype(np.float32)


def _create_grayscale_fill_test_image(channels: int, dtype: type) -> np.ndarray:
    return _create_grayscale_fill_test_array((64, 64, channels), dtype)


@pytest.mark.parametrize(("augmentation_cls", "params", "needs_mask"), BASE_DROPOUT_GRAYSCALE_CASES)
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_base_dropout_grayscale_fill_supports_any_channel_count(augmentation_cls, params, needs_mask, channels, dtype):
    image = _create_grayscale_fill_test_image(channels, dtype)
    data = {"image": image}
    if needs_mask:
        mask = np.zeros((64, 64, 1), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        data["mask"] = mask

    transform = augmentation_cls(**params, fill="grayscale", p=1.0)
    pipeline = A.Compose([transform], seed=137, strict=True)

    result = pipeline(**data)["image"]

    _assert_grayscale_fill_behavior(image, result)


@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_mask_dropout_grayscale_fill_supports_any_channel_count(channels, dtype):
    image = _create_grayscale_fill_test_image(channels, dtype)
    mask = np.zeros((64, 64, 1), dtype=np.uint8)
    mask[16:48, 16:48] = 1

    transform = A.MaskDropout(max_objects_range=(1, 1), fill="grayscale", fill_mask=0, p=1.0)
    pipeline = A.Compose([transform], seed=137, strict=True)

    result = pipeline(image=image, mask=mask)["image"]

    changed_mask = _assert_grayscale_fill_behavior(image, result)
    if channels > 1:
        np.testing.assert_array_equal(changed_mask, mask[:, :, 0].astype(bool))


def test_erasing_grayscale_fill_converts_only_sampled_patch():
    image = _create_grayscale_fill_test_image(3, np.uint8)

    transform = A.Erasing(
        scale=(0.2, 0.2),
        ratio=(1.0, 1.0),
        fill="grayscale",
        p=1.0,
    )
    transform.set_random_seed(137)

    result = transform(image=image)["image"]
    _assert_grayscale_fill_behavior(image, result)


def test_erasing_grayscale_fill_requires_fill_mask_none():
    with pytest.raises(ValueError, match="fill_mask must be None"):
        A.Erasing(
            scale=(0.2, 0.2),
            ratio=(1.0, 1.0),
            fill="grayscale",
            fill_mask=0,
            p=1.0,
        )


@pytest.mark.parametrize("channels", [1, 3, 5])
def test_erasing_grayscale_fill_on_volume(channels):
    volume = _create_grayscale_fill_test_array((4, 64, 64, channels), np.uint8)

    transform = A.Erasing(
        scale=(0.2, 0.2),
        ratio=(1.0, 1.0),
        fill="grayscale",
        p=1.0,
    )
    transform.set_random_seed(137)

    result = transform(volume=volume)["volume"]
    _assert_grayscale_fill_behavior(volume, result)


def test_salt_and_pepper_noise():
    # Test image setup - create all gray image instead of black with gray square
    image = np.full((100, 100, 3), 128, dtype=np.uint8)  # All gray image

    # Fixed parameters for deterministic testing
    amount = (0.05, 0.05)  # Exactly 5% of pixels
    salt_vs_pepper = (0.6, 0.6)  # Exactly 60% salt, 40% pepper

    transform = A.SaltAndPepper(
        amount_range=amount,
        salt_vs_pepper_range=salt_vs_pepper,
        p=1.0,
    )
    transform.set_random_seed(137)

    transformed = transform(image=image)["image"]

    # Count all changes
    salt_pixels = (transformed == 255).all(axis=2)
    pepper_pixels = (transformed == 0).all(axis=2)

    total_changes = salt_pixels.sum() + pepper_pixels.sum()

    expected_pixels = int(image.shape[0] * image.shape[1] * amount[0])
    assert total_changes == expected_pixels, f"Expected {expected_pixels} noisy pixels, got {total_changes}"

    # Verify salt vs pepper ratio
    expected_salt = int(expected_pixels * salt_vs_pepper[0])
    assert salt_pixels.sum() == expected_salt, f"Expected {expected_salt} salt pixels, got {salt_pixels.sum()}"


def test_salt_and_pepper_float_image():
    """Test salt and pepper noise on float32 images"""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    image[25:75, 25:75] = 0.5  # Gray square

    transform = A.SaltAndPepper(
        amount_range=(0.05, 0.05),
        salt_vs_pepper_range=(0.6, 0.6),
        p=1.0,
    )
    transform.set_random_seed(137)

    transformed = transform(image=image)["image"]

    # Check that salt pixels are 1.0 and pepper pixels are 0.0
    (transformed != image).any(axis=2)
    np.testing.assert_allclose(
        transformed[transformed > 0.9],
        1.0,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
        err_msg="Salt pixels should be exactly 1.0 for float images",
    )
    np.testing.assert_allclose(
        transformed[transformed < 0.1],
        0.0,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
        err_msg="Pepper pixels should be exactly 0.0 for float images",
    )


def test_salt_and_pepper_grayscale():
    """Test salt and pepper noise on single-channel images"""
    image = np.zeros((100, 100, 1), dtype=np.uint8)
    image[25:75, 25:75] = 128

    transform = A.SaltAndPepper(
        amount_range=(0.05, 0.05),
        salt_vs_pepper_range=(0.6, 0.6),
        p=1.0,
    )
    transform.set_random_seed(137)

    transformed = transform(image=image)["image"]

    assert transformed.shape == image.shape, "Transform should preserve single-channel image shape"

    # Check noise values
    changed_mask = transformed != image
    salt_pixels = (transformed == 255) & changed_mask
    pepper_pixels = (transformed == 0) & changed_mask

    assert (salt_pixels | pepper_pixels | ~changed_mask).all(), "Changed pixels should only be salt (255) or pepper (0)"


@pytest.mark.parametrize(
    ["slant_range", "expected_slant_range"],
    [
        ((-10, -5), (-10, -5)),  # negative slant range
        ((5, 10), (5, 10)),  # positive slant range
        ((-5, 5), (-5, 5)),  # range crossing zero
    ],
)
def test_random_rain_slant(slant_range, expected_slant_range):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create transform with 100% probability and specific slant range
    transform = A.RandomRain(
        slant_range=slant_range,
        p=1.0,
        rain_type="heavy",  # Use heavy to ensure enough rain drops
    )

    # Run multiple iterations with different seeds to ensure slant stays within range
    slants = []
    for iteration in range(50):  # Run 50 times to get a good sample
        # Use different seed for each iteration
        transform.set_random_seed(137 + iteration)
        # Get params without actually applying transform
        params = transform.sample_parameters(
            *make_sampling_args(transform, {"image": image}),
            SamplingContext.from_owner(transform, {}),
        ).params
        slants.append(params["slant"])

    # Assert all slants are within the expected range
    assert all(expected_slant_range[0] <= s <= expected_slant_range[1] for s in slants), (
        f"Slants {slants} not within range {expected_slant_range}"
    )

    # Assert we get at least some variation in slants
    assert len(set(slants)) > 1, "Slant values show no variation"

    # Assert we get values from both halves of the range
    slant_mid = (expected_slant_range[0] + expected_slant_range[1]) / 2
    has_lower = any(s < slant_mid for s in slants)
    has_upper = any(s > slant_mid for s in slants)
    if expected_slant_range[0] != expected_slant_range[1]:  # Skip if range is single value
        assert has_lower and has_upper, f"Slant values {slants} don't cover both halves of range {expected_slant_range}"


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_random_rain_compose_preserves_rgb_shape_dtype_and_input(dtype):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((64, 64, 3), dtype=np.float32).astype(dtype)
    )
    original = image.copy()
    transform = A.Compose(
        [
            A.RandomRain(
                slant_range=(0, 0),
                drop_length=12,
                drop_width=1,
                blur_value=1,
                brightness_coefficient=1.0,
                rain_type="heavy",
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(image, original)


def test_random_rain_float16_matches_float32_reference() -> None:
    half_image = np.random.default_rng(137).random((64, 64, 3), dtype=np.float32).astype(np.float16)
    float_image = half_image.astype(np.float32)
    params = {
        "slant_range": (0, 0),
        "drop_length": 12,
        "drop_width": 1,
        "blur_value": 1,
        "brightness_coefficient": 1.0,
        "rain_type": "heavy",
        "p": 1.0,
    }

    half_result = A.Compose([A.RandomRain(**params)], seed=137)(image=half_image)["image"]
    float_result = A.Compose([A.RandomRain(**params)], seed=137)(image=float_image)["image"]

    np.testing.assert_array_equal(half_result, float_result.astype(np.float16))


@pytest.mark.parametrize("slant", [-10, 0, 10])
def test_random_rain_visual_effect(slant):
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    # Create transform with fixed slant for visual verification
    transform = A.RandomRain(
        slant_range=(slant, slant),  # Force specific slant
        drop_length=20,
        drop_width=2,
        drop_color=(0, 0, 0),  # Black rain drops for contrast
        blur_value=1,  # Minimal blur for clearer lines
        brightness_coefficient=1.0,  # No brightness change
        p=1.0,
        rain_type="heavy",
    )

    transform.set_random_seed(137)

    result = transform(image=image)["image"]

    # Find non-white pixels (rain drops)
    rain_pixels = np.where(result != 255)

    if len(rain_pixels[0]) > 0:  # If we found rain drops
        # Calculate the average slope of rain drops
        # We can do this by looking at the leftmost and rightmost points
        left_x = min(rain_pixels[1])
        right_x = max(rain_pixels[1])
        if right_x > left_x:  # Ensure we have horizontal spread
            left_y = rain_pixels[0][rain_pixels[1] == left_x].mean()
            right_y = rain_pixels[0][rain_pixels[1] == right_x].mean()

            # Calculate observed slant direction
            observed_slant = 1 if right_y > left_y else -1 if right_y < left_y else 0
            expected_slant = 1 if slant > 0 else -1 if slant < 0 else 0

            # The slant direction should match the expected direction
            assert observed_slant == expected_slant, (
                f"Rain drops slant direction {observed_slant} doesn't match expected {expected_slant}"
            )


def test_to_sepia_rgb():
    transform = A.ToSepia(p=1.0)
    # White image to see some effect:
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    transformed = transform(image=image)["image"]

    assert image.shape == transformed.shape
    assert not np.array_equal(image, transformed)


@pytest.mark.parametrize(
    "image",
    [
        np.random.randint(low=0, high=255, size=(10, 10, 1), dtype=np.uint8),
        np.random.randint(low=0, high=255, size=(2, 10, 10, 1), dtype=np.uint8),
    ],
)
def test_to_sepia_gray(image: np.ndarray):
    transform = A.ToSepia(p=1.0)

    transformed = transform(image=image)["image"]

    np.testing.assert_array_equal(image, transformed)


def test_to_sepia_rgb_multiple_images():
    transform = A.ToSepia(p=1.0)
    images = np.ones((2, 10, 10, 3), dtype=np.uint8) * 255

    transformed = transform.apply_to_images(images)

    assert images.shape == transformed.shape
    assert np.all([not np.array_equal(im, tr) for im, tr in zip(images, transformed, strict=False)])


def _restore_median_blur_channel_axis(image: np.ndarray, num_channels: int) -> np.ndarray:
    return image[..., np.newaxis] if num_channels == 1 and image.ndim == 2 else image


def _median_blur_uint8_reference(image: np.ndarray, kernel: int) -> np.ndarray:
    num_channels = image.shape[-1]
    if kernel in (3, 5) or num_channels <= 4:
        return _restore_median_blur_channel_axis(cv2.medianBlur(image, kernel), num_channels)
    return np.concatenate(
        [
            _restore_median_blur_channel_axis(cv2.medianBlur(image[..., index : index + 1], kernel), 1)
            for index in range(num_channels)
        ],
        axis=-1,
    )


def _median_blur_reference(image: np.ndarray, kernel: int) -> np.ndarray:
    if image.dtype == np.uint8:
        return _median_blur_uint8_reference(image, kernel)
    if kernel in (3, 5):
        return _restore_median_blur_channel_axis(cv2.medianBlur(image, kernel), image.shape[-1])
    quantized = convert_dtype(image, np.dtype(np.uint8))
    return convert_dtype(_median_blur_uint8_reference(quantized, kernel), np.float32)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.float32],
)
@pytest.mark.parametrize(
    "num_channels",
    [1, 3, 5],
)
@pytest.mark.parametrize(
    "kernel",
    [3, 5, 7],
)
def test_median_blur_compose_target_routes(
    dtype: type[np.uint8 | np.float32],
    num_channels: int,
    kernel: int,
) -> None:
    """MedianBlur preserves its dtype route across image, batch, and volume targets."""
    rng = np.random.default_rng(137)

    if dtype == np.uint8:
        image = rng.integers(0, 256, size=(31, 37, num_channels), dtype=np.uint8)
    else:
        image = rng.random((31, 37, num_channels), dtype=np.float32)

    images = np.stack([image, np.flip(image, axis=0)], axis=0)
    volume = np.stack([image, np.flip(image, axis=1)], axis=0)
    transform = A.Compose(
        [A.MedianBlur(blur_range=(kernel, kernel), p=1.0)],
        save_applied_params=True,
        seed=137,
    )

    image_result = transform(image=image)
    images_result = transform(images=images)
    volume_result = transform(volume=volume)

    expected_image = _median_blur_reference(image, kernel)
    expected_images = np.stack([_median_blur_reference(item, kernel) for item in images])
    expected_volume = np.stack([_median_blur_reference(item, kernel) for item in volume])

    for result, target_name, source, expected in (
        (image_result, "image", image, expected_image),
        (images_result, "images", images, expected_images),
        (volume_result, "volume", volume, expected_volume),
    ):
        transformed = result[target_name]
        assert transformed.shape == source.shape
        assert transformed.dtype == source.dtype
        assert transformed.flags.c_contiguous
        np.testing.assert_array_equal(transformed, expected)
        assert result["applied_transforms"] == [("MedianBlur", {"p": 1.0, "blur_range": kernel})]

    per_image_results = np.stack([transform(image=item)["image"] for item in images])
    per_volume_results = np.stack([transform(image=item)["image"] for item in volume])
    np.testing.assert_array_equal(images_result["images"], per_image_results)
    np.testing.assert_array_equal(volume_result["volume"], per_volume_results)


@pytest.mark.parametrize("kernel", [3, 5])
def test_median_blur_native_float32_preserves_sub_uint8_precision(kernel: int) -> None:
    image = np.linspace(0.1001, 0.1099, 31 * 37, dtype=np.float32).reshape(31, 37, 1)
    transform = A.Compose([A.MedianBlur(blur_range=(kernel, kernel), p=1.0)])

    result = transform(image=image)["image"]
    native = _restore_median_blur_channel_axis(cv2.medianBlur(image, kernel), 1)
    legacy = convert_dtype(_median_blur_uint8_reference(convert_dtype(image, np.dtype(np.uint8)), kernel), np.float32)

    np.testing.assert_array_equal(result, native)
    assert not np.array_equal(result, legacy)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.float32],
)
def test_unsharp_mask_apply_to_images(dtype):
    if dtype == np.uint8:
        images = np.random.RandomState(137).randint(0, 256, (2, 100, 100, 3), dtype=np.uint8)
    else:
        images = np.random.RandomState(137).random((2, 100, 100, 3)).astype(np.float32)

    transform = A.UnsharpMask(
        blur_range=(3, 3),
        sigma_range=(0.5, 0.5),
        alpha_range=(0.3, 0.3),
        threshold=10,
        p=1.0,
    )

    transformed = transform(images=images)["images"]

    assert transformed.shape == images.shape
    assert transformed.dtype == images.dtype

    expected = np.stack([transform(image=images[i])["image"] for i in range(images.shape[0])])
    np.testing.assert_array_equal(transformed, expected)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(np.uint8, 137), (np.float16, 0.375), (np.float32, 0.375)],
)
def test_sharpen_compose_preserves_singleton_channel(dtype, value):
    image = np.full((17, 13, 1), value, dtype=dtype)
    transform = A.Compose(
        [A.Sharpen(alpha_range=(0.3, 0.3), lightness_range=(0.7, 0.7), method="kernel", p=1.0)],
        seed=137,
    )

    result = transform(image=image)["image"]

    expected = image.astype(np.float32) * np.float32(0.91)
    if dtype is np.uint8:
        expected = np.rint(expected).astype(np.uint8)
    elif dtype is np.float16:
        expected = expected.astype(np.float16)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    if dtype is np.float32:
        # CV_32F convolution may differ from scalar multiplication by a few ULPs across CPUs.
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
    else:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_sharpen_gaussian_compose_preserves_singleton_channel(dtype):
    rng = np.random.default_rng(137)
    single = (
        rng.integers(0, 256, (17, 13, 1), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((17, 13, 1), dtype=np.float32).astype(dtype)
    )
    original = single.copy()
    rgb = np.repeat(single, 3, axis=-1)
    transform = A.Sharpen(
        alpha_range=(0.3, 0.3),
        method="gaussian",
        kernel_size=5,
        sigma=1.0,
        p=1.0,
    )

    single_result = A.Compose([transform], seed=137)(image=single)["image"]
    rgb_result = A.Compose(
        [A.Sharpen(alpha_range=(0.3, 0.3), method="gaussian", kernel_size=5, sigma=1.0, p=1.0)],
        seed=137,
    )(image=rgb)["image"]

    assert single_result.shape == single.shape
    assert single_result.dtype == single.dtype
    np.testing.assert_allclose(
        single_result[..., 0],
        rgb_result[..., 0],
        rtol=0,
        atol=1e-7 if dtype is np.float32 else 0,
    )
    np.testing.assert_array_equal(single, original)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_sharpen_gaussian_half_clipping_matches_float32(container):
    image = np.zeros((15, 15, 1), dtype=np.float16)
    image[7, 7, 0] = 1
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    def pipeline():
        return A.Compose(
            [A.Sharpen(alpha_range=(1.0, 1.0), method="gaussian", kernel_size=5, sigma=1.0, p=1.0)],
            seed=137,
            strict=True,
        )

    half_result = pipeline()(image=source)["image"]
    actual = tensor_to_numpy_spatial(half_result, "image") if container == "tensor" else half_result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    float32_source = (
        numpy_to_tensor_spatial(image.astype(np.float32), "image")
        if container == "tensor"
        else image.astype(np.float32)
    )
    float32_result = pipeline()(image=float32_source)["image"]
    if container == "tensor":
        float32_result = tensor_to_numpy_spatial(float32_result, "image")

    assert actual.dtype == np.float16
    assert actual.shape == image.shape
    assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_array_equal(actual, float32_result.astype(np.float16))
    np.testing.assert_array_equal(preserved, original)


@pytest.mark.parametrize(
    ["dtype", "method"],
    [
        (np.uint8, "kernel"),
        (np.float32, "kernel"),
        (np.uint8, "gaussian"),
        (np.float32, "gaussian"),
    ],
)
def test_sharpen_apply_to_images(dtype, method):
    if dtype == np.uint8:
        images = np.random.RandomState(137).randint(0, 256, (2, 100, 100, 3), dtype=np.uint8)
    else:
        images = np.random.RandomState(137).random((2, 100, 100, 3)).astype(np.float32)

    transform = A.Sharpen(
        alpha_range=(0.3, 0.3),
        lightness_range=(0.7, 0.7),
        method=method,
        kernel_size=5,
        sigma=1.0,
        p=1.0,
    )

    transformed = transform(images=images)["images"]

    assert transformed.shape == images.shape
    assert transformed.dtype == images.dtype

    expected = np.stack([transform(image=images[i])["image"] for i in range(images.shape[0])])
    np.testing.assert_array_equal(transformed, expected)


@pytest.mark.parametrize("mode", ["edge", "detail"])
def test_enhance_alpha_zero_is_identity(mode):
    image = np.random.RandomState(137).randint(0, 256, (64, 64, 3), dtype=np.uint8)
    transform = A.Compose([A.Enhance(mode=mode, alpha_range=(0.0, 0.0), p=1.0)])
    np.testing.assert_array_equal(transform(image=image)["image"], image)


@pytest.mark.parametrize("mode", ["edge", "detail"])
def test_enhance_kernel_matches_pillow_preset(mode):
    """alpha=1 must reproduce the Pillow preset kernel exactly (DC-preserving)."""
    expected = {
        "edge": np.array(
            [[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]],
            dtype=np.float32,
        ),
        "detail": np.array(
            [
                [0.0, -1.0 / 6.0, 0.0],
                [-1.0 / 6.0, 10.0 / 6.0, -1.0 / 6.0],
                [0.0, -1.0 / 6.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }[mode]
    np.testing.assert_allclose(
        fpixel.generate_enhance_matrix(mode, 1.0),
        expected,
        rtol=1e-6,
    )
    np.testing.assert_allclose(fpixel.generate_enhance_matrix(mode, 1.0).sum(), 1.0, rtol=1e-6)


def test_enhance_edge_alpha_two_matches_edge_enhance_more():
    """alpha=2 with mode=edge must reproduce Pillow's EDGE_ENHANCE_MORE kernel."""
    expected = np.array(
        [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(fpixel.generate_enhance_matrix("edge", 2.0), expected, rtol=1e-6)


@pytest.mark.parametrize("mode", ["edge", "detail"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
def test_enhance_preserves_shape_and_dtype(mode, dtype, num_channels):
    rng = np.random.RandomState(137)
    shape = (64, 64) if num_channels == 1 else (64, 64, num_channels)
    if dtype == np.uint8:
        image = rng.randint(0, 256, shape, dtype=np.uint8)
    else:
        image = rng.random(shape).astype(np.float32)

    transform = A.Compose([A.Enhance(mode=mode, alpha_range=(0.5, 1.5), p=1.0)])
    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    if dtype == np.float32:
        assert result.min() >= 0.0
        assert result.max() <= 1.0


@pytest.mark.parametrize("mode", ["edge", "detail"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_enhance_apply_to_images_matches_per_image(mode, dtype):
    rng = np.random.RandomState(137)
    shape = (3, 48, 48, 3)
    if dtype == np.uint8:
        images = rng.randint(0, 256, shape, dtype=np.uint8)
    else:
        images = rng.random(shape).astype(np.float32)

    transform = A.Compose([A.Enhance(mode=mode, alpha_range=(0.7, 0.7), p=1.0)])

    batched = transform(images=images)["images"]
    per_image = np.stack([transform(image=images[i])["image"] for i in range(images.shape[0])])

    assert batched.shape == images.shape
    assert batched.dtype == images.dtype
    np.testing.assert_array_equal(batched, per_image)


@pytest.mark.parametrize("target_name", ["images", "volume"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize(
    ("brightness_factor", "contrast_factor"),
    [(0.0, 0.0), (0.19, 0.0), (0.19, -0.23)],
)
def test_plasma_brightness_contrast_batch_routes_match_per_image(
    target_name,
    dtype,
    num_channels,
    batch_size,
    brightness_factor,
    contrast_factor,
):
    rng = np.random.default_rng(137 + num_channels + batch_size)
    source_shape = (batch_size, 11, 26, num_channels)
    if dtype == np.uint8:
        source = rng.integers(0, 256, source_shape, dtype=np.uint8)
    else:
        source = rng.random(source_shape, dtype=np.float32)
    data = source[:, :, ::2]
    data.setflags(write=False)
    data_before = data.copy()
    transform = A.PlasmaBrightnessContrast(
        brightness_range=(brightness_factor, brightness_factor),
        contrast_range=(contrast_factor, contrast_factor),
        plasma_size=8,
        roughness=2.0,
        p=1.0,
    )
    compose = A.Compose([transform], seed=137, strict=True, save_applied_params=True)

    actual = compose(**{target_name: data})[target_name]
    params = get_resolved_applied_params(transform, target_name)
    expected = np.stack([transform.apply(image, **params) for image in data])

    assert actual.shape == data.shape
    assert actual.dtype == data.dtype
    assert actual.flags.c_contiguous
    assert actual.flags.writeable
    assert not np.shares_memory(actual, data)
    np.testing.assert_array_equal(data, data_before)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target_name", ["images", "volume"])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        pytest.param(np.dtype(np.float16).newbyteorder("S"), id="swapped-float16"),
        pytest.param(np.dtype(np.float32).newbyteorder("S"), id="swapped-float32"),
    ],
)
def test_plasma_brightness_contrast_zero_effect_clips_floating_input(target_name, num_channels, dtype):
    source = np.linspace(-0.25, 1.25, num=4 * 9 * 22 * num_channels, dtype=dtype).reshape(4, 9, 22, num_channels)
    data = source[:, :, ::2]
    data.setflags(write=False)
    before = data.copy()
    compose = A.Compose(
        [A.PlasmaBrightnessContrast(brightness_range=(0, 0), contrast_range=(0, 0), p=1)],
        seed=137,
        strict=True,
    )

    result = compose(**{target_name: data})[target_name]

    assert result.dtype == data.dtype
    assert result.flags.c_contiguous
    assert result.flags.writeable
    assert not np.shares_memory(result, data)
    np.testing.assert_array_equal(result, np.clip(before, 0, 1))
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_random_gravel_compose_preserves_dtype_and_input(dtype, container):
    rng = np.random.default_rng(137)
    uint8_image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image = uint8_image if dtype is np.uint8 else (uint8_image.astype(np.float32) / 255).astype(dtype)
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    def make_pipeline():
        return A.Compose(
            [A.RandomGravel(gravel_roi=(0, 0, 1, 1), number_of_patches=2, p=1.0)],
            seed=137,
            strict=True,
        )

    result = make_pipeline()(image=source)["image"]

    assert type(result) is type(source)
    actual = tensor_to_numpy_spatial(result, "image") if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.any(actual != image)
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_source = image.astype(np.float32)
        if container == "tensor":
            float32_source = numpy_to_tensor_spatial(float32_source, "image")
        float32_result = make_pipeline()(image=float32_source)["image"]
        if container == "tensor":
            float32_result = tensor_to_numpy_spatial(float32_result, "image")
        np.testing.assert_array_equal(actual, float32_result.astype(np.float16))


def _overlapping_shadow_params() -> tuple[list[np.ndarray], np.ndarray]:
    vertices_list = [
        np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.int32),
        np.array([[3, 3], [7, 3], [7, 7], [3, 7]], dtype=np.int32),
    ]
    return vertices_list, np.array([0.5, 0.25], dtype=np.float64)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_random_shadow_grayscale_compose_preserves_channel(dtype, container):
    expected = np.full((17, 19, 1), 127 if dtype is np.uint8 else 0.5, dtype=dtype)
    source = numpy_to_tensor_spatial(expected.copy(), "image") if container == "tensor" else expected.copy()
    if container == "numpy":
        source.flags.writeable = False

    def make_pipeline():
        return A.Compose(
            [A.RandomShadow(num_shadows_range=(1, 1), shadow_intensity_range=(0.5, 0.5), p=1.0)],
            seed=137,
            strict=True,
        )

    result = make_pipeline()(image=source)["image"]

    assert type(result) is type(source)
    actual = tensor_to_numpy_spatial(result, "image") if container == "tensor" else result
    original = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.any(actual != expected)
    max_value = 255 if dtype is np.uint8 else 1
    assert np.all((actual >= 0) & (actual <= max_value))
    np.testing.assert_array_equal(original, expected)
    if dtype is np.float16:
        float32_source = np.full(expected.shape, 0.5, dtype=np.float32)
        if container == "tensor":
            float32_source = numpy_to_tensor_spatial(float32_source, "image")
        float32_result = make_pipeline()(image=float32_source)["image"]
        if container == "tensor":
            float32_result = tensor_to_numpy_spatial(float32_result, "image")
        np.testing.assert_array_equal(actual, float32_result.astype(np.float16))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
def test_random_shadow_apply_to_images_matches_overlapping_per_image_results(dtype, num_channels):
    values = np.array([137, 200], dtype=np.uint8)
    if dtype == np.float32:
        values = values.astype(np.float32) / np.float32(255)

    source = np.broadcast_to(values[:, np.newaxis, np.newaxis, np.newaxis], (2, 18, 9, num_channels)).copy()
    images = source[:, ::2]
    images.setflags(write=False)
    images_before = images.copy()
    vertices_list, intensities = _overlapping_shadow_params()
    transform = A.RandomShadow(p=1.0)

    actual = transform.apply_to_images(images, vertices_list=vertices_list, intensities=intensities)
    expected = np.stack(
        [transform.apply(image, vertices_list=vertices_list, intensities=intensities) for image in images],
    )

    assert actual.shape == images.shape
    assert actual.dtype == images.dtype
    assert actual.flags.c_contiguous
    assert actual.flags.writeable
    assert not np.shares_memory(actual, images)
    np.testing.assert_array_equal(images, images_before)
    np.testing.assert_array_equal(actual, expected)

    expected_pixels = np.array([[68, 51, 102], [100, 75, 150]], dtype=np.uint8)
    if dtype == np.float32:
        expected_pixels = expected_pixels.astype(np.float32) / np.float32(255)
    np.testing.assert_array_equal(actual[:, 1, 1, 0], expected_pixels[:, 0])
    np.testing.assert_array_equal(actual[:, 3, 3, 0], expected_pixels[:, 1])
    np.testing.assert_array_equal(actual[:, 7, 7, 0], expected_pixels[:, 2])


def test_random_shadow_apply_to_images_rasterizes_each_polygon_once(monkeypatch):
    vertices_list, intensities = _overlapping_shadow_params()
    images = np.full((4, 9, 9, 3), 137, dtype=np.uint8)
    transform = A.RandomShadow(p=1.0)
    expected = np.stack(
        [transform.apply(image, vertices_list=vertices_list, intensities=intensities) for image in images],
    )
    fill_poly_calls = 0
    original_fill_poly = cv2.fillPoly

    def counting_fill_poly(*args, **kwargs):
        nonlocal fill_poly_calls
        fill_poly_calls += 1
        return original_fill_poly(*args, **kwargs)

    monkeypatch.setattr(cv2, "fillPoly", counting_fill_poly)

    actual = transform.apply_to_images(images, vertices_list=vertices_list, intensities=intensities)

    assert fill_poly_calls == len(vertices_list)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_random_shadow_apply_to_images_matches_out_of_bounds_polygons(dtype):
    images = np.full((2, 9, 9, 3), 137, dtype=np.uint8)
    if dtype == np.float32:
        images = images.astype(np.float32) / np.float32(255)
    vertices_list = [
        np.array([[-4, -4], [4, -4], [4, 4], [-4, 4]], dtype=np.int32),
        np.array([[20, 20], [24, 20], [24, 24], [20, 24]], dtype=np.int32),
    ]
    intensities = np.array([0.5, 0.25], dtype=np.float64)
    transform = A.RandomShadow(p=1.0)

    actual = transform.apply_to_images(images, vertices_list=vertices_list, intensities=intensities)
    expected = np.stack(
        [transform.apply(image, vertices_list=vertices_list, intensities=intensities) for image in images],
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("writeable", [False, True])
def test_random_shadow_apply_to_images_large_float32_conversion_matches_per_image(writeable):
    images = np.random.default_rng(137).random((2, 257, 341, 3), dtype=np.float32)
    images.setflags(write=writeable)
    vertices_list, intensities = _overlapping_shadow_params()
    transform = A.RandomShadow(p=1.0)

    actual = transform.apply_to_images(images, vertices_list=vertices_list, intensities=intensities)
    expected = np.stack(
        [transform.apply(image, vertices_list=vertices_list, intensities=intensities) for image in images],
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(2, 256, 256, 5), (2, 512, 512, 3)])
def test_random_shadow_apply_to_images_preserves_per_image_float32_conversion_working_set(monkeypatch, shape):
    images = np.random.default_rng(137).random(shape, dtype=np.float32)
    vertices_list, intensities = _overlapping_shadow_params()
    transform = A.RandomShadow(p=1.0)
    expected = np.stack(
        [transform.apply(image, vertices_list=vertices_list, intensities=intensities) for image in images],
    )
    conversion_sizes: list[int] = []
    original_convert_dtype = fweather.convert_dtype

    def tracked_convert_dtype(
        image: np.ndarray,
        dtype: np.dtype[np.generic],
        *,
        max_value: float | None = None,
    ) -> np.ndarray:
        if np.dtype(dtype) == np.uint8:
            conversion_sizes.append(image.size)
        return original_convert_dtype(image, dtype=dtype, max_value=max_value)

    monkeypatch.setattr(fweather, "convert_dtype", tracked_convert_dtype)

    actual = transform.apply_to_images(images, vertices_list=vertices_list, intensities=intensities)

    assert conversion_sizes == [images[0].size] * len(images)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target_name", ["images", "volume"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("seed", [137, 138])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_random_shadow_batch_compose_routes_match_per_image_results(target_name, dtype, seed, batch_size):
    rng = np.random.default_rng(seed)
    shape = (batch_size, 17, 19, 5)
    if dtype == np.uint8:
        data = rng.integers(0, 256, shape, dtype=np.uint8)
    else:
        data = rng.random(shape, dtype=np.float32)
    data_before = data.copy()
    transform = A.RandomShadow(
        num_shadows_range=(2, 2),
        shadow_intensity_range=(0.2, 0.7),
        p=1.0,
    )
    compose = A.Compose([transform], seed=seed, strict=True, save_applied_params=True)

    actual = compose(**{target_name: data})[target_name]
    params = get_resolved_applied_params(transform, target_name)
    expected = np.stack([transform.apply(image, **params) for image in data])

    np.testing.assert_array_equal(data, data_before)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_random_shadow_apply_to_images_preserves_empty_batch_identity(dtype):
    images = np.empty((0, 17, 19, 3), dtype=dtype)
    transform = A.RandomShadow(p=1.0)
    vertices_list, intensities = _overlapping_shadow_params()

    assert transform.apply_to_images(images, vertices_list=vertices_list, intensities=intensities) is images
    assert A.Compose([transform], seed=137, strict=True)(images=images)["images"] is images


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("width", [1, 3])
def test_random_shadow_apply_to_images_preserves_raw_grayscale_batch(dtype, width):
    images = np.arange(3 * 9 * width, dtype=np.uint8).reshape(3, 9, width)
    if dtype == np.float32:
        images = images.astype(np.float32) / np.float32(255)
    vertices_list = [np.array([[0, 0], [width - 1, 0], [width - 1, 8], [0, 8]], dtype=np.int32)]
    intensities = np.array([0.5], dtype=np.float64)
    transform = A.RandomShadow(p=1.0)

    actual = transform.apply_to_images(images, vertices_list=vertices_list, intensities=intensities)
    expected = np.stack(
        [transform.apply(image, vertices_list=vertices_list, intensities=intensities) for image in images],
    )

    assert actual.shape == images.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["rain", "mud"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("seed", [137, 138, 139])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_spatter_apply_to_images_matches_inherited_fallback(mode, dtype, seed, batch_size):
    rng = np.random.default_rng(seed)
    source_shape = (batch_size, 37, 82, 3)
    if dtype == np.uint8:
        source = rng.integers(0, 256, source_shape, dtype=np.uint8)
    else:
        source = rng.random(source_shape, dtype=np.float32)

    images = source[:, :, ::2, :]
    images.setflags(write=False)
    images_before = images.copy()
    transform = A.Spatter(mode=mode, color=(137, 89, 211), p=1.0)
    compose_result = A.Compose([transform], save_applied_params=True, seed=seed, strict=True)(images=images)["images"]
    params = get_resolved_applied_params(transform, target="images")
    expected = ImageOnlyTransform.apply_to_images(transform, images, **params)
    direct_result = transform.apply_to_images(images, **params)

    assert direct_result.shape == images.shape
    assert direct_result.dtype == images.dtype
    assert direct_result.flags.c_contiguous
    assert direct_result.flags.writeable
    assert not np.shares_memory(direct_result, images)
    assert not np.shares_memory(compose_result, images)
    np.testing.assert_array_equal(images, images_before)
    if dtype == np.uint8:
        np.testing.assert_array_equal(direct_result, expected)
        np.testing.assert_array_equal(compose_result, expected)
    else:
        np.testing.assert_allclose(direct_result, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(compose_result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_random_sun_flare_physics_compose_preserves_dtype_and_input(dtype, container):
    rng = np.random.default_rng(137)
    uint8_image = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    image = uint8_image if dtype is np.uint8 else (uint8_image.astype(np.float32) / 255).astype(dtype)
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    transform = A.RandomSunFlare(
        flare_roi=(0, 0, 1, 1),
        angle_range=(0, 0),
        num_flare_circles_range=(2, 2),
        src_radius=15,
        src_color=(255, 255, 210),
        method="physics_based",
        p=1.0,
    )
    pipeline = A.Compose([transform], seed=137, strict=True, save_applied_params=True)
    result = pipeline(image=source)["image"]
    params = get_resolved_applied_params(transform, target="image")
    expected = transform.apply(image, **params)

    assert type(result) is type(source)
    actual = tensor_to_numpy_spatial(result, "image") if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_image = image.astype(np.float32)
        if container == "tensor":
            float32_image = numpy_to_tensor_spatial(float32_image, "image")
        float32_result = A.Compose(
            [
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 1),
                    angle_range=(0, 0),
                    num_flare_circles_range=(2, 2),
                    src_radius=15,
                    src_color=(255, 255, 210),
                    method="physics_based",
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )(image=float32_image)["image"]
        if container == "tensor":
            float32_result = tensor_to_numpy_spatial(float32_result, "image")
        np.testing.assert_array_equal(actual, float32_result.astype(np.float16))


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_iso_noise_float16_compose_matches_float32_reference(target, container):
    rng = np.random.default_rng(137)
    shapes = {
        "image": (19, 23, 3),
        "images": (2, 19, 23, 3),
        "volume": (3, 19, 23, 3),
        "volumes": (2, 3, 19, 23, 3),
    }
    half_image = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
    original = half_image.copy()

    def make_source(image):
        source = image.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(image):
        source = make_source(image)
        pipeline = A.Compose(
            [A.ISONoise(color_shift_range=(0.03, 0.03), intensity_range=(0.7, 0.7), p=1.0)],
            seed=137,
            strict=True,
        )
        return pipeline(**{target: source})[target], source

    half_result, half_source = run(half_image)
    float32_result, _ = run(half_image.astype(np.float32))
    actual = tensor_to_numpy_spatial(half_result, target) if container == "tensor" else half_result
    expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
    preserved = tensor_to_numpy_spatial(half_source, target) if container == "tensor" else half_source

    assert actual.shape == original.shape
    assert actual.dtype == np.float16
    assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_array_equal(actual, expected.astype(np.float16))
    np.testing.assert_array_equal(preserved, original)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_color_jitter_saturation_float16_compose_clips_and_matches_float32(container):
    half_image = np.broadcast_to(np.array([1.0, 0.05, 0.2], dtype=np.float16), (19, 23, 3)).copy()
    original = half_image.copy()

    def make_source(image):
        source = image.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, "image")
        source.flags.writeable = False
        return source

    def run(image):
        source = make_source(image)
        pipeline = A.Compose(
            [
                A.ColorJitter(
                    brightness_range=(1.0, 1.0),
                    contrast_range=(1.0, 1.0),
                    saturation_range=(1.7, 1.7),
                    hue_range=(0.0, 0.0),
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )
        return pipeline(image=source)["image"], source

    half_result, half_source = run(half_image)
    float32_result, _ = run(half_image.astype(np.float32))
    actual = tensor_to_numpy_spatial(half_result, "image") if container == "tensor" else half_result
    expected = tensor_to_numpy_spatial(float32_result, "image") if container == "tensor" else float32_result
    preserved = tensor_to_numpy_spatial(half_source, "image") if container == "tensor" else half_source

    assert actual.shape == original.shape
    assert actual.dtype == np.float16
    assert np.all((actual >= 0) & (actual <= 1))
    assert np.any(actual == 1)
    np.testing.assert_allclose(actual, expected.astype(np.float16), rtol=0, atol=1 / 2048)
    np.testing.assert_array_equal(preserved, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_linear_illumination_compose_preserves_dtype_for_all_targets(target, container, dtype):
    shapes = {
        "image": (19, 23, 3),
        "images": (2, 19, 23, 3),
        "volume": (3, 19, 23, 3),
        "volumes": (2, 3, 19, 23, 3),
    }
    rng = np.random.default_rng(137)
    if dtype is np.uint8:
        image = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    else:
        half_image = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
        image = half_image if dtype is np.float16 else half_image.astype(np.float32)
    original = image.copy()

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose(
            [
                A.Illumination(
                    mode="linear",
                    intensity_range=(0.1, 0.1),
                    effect_type="brighten",
                    angle_range=(45, 45),
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source

    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.isfinite(actual).all()
    if dtype is np.uint8:
        assert np.all((actual >= 0) & (actual <= 255))
    else:
        assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_array_equal(preserved, original)

    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_plasma_shadow_compose_preserves_dtype_and_input(channels, dtype, container):
    rng = np.random.default_rng(137)
    uint8_image = rng.integers(0, 256, (64, 80, channels), dtype=np.uint8)
    if dtype is np.uint8:
        image = uint8_image
    else:
        half_image = (uint8_image.astype(np.float32) / 255).astype(np.float16)
        image = half_image if dtype is np.float16 else half_image.astype(np.float32)
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    transform = A.PlasmaShadow(shadow_intensity_range=(0.4, 0.4), roughness=1.0, p=1.0)
    pipeline = A.Compose([transform], seed=137, strict=True, save_applied_params=True)
    result = pipeline(image=source)["image"]
    params = get_resolved_applied_params(transform, target="image")
    expected = transform.apply(image, **params)

    assert type(result) is type(source)
    actual = tensor_to_numpy_spatial(result, "image") if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.all(np.isfinite(actual))
    assert np.all(actual <= image)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_image = image.astype(np.float32)
        float32_source = numpy_to_tensor_spatial(float32_image, "image") if container == "tensor" else float32_image
        float32_result = A.Compose(
            [A.PlasmaShadow(shadow_intensity_range=(0.4, 0.4), roughness=1.0, p=1.0)],
            seed=137,
            strict=True,
        )(image=float32_source)["image"]
        if container == "tensor":
            float32_result = tensor_to_numpy_spatial(float32_result, "image")
        np.testing.assert_array_equal(actual, float32_result.astype(np.float16))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_add_fog_direct_grayscale_keeps_singleton_channel(dtype):
    rng = np.random.default_rng(137)
    uint8_image = rng.integers(0, 256, (64, 80, 1), dtype=np.uint8)
    image = uint8_image if dtype is np.uint8 else (uint8_image.astype(np.float32) / 255).astype(dtype)
    original = image.copy()

    result = fpixel.add_fog(
        image,
        fog_intensity=0.7,
        alpha_coef=0.4,
        fog_particle_positions=[(18, 24), (48, 56)],
        fog_particle_radiuses=[12, 9],
    )

    assert result.shape == image.shape
    assert result.dtype == dtype
    assert np.any(result != image)
    assert np.all(np.isfinite(result))
    if dtype is np.uint8:
        assert np.all((result >= 0) & (result <= 255))
    else:
        assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_random_fog_compose_preserves_dtype_and_matches_reference(dtype, container):
    rng = np.random.default_rng(137)
    uint8_image = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    image = uint8_image if dtype is np.uint8 else (uint8_image.astype(np.float32) / 255).astype(dtype)
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    transform = A.RandomFog(alpha_coef=0.2, fog_coef_range=(0.6, 0.6), p=1.0)
    pipeline = A.Compose([transform], seed=137, strict=True, save_applied_params=True)
    result = pipeline(image=source)["image"]
    params = get_resolved_applied_params(transform, target="image")
    expected = transform.apply(image, **params)

    assert type(result) is type(source)
    actual = tensor_to_numpy_spatial(result, "image") if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_image = image.astype(np.float32)
        if container == "tensor":
            float32_image = numpy_to_tensor_spatial(float32_image, "image")
        float32_result = A.Compose(
            [A.RandomFog(alpha_coef=0.2, fog_coef_range=(0.6, 0.6), p=1.0)],
            seed=137,
            strict=True,
        )(image=float32_image)["image"]
        if container == "tensor":
            float32_result = tensor_to_numpy_spatial(float32_result, "image")
        np.testing.assert_array_equal(actual, float32_result.astype(np.float16))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_random_sun_flare_overlay_compose_preserves_dtype_and_input(dtype, container):
    rng = np.random.default_rng(137)
    uint8_image = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    image = uint8_image if dtype is np.uint8 else (uint8_image.astype(np.float32) / 255).astype(dtype)
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    transform = A.RandomSunFlare(
        flare_roi=(0, 0, 1, 1),
        angle_range=(0, 0),
        num_flare_circles_range=(2, 2),
        src_radius=15,
        src_color=(255, 255, 255),
        method="overlay",
        p=1.0,
    )
    pipeline = A.Compose([transform], seed=137, strict=True, save_applied_params=True)
    result = pipeline(image=source)["image"]
    params = get_resolved_applied_params(transform, target="image")
    expected = transform.apply(image, **params)

    assert type(result) is type(source)
    actual = tensor_to_numpy_spatial(result, "image") if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_image = image.astype(np.float32)
        if container == "tensor":
            float32_image = numpy_to_tensor_spatial(float32_image, "image")
        float32_result = A.Compose(
            [
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 1),
                    angle_range=(0, 0),
                    num_flare_circles_range=(2, 2),
                    src_radius=15,
                    src_color=(255, 255, 255),
                    method="overlay",
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )(image=float32_image)["image"]
        if container == "tensor":
            float32_result = tensor_to_numpy_spatial(float32_result, "image")
        np.testing.assert_array_equal(actual, float32_result.astype(np.float16))


@pytest.mark.parametrize("mode", ["rain", "mud"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_spatter_single_image_compose_preserves_dtype_and_matches_reference(mode, dtype, container):
    rng = np.random.default_rng(137)
    uint8_image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image = uint8_image if dtype is np.uint8 else (uint8_image.astype(np.float32) / 255).astype(dtype)
    original = image.copy()
    source = numpy_to_tensor_spatial(image.copy(), "image") if container == "tensor" else image.copy()
    if container == "numpy":
        source.flags.writeable = False

    def make_pipeline():
        transform = A.Spatter(mode=mode, color=(137, 89, 211), p=1.0)
        return transform, A.Compose([transform], seed=137, strict=True, save_applied_params=True)

    transform, pipeline = make_pipeline()
    result = pipeline(image=source)["image"]
    params = get_resolved_applied_params(transform, target="image")
    expected = transform.apply(image, **params)

    assert type(result) is type(source)
    actual = tensor_to_numpy_spatial(result, "image") if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, "image") if container == "tensor" else source
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        _, float32_pipeline = make_pipeline()
        float32_source = image.astype(np.float32)
        if container == "tensor":
            float32_source = numpy_to_tensor_spatial(float32_source, "image")
        float32_result = float32_pipeline(image=float32_source)["image"]
        if container == "tensor":
            float32_result = tensor_to_numpy_spatial(float32_result, "image")
        np.testing.assert_array_equal(actual, float32_result.astype(np.float16))


@pytest.mark.parametrize("mode", ["rain", "mud"])
@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_spatter_float16_compose_matches_float32_for_all_targets(mode, target, container):
    shapes = {
        "image": (24, 31, 3),
        "images": (2, 24, 31, 3),
        "volume": (3, 24, 31, 3),
        "volumes": (2, 3, 24, 31, 3),
    }
    rng = np.random.default_rng(137)
    half_image = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
    original = half_image.copy()

    def make_source(image):
        source = image.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(image):
        source = make_source(image)
        pipeline = A.Compose(
            [A.Spatter(mode=mode, color=(137, 89, 211), p=1.0)],
            seed=137,
            strict=True,
        )
        return pipeline(**{target: source})[target], source

    half_result, half_source = run(half_image)
    float32_result, _ = run(half_image.astype(np.float32))
    if container == "tensor":
        half_result = tensor_to_numpy_spatial(half_result, target)
        float32_result = tensor_to_numpy_spatial(float32_result, target)
        half_source = tensor_to_numpy_spatial(half_source, target)

    assert half_result.shape == half_image.shape
    assert half_result.dtype == np.float16
    assert np.all((half_result >= 0) & (half_result <= 1))
    np.testing.assert_allclose(half_result, float32_result.astype(np.float16), rtol=0, atol=1 / 2048)
    np.testing.assert_array_equal(half_source, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_chromatic_aberration_compose_supports_all_targets(target, container, dtype):
    shapes = {
        "image": (31, 37, 3),
        "images": (2, 31, 37, 3),
        "volume": (3, 31, 37, 3),
        "volumes": (2, 3, 31, 37, 3),
    }
    rng = np.random.default_rng(137)
    raw = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    image = raw if dtype is np.uint8 else (raw.astype(np.float32) / 255).astype(dtype)
    original = image.copy()

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose(
            [
                A.ChromaticAberration(
                    primary_distortion_range=(0.02, 0.02),
                    secondary_distortion_range=(0.03, 0.03),
                    mode="red_blue",
                    interpolation=cv2.INTER_LINEAR,
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source
    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))


@pytest.mark.parametrize(
    ("mode", "effect_params"),
    [
        (
            "rain",
            {
                "drops": np.array(
                    [[[-1 / 255, 0.5 / 255, 1.5 / 255], [254.5 / 255, 255.5 / 255, 2]]],
                    dtype=np.float32,
                ),
                "non_mud": None,
                "mud": None,
            },
        ),
        (
            "mud",
            {
                "drops": None,
                "non_mud": np.array([[[1, 0.5, 2], [-1, 1, 1]]], dtype=np.float32),
                "mud": np.array(
                    [[[-1 / 255, 0.5 / 255, -0.5 / 255], [2, -0.5 / 255, 0.5 / 255]]],
                    dtype=np.float32,
                ),
            },
        ),
    ],
)
def test_spatter_apply_to_images_uint8_clipping_and_half_rounding_boundaries(mode, effect_params):
    images = np.array([[[[0, 0, 1], [255, 255, 254]]]], dtype=np.uint8)
    transform = A.Spatter(mode=mode, p=1.0)
    params = {"mode": mode, **effect_params}

    expected = ImageOnlyTransform.apply_to_images(transform, images, **params)
    actual = transform.apply_to_images(images, **params)

    np.testing.assert_array_equal(actual, expected)
    if mode == "rain":
        np.testing.assert_array_equal(actual, np.array([[[[0, 0, 2], [255, 255, 255]]]], dtype=np.uint8))
    else:
        np.testing.assert_array_equal(actual, np.array([[[[0, 0, 2], [255, 254, 254]]]], dtype=np.uint8))


@pytest.mark.parametrize("mode", ["rain", "mud"])
def test_spatter_apply_to_images_float32_clipping_vectors(mode):
    images = np.array([[[[-0.25, 0.25, 1.25]]]], dtype=np.float32)
    transform = A.Spatter(mode=mode, p=1.0)
    if mode == "rain":
        params = {
            "mode": mode,
            "drops": np.array([[[-0.5, 0.5, 0.5]]], dtype=np.float32),
            "non_mud": None,
            "mud": None,
        }
    else:
        params = {
            "mode": mode,
            "drops": None,
            "non_mud": np.array([[[2, 2, 2]]], dtype=np.float32),
            "mud": np.array([[[-0.5, 0.25, -0.25]]], dtype=np.float32),
        }

    expected = ImageOnlyTransform.apply_to_images(transform, images, **params)
    actual = transform.apply_to_images(images, **params)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual, np.array([[[[0, 0.75, 1]]]], dtype=np.float32))


@pytest.mark.parametrize(
    ("mode", "dtype", "height", "width", "batch_at_threshold"),
    [
        ("rain", np.uint8, 256, 256, 4),
        ("mud", np.uint8, 256, 256, 8),
        ("mud", np.float32, 512, 512, 8),
    ],
)
def test_spatter_apply_to_images_switches_at_working_set_guard(
    monkeypatch,
    mode,
    dtype,
    height,
    width,
    batch_at_threshold,
):
    fallback_calls = 0

    def fallback(transform, images, *args, **forwarded_params):
        nonlocal fallback_calls
        fallback_calls += 1
        assert not args
        assert forwarded_params["mode"] == mode
        assert forwarded_params["drops"] is params["drops"]
        assert forwarded_params["non_mud"] is params["non_mud"]
        assert forwarded_params["mud"] is params["mud"]
        return np.full_like(images, 137 if dtype == np.uint8 else 0.5)

    monkeypatch.setattr(ImageOnlyTransform, "apply_to_images", fallback)
    transform = A.Spatter(mode=mode, p=1.0)
    if mode == "rain":
        params = {
            "mode": mode,
            "drops": np.zeros((height, width, 3), dtype=np.float32),
            "non_mud": None,
            "mud": None,
        }
    else:
        params = {
            "mode": mode,
            "drops": None,
            "non_mud": np.ones((height, width, 3), dtype=np.float32),
            "mud": np.zeros((height, width, 3), dtype=np.float32),
        }

    below = np.zeros((batch_at_threshold - 1, height, width, 3), dtype=dtype)
    below_result = transform.apply_to_images(below, **params)
    assert fallback_calls == 0
    np.testing.assert_array_equal(below_result, below)

    at_threshold = np.zeros((batch_at_threshold, height, width, 3), dtype=dtype)
    threshold_result = transform.apply_to_images(at_threshold, **params)
    assert fallback_calls == 1
    np.testing.assert_array_equal(
        threshold_result,
        np.full_like(at_threshold, 137 if dtype == np.uint8 else 0.5),
    )


@pytest.mark.parametrize(
    ("mode", "dtype"),
    [("rain", np.uint8), ("mud", np.uint8), ("mud", np.float32)],
)
def test_spatter_fallback_passes_explicit_parameters_to_base_handler(monkeypatch, mode, dtype):
    images = np.zeros((2, 5, 7, 3), dtype=dtype)
    effect = np.zeros(images.shape[1:], dtype=np.float32)
    params = (
        {"mode": mode, "drops": effect, "non_mud": None, "mud": None}
        if mode == "rain"
        else {"mode": mode, "drops": None, "non_mud": np.ones_like(effect), "mud": effect}
    )
    transform = A.Spatter(mode=mode, p=1.0)
    monkeypatch.setitem(weather._SPATTER_BATCH_FALLBACK_WORKING_SET_BYTES, (mode, images.dtype.name), 0)

    actual = transform.apply_to_images(images, **params)
    expected = ImageOnlyTransform.apply_to_images(transform, images, **params)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_spatter_apply_to_images_preserves_empty_batch_identity(dtype):
    images = np.empty((0, 17, 19, 3), dtype=dtype)
    transform = A.Spatter(p=1.0)

    assert transform.apply_to_images(images, mode="rain", drops=None, non_mud=None, mud=None) is images
    assert A.Compose([transform], seed=137, strict=True)(images=images)["images"] is images


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 17, 19, 1), id="one-channel"),
        pytest.param((2, 17, 19, 5), id="five-channel"),
        pytest.param((2, 17, 19), id="raw-grayscale"),
        pytest.param((3, 3, 3), id="raw-grayscale-width-three"),
    ],
)
def test_spatter_apply_to_images_rejects_non_rgb_batches(shape):
    images = np.zeros(shape, dtype=np.uint8)
    transform = A.Spatter(p=1.0)

    with pytest.raises(ValueError, match="This transformation expects 3-channel images"):
        transform.apply_to_images(
            images,
            mode="rain",
            drops=np.zeros((*shape[1:3], 3), dtype=np.float32),
            non_mud=None,
            mud=None,
        )


@pytest.mark.parametrize("mode", ["rain", "mud"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_spatter_batch_path_preserves_volume_routing(mode, dtype):
    rng = np.random.default_rng(137)
    if dtype == np.uint8:
        volume = rng.integers(0, 256, (3, 17, 19, 3), dtype=np.uint8)
    else:
        volume = rng.random((3, 17, 19, 3), dtype=np.float32)
    transform = A.Spatter(mode=mode, p=1.0)

    actual = A.Compose([transform], save_applied_params=True, seed=137, strict=True)(volume=volume)["volume"]
    params = get_resolved_applied_params(transform, "volume")
    expected = ImageOnlyTransform.apply_to_images(transform, volume, **params)

    if dtype == np.uint8:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_enhance_invalid_mode_raises():
    with pytest.raises(ValueError):
        A.Enhance(mode="invalid", alpha_range=(0.5, 1.0), p=1.0)


@pytest.mark.parametrize(
    "alpha_range",
    [
        (-0.1, 0.5),  # negative lower bound
        (1.0, 0.5),  # decreasing
    ],
)
def test_enhance_invalid_alpha_range_raises(alpha_range):
    with pytest.raises(ValueError):
        A.Enhance(mode="edge", alpha_range=alpha_range, p=1.0)


def test_enhance_invalid_mode_in_functional_raises():
    """generate_enhance_matrix must raise ValueError (not KeyError) on bad mode."""
    with pytest.raises(ValueError, match="Unsupported enhance mode"):
        fpixel.generate_enhance_matrix("not_a_mode", 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["edge", "detail"])
def test_enhance_applied_config_resolves_range_to_scalar(mode):
    """applied_config must record the *sampled* alpha (a scalar in alpha_range), not the range itself.

    This enforces the contract documented on BasicTransform.get_applied_config:
    "all constructor params and range params resolved to sampled scalar values".
    """
    image = np.random.RandomState(137).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    alpha_range = (0.3, 0.9)
    aug = A.Enhance(mode=mode, alpha_range=alpha_range, p=1.0)
    aug(image=image)

    sampled = aug.applied_config["alpha_range"]
    assert isinstance(sampled, float), f"expected sampled scalar, got {type(sampled).__name__}: {sampled!r}"
    assert alpha_range[0] <= sampled <= alpha_range[1]
    # mode is already covered by the base init args; verify it's preserved
    assert aug.applied_config["mode"] == mode


@pytest.mark.parametrize(
    ("mode", "alpha", "pil_filter_name"),
    [
        ("edge", 1.0, "EDGE_ENHANCE"),
        ("edge", 2.0, "EDGE_ENHANCE_MORE"),
        ("detail", 1.0, "DETAIL"),
    ],
)
def test_enhance_matches_pillow_interior(mode, alpha, pil_filter_name):
    """Enhance must reproduce PIL's preset on interior pixels (within 1 LSB).

    Border pixels are excluded because cv2 defaults to BORDER_REFLECT_101 while
    PIL uses replicate at borders. Integer-only kernel (EDGE_ENHANCE_MORE) is
    bit-exact; fractional kernels can differ by 1 LSB due to rounding mode.
    """
    pytest.importorskip("PIL")
    from PIL import Image, ImageFilter

    image = np.random.RandomState(137).randint(0, 256, (96, 96, 3), dtype=np.uint8)

    pil_filter = getattr(ImageFilter, pil_filter_name)
    pil_out = np.array(Image.fromarray(image).filter(pil_filter))

    transform = A.Compose([A.Enhance(mode=mode, alpha_range=(alpha, alpha), p=1.0)])
    ours = transform(image=image)["image"]

    pil_interior = pil_out[1:-1, 1:-1]
    ours_interior = ours[1:-1, 1:-1]

    diff = np.abs(pil_interior.astype(np.int16) - ours_interior.astype(np.int16))
    assert diff.max() <= 1, f"max abs diff {diff.max()} exceeds 1 LSB tolerance vs PIL {pil_filter_name}"
    if pil_filter_name == "EDGE_ENHANCE_MORE":
        np.testing.assert_array_equal(pil_interior, ours_interior)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_auto_contrast_compose_supports_all_targets_and_dtypes(target, container, channels, dtype):
    shapes = {
        "image": (19, 23, channels),
        "images": (2, 19, 23, channels),
        "volume": (3, 19, 23, channels),
        "volumes": (2, 3, 19, 23, channels),
    }
    rng = np.random.default_rng(137)
    if dtype is np.uint8:
        image = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    else:
        half = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
        image = half if dtype is np.float16 else half.astype(np.float32)
    original = image.copy()

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose([A.AutoContrast(cutoff=0, method="cdf", p=1.0)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source
    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_halftone_compose_supports_all_targets_and_dtypes(target, container, channels, dtype):
    shapes = {
        "image": (19, 23, channels),
        "images": (2, 19, 23, channels),
        "volume": (3, 19, 23, channels),
        "volumes": (2, 3, 19, 23, channels),
    }
    rng = np.random.default_rng(137)
    if dtype is np.uint8:
        image = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    else:
        half = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
        image = half if dtype is np.float16 else half.astype(np.float32)
    original = image.copy()

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose([A.Halftone(dot_size_range=(4, 4), blend_range=(0, 0), p=1.0)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source
    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    np.testing.assert_array_equal(preserved, original)
    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))
