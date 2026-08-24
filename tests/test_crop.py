import math
from typing import Literal

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.crops.functional import crop_bboxes_by_coords
from albumentations.core.invocation import SamplingContext

from .conftest import IMAGES, RECTANGULAR_UINT8_IMAGE
from .utils import make_sampling_input


def test_random_crop_vs_crop(bboxes, keypoints):
    image = RECTANGULAR_UINT8_IMAGE
    image_height, image_width = image.shape[:2]

    mask = np.random.randint(0, 2, image.shape[:2], dtype=np.uint8)

    random_crop_transform = A.Compose(
        [A.RandomCrop(height=image_height, width=image_width, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        strict=True,
    )
    crop_transform = A.Compose(
        [A.Crop(x_min=0, y_min=0, x_max=image_width, y_max=image_height, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        strict=True,
    )

    random_crop_result = random_crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    crop_result = crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(random_crop_result["image"], crop_result["image"])
    np.testing.assert_array_equal(random_crop_result["mask"], crop_result["mask"])

    np.testing.assert_equal(random_crop_result["bboxes"], crop_result["bboxes"])
    np.testing.assert_equal(random_crop_result["keypoints"], crop_result["keypoints"])


def test_center_crop_vs_crop(bboxes, keypoints):
    image = RECTANGULAR_UINT8_IMAGE
    height, width = 50, 50
    img_height, img_width = image.shape[:2]
    mask = np.random.randint(0, 2, image.shape[:2], dtype=np.uint8)

    center_crop_transform = A.Compose(
        [A.CenterCrop(height=height, width=width, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        strict=True,
    )
    crop_transform = A.Compose(
        [
            A.Crop(
                x_min=(img_width - width) // 2,
                y_min=(img_height - height) // 2,
                x_max=(img_width + width) // 2,
                y_max=(img_height + height) // 2,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        strict=True,
    )

    center_crop_result = center_crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    crop_result = crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(center_crop_result["image"], crop_result["image"])
    np.testing.assert_array_equal(center_crop_result["mask"], crop_result["mask"])

    np.testing.assert_equal(center_crop_result["bboxes"], crop_result["bboxes"])
    np.testing.assert_equal(center_crop_result["keypoints"], crop_result["keypoints"])


@pytest.mark.parametrize("image", IMAGES)
def test_crop_near_bbox(image, bboxes, keypoints):
    bbox_key = "target_bbox"
    aug = A.Compose(
        [A.RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key=bbox_key, p=1)],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        strict=True,
    )

    aug(image=image, bboxes=bboxes, target_bbox=[0, 5, 10, 20], keypoints=keypoints)

    target_keys = {
        "image",
        "images",
        "bboxes",
        "labels",
        "mask",
        "masks",
        "keypoints",
        "volume",
        "mask3d",
        "user_data",
        bbox_key,
    }

    assert aug._available_keys == target_keys

    aug2 = A.Compose(
        [A.Sequential([A.RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key=bbox_key, p=1)])],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        strict=True,
    )

    assert aug2._available_keys == target_keys


def test_crop_bbox_by_coords():
    cropped_bbox = crop_bboxes_by_coords(
        np.array([[0.5, 0.2, 0.9, 0.7]]),
        (18, 18, 82, 82),
        (100, 100),
    )
    np.testing.assert_array_almost_equal(cropped_bbox, np.array([[0.5, 0.03125, 1.125, 0.8125]]))


@pytest.mark.parametrize(
    ["transforms", "bboxes", "expected_bboxes", "min_area", "min_visibility"],
    [
        [[A.Crop(10, 10, 20, 20)], [[0, 0, 10, 10, 0]], [], 0, 0],
        [
            [A.Crop(0, 0, 90, 90)],
            [[0, 0, 91, 91, 0], [0, 0, 89, 89, 0]],
            [[0, 0, 90, 90, 0], [0, 0, 89, 89, 0]],
            0,
            0.9,
        ],
        [
            [A.Crop(0, 0, 90, 90)],
            [[0, 0, 1, 10, 0], [0, 0, 1, 11, 0]],
            [[0, 0, 1, 10, 0], [0, 0, 1, 11, 0]],
            10,
            0,
        ],
    ],
)
def test_bbox_params_edges(
    transforms,
    bboxes,
    expected_bboxes,
    min_area: float,
    min_visibility: float,
) -> None:
    image = np.empty([100, 100, 3], dtype=np.uint8)
    aug = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            "pascal_voc",
            min_area=min_area,
            min_visibility=min_visibility,
        ),
        strict=True,
    )
    res = aug(image=image, bboxes=bboxes)["bboxes"]
    # Handle comparison when expected is empty list vs result is empty array with shape
    if len(expected_bboxes) == 0:
        assert len(res) == 0
    else:
        np.testing.assert_allclose(res, expected_bboxes, rtol=1e-6, atol=1e-6)


POSITIONS = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]


@pytest.mark.parametrize(
    ["crop_cls", "crop_params"],
    [
        (A.RandomCrop, {"height": 150, "width": 150}),
        (A.CenterCrop, {"height": 150, "width": 150}),
    ],
)
@pytest.mark.parametrize("pad_position", POSITIONS)
@pytest.mark.parametrize("border_mode", [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT])
def test_pad_position_equivalence(
    image: np.ndarray,
    crop_cls: type[A.DualTransform],
    crop_params: dict[str, int],
    pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"],
    border_mode: int,
    mask: np.ndarray,
    bboxes: np.ndarray,
    keypoints: np.ndarray,
):
    """Test that pad_position works identically for both padding approaches."""
    # Approach 1: Crop with built-in padding
    transform1 = A.Compose(
        [
            crop_cls(
                **crop_params,
                pad_if_needed=True,
                border_mode=border_mode,
                fill=0,
                pad_position=pad_position,
            ),
        ],
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        strict=True,
    )

    # Approach 2: Separate pad and crop
    transform2 = A.Compose(
        [
            A.PadIfNeeded(
                min_height=crop_params["height"],
                min_width=crop_params["width"],
                border_mode=border_mode,
                fill=0,
                position=pad_position,
            ),
            crop_cls(
                **crop_params,
                pad_if_needed=False,
            ),
        ],
        keypoint_params=A.KeypointParams(coord_format="xyas"),
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        strict=True,
    )

    result1 = transform1(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    result2 = transform2(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(
        result1["image"],
        result2["image"],
        err_msg=f"Images don't match for position {pad_position}",
    )
    np.testing.assert_array_equal(
        result1["mask"],
        result2["mask"],
        err_msg=f"Masks don't match for position {pad_position}",
    )
    np.testing.assert_array_equal(
        result1["bboxes"],
        result2["bboxes"],
        err_msg=f"Bboxes don't match for position {pad_position}",
    )
    np.testing.assert_array_equal(
        result1["keypoints"],
        result2["keypoints"],
        err_msg=f"Keypoints don't match for position {pad_position}",
    )


def test_base_crop_and_pad_fill():
    # tests whether BaseCropAndPad usues correct values for constant borders
    c = A.CenterCrop(4, 4, pad_if_needed=True, fill=100, fill_mask=200)
    c1 = A.CenterCrop(4, 4, pad_if_needed=True, fill=201)

    im = np.zeros((2, 6, 3)).astype(np.float32)
    msk = np.zeros((2, 6, 1)).astype(np.uint8)
    mask3d = np.zeros((2, 2, 6, 1), dtype=np.uint8)

    out = c(image=im, mask=msk, mask3d=mask3d)
    out1 = c1(image=im, mask=msk, mask3d=mask3d)

    expected_img = np.ones((4, 4, 3)).astype(np.float32)
    expected_img[1:3, ...] = 0

    expected_msk = np.ones((4, 4, 1)).astype(np.uint8)
    expected_msk[1:3, ...] = 0
    expected_mask3d = np.ones((2, 4, 4, 1), dtype=np.uint8)
    expected_mask3d[:, 1:3, ...] = 0

    np.testing.assert_array_equal(out["image"], expected_img * 100)
    np.testing.assert_array_equal(out["mask"], expected_msk * 200)
    np.testing.assert_array_equal(out["mask3d"], expected_mask3d * 200)

    np.testing.assert_array_equal(out1["image"], expected_img * 201)
    np.testing.assert_array_equal(out1["mask"], expected_msk * 0)  # 0 is the default for fill_mask
    np.testing.assert_array_equal(out1["mask3d"], expected_mask3d * 0)


@pytest.mark.parametrize(
    ["image_shape", "crop_coords", "pad_position"],
    [
        # Case 1: Inside crop, no padding needed
        ((100, 100, 3), (10, 20, 60, 80), "center"),
        # Case 2: Width > image_width, requires padding (center)
        ((100, 100, 3), (10, 20, 120, 80), "center"),
        # Case 3: Crop extends beyond image height, but crop_height <= image_height, no padding needed
        ((100, 100, 3), (10, 20, 60, 120), "center"),
        # Case 4: Width > image_width and Height > image_height, requires padding (center)
        ((100, 100, 3), (10, 20, 120, 130), "center"),
        # Case 7: Crop partially outside (large x, y), no padding needed, clips crop region
        ((100, 100, 3), (90, 90, 120, 120), "center"),
        # Case 9: Width > image_width, requires padding (top_left)
        ((100, 100, 3), (10, 20, 120, 80), "top_left"),
        # Case 10: Width > image_width and Height > image_height, requires padding (top_left)
        ((100, 100, 3), (10, 20, 120, 130), "top_left"),
    ],
)
def test_crop_pad_if_needed(image_shape, crop_coords, pad_position):
    """Tests Crop transform with pad_if_needed=True ensures output has requested crop shape."""
    image = np.ones(image_shape, dtype=np.uint8) * 255
    x_min, y_min, x_max, y_max = crop_coords

    expected_h = y_max - y_min
    expected_w = x_max - x_min
    expected_shape = (expected_h, expected_w, image_shape[2])

    transform = A.Crop(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        pad_if_needed=True,
        pad_position=pad_position,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,  # Fill value doesn't affect shape test
        p=1.0,
    )

    result = transform(image=image)
    transformed_image = result["image"]

    assert transformed_image.shape == expected_shape


def test_at_least_one_bbox_random_crop_with_multiple_bboxes():
    """Test that AtLeastOneBBoxRandomCrop works with multiple bboxes without ValueError.

    Regression test for issue #104: py_random.choice(numpy_array) raises ValueError
    in Python 3.11 when array has more than one element.
    """
    image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (300, 300), dtype=np.uint8)

    # Create multiple bounding boxes to trigger the bug
    bboxes = np.array(
        [
            [30, 50, 100, 140],
            [150, 120, 270, 250],
            [200, 30, 280, 90],
        ],
        dtype=np.float32,
    )
    bbox_labels = [1, 2, 3]

    keypoints = np.array(
        [
            [50, 70],
            [190, 170],
            [240, 60],
        ],
        dtype=np.float32,
    )
    keypoint_labels = [0, 1, 2]

    transform = A.Compose(
        [
            A.AtLeastOneBBoxRandomCrop(
                height=200,
                width=200,
                erosion_factor=0.2,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
    )

    # This should not raise ValueError
    transformed = transform(
        image=image,
        mask=mask,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
        keypoints=keypoints,
        keypoint_labels=keypoint_labels,
    )

    # Verify that at least one bounding box was preserved
    assert len(transformed["bboxes"]) > 0
    assert len(transformed["bbox_labels"]) > 0
    assert transformed["image"].shape == (200, 200, 3)
    assert transformed["mask"].shape == (200, 200)


@pytest.mark.parametrize("num_bboxes", [1, 10, 100, 1000])
def test_at_least_one_bbox_random_crop_efficiency(num_bboxes):
    """Test that AtLeastOneBBoxRandomCrop efficiently handles varying numbers of bboxes."""
    image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

    # Generate random valid bboxes
    np.random.seed(137)
    x1 = np.random.randint(10, 250, num_bboxes).astype(np.float32)
    y1 = np.random.randint(10, 250, num_bboxes).astype(np.float32)
    x2 = (x1 + np.random.randint(20, 40, num_bboxes)).astype(np.float32)
    y2 = (y1 + np.random.randint(20, 40, num_bboxes)).astype(np.float32)
    bboxes = np.stack([x1, y1, x2, y2], axis=1)
    bbox_labels = list(range(num_bboxes))

    transform = A.Compose(
        [
            A.AtLeastOneBBoxRandomCrop(
                height=200,
                width=200,
                erosion_factor=0.2,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
    )

    # Should work efficiently without converting to list
    transformed = transform(image=image, bboxes=bboxes, bbox_labels=bbox_labels)

    assert len(transformed["bboxes"]) > 0
    assert len(transformed["bbox_labels"]) > 0
    assert transformed["image"].shape == (200, 200, 3)


def _subset_safe_crop_bboxes() -> np.ndarray:
    # Six spatially separated normalized boxes so any subset yields a distinct union.
    return np.array(
        [
            [0.02, 0.02, 0.12, 0.12],
            [0.20, 0.02, 0.30, 0.12],
            [0.02, 0.20, 0.12, 0.30],
            [0.20, 0.20, 0.30, 0.30],
            [0.40, 0.40, 0.50, 0.50],
            [0.60, 0.60, 0.98, 0.98],
        ],
        dtype=np.float32,
    )


def test_bbox_subset_safe_random_crop_selects_valid_subset_size():
    bboxes = _subset_safe_crop_bboxes()
    num_bboxes = len(bboxes)
    subset_fraction_range = (0.3, 0.8)
    min_subset_size = math.ceil(num_bboxes * subset_fraction_range[0])
    max_subset_size = min(math.ceil(num_bboxes * subset_fraction_range[1]), num_bboxes)

    transform = A.BBoxSubsetSafeRandomCrop(
        subset_fraction_range=subset_fraction_range,
        erosion_rate=0.0,
        aspect_ratio_range=(0.01, 100.0),
        p=1.0,
    )

    observed_counts = set()
    for seed in range(40):
        transform.set_random_seed(seed)
        sampling = SamplingContext.from_owner(transform, {})
        data = {"image": np.zeros((400, 400, 3), dtype=np.uint8), "bboxes": bboxes}
        result = transform.sample_parameters(make_sampling_input(transform, data), sampling).shared
        selected = result["bbox_indices"]

        assert min_subset_size <= len(selected) <= max_subset_size
        assert len(set(selected)) == len(selected)
        observed_counts.add(len(selected))

    assert observed_counts == set(range(min_subset_size, max_subset_size + 1))


def test_bbox_subset_safe_random_crop_preserves_selected_boxes_without_erosion():
    bboxes = _subset_safe_crop_bboxes()
    image_shape = (400, 400)

    transform = A.BBoxSubsetSafeRandomCrop(
        subset_fraction_range=(0.3, 0.8),
        erosion_rate=0.0,
        aspect_ratio_range=(0.01, 100.0),
        p=1.0,
    )

    for seed in range(40):
        transform.set_random_seed(seed)
        sampling = SamplingContext.from_owner(transform, {})
        data = {"image": np.zeros((*image_shape, 3), dtype=np.uint8), "bboxes": bboxes}
        result = transform.sample_parameters(make_sampling_input(transform, data), sampling).shared
        selected = result["bbox_indices"]
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = result["crop_coords"]

        for idx in selected:
            box_x_min, box_y_min, box_x_max, box_y_max = bboxes[idx]
            px_min = box_x_min * image_shape[1]
            py_min = box_y_min * image_shape[0]
            px_max = box_x_max * image_shape[1]
            py_max = box_y_max * image_shape[0]

            assert crop_x_min <= px_min
            assert crop_y_min <= py_min
            assert crop_x_max >= px_max
            assert crop_y_max >= py_max


def test_bbox_subset_safe_random_crop_enforces_feasible_aspect_ratio():
    bboxes = np.array([[0.45, 0.35, 0.55, 0.65]], dtype=np.float32)
    transform = A.BBoxSubsetSafeRandomCrop(
        subset_fraction_range=(1.0, 1.0),
        aspect_ratio_range=(0.2, 0.2),
        p=1.0,
    )

    for seed in range(20):
        transform.set_random_seed(seed)
        sampling = SamplingContext.from_owner(transform, {})
        data = {"image": np.zeros((320, 640, 3), dtype=np.uint8), "bboxes": bboxes}
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = transform.sample_parameters(
            make_sampling_input(transform, data), sampling
        ).shared["crop_coords"]
        assert (crop_y_max - crop_y_min) / (crop_x_max - crop_x_min) == 0.2


def test_bbox_subset_safe_random_crop_applies_aspect_ratio_without_protected_boxes():
    transform = A.BBoxSubsetSafeRandomCrop(
        subset_fraction_range=(1.0, 1.0),
        erosion_rate=1.0,
        aspect_ratio_range=(0.2, 0.2),
        p=1.0,
    )
    sampling = SamplingContext.from_owner(transform, {})
    data = {
        "image": np.zeros((320, 640, 3), dtype=np.uint8),
        "bboxes": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
    }
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = transform.sample_parameters(
        make_sampling_input(transform, data), sampling
    ).shared["crop_coords"]

    assert (crop_y_max - crop_y_min) / (crop_x_max - crop_x_min) == 0.2


def test_bbox_subset_safe_random_crop_returns_full_image_when_aspect_ratio_is_infeasible():
    transform = A.BBoxSubsetSafeRandomCrop(
        subset_fraction_range=(1.0, 1.0),
        aspect_ratio_range=(1.5, 2.0),
        p=1.0,
    )
    sampling = SamplingContext.from_owner(transform, {})
    data = {
        "image": np.zeros((100, 200, 3), dtype=np.uint8),
        "bboxes": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
    }
    result = transform.sample_parameters(make_sampling_input(transform, data), sampling).shared

    assert result == {"crop_coords": (0, 0, 200, 100), "bbox_indices": (0,)}
