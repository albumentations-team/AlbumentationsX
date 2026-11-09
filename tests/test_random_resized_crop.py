"""Focused regression tests for rejection-free 2D RandomResizedCrop sampling."""

import math
from collections import Counter
from typing import Any, Literal, cast
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.crops._sampling import (
    SamplingMethod,
    _integral_linear_exp,
    _sample_standard_2d_area,
    _standard_2d_segments,
    relax_2d_crop_ranges,
    sample_2d_crop_shape,
)
from albumentations.augmentations.crops.sized import _bbox_envelope, _clip_polygon_to_image
from albumentations.core.bbox_utils import obb_to_polygons


@pytest.mark.parametrize("sampling_method", ("standard", "uniform_scale"))
def test_random_resized_crop_samples_a_feasible_shape_without_retries(sampling_method: SamplingMethod) -> None:
    random = Mock()
    random.random.side_effect = (0.2, 0.8)

    crop_shape = sample_2d_crop_shape(
        height=120,
        width=240,
        scale=(0.08, 1.0),
        ratio=(0.75, 4.0 / 3.0),
        sampling_method=sampling_method,
        py_random=random,
    )

    assert crop_shape is not None
    crop_height, crop_width = crop_shape
    assert 1 <= crop_height <= 120
    assert 1 <= crop_width <= 240
    assert random.random.call_count == 2


def test_random_resized_crop_keeps_a_degenerate_feasible_crop() -> None:
    assert sample_2d_crop_shape(
        height=80,
        width=120,
        scale=(1.0, 1.0),
        ratio=(1.5, 1.5),
        sampling_method="standard",
        py_random=Mock(random=Mock(return_value=0.5)),
    ) == (80, 120)


def test_standard_area_cdf_inversion_is_numerically_precise() -> None:
    segments = _standard_2d_segments(120, 240, 0.08, 1.0, 0.75, 4.0 / 3.0)
    total_mass = segments[-1][4] + segments[-1][5]

    for quantile in np.linspace(1e-9, 1.0 - 1e-9, 101):
        area = _sample_standard_2d_area(segments, quantile)
        log_area = math.log(area)
        cumulative = 0.0
        for left, right, slope, intercept, prior_mass, mass in segments:
            if log_area <= right:
                cumulative = (
                    prior_mass
                    + _integral_linear_exp(slope, intercept, log_area)
                    - _integral_linear_exp(
                        slope,
                        intercept,
                        left,
                    )
                )
                break
            cumulative = prior_mass + mass
        assert cumulative / total_mass == pytest.approx(quantile, abs=1e-10)


def test_random_resized_crop_reports_no_shape_only_for_an_empty_region() -> None:
    assert (
        sample_2d_crop_shape(
            height=80,
            width=120,
            scale=(0.0, 0.0),
            ratio=(0.75, 4.0 / 3.0),
            sampling_method="standard",
            py_random=Mock(),
        )
        is None
    )


@pytest.mark.parametrize("sampling_method", ("standard", "uniform_scale"))
def test_random_resized_crop_shape_respects_minimum_crop_shape(sampling_method: SamplingMethod) -> None:
    sampled_shape = sample_2d_crop_shape(
        height=80,
        width=120,
        scale=(0.08, 0.2),
        ratio=(0.75, 1.5),
        sampling_method=sampling_method,
        py_random=Mock(random=Mock(return_value=0.5)),
        min_crop_shape=(60, 70),
    )
    assert sampled_shape is None

    relaxed_scale, relaxed_ratio = relax_2d_crop_ranges(
        height=80,
        width=120,
        scale=(0.08, 0.2),
        ratio=(0.75, 1.5),
        min_crop_shape=(60, 70),
    )
    assert relaxed_scale[0] > 0.2

    sampled_shape = sample_2d_crop_shape(
        height=80,
        width=120,
        scale=relaxed_scale,
        ratio=relaxed_ratio,
        sampling_method=sampling_method,
        py_random=Mock(random=Mock(return_value=0.5)),
        min_crop_shape=(60, 70),
    )
    assert sampled_shape is not None
    crop_height, crop_width = sampled_shape
    assert crop_height >= 60
    assert crop_width >= 70


def test_relax_2d_crop_ranges_keeps_closest_ratio_before_relaxing_scale() -> None:
    relaxed_scale, relaxed_ratio = relax_2d_crop_ranges(
        height=80,
        width=120,
        scale=(0.08, 0.2),
        ratio=(0.4, 0.5),
        min_crop_shape=(70, 50),
    )

    assert relaxed_ratio[0] == relaxed_ratio[1]
    assert relaxed_ratio[0] == pytest.approx(49.5 / 80.5)
    assert relaxed_scale[0] == relaxed_scale[1]
    assert relaxed_scale[0] > 0.2

    sampled_shape = sample_2d_crop_shape(
        height=80,
        width=120,
        scale=relaxed_scale,
        ratio=relaxed_ratio,
        sampling_method="standard",
        py_random=Mock(random=Mock(return_value=0.5)),
        min_crop_shape=(70, 50),
    )
    assert sampled_shape == (80, 50)


def test_clip_polygon_to_image_returns_visible_polygon_envelope() -> None:
    polygon = np.array([[-100.0, -50.0], [-100.0, 100.0], [50.0, 50.0]], dtype=np.float32)
    image_polygon = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=np.float32)
    area, intersection = cv2.intersectConvexConvex(polygon, image_polygon)

    visible_envelope = _clip_polygon_to_image(polygon, image_height=100, image_width=100)

    assert area > 0
    assert intersection is not None
    clipped_points = intersection.reshape(-1, 2)
    expected = (
        float(clipped_points[:, 0].min()),
        float(clipped_points[:, 1].min()),
        float(clipped_points[:, 0].max()),
        float(clipped_points[:, 1].max()),
    )
    assert visible_envelope == pytest.approx(expected)
    assert visible_envelope is not None
    assert visible_envelope[1] > 0.0
    assert _clip_polygon_to_image(polygon - 1000.0, image_height=100, image_width=100) is None
    assert _clip_polygon_to_image(np.array([[0.0, 5.0], [5.0, 0.0], [10.0, -5.0]]), 10, 10) is None


def test_random_resized_crop_projects_partial_obb_in_pixel_coordinates() -> None:
    image_shape = (100, 200)
    bbox = np.array([0.0, 0.3, 0.4, 0.7, 45.0], dtype=np.float32)
    bbox_pixels = bbox.copy()
    bbox_pixels[[0, 2]] *= image_shape[1]
    bbox_pixels[[1, 3]] *= image_shape[0]
    polygon = obb_to_polygons(bbox_pixels.reshape(1, 5))[0]
    image_polygon = np.array(
        [[0, 0], [image_shape[1], 0], [image_shape[1], image_shape[0]], [0, image_shape[0]]],
        dtype=np.float32,
    )
    area, clipped = cv2.intersectConvexConvex(polygon, image_polygon)

    envelope = _bbox_envelope(bbox, "obb", image_shape)

    assert area > 0
    assert clipped is not None
    clipped_points = clipped.reshape(-1, 2)
    expected = (
        float(clipped_points[:, 0].min()),
        float(clipped_points[:, 1].min()),
        float(clipped_points[:, 0].max()),
        float(clipped_points[:, 1].max()),
    )
    assert envelope == pytest.approx(expected)


@pytest.mark.parametrize(
    "bbox",
    (
        (0.4, 0.3, 0.65, 0.6, 45.0),
        (-0.05, 0.3, 0.2, 0.7, 45.0),
        (1.2, 0.2, 1.4, 0.4, 20.0),
    ),
)
def test_obb_envelope_matches_clipped_polygon(bbox: tuple[float, ...]) -> None:
    image_shape = (100, 200)
    bbox_pixels = np.asarray(bbox, dtype=np.float64)
    bbox_pixels[[0, 2]] *= image_shape[1]
    bbox_pixels[[1, 3]] *= image_shape[0]
    polygon = obb_to_polygons(bbox_pixels.reshape(1, 5))[0].astype(np.float32)
    frame_polygon = np.array(
        [[0, 0], [image_shape[1], 0], [image_shape[1], image_shape[0]], [0, image_shape[0]]],
        dtype=np.float32,
    )
    area, clipped = cv2.intersectConvexConvex(polygon, frame_polygon)

    result = _bbox_envelope(np.asarray(bbox), "obb", image_shape)

    if area <= 0 or clipped is None:
        assert result is None
    else:
        clipped_points = clipped.reshape(-1, 2)
        assert result == pytest.approx(
            (
                float(clipped_points[:, 0].min()),
                float(clipped_points[:, 1].min()),
                float(clipped_points[:, 0].max()),
                float(clipped_points[:, 1].max()),
            ),
        )


def test_random_resized_crop_normalizes_large_finite_obb_angles() -> None:
    image_shape = (100, 200)
    bbox = np.array([0.4, 0.3, 0.65, 0.6, 0.0], dtype=np.float32)
    equivalent_bbox = bbox.copy()
    equivalent_bbox[4] += 360.0 * 2**18

    assert _bbox_envelope(equivalent_bbox, "obb", image_shape) == pytest.approx(
        _bbox_envelope(bbox, "obb", image_shape),
    )


def test_random_resized_crop_preserves_one_random_box() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox_sets = [[(5, 5, 15, 15)], [(80, 80, 90, 90)]]
    selected_counts: Counter[int] = Counter()
    transform = A.ReplayCompose(
        [
            A.RandomResizedCrop(
                size=(32, 32),
                scale=(0.01, 0.02),
                ratio=(1.0, 1.0),
                bbox_preservation="one_box",
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        additional_targets={"teacher_bboxes": "bboxes"},
        seed=137,
    )

    for seed in range(100, 200):
        transform.set_random_seed(seed)
        result = transform(image=image, bboxes=bbox_sets[0], teacher_bboxes=bbox_sets[1])
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = result["replay"]["transforms"][0]["params"]["params"][
            "crop_coords"
        ]

        contained_ids = [
            index
            for index, (x_min, y_min, x_max, y_max) in enumerate(boxes[0] for boxes in bbox_sets)
            if crop_x_min <= x_min and crop_y_min <= y_min and crop_x_max >= x_max and crop_y_max >= y_max
        ]
        assert len(contained_ids) == 1
        assert len(result["bboxes"]) == int(contained_ids[0] == 0)
        assert len(result["teacher_bboxes"]) == int(contained_ids[0] == 1)
        selected_counts[contained_ids[0]] += 1

    assert 40 <= selected_counts[0] <= 60
    assert 40 <= selected_counts[1] <= 60


def test_random_resized_crop_selects_boxes_even_when_scale_needs_relaxing() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = [(0, 0, 10, 10), (20, 20, 90, 90)]
    large_box_count = 0
    small_box_count = 0
    transform = A.ReplayCompose(
        [
            A.RandomResizedCrop(
                size=(32, 32),
                scale=(0.01, 0.1),
                ratio=(1.0, 1.0),
                bbox_preservation="one_box",
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        seed=137,
    )

    for seed in range(100, 200):
        transform.set_random_seed(seed)
        result = transform(image=image, bboxes=bboxes)
        crop_coords = result["replay"]["transforms"][0]["params"]["params"]["crop_coords"]
        crop_width = crop_coords[2] - crop_coords[0]
        if crop_width >= 60:
            large_box_count += 1
        else:
            assert crop_width <= 32
            small_box_count += 1

    assert 40 <= large_box_count <= 60
    assert 40 <= small_box_count <= 60


def test_random_resized_crop_preserves_boxes_from_all_bbox_fields() -> None:
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    bboxes = [(10, 10, 50, 40)]
    auxiliary_bboxes = [(95, 70, 135, 110)]
    transform = A.ReplayCompose(
        [
            A.RandomResizedCrop(
                size=(48, 48),
                scale=(0.08, 0.2),
                ratio=(1.25, 1.25),
                bbox_preservation="all_boxes",
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        additional_targets={"auxiliary_bboxes": "bboxes"},
        seed=137,
    )

    result = transform(image=image, bboxes=bboxes, auxiliary_bboxes=auxiliary_bboxes)
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = result["replay"]["transforms"][0]["params"]["params"][
        "crop_coords"
    ]

    assert crop_x_min <= 10 and crop_x_max >= 135
    assert crop_y_min <= 10 and crop_y_max >= 110
    assert result["bboxes"].shape[0] == 1
    assert result["auxiliary_bboxes"].shape[0] == 1
    assert (crop_x_max - crop_x_min) / (crop_y_max - crop_y_min) == pytest.approx(1.25)


def test_random_resized_crop_accepts_an_alias_without_main_bbox_key() -> None:
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    alias_bbox = [(20, 25, 80, 85)]
    transform = A.Compose(
        [A.RandomResizedCrop(size=(48, 48), bbox_preservation="all_boxes", p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        additional_targets={"teacher_bboxes": "bboxes"},
        seed=137,
        strict=True,
    )

    result = transform(image=image, teacher_bboxes=alias_bbox)

    assert len(result["teacher_bboxes"]) == 1


def test_random_resized_crop_preserves_boxes_across_bound_frames() -> None:
    image = np.zeros((2, 100, 120, 3), dtype=np.uint8)
    frame_annotations = [
        {"bboxes": [(5, 5, 35, 35)], "labels": [0]},
        {"bboxes": [(80, 60, 115, 95)], "labels": [1]},
    ]
    transform = A.ReplayCompose(
        [
            A.RandomResizedCrop(
                size=(32, 32),
                scale=(0.08, 0.2),
                ratio=(1.0, 1.0),
                bbox_preservation="all_boxes",
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        frame_binding=["images", "frame_annotations"],
        seed=137,
    )

    result = transform(images=image, frame_annotations=frame_annotations)
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = result["replay"]["transforms"][0]["params"]["params"][
        "crop_coords"
    ]

    assert result["images"].shape == (2, 32, 32, 3)
    assert result["frame_annotations"][0]["labels"] == [0]
    assert result["frame_annotations"][1]["labels"] == [1]
    assert crop_x_min <= 5 and crop_x_max >= 115
    assert crop_y_min <= 5 and crop_y_max >= 95


def test_random_resized_crop_preserves_visible_part_of_out_of_frame_hbb() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    transform = A.ReplayCompose(
        [
            A.RandomResizedCrop(
                size=(32, 32),
                scale=(0.01, 0.1),
                ratio=(1.0, 1.0),
                bbox_preservation="all_boxes",
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            label_fields=["labels"],
            clip_after_transform=False,
        ),
        seed=137,
    )

    result = transform(image=image, bboxes=[(-20, 30, 40, 70)], labels=[137])
    crop_coords = result["replay"]["transforms"][0]["params"]["params"]["crop_coords"]

    assert crop_coords == (0, 30, 40, 70)
    assert result["labels"] == [137]


def test_random_resized_crop_uses_ordinary_path_for_boxes_outside_frame() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    ordinary = A.Compose(
        [A.RandomResizedCrop(size=(32, 48), p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", clip_after_transform=False),
        seed=137,
    )
    configured = A.Compose(
        [A.RandomResizedCrop(size=(32, 48), bbox_preservation="all_boxes", p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", clip_after_transform=False),
        seed=137,
    )
    bboxes = [(150, 150, 160, 160)]

    ordinary_result = ordinary(image=image, bboxes=bboxes)
    configured_result = configured(image=image, bboxes=bboxes)

    np.testing.assert_array_equal(configured_result["image"], ordinary_result["image"])


def test_random_resized_crop_keeps_bbox_params_min_area_filter() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    transform = A.Compose(
        [A.RandomResizedCrop(size=(32, 32), bbox_preservation="all_boxes", p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"], min_area=2048),
        seed=137,
    )

    result = transform(image=image, bboxes=[(20, 20, 40, 40)], labels=[137])

    assert len(result["bboxes"]) == 0
    assert result["labels"] == []


@pytest.mark.parametrize("bbox_preservation", ("one_box", "all_boxes"))
def test_random_resized_crop_uses_ordinary_path_without_boxes(
    bbox_preservation: Literal["one_box", "all_boxes"],
) -> None:
    image = np.arange(120 * 160, dtype=np.uint8).reshape(120, 160, 1)
    ordinary = A.Compose([A.RandomResizedCrop(size=(48, 64), p=1.0)], seed=137, strict=True)
    configured = A.Compose(
        [A.RandomResizedCrop(size=(48, 64), bbox_preservation=bbox_preservation, p=1.0)],
        seed=137,
        strict=True,
    )

    np.testing.assert_array_equal(configured(image=image)["image"], ordinary(image=image)["image"])

    bbox_ordinary = A.Compose(
        [A.RandomResizedCrop(size=(48, 64), p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        seed=137,
        strict=True,
    )
    bbox_configured = A.Compose(
        [A.RandomResizedCrop(size=(48, 64), bbox_preservation=bbox_preservation, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        seed=137,
        strict=True,
    )
    empty = np.empty((0, 4), dtype=np.float32)
    np.testing.assert_array_equal(
        bbox_configured(image=image, bboxes=empty)["image"],
        bbox_ordinary(image=image, bboxes=empty)["image"],
    )


def test_random_resized_crop_keeps_the_best_ratio_when_scale_is_too_small() -> None:
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    transform = A.ReplayCompose(
        [
            A.RandomResizedCrop(
                size=(128, 128),
                scale=(0.08, 0.2),
                ratio=(1.0, 1.0),
                bbox_preservation="one_box",
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        seed=137,
    )

    result = transform(image=image, bboxes=[(300, 200, 900, 600)], labels=[137])
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = result["replay"]["transforms"][0]["params"]["params"][
        "crop_coords"
    ]

    assert crop_x_max - crop_x_min == 600
    assert crop_y_max - crop_y_min == 600
    assert crop_x_min <= 300 and crop_x_max >= 900
    assert crop_y_min <= 200 and crop_y_max >= 600


@pytest.mark.parametrize("bbox_preservation", ("one_box", "all_boxes"))
def test_random_resized_crop_replay_reuses_sampled_bbox_crop(
    bbox_preservation: Literal["one_box", "all_boxes"],
) -> None:
    image = np.arange(128 * 160, dtype=np.uint8).reshape(128, 160, 1)
    bboxes = [(10, 15, 45, 55), (100, 65, 150, 115)]
    labels = [11, 29]
    transform = A.ReplayCompose(
        [
            A.RandomResizedCrop(
                size=(48, 64),
                bbox_preservation=bbox_preservation,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        seed=137,
    )

    result = transform(image=image, bboxes=bboxes, labels=labels)
    replayed = A.ReplayCompose.replay(result["replay"], image=image, bboxes=bboxes, labels=labels)

    np.testing.assert_array_equal(replayed["image"], result["image"])
    np.testing.assert_array_equal(replayed["bboxes"], result["bboxes"])
    assert replayed["labels"] == result["labels"]


@pytest.mark.parametrize(
    "ratio",
    ((0.0, 1.0), (float("inf"), 1.0), (float("nan"), 1.0)),
)
def test_random_resized_crop_rejects_nonpositive_or_nonfinite_ratios(ratio: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        A.RandomResizedCrop(size=(32, 32), ratio=ratio)


def test_random_resized_crop_rejects_unknown_bbox_preservation_at_init() -> None:
    with pytest.raises(ValueError):
        A.RandomResizedCrop(size=(32, 32), bbox_preservation=cast("Any", "largest"))


def test_random_resized_crop_uniform_scale_replays_exactly() -> None:
    image = np.arange(120 * 240, dtype=np.uint8).reshape(120, 240, 1)
    transform = A.ReplayCompose(
        [A.RandomResizedCrop(size=(32, 48), sampling_method="uniform_scale", p=1.0)],
        seed=137,
    )

    result = transform(image=image)
    replayed = A.ReplayCompose.replay(result["replay"], image=image)

    np.testing.assert_array_equal(replayed["image"], result["image"])


def test_random_resized_crop_applied_config_replaces_policy_with_realized_shape() -> None:
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    pipeline = A.Compose(
        [A.RandomResizedCrop(size=(32, 48), sampling_method="standard", p=1.0)],
        save_applied_params=True,
        seed=137,
        strict=True,
    )

    result = pipeline(image=image)
    _, applied_config = result["applied_transforms"][0]

    assert applied_config["sampling_method"] == "standard"
    assert applied_config["scale"][0] == applied_config["scale"][1]
    assert applied_config["ratio"][0] == applied_config["ratio"][1]
    assert math.isfinite(applied_config["scale"][0])
