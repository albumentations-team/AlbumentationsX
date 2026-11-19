from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize("distortion", [-0.3, 0.0, 0.3])
def test_fisheye_maps_fractional_keypoints_and_replays(distortion: float) -> None:
    image = np.arange(96 * 128, dtype=np.uint8).reshape(96, 128)
    center = np.array([64.0, 48.0])
    radius_squared = np.sum(center**2)
    # The two output points round to the same pixel but are distinct annotations.
    expected = np.array([[73.15, 42.25], [73.35, 42.45], [27.75, 66.25]])
    offset = expected - center
    source = center + offset * (1 + distortion * np.sum(offset**2, axis=1) / radius_squared)[:, None]
    data = dict(image=image, keypoints=source)
    pipeline = A.ReplayCompose(
        [A.OpticalDistortion(mode="fisheye", distort_range=(distortion, distortion), p=1)],
        keypoint_params=A.KeypointParams(coord_format="xy"),
    )

    result = pipeline(**data)
    replayed = A.ReplayCompose.replay(result["replay"], **data)

    np.testing.assert_allclose(result["keypoints"], expected, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(replayed["keypoints"], result["keypoints"])
    np.testing.assert_array_equal(replayed["image"], result["image"])


@pytest.mark.parametrize(("bbox_type", "angle"), [("hbb", ()), ("obb", (23.0,))])
def test_fisheye_fits_box_edges_between_the_corners(bbox_type: str, angle: tuple[float, ...]) -> None:
    image = np.zeros((96, 128), dtype=np.uint8)
    center = np.array([64.0, 48.0])
    output_half_sizes = np.array([26.0, 19.0])
    distortion = 0.3
    # For a centered box under this contraction, the side midpoints give the widest extent.
    input_half_sizes = output_half_sizes * (1 + distortion * output_half_sizes**2 / np.sum(center**2))
    expected = [*(center - output_half_sizes), *(center + output_half_sizes), *angle]
    source = [*(center - input_half_sizes), *(center + input_half_sizes), *angle]

    pipeline = A.Compose(
        [A.OpticalDistortion(mode="fisheye", distort_range=(distortion, distortion), p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", bbox_type=bbox_type),
    )

    result = pipeline(image=image, bboxes=[source])
    np.testing.assert_allclose(result["bboxes"], [expected], atol=0.3, rtol=0)


@pytest.mark.parametrize(("bbox_type", "angle"), [("hbb", ()), ("obb", (0.0,))])
def test_fisheye_drops_invisible_annotations_and_keeps_labels(bbox_type: str, angle: tuple[float, ...]) -> None:
    image = np.zeros((96, 128), dtype=np.uint8)
    pipeline = A.Compose(
        [A.OpticalDistortion(mode="fisheye", distort_range=(-0.3, -0.3), p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", bbox_type=bbox_type, label_fields=["labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy"),
    )

    result = pipeline(
        image=image,
        bboxes=[(1, 1, 7, 6, *angle), (1, 34, 48, 62, *angle)],
        keypoints=[(3, 4), (64, 48)],
        labels=["outside", "crossing"],
    )

    assert result["labels"] == ["crossing"]
    assert result["bboxes"][0][0] == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_array_equal(result["keypoints"], [[64, 48]])


@pytest.mark.parametrize(
    "settings",
    [
        {"map_resolution_range": (0.75, 0.75)},
        {"border_mode": cv2.BORDER_REFLECT_101},
        {"distort_range": (-0.4, -0.4)},
        {"mode": "camera"},
    ],
)
def test_optical_distortion_uses_configured_keypoint_fallback(
    settings: dict,
    mocker: MockerFixture,
) -> None:
    image = np.zeros((96, 128), dtype=np.uint8)
    transform_args = {"mode": "fisheye", "distort_range": (-0.3, -0.3), **settings}
    transform = A.ReplayCompose(
        [A.OpticalDistortion(**transform_args, p=1)],
        keypoint_params=A.KeypointParams(coord_format="xy"),
    )

    rasterize = mocker.spy(fgeometric, "remap_keypoints_via_mask")
    result = transform(image=image, keypoints=[(64.1, 48.1)])

    rasterize.assert_called_once()
    np.testing.assert_array_equal(result["keypoints"], rasterize.spy_return[:, :2])
