import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import _functional_images
from tests.helpers import obb_corners_equivalent


@pytest.mark.parametrize(
    ("transform", "resized_shape", "output_shape", "offset"),
    [
        pytest.param(A.LongestMaxSize(max_size=4), (3, 4), (3, 4), (0, 0), id="longest"),
        pytest.param(A.SmallestMaxSize(max_size=3), (3, 5), (3, 5), (0, 0), id="smallest"),
        pytest.param(A.LetterBox(size=(5, 4), p=1), (3, 4), (5, 4), (0, 1), id="letterbox-with-padding"),
    ],
)
def test_rounded_resize_obb_contains_transformed_corners(transform, resized_shape, output_shape, offset):
    height, width = 7, 11
    image = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    bbox = (5.5, 3.5, 4.0, 1.5, 45.0)
    pipeline = A.Compose(
        [transform],
        bbox_params=A.BboxParams(coord_format="cxcywh", bbox_type="obb", label_fields=["labels"]),
    )

    result = pipeline(image=image, bboxes=[bbox], labels=[137])

    assert result["image"].shape == output_shape
    assert result["labels"] == [137]
    resized_height, resized_width = resized_shape
    source_corners = cv2.boxPoints((bbox[:2], bbox[2:4], bbox[4]))
    expected_corners = source_corners * (resized_width / width, resized_height / height) + offset
    center_x, center_y, box_width, box_height, angle = result["bboxes"][0]
    actual_corners = cv2.boxPoints(((center_x, center_y), (box_width, box_height), angle))
    for x, y in expected_corners:
        assert cv2.pointPolygonTest(actual_corners, (x, y), measureDist=True) >= -1e-5


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        pytest.param(A.CropAndPad(px=(0, 0, 0, -2), p=1), (0.1, 0.25, 0.7, 0.75), id="crop-and-pad"),
        pytest.param(A.RandomSizedCrop((8, 8), (4, 6), w2h_ratio=1.5, p=1), (0.25, 0.25, 0.75, 0.75), id="sized"),
        pytest.param(
            A.RandomResizedCrop((4, 6), scale=(1, 1), ratio=(1.5, 1.5), p=1),
            (0.25, 0.25, 0.75, 0.75),
            id="resized",
        ),
        pytest.param(A.LetterBox((8, 16), p=1), (5 / 16, 0.25, 11 / 16, 0.75), id="letterbox"),
    ],
)
def test_direct_crop_and_resize_keep_normalized_hbb_support(transform, expected):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    bboxes = np.array([[0.25, 0.25, 0.75, 0.75, 137]], dtype=np.float32)

    result = transform(image=image, bboxes=bboxes)

    np.testing.assert_allclose(result["bboxes"], [[*expected, 137]], atol=1e-6, rtol=0)


def test_direct_flip_maps_normalized_hbb_alias():
    transform = A.HorizontalFlip(p=1)
    transform.add_targets({"boxes": "bboxes"})
    boxes = np.array([[0.125, 0.25, 0.5, 0.75, 137]], dtype=np.float32)

    result = transform(image=np.zeros((8, 12, 3), dtype=np.uint8), boxes=boxes)

    np.testing.assert_array_equal(result["boxes"], [[0.5, 0.25, 0.875, 0.75, 137]])


def test_legacy_crop_and_pad_replays_normalized_hbb_without_bbox_params():
    transform = A.ReplayCompose([A.CropAndPad(px=(0, 0, 0, -2), p=1)])
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    boxes = np.array([[0.25, 0.25, 0.75, 0.75, 137]], dtype=np.float32)
    result = transform(image=image, bboxes=boxes)
    replay = result["replay"]
    del replay["transforms"][0]["params"]["params"]["bbox_type"]

    replayed = A.ReplayCompose.replay(replay, image=image, bboxes=boxes)

    np.testing.assert_array_equal(replayed["bboxes"], result["bboxes"])


@pytest.mark.parametrize(
    ("transform", "input_shape", "output_shape"),
    [
        pytest.param(A.CropAndPad(px=(-4, -27, 8, 3), keep_size=True, p=1), (60, 100), (60, 100), id="crop-and-pad"),
        pytest.param(A.RandomSizedCrop(min_max_height=(36, 36), size=(36, 76), p=1), (48, 54), (36, 76), id="sized"),
        pytest.param(
            A.RandomResizedCrop(size=(36, 76), scale=(1, 1), ratio=(1, 1), p=1),
            (36, 36),
            (36, 76),
            id="resized",
        ),
        pytest.param(A.RandomSizedBBoxSafeCrop(36, 76, p=1), (48, 54), (36, 76), id="bbox-safe-sized"),
    ],
)
def test_crop_resize_refits_obb_in_output_pixels(transform, input_shape, output_shape):
    height, width = input_shape
    image = np.arange(height * width, dtype=np.uint8).reshape(input_shape)
    bbox = (width / 2, height / 2, 12.0, 6.0, 45.0)
    pipeline = A.ReplayCompose(
        [transform],
        bbox_params=A.BboxParams(coord_format="cxcywh", bbox_type="obb", label_fields=["labels"]),
        seed=137,
    )

    result = pipeline(image=image, bboxes=[[*bbox, 23]], labels=[137])

    assert result["image"].shape == output_shape
    assert result["labels"] == [137]
    recorded = result["replay"]["transforms"][0]["params"]["params"]
    if isinstance(transform, A.CropAndPad):
        offset = (3, -4)
        intermediate_height, intermediate_width = 64, 76
    else:
        x_min, y_min, x_max, y_max = recorded["crop_coords"]
        offset = (-x_min, -y_min)
        intermediate_height, intermediate_width = y_max - y_min, x_max - x_min
    output_height, output_width = output_shape
    scales = (output_width / intermediate_width, output_height / intermediate_height)
    assert scales[0] != scales[1]
    corners = cv2.boxPoints((bbox[:2], bbox[2:4], bbox[4]))
    expected = cv2.boxPoints(cv2.minAreaRect(((corners + offset) * scales).astype(np.float32)))
    center_x, center_y, box_width, box_height, angle, metadata = result["bboxes"][0]
    actual = cv2.boxPoints(((center_x, center_y), (box_width, box_height), angle))
    assert obb_corners_equivalent(actual, expected, atol=1e-4)
    assert metadata == 23


@pytest.mark.parametrize("keep_size", [False, True])
@pytest.mark.parametrize("fit_output", [False, True])
def test_perspective_refits_obb_in_pixels_on_rectangular_frame(keep_size, fit_output):
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    bbox = (40.0, 30.0, 14.0, 6.0, 32.0)
    pipeline = A.ReplayCompose(
        [A.Perspective(scale=(0.12, 0.12), keep_size=keep_size, fit_output=fit_output, p=1)],
        bbox_params=A.BboxParams(coord_format="cxcywh", bbox_type="obb", label_fields=["labels"]),
        seed=137,
    )

    result = pipeline(image=image, bboxes=[[*bbox, 23]], labels=[137])

    recorded = result["replay"]["transforms"][0]["params"]["params"]
    height, width = result["image"].shape[:2]
    assert (height, width) == (image.shape[:2] if keep_size else (recorded["max_height"], recorded["max_width"]))
    corners = cv2.boxPoints((bbox[:2], bbox[2:4], bbox[4]))
    mapped = cv2.perspectiveTransform(corners[:, None, :], recorded["matrix"])[:, 0, :]
    mapped *= (width / recorded["max_width"], height / recorded["max_height"])
    expected = cv2.boxPoints(cv2.minAreaRect(mapped))
    center_x, center_y, box_width, box_height, angle, metadata = result["bboxes"][0]
    actual = cv2.boxPoints(((center_x, center_y), (box_width, box_height), angle))
    assert obb_corners_equivalent(actual, expected, atol=1e-4)
    assert result["labels"] == [137]
    assert metadata == 23


@pytest.mark.parametrize(
    "recorded_shape",
    [None, (60, 100, 3), (54, 84, 3)],
    ids=["sampled", "legacy-replay", "legacy-replay-new-shape"],
)
def test_perspective_keep_size_shares_output_matrix(mocker, recorded_shape):
    image = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
    data = {
        "image": image,
        "images": np.stack((image, image)),
        "mask": image[..., 0].copy(),
        "bboxes": [(20, 10, 42, 27)],
        "keypoints": [(20, 10)],
    }
    pipeline = A.ReplayCompose(
        [A.Perspective(scale=(0.12, 0.12), keep_size=True, interpolation=cv2.INTER_NEAREST, p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        seed=137,
    )
    warp = mocker.spy(_functional_images, "warp_perspective")
    project = mocker.spy(cv2, "perspectiveTransform")
    if recorded_shape is not None:
        matrix = np.array([[1.08, 0.1, 1.5], [0.03, 1.12, 0.9], [0.001, -0.001, 1]])
        recorded = {
            "matrix": matrix,
            "matrix_bbox": matrix.copy(),
            "max_height": 54,
            "max_width": 84,
            "shape": recorded_shape,
            "bbox_type": "hbb",
        }
        record = A.to_dict(pipeline)["transform"]
        record.update(applied=True, params=None)
        record["transforms"][0].update(
            applied=True,
            params={"parameter_schema": 3, "target_schema": None, "target_params": [], "params": recorded},
        )
        sample = mocker.spy(A.Perspective, "sample_parameters")
        result = A.ReplayCompose.replay(record, **data)
        sample.assert_not_called()
    else:
        result = pipeline(**data)
        recorded = result["replay"]["transforms"][0]["params"]["params"]

    assert result["image"].shape == image.shape
    np.testing.assert_array_equal(result["images"], np.stack((result["image"], result["image"])))
    matrices = [call.args[1] for call in (*warp.call_args_list, *project.call_args_list)]
    assert len(matrices) == 6
    expected = np.diag([100 / recorded["max_width"], 60 / recorded["max_height"], 1]) @ recorded["matrix"]
    np.testing.assert_array_equal(matrices[0], expected)
    assert all(matrix is matrices[0] for matrix in matrices)


@pytest.mark.parametrize(
    ("transform", "output_shape"),
    [
        pytest.param(A.RandomScale(scale_range=(-0.5, -0.5), p=1), (4, 6), id="random-scale"),
        pytest.param(A.LongestMaxSize(max_size=4), (3, 5), id="longest"),
    ],
)
@pytest.mark.parametrize("legacy_record", [False, True], ids=["current-record", "scale-only-record"])
def test_resize_replay_maps_keypoints_to_current_frame(transform, output_shape, legacy_record):
    pipeline = A.ReplayCompose([transform], keypoint_params=A.KeypointParams(coord_format="xy"))
    keypoints = [(4.0, 2.0)]
    replay = pipeline(image=np.zeros((7, 11), dtype=np.uint8), keypoints=keypoints)["replay"]
    if legacy_record:
        replay["transforms"][0]["params"]["params"].pop("output_shape")
    image = np.zeros((9, 13), dtype=np.uint8)

    result = A.ReplayCompose.replay(replay, image=image, keypoints=keypoints)

    assert result["image"].shape == output_shape
    input_height, input_width = image.shape
    output_height, output_width = output_shape
    x, y = keypoints[0]
    expected = [[(x + 0.5) * output_width / input_width - 0.5, (y + 0.5) * output_height / input_height - 0.5]]
    np.testing.assert_allclose(result["keypoints"], expected, atol=1e-6)


@pytest.mark.parametrize(
    ("transform", "pad_top", "pad_bottom"),
    [
        pytest.param(A.Resize(9, 24, interpolation=cv2.INTER_NEAREST), 0, 0, id="resize"),
        pytest.param(
            A.LetterBox((28, 24), interpolation=cv2.INTER_NEAREST, fill=11, fill_mask=5),
            9,
            10,
            id="letterbox",
        ),
    ],
)
@pytest.mark.parametrize("target", ["bboxes", "keypoints"])
def test_fixed_resize_replay_preserves_output_geometry_and_maps_current_annotations(
    transform, pad_top, pad_bottom, target
):
    pipeline = A.ReplayCompose(
        [transform],
        bbox_params=A.BboxParams(coord_format="cxcywh", bbox_type="obb", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
    )
    recorded = pipeline(
        image=np.zeros((12, 32), dtype=np.uint8),
        bboxes=[(16, 6, 5, 3, 24)],
        labels=[137],
        keypoints=[(16, 6, 17, 2)],
    )["replay"]
    image = np.arange(24 * 18, dtype=np.uint8).reshape(24, 18)
    mask = image % 4
    bbox = (9.0, 12.0, 6.0, 4.0, 37.0)
    x, y, angle, scale = (9.25, 12.5, 17, 2)

    result = A.ReplayCompose.replay(
        recorded,
        image=image,
        mask=mask,
        bboxes=[[*bbox, 23]],
        labels=[137],
        keypoints=[(x, y, angle, scale)],
    )

    expected_image = cv2.resize(image, (24, 9), interpolation=cv2.INTER_NEAREST)
    expected_mask = cv2.resize(mask, (24, 9), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(
        result["image"], np.pad(expected_image, ((pad_top, pad_bottom), (0, 0)), constant_values=11)
    )
    np.testing.assert_array_equal(
        result["mask"], np.pad(expected_mask, ((pad_top, pad_bottom), (0, 0)), constant_values=5)
    )

    scale_x, scale_y = 24 / 18, 9 / 24
    if target == "keypoints":
        expected = [[(x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5 + pad_top, angle, scale * scale_x]]
        np.testing.assert_allclose(result["keypoints"], expected, atol=1e-5)
    else:
        corners = cv2.boxPoints((bbox[:2], bbox[2:4], bbox[4]))
        mapped = corners * (scale_x, scale_y) + (0, pad_top)
        expected = cv2.boxPoints(cv2.minAreaRect(mapped.astype(np.float32)))
        cx, cy, width, height, output_angle, metadata = result["bboxes"][0]
        actual = cv2.boxPoints(((cx, cy), (width, height), output_angle))
        assert obb_corners_equivalent(actual, expected, atol=1e-4)
        assert result["labels"] == [137]
        assert metadata == 23


def test_identity_remap_preserves_fractional_obb():
    image = np.arange(72 * 128, dtype=np.uint8).reshape(72, 128)
    bboxes = np.array([[0.23, 0.27, 0.62, 0.43, 111.75, 23]], dtype=np.float32)
    pipeline = A.Compose(
        [A.OpticalDistortion(distort_range=(0, 0), p=1)],
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb", label_fields=["labels"]),
    )

    result = pipeline(image=image, bboxes=bboxes, labels=[137])

    np.testing.assert_array_equal(result["image"], image)
    np.testing.assert_array_equal(result["bboxes"], bboxes)
    assert result["labels"] == [137]


def test_remap_refits_obb_to_warped_rotated_support():
    image = np.zeros((72, 128), dtype=np.uint8)
    mask = np.zeros_like(image)
    bbox = (22.0, 19.0, 18.0, 7.0, 33.0)
    corners = cv2.boxPoints((bbox[:2], bbox[2:4], bbox[4]))
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1)
    pipeline = A.Compose(
        [A.OpticalDistortion(distort_range=(0.25, 0.25), p=1)],
        bbox_params=A.BboxParams(coord_format="cxcywh", bbox_type="obb", label_fields=["labels"]),
        seed=137,
    )

    result = pipeline(image=image, mask=mask, bboxes=[[*bbox, 23]], labels=[137])

    assert not np.array_equal(result["mask"], mask)
    expected = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(result["mask"])))
    cx, cy, width, height, angle, metadata = result["bboxes"][0]
    actual = cv2.boxPoints(((cx, cy), (width, height), angle))
    assert obb_corners_equivalent(actual, expected, atol=1e-4)
    assert result["labels"] == [137]
    assert metadata == 23


@pytest.mark.parametrize(
    ("transform", "output_shape"),
    [
        pytest.param(A.RandomScale(scale_range=(-0.5, -0.5), p=1), (4, 6), id="random-scale"),
        pytest.param(A.LongestMaxSize(max_size=4), (3, 4), id="longest"),
    ],
)
def test_resize_uses_nonempty_target_shape(transform, output_shape):
    pipeline = A.Compose([transform], is_check_shapes=False)
    images = np.empty((0, 7, 11, 3), dtype=np.uint8)
    mask = np.zeros((9, 13), dtype=np.uint8)

    result = pipeline(images=images, mask=mask)

    assert result["images"].shape == images.shape
    assert result["mask"].shape == output_shape


@pytest.mark.parametrize("legacy_replay", [False, True], ids=["sampled", "legacy-replay"])
def test_grid_elastic_shares_cell_matrices_across_targets(mocker, legacy_replay):
    image = np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)
    mask = np.zeros_like(image)
    mask[2:9, 3:12] = 3
    data = {
        "image": image,
        "images": np.stack((image, image)),
        "mask": mask,
        "bboxes": [(3, 2, 12, 9)],
        "labels": [137],
        "keypoints": [(4, 3), (8, 3), (8, 6), (12, 9)],
    }
    pipeline = A.ReplayCompose(
        [A.GridElasticDeform((2, 2), 2, interpolation=cv2.INTER_NEAREST)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        seed=137,
    )
    fit_matrix = mocker.spy(cv2, "getPerspectiveTransform")

    if legacy_replay:
        record = A.to_dict(pipeline)["transform"]
        record.update(applied=True, params=None)
        record["transforms"][0].update(
            applied=True,
            params={
                "parameter_schema": 3,
                "target_schema": None,
                "target_params": [],
                "params": {
                    "generated_mesh": np.array(
                        [
                            [0, 0, 8, 6, 0, 0, 8, 0, 7, 5, 0, 6],
                            [8, 0, 16, 6, 8, 0, 16, 0, 16, 6, 7, 5],
                            [0, 6, 8, 12, 0, 6, 7, 5, 8, 12, 0, 12],
                            [8, 6, 16, 12, 7, 5, 16, 6, 16, 12, 8, 12],
                        ],
                        dtype=np.float32,
                    ),
                    "shape": (12, 16, 1),
                    "interpolation": cv2.INTER_NEAREST,
                    "bbox_type": "hbb",
                },
            },
        )
        sample = mocker.spy(A.GridElasticDeform, "sample_parameters")
        result = A.ReplayCompose.replay(record, **data)
        sample.assert_not_called()
    else:
        result = pipeline(**data)
        saved_params = result["replay"]["transforms"][0]["params"]["params"]
        assert saved_params["generated_mesh"].shape == (4, 12)

    np.testing.assert_array_equal(result["images"], np.stack((result["image"], result["image"])))
    assert result["image"][6, 7] == 120
    assert set(np.unique(result["mask"])) == {0, 3}
    assert cv2.boundingRect(result["mask"]) == (2, 2, 10, 7)
    np.testing.assert_array_equal(result["bboxes"], [[2, 2, 12, 9]])
    assert result["labels"] == [137]
    np.testing.assert_allclose(
        result["keypoints"],
        [[4.097561, 2.9268293], [7.529412, 2.3529413], [7, 5], [12.072727, 8.945455]],
        atol=1e-6,
    )
    assert fit_matrix.call_count == 4


def test_grid_elastic_warps_bbox_masks_on_wide_images():
    image = np.zeros((16, 640, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[2:6, 32:64] = 1
    transform = A.Compose(
        [A.GridElasticDeform((2, 2), 2)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        seed=137,
    )

    result = transform(image=image, mask=mask, bboxes=[(32, 2, 64, 6)])

    assert result["mask"].shape == mask.shape
    x, y, width, height = cv2.boundingRect(result["mask"])
    np.testing.assert_allclose(result["bboxes"], [[x, y, x + width, y + height]], atol=1e-5)


@pytest.mark.parametrize(
    ("px", "pixel", "pad_width"),
    [
        pytest.param((-3, -4, -3, -4), (3, 4), ((0, 0), (0, 0)), id="single-pixel"),
        pytest.param(-64, (3, 4), ((0, 0), (0, 0)), id="symmetric-overcrop"),
        pytest.param((-64, -64, 3, 2), (6, 0), ((0, 3), (2, 0)), id="one-sided-overcrop-with-pad"),
    ],
)
def test_crop_and_pad_to_single_pixel_maps_annotations(px, pixel, pad_width):
    image = np.arange(7 * 9, dtype=np.uint8).reshape(7, 9)
    mask = image % 4
    pipeline = A.ReplayCompose(
        [A.CropAndPad(px=px, keep_size=False, fill=37, fill_mask=7, p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
    )
    y, x = pixel
    data = {"image": image, "mask": mask, "bboxes": [(0, 0, 9, 7)], "keypoints": [(x, y)]}

    result = pipeline(**data)

    np.testing.assert_array_equal(result["image"], np.pad(image[y : y + 1, x : x + 1], pad_width, constant_values=37))
    np.testing.assert_array_equal(result["mask"], np.pad(mask[y : y + 1, x : x + 1], pad_width, constant_values=7))
    pad_top, pad_left = pad_width[0][0], pad_width[1][0]
    np.testing.assert_allclose(result["bboxes"], [[pad_left, pad_top, pad_left + 1, pad_top + 1]], atol=1e-6)
    np.testing.assert_array_equal(result["keypoints"], [[pad_left, pad_top]])
    replayed = A.ReplayCompose.replay(result["replay"], **data)
    for target in data:
        np.testing.assert_array_equal(replayed[target], result[target])


def test_large_asymmetric_padding_preserves_fill_and_annotation_offsets():
    image = np.arange(5 * 8, dtype=np.uint8).reshape(5, 8)
    mask = image % 4
    pipeline = A.Compose(
        [A.Pad(padding=(137, 3, 11, 79), fill=37, fill_mask=7, p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
    )

    result = pipeline(image=image, mask=mask, bboxes=[(1, 1, 4, 4)], keypoints=[(1.25, 2.5)])

    pad_width = ((3, 79), (137, 11))
    np.testing.assert_array_equal(result["image"], np.pad(image, pad_width, constant_values=37))
    np.testing.assert_array_equal(result["mask"], np.pad(mask, pad_width, constant_values=7))
    np.testing.assert_allclose(result["bboxes"], [[138, 4, 141, 7]], atol=1e-5)
    np.testing.assert_array_equal(result["keypoints"], [[138.25, 5.5]])


def test_affine_near_zero_axis_scale_preserves_subpixel_annotations():
    image = np.arange(5 * 9, dtype=np.uint8).reshape(5, 9)
    mask = image % 4
    scale_x = 1 / 256
    pipeline = A.Compose(
        [
            A.Affine(
                scale={"x": (scale_x, scale_x), "y": (1, 1)},
                keep_ratio=False,
                interpolation=cv2.INTER_NEAREST,
                fill=37,
                fill_mask=7,
                p=1,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
    )

    result = pipeline(image=image, mask=mask, bboxes=[(0, 0, 9, 5)], keypoints=[(3.25, 1.25)])

    expected_image = np.full_like(image, 37)
    expected_mask = np.full_like(mask, 7)
    expected_image[:, 4] = image[:, 4]
    expected_mask[:, 4] = mask[:, 4]
    np.testing.assert_array_equal(result["image"], expected_image)
    np.testing.assert_array_equal(result["mask"], expected_mask)
    np.testing.assert_allclose(result["bboxes"], [[4.5 - 4.5 * scale_x, 0, 4.5 + 4.5 * scale_x, 5]], atol=1e-6)
    np.testing.assert_allclose(result["keypoints"], [[4 - 0.75 * scale_x, 1.25]], atol=1e-6)


@pytest.mark.parametrize(("keep_size", "output_shape"), [(True, (1, 3)), (False, (2, 2))])
def test_strong_perspective_on_single_row_frame_keeps_targets_aligned(keep_size, output_shape):
    mask = np.array([[1, 2, 3]], dtype=np.uint8)
    image = mask * 17
    pipeline = A.Compose(
        [
            A.Perspective(
                scale=(2, 2),
                keep_size=keep_size,
                interpolation=cv2.INTER_NEAREST,
                fill=119,
                fill_mask=7,
                p=1,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        seed=137,
    )

    result = pipeline(image=image, mask=mask, bboxes=[(0, 0, 3, 1)], labels=[137], keypoints=[(0.5, 0.25)])

    assert result["image"].shape == result["mask"].shape == output_shape
    assert result["image"].dtype == result["mask"].dtype == np.uint8
    assert set(np.unique(result["mask"])) <= {1, 2, 3, 7}
    assert np.any(result["mask"] != 7)
    np.testing.assert_array_equal(result["image"], result["mask"] * 17)
    height, width = output_shape
    boxes = result["bboxes"]
    assert boxes.shape == (1, 4)
    assert ((boxes >= 0) & (boxes <= [width, height, width, height])).all()
    assert (boxes[:, :2] < boxes[:, 2:]).all()
    assert result["labels"] == [137]
    points = result["keypoints"]
    assert points.shape == (1, 2)
    assert ((points >= -0.5) & (points <= [width - 0.5, height - 0.5])).all()


@pytest.mark.parametrize("bbox_type", ["hbb", "obb"])
@pytest.mark.parametrize("check_each_transform", [False, True])
def test_crop_removes_all_annotations_and_label_fields(bbox_type, check_each_transform):
    image = np.zeros((9, 13), dtype=np.uint8)
    boxes = np.array([[1, 1, 3, 3], [3, 2, 5, 4]], dtype=np.float32)
    if bbox_type == "obb":
        boxes = np.column_stack((boxes, np.array([27, -33], dtype=np.float32)))
    pipeline = A.Compose(
        [A.Crop(x_min=8, y_min=5, x_max=13, y_max=9, p=1)],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            bbox_type=bbox_type,
            label_fields=["box_labels", "box_scores"],
            check_each_transform=check_each_transform,
        ),
        keypoint_params=A.KeypointParams(
            coord_format="xy",
            label_fields=["point_labels"],
            label_mapping={},
            check_each_transform=check_each_transform,
        ),
    )

    result = pipeline(
        image=image,
        bboxes=boxes,
        box_labels=["left", "right"],
        box_scores=np.array([0.25, 0.75], dtype=np.float32),
        keypoints=np.array([[2, 2], [4, 3]], dtype=np.float32),
        point_labels=[137, 139],
    )

    assert result["bboxes"].shape == (0, boxes.shape[1])
    assert result["keypoints"].shape == (0, 2)
    assert result["box_labels"] == []
    assert len(result["box_scores"]) == 0
    assert result["point_labels"] == []
