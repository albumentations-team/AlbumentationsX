import numpy as np
import pytest

import albumentations as A


def _make_image(height: int = 100, width: int = 100) -> np.ndarray:
    return np.random.default_rng(137).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _make_mask(height: int = 100, width: int = 100, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if region is not None:
        y1, y2, x1, x2 = region
        mask[y1:y2, x1:x2] = 1
    return mask


class TestInstanceBindingInit:
    def test_valid_binding(self) -> None:
        t = A.Compose(
            [A.HorizontalFlip(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
        )
        assert t._instance_binding == frozenset({"masks", "bboxes"})

    def test_binding_requires_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            A.Compose(
                [A.HorizontalFlip(p=1)],
                bbox_params=A.BboxParams(coord_format="pascal_voc"),
                instance_binding=["bboxes"],
            )

    def test_invalid_target_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid instance_binding"):
            A.Compose(
                [A.HorizontalFlip(p=1)],
                instance_binding=["masks", "invalid_target"],
            )

    def test_mask_and_masks_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="both 'mask' and 'masks'"):
            A.Compose(
                [A.HorizontalFlip(p=1)],
                bbox_params=A.BboxParams(coord_format="pascal_voc"),
                instance_binding=["mask", "masks", "bboxes"],
            )

    def test_bboxes_requires_bbox_params(self) -> None:
        with pytest.raises(ValueError, match="bbox_params must be set"):
            A.Compose(
                [A.HorizontalFlip(p=1)],
                instance_binding=["masks", "bboxes"],
            )

    def test_keypoints_requires_keypoint_params(self) -> None:
        with pytest.raises(ValueError, match="keypoint_params must be set"):
            A.Compose(
                [A.HorizontalFlip(p=1)],
                bbox_params=A.BboxParams(coord_format="pascal_voc"),
                instance_binding=["masks", "bboxes", "keypoints"],
            )

    def test_none_binding(self) -> None:
        t = A.Compose([A.HorizontalFlip(p=1)])
        assert t._instance_binding is None


class TestUnpackRepack:
    def test_basic_roundtrip(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                "keypoints": np.array([[20.0, 20.0], [30.0, 30.0]], dtype=np.float32),
                "bbox_labels": {"class_id": "cat"},
            },
            {
                "mask": _make_mask(region=(60, 90, 60, 90)),
                "bbox": np.array([60, 60, 90, 90], dtype=np.float32),
                "keypoints": np.array([[70.0, 70.0]], dtype=np.float32),
                "bbox_labels": {"class_id": "dog"},
            },
        ]

        result = transform(image=image, instances=instances)
        assert len(result["instances"]) == 2
        assert result["instances"][0]["mask"].shape == (100, 100)
        assert result["instances"][0]["bbox_labels"]["class_id"] == "cat"
        assert result["instances"][1]["bbox_labels"]["class_id"] == "dog"
        assert result["instances"][0]["keypoints"].shape == (2, 2)
        assert result["instances"][1]["keypoints"].shape == (1, 2)

    def test_variable_keypoints_per_instance(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                "keypoints": np.array([[20.0, 20.0]] * 17, dtype=np.float32),
            },
            {
                "mask": _make_mask(region=(60, 90, 60, 90)),
                "bbox": np.array([60, 60, 90, 90], dtype=np.float32),
                "keypoints": np.array([[70.0, 70.0]], dtype=np.float32),
            },
        ]

        result = transform(image=image, instances=instances)
        assert result["instances"][0]["keypoints"].shape == (17, 2)
        assert result["instances"][1]["keypoints"].shape == (1, 2)


class TestBboxFiltering:
    def test_removed_bbox_removes_mask_and_keypoints(self) -> None:
        transform = A.Compose(
            [A.Crop(x_min=0, y_min=0, x_max=55, y_max=55, p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"], min_area=1),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                "keypoints": np.array([[20.0, 20.0], [30.0, 30.0]], dtype=np.float32),
                "bbox_labels": {"class_id": "cat"},
            },
            {
                "mask": _make_mask(region=(60, 90, 60, 90)),
                "bbox": np.array([60, 60, 90, 90], dtype=np.float32),
                "keypoints": np.array([[70.0, 70.0]], dtype=np.float32),
                "bbox_labels": {"class_id": "dog"},
            },
        ]

        result = transform(image=image, instances=instances)
        assert len(result["instances"]) == 1
        assert result["instances"][0]["bbox_labels"]["class_id"] == "cat"
        assert result["instances"][0]["keypoints"].shape[0] == 2

    def test_all_bboxes_removed(self) -> None:
        transform = A.Compose(
            [A.Crop(x_min=40, y_min=40, x_max=60, y_max=60, p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", min_area=100),
            instance_binding=["masks", "bboxes"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 20, 10, 20)),
                "bbox": np.array([10, 10, 20, 20], dtype=np.float32),
            },
        ]

        result = transform(image=image, instances=instances)
        assert result["instances"] == []


class TestEmptyInput:
    def test_empty_instances_list(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
        )

        image = _make_image()
        result = transform(image=image, instances=[])
        assert result["instances"] == []

    def test_instance_with_zero_keypoints(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                "keypoints": np.zeros((0, 2), dtype=np.float32),
            },
        ]

        result = transform(image=image, instances=instances)
        assert len(result["instances"]) == 1
        assert result["instances"][0]["keypoints"].shape == (0, 2)


class TestKeypoints:
    def test_out_of_bounds_keypoints_preserved(self) -> None:
        transform = A.Compose(
            [A.Crop(x_min=20, y_min=20, x_max=80, y_max=80, p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(20, 80, 20, 80)),
                "bbox": np.array([20, 20, 80, 80], dtype=np.float32),
                "keypoints": np.array(
                    [
                        [50.0, 50.0],
                        [5.0, 5.0],
                    ],
                    dtype=np.float32,
                ),
            },
        ]

        result = transform(image=image, instances=instances)
        assert len(result["instances"]) == 1
        assert result["instances"][0]["keypoints"].shape[0] == 2


class TestOverlappingLabelNames:
    def test_same_label_name_bbox_and_keypoint(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class"]),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["class"]),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                "keypoints": np.array([[20.0, 20.0], [30.0, 30.0]], dtype=np.float32),
                "bbox_labels": {"class": "cat"},
                "keypoint_labels": {"class": ["left_eye", "right_eye"]},
            },
        ]

        result = transform(image=image, instances=instances)
        assert result["instances"][0]["bbox_labels"]["class"] == "cat"
        assert result["instances"][0]["keypoint_labels"]["class"] == ["left_eye", "right_eye"]


class TestValidation:
    def test_missing_mask(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
        )

        image = _make_image()
        with pytest.raises(ValueError, match="missing required key 'mask'"):
            transform(
                image=image,
                instances=[{"bbox": np.array([10, 10, 50, 50], dtype=np.float32)}],
            )

    def test_missing_bbox(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
        )

        image = _make_image()
        with pytest.raises(ValueError, match="missing required key 'bbox'"):
            transform(
                image=image,
                instances=[{"mask": _make_mask()}],
            )

    def test_missing_bbox_labels(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
            instance_binding=["masks", "bboxes"],
        )

        image = _make_image()
        with pytest.raises(ValueError, match="missing 'bbox_labels'"):
            transform(
                image=image,
                instances=[
                    {
                        "mask": _make_mask(),
                        "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                    },
                ],
            )

    def test_missing_bbox_label_key(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id", "score"]),
            instance_binding=["masks", "bboxes"],
        )

        image = _make_image()
        with pytest.raises(ValueError, match="missing keys"):
            transform(
                image=image,
                instances=[
                    {
                        "mask": _make_mask(),
                        "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                        "bbox_labels": {"class_id": "cat"},
                    },
                ],
            )

    def test_keypoint_label_length_mismatch(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["name"]),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        with pytest.raises(ValueError, match="values but keypoints has"):
            transform(
                image=image,
                instances=[
                    {
                        "mask": _make_mask(),
                        "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                        "keypoints": np.array([[20.0, 20.0], [30.0, 30.0]], dtype=np.float32),
                        "keypoint_labels": {"name": ["left_eye"]},
                    },
                ],
            )


class TestWithoutBboxes:
    def test_masks_and_keypoints_only(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "keypoints": np.array([[20.0, 20.0], [30.0, 30.0]], dtype=np.float32),
            },
            {
                "mask": _make_mask(region=(60, 90, 60, 90)),
                "keypoints": np.array([[70.0, 70.0]], dtype=np.float32),
            },
        ]

        result = transform(image=image, instances=instances)
        assert len(result["instances"]) == 2


class TestSerialization:
    def test_to_dict_excludes_hidden_fields(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        d = transform.to_dict_private()
        bbox_label_fields = d["bbox_params"]["label_fields"]
        kp_label_fields = d["keypoint_params"]["label_fields"]

        assert "_bbox_instance_id" not in bbox_label_fields
        assert "_ibl_bbox_class_id" not in bbox_label_fields
        assert "_kp_instance_id" not in kp_label_fields
        assert "class_id" in bbox_label_fields
        assert d["instance_binding"] == ["bboxes", "keypoints", "masks"]

    def test_to_dict_omits_binding_when_none(self) -> None:
        transform = A.Compose([A.NoOp(p=1)])
        d = transform.to_dict_private()
        assert "instance_binding" not in d

    def test_get_init_params_clean(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
            instance_binding=["masks", "bboxes"],
        )

        params = transform._get_init_params()
        bbox_params = params["bbox_params"]
        assert "_bbox_instance_id" not in (bbox_params.label_fields or [])
        assert params["instance_binding"] == ["bboxes", "masks"]


class TestChannelMask:
    def test_mask_channel_binding(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["mask", "bboxes"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
            },
            {
                "mask": _make_mask(region=(60, 90, 60, 90)),
                "bbox": np.array([60, 60, 90, 90], dtype=np.float32),
            },
        ]

        result = transform(image=image, instances=instances)
        assert len(result["instances"]) == 2
        assert result["instances"][0]["mask"].shape == (100, 100)
