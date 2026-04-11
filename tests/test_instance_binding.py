from typing import Any

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

    def test_mask_repack_uses_original_instance_index(self) -> None:
        """When instance 0 is bbox-filtered but instance 1 survives, mask must index stack by old id."""
        transform = A.Compose(
            [A.Crop(x_min=40, y_min=40, x_max=60, y_max=60, p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", min_area=1),
            instance_binding=["masks", "bboxes"],
        )
        image = _make_image()
        instances = [
            {
                "mask": np.zeros((100, 100), dtype=np.uint8),
                "bbox": np.array([10.0, 10.0, 15.0, 15.0], dtype=np.float32),
            },
            {
                "mask": _make_mask(region=(42, 58, 42, 58)),
                "bbox": np.array([45.0, 45.0, 55.0, 55.0], dtype=np.float32),
            },
        ]
        result = transform(image=image, instances=instances)
        assert len(result["instances"]) == 1
        assert result["instances"][0]["mask"].sum() > 50

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

    def test_zero_keypoints_with_label_fields_no_keypoint_labels_required(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["name"]),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(region=(10, 50, 10, 50)),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                "keypoints": np.empty((0, 2), dtype=np.float32),
                "bbox_labels": {"class_id": 1},
            },
        ]

        result = transform(image=image, instances=instances)
        np.testing.assert_array_equal(
            result["instances"][0]["keypoints"],
            np.empty((0, 2), dtype=np.float32),
        )

    def test_empty_instances_obb_bbox_processor_shape(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", bbox_type="obb"),
            instance_binding=["masks", "bboxes"],
        )
        image = _make_image()
        result = transform(image=image, instances=[])
        assert result["instances"] == []


class TestParamsIsolation:
    def test_shared_bbox_params_not_mutated_by_instance_binding(self) -> None:
        shared = A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"])
        A.Compose(
            [A.NoOp(p=1)],
            bbox_params=shared,
            instance_binding=["masks", "bboxes"],
        )
        assert shared.label_fields == ["class_id"]

    def test_shared_keypoint_params_not_mutated_by_instance_binding(self) -> None:
        shared = A.KeypointParams(coord_format="xy", remove_invisible=True, check_each_transform=True)
        A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=shared,
            instance_binding=["masks", "bboxes", "keypoints"],
        )
        assert shared.remove_invisible is True
        assert shared.check_each_transform is True
        assert shared.label_fields is None


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
    def test_missing_instances_key(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
        )
        image = _make_image()
        with pytest.raises(ValueError, match="`instances` must be provided"):
            transform(image=image)

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

    def test_missing_keypoints_when_bound(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )
        image = _make_image()
        with pytest.raises(ValueError, match="missing required key 'keypoints'"):
            transform(
                image=image,
                instances=[
                    {
                        "mask": _make_mask(),
                        "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                    },
                ],
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

    def test_get_init_params_masks_keypoints_preserves_bbox_params(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["name"]),
            instance_binding=["masks", "keypoints"],
        )

        params = transform._get_init_params()
        assert params["bbox_params"].label_fields == ["class_id"]

    def test_get_init_params_masks_bboxes_preserves_keypoint_params(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(
                coord_format="xy",
                label_fields=["name"],
                remove_invisible=False,
                check_each_transform=False,
            ),
            instance_binding=["masks", "bboxes"],
        )

        params = transform._get_init_params()
        assert params["keypoint_params"].label_fields == ["name"]
        assert params["keypoint_params"].remove_invisible is False
        assert params["keypoint_params"].check_each_transform is False

    def test_get_init_params_keypoints_binding_reflects_runtime_flags(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(
                coord_format="xy",
                remove_invisible=True,
                check_each_transform=True,
            ),
            instance_binding=["masks", "bboxes", "keypoints"],
        )
        params = transform._get_init_params()
        kp = params["keypoint_params"]
        assert kp.remove_invisible is False
        assert kp.check_each_transform is False


class TestNestedComposeInstanceBinding:
    def test_inner_compose_preprocess_after_unpack(self) -> None:
        inner = A.Compose([A.NoOp(p=1)])
        transform = A.Compose(
            [inner],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
        )
        image = _make_image()
        out = transform(
            image=image,
            instances=[
                {
                    "mask": _make_mask(),
                    "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                },
            ],
        )
        assert len(out["instances"]) == 1


class TestInstanceUnpackCollisions:
    def test_rejects_existing_masks_key(self) -> None:
        transform = A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
        )
        image = _make_image()
        existing = np.zeros((1, 100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="would overwrite existing data keys"):
            transform(
                image=image,
                masks=existing,
                instances=[
                    {
                        "mask": _make_mask(),
                        "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                    },
                ],
            )


class TestInstanceBindingCallState:
    def test_state_cleared_when_transform_raises(self) -> None:
        def boom(img: np.ndarray, **kwargs: Any) -> np.ndarray:
            msg = "intentional"
            raise RuntimeError(msg)

        transform = A.Compose(
            [A.NoOp(p=1), A.Lambda(image=boom, p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            keypoint_params=A.KeypointParams(coord_format="xy"),
            instance_binding=["masks", "bboxes", "keypoints"],
        )

        image = _make_image()
        instances = [
            {
                "mask": _make_mask(),
                "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
                "keypoints": np.array([[20.0, 20.0]], dtype=np.float32),
            },
        ]

        with pytest.raises(RuntimeError, match="intentional"):
            transform(image=image, instances=instances)

        assert not getattr(transform, "_repack_after_processors", False)
        assert not hasattr(transform, "_instance_count")


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
