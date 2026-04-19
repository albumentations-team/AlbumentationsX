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


class TestMixingTransformsInstanceBinding:
    def test_mosaic_instance_binding_masks_one_cell(self) -> None:
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(1, 1),
                    target_size=(64, 64),
                    cell_shape=(64, 64),
                    center_range=(0.5, 0.5),
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        image = _make_image(64, 64)
        m1 = _make_mask(64, 64, (10, 40, 10, 40))
        m2 = _make_mask(64, 64, (45, 60, 45, 60))
        instances = [
            {
                "mask": m1,
                "bbox": np.array([10, 10, 40, 40], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
            {
                "mask": m2,
                "bbox": np.array([45, 45, 60, 60], dtype=np.float32),
                "bbox_labels": {"cid": 2},
            },
        ]
        result = transform(image=image, instances=instances, mosaic_metadata=[])
        assert len(result["instances"]) == 2
        assert result["instances"][0]["mask"].shape == (64, 64)
        assert result["instances"][1]["mask"].shape == (64, 64)

    def test_mosaic_instance_binding_two_sources_two_instances(self) -> None:
        ch, cw = 64, 64
        rng = np.random.default_rng(137)
        img_p = rng.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 36, 4, 36)),
                "bbox": np.array([4, 4, 36, 36], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (8, 44, 8, 44))]),
                "bboxes": np.array([[8.0, 8.0, 44.0, 44.0]], dtype=np.float32),
                "bbox_labels": {"cid": [2]},
            },
        ]
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(2, 1),
                    target_size=(ch * 2, cw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    fit_mode="cover",
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=img_p, instances=instances, mosaic_metadata=mosaic_metadata)
        assert len(result["instances"]) == 2
        assert result["instances"][0]["mask"].shape == (ch * 2, cw)
        assert result["instances"][1]["mask"].shape == (ch * 2, cw)
        cids = {result["instances"][i]["bbox_labels"]["cid"] for i in range(2)}
        assert cids == {1, 2}

    def test_copy_paste_instance_binding_keypoints_survive_by_instance_id(self) -> None:
        """Full-cover paste deterministically occludes all primaries; only the PASTED instance
        survives. The binding contract still has to allocate the pasted ID without IndexError.
        """
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        m0 = _make_mask(80, 80, (5, 30, 5, 30))
        m1 = _make_mask(80, 80, (50, 75, 50, 75))
        instances = [
            {
                "mask": m0,
                "bbox": np.array([5, 5, 30, 30], dtype=np.float32),
                "bbox_labels": {"cid": 1},
                "keypoints": np.array([[10.0, 10.0], [15.0, 15.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [1, 1]},
            },
            {
                "mask": m1,
                "bbox": np.array([50, 50, 75, 75], dtype=np.float32),
                "bbox_labels": {"cid": 2},
                "keypoints": np.array([[60.0, 60.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [2]},
            },
        ]
        # Full-cover donor: deterministic paste over the entire 80x80 target.
        obj: dict[str, Any] = {
            "image": np.full((80, 80, 3), 200, dtype=np.uint8),
            "mask": np.ones((80, 80), dtype=np.uint8),
            "bbox": [0, 0, 80, 80],
            "bbox_labels": {"cid": 99},
            "keypoints": np.array([[20.0, 20.0]], dtype=np.float32),
            "keypoint_labels": {"vis": [9]},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.05, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["vis"]),
            instance_binding=["masks", "bboxes", "keypoints"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[obj])
        assert len(result["instances"]) == 1
        assert result["instances"][0]["bbox_labels"]["cid"] == 99
        assert result["instances"][0]["keypoints"].shape == (1, 2)


# ---------------------------------------------------------------------------
# Mixing transforms × instance binding — matrix + corner cases
# ---------------------------------------------------------------------------


@pytest.fixture
def rng_137() -> np.random.Generator:
    return np.random.default_rng(137)


def _instances_by_cid(instances: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {inst["bbox_labels"]["cid"]: inst for inst in instances}


def _make_mosaic_compose(
    *,
    grid_yx: tuple[int, int],
    target_size: tuple[int, int],
    cell_shape: tuple[int, int],
    fit_mode: str = "cover",
    strict: bool = False,
    instance_binding: list[str] | None = None,
    with_keypoints: bool = False,
) -> A.Compose:
    binding = instance_binding if instance_binding is not None else ["masks", "bboxes"]
    kp_params = None
    if with_keypoints or "keypoints" in binding:
        kp_params = A.KeypointParams(coord_format="xy", label_fields=["vis"])
    return A.Compose(
        [
            A.Mosaic(
                grid_yx=grid_yx,
                target_size=target_size,
                cell_shape=cell_shape,
                center_range=(0.5, 0.5),
                fit_mode=fit_mode,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
        keypoint_params=kp_params,
        instance_binding=binding,
        seed=137,
        strict=strict,
    )


def _two_source_mosaic_payload(
    *,
    ch: int,
    cw: int,
    layout: str,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
    tuple[int, int],
    tuple[int, int],
]:
    """Primary + metadata for a 2-cell mosaic; layout is 'vertical' (2,1) or 'horizontal' (1,2)."""
    img_p = rng.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
    img_m = rng.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
    instances = [
        {
            "mask": _make_mask(ch, cw, (4, 36, 4, 36)),
            "bbox": np.array([4, 4, 36, 36], dtype=np.float32),
            "bbox_labels": {"cid": 1},
        },
    ]
    mosaic_metadata = [
        {
            "image": img_m,
            "masks": np.stack([_make_mask(ch, cw, (8, 44, 8, 44))]),
            "bboxes": np.array([[8.0, 8.0, 44.0, 44.0]], dtype=np.float32),
            "bbox_labels": {"cid": [2]},
        },
    ]
    if layout == "vertical":
        target_size, grid_yx = (ch * 2, cw), (2, 1)
    elif layout == "horizontal":
        target_size, grid_yx = (ch, cw * 2), (1, 2)
    else:
        raise ValueError(layout)
    return img_p, instances, mosaic_metadata, target_size, grid_yx


class TestMosaicInstanceBindingExhaustive:
    @pytest.mark.parametrize("strict", [False, True], ids=["loose", "strict"])
    def test_strict_compose_accepts_mosaic_with_masks(self, strict: bool, rng_137: np.random.Generator) -> None:
        ch, cw = 48, 48
        transform = _make_mosaic_compose(
            grid_yx=(1, 1),
            target_size=(ch, cw),
            cell_shape=(ch, cw),
            strict=strict,
        )
        image = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (5, 25, 5, 25)),
                "bbox": np.array([5, 5, 25, 25], dtype=np.float32),
                "bbox_labels": {"cid": 7},
            },
        ]
        result = transform(image=image, instances=instances, mosaic_metadata=[])
        assert len(result["instances"]) == 1
        assert result["instances"][0]["bbox_labels"]["cid"] == 7
        assert result["instances"][0]["mask"].shape == (ch, cw)

    @pytest.mark.parametrize(
        ("fit_mode", "layout"),
        [
            ("cover", "vertical"),
            ("cover", "horizontal"),
            ("contain", "vertical"),
            ("contain", "horizontal"),
        ],
        ids=["cv", "ch", "ktv", "kth"],
    )
    def test_two_cell_mosaic_fit_mode_and_layout(
        self,
        fit_mode: str,
        layout: str,
        rng_137: np.random.Generator,
    ) -> None:
        ch, cw = 56, 56
        img_p, instances, mosaic_metadata, target_size, grid_yx = _two_source_mosaic_payload(
            ch=ch,
            cw=cw,
            layout=layout,
            rng=rng_137,
        )
        transform = _make_mosaic_compose(
            grid_yx=grid_yx,
            target_size=target_size,
            cell_shape=(ch, cw),
            fit_mode=fit_mode,
        )
        result = transform(image=img_p, instances=instances, mosaic_metadata=mosaic_metadata)
        assert len(result["instances"]) == 2
        th, tw = target_size
        for inst in result["instances"]:
            assert inst["mask"].shape == (th, tw)
        assert _instances_by_cid(result["instances"]).keys() == {1, 2}

    def test_mosaic_triple_binding_keypoint_counts_per_instance(self, rng_137: np.random.Generator) -> None:
        ch, cw = 48, 48
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 30, 4, 30)),
                "bbox": np.array([4, 4, 30, 30], dtype=np.float32),
                "bbox_labels": {"cid": 10},
                "keypoints": np.array([[12.0, 12.0], [18.0, 18.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [1, 1]},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (6, 40, 6, 40))]),
                "bboxes": np.array([[6.0, 6.0, 40.0, 40.0]], dtype=np.float32),
                "bbox_labels": {"cid": [20]},
                "keypoints": np.array([[22.0, 22.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [2]},
            },
        ]
        transform = _make_mosaic_compose(
            grid_yx=(2, 1),
            target_size=(ch * 2, cw),
            cell_shape=(ch, cw),
            with_keypoints=True,
            instance_binding=["masks", "bboxes", "keypoints"],
        )
        result = transform(image=img_p, instances=instances, mosaic_metadata=mosaic_metadata)
        assert len(result["instances"]) == 2
        by_cid = _instances_by_cid(result["instances"])
        assert by_cid[10]["keypoints"].shape == (2, 2)
        assert by_cid[20]["keypoints"].shape == (1, 2)
        np.testing.assert_array_equal(by_cid[10]["keypoint_labels"]["vis"], np.array([1, 1]))
        np.testing.assert_array_equal(by_cid[20]["keypoint_labels"]["vis"], np.array([2]))

    def test_mosaic_masks_plus_bboxes_only_no_keypoints_in_binding(self, rng_137: np.random.Generator) -> None:
        ch, cw = 40, 40
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (5, 25, 5, 25)),
                "bbox": np.array([5, 5, 25, 25], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
            {
                "mask": _make_mask(ch, cw, (28, 38, 28, 38)),
                "bbox": np.array([28, 28, 38, 38], dtype=np.float32),
                "bbox_labels": {"cid": 2},
            },
        ]
        transform = _make_mosaic_compose(
            grid_yx=(1, 1),
            target_size=(ch, cw),
            cell_shape=(ch, cw),
            instance_binding=["masks", "bboxes"],
        )
        result = transform(image=img_p, instances=instances, mosaic_metadata=[])
        assert len(result["instances"]) == 2
        assert "keypoints" not in result["instances"][0]

    def test_mosaic_empty_metadata_replicates_primary_into_both_cells(self, rng_137: np.random.Generator) -> None:
        """Two cells, no donors: primary is cloned into each cell → two bbox/mask rows after fuse."""
        ch, cw = 32, 32
        transform = _make_mosaic_compose(
            grid_yx=(2, 1),
            target_size=(ch * 2, cw),
            cell_shape=(ch, cw),
        )
        img = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 20, 4, 20)),
                "bbox": np.array([4, 4, 20, 20], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        result = transform(image=img, instances=instances, mosaic_metadata=[])
        assert len(result["instances"]) == 2
        for inst in result["instances"]:
            assert inst["mask"].shape == (ch * 2, cw)
            assert int(inst["bbox_labels"]["cid"]) == 1

    def test_mosaic_zero_instances_returns_empty(self) -> None:
        transform = _make_mosaic_compose(
            grid_yx=(1, 1),
            target_size=(24, 24),
            cell_shape=(24, 24),
        )
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        result = transform(image=image, instances=[], mosaic_metadata=[])
        assert result["instances"] == []

    def test_mosaic_chained_with_crop_repacks_instances(self, rng_137: np.random.Generator) -> None:
        ch, cw = 48, 48
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(1, 1),
                    target_size=(ch, cw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    p=1.0,
                ),
                A.Crop(x_min=0, y_min=0, x_max=32, y_max=32, p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"], min_area=1),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        image = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (5, 30, 5, 30)),
                "bbox": np.array([5, 5, 30, 30], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        result = transform(image=image, instances=instances, mosaic_metadata=[])
        assert len(result["instances"]) >= 1
        assert result["instances"][0]["mask"].shape == (32, 32)


class TestCopyPasteInstanceBindingExhaustive:
    @pytest.fixture
    def base_two_instance_payload(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Shared primary image + two non-overlapping instances for CopyAndPaste matrices."""
        image = np.zeros((72, 72, 3), dtype=np.uint8)
        m0 = _make_mask(72, 72, (6, 34, 6, 34))
        m1 = _make_mask(72, 72, (40, 68, 40, 68))
        instances = [
            {
                "mask": m0,
                "bbox": np.array([6, 6, 34, 34], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
            {
                "mask": m1,
                "bbox": np.array([40, 40, 68, 68], dtype=np.float32),
                "bbox_labels": {"cid": 2},
            },
        ]
        return image, instances

    def test_masks_bboxes_only_pasted_labels_and_counts(
        self,
        base_two_instance_payload: tuple[np.ndarray, list[dict[str, Any]]],
    ) -> None:
        image, instances = base_two_instance_payload
        paste_mask = np.zeros((72, 72), dtype=np.uint8)
        paste_mask[:28, :28] = 1
        donor: dict[str, Any] = {
            "image": np.full((72, 72, 3), 99, dtype=np.uint8),
            "mask": paste_mask,
            "bbox": [0, 0, 28, 28],
            "bbox_labels": {"cid": 77},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.05, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        by_cid = _instances_by_cid(result["instances"])
        assert 77 in by_cid
        assert by_cid[77]["bbox_labels"]["cid"] == 77
        assert len(result["instances"]) >= 2

    @pytest.mark.parametrize("min_vis", [0.0, 0.45, 0.9], ids=["v0", "v045", "v09"])
    def test_copy_paste_min_visibility_param_survival_matrix(
        self,
        base_two_instance_payload: tuple[np.ndarray, list[dict[str, Any]]],
        min_vis: float,
    ) -> None:
        image, instances = base_two_instance_payload
        paste_mask = np.zeros((72, 72), dtype=np.uint8)
        paste_mask[:20, :20] = 1
        donor: dict[str, Any] = {
            "image": np.full((72, 72, 3), 50, dtype=np.uint8),
            "mask": paste_mask,
            "bbox": [0, 0, 20, 20],
            "bbox_labels": {"cid": 900},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=min_vis, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        assert len(result["instances"]) >= 1
        cids = {inst["bbox_labels"]["cid"] for inst in result["instances"]}
        assert 900 in cids

    def test_all_primaries_occluded_only_paste_remains(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        full = np.ones((64, 64), dtype=np.uint8)
        instances = [
            {
                "mask": full,
                "bbox": np.array([0, 0, 64, 64], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        paste_mask = np.ones((64, 64), dtype=np.uint8)
        donor: dict[str, Any] = {
            "image": np.full((64, 64, 3), 200, dtype=np.uint8),
            "mask": paste_mask,
            "bbox": [0, 0, 64, 64],
            "bbox_labels": {"cid": 500},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.99, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        assert len(result["instances"]) == 1
        assert result["instances"][0]["bbox_labels"]["cid"] == 500

    def test_two_donors_distinct_cids_and_mask_stack(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        m0 = _make_mask(80, 80, (50, 78, 50, 78))
        instances = [
            {
                "mask": m0,
                "bbox": np.array([50, 50, 78, 78], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        d1_mask = np.zeros((80, 80), dtype=np.uint8)
        d1_mask[0:15, 0:15] = 1
        d2_mask = np.zeros((80, 80), dtype=np.uint8)
        d2_mask[0:15, 20:35] = 1
        meta = [
            {
                "image": np.full((80, 80, 3), 10, dtype=np.uint8),
                "mask": d1_mask,
                "bbox": [0, 0, 15, 15],
                "bbox_labels": {"cid": 101},
            },
            {
                "image": np.full((80, 80, 3), 20, dtype=np.uint8),
                "mask": d2_mask,
                "bbox": [20, 0, 35, 15],
                "bbox_labels": {"cid": 102},
            },
        ]
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=meta)
        cids = {inst["bbox_labels"]["cid"] for inst in result["instances"]}
        assert {1, 101, 102}.issubset(cids)
        stacked = np.stack([inst["mask"] for inst in result["instances"]], axis=0)
        assert stacked.shape[0] == len(result["instances"])

    def test_empty_metadata_no_paste_instances_unchanged_count(self) -> None:
        image = _make_image(48, 48)
        instances = [
            {
                "mask": _make_mask(48, 48, (5, 30, 5, 30)),
                "bbox": np.array([5, 5, 30, 30], dtype=np.float32),
                "bbox_labels": {"cid": 3},
            },
        ]
        transform = A.Compose(
            [A.CopyAndPaste(p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[])
        assert len(result["instances"]) == 1
        assert result["instances"][0]["bbox_labels"]["cid"] == 3

    def test_donor_mask_empty_skipped(self) -> None:
        image = _make_image(40, 40)
        instances = [
            {
                "mask": _make_mask(40, 40, (5, 25, 5, 25)),
                "bbox": np.array([5, 5, 25, 25], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        donor: dict[str, Any] = {
            "image": np.zeros((40, 40, 3), dtype=np.uint8),
            "mask": np.zeros((40, 40), dtype=np.uint8),
            "bbox": [0, 0, 1, 1],
            "bbox_labels": {"cid": 99},
        }
        transform = A.Compose(
            [A.CopyAndPaste(p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        assert len(result["instances"]) == 1

    def test_triple_binding_assert_pasted_keypoint_row_and_survivor_cids(self) -> None:
        image = np.zeros((88, 88, 3), dtype=np.uint8)
        m0 = _make_mask(88, 88, (8, 40, 8, 40))
        m1 = _make_mask(88, 88, (52, 84, 52, 84))
        instances = [
            {
                "mask": m0,
                "bbox": np.array([8, 8, 40, 40], dtype=np.float32),
                "bbox_labels": {"cid": 1},
                "keypoints": np.array([[12.0, 12.0], [20.0, 20.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [1, 1]},
            },
            {
                "mask": m1,
                "bbox": np.array([52, 52, 84, 84], dtype=np.float32),
                "bbox_labels": {"cid": 2},
                "keypoints": np.array([[70.0, 70.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [2]},
            },
        ]
        paste_mask = np.zeros((88, 88), dtype=np.uint8)
        paste_mask[:44, :44] = 1
        donor: dict[str, Any] = {
            "image": np.full((88, 88, 3), 123, dtype=np.uint8),
            "mask": paste_mask,
            "bbox": [0, 0, 44, 44],
            "bbox_labels": {"cid": 999},
            "keypoints": np.array([[22.0, 22.0], [30.0, 30.0]], dtype=np.float32),
            "keypoint_labels": {"vis": [8, 8]},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.05, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["vis"]),
            instance_binding=["masks", "bboxes", "keypoints"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        by_cid = _instances_by_cid(result["instances"])
        assert 999 in by_cid
        assert by_cid[999]["keypoints"].shape[0] == 2
        np.testing.assert_array_equal(by_cid[999]["keypoint_labels"]["vis"], np.array([8, 8]))
        assert 2 in by_cid
        assert by_cid[2]["keypoints"].shape[0] == 1

    def test_copy_paste_min_visibility_one_excludes_all_primaries(self) -> None:
        """min_visibility_after_paste=1.0 requires untouched masks — any overlap removes instance."""
        image = np.zeros((56, 56, 3), dtype=np.uint8)
        m = _make_mask(56, 56, (10, 46, 10, 46))
        instances = [
            {
                "mask": m,
                "bbox": np.array([10, 10, 46, 46], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        paste = np.zeros((56, 56), dtype=np.uint8)
        paste[20:30, 20:30] = 1
        donor: dict[str, Any] = {
            "image": np.ones((56, 56, 3), dtype=np.uint8),
            "mask": paste,
            "bbox": [20, 20, 30, 30],
            "bbox_labels": {"cid": 88},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=1.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        assert len(result["instances"]) == 1
        assert int(result["instances"][0]["bbox_labels"]["cid"]) == 88

    @pytest.mark.parametrize("dtype_mask", [np.uint8, np.int32], ids=["u8", "i32"])
    def test_copy_paste_mask_dtype_variants(self, dtype_mask: np.dtype) -> None:
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        m = np.zeros((50, 50), dtype=dtype_mask)
        m[10:30, 10:30] = 1
        instances = [
            {
                "mask": m,
                "bbox": np.array([10, 10, 30, 30], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        paste = np.zeros((50, 50), dtype=dtype_mask)
        paste[:12, :12] = 1
        donor: dict[str, Any] = {
            "image": np.ones((50, 50, 3), dtype=np.uint8),
            "mask": paste,
            "bbox": [0, 0, 12, 12],
            "bbox_labels": {"cid": 2},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        assert len(result["instances"]) == 2


class TestMixingInstanceBindingRegression:
    def test_mosaic_then_horizontal_flip_deterministic_seed(self, rng_137: np.random.Generator) -> None:
        ch, cw = 40, 40
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(1, 1),
                    target_size=(ch, cw),
                    cell_shape=(ch, cw),
                    p=1.0,
                ),
                A.HorizontalFlip(p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        image = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (5, 25, 5, 30)),
                "bbox": np.array([5, 5, 30, 25], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        r1 = transform(image=image.copy(), instances=instances, mosaic_metadata=[])
        r2 = transform(image=image.copy(), instances=instances, mosaic_metadata=[])
        assert len(r1["instances"]) == len(r2["instances"]) == 1
        np.testing.assert_array_equal(r1["instances"][0]["mask"], r2["instances"][0]["mask"])


class TestMosaicCopyPasteInstanceBinding:
    """Mosaic then CopyAndPaste: fused mosaic stack must flow through paste visibility + repack."""

    def test_mosaic_then_copy_paste_masks_bboxes_occluded_primary_dropped_pasted_kept(
        self,
        rng_137: np.random.Generator,
    ) -> None:
        ch, cw = 48, 48
        th, tw = ch * 2, cw
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 40, 4, 40)),
                "bbox": np.array([4, 4, 40, 40], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (6, 42, 6, 42))]),
                "bboxes": np.array([[6.0, 6.0, 42.0, 42.0]], dtype=np.float32),
                "bbox_labels": {"cid": [2]},
            },
        ]
        # Full-cover donor: deterministic paste over the entire mosaic canvas; both primaries
        # are occluded and only the pasted instance survives.
        donor: dict[str, Any] = {
            "image": np.full((th, tw, 3), 222, dtype=np.uint8),
            "mask": np.ones((th, tw), dtype=np.uint8),
            "bbox": [0, 0, int(tw), int(th)],
            "bbox_labels": {"cid": 900},
        }
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(2, 1),
                    target_size=(th, tw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    fit_mode="cover",
                    p=1.0,
                ),
                A.CopyAndPaste(min_visibility_after_paste=0.05, p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(
            image=img_p,
            instances=instances,
            mosaic_metadata=mosaic_metadata,
            copy_paste_metadata=[donor],
        )
        cids = {int(inst["bbox_labels"]["cid"]) for inst in result["instances"]}
        assert cids == {900}
        for inst in result["instances"]:
            assert inst["mask"].shape == (th, tw)

    def test_mosaic_then_copy_paste_triple_binding_keypoints_follow_survivors(
        self,
        rng_137: np.random.Generator,
    ) -> None:
        ch, cw = 48, 48
        th, tw = ch * 2, cw
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 40, 4, 40)),
                "bbox": np.array([4, 4, 40, 40], dtype=np.float32),
                "bbox_labels": {"cid": 1},
                "keypoints": np.array([[12.0, 12.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [1]},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (6, 42, 6, 42))]),
                "bboxes": np.array([[6.0, 6.0, 42.0, 42.0]], dtype=np.float32),
                "bbox_labels": {"cid": [2]},
                "keypoints": np.array([[24.0, 24.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [2]},
            },
        ]
        # Full-cover donor for deterministic paste; both primaries get occluded, leaving only
        # the pasted instance + its keypoints.
        donor: dict[str, Any] = {
            "image": np.full((th, tw, 3), 111, dtype=np.uint8),
            "mask": np.ones((th, tw), dtype=np.uint8),
            "bbox": [0, 0, int(tw), int(th)],
            "bbox_labels": {"cid": 900},
            "keypoints": np.array([[8.0, 8.0]], dtype=np.float32),
            "keypoint_labels": {"vis": [9]},
        }
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(2, 1),
                    target_size=(th, tw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    fit_mode="cover",
                    p=1.0,
                ),
                A.CopyAndPaste(min_visibility_after_paste=0.05, p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["vis"]),
            instance_binding=["masks", "bboxes", "keypoints"],
            seed=137,
        )
        result = transform(
            image=img_p,
            instances=instances,
            mosaic_metadata=mosaic_metadata,
            copy_paste_metadata=[donor],
        )
        assert len(result["instances"]) == 1
        by_cid = _instances_by_cid(result["instances"])
        assert set(by_cid) == {900}
        assert by_cid[900]["keypoints"].shape == (1, 2)
        np.testing.assert_array_equal(by_cid[900]["keypoint_labels"]["vis"], np.array([9]))


# ---------------------------------------------------------------------------
# CopyAndPaste regression tests for ID-driven survivor selection.
# These pin the contract that mask rows stay row-aligned with surviving
# bboxes (and that pasted IDs cannot collide with existing instance IDs)
# even when an upstream transform has filtered some bboxes.
# ---------------------------------------------------------------------------


def _three_instance_payload(
    side: int = 80,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Three non-overlapping instances laid out along the diagonal: A (top-left), B (middle), C (bottom-right)."""
    image = np.zeros((side, side, 3), dtype=np.uint8)
    third = side // 3
    a = _make_mask(side, side, (2, third - 2, 2, third - 2))
    b = _make_mask(side, side, (third + 2, 2 * third - 2, third + 2, 2 * third - 2))
    c = _make_mask(side, side, (2 * third + 2, side - 2, 2 * third + 2, side - 2))
    instances = [
        {
            "mask": a,
            "bbox": np.array([2, 2, third - 2, third - 2], dtype=np.float32),
            "bbox_labels": {"cid": "A"},
        },
        {
            "mask": b,
            "bbox": np.array([third + 2, third + 2, 2 * third - 2, 2 * third - 2], dtype=np.float32),
            "bbox_labels": {"cid": "B"},
        },
        {
            "mask": c,
            "bbox": np.array([2 * third + 2, 2 * third + 2, side - 2, side - 2], dtype=np.float32),
            "bbox_labels": {"cid": "C"},
        },
    ]
    return image, instances


def _mask_matches_cid_region(
    mask: np.ndarray,
    expected_region: tuple[int, int, int, int],
) -> bool:
    """Return True iff every non-zero pixel of `mask` falls inside `expected_region` (y1, y2, x1, x2)."""
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return False
    y1, y2, x1, x2 = expected_region
    inside = (nz[:, 0] >= y1) & (nz[:, 0] < y2) & (nz[:, 1] >= x1) & (nz[:, 1] < x2)
    return bool(inside.all())


class TestCopyPasteBindingRegression:
    """Regression tests for the ID-vs-position alignment bug between CopyAndPaste and instance binding."""

    def test_crop_filters_then_copy_paste_no_indexerror(self) -> None:
        """Crop drops middle instance B by min_area; CopyAndPaste then occludes A. Should survive C + paste."""
        side = 90
        image = np.zeros((side, side, 3), dtype=np.uint8)
        third = side // 3
        # A and C are big (area 676); B is intentionally tiny (area 4) so min_area filters only B.
        a_region = (2, third - 2, 2, third - 2)
        b_region = (third + 14, third + 16, third + 14, third + 16)
        c_region = (2 * third + 2, side - 2, 2 * third + 2, side - 2)
        instances = [
            {
                "mask": _make_mask(side, side, a_region),
                "bbox": np.array([2, 2, third - 2, third - 2], dtype=np.float32),
                "bbox_labels": {"cid": "A"},
            },
            {
                "mask": _make_mask(side, side, b_region),
                "bbox": np.array([third + 14, third + 14, third + 16, third + 16], dtype=np.float32),
                "bbox_labels": {"cid": "B"},
            },
            {
                "mask": _make_mask(side, side, c_region),
                "bbox": np.array([2 * third + 2, 2 * third + 2, side - 2, side - 2], dtype=np.float32),
                "bbox_labels": {"cid": "C"},
            },
        ]
        # Full-cover donor → deterministic paste over the entire post-crop canvas. Combined with
        # min_area filtering of B, this exercises the id-allocation path when the binding mask
        # row count is reduced before the pasted instance is appended.
        donor: dict[str, Any] = {
            "image": np.full((side, side, 3), 200, dtype=np.uint8),
            "mask": np.ones((side, side), dtype=np.uint8),
            "bbox": [0, 0, side, side],
            "bbox_labels": {"cid": "PASTED"},
        }
        transform = A.Compose(
            [
                A.Crop(x_min=0, y_min=0, x_max=side, y_max=side, p=1.0),
                A.CopyAndPaste(min_visibility_after_paste=0.3, p=1.0),
            ],
            bbox_params=A.BboxParams(
                coord_format="pascal_voc",
                label_fields=["cid"],
                min_area=10,
            ),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        cids = {inst["bbox_labels"]["cid"] for inst in result["instances"]}
        assert cids == {"PASTED"}, "B filtered by min_area; A and C fully occluded by paste"
        for inst in result["instances"]:
            assert inst["mask"].shape == (side, side)

    def test_copy_paste_full_cover_drops_all_primaries_with_binding(self) -> None:
        """Full-cover paste deterministically drops every primary; binding must allocate the
        pasted id without IndexError, and the lone surviving instance is the pasted one with
        a mask matching the full target footprint.
        """
        side = 90
        image, instances = _three_instance_payload(side)
        donor: dict[str, Any] = {
            "image": np.full((side, side, 3), 100, dtype=np.uint8),
            "mask": np.ones((side, side), dtype=np.uint8),
            "bbox": [0, 0, side, side],
            "bbox_labels": {"cid": "PASTED"},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.3, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        by_cid = _instances_by_cid(result["instances"])
        assert set(by_cid) == {"PASTED"}
        assert by_cid["PASTED"]["mask"].sum() == side * side

    def test_copy_paste_id_collision_when_crop_drops_low_id_instance(self) -> None:
        """After Crop drops A (id 0), pasted IDs must allocate above N_masks-1 to avoid colliding with B/C ids."""
        side = 96
        image, instances = _three_instance_payload(side)
        third = side // 3
        crop_offset = third + 1
        cropped_side = side - crop_offset
        # Full-cover donor in post-crop coordinates; deterministic paste over everything.
        donor: dict[str, Any] = {
            "image": np.full((cropped_side, cropped_side, 3), 50, dtype=np.uint8),
            "mask": np.ones((cropped_side, cropped_side), dtype=np.uint8),
            "bbox": [0, 0, cropped_side, cropped_side],
            "bbox_labels": {"cid": "PASTED"},
        }
        transform = A.Compose(
            [
                A.Crop(x_min=crop_offset, y_min=crop_offset, x_max=side, y_max=side, p=1.0),
                A.CopyAndPaste(min_visibility_after_paste=0.3, p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        cids = {inst["bbox_labels"]["cid"] for inst in result["instances"]}
        assert cids == {"PASTED"}, "A cropped, B and C fully occluded by paste"
        for inst in result["instances"]:
            assert inst["mask"].shape[:2] == (cropped_side, cropped_side)

    def test_copy_paste_keypoint_id_filter_uses_ids_not_positions(self) -> None:
        """Triple binding: Crop drops A, paste occludes B. Keypoints must follow C and the pasted donor."""
        side = 96
        image = np.zeros((side, side, 3), dtype=np.uint8)
        third = side // 3
        instances = [
            {
                "mask": _make_mask(side, side, (2, third - 2, 2, third - 2)),
                "bbox": np.array([2, 2, third - 2, third - 2], dtype=np.float32),
                "bbox_labels": {"cid": "A"},
                "keypoints": np.array([[10.0, 10.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [1]},
            },
            {
                "mask": _make_mask(side, side, (third + 2, 2 * third - 2, third + 2, 2 * third - 2)),
                "bbox": np.array(
                    [third + 2, third + 2, 2 * third - 2, 2 * third - 2],
                    dtype=np.float32,
                ),
                "bbox_labels": {"cid": "B"},
                "keypoints": np.array([[third + 5.0, third + 5.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [2]},
            },
            {
                "mask": _make_mask(side, side, (2 * third + 2, side - 2, 2 * third + 2, side - 2)),
                "bbox": np.array(
                    [2 * third + 2, 2 * third + 2, side - 2, side - 2],
                    dtype=np.float32,
                ),
                "bbox_labels": {"cid": "C"},
                "keypoints": np.array([[2 * third + 5.0, 2 * third + 5.0]], dtype=np.float32),
                "keypoint_labels": {"vis": [3]},
            },
        ]
        crop_offset = third + 1
        cropped_side = side - crop_offset
        # Full-cover donor in post-crop coordinates; A is cropped out, B and C are fully occluded
        # by the paste. Only the pasted instance and its keypoint should survive.
        donor: dict[str, Any] = {
            "image": np.full((cropped_side, cropped_side, 3), 99, dtype=np.uint8),
            "mask": np.ones((cropped_side, cropped_side), dtype=np.uint8),
            "bbox": [0, 0, cropped_side, cropped_side],
            "bbox_labels": {"cid": "PASTED"},
            "keypoints": np.array([[3.0, 3.0]], dtype=np.float32),
            "keypoint_labels": {"vis": [9]},
        }
        transform = A.Compose(
            [
                A.Crop(x_min=crop_offset, y_min=crop_offset, x_max=side, y_max=side, p=1.0),
                A.CopyAndPaste(min_visibility_after_paste=0.3, p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["vis"]),
            instance_binding=["masks", "bboxes", "keypoints"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        by_cid = _instances_by_cid(result["instances"])
        assert set(by_cid) == {"PASTED"}
        np.testing.assert_array_equal(by_cid["PASTED"]["keypoint_labels"]["vis"], np.array([9]))

    def test_copy_paste_min_visibility_zero_keeps_all_primaries_with_binding(self) -> None:
        """min_visibility_after_paste=0.0 means every primary survives regardless of overlap; pasted appended."""
        side = 64
        image, instances = _three_instance_payload(side)
        paste_mask = np.ones((side, side), dtype=np.uint8)
        donor: dict[str, Any] = {
            "image": np.full((side, side, 3), 200, dtype=np.uint8),
            "mask": paste_mask,
            "bbox": [0, 0, side, side],
            "bbox_labels": {"cid": "PASTED"},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        cids = [inst["bbox_labels"]["cid"] for inst in result["instances"]]
        assert cids == ["A", "B", "C", "PASTED"]

    def test_copy_paste_min_visibility_one_drops_all_primaries_with_binding(self) -> None:
        """min_visibility_after_paste=1.0 with any overlap removes all primaries; only pasted survives."""
        side = 64
        image, instances = _three_instance_payload(side)
        paste_mask = np.ones((side, side), dtype=np.uint8)
        donor: dict[str, Any] = {
            "image": np.full((side, side, 3), 200, dtype=np.uint8),
            "mask": paste_mask,
            "bbox": [0, 0, side, side],
            "bbox_labels": {"cid": "PASTED"},
        }
        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=1.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances, copy_paste_metadata=[donor])
        assert [inst["bbox_labels"]["cid"] for inst in result["instances"]] == ["PASTED"]


class TestMosaicBindingRegression:
    """Mosaic edge cases that can stress the ID/position invariant in masks vs bboxes."""

    def test_mosaic_min_area_filters_some_cells_then_repack_ok(self, rng_137: np.random.Generator) -> None:
        """Mosaic with bbox min_area drops a tiny instance; remaining mask rows still attach to right ids."""
        ch, cw = 64, 64
        th, tw = ch * 2, cw
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 30, 4, 30)),
                "bbox": np.array([4, 4, 30, 30], dtype=np.float32),
                "bbox_labels": {"cid": "P_BIG"},
            },
            {
                "mask": _make_mask(ch, cw, (60, 62, 60, 62)),
                "bbox": np.array([60, 60, 62, 62], dtype=np.float32),
                "bbox_labels": {"cid": "P_TINY"},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (5, 28, 5, 28))]),
                "bboxes": np.array([[5, 5, 28, 28]], dtype=np.float32),
                "bbox_labels": {"cid": ["M_BIG"]},
            },
        ]
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(2, 1),
                    target_size=(th, tw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    fit_mode="cover",
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"], min_area=100),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=img_p, instances=instances, mosaic_metadata=mosaic_metadata)
        cids = {inst["bbox_labels"]["cid"] for inst in result["instances"]}
        assert "P_TINY" not in cids
        assert {"P_BIG", "M_BIG"}.issubset(cids)
        for inst in result["instances"]:
            assert inst["mask"].shape == (th, tw)

    def test_mosaic_obb_with_binding(self, rng_137: np.random.Generator) -> None:
        """Mosaic must preserve the angle column on OBB bboxes when instance binding is active."""
        ch, cw = 48, 48
        th, tw = ch * 2, cw
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 40, 4, 40)),
                "bbox": np.array([4, 4, 40, 40, 0.0], dtype=np.float32),
                "bbox_labels": {"cid": 1},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (6, 42, 6, 42))]),
                "bboxes": np.array([[6.0, 6.0, 42.0, 42.0, 0.0]], dtype=np.float32),
                "bbox_labels": {"cid": [2]},
            },
        ]
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(2, 1),
                    target_size=(th, tw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    fit_mode="cover",
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"], bbox_type="obb"),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=img_p, instances=instances, mosaic_metadata=mosaic_metadata)
        for inst in result["instances"]:
            assert inst["bbox"].shape[-1] == 5

    def test_mosaic_then_crop_then_copy_paste_chain(self, rng_137: np.random.Generator) -> None:
        """3-step chain that historically tripped the IndexError; all three exercise ID remapping."""
        ch, cw = 64, 64
        th, tw = ch * 2, cw
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 30, 4, 30)),
                "bbox": np.array([4, 4, 30, 30], dtype=np.float32),
                "bbox_labels": {"cid": "P0"},
            },
            {
                "mask": _make_mask(ch, cw, (40, 60, 40, 60)),
                "bbox": np.array([40, 40, 60, 60], dtype=np.float32),
                "bbox_labels": {"cid": "P1"},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (8, 36, 8, 36))]),
                "bboxes": np.array([[8.0, 8.0, 36.0, 36.0]], dtype=np.float32),
                "bbox_labels": {"cid": ["M0"]},
            },
        ]
        crop_h = th // 2 + ch // 4
        paste_mask = np.zeros((crop_h, tw), dtype=np.uint8)
        paste_mask[: ch // 2, : cw // 2] = 1
        donor: dict[str, Any] = {
            "image": np.full((crop_h, tw, 3), 200, dtype=np.uint8),
            "mask": paste_mask,
            "bbox": [0, 0, cw // 2, ch // 2],
            "bbox_labels": {"cid": "PASTED"},
        }
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(2, 1),
                    target_size=(th, tw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    fit_mode="cover",
                    p=1.0,
                ),
                A.Crop(x_min=0, y_min=0, x_max=tw, y_max=crop_h, p=1.0),
                A.CopyAndPaste(min_visibility_after_paste=0.3, p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(
            image=img_p,
            instances=instances,
            mosaic_metadata=mosaic_metadata,
            copy_paste_metadata=[donor],
        )
        cids = {inst["bbox_labels"]["cid"] for inst in result["instances"]}
        assert "PASTED" in cids
        for inst in result["instances"]:
            assert inst["mask"].shape == (crop_h, tw)


class TestMosaicCopyPasteFourInstanceOcclusion:
    """Mosaic emits 4 instances; full-cover CopyAndPaste occludes all of them.
    The pasted instance must still be allocated a fresh id without IndexError, and any
    surviving (non-pasted) mask rows must remain positionally aligned to their owners.
    """

    def test_mosaic_4_instances_full_cover_paste(self, rng_137: np.random.Generator) -> None:
        ch, cw = 48, 48
        th, tw = ch * 2, cw * 2
        img_p = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        img_m = rng_137.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
        instances = [
            {
                "mask": _make_mask(ch, cw, (4, 40, 4, 40)),
                "bbox": np.array([4, 4, 40, 40], dtype=np.float32),
                "bbox_labels": {"cid": "P0"},
            },
        ]
        mosaic_metadata = [
            {
                "image": img_m,
                "masks": np.stack([_make_mask(ch, cw, (4, 40, 4, 40))]),
                "bboxes": np.array([[4.0, 4.0, 40.0, 40.0]], dtype=np.float32),
                "bbox_labels": {"cid": [f"M{i}"]},
            }
            for i in range(3)
        ]
        # Full-cover donor: deterministic paste over the entire mosaic canvas.
        donor: dict[str, Any] = {
            "image": np.full((th, tw, 3), 88, dtype=np.uint8),
            "mask": np.ones((th, tw), dtype=np.uint8),
            "bbox": [0, 0, tw, th],
            "bbox_labels": {"cid": "PASTED"},
        }
        transform = A.Compose(
            [
                A.Mosaic(
                    grid_yx=(2, 2),
                    target_size=(th, tw),
                    cell_shape=(ch, cw),
                    center_range=(0.5, 0.5),
                    fit_mode="cover",
                    p=1.0,
                ),
                A.CopyAndPaste(min_visibility_after_paste=0.3, p=1.0),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"]),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(
            image=img_p,
            instances=instances,
            mosaic_metadata=mosaic_metadata,
            copy_paste_metadata=[donor],
        )
        cids = [inst["bbox_labels"]["cid"] for inst in result["instances"]]
        assert cids == ["PASTED"]
        pasted = result["instances"][0]
        assert pasted["mask"].shape == (th, tw)
        assert pasted["mask"].sum() == th * tw


class TestFilteringTransformBinding:
    """Generic sweep: filtering transforms must keep mask rows positionally aligned to instance ids."""

    @staticmethod
    def _three_inline_instances(side: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """A in left third, B in middle (will be filtered), C in right third."""
        image = np.zeros((side, side, 3), dtype=np.uint8)
        third = side // 3
        instances = [
            {
                "mask": _make_mask(side, side, (10, side - 10, 2, third - 2)),
                "bbox": np.array([2, 10, third - 2, side - 10], dtype=np.float32),
                "bbox_labels": {"cid": "A"},
            },
            {
                "mask": _make_mask(side, side, (10, side - 10, third + 2, 2 * third - 2)),
                "bbox": np.array([third + 2, 10, 2 * third - 2, side - 10], dtype=np.float32),
                "bbox_labels": {"cid": "B"},
            },
            {
                "mask": _make_mask(side, side, (10, side - 10, 2 * third + 2, side - 2)),
                "bbox": np.array([2 * third + 2, 10, side - 2, side - 10], dtype=np.float32),
                "bbox_labels": {"cid": "C"},
            },
        ]
        return image, instances

    @pytest.mark.parametrize(
        ("transform_factory", "expected_dropped"),
        [
            pytest.param(
                lambda side: A.Crop(x_min=0, y_min=0, x_max=side // 3, y_max=side, p=1.0),
                {"B", "C"},
                id="crop-keep-A",
            ),
            pytest.param(
                lambda side: A.CenterCrop(height=side // 3 - 2, width=side // 3 - 2, p=1.0),
                {"A", "C"},
                id="center-crop-keep-B",
            ),
        ],
    )
    def test_crop_family_drops_some_instances_with_binding(
        self,
        transform_factory: Any,
        expected_dropped: set[str],
    ) -> None:
        side = 90
        image, instances = self._three_inline_instances(side)
        transform = A.Compose(
            [transform_factory(side)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"], min_area=10),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances)
        cids = {inst["bbox_labels"]["cid"] for inst in result["instances"]}
        expected_survivors = {"A", "B", "C"} - expected_dropped
        assert cids == expected_survivors, (
            f"survivors mismatch: got {cids}, expected {expected_survivors} (dropped={expected_dropped})"
        )
        for inst in result["instances"]:
            label = inst["bbox_labels"]["cid"]
            mask = inst["mask"]
            assert mask.shape[:2] == result["image"].shape[:2]
            nz = np.argwhere(mask > 0)
            if nz.size == 0:
                continue
            x_min = float(inst["bbox"][0])
            x_max = float(inst["bbox"][2])
            assert nz[:, 1].min() + 0.5 >= x_min - 1, (
                f"Mask of instance {label} has pixels left of its bbox - row was attached to wrong instance id."
            )
            assert nz[:, 1].max() <= x_max + 1, (
                f"Mask of instance {label} has pixels right of its bbox - row was attached to wrong instance id."
            )

    def test_coarse_dropout_with_binding_filters_instance_under_hole(self, rng_137: np.random.Generator) -> None:
        """A hole over instance B drops B from bboxes; A and C must keep their original masks."""
        side = 96
        image, instances = self._three_inline_instances(side)
        third = side // 3
        # Force the dropout hole to fully cover B by making it large and well-positioned.
        transform = A.Compose(
            [
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(side - 20, side - 20),
                    hole_width_range=(third - 2, third - 2),
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                coord_format="pascal_voc",
                label_fields=["cid"],
                min_area=10,
                min_visibility=0.5,
            ),
            instance_binding=["masks", "bboxes"],
            seed=137,
        )
        result = transform(image=image, instances=instances)
        for inst in result["instances"]:
            label = inst["bbox_labels"]["cid"]
            x_min = float(inst["bbox"][0])
            x_max = float(inst["bbox"][2])
            assert x_min >= 0 and x_max <= side
            nz = np.argwhere(inst["mask"] > 0)
            if nz.size > 0:
                assert nz[:, 1].min() + 0.5 >= x_min - 1, (
                    f"Mask of instance {label} bleeds left of bbox - id/position mismatch in dropout path."
                )
                assert nz[:, 1].max() <= x_max + 1, (
                    f"Mask of instance {label} bleeds right of bbox - id/position mismatch in dropout path."
                )

    def test_keypoint_binding_through_crop(self) -> None:
        """Crop drops B by leaving it outside the crop region; A and C survive with right keypoints attached."""
        side = 90
        image, instances = self._three_inline_instances(side)
        third = side // 3
        for inst, kp_xy in zip(
            instances,
            [(10.0, 10.0), (third + 5.0, 10.0), (2 * third + 5.0, 10.0)],
            strict=True,
        ):
            inst["keypoints"] = np.array([list(kp_xy)], dtype=np.float32)
            inst["keypoint_labels"] = {"vis": [int(inst["bbox_labels"]["cid"] != "B")]}
        transform = A.Compose(
            [A.Crop(x_min=0, y_min=0, x_max=third + 4, y_max=side, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["cid"], min_area=10),
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["vis"]),
            instance_binding=["masks", "bboxes", "keypoints"],
            seed=137,
        )
        result = transform(image=image, instances=instances)
        by_cid = _instances_by_cid(result["instances"])
        assert "A" in by_cid
        assert by_cid["A"]["keypoints"].shape[0] == 1
        np.testing.assert_array_equal(by_cid["A"]["keypoint_labels"]["vis"], np.array([1]))
