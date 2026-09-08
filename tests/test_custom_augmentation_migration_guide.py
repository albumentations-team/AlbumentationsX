import numpy as np

import albumentations as A


def test_sequence_recipe_shares_one_spatial_realization() -> None:
    frames = np.arange(3 * 6 * 8, dtype=np.uint8).reshape(3, 6, 8, 1)
    transform = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    result = transform(images=frames)["images"]

    np.testing.assert_array_equal(result, np.flip(frames, axis=2))


def test_volume_recipe_preserves_depth_and_shares_geometry() -> None:
    volume = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8, 1)
    transform = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    result = transform(volume=volume)["volume"]

    np.testing.assert_array_equal(result, np.flip(volume, axis=2))


def test_additional_target_recipe_keeps_paired_geometry_aligned() -> None:
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8, 1)
    depth = np.arange(6 * 8, dtype=np.float32).reshape(6, 8, 1)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        additional_targets={"depth": "image"},
        strict=True,
    )

    result = transform(image=image, depth=depth)

    np.testing.assert_array_equal(result["image"], np.flip(image, axis=1))
    np.testing.assert_array_equal(result["depth"], np.flip(depth, axis=1))


def test_structured_recipe_filters_complete_bound_instance() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)

    def mask_at(start: int, stop: int) -> np.ndarray:
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[start:stop, start:stop] = 1
        return mask

    instances = [
        {
            "mask": mask_at(5, 25),
            "bbox": np.array([5, 5, 25, 25], dtype=np.float32),
            "keypoints": np.array([[15.0, 15.0]], dtype=np.float32),
            "bbox_labels": {"class_id": "cat"},
        },
        {
            "mask": mask_at(50, 70),
            "bbox": np.array([50, 50, 70, 70], dtype=np.float32),
            "keypoints": np.array([[60.0, 60.0]], dtype=np.float32),
            "bbox_labels": {"class_id": "dog"},
        },
    ]
    transform = A.Compose(
        [A.Crop(x_min=0, y_min=0, x_max=40, y_max=40, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
        keypoint_params=A.KeypointParams(coord_format="xy", remove_invisible=True),
        instance_binding=["masks", "bboxes", "keypoints"],
        strict=True,
    )

    result = transform(image=image, instances=instances)

    assert len(result["instances"]) == 1
    assert result["instances"][0]["bbox_labels"]["class_id"] == "cat"
    assert result["instances"][0]["mask"].shape == (40, 40)
    np.testing.assert_array_equal(result["instances"][0]["keypoints"], [[15.0, 15.0]])


def test_user_data_recipe_preserves_metadata() -> None:
    image = np.zeros((6, 8, 1), dtype=np.uint8)
    metadata = {"clip_id": "train-137", "start_seconds": 4.5}
    transform = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    result = transform(image=image, user_data=metadata)

    assert result["user_data"] is metadata


def test_replay_recipe_reuses_realized_parameters() -> None:
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8, 1)
    transform = A.ReplayCompose(
        [A.RandomRotate90(p=1.0), A.HorizontalFlip(p=0.5)],
        seed=137,
        strict=True,
    )

    first = transform(image=image)
    replayed = A.ReplayCompose.replay(first["replay"], image=image)

    np.testing.assert_array_equal(first["image"], replayed["image"])


def test_recipe_constructor_configuration_round_trip() -> None:
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8, 1)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        additional_targets={"paired": "image"},
        seed=137,
        strict=True,
    )

    restored = A.from_dict(A.to_dict(transform))
    expected = transform(image=image, paired=image.copy())
    actual = restored(image=image, paired=image.copy())

    np.testing.assert_array_equal(actual["image"], expected["image"])
    np.testing.assert_array_equal(actual["paired"], expected["paired"])


def test_selective_channel_recipe_preserves_unselected_channels() -> None:
    image = np.zeros((6, 8, 5), dtype=np.uint8)
    image[..., :3] = 10
    image[..., 3:] = 137
    transform = A.Compose(
        [
            A.SelectiveChannelTransform(
                [A.InvertImg(p=1.0)],
                channels=(0, 1, 2),
                p=1.0,
            ),
        ],
        strict=True,
    )

    result = transform(image=image)["image"]

    np.testing.assert_array_equal(result[..., :3], 245)
    np.testing.assert_array_equal(result[..., 3:], image[..., 3:])
