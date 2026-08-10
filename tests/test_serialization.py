import copy
import inspect
import io
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from deepdiff import DeepDiff

import albumentations as A
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.core.serialization import SERIALIZABLE_REGISTRY, shorten_class_name
from albumentations.core.transforms_interface import ImageOnlyTransform
from tests.conftest import IMAGES, SQUARE_FLOAT_IMAGE, SQUARE_UINT8_IMAGE
from tests.helpers.contract_assertions import assert_contract_values_equal
from tests.helpers.transform_cases import (
    PRIMARY_TRANSFORM_CONTRACT_CASES,
    TRANSFORM_CONTRACT_CASES,
    TransformContractCase,
)

images = []


## Can use several seeds, but just too slow.
TEST_SEEDS = (137,)


def _make_case_pipeline(case: TransformContractCase, seed: int) -> A.Compose:
    transform = case.transform_cls(p=1.0, **copy.deepcopy(dict(case.init_kwargs)))
    return A.Compose(
        [transform],
        seed=seed,
        **copy.deepcopy(dict(case.primary_compose_kwargs)),
    )


def _assert_case_roundtrip(
    case: TransformContractCase,
    original: A.Compose,
    restored: A.Compose,
    seed: int,
) -> None:
    original.set_random_seed(seed)
    restored.set_random_seed(seed)
    original_data = case.make_data(np.random.default_rng(seed))
    restored_data = case.make_data(np.random.default_rng(seed))
    assert_contract_values_equal(original(**original_data), restored(**restored_data), "result")


@pytest.mark.parametrize("case", TRANSFORM_CONTRACT_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_case_dict_serialization(case: TransformContractCase, seed: int) -> None:
    original = _make_case_pipeline(case, seed)
    restored = A.from_dict(A.to_dict(original))

    assert isinstance(restored, A.Compose)
    _assert_case_roundtrip(case, original, restored, seed)


@pytest.mark.parametrize("case", TRANSFORM_CONTRACT_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("data_format", ("yaml", "json"))
def test_transform_case_file_serialization(
    case: TransformContractCase,
    seed: int,
    data_format: str,
) -> None:
    buffer = StringIO()
    original = _make_case_pipeline(case, seed)
    A.save(original, buffer, data_format=data_format)
    buffer.seek(0)
    restored = A.load(buffer, data_format=data_format)

    assert isinstance(restored, A.Compose)
    _assert_case_roundtrip(case, original, restored, seed)


_DUAL_SERIALIZATION_CASES = tuple(
    case
    for case in PRIMARY_TRANSFORM_CONTRACT_CASES
    if issubclass(case.transform_cls, A.DualTransform)
    and not issubclass(case.transform_cls, A.Transform3D)
    and case.transform_cls
    not in {
        A.CoarseDropout,
        A.CropNonEmptyMaskIfExists,
        A.OverlayElements,
        A.TextImage,
        A.Mosaic,
        A.CopyAndPaste,
    }
)


@pytest.mark.parametrize("case", _DUAL_SERIALIZATION_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_augmentations_for_bboxes_serialization(
    case: TransformContractCase,
    p,
    seed,
    albumentations_bboxes,
):
    augmentation_cls = case.transform_cls
    image = SQUARE_FLOAT_IMAGE if augmentation_cls == A.FromFloat else SQUARE_UINT8_IMAGE
    transform = augmentation_cls(p=p, **copy.deepcopy(dict(case.init_kwargs)))
    aug = A.Compose(
        [transform],
        bbox_params={"coord_format": "pascal_voc"},
    )
    aug.set_random_seed(seed)
    data = case.make_data(np.random.default_rng(seed))
    data.update(image=image, bboxes=albumentations_bboxes)
    data.pop("bbox_labels", None)
    if "mask" in data:
        data["mask"] = np.zeros((*image.shape[:2], 1), dtype=np.uint8)
        data["mask"][:20, :20] = 1

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    deserialized_aug.set_random_seed(seed)
    aug_data = aug(**data)
    deserialized_aug_data = deserialized_aug(**data)
    np.testing.assert_array_equal(aug_data["image"], deserialized_aug_data["image"])
    np.testing.assert_array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])


_KEYPOINT_SERIALIZATION_CASES = tuple(
    case
    for case in _DUAL_SERIALIZATION_CASES
    if case.transform_cls
    not in {
        A.CropNonEmptyMaskIfExists,
        A.RandomSizedBBoxSafeCrop,
        A.BBoxSafeRandomCrop,
    }
)


@pytest.mark.parametrize("case", _KEYPOINT_SERIALIZATION_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_augmentations_for_keypoints_serialization(
    case: TransformContractCase,
    p,
    seed,
    keypoints,
):
    augmentation_cls = case.transform_cls
    image = SQUARE_FLOAT_IMAGE if augmentation_cls == A.FromFloat else SQUARE_UINT8_IMAGE
    aug = augmentation_cls(p=p, **copy.deepcopy(dict(case.init_kwargs)))
    aug.set_random_seed(seed)
    data = case.make_data(np.random.default_rng(seed))
    data.update(image=image, keypoints=keypoints)
    if "mask" in data:
        data["mask"] = np.zeros((*image.shape[:2], 1), dtype=np.uint8)
        data["mask"][:20, :20] = 1

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    deserialized_aug.set_random_seed(seed)
    aug_data = aug(**data)
    deserialized_aug_data = deserialized_aug(**data)
    np.testing.assert_array_equal(aug_data["image"], deserialized_aug_data["image"])
    np.testing.assert_array_equal(
        aug_data["keypoints"],
        deserialized_aug_data["keypoints"],
    )


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("image", IMAGES)
def test_transform_pipeline_serialization(seed, image):
    mask = image.copy()
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.Resize(1024, 1024),
                        A.RandomSizedCrop(
                            min_max_height=(256, 1024),
                            size=(512, 512),
                            p=1,
                        ),
                        A.OneOf(
                            [
                                A.RandomSizedCrop(
                                    min_max_height=(256, 512),
                                    size=(384, 384),
                                    p=0.5,
                                ),
                                A.RandomSizedCrop(
                                    min_max_height=(256, 512),
                                    size=(512, 512),
                                    p=0.5,
                                ),
                            ],
                        ),
                    ],
                ),
                A.Compose(
                    [
                        A.Resize(1024, 1024),
                        A.RandomSizedCrop(
                            min_max_height=(256, 1025),
                            size=(256, 256),
                            p=1,
                        ),
                        A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
                    ],
                ),
            ),
            A.SomeOf(
                [
                    A.HorizontalFlip(p=1),
                    A.D4(p=1),
                    A.HueSaturationValue(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                2,
                replace=False,
            ),
        ],
    )
    aug.set_random_seed(seed)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    deserialized_aug.set_random_seed(seed)
    aug_data = aug(image=image, mask=mask)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    np.testing.assert_array_equal(aug_data["image"], deserialized_aug_data["image"])
    np.testing.assert_array_equal(aug_data["mask"], deserialized_aug_data["mask"])


@pytest.mark.parametrize(
    ["bboxes", "bbox_format", "labels"],
    [
        ([(20, 30, 40, 50)], "coco", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "coco", [1, 2]),
        ([(20, 30, 60, 80)], "pascal_voc", [2]),
        ([(20, 30, 60, 80, 99)], "pascal_voc", [1]),
        ([(0.2, 0.3, 0.4, 0.5)], "yolo", [2]),
        ([(0.2, 0.3, 0.4, 0.5, 99)], "yolo", [1]),
    ],
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("image", IMAGES)
def test_transform_pipeline_serialization_with_bboxes(
    seed,
    image,
    bboxes,
    bbox_format,
    labels,
):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.RandomRotate90(),
                        A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
                    ],
                ),
                A.Compose(
                    [
                        A.Rotate(p=0.5),
                        A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
                    ],
                ),
            ),
            A.SomeOf(
                [
                    A.HorizontalFlip(p=1),
                    A.D4(p=1),
                    A.HueSaturationValue(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                n=5,
            ),
        ],
        bbox_params={"coord_format": bbox_format, "label_fields": ["labels"]},
    )
    aug.set_random_seed(seed)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    deserialized_aug.set_random_seed(seed)
    aug_data = aug(image=image, bboxes=bboxes, labels=labels)
    deserialized_aug_data = deserialized_aug(image=image, bboxes=bboxes, labels=labels)
    np.testing.assert_array_equal(aug_data["image"], deserialized_aug_data["image"])
    np.testing.assert_array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format", "labels"],
    [
        ([(20, 30, 40, 50)], "xyas", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "xy", [1, 2]),
        ([(20, 30, 60, 80)], "yx", [2]),
        ([(20, 30, 60, 80, 99)], "xys", [1]),
    ],
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("image", IMAGES)
def test_transform_pipeline_serialization_with_keypoints(
    seed,
    image,
    keypoints,
    keypoint_format,
    labels,
):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.RandomRotate90(),
                        A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
                    ],
                ),
                A.Compose(
                    [
                        A.Rotate(p=0.5),
                        A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
                    ],
                ),
            ),
            A.SomeOf(
                n=2,
                transforms=[
                    A.HorizontalFlip(p=1),
                    A.Transpose(p=1),
                    A.HueSaturationValue(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                replace=False,
            ),
        ],
        keypoint_params={"coord_format": keypoint_format, "label_fields": ["labels"]},
        seed=seed,
    )

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    deserialized_aug.set_random_seed(seed)

    aug_data = aug(image=image, keypoints=keypoints, labels=labels)
    deserialized_aug_data = deserialized_aug(
        image=image,
        keypoints=keypoints,
        labels=labels,
    )

    np.testing.assert_array_equal(aug_data["image"], deserialized_aug_data["image"])
    np.testing.assert_array_equal(
        aug_data["keypoints"],
        deserialized_aug_data["keypoints"],
    )


_IMAGE_ONLY_SERIALIZATION_CASES = tuple(
    case
    for case in PRIMARY_TRANSFORM_CONTRACT_CASES
    if issubclass(case.transform_cls, A.ImageOnlyTransform) and case.transform_cls is not A.TextImage
)


@pytest.mark.parametrize("case", _IMAGE_ONLY_SERIALIZATION_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_additional_targets_for_image_only_serialization(
    case: TransformContractCase,
    seed,
):
    augmentation_cls = case.transform_cls
    data = case.make_data(np.random.default_rng(seed))
    image = data["image"]
    aug = A.Compose(
        [augmentation_cls(p=1.0, **copy.deepcopy(dict(case.init_kwargs)))],
        additional_targets={"image2": "image"},
        seed=seed,
        strict=True,
    )

    data["image2"] = image.copy()

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    deserialized_aug.set_random_seed(seed)

    aug_data = aug(**data)
    deserialized_aug_data = deserialized_aug(**data)

    np.testing.assert_array_equal(aug_data["image"], deserialized_aug_data["image"])
    np.testing.assert_array_equal(aug_data["image2"], deserialized_aug_data["image2"])


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("p", [1])
@pytest.mark.parametrize("image", IMAGES)
def test_lambda_serialization(image, albumentations_bboxes, keypoints, seed, p):
    def vflip_image(image, **kwargs):
        return fgeometric.vflip(image)

    def vflip_mask(mask, **kwargs):
        return fgeometric.vflip(mask)

    def vflip_bbox(bboxes, **kwargs):
        return fgeometric.bboxes_vflip(bboxes, bbox_type=kwargs["bbox_type"])

    def vflip_keypoint(keypoints, **kwargs):
        return fgeometric.keypoints_vflip(keypoints, kwargs["shape"][0])

    mask = image.copy()

    aug = A.Compose(
        [
            A.Lambda(
                name="vflip",
                image=vflip_image,
                mask=vflip_mask,
                bboxes=vflip_bbox,
                keypoints=vflip_keypoint,
                p=p,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="albumentations"),
        keypoint_params=A.KeypointParams(coord_format="xyas"),
    )
    aug.set_random_seed(seed)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug, nonserializable={"vflip": aug.transforms[0]})
    deserialized_aug.set_random_seed(seed)
    aug_data = aug(
        image=image,
        mask=mask,
        bboxes=albumentations_bboxes,
        keypoints=keypoints,
    )
    deserialized_aug_data = deserialized_aug(
        image=image,
        mask=mask,
        bboxes=albumentations_bboxes,
        keypoints=keypoints,
    )
    np.testing.assert_array_equal(aug_data["image"], deserialized_aug_data["image"])
    np.testing.assert_array_equal(aug_data["mask"], deserialized_aug_data["mask"])
    np.testing.assert_array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])
    np.testing.assert_array_equal(
        aug_data["keypoints"],
        deserialized_aug_data["keypoints"],
    )


@pytest.mark.parametrize(
    "transform_file_name",
    [
        "transform_serialization_v2_without_totensor.json",
    ],
)
@pytest.mark.parametrize("data_format", ("yaml", "json"))
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_serialization_conversion_without_totensor(
    transform_file_name,
    data_format,
    seed,
):
    image = SQUARE_UINT8_IMAGE

    # Step 1: Load transform from file
    current_directory = Path(__file__).resolve().parent
    files_directory = current_directory / "files"
    transform_file_path = files_directory / transform_file_name
    transform = A.load(transform_file_path, data_format="json")
    transform.set_random_seed(seed)
    # Step 2: Serialize it to buffer in memory
    buffer = io.StringIO()
    A.save(transform, buffer, data_format=data_format)
    buffer.seek(0)  # Reset buffer position to the beginning

    # Step 3: Load transform from this memory buffer
    transform_from_buffer = A.load(buffer, data_format=data_format)
    transform_from_buffer.set_random_seed(seed)
    # Ensure the buffer is closed after use
    buffer.close()

    assert (
        DeepDiff(
            transform.to_dict(),
            transform_from_buffer.to_dict(),
            ignore_type_in_groups=[(tuple, list)],
        )
        == {}
    ), (
        f"The loaded transform is not equal to the original one {DeepDiff(transform.to_dict(), transform_from_buffer.to_dict(), ignore_type_in_groups=[(tuple, list)])}"
    )

    image1 = transform(image=image)["image"]
    image2 = transform_from_buffer(image=image)["image"]

    np.testing.assert_array_equal(
        image1,
        image2,
        err_msg=f"The transformed images are not equal {(image1 - image2).mean()}",
    )


@pytest.mark.pytorch
@pytest.mark.parametrize(
    "transform_file_name",
    [
        "transform_serialization_v2_with_totensor.json",
    ],
)
@pytest.mark.parametrize("data_format", ("yaml", "json"))
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_serialization_conversion_with_totensor(
    transform_file_name: str,
    data_format: str,
    seed: int,
) -> None:
    image = SQUARE_UINT8_IMAGE

    # Load transform from file
    current_directory = Path(__file__).resolve().parent
    files_directory = current_directory / "files"
    transform_file_path = files_directory / transform_file_name

    transform = A.load(transform_file_path, data_format="json")
    transform.set_random_seed(seed)

    # Serialize it to buffer in memory
    buffer = io.StringIO()
    A.save(transform, buffer, data_format=data_format)
    buffer.seek(0)  # Reset buffer position to the beginning

    # Load transform from this memory buffer
    transform_from_buffer = A.load(buffer, data_format=data_format)
    transform_from_buffer.set_random_seed(seed)
    buffer.close()  # Ensure the buffer is closed after use

    assert (
        DeepDiff(
            transform.to_dict(),
            transform_from_buffer.to_dict(),
            ignore_type_in_groups=[(tuple, list)],
        )
        == {}
    ), (
        f"The loaded transform is not equal to the original one {DeepDiff(transform.to_dict(), transform_from_buffer.to_dict(), ignore_type_in_groups=[(tuple, list)])}"
    )

    image1 = transform(image=image)["image"]
    image2 = transform_from_buffer(image=image)["image"]

    (
        np.testing.assert_array_equal(image1, image2),
        f"The transformed images are not equal {(image1 - image2).mean()}",
    )


def test_custom_transform_with_overlapping_name():
    class HorizontalFlip(ImageOnlyTransform):
        pass

    assert SERIALIZABLE_REGISTRY["HorizontalFlip"] == A.HorizontalFlip
    assert SERIALIZABLE_REGISTRY["tests.test_serialization.HorizontalFlip"] == HorizontalFlip


def test_serialization_v2_to_dict() -> None:
    transform = A.Compose([A.HorizontalFlip()])
    transform_dict = A.to_dict(transform)["transform"]
    assert transform_dict == {
        "__class_fullname__": "Compose",
        "p": 1.0,
        "transforms": [{"__class_fullname__": "HorizontalFlip", "p": 0.5}],
        "bbox_params": None,
        "keypoint_params": None,
        "additional_targets": {},
        "semantic_mask_label_mappings": None,
        "is_check_shapes": True,
        "strict": False,
        "mask_interpolation": None,
        "seed": None,
        "save_applied_params": False,
        "telemetry": True,
        "instance_binding": None,
        "strict_instance_invariant": True,
    }


def test_compose_roundtrip_preserves_all_behavioral_constructor_state() -> None:
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            filter_invalid_bboxes=True,
        ),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        additional_targets={"image2": "image"},
        p=0.75,
        is_check_shapes=False,
        strict=True,
        mask_interpolation=cv2.INTER_LINEAR,
        seed=137,
        save_applied_params=True,
        telemetry=False,
        strict_instance_invariant=False,
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
    )

    restored = A.from_dict(A.to_dict(transform))

    assert isinstance(restored, A.Compose)
    assert restored.p == 0.75
    assert restored.is_check_shapes is False
    assert restored.strict is True
    assert restored.mask_interpolation == cv2.INTER_LINEAR
    assert restored.seed == 137
    assert restored.save_applied_params is True
    assert restored.telemetry is False
    assert restored._strict_instance_invariant is False
    assert restored.additional_targets == {"image2": "image"}
    assert restored.semantic_mask_label_mappings == {"HorizontalFlip": {2: 3, 3: 2}}
    assert restored.processors["bboxes"].params.filter_invalid_bboxes is True


@pytest.mark.parametrize("data_format", ("json", "yaml"))
@pytest.mark.parametrize(
    "marker_shaped_mapping",
    (
        {
            "__albumentations_serialized_type__": "mapping",
            "items": {"left": "right"},
        },
        {
            "__albumentations_serialized_type__": "mapping",
            "items": [["left", "right"]],
            "__albumentations_serialized_mapping_version__": 1,
        },
    ),
)
def test_mapping_codec_preserves_marker_shaped_label_mappings(
    data_format: str,
    marker_shaped_mapping: dict[str, object],
) -> None:
    pipeline = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        keypoint_params=A.KeypointParams(
            coord_format="xy",
            label_fields=["label"],
            label_mapping={"HorizontalFlip": {"label": marker_shaped_mapping}},
        ),
    )

    restored_from_dict = A.from_dict(A.to_dict(pipeline))
    assert isinstance(restored_from_dict, A.Compose)
    assert restored_from_dict.processors["keypoints"].params.label_mapping == {
        "HorizontalFlip": {"label": marker_shaped_mapping},
    }

    buffer = StringIO()
    A.save(pipeline, buffer, data_format=data_format)
    buffer.seek(0)
    restored_from_file = A.load(buffer, data_format=data_format)

    assert isinstance(restored_from_file, A.Compose)
    assert restored_from_file.processors["keypoints"].params.label_mapping == {
        "HorizontalFlip": {"label": marker_shaped_mapping},
    }


def test_from_dict_preserves_raw_mapping_escape_prefix_keys() -> None:
    escape_prefix = "__albumentations_escaped_mapping_key__"
    additional_targets = {f"{escape_prefix}image2": "image"}
    pipeline = A.Compose([A.NoOp(p=1.0)], additional_targets=additional_targets)
    payload = A.to_dict(pipeline)
    canonical_restored = A.from_dict(payload)
    payload["transform"]["additional_targets"] = additional_targets

    restored = A.from_dict(payload)

    assert isinstance(canonical_restored, A.Compose)
    assert canonical_restored.additional_targets == additional_targets
    assert isinstance(restored, A.Compose)
    assert restored.additional_targets == additional_targets


def test_compose_roundtrip_preserves_mask_validation_and_applied_output_behavior() -> None:
    transform = A.Compose(
        [
            A.Rotate(
                angle_range=(45, 45),
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=1.0,
            ),
        ],
        strict=True,
        mask_interpolation=cv2.INTER_LINEAR,
        save_applied_params=True,
        seed=137,
    )
    restored = A.from_dict(A.to_dict(transform))
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255

    expected = transform(image=image, mask=mask)
    actual = restored(image=image, mask=mask)

    np.testing.assert_array_equal(actual["mask"], expected["mask"])
    assert set(actual["mask"].ravel()) - {0, 255}
    assert actual["applied_transforms"] == expected["applied_transforms"]
    for pipeline in (transform, restored):
        with pytest.raises(ValueError, match="Key unexpected is not in available keys"):
            pipeline(image=image, unexpected=np.zeros((5, 5), dtype=np.uint8))


def test_composition_subclasses_preserve_class_policy_in_roundtrip_and_operators() -> None:
    selective = A.SelectiveChannelTransform(
        [A.InvertImg(p=1.0)],
        channels=[1],
        p=0.8,
    )
    some_of = A.SomeOf(
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
        n=2,
        replace=True,
        p=0.9,
    )
    replay = A.ReplayCompose(
        [selective, some_of],
        p=0.75,
        save_key="replay_state",
        strict=True,
        mask_interpolation=cv2.INTER_LINEAR,
        save_applied_params=True,
        telemetry=False,
        strict_instance_invariant=False,
        seed=137,
    )

    restored = A.from_dict(A.to_dict(replay))
    extended = replay + A.VerticalFlip(p=1.0)

    assert isinstance(restored, A.ReplayCompose)
    assert restored.save_key == "replay_state"
    assert restored.strict is True
    assert restored.mask_interpolation == cv2.INTER_LINEAR
    assert restored.save_applied_params is True
    assert restored.telemetry is False
    assert restored._strict_instance_invariant is False
    assert restored.transforms[0].channels == [1]
    assert restored.transforms[1].n == 2
    assert restored.transforms[1].replace is True
    assert isinstance(extended, A.ReplayCompose)
    assert extended.strict is True
    assert extended.transforms[0].channels == [1]


@pytest.mark.parametrize(
    ["class_fullname", "expected_short_class_name"],
    [
        ["albumentations.augmentations.transforms.HorizontalFlip", "HorizontalFlip"],
        ["HorizontalFlip", "HorizontalFlip"],
        ["some_module.HorizontalFlip", "some_module.HorizontalFlip"],
    ],
)
def test_shorten_class_name(class_fullname, expected_short_class_name):
    assert shorten_class_name(class_fullname) == expected_short_class_name


@pytest.mark.parametrize("case", TRANSFORM_CONTRACT_CASES, ids=lambda case: case.case_id)
def test_serialized_fields_match_public_constructor(case: TransformContractCase) -> None:
    instance = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)))
    signature = inspect.signature(case.transform_cls.__init__)
    public_fields = {
        name
        for name, parameter in signature.parameters.items()
        if name not in {"self", "strict"}
        and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    }
    serialized_fields = set(instance.to_dict()["transform"]) - {"__class_fullname__"}

    assert serialized_fields <= public_fields | {"p"}


def test_serialization_includes_compose_strict_but_excludes_transform_strict() -> None:
    transform = A.Compose([A.HorizontalFlip()])
    transform_dict = A.to_dict(transform)["transform"]
    assert transform_dict["strict"] is False
    assert "strict" not in transform_dict["transforms"][0]

    transform = A.HorizontalFlip(strict=True)
    transform_dict = A.to_dict(transform)["transform"]
    assert "strict" not in transform_dict


def test_serialization_from_float() -> None:
    dtype = "uint8"
    max_value = 137
    transform = A.FromFloat(dtype=dtype, max_value=max_value)
    transform_dict = A.to_dict(transform)["transform"]
    assert transform_dict["dtype"] == dtype
    assert transform_dict["max_value"] == max_value
