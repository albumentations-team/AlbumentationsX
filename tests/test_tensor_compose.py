"""CPU Tensor boundary tests for the Tensor-native Compose foundation."""

from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core import composition
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.utils import get_image_data, get_shape, get_volume_shape


class _TensorProbe(ImageOnlyTransform):
    """Private test transform that proves Compose carries Tensor values to dispatch."""

    _supports_cpu_tensor = True

    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.seen_shapes: list[tuple[int, ...]] = []

    def apply(self, image: torch.Tensor, **params: Any) -> torch.Tensor:
        self.seen_shapes.append(params["shape"])
        return image + 1

    def apply_to_images(self, images: torch.Tensor, **params: Any) -> torch.Tensor:
        self.seen_shapes.append(params["shape"])
        return images + 1

    def apply_to_volume(self, volume: torch.Tensor, **params: Any) -> torch.Tensor:
        self.seen_shapes.append(params["shape"])
        return volume + 1


class _NumpyOnlyProbe(ImageOnlyTransform):
    """Private probe that accepts a Tensor caller only through Compose's whole-pipeline NumPy route."""

    def __init__(self) -> None:
        super().__init__(p=1.0)

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        assert isinstance(image, np.ndarray)
        return image + 1


@pytest.mark.parametrize(
    ("target", "shape", "expected_shape"),
    [
        ("image", (3, 11, 13), (11, 13, 3)),
        ("images", (3, 5, 11, 13), (11, 13, 3)),
        ("volume", (3, 7, 11, 13), (11, 13, 3)),
    ],
)
def test_compose_preserves_cpu_tensor_representation_and_logical_shape(
    target: str,
    shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    probe = _TensorProbe()
    tensor = torch.zeros(shape, dtype=torch.uint8)

    result = A.Compose([probe], strict=True)(**{target: tensor})

    assert isinstance(result[target], torch.Tensor)
    assert result[target].shape == tensor.shape
    assert result[target].dtype == tensor.dtype
    torch.testing.assert_close(result[target], tensor + 1)
    assert probe.seen_shapes == [expected_shape]


def test_empty_compose_preserves_cpu_tensor_identity() -> None:
    tensor = torch.zeros((3, 11, 13), dtype=torch.float32)

    result = A.Compose([])(image=tensor)

    assert result["image"] is tensor


def test_noop_preserves_non_contiguous_cpu_tensor_without_copy() -> None:
    tensor = torch.arange(3 * 11 * 13, dtype=torch.float32).reshape(3, 11, 13)[:, :, ::2]
    assert not tensor.is_contiguous()

    result = A.Compose([A.NoOp(p=1.0)], strict=True)(image=tensor)

    assert result["image"] is tensor
    assert not result["image"].is_contiguous()


def test_tensor_additional_target_uses_channel_first_shape_contract() -> None:
    probe = _TensorProbe()
    image = torch.zeros((3, 11, 13), dtype=torch.uint8)
    image2 = torch.full((3, 11, 13), 7, dtype=torch.uint8)
    compose = A.Compose([probe], additional_targets={"image2": "image"}, strict=True)

    result = compose(image=image, image2=image2)

    torch.testing.assert_close(result["image"], image + 1)
    torch.testing.assert_close(result["image2"], image2 + 1)
    assert probe.seen_shapes == [(11, 13, 3), (11, 13, 3)]


def test_tensor_nested_compose_uses_declared_child_capability() -> None:
    tensor = torch.zeros((3, 11, 13), dtype=torch.float32)
    compose = A.Compose([A.Sequential([_TensorProbe()], p=1.0)], strict=True)

    result = compose(image=tensor)

    torch.testing.assert_close(result["image"], tensor + 1)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("channels", [1, 3])
def test_noop_preserves_tensor_image_sequences(dtype: torch.dtype, channels: int) -> None:
    images = torch.zeros((channels, 5, 11, 13), dtype=dtype)

    result = A.Compose([A.NoOp(p=1.0)], strict=True)(images=images)["images"]

    assert isinstance(result, torch.Tensor)
    assert result.data_ptr() == images.data_ptr()
    assert result.shape == images.shape
    assert result.dtype == images.dtype


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("images", np.zeros((2, 11, 13, 3), dtype=np.uint8)),
        ("masks", np.zeros((2, 11, 13, 3), dtype=np.uint8)),
        ("volume", np.zeros((2, 11, 13, 3), dtype=np.uint8)),
    ],
)
def test_noop_preserves_numpy_batch_target_without_copy(target: str, value: np.ndarray) -> None:
    result = A.Compose([A.NoOp(p=1.0)], strict=True)(**{target: value})[target]

    assert result is value


def test_nested_compose_preserves_tensor_annotations() -> None:
    image = torch.zeros((3, 11, 13), dtype=torch.uint8)
    bboxes = torch.tensor([[1, 1, 5, 4]], dtype=torch.float32)
    keypoints = torch.tensor([[2, 3]], dtype=torch.float32)
    compose = A.Compose(
        [A.Compose([A.NoOp(p=1.0)], strict=True)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        strict=True,
    )

    result = compose(image=image, bboxes=bboxes, keypoints=keypoints)

    for target, expected in (("image", image), ("bboxes", bboxes), ("keypoints", keypoints)):
        assert isinstance(result[target], torch.Tensor)
        torch.testing.assert_close(result[target], expected)


@pytest.mark.parametrize("terminal", [A.ToTensorV2(p=1.0), A.ToTensor3D(p=1.0)])
def test_tensor_rejects_legacy_terminal_transforms(terminal: A.BasicTransform) -> None:
    with pytest.raises(TypeError, match="remove ToTensorV2 or ToTensor3D"):
        A.Compose([terminal])(image=torch.zeros((3, 11, 13), dtype=torch.uint8))


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (torch.zeros((11, 13), dtype=torch.uint8), "must have shape"),
        (torch.zeros((3, 11, 13), dtype=torch.int64), "dtype one of"),
        (torch.zeros((3, 11, 13), dtype=torch.float32, requires_grad=True), "requires_grad=False"),
    ],
)
def test_tensor_boundary_rejects_invalid_image_contract(tensor: torch.Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        A.Compose([])(image=tensor)


@pytest.mark.parametrize(
    ("target", "tensor", "message"),
    [
        ("mask", torch.zeros((1, 11, 13), dtype=torch.uint8), "must have shape"),
        ("masks", torch.zeros((11, 13), dtype=torch.uint8), "must have shape"),
        ("bboxes", torch.zeros((1, 4), dtype=torch.int64), "dtype one of"),
        ("keypoints", torch.zeros((1, 2, 1), dtype=torch.float32), "must have shape"),
    ],
)
def test_tensor_boundary_rejects_invalid_annotation_contract(
    target: str,
    tensor: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        A.Compose([])(
            image=torch.zeros((3, 11, 13), dtype=torch.uint8),
            **{target: tensor},
        )


@pytest.mark.parametrize(
    ("target", "shape", "dtype"),
    [
        ("mask", (11, 13), torch.int64),
        ("masks", (2, 11, 13), torch.int64),
        ("mask3d", (2, 11, 13), torch.int64),
        ("mask", (11, 13), torch.int32),
        ("masks", (2, 11, 13), torch.int32),
        ("mask3d", (2, 11, 13), torch.int32),
        ("mask", (11, 13), torch.bool),
        ("masks", (2, 11, 13), torch.bool),
        ("mask3d", (2, 11, 13), torch.bool),
    ],
)
def test_tensor_boundary_rejects_unsupported_mask_dtypes(
    target: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    with pytest.raises(TypeError, match="dtype one of"):
        A.Compose([])(
            image=torch.zeros((3, 11, 13), dtype=torch.uint8),
            **{target: torch.zeros(shape, dtype=dtype)},
        )


def test_noop_preserves_tensor_spatial_targets_at_compose_boundary() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7)
    mask = torch.arange(5 * 7, dtype=torch.uint8).reshape(5, 7)
    masks = torch.arange(2 * 5 * 7, dtype=torch.uint8).reshape(2, 5, 7)
    bboxes = torch.tensor([[1.0, 1.0, 5.0, 4.0]], dtype=torch.float32)
    keypoints = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    compose = A.Compose(
        [A.NoOp(p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        strict=True,
    )

    result = compose(image=image, mask=mask, masks=masks, bboxes=bboxes, keypoints=keypoints)

    for target, expected in {
        "image": image,
        "mask": mask,
        "masks": masks,
        "bboxes": bboxes,
        "keypoints": keypoints,
    }.items():
        assert isinstance(result[target], torch.Tensor)
        torch.testing.assert_close(result[target], expected)


def test_noop_preserves_tensor_volume_and_mask3d_without_numpy_batch_dispatch() -> None:
    volume = torch.arange(3 * 5 * 7 * 9, dtype=torch.float32).reshape(3, 5, 7, 9)
    mask3d = torch.zeros((5, 7, 9), dtype=torch.uint8)

    result = A.Compose([A.NoOp(p=1.0)], strict=True)(volume=volume, mask3d=mask3d)

    assert result["volume"] is volume
    assert result["mask3d"] is mask3d


def test_flip3d_tensor_compose_preserves_volume_mask_and_semantic_mapping() -> None:
    volume = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5)
    mask3d = (torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5) % 2) + 2

    result = A.Compose(
        [A.Flip3D(flip_axes=(2,), p=1.0)],
        semantic_mask_label_mappings={"Flip3D": {2: 3, 3: 2}},
        strict=True,
    )(volume=volume, mask3d=mask3d)

    expected_mask3d = torch.flip(mask3d, dims=(2,))
    expected_mask3d = torch.where(expected_mask3d == 2, 3, 2).to(dtype=mask3d.dtype)
    assert isinstance(result["volume"], torch.Tensor)
    assert isinstance(result["mask3d"], torch.Tensor)
    assert result["volume"].dtype == volume.dtype
    assert result["mask3d"].dtype == mask3d.dtype
    torch.testing.assert_close(result["volume"], torch.flip(volume, dims=(3,)))
    torch.testing.assert_close(result["mask3d"], expected_mask3d)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("group_element", ["e", "r90", "r180", "r270", "v", "hvt", "h", "t"])
@pytest.mark.parametrize("strided", [False, True])
def test_square_symmetry_volume_tensor_compose_stays_on_direct_route(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    group_element: str,
    strided: bool,
) -> None:
    volume = torch.arange(2 * 3 * 5 * 5, dtype=dtype).reshape(2, 3, 5, 5)
    mask3d = torch.arange(3 * 5 * 5, dtype=torch.uint8).reshape(3, 5, 5)
    if strided:
        volume = volume.transpose(-2, -1)
        mask3d = mask3d.transpose(-2, -1)

    def fail_on_numpy_bridge(*args: object, **kwargs: object) -> None:
        raise AssertionError("SquareSymmetry volume Tensor route entered the NumPy bridge")

    monkeypatch.setattr(composition, "tensor_to_numpy_spatial", fail_on_numpy_bridge)
    result = A.Compose([A.SquareSymmetry(group_element=group_element, p=1.0)], strict=True)(
        volume=volume,
        mask3d=mask3d,
    )

    expected_volume = _d4_tensor_reference(volume, group_element)
    expected_mask3d = _d4_tensor_reference(mask3d, group_element)
    assert result["volume"].dtype == volume.dtype
    assert result["mask3d"].dtype == mask3d.dtype
    assert result["volume"].stride() == expected_volume.stride()
    assert result["mask3d"].stride() == expected_mask3d.stride()
    torch.testing.assert_close(result["volume"], expected_volume)
    torch.testing.assert_close(result["mask3d"], expected_mask3d)


def _d4_tensor_reference(value: torch.Tensor, group_element: str) -> torch.Tensor:
    match group_element:
        case "e":
            result = value
        case "r90":
            result = torch.rot90(value, 1, (-2, -1))
        case "r180":
            result = torch.rot90(value, 2, (-2, -1))
        case "r270":
            result = torch.rot90(value, 3, (-2, -1))
        case "v":
            result = torch.flip(value, (-2,))
        case "hvt":
            result = torch.transpose(torch.flip(value, (-2, -1)), -2, -1)
        case "h":
            result = torch.flip(value, (-1,))
        case "t":
            result = torch.transpose(value, -2, -1)
        case _:
            raise AssertionError(f"Unexpected D4 group element: {group_element}")
    return result


def test_tensor_compose_combines_3d_transforms() -> None:
    volume = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5)
    mask3d = (torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5) % 2) + 2
    transforms = [
        A.Flip3D(flip_axes=(2,), p=1.0),
        A.CenterCrop3D(size=(2, 3, 4), p=1.0),
    ]
    tensor_compose = A.Compose(
        transforms,
        semantic_mask_label_mappings={"Flip3D": {2: 3, 3: 2}},
        strict=True,
    )
    numpy_compose = A.Compose(
        transforms,
        semantic_mask_label_mappings={"Flip3D": {2: 3, 3: 2}},
        strict=True,
    )

    result = tensor_compose(volume=volume, mask3d=mask3d)
    expected = numpy_compose(
        volume=volume.permute(1, 2, 3, 0).numpy(),
        mask3d=mask3d.numpy(),
    )

    assert isinstance(result["volume"], torch.Tensor)
    assert isinstance(result["mask3d"], torch.Tensor)
    expected_volume = torch.from_numpy(np.ascontiguousarray(expected["volume"])).permute(3, 0, 1, 2)
    expected_mask3d = torch.from_numpy(np.ascontiguousarray(expected["mask3d"]))
    torch.testing.assert_close(result["volume"], expected_volume)
    torch.testing.assert_close(result["mask3d"], expected_mask3d)


def test_tensor_compose_bridges_numpy_2d_targets_and_annotations() -> None:
    image = torch.arange(5 * 5 * 7, dtype=torch.uint8).reshape(5, 5, 7)
    images = torch.arange(5 * 2 * 5 * 7, dtype=torch.uint8).reshape(5, 2, 5, 7)
    mask = torch.arange(5 * 7, dtype=torch.uint8).reshape(5, 7)
    masks = torch.arange(2 * 5 * 7, dtype=torch.uint8).reshape(2, 5, 7)
    bboxes = torch.tensor([[1, 1, 5, 4]], dtype=torch.float32)
    keypoints = torch.tensor([[2, 3]], dtype=torch.float32)
    compose_kwargs = {
        "bbox_params": A.BboxParams(coord_format="pascal_voc"),
        "keypoint_params": A.KeypointParams(coord_format="xy"),
        "strict": True,
    }
    tensor_result = A.Compose([A.CenterCrop(height=3, width=5, p=1.0)], **compose_kwargs)(
        image=image,
        images=images,
        mask=mask,
        masks=masks,
        bboxes=bboxes,
        keypoints=keypoints,
    )
    numpy_result = A.Compose([A.CenterCrop(height=3, width=5, p=1.0)], **compose_kwargs)(
        image=image.permute(1, 2, 0).numpy(),
        images=images.permute(1, 2, 3, 0).numpy(),
        mask=mask.numpy(),
        masks=masks.numpy(),
        bboxes=bboxes.numpy(),
        keypoints=keypoints.numpy(),
    )

    expected_image = torch.from_numpy(np.ascontiguousarray(numpy_result["image"])).permute(2, 0, 1)
    expected_images = torch.from_numpy(np.ascontiguousarray(numpy_result["images"])).permute(3, 0, 1, 2)
    torch.testing.assert_close(tensor_result["image"], expected_image)
    torch.testing.assert_close(tensor_result["images"], expected_images)
    for target in ("mask", "masks", "bboxes", "keypoints"):
        assert isinstance(tensor_result[target], torch.Tensor)
        expected = torch.from_numpy(np.ascontiguousarray(numpy_result[target])).to(dtype=tensor_result[target].dtype)
        torch.testing.assert_close(tensor_result[target], expected)


def test_tensor_compose_bridges_nested_numpy_transform() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    tensor_compose = A.Compose(
        [A.Sequential([A.CenterCrop(height=3, width=5, p=1.0)], p=1.0)],
        strict=True,
    )
    numpy_compose = A.Compose(
        [A.Sequential([A.CenterCrop(height=3, width=5, p=1.0)], p=1.0)],
        strict=True,
    )

    result = tensor_compose(image=image)["image"]
    expected = numpy_compose(image=image.permute(1, 2, 0).numpy())["image"]

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.from_numpy(np.ascontiguousarray(expected)).permute(2, 0, 1))


def test_tensor_compose_bridges_all_selectable_oneof_branches() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)

    result = A.Compose([A.OneOf([_NumpyOnlyProbe()], p=1.0)], strict=True)(image=image)["image"]

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, image + 1)


@pytest.mark.parametrize(
    ("image", "mask"),
    [
        (torch.zeros((3, 11, 13), dtype=torch.uint8), np.zeros((11, 13), dtype=np.uint8)),
        (np.zeros((11, 13, 3), dtype=np.uint8), torch.zeros((11, 13), dtype=torch.uint8)),
    ],
)
def test_tensor_boundary_rejects_mixed_spatial_representations(image: Any, mask: Any) -> None:
    with pytest.raises(TypeError, match="cannot be mixed across spatial targets"):
        A.Compose([])(image=image, mask=mask)


def test_tensor_boundary_rejects_numpy_bbox_with_tensor_image() -> None:
    with pytest.raises(TypeError, match="must also be Tensors"):
        A.Compose([])(
            image=torch.zeros((3, 11, 13), dtype=torch.uint8),
            bboxes=[[1.0, 1.0, 5.0, 4.0]],
        )


def test_tensor_shape_helpers_follow_channel_first_images_sequence_and_volume_contracts() -> None:
    image = torch.zeros((3, 11, 13), dtype=torch.uint8)
    images = torch.zeros((3, 5, 11, 13), dtype=torch.uint8)
    volume = torch.zeros((3, 7, 11, 13), dtype=torch.uint8)

    assert get_shape({"image": image}) == (11, 13)
    assert get_shape({"images": images}) == (11, 13)
    assert get_shape({"volume": volume}) == (11, 13)
    assert get_volume_shape({"volume": volume}) == (7, 11, 13)
    assert get_image_data({"image": image}) == {
        "dtype": torch.uint8,
        "height": 11,
        "width": 13,
        "num_channels": 3,
    }
    assert get_image_data({"images": images}) == {
        "dtype": torch.uint8,
        "height": 11,
        "width": 13,
        "num_channels": 3,
    }
    assert get_image_data({"volume": volume}) == {
        "dtype": torch.uint8,
        "height": 11,
        "width": 13,
        "num_channels": 3,
    }


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("channels", [1, 3])
def test_transpose_preserves_accepted_tensor_image_view_contract(dtype: torch.dtype, channels: int) -> None:
    image = torch.arange(channels * 5 * 7, dtype=torch.int64).reshape(channels, 5, 7).to(dtype)

    result = A.Compose([A.Transpose(p=1.0)], strict=True)(image=image)["image"]

    assert isinstance(result, torch.Tensor)
    assert result.shape == (channels, 7, 5)
    assert result.dtype == image.dtype
    assert result.data_ptr() == image.data_ptr()
    assert not result.is_contiguous()
    torch.testing.assert_close(result, image.mT)


def test_transpose_keeps_tensor_masks_bboxes_and_keypoints_aligned_with_tensor_image() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    numpy_image = image.permute(1, 2, 0).numpy().copy()
    mask = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)
    masks = np.arange(2 * 5 * 7, dtype=np.uint8).reshape(2, 5, 7)
    mask3d = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    bboxes = np.array([[1, 1, 5, 4]], dtype=np.float32)
    keypoints = np.array([[2, 3]], dtype=np.float32)
    bbox_params = A.BboxParams(coord_format="pascal_voc")
    keypoint_params = A.KeypointParams(coord_format="xy")

    tensor_result = A.Compose(
        [A.Transpose(p=1.0)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        strict=True,
    )(
        image=image,
        mask=torch.from_numpy(mask.copy()),
        masks=torch.from_numpy(masks.copy()),
        mask3d=torch.from_numpy(mask3d.copy()),
        bboxes=torch.from_numpy(bboxes.copy()),
        keypoints=torch.from_numpy(keypoints.copy()),
    )
    numpy_result = A.Compose(
        [A.Transpose(p=1.0)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        strict=True,
    )(
        image=numpy_image,
        mask=mask,
        masks=masks,
        mask3d=mask3d,
        bboxes=bboxes,
        keypoints=keypoints,
    )

    torch.testing.assert_close(
        tensor_result["image"],
        torch.from_numpy(numpy_result["image"]).permute(2, 0, 1),
    )
    for target in ("mask", "masks", "mask3d", "bboxes", "keypoints"):
        assert isinstance(tensor_result[target], torch.Tensor)
        expected = torch.from_numpy(numpy_result[target]).to(dtype=tensor_result[target].dtype)
        torch.testing.assert_close(tensor_result[target], expected)


@pytest.mark.parametrize(
    ("transform_cls", "dimensions"),
    [
        (A.HorizontalFlip, (-1,)),
        (A.VerticalFlip, (-2,)),
        (A.Transpose, None),
    ],
)
@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("image", (5, 3, 4)),
        ("images", (5, 2, 3, 4)),
        ("volume", (5, 2, 3, 4)),
        ("mask", (3, 4)),
        ("masks", (2, 3, 4)),
        ("mask3d", (2, 3, 4)),
    ],
)
def test_flip_transforms_support_all_tensor_spatial_targets(
    transform_cls: type[A.DualTransform],
    dimensions: tuple[int, ...] | None,
    target: str,
    shape: tuple[int, ...],
) -> None:
    value = torch.arange(int(np.prod(shape)), dtype=torch.uint8).reshape(shape)

    result = A.Compose([transform_cls(p=1.0)], strict=True)(**{target: value})[target]

    expected = value.transpose(-1, -2) if dimensions is None else torch.flip(value, dimensions)
    assert result.dtype == value.dtype
    torch.testing.assert_close(result, expected)
