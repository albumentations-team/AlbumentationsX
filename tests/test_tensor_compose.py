"""Public CPU Tensor routing contracts for :class:`albumentations.Compose`."""

from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core import transforms_interface
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetSet
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.utils import get_image_data, get_shape, get_volume_shape


class _TensorProbe(ImageOnlyTransform):
    """A complete leaf-local Tensor route used to verify native dispatch."""

    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.seen: list[tuple[type[Any], tuple[int, ...]]] = []

    def apply(self, image: torch.Tensor, **params: Any) -> torch.Tensor:
        self.seen.append((type(image), params["shape"]))
        return image + 1

    def apply_to_images(self, images: torch.Tensor, **params: Any) -> torch.Tensor:
        self.seen.append((type(images), params["shape"]))
        return images + 1

    def apply_to_volume(self, volume: torch.Tensor, **params: Any) -> torch.Tensor:
        self.seen.append((type(volume), params["shape"]))
        return volume + 1


class _NumpyOnlyProbe(ImageOnlyTransform):
    """A user-style transform that gains Tensor support through the base fallback."""

    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.seen: list[tuple[type[Any], tuple[int, ...]]] = []

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        self.seen.append((type(image), image.shape))
        return image + 1


class _NegativeStrideProbe(ImageOnlyTransform):
    def __init__(self) -> None:
        super().__init__(p=1.0)

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        return image[:, ::-1]


class _TargetsAsParamsProbe(ImageOnlyTransform):
    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.seen: np.ndarray | None = None

    @property
    def targets_as_params(self) -> list[str]:
        return ["offset"]

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        self.seen = data["offset"]
        return SampledParams(params={"offset": int(self.seen[0])})

    def apply(self, image: np.ndarray, offset: int, **params: Any) -> np.ndarray:
        return image + offset


class _TensorSubclass(torch.Tensor):
    """A deliberately unsupported Tensor subclass for boundary validation."""


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("image", torch.zeros((11, 13), dtype=torch.float32)),
        ("image", torch.zeros((3, 11, 13), dtype=torch.uint8)),
        ("images", torch.zeros((2, 3, 11, 13), dtype=torch.float32)),
        ("volume", torch.zeros((3, 5, 11, 13), dtype=torch.uint8)),
        ("mask", torch.zeros((11, 13), dtype=torch.int16)),
        ("mask", torch.zeros((2, 11, 13), dtype=torch.float32)),
        ("masks", torch.zeros((2, 11, 13), dtype=torch.uint8)),
        ("masks", torch.zeros((2, 3, 11, 13), dtype=torch.int16)),
        ("mask3d", torch.zeros((5, 11, 13), dtype=torch.uint8)),
        ("mask3d", torch.zeros((2, 5, 11, 13), dtype=torch.float32)),
    ],
)
def test_noop_preserves_supported_tensor_object(target: str, value: torch.Tensor) -> None:
    result = A.Compose([A.NoOp(p=1.0)], strict=True)(**{target: value})[target]

    assert result is value


@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("mask", (11, 13)),
        ("masks", (2, 11, 13)),
        ("mask3d", (5, 11, 13)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.float32])
def test_tensor_masks_accept_all_declared_public_dtypes(
    target: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    value = torch.zeros(shape, dtype=dtype)

    assert A.Compose([])(**{target: value})[target] is value


@pytest.mark.parametrize(
    "dtype",
    [torch.uint8, torch.int16, torch.float32],
)
@pytest.mark.parametrize("target", ["mask", "masks", "mask3d"])
def test_tensor_mask_crop_fallback_preserves_every_accepted_dtype_exactly(target: str, dtype: torch.dtype) -> None:
    if target == "mask":
        value = torch.arange(25).reshape(5, 5).to(dtype)
        result = A.Compose([A.CenterCrop(height=3, width=3, p=1.0)], strict=True)(
            image=torch.zeros((1, 5, 5), dtype=torch.uint8),
            mask=value,
        )[target]
        expected = value[1:4, 1:4]
    elif target == "masks":
        value = torch.arange(2 * 25).reshape(2, 5, 5).to(dtype)
        result = A.Compose([A.CenterCrop(height=3, width=3, p=1.0)], strict=True)(
            image=torch.zeros((1, 5, 5), dtype=torch.uint8),
            masks=value,
        )[target]
        expected = value[:, 1:4, 1:4]
    else:
        value = torch.arange(5 * 5 * 5).reshape(5, 5, 5).to(dtype)
        result = A.Compose([A.CenterCrop3D(size=(3, 3, 3), p=1.0)], strict=True)(
            volume=torch.zeros((1, 5, 5, 5), dtype=torch.uint8),
            mask3d=value,
        )[target]
        expected = value[1:4, 1:4, 1:4]

    assert result.dtype == dtype
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("images", torch.zeros((3, 11, 13), dtype=torch.uint8), "must have shape"),
        ("volume", torch.zeros((5, 11, 13), dtype=torch.uint8), "must have shape"),
        ("image", torch.zeros((3, 11, 13), dtype=torch.int64), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.bool), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.int8), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.int32), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.int64), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.uint16), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.uint32), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.uint64), "dtype one of"),
        ("mask", torch.zeros((11, 13), dtype=torch.float64), "dtype one of"),
        ("bboxes", torch.zeros((1, 4), dtype=torch.int64), "dtype one of"),
        ("keypoints", torch.zeros((1, 2, 1), dtype=torch.float32), "must have shape"),
        ("image", torch.zeros((3, 11, 13), dtype=torch.float32, requires_grad=True), "requires_grad=False"),
        ("image", torch.zeros((3, 11, 13), dtype=torch.uint8).to_sparse(), "torch.strided"),
        ("image", torch.zeros((3, 11, 13), dtype=torch.uint8).as_subclass(_TensorSubclass), "plain torch.Tensor"),
    ],
)
def test_tensor_boundary_rejects_unsupported_inputs_before_execution(
    target: str,
    value: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        A.Compose([_NumpyOnlyProbe()], strict=True)(**{target: value})


@pytest.mark.parametrize(
    "pipeline",
    [
        lambda: A.Compose([A.ToTensorV2(p=1.0)], strict=True),
        lambda: A.Compose([A.Sequential([A.ToTensorV2(p=1.0)], p=1.0)], strict=True),
        lambda: A.Compose([A.OneOf([A.ToTensor3D(p=1.0)], p=1.0)], strict=True),
    ],
)
def test_tensor_input_rejects_tensor_terminal_anywhere_in_the_graph(pipeline: Any) -> None:
    with pytest.raises(TypeError, match="accept NumPy input only"):
        pipeline()(image=torch.zeros((3, 11, 13), dtype=torch.uint8))


@pytest.mark.parametrize("terminal", [A.ToTensorV2(p=1.0), A.ToTensor3D(p=1.0)])
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_tensor_terminal_rejects_direct_tensor_calls(terminal: A.BasicTransform, p: float) -> None:
    terminal.p = p

    with pytest.raises(TypeError, match="accept NumPy input only"):
        terminal(image=torch.zeros((3, 11, 13), dtype=torch.uint8))


@pytest.mark.parametrize(
    ("target", "shape", "expected_shape"),
    [
        ("image", (3, 11, 13), (11, 13, 3)),
        ("images", (5, 3, 11, 13), (11, 13, 3)),
        ("volume", (3, 7, 11, 13), (11, 13, 3)),
    ],
)
def test_tensor_aware_leaf_receives_canonical_tensor_layout(
    target: str,
    shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    probe = _TensorProbe()
    value = torch.zeros(shape, dtype=torch.uint8)

    result = A.Compose([probe], strict=True)(**{target: value})[target]

    torch.testing.assert_close(result, value + 1)
    assert probe.seen == [(torch.Tensor, expected_shape)]


def test_numpy_pipeline_never_enters_tensor_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("NumPy input entered a Tensor adapter")

    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", fail)
    monkeypatch.setattr(transforms_interface, "numpy_to_tensor_spatial", fail)
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    result = A.Compose([A.CenterCrop(height=3, width=5, p=1.0), A.HorizontalFlip(p=1.0)], strict=True)(image=image)

    np.testing.assert_array_equal(result["image"], image[1:4, 1:6][:, ::-1])


def test_leaf_does_not_bridge_tensor_targets_it_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("pass-through target entered a Tensor adapter")

    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", fail)
    monkeypatch.setattr(transforms_interface, "numpy_to_tensor_spatial", fail)
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    mask = torch.zeros((5, 7), dtype=torch.uint8)

    result = A.Compose([A.RandomBrightnessContrast(p=1.0)], strict=True)(image=image, mask=mask)

    assert isinstance(result["image"], np.ndarray)
    assert result["mask"] is mask


def test_numpy_only_user_transform_uses_a_leaf_local_tensor_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    conversions: list[str] = []
    to_numpy = transforms_interface.tensor_to_numpy_spatial
    to_tensor = transforms_interface.numpy_to_tensor_spatial

    def track_to_numpy(value: torch.Tensor, target: str) -> np.ndarray:
        conversions.append(f"to_numpy:{target}")
        return to_numpy(value, target)

    def track_to_tensor(value: np.ndarray, target: str) -> torch.Tensor:
        conversions.append(f"to_tensor:{target}")
        return to_tensor(value, target)

    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", track_to_numpy)
    monkeypatch.setattr(transforms_interface, "numpy_to_tensor_spatial", track_to_tensor)
    first, second = _NumpyOnlyProbe(), _NumpyOnlyProbe()
    image = torch.zeros((3, 5, 7), dtype=torch.uint8)

    result = A.Compose([first, second], strict=True)(image=image)["image"]

    torch.testing.assert_close(result, image + 2)
    assert first.seen == [(np.ndarray, (5, 7, 3))]
    assert second.seen == [(np.ndarray, (5, 7, 3))]
    assert conversions == ["to_numpy:image", "to_tensor:image", "to_numpy:image", "to_tensor:image"]


def test_lambda_tensor_input_uses_its_numpy_handler() -> None:
    seen: list[type[Any]] = []

    def add_one(image: np.ndarray, **params: Any) -> np.ndarray:
        seen.append(type(image))
        return image + 1

    image = torch.zeros((3, 5, 7), dtype=torch.uint8)
    result = A.Compose([A.Lambda(image=add_one, p=1.0)], strict=True)(image=image)["image"]

    assert seen == [np.ndarray]
    torch.testing.assert_close(result, image + 1)


def test_tensor_fallback_accepts_noncontiguous_input() -> None:
    image = torch.arange(3 * 7 * 5, dtype=torch.uint8).reshape(3, 7, 5).transpose(1, 2)

    result = A.Compose([A.CenterCrop(height=3, width=5, p=1.0)], strict=True)(image=image)["image"]

    assert not image.is_contiguous()
    torch.testing.assert_close(result, image[:, 1:4, 1:6])


def test_tensor_fallback_materializes_negative_stride_numpy_output() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)

    result = A.Compose([_NegativeStrideProbe()], strict=True)(image=image)["image"]

    torch.testing.assert_close(result, torch.flip(image, (-1,)))


@pytest.mark.parametrize(
    "transform",
    [
        A.Compose([], strict=True),
        A.Compose([A.CenterCrop(height=3, width=5, p=1.0)], p=0.0, strict=True),
        A.Compose([A.CenterCrop(height=3, width=5, p=0.0)], strict=True),
    ],
)
def test_unapplied_tensor_pipeline_preserves_object_identity(transform: A.Compose) -> None:
    image = torch.zeros((3, 5, 7), dtype=torch.uint8)

    assert transform(image=image)["image"] is image


def test_mixed_target_containers_are_independent_through_numpy_fallback() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    mask = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)

    result = A.Compose([A.CenterCrop(height=3, width=5, p=1.0)], strict=True)(image=image, mask=mask)

    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["mask"], np.ndarray)
    torch.testing.assert_close(result["image"], image[:, 1:4, 1:6])
    np.testing.assert_array_equal(result["mask"], mask[1:4, 1:6])


def test_tensor_fallback_preserves_optional_mask_rank_and_annotation_containers() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    mask = torch.arange(5 * 7, dtype=torch.int16).reshape(5, 7)
    masks = torch.arange(2 * 5 * 7, dtype=torch.uint8).reshape(2, 5, 7)
    bboxes = torch.tensor([[1.0, 1.0, 5.0, 4.0]], dtype=torch.float32)
    keypoints = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    transform = A.Compose(
        [A.CenterCrop(height=3, width=5, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        strict=True,
    )

    result = transform(image=image, mask=mask, masks=masks, bboxes=bboxes, keypoints=keypoints)

    torch.testing.assert_close(result["image"], image[:, 1:4, 1:6])
    torch.testing.assert_close(result["mask"], mask[1:4, 1:6])
    torch.testing.assert_close(result["masks"], masks[:, 1:4, 1:6])
    torch.testing.assert_close(result["bboxes"], torch.tensor([[0.0, 0.0, 4.0, 3.0]]))
    torch.testing.assert_close(result["keypoints"], torch.tensor([[1.0, 2.0]]))
    assert result["mask"].ndim == 2
    assert result["masks"].ndim == 3


def test_channel_changing_transform_keeps_a_tensor_channel_axis() -> None:
    image = torch.arange(5 * 7, dtype=torch.uint8).reshape(5, 7)

    result = A.Compose([A.ToRGB(p=1.0)], strict=True)(image=image)["image"]

    assert result.shape == (3, 5, 7)
    torch.testing.assert_close(result[0], image)
    torch.testing.assert_close(result[1], image)
    torch.testing.assert_close(result[2], image)


@pytest.mark.parametrize(
    ("transform_cls", "expected"),
    [
        (A.HorizontalFlip, lambda value: torch.flip(value, (-1,))),
        (A.VerticalFlip, lambda value: torch.flip(value, (-2,))),
        (A.Transpose, lambda value: value.transpose(-1, -2)),
    ],
)
@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("image", (3, 5, 7)),
        ("images", (2, 3, 5, 7)),
        ("volume", (2, 4, 5, 7)),
        ("mask", (5, 7)),
        ("masks", (2, 5, 7)),
        ("mask3d", (4, 5, 7)),
    ],
)
def test_existing_flip_tensor_paths_stay_native(
    monkeypatch: pytest.MonkeyPatch,
    transform_cls: type[A.DualTransform],
    expected: Any,
    target: str,
    shape: tuple[int, ...],
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("native Tensor flip entered the NumPy fallback")

    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", fail)
    value = torch.arange(int(np.prod(shape)), dtype=torch.uint8).reshape(shape)

    result = A.Compose([transform_cls(p=1.0)], strict=True)(**{target: value})[target]

    torch.testing.assert_close(result, expected(value))


@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("images", (2, 3, 5, 7)),
        ("volume", (2, 4, 5, 7)),
        ("mask3d", (4, 5, 7)),
    ],
)
def test_d4_tensor_handlers_stay_native(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    shape: tuple[int, ...],
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("native Tensor D4 handler entered the NumPy fallback")

    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", fail)
    value = torch.arange(int(np.prod(shape)), dtype=torch.uint8).reshape(shape)

    result = A.Compose([A.D4(group_element="r90", p=1.0)], strict=True)(**{target: value})[target]

    torch.testing.assert_close(result, value.rot90(1, (-2, -1)))


def test_tensor_3d_pipeline_keeps_native_and_fallback_routes_composable() -> None:
    volume = torch.arange(2 * 4 * 5 * 7, dtype=torch.float32).reshape(2, 4, 5, 7)
    mask3d = torch.arange(4 * 5 * 7, dtype=torch.int16).reshape(4, 5, 7)

    result = A.Compose([A.Flip3D(flip_axes=(2,), p=1.0), A.CenterCrop3D(size=(2, 3, 5), p=1.0)], strict=True)(
        volume=volume,
        mask3d=mask3d,
    )

    torch.testing.assert_close(result["volume"], torch.flip(volume, (-1,))[:, 1:3, 1:4, 1:6])
    torch.testing.assert_close(result["mask3d"], torch.flip(mask3d, (-1,))[1:3, 1:4, 1:6])
    assert result["mask3d"].dtype == torch.int16


@pytest.mark.parametrize(
    ("image", "channels"),
    [
        (
            torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5),
            (1,),
        ),
        (
            torch.arange(4 * 5, dtype=torch.uint8).reshape(4, 5),
            (0,),
        ),
    ],
)
def test_selective_channel_transform_uses_tensor_numpy_fallback(
    image: torch.Tensor,
    channels: tuple[int, ...],
) -> None:
    source = image.clone()
    result = A.Compose(
        [A.SelectiveChannelTransform([A.InvertImg(p=1.0)], channels=channels, p=1.0)],
        strict=True,
    )(image=image)["image"]

    assert isinstance(result, torch.Tensor)
    expected = 255 - source if source.ndim == 2 else source.clone()
    if source.ndim == 3:
        expected[list(channels)] = 255 - expected[list(channels)]
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(image, source)


def test_selective_channel_transform_trace_snapshot_preserves_tensor_representation() -> None:
    image = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5)
    traced = A.Compose(
        [A.SelectiveChannelTransform([A.InvertImg(p=1.0)], channels=(1,), p=1.0)],
        strict=True,
    ).run_with_trace(image=image, options=A.TraceOptions(snapshot_targets=("image",)))

    snapshot = traced.records[0].snapshot["image"]
    assert isinstance(snapshot, torch.Tensor)
    expected = image.clone()
    expected[1] = 255 - expected[1]
    torch.testing.assert_close(snapshot, expected)


@pytest.mark.parametrize(
    "transform",
    [
        A.Affine3D(p=1.0),
        A.Anisotropy3D(axes=(0,), num_axes_range=(1, 1), downscale_factor_range=(2.0, 2.0), p=1.0),
        A.Resize3D(size=(2, 3, 5), p=1.0),
    ],
)
def test_existing_3d_tensor_backends_stay_native(
    monkeypatch: pytest.MonkeyPatch,
    transform: A.BasicTransform,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("native Tensor 3D transform entered the NumPy fallback")

    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", fail)
    volume = torch.arange(2 * 4 * 5 * 7, dtype=torch.float32).reshape(2, 4, 5, 7)

    result = A.Compose([transform], strict=True)(volume=volume)["volume"]

    assert isinstance(result, torch.Tensor)


def test_nested_and_selectable_compositions_use_leaf_fallback() -> None:
    image = torch.zeros((3, 5, 7), dtype=torch.uint8)
    nested = A.Compose([A.Sequential([_NumpyOnlyProbe()], p=1.0)], strict=True)
    selected = A.Compose([A.OneOf([_NumpyOnlyProbe()], p=1.0)], strict=True)
    sampled = A.Compose([A.SomeOf([_NumpyOnlyProbe()], n=1, p=1.0)], strict=True)

    torch.testing.assert_close(nested(image=image)["image"], image + 1)
    torch.testing.assert_close(selected(image=image)["image"], image + 1)
    torch.testing.assert_close(sampled(image=image)["image"], image + 1)


def test_direct_container_uses_tensor_fallback_without_a_compose_root() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    transform = A.OneOf([A.CenterCrop(height=3, width=5, p=1.0)], p=1.0)

    result = transform(image=image)["image"]

    torch.testing.assert_close(result, image[:, 1:4, 1:6])


def test_direct_leaf_rejects_optional_channel_tensor_rank() -> None:
    with pytest.raises(TypeError, match="pass optional-channel inputs through Compose"):
        A.HorizontalFlip(p=1.0)(image=torch.zeros((5, 7), dtype=torch.uint8))


def test_replay_compose_replays_tensor_leaf_fallback() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    transform = A.ReplayCompose([A.CenterCrop(height=3, width=5, p=1.0)], strict=True)

    first = transform(image=image)
    replayed = A.ReplayCompose.replay(first["replay"], image=image)

    torch.testing.assert_close(first["image"], image[:, 1:4, 1:6])
    torch.testing.assert_close(replayed["image"], first["image"])


def test_targets_as_params_adapts_tensor_reference_images_and_preserves_metadata_object() -> None:
    image = torch.zeros((3, 5, 7), dtype=torch.uint8)
    references = [torch.full((3, 5, 7), 255, dtype=torch.uint8)]
    transform = A.Compose([A.HistogramMatching(blend_ratio=(1.0, 1.0), metadata_key="refs", p=1.0)], strict=True)

    result = transform(image=image, refs=references)

    torch.testing.assert_close(result["image"], references[0])
    assert result["refs"] is references
    assert result["refs"][0] is references[0]


def test_tensor_reference_metadata_accepts_optional_channel_rank() -> None:
    image = torch.zeros((1, 5, 7), dtype=torch.uint8)
    references = [torch.full((5, 7), 255, dtype=torch.uint8)]
    transform = A.Compose([A.HistogramMatching(blend_ratio=(1.0, 1.0), metadata_key="refs", p=1.0)], strict=True)

    result = transform(image=image, refs=references)

    torch.testing.assert_close(result["image"], references[0].unsqueeze(0))
    assert result["refs"] is references


def test_targets_as_params_is_the_only_declaration_needed_for_raw_tensor_metadata() -> None:
    image = torch.zeros((1, 5, 7), dtype=torch.uint8)
    offset = torch.tensor([3], dtype=torch.int64)
    probe = _TargetsAsParamsProbe()

    result = A.Compose([probe], strict=True)(image=image, offset=offset)

    assert isinstance(probe.seen, np.ndarray)
    assert probe.seen.dtype == np.int64
    torch.testing.assert_close(result["image"], image + 3)
    assert result["offset"] is offset


def test_tensor_overlay_metadata_uses_known_target_fields_only() -> None:
    image = torch.zeros((3, 5, 7), dtype=torch.uint8)
    overlay = torch.full((3, 2, 3), 255, dtype=torch.uint8)
    nested_image = torch.zeros((3, 1, 1), dtype=torch.float64)
    bbox = torch.tensor([0.0, 0.0, 3 / 7, 2 / 5], dtype=torch.float32)
    metadata = [{"image": overlay, "bbox": bbox, "note": {"image": nested_image}}]

    result = A.Compose([A.OverlayElements(metadata_key="overlays", p=1.0)], strict=True)(
        image=image,
        overlays=metadata,
    )

    assert torch.count_nonzero(result["image"] == 255) == overlay.numel()
    assert result["overlays"] is metadata
    assert result["overlays"][0]["image"] is overlay
    assert result["overlays"][0]["bbox"] is bbox
    assert result["overlays"][0]["note"]["image"] is nested_image


def test_tensor_fallback_does_not_mutate_overlay_mask_input() -> None:
    image = torch.zeros((3, 5, 7), dtype=torch.uint8)
    mask = torch.zeros((1, 5, 7), dtype=torch.int16)
    metadata = [
        {
            "image": torch.full((3, 2, 3), 255, dtype=torch.uint8),
            "mask": torch.ones((2, 3), dtype=torch.uint8),
            "mask_id": 9,
            "bbox": [0.0, 0.0, 3 / 7, 2 / 5],
        },
    ]

    result = A.Compose([A.OverlayElements(metadata_key="overlays", p=1.0)], strict=True)(
        image=image,
        mask=mask,
        overlays=metadata,
    )

    assert torch.equal(mask, torch.zeros_like(mask))
    assert torch.count_nonzero(result["mask"] == 9) == 6


def test_tensor_copy_paste_metadata_adapts_semantic_mask_alias() -> None:
    image = torch.zeros((3, 8, 10), dtype=torch.uint8)
    mask = torch.zeros((8, 10), dtype=torch.uint8)
    donor_mask = torch.ones((3, 4), dtype=torch.uint8)
    metadata = [
        {
            "image": torch.full((3, 3, 4), 255, dtype=torch.uint8),
            "mask": donor_mask,
            "semantic_mask": torch.full((3, 4), 7, dtype=torch.uint8),
        },
    ]

    result = A.Compose([A.CopyAndPaste(p=1.0)], seed=137, strict=True)(
        image=image,
        mask=mask,
        copy_paste_metadata=metadata,
    )

    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["mask"], torch.Tensor)
    assert torch.count_nonzero(result["mask"] == 7) == donor_mask.numel()
    assert result["copy_paste_metadata"] is metadata


def test_tensor_crop_metadata_uses_targets_as_params_without_a_tensor_declaration() -> None:
    image = torch.arange(3 * 8 * 10, dtype=torch.uint8).reshape(3, 8, 10)
    cropping_bbox = torch.tensor([2, 2, 8, 6], dtype=torch.int64)
    transform = A.Compose(
        [A.RandomCropNearBBox(max_part_shift=(0.0, 0.0), cropping_bbox_key="crop", p=1.0)],
        strict=True,
    )

    result = transform(image=image, crop=cropping_bbox)

    assert result["image"].shape == (3, 4, 6)
    assert result["crop"] is cropping_bbox


def test_invalid_targets_as_params_tensor_fails_before_transform_probability() -> None:
    transform = A.Compose([A.HistogramMatching(metadata_key="refs", p=0.0)], strict=True)
    refs = [torch.zeros((3, 5, 7), dtype=torch.float64)]

    with pytest.raises(TypeError, match=r"refs\[0\].*dtype one of"):
        transform(image=np.zeros((5, 7, 3), dtype=np.uint8), refs=refs)


def test_tensor_shape_helpers_use_nchw_and_cdhw_contracts() -> None:
    image = torch.zeros((3, 11, 13), dtype=torch.uint8)
    images = torch.zeros((5, 3, 11, 13), dtype=torch.uint8)
    volume = torch.zeros((3, 7, 11, 13), dtype=torch.uint8)

    assert get_shape({"image": image}) == (11, 13)
    assert get_shape({"images": images}) == (11, 13)
    assert get_shape({"volume": volume}) == (11, 13)
    assert get_volume_shape({"volume": volume}) == (7, 11, 13)
    assert get_image_data({"images": images}) == {
        "dtype": torch.uint8,
        "height": 11,
        "width": 13,
        "num_channels": 3,
    }
