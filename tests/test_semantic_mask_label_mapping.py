import json
from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.transform_params import SampledParams, TargetSet


class _ApplyWithParamsTrackingFlip(A.HorizontalFlip):
    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.apply_with_params_calls = 0

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.apply_with_params_calls += 1
        return super().apply_with_params(params, *args, **kwargs)


def test_semantic_mask_label_mapping_swaps_labels_after_horizontal_flip() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    np.testing.assert_array_equal(result["mask"], np.array([[3, 2, 0, 3]], dtype=np.uint8))
    np.testing.assert_array_equal(mask, np.array([[2, 0, 3, 2]], dtype=np.uint8))


@pytest.mark.parametrize("flip_axes", [(0,), (1,), (2,), (0, 1, 2)])
def test_semantic_mask_label_mapping_swaps_mask3d_after_odd_flip3d(flip_axes: tuple[int, ...]) -> None:
    volume = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4, 1)
    mask3d = np.array(
        [
            [[2, 0, 3, 4], [4, 3, 0, 2]],
            [[3, 2, 4, 0], [0, 4, 2, 3]],
        ],
        dtype=np.uint8,
    )
    transform = A.Compose(
        [A.Flip3D(flip_axes=flip_axes, p=1.0)],
        additional_targets={"mask3d_alias": "mask3d"},
        semantic_mask_label_mappings={"Flip3D": {2: 3, 3: 2}},
        strict=True,
        telemetry=False,
    )

    result = transform(volume=volume, mask3d=mask3d, mask3d_alias=mask3d.copy())

    np.testing.assert_array_equal(result["volume"], np.flip(volume, axis=flip_axes))
    expected_mask3d = np.flip(mask3d, axis=flip_axes).copy()
    source_labels = expected_mask3d.copy()
    expected_mask3d[source_labels == 2] = 3
    expected_mask3d[source_labels == 3] = 2
    np.testing.assert_array_equal(result["mask3d"], expected_mask3d)
    np.testing.assert_array_equal(result["mask3d_alias"], expected_mask3d)


def test_semantic_mask_label_mapping_flip3d_does_not_remap_2d_masks() -> None:
    volume = np.zeros((1, 2, 3, 1), dtype=np.float32)
    mask = np.array([[2, 0, 3], [3, 2, 0]], dtype=np.uint8)
    masks = np.stack([mask, np.array([[0, 3, 2], [2, 0, 3]], dtype=np.uint8)])
    transform = A.Compose(
        [A.Flip3D(flip_axes=(2,), p=1.0)],
        semantic_mask_label_mappings={"Flip3D": {2: 3, 3: 2}},
        strict=True,
        telemetry=False,
    )

    result = transform(volume=volume, mask=mask, masks=masks)

    np.testing.assert_array_equal(result["mask"], mask)
    np.testing.assert_array_equal(result["masks"], masks)


def test_semantic_mask_label_mapping_flip3d_identity_does_not_remap_mask3d() -> None:
    volume = np.zeros((1, 2, 3, 1), dtype=np.float32)
    mask3d = np.array([[[2, 0, 3], [3, 2, 0]]], dtype=np.uint8)
    transform = A.Compose(
        [A.Flip3D(flip_axes=(), p=1.0)],
        semantic_mask_label_mappings={"Flip3D": {2: 3, 3: 2}},
        strict=True,
        telemetry=False,
    )

    result = transform(volume=volume, mask3d=mask3d)

    np.testing.assert_array_equal(result["mask3d"], mask3d)


@pytest.mark.parametrize("flip_axes", [(0, 1), (0, 2), (1, 2)])
def test_semantic_mask_label_mapping_flip3d_ignores_even_reflections(flip_axes: tuple[int, ...]) -> None:
    volume = np.zeros((2, 2, 3, 1), dtype=np.float32)
    mask3d = np.array(
        [
            [[2, 0, 3], [3, 2, 0]],
            [[0, 3, 2], [2, 0, 3]],
        ],
        dtype=np.uint8,
    )
    transform = A.Compose(
        [A.Flip3D(flip_axes=flip_axes, p=1.0)],
        semantic_mask_label_mappings={"Flip3D": {2: 3, 3: 2}},
        strict=True,
        telemetry=False,
    )

    result = transform(volume=volume, mask3d=mask3d)

    np.testing.assert_array_equal(result["mask3d"], np.flip(mask3d, axis=flip_axes))


@pytest.mark.parametrize(("index", "swap_labels"), [(0, False), (24, True)])
def test_semantic_mask_label_mapping_follows_realized_cubic_symmetry_operation(
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    swap_labels: bool,
) -> None:
    """Only rotoreflections rename class IDs after their mask3d geometry is transformed."""
    volume = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 1)
    mask3d = np.array(
        [
            [[2, 0, 3, 4], [4, 3, 0, 2], [2, 4, 3, 0], [0, 2, 4, 3]],
            [[3, 2, 4, 0], [0, 4, 2, 3], [4, 3, 0, 2], [2, 0, 3, 4]],
        ],
        dtype=np.uint8,
    )

    def fixed_index(
        self: A.CubicSymmetry,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: Any,
    ) -> SampledParams:
        del self, params, targets, sampling
        return SampledParams.shared_only({"index": index, "volume_shape": data["volume"].shape})

    monkeypatch.setattr(A.CubicSymmetry, "sample_parameters", fixed_index)
    result = A.Compose(
        [A.CubicSymmetry(p=1.0)],
        additional_targets={"mask3d_alias": "mask3d"},
        semantic_mask_label_mappings={"CubicSymmetry": {2: 3, 3: 2}},
        strict=True,
        telemetry=False,
    )(volume=volume, mask3d=mask3d, mask3d_alias=mask3d.copy())

    expected = A.CubicSymmetry().apply_to_mask3d(mask3d, index=index)
    if swap_labels:
        source_labels = expected.copy()
        expected[source_labels == 2] = 3
        expected[source_labels == 3] = 2
    np.testing.assert_array_equal(result["mask3d"], expected)
    np.testing.assert_array_equal(result["mask3d_alias"], expected)


def test_semantic_mask_label_mapping_preserves_custom_apply_with_params_hook() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    flip = _ApplyWithParamsTrackingFlip()
    transform = A.Compose(
        [flip],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    assert flip.apply_with_params_calls == 1
    np.testing.assert_array_equal(result["mask"], np.array([[3, 2, 0, 3]], dtype=np.uint8))


def test_semantic_mask_label_mapping_preserves_nested_compose_configuration() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    nested = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        telemetry=False,
    )
    transform = A.Compose(
        [nested],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 4, 4: 2}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    np.testing.assert_array_equal(result["mask"], np.array([[3, 2, 0, 3]], dtype=np.uint8))


def test_semantic_mask_label_mapping_propagates_to_unconfigured_nested_compose() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    transform = A.Compose(
        [A.Compose([A.HorizontalFlip(p=1.0)], telemetry=False)],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    np.testing.assert_array_equal(result["mask"], np.array([[3, 2, 0, 3]], dtype=np.uint8))


def test_semantic_mask_label_mapping_normalizes_string_target_labels() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {"2": "3", "3": "2"}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    assert transform.semantic_mask_label_mappings == {"HorizontalFlip": {2: 3, 3: 2}}
    np.testing.assert_array_equal(result["mask"], np.array([[3, 2, 0, 3]], dtype=np.uint8))


def test_semantic_mask_label_mapping_survives_json_replay() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    transform = A.ReplayCompose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
    )

    first = transform(image=image, mask=mask)
    replay = json.loads(json.dumps(first["replay"], allow_nan=False))
    replayed = A.ReplayCompose.replay(replay, image=image, mask=mask)

    np.testing.assert_array_equal(replayed["mask"], first["mask"])


def test_semantic_mask_label_mapping_survives_pipeline_json_roundtrip() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    original = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        telemetry=False,
    )

    transported = json.loads(json.dumps(A.to_dict(original), allow_nan=False))
    restored = A.from_dict(transported)

    assert isinstance(restored, A.Compose)
    assert restored.semantic_mask_label_mappings == {"HorizontalFlip": {2: 3, 3: 2}}
    np.testing.assert_array_equal(restored(image=image, mask=mask)["mask"], original(image=image, mask=mask)["mask"])


def test_compose_without_semantic_mapping_clears_reused_transform_mapping() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 0, 3, 2]], dtype=np.uint8)
    flip = A.HorizontalFlip(p=1.0)
    A.Compose(
        [flip],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        telemetry=False,
    )

    transform = A.Compose([flip], telemetry=False)
    result = transform(image=image, mask=mask)

    np.testing.assert_array_equal(result["mask"], np.array([[2, 3, 0, 2]], dtype=np.uint8))


def test_semantic_mask_label_mapping_one_way_preserves_unmapped_and_ignore_labels() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[2, 7, 255, 4]], dtype=np.uint8)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    np.testing.assert_array_equal(result["mask"], np.array([[4, 255, 7, 3]], dtype=np.uint8))


def test_semantic_mask_label_mapping_preserves_uint16_mask_dtype() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    mask = np.array([[300, 0, 400, 300]], dtype=np.uint16)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {300: 400, 400: 300}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    assert result["mask"].dtype == np.uint16
    np.testing.assert_array_equal(result["mask"], np.array([[400, 300, 0, 400]], dtype=np.uint16))


def test_semantic_mask_label_mapping_covers_masks_mask3d_and_aliases() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    first_mask = np.array([[2, 0, 3, 255]], dtype=np.uint8)
    second_mask = np.array([[3, 3, 2, 0]], dtype=np.uint8)
    masks = np.stack([first_mask, second_mask])
    mask3d = masks.copy()
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        additional_targets={
            "mask_alias": "mask",
            "masks_alias": "masks",
            "mask3d_alias": "mask3d",
        },
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        strict=True,
        telemetry=False,
    )

    result = transform(
        image=image,
        mask=first_mask,
        mask_alias=first_mask.copy(),
        masks=masks,
        masks_alias=masks.copy(),
        mask3d=mask3d,
        mask3d_alias=mask3d.copy(),
    )

    expected_mask = np.array([[255, 2, 0, 3]], dtype=np.uint8)
    expected_stacked = np.stack([expected_mask, np.array([[0, 3, 2, 2]], dtype=np.uint8)])
    np.testing.assert_array_equal(result["mask"], expected_mask)
    np.testing.assert_array_equal(result["mask_alias"], expected_mask)
    np.testing.assert_array_equal(result["masks"], expected_stacked)
    np.testing.assert_array_equal(result["masks_alias"], expected_stacked)
    np.testing.assert_array_equal(result["mask3d"], expected_stacked)
    np.testing.assert_array_equal(result["mask3d_alias"], expected_stacked)


@pytest.mark.parametrize(("image_target", "mask_target"), [("images", "masks"), ("volume", "mask3d")])
def test_semantic_mask_label_mapping_matches_image_batch_and_volume_routes(
    image_target: str,
    mask_target: str,
) -> None:
    first_mask = np.array([[2, 0, 3, 255]], dtype=np.uint8)
    second_mask = np.array([[3, 3, 2, 0]], dtype=np.uint8)
    masks = np.stack([first_mask, second_mask])
    images = np.zeros((2, 1, 4, 3), dtype=np.uint8)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        telemetry=False,
    )

    result = transform(**{image_target: images, mask_target: masks})

    expected_mask = np.array([[255, 2, 0, 3]], dtype=np.uint8)
    expected_stacked = np.stack([expected_mask, np.array([[0, 3, 2, 2]], dtype=np.uint8)])
    np.testing.assert_array_equal(result[mask_target], expected_stacked)


@pytest.mark.parametrize(
    ("group_element", "expected_label"),
    [
        ("e", 2),
        ("r90", 2),
        ("r180", 2),
        ("r270", 2),
        ("h", 3),
        ("v", 3),
        ("t", 3),
        ("hvt", 3),
    ],
)
def test_semantic_mask_label_mapping_follows_realized_d4_operation(
    group_element: str,
    expected_label: int,
) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[2, 0], [0, 0]], dtype=np.uint8)
    transform = A.Compose(
        [A.D4(p=1.0, group_element=group_element)],
        semantic_mask_label_mappings={
            "HorizontalFlip": {2: 3},
            "VerticalFlip": {2: 3},
            "Transpose": {2: 3},
        },
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    assert np.count_nonzero(result["mask"] == expected_label) == 1
    assert np.count_nonzero(result["mask"] == (3 if expected_label == 2 else 2)) == 0


@pytest.mark.parametrize("group_element", ["e", "r90", "r180", "r270"])
def test_semantic_mask_label_mapping_does_not_remap_d4_identity_or_rotations(group_element: str) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[2, 0], [0, 0]], dtype=np.uint8)
    transform = A.Compose(
        [A.D4(p=1.0, group_element=group_element)],
        semantic_mask_label_mappings={"D4": {2: 3}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    assert np.count_nonzero(result["mask"] == 2) == 1
    assert np.count_nonzero(result["mask"] == 3) == 0


@pytest.mark.pytorch
def test_semantic_mask_label_mapping_preserves_uint16_tensor_masks() -> None:
    image = torch.zeros((3, 2, 3), dtype=torch.uint8)
    mask = torch.tensor([[300, 0, 400], [400, 300, 0]], dtype=torch.uint16)
    other_mask = torch.tensor([[400, 300, 0], [0, 400, 300]], dtype=torch.uint16)
    stacked_masks = torch.stack([mask, other_mask])
    transform = A.Compose(
        [A.Transpose(p=1.0)],
        semantic_mask_label_mappings={"Transpose": {300: 400, 400: 300}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask, masks=stacked_masks, mask3d=stacked_masks)

    expected_mask = torch.tensor([[400, 300], [0, 400], [300, 0]], dtype=torch.uint16)
    expected_other_mask = torch.tensor([[300, 0], [400, 300], [0, 400]], dtype=torch.uint16)
    expected_stacked_masks = torch.stack([expected_mask, expected_other_mask])
    assert isinstance(result["mask"], torch.Tensor)
    assert isinstance(result["masks"], torch.Tensor)
    assert isinstance(result["mask3d"], torch.Tensor)
    assert result["mask"].dtype == torch.uint16
    assert result["masks"].dtype == torch.uint16
    assert result["mask3d"].dtype == torch.uint16
    torch.testing.assert_close(result["mask"], expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(result["masks"], expected_stacked_masks, rtol=0, atol=0)
    torch.testing.assert_close(result["mask3d"], expected_stacked_masks, rtol=0, atol=0)


@pytest.mark.pytorch
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_semantic_mask_label_mapping_preserves_tensor_mask_dtype(dtype: torch.dtype) -> None:
    image = torch.zeros((3, 2, 3), dtype=torch.uint8)
    mask = torch.tensor([[2, 0, 3], [3, 2, 0]], dtype=dtype)
    transform = A.Compose(
        [A.Transpose(p=1.0)],
        semantic_mask_label_mappings={"Transpose": {2: 3, 3: 2}},
        telemetry=False,
    )

    result = transform(image=image, mask=mask)

    expected = torch.tensor([[3, 2], [0, 3], [2, 0]], dtype=dtype)
    assert result["mask"].dtype == dtype
    torch.testing.assert_close(result["mask"], expected, rtol=0, atol=0)
