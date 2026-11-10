"""Public target layouts and CPU Tensor conversion for vector fields."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.tensor import numpy_to_tensor_spatial, tensor_to_numpy_spatial
from albumentations.core.transform_params import TargetSet
from albumentations.core.type_definitions import VECTOR_FIELD_TARGET_INFO
from tests.helpers.contract_assertions import assert_contract_values_equal

_VECTOR_FIELD_TARGET_CASES = tuple(
    (target.name.lower(), *layout) for target, layout in VECTOR_FIELD_TARGET_INFO.items()
)


@pytest.mark.parametrize(("target", "spatial_rank", "components"), _VECTOR_FIELD_TARGET_CASES)
@pytest.mark.parametrize("dtype", (np.float16, np.float32))
def test_vector_field_numpy_tensor_roundtrip(
    target: str,
    spatial_rank: int,
    components: int,
    dtype: type[np.float16 | np.float32],
) -> None:
    spatial_shape = (7, 9) if spatial_rank == 2 else (3, 7, 9)
    source = np.arange(np.prod((*spatial_shape, components)), dtype=np.float32)
    source = source.reshape(*spatial_shape, components).astype(dtype)
    source[0, 0, ...] = np.nan

    tensor = numpy_to_tensor_spatial(source, target)
    result = tensor_to_numpy_spatial(tensor, target)

    expected_shape = (components, *spatial_shape)
    assert tuple(tensor.shape) == expected_shape
    assert result.dtype == source.dtype
    assert np.shares_memory(result, source)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(source))
    valid = np.isfinite(source)
    np.testing.assert_array_equal(result[valid], source[valid])


@pytest.mark.parametrize(("target", "spatial_rank", "components"), _VECTOR_FIELD_TARGET_CASES)
def test_vector_field_descriptor_uses_spatial_axes_without_image_scaling(
    target: str,
    spatial_rank: int,
    components: int,
) -> None:
    spatial_shape = (5, 9) if spatial_rank == 2 else (3, 5, 9)
    field = np.zeros((*spatial_shape, components), dtype=np.float32)
    source = {"image": np.zeros((5, 9, 1), dtype=np.float32), target: field}
    targets = TargetSet.from_data(source, {name: name for name in source})

    descriptor = targets.by_name(target).descriptor
    assert descriptor.spatial_shape == spatial_shape
    assert descriptor.channels == components
    assert descriptor.value_scale is None
    assert targets.image_like()[0].name == "image"
    assert targets.require_aligned_spatial_shape(2) == (5, 9)
    if spatial_rank == 3:
        source["volume"] = np.zeros((3, 5, 9, 1), dtype=np.float32)
        source.pop("image")
        targets = TargetSet.from_data(source, {name: name for name in source})
        assert targets.require_aligned_spatial_shape(3) == (3, 5, 9)


@pytest.mark.parametrize(("target", "spatial_rank", "components"), _VECTOR_FIELD_TARGET_CASES)
def test_empty_compose_preserves_field_targets_and_mixed_containers(
    target: str,
    spatial_rank: int,
    components: int,
) -> None:
    spatial_shape = (5, 9) if spatial_rank == 2 else (3, 5, 9)
    field = np.zeros((*spatial_shape, components), dtype=np.float32)
    image = torch.zeros((1, 5, 9), dtype=torch.float32)
    main_key = "volume" if spatial_rank == 3 else "image"
    primary = np.zeros((3, 5, 9, 1), dtype=np.float32) if spatial_rank == 3 else image
    pipeline = A.Compose([], strict=True)

    result = pipeline(**{main_key: primary, target: field})

    assert result[main_key] is primary
    assert result[target] is field
    assert isinstance(result[main_key], type(primary))
    assert isinstance(result[target], np.ndarray)


@pytest.mark.parametrize(("target", "spatial_rank", "components"), _VECTOR_FIELD_TARGET_CASES)
def test_empty_compose_accepts_additional_vector_field_aliases(
    target: str,
    spatial_rank: int,
    components: int,
) -> None:
    spatial_shape = (5, 9) if spatial_rank == 2 else (3, 5, 9)
    field = np.zeros((*spatial_shape, components), dtype=np.float16)
    primary = np.zeros((3, 5, 9, 1), dtype=np.float16) if spatial_rank == 3 else np.zeros((5, 9, 1), dtype=np.float16)
    alias = f"{target}_extra"
    pipeline = A.Compose([], additional_targets={alias: target}, strict=True)

    result = pipeline(**{"volume" if spatial_rank == 3 else "image": primary, target: field, alias: field.copy()})

    assert result[target] is field
    np.testing.assert_array_equal(result[alias], field)


@pytest.mark.parametrize(
    "transform",
    (
        A.CopyAndPaste(p=1.0),
        A.Mosaic(p=1.0),
        A.OverlayElements(p=1.0),
        A.Morphological(p=1.0),
        A.PixelSpread(p=1.0),
    ),
    ids=("copy_and_paste", "mosaic", "overlay", "morphological", "pixel_spread"),
)
def test_unsupported_mix_transforms_skip_before_sampling_when_fields_are_active(transform: Any) -> None:
    image = np.arange(5 * 7, dtype=np.float32).reshape(5, 7, 1)
    field = np.full((5, 7, 2), (2.0, -1.0), dtype=np.float32)

    direct = transform(image=image, gradient=field)
    forced = transform(force_apply=True, image=image, gradient=field)

    assert direct["image"] is image
    assert direct["gradient"] is field
    assert forced["image"] is image
    assert forced["gradient"] is field


def test_unsupported_transform_skips_vector_field_aliases_and_trace_reports_the_reason() -> None:
    image = np.zeros((5, 7, 1), dtype=np.float32)
    field = np.ones((5, 7, 2), dtype=np.float32)
    pipeline = A.Compose(
        [A.Mosaic(p=1.0)],
        additional_targets={"gradient_aux": "gradient"},
        strict=True,
        seed=137,
    )

    traced = pipeline.run_with_trace(image=image, gradient=field, gradient_aux=field.copy())

    assert traced.data["image"] is image
    assert traced.data["gradient"] is field
    np.testing.assert_array_equal(traced.data["gradient_aux"], field)
    leaf = next(record for record in traced.records if record.node_kind == "leaf")
    assert leaf.status == "skipped_unsupported_target"


def test_direct_transform_validates_vector_field_aliases() -> None:
    image = np.zeros((5, 7, 1), dtype=np.float32)
    field = np.empty((5, 7, 2), dtype=np.float32)
    field[...] = (2.0, 4.0)
    transform = A.HorizontalFlip(p=1.0)
    transform.add_targets({"gradient_aux": "gradient"})

    result = transform(image=image, gradient=field, gradient_aux=field.copy())

    np.testing.assert_array_equal(result["gradient"], np.broadcast_to((-2.0, 4.0), field.shape))
    np.testing.assert_array_equal(result["gradient_aux"], result["gradient"])
    with pytest.raises(TypeError, match="float16 or float32"):
        transform(image=image, gradient=field.astype(np.float64))


def test_unsupported_transform_does_not_inspect_donor_metadata_when_field_is_active() -> None:
    class UnreadableDonorMetadata:
        def __iter__(self) -> Any:
            raise AssertionError("unsupported node inspected donor metadata")

    image = np.zeros((5, 7, 1), dtype=np.float32)
    field = np.ones((5, 7, 2), dtype=np.float32)
    pipeline = A.Compose([A.Mosaic(p=1.0)], strict=True, seed=137)

    result = pipeline(image=image, gradient=field, mosaic_metadata=UnreadableDonorMetadata())

    assert result["image"] is image
    assert result["gradient"] is field


def test_unsupported_transform_remains_skipped_during_replay_with_active_field() -> None:
    image = np.zeros((5, 7, 1), dtype=np.float32)
    field = np.ones((5, 7, 2), dtype=np.float32)
    donor = np.ones_like(image)
    pipeline = A.ReplayCompose(
        [A.Mosaic(grid_yx=(1, 2), target_size=(5, 7), cell_shape=(5, 4), p=1.0)],
        strict=True,
        seed=137,
    )

    first = pipeline(image=image, mosaic_metadata=[{"image": donor}])
    assert first["replay"]["transforms"][0]["applied"]
    replayed = A.ReplayCompose.replay(first["replay"], image=image, gradient=field)

    assert replayed["image"] is image
    assert replayed["gradient"] is field


def test_selected_unsupported_child_is_a_noop_without_reselecting_oneof() -> None:
    image = np.arange(5 * 7, dtype=np.float32).reshape(5, 7, 1)
    field = np.ones((5, 7, 2), dtype=np.float32)
    pipeline = A.Compose(
        [A.OneOf([A.Mosaic(p=1.0), A.HorizontalFlip(p=1.0)], p=1.0)],
        strict=True,
        seed=139,
    )

    traced = pipeline.run_with_trace(image=image, gradient=field)

    assert traced.data["image"] is image
    assert traced.data["gradient"] is field
    leaf_records = [record for record in traced.records if record.node_kind == "leaf"]
    selected_records = [record for record in leaf_records if record.status != "skipped_selection"]
    assert len(selected_records) == 1
    assert selected_records[0].class_fullname == "Mosaic"
    assert selected_records[0].status == "skipped_unsupported_target"


@pytest.mark.parametrize(
    "transform",
    (
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.4, 0.4),
            hole_width_range=(0.4, 0.4),
            p=1.0,
        ),
        A.PixelDropout(dropout_prob=1.0, p=1.0),
    ),
    ids=("coarse_dropout", "pixel_dropout"),
)
def test_dropout_preserves_vector_fields_unchanged(transform: Any) -> None:
    image = np.ones((5, 7, 1), dtype=np.float32)
    field = np.full((5, 7, 2), (2.0, -1.0), dtype=np.float16)

    result = A.Compose([transform])(image=image, vector_field=field)

    assert result["vector_field"] is field


def test_to_tensor_v2_moves_field_components_to_the_channel_axis() -> None:
    image = np.zeros((5, 7, 1), dtype=np.float32)
    gradient = np.full((5, 7, 2), (2.0, -1.0), dtype=np.float16)
    gradient[0, 0, 0] = np.nan

    result = A.Compose([A.ToTensorV2()])(image=image, gradient=gradient)

    assert result["gradient"].shape == (2, 5, 7)
    assert result["gradient"].dtype == torch.float16
    assert torch.isnan(result["gradient"][0, 0, 0])
    assert result["gradient"][1, 0, 0] == -1


def test_to_tensor3d_moves_field_components_to_the_channel_axis() -> None:
    volume = np.zeros((3, 5, 7, 1), dtype=np.float32)
    normal = np.full((3, 5, 7, 3), (0.0, 0.0, 1.0), dtype=np.float32)

    result = A.Compose([A.ToTensor3D()])(volume=volume, normal3d=normal)

    assert result["normal3d"].shape == (3, 3, 5, 7)
    assert result["normal3d"].dtype == torch.float32
    torch.testing.assert_close(result["normal3d"][:, 1, 2, 3], torch.tensor([0.0, 0.0, 1.0]))


def test_vector_fields_do_not_change_raster_rng_or_applied_configuration() -> None:
    image = np.arange(9 * 13, dtype=np.uint8).reshape(9, 13, 1)
    field = np.full((9, 13, 2), (0.25, -0.5), dtype=np.float32)

    image_only = A.Compose([A.RandomRotate90(p=1.0)], seed=137, save_applied_params=True)(image=image)
    with_field = A.Compose([A.RandomRotate90(p=1.0)], seed=137, save_applied_params=True)(image=image, gradient=field)

    np.testing.assert_array_equal(with_field["image"], image_only["image"])
    assert with_field["applied_transforms"] == image_only["applied_transforms"]


def test_contract_comparison_accepts_matching_nan_sentinels_only() -> None:
    expected = np.asarray(((1.0, np.nan), (2.0, 3.0)), dtype=np.float32)
    matching = expected.copy()
    mismatched = expected.copy()
    mismatched[0, 0] = np.nan

    assert_contract_values_equal(matching, expected)
    with pytest.raises(AssertionError):
        assert_contract_values_equal(mismatched, expected)


@pytest.mark.parametrize(
    ("target", "field"),
    (
        ("gradient", np.zeros((5, 7, 2), dtype=np.float64)),
        ("gradient", np.zeros((5, 7, 2), dtype=np.uint8)),
        ("gradient", np.zeros((5, 7), dtype=np.float32)),
        ("normal", np.zeros((5, 7, 2), dtype=np.float32)),
        ("gradient3d", np.zeros((3, 5, 7, 2), dtype=np.float32)),
    ),
)
def test_vector_field_input_rejects_unsupported_dtypes_and_layouts(target: str, field: np.ndarray) -> None:
    image = np.zeros((5, 7, 1), dtype=np.float32)
    pipeline = A.Compose([], strict=True)

    with pytest.raises(TypeError):
        pipeline(image=image, **{target: field})


@pytest.mark.parametrize(("target", "spatial_rank", "components"), _VECTOR_FIELD_TARGET_CASES)
def test_vector_field_tensor_rejects_wrong_component_count(target: str, spatial_rank: int, components: int) -> None:
    spatial_shape = (5, 7) if spatial_rank == 2 else (3, 5, 7)
    invalid_field = torch.zeros((components + 1, *spatial_shape), dtype=torch.float32)
    primary_key = "image" if spatial_rank == 2 else "volume"
    primary = torch.zeros((1, *spatial_shape), dtype=torch.float32)

    with pytest.raises(TypeError, match="shape"):
        A.Compose([], strict=True)(**{primary_key: primary, target: invalid_field})


@pytest.mark.parametrize("transform_cls", (A.CenterCrop3D, A.RandomCrop3D))
def test_3d_crop_padding_keeps_vector_field_aligned_with_volume(transform_cls: type[Any]) -> None:
    volume = np.zeros((2, 3, 4, 1), dtype=np.float32)
    field = np.ones((2, 3, 4, 3), dtype=np.float32)
    transform = A.Compose([transform_cls(size=(4, 3, 4), pad_if_needed=True, p=1.0)], strict=True, seed=137)

    result = transform(volume=volume, gradient3d=field)

    assert result["volume"].shape == (4, 3, 4, 1)
    assert result["gradient3d"].shape == (4, 3, 4, 3)


def test_none_vector_field_is_ignored_by_non_strict_compose() -> None:
    image = np.zeros((4, 6, 1), dtype=np.float32)
    pipeline = A.Compose([A.Resize(height=8, width=12, p=1.0)], strict=False)

    result = pipeline(image=image, gradient=None)

    assert result["image"].shape == (8, 12, 1)
    assert result["gradient"] is None


def test_strict_compose_rejects_none_vector_field() -> None:
    image = np.zeros((4, 6, 1), dtype=np.float32)

    with pytest.raises(TypeError, match=r"NumPy array or torch\.Tensor"):
        A.Compose([], strict=True)(image=image, gradient=None)
