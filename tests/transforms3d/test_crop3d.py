"""Exact explicit-origin Crop3D public contracts."""

import json

import numpy as np
import pytest
import torch

import albumentations as A


@pytest.mark.parametrize("scalar", [np.int64(2**60 + 3), np.uint64(2**63 + 3), np.uint64(2**64 - 1)])
@pytest.mark.parametrize("override", [False, True])
def test_numpy_integer_fill_preserves_exact_label_through_json_and_replay(scalar, override):
    value = int(scalar)
    config = {"size": (2, 3, 4), "origin": (-1, 0, 0), "pad_if_needed": True}
    config.update({"target_overrides": {"mask3d": {"fill": scalar}}} if override else {"fill_mask": scalar})
    data = {"volume": np.zeros((3, 5, 7), np.uint8), "mask3d": np.ones((3, 5, 7), dtype=scalar.dtype)}
    transform = A.Crop3D(**config)
    constructor = json.loads(json.dumps(A.to_dict(transform), allow_nan=False))
    restored = A.from_dict(constructor)
    original = A.Compose([restored], save_applied_params=True)(**data)
    applied = json.loads(json.dumps(original["applied_transforms"], allow_nan=False))
    reconstructed = A.Compose.from_applied_transforms(applied)(**data)
    captured = A.ReplayCompose([A.Crop3D(**config)])(**data)
    replayed = A.ReplayCompose.replay(json.loads(json.dumps(captured["replay"], allow_nan=False)), **data)
    expected = np.full((2, 3, 4), value, dtype=scalar.dtype)
    expected[1] = 1
    for result in (original, reconstructed, captured, replayed):
        np.testing.assert_array_equal(result["mask3d"], expected)


@pytest.mark.parametrize("scalar", [np.int64(2**60 + 3), np.uint64(2**63 + 3)])
def test_crop3d_numpy_integer_volume_fill_serializes_without_rounding(scalar):
    transform = A.Crop3D((1, 1, 1), origin=(0, 0, 0), fill=scalar)
    payload = json.loads(json.dumps(A.to_dict(transform), allow_nan=False))
    assert payload["transform"]["fill"] == int(scalar)
    assert isinstance(payload["transform"]["fill"], int)


@pytest.mark.parametrize("shape", [(0, 5, 7), (3, 0, 7), (3, 5, 0)])
@pytest.mark.parametrize("name", ["mask3d", "coverage"])
@pytest.mark.parametrize("tensor", [False, True])
def test_crop3d_rejects_empty_axis_mask_mismatch(shape, name, tensor):
    volume = torch.zeros((1, 3, 5, 7)) if tensor else np.zeros((3, 5, 7), np.float32)
    mask = torch.zeros(shape, dtype=torch.uint8) if tensor else np.zeros(shape, np.uint8)
    pipeline = A.Compose(
        [A.Crop3D((2, 3, 4), origin=(-1, 0, 0), pad_if_needed=True)],
        additional_targets={"coverage": "mask3d"},
        is_check_shapes=False,
    )
    masks = {name: mask}
    if name == "coverage":
        masks["mask3d"] = torch.zeros((3, 5, 7), dtype=torch.uint8) if tensor else np.zeros((3, 5, 7), np.uint8)
    with pytest.raises(ValueError, match=r"shape|align"):
        pipeline(volume=volume, **masks)


def expected_crop(volume, origin, size, fill=0):
    """Independent elementwise indexing oracle, including external windows."""
    output = np.full((*size, *volume.shape[3:]), fill, dtype=volume.dtype)
    for destination in np.ndindex(size):
        source = tuple(a + b for a, b in zip(destination, origin, strict=True))
        if all(0 <= index < bound for index, bound in zip(source, volume.shape[:3], strict=True)):
            output[destination] = volume[source]
    return output


@pytest.mark.parametrize(
    "origin", [(0, 1, 2), (-1, 1, 1), (1, -2, 1), (1, 1, -3), (2, 1, 1), (1, 4, 1), (1, 1, 6), (99, 0, 0), (-99, 0, 0)]
)
@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_crop3d_matches_independent_oracle(origin, tensor, channels):
    volume = np.arange(3 * 5 * 7 * channels, dtype=np.float32).reshape(3, 5, 7, channels)
    before = volume.copy()
    source = torch.from_numpy(volume).permute(3, 0, 1, 2) if tensor else volume
    result = A.Compose([A.Crop3D((2, 3, 4), origin=origin, pad_if_needed=True, fill=-7)], strict=True)(
        volume=source,
    )["volume"]
    actual = result.permute(1, 2, 3, 0).numpy() if tensor else result
    np.testing.assert_array_equal(actual, expected_crop(volume, origin, (2, 3, 4), -7))
    np.testing.assert_array_equal(volume, before)


@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.uint8, 251),
        (np.int16, 30000),
        (np.int32, 16777217),
        (np.int64, 2**60 + 3),
        (np.uint64, 2**63 + 3),
        (np.bool_, 1),
    ],
)
def test_crop3d_preserves_primary_and_alias_mask_values(dtype, value):
    volume = np.zeros((3, 5, 7), np.uint8)
    mask = np.full(volume.shape, value, dtype=dtype)
    transform = A.Compose(
        [
            A.Crop3D(
                (4, 3, 5),
                origin=(-1, 2, 3),
                pad_if_needed=True,
                fill_mask=value,
                target_overrides={"coverage": {"fill": 0}},
            )
        ],
        additional_targets={"coverage": "mask3d"},
        strict=True,
    )
    result = transform(volume=volume, mask3d=mask, coverage=mask.copy())
    np.testing.assert_array_equal(result["mask3d"], expected_crop(mask, (-1, 2, 3), (4, 3, 5), value))
    np.testing.assert_array_equal(result["coverage"], expected_crop(mask, (-1, 2, 3), (4, 3, 5), 0))


@pytest.mark.parametrize("fill", [-1, 256, 0.5])
def test_crop3d_rejects_unrepresentable_integer_fill(fill):
    with pytest.raises(ValueError, match="fill"):
        A.Compose([A.Crop3D((2, 3, 4), origin=(-1, 0, 0), pad_if_needed=True, fill_mask=fill)])(
            volume=np.zeros((3, 5, 7), np.uint8),
            mask3d=np.zeros((3, 5, 7), np.uint8),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("origin", (True, 0, 0)),
        ("origin", (1.0, 0, 0)),
        ("origin", (0, 0)),
        ("size", (True, 3, 4)),
        ("size", (2.0, 3, 4)),
        ("size", (0, 3, 4)),
    ],
)
def test_crop3d_rejects_noninteger_controls(field, value):
    kwargs = {"size": (2, 3, 4), "origin": (0, 0, 0), field: value}
    with pytest.raises(ValueError):
        A.Crop3D(**kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"origin": (0, 0, 0), "origin_key": "start"}, {"origin_key": ""}])
def test_crop3d_requires_one_origin_source(kwargs):
    with pytest.raises(ValueError):
        A.Crop3D((2, 3, 4), **kwargs)


def test_crop3d_applied_config_freezes_origin_and_replay_checks_fresh_inputs():
    volume = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    transform = A.Compose([A.Crop3D((2, 3, 4), origin_key="start")], save_applied_params=True, strict=True)
    output = transform(volume=volume, user_data={"start": [1, 1, 2], "extra": {"x": 137}})
    applied = json.loads(json.dumps(output["applied_transforms"], allow_nan=False))
    assert applied[0][1]["origin"] == [1, 1, 2]
    assert applied[0][1]["origin_key"] is None
    np.testing.assert_array_equal(A.Compose.from_applied_transforms(applied)(volume=volume)["volume"], output["volume"])
    restored = A.from_dict(json.loads(json.dumps(A.to_dict(transform), allow_nan=False)))
    np.testing.assert_array_equal(restored(volume=volume, user_data={"start": (1, 1, 2)})["volume"], output["volume"])
    replay = A.ReplayCompose([A.Crop3D((2, 3, 4), origin_key="start")])
    captured = replay(volume=volume, user_data={"start": (1, 1, 2)})["replay"]
    trace = json.loads(json.dumps(captured, allow_nan=False))
    fresh = volume + 10
    expected = fresh[1:3, 1:4, 2:6]
    np.testing.assert_array_equal(A.ReplayCompose.replay(trace, volume=fresh)["volume"], expected)
    np.testing.assert_array_equal(
        A.ReplayCompose.replay(trace, volume=fresh, user_data={"start": [1, 1, 2]})["volume"],
        expected,
    )
    with pytest.raises(ValueError, match="origin"):
        A.ReplayCompose.replay(trace, volume=fresh, user_data={"start": (0, 0, 0)})
    with pytest.raises(ValueError, match=r"shape|requirements"):
        A.ReplayCompose.replay(trace, volume=np.zeros((4, 5, 7), np.float32))


def test_crop3d_keypoints_translate_and_filter_labels():
    pipeline = A.Compose(
        [A.Crop3D((2, 3, 4), origin=(1, 1, 2))],
        keypoint_params=A.KeypointParams(coord_format="xyz", label_fields=["labels"]),
    )
    result = pipeline(
        volume=np.zeros((3, 5, 7), np.uint8),
        keypoints=[[3, 2, 1], [0, 0, 0], [6, 4, 2]],
        labels=["keep", "left", "right"],
    )
    np.testing.assert_array_equal(result["keypoints"], [[1, 1, 0]])
    assert result["labels"] == ["keep"]


def test_crop3d_skip_retry_and_ownership():
    source = np.arange(105, dtype=np.float32).reshape(3, 5, 7, 1)
    transform = A.Crop3D((2, 3, 4), origin_key="start")
    skipped = A.Compose([A.Crop3D((2, 3, 4), origin_key="missing", p=0)])(volume=source)
    assert skipped["volume"] is source
    with pytest.raises(ValueError, match="start"):
        transform(volume=source)
    with pytest.raises(ValueError, match="bounds"):
        transform(volume=source, user_data={"start": (-1, 0, 0)})
    result = transform(volume=source, user_data={"start": (0, 0, 0)})["volume"]
    np.testing.assert_array_equal(result, source[:2, :3, :4])
    assert np.shares_memory(source, result)
    padded = A.Crop3D((2, 3, 4), origin=(-1, 0, 0), pad_if_needed=True)(volume=source)["volume"]
    assert not np.shares_memory(source, padded)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7)])
def test_crop3d_preserves_nonfinite_and_strided_values(shape):
    source = np.ones((*shape, 1), np.float32)
    source[0, 0, 0, 0] = np.nan
    source = source[::-1]
    source.setflags(write=False)
    result = A.Compose([A.Crop3D(shape, origin=(0, 0, 0))])(volume=source)["volume"]
    np.testing.assert_array_equal(result, source)


@pytest.mark.parametrize("override", [{"missing": {"fill": 0}}, {"volume": {"fill": 0}}])
def test_crop3d_rejects_missing_or_nonmask_overrides(override):
    with pytest.raises(ValueError, match="override"):
        A.Compose([A.Crop3D((1, 1, 1), origin=(0, 0, 0), target_overrides=override)])(
            volume=np.zeros((3, 5, 7), np.uint8),
        )


def test_crop3d_rejects_alias_shape_mismatch():
    transform = A.Compose(
        [A.Crop3D((1, 1, 1), origin=(0, 0, 0))], additional_targets={"coverage": "mask3d"}, is_check_shapes=False
    )
    with pytest.raises(ValueError, match="align"):
        transform(
            volume=np.zeros((3, 5, 7), np.uint8),
            mask3d=np.zeros((3, 5, 7), np.uint8),
            coverage=np.zeros((2, 5, 7), np.uint8),
        )


@pytest.mark.parametrize("dtype,value", [(torch.uint8, 251), (torch.int16, 30000), (torch.float32, -1.25)])
@pytest.mark.parametrize("channels", [None, 1, 5])
def test_crop3d_tensor_masks_and_numpy_alias_share_window(dtype, value, channels):
    shape = (3, 5, 7) if channels is None else (channels, 3, 5, 7)
    mask = torch.full(shape, value, dtype=dtype)
    volume = torch.zeros((1, 3, 5, 7), dtype=torch.float32)
    coverage = np.ones((3, 5, 7), np.uint8)
    transform = A.Compose(
        [
            A.Crop3D(
                (2, 3, 4),
                origin=(-1, 1, 2),
                pad_if_needed=True,
                fill_mask=value,
                target_overrides={"coverage": {"fill": 0}},
            )
        ],
        additional_targets={"coverage": "mask3d"},
    )
    result = transform(volume=volume, mask3d=mask, coverage=coverage)
    expected_shape = (2, 3, 4) if channels is None else (channels, 2, 3, 4)
    torch.testing.assert_close(result["mask3d"], torch.full(expected_shape, value, dtype=dtype), rtol=0, atol=0)
    np.testing.assert_array_equal(result["coverage"][0], 0)
    np.testing.assert_array_equal(result["coverage"][1], 1)
    assert result["mask3d"].untyped_storage().data_ptr() != mask.untyped_storage().data_ptr()


@pytest.mark.parametrize("compose", [False, True])
@pytest.mark.parametrize("target", ["mask3d", "coverage"])
def test_crop3d_rejects_bool_tensor_masks_at_input_boundary(compose, target):
    transform = A.Crop3D((2, 3, 4), origin=(-1, 1, 2), pad_if_needed=True)
    transform.add_targets({"coverage": "mask3d"})
    pipeline = A.Compose([transform], additional_targets={"coverage": "mask3d"}) if compose else transform
    data = {
        "volume": np.zeros((3, 5, 7, 1), np.float32),
        "mask3d": np.ones((3, 5, 7, 1), np.uint8),
        target: torch.ones((1, 3, 5, 7), dtype=torch.bool),
    }
    with pytest.raises(TypeError, match=rf"{target} must have dtype one of .*got torch.bool"):
        pipeline(**data)


def test_crop3d_big_mask_fill_survives_applied_config_and_replay():
    fill = 2**63 + 3
    data = {"volume": np.zeros((3, 5, 7), np.uint8), "mask3d": np.ones((3, 5, 7), np.uint64)}
    config = {"size": (2, 3, 4), "origin": (-1, 1, 2), "pad_if_needed": True, "fill_mask": fill}
    original = A.Compose([A.Crop3D(**config)], save_applied_params=True)(**data)
    applied = json.loads(json.dumps(original["applied_transforms"], allow_nan=False))
    reconstructed = A.Compose.from_applied_transforms(applied)(**data)
    trace = A.ReplayCompose([A.Crop3D(**config)])(**data)["replay"]
    replayed = A.ReplayCompose.replay(json.loads(json.dumps(trace, allow_nan=False)), **data)
    expected = np.full((2, 3, 4), fill, np.uint64)
    expected[1] = 1
    for result in (original, reconstructed, replayed):
        np.testing.assert_array_equal(result["mask3d"], expected)


@pytest.mark.parametrize("target", ["volumes", "masks3d"])
def test_crop3d_rejects_batches(target):
    with pytest.raises(ValueError, match="batches"):
        A.Crop3D((2, 3, 4), origin=(0, 0, 0))(
            volume=np.zeros((3, 5, 7, 1), np.uint8),
            **{target: np.zeros((2, 3, 5, 7, 1), np.uint8)},
        )


@pytest.mark.parametrize("tensor", [False, True])
def test_crop3d_rejects_fill_that_overflows_float32(tensor):
    source = torch.zeros((1, 3, 5, 7), dtype=torch.float32) if tensor else np.zeros((3, 5, 7), np.float32)
    with pytest.raises(ValueError, match="fill"):
        A.Compose([A.Crop3D((2, 3, 4), origin=(-1, 0, 0), pad_if_needed=True, fill=1e40)])(volume=source)
