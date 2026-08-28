import json

import cv2
import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.transforms3d import functional as f3d


def _forward_elastic_point(
    point: np.ndarray,
    control_coefficients: dict[str, np.ndarray],
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    displacement = np.zeros(3, dtype=np.float64)
    if "xy" in control_coefficients:
        displacement[:2] += fgeometric.evaluate_control_grid(
            point[np.newaxis, :2],
            control_coefficients["xy"],
            volume_shape[1:],
        )[0]
    if "xz" in control_coefficients:
        xz_displacement = fgeometric.evaluate_control_grid(
            point[np.newaxis, (0, 2)],
            control_coefficients["xz"],
            (volume_shape[0], volume_shape[2]),
        )[0]
        displacement[0] += xz_displacement[0]
        displacement[2] += xz_displacement[1]
    if "yz" in control_coefficients:
        yz_displacement = fgeometric.evaluate_control_grid(
            point[np.newaxis, 1:3],
            control_coefficients["yz"],
            volume_shape[:2],
        )[0]
        displacement[1] += yz_displacement[0]
        displacement[2] += yz_displacement[1]
    return point + displacement / len(control_coefficients)


def test_elastic_transform3d_shares_one_field_between_volume_mask_and_keypoints() -> None:
    mask3d = np.zeros((9, 9, 9), dtype=np.uint8)
    mask3d[4, 4, 4] = 137
    volume = mask3d[..., np.newaxis]
    keypoints = np.array([[4.0, 4.0, 4.0, 17.0]], dtype=np.float32)
    transform = A.Compose(
        [
            A.ElasticTransform3D(
                displacement_range=(0.05, 0.05),
                control_grid_shape=(7, 7),
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        seed=137,
        strict=True,
    )

    result = transform(volume=volume, mask3d=mask3d, keypoints=keypoints)

    np.testing.assert_array_equal(result["volume"][..., 0], result["mask3d"])
    mask_keypoint = np.argwhere(result["mask3d"] == 137)[0][::-1]
    np.testing.assert_allclose(result["keypoints"][0, :3], mask_keypoint, atol=1.0)
    np.testing.assert_array_equal(result["keypoints"][:, 3:], keypoints[:, 3:])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"displacement_range": (-0.01, 0.05)},
        {"displacement_range": (0.05, 0.01)},
        {"control_grid_shape": (3, 7)},
        {"displacement_range": (0.2, 0.2)},
    ],
)
def test_elastic_transform3d_rejects_invalid_constructor_contract(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A.ElasticTransform3D(**kwargs)


def test_elastic_transform3d_topology_bound_is_strict() -> None:
    control_grid_shape = (4, 5)
    limit = 0.75 / (2 * np.sqrt((control_grid_shape[0] - 3) ** 2 + (control_grid_shape[1] - 3) ** 2))

    A.ElasticTransform3D(
        displacement_range=(0.0, float(np.nextafter(limit, 0.0))),
        control_grid_shape=control_grid_shape,
    )
    with pytest.raises(ValueError, match="strict topology bound"):
        A.ElasticTransform3D(
            displacement_range=(0.0, float(np.nextafter(limit, np.inf))),
            control_grid_shape=control_grid_shape,
        )


def test_elastic_transform3d_yz_grid_matches_its_voxel_displacement() -> None:
    volume_shape = (5, 7, 11)
    control_coefficients = {
        "yz": np.random.default_rng(137).uniform(-0.2, 0.2, (4, 5, 2)).astype(np.float32),
    }
    output_points = np.array(((3.0, 2.0, 1.0), (7.0, 5.0, 3.0)), dtype=np.float64)

    sampling_grid = f3d.create_elastic_grid_3d(control_coefficients, volume_shape)
    grid_points = sampling_grid[
        output_points[:, 2].astype(int),
        output_points[:, 1].astype(int),
        output_points[:, 0].astype(int),
    ]
    source_points = (grid_points + 1.0) * np.asarray((volume_shape[2], volume_shape[1], volume_shape[0])) / 2.0 - 0.5
    expected_points = np.stack(
        [_forward_elastic_point(point, control_coefficients, volume_shape) for point in output_points],
    )

    np.testing.assert_allclose(source_points, expected_points, atol=1e-6)


def test_elastic_transform3d_keypoint_inverse_recovers_xyz_reference() -> None:
    volume_shape = (17, 19, 23)
    random_generator = np.random.default_rng(137)
    control_coefficients = {
        plane: random_generator.uniform(-0.2, 0.2, (4, 5, 2)).astype(np.float32) for plane in ("xy", "xz", "yz")
    }
    output_points = random_generator.uniform([1.0, 1.0, 1.0], [21.0, 17.0, 15.0], (32, 3))
    source_points = np.stack(
        [_forward_elastic_point(point, control_coefficients, volume_shape) for point in output_points],
    )
    keypoints = np.column_stack(
        [source_points, np.full(len(source_points), 17.0, dtype=np.float32)],
    ).astype(np.float32)

    transformed = f3d.remap_elastic_keypoints_3d(keypoints, control_coefficients, volume_shape)

    np.testing.assert_allclose(transformed[:, :3], output_points, atol=1e-3)
    np.testing.assert_array_equal(transformed[:, 3:], keypoints[:, 3:])


def test_elastic_transform3d_remap_respects_constant_volume_and_mask_fills() -> None:
    volume = np.ones((3, 5, 7, 1), dtype=np.uint8)
    mask3d = np.ones((3, 5, 7), dtype=np.uint16)
    sampling_grid = f3d.create_elastic_grid_3d({}, volume.shape[:3])
    sampling_grid[..., 0] = 2.0

    remapped_volume = f3d.remap_3d(
        volume,
        sampling_grid,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
        fill=17,
    )
    remapped_mask = f3d.remap_3d(
        mask3d,
        sampling_grid,
        cv2.INTER_NEAREST,
        cv2.BORDER_CONSTANT,
        fill=31,
        is_mask=True,
    )

    np.testing.assert_array_equal(remapped_volume, np.full_like(volume, 17))
    np.testing.assert_array_equal(remapped_mask, np.full_like(mask3d, 31))


def test_elastic_transform3d_identity_skips_resampling_for_zero_magnitude() -> None:
    zero_magnitude_volume = np.random.default_rng(137).random((7, 11, 13, 1), dtype=np.float32)
    zero_magnitude = A.Compose([A.ElasticTransform3D(displacement_range=(0.0, 0.0), p=1.0)], strict=True)

    zero_result = zero_magnitude(volume=zero_magnitude_volume)["volume"]

    assert zero_result is zero_magnitude_volume


def test_elastic_transform3d_seeded_compose_reproduces_the_compact_field() -> None:
    volume = np.random.default_rng(137).random((7, 11, 13, 1), dtype=np.float32)
    kwargs = {"displacement_range": (0.02, 0.05), "control_grid_shape": (7, 7), "p": 1.0}
    first = A.Compose([A.ElasticTransform3D(**kwargs)], seed=137, strict=True)
    second = A.Compose([A.ElasticTransform3D(**kwargs)], seed=137, strict=True)

    np.testing.assert_array_equal(first(volume=volume)["volume"], second(volume=volume)["volume"])


def test_elastic_transform3d_shares_geometry_with_additional_volume_and_mask_targets() -> None:
    volume = np.random.default_rng(137).integers(0, 256, (7, 11, 13, 1), dtype=np.uint8)
    mask3d = np.arange(7 * 11 * 13, dtype=np.uint8).reshape(7, 11, 13)
    transform = A.Compose(
        [A.ElasticTransform3D(displacement_range=(0.05, 0.05), p=1.0)],
        additional_targets={"second_volume": "volume", "second_mask3d": "mask3d"},
        seed=137,
        strict=True,
    )

    result = transform(
        volume=volume,
        mask3d=mask3d,
        second_volume=volume,
        second_mask3d=mask3d,
    )

    np.testing.assert_array_equal(result["volume"], result["second_volume"])
    np.testing.assert_array_equal(result["mask3d"], result["second_mask3d"])


def test_elastic_transform3d_rejects_keypoint_only_input_without_a_spatial_domain() -> None:
    transform = A.Compose(
        [A.ElasticTransform3D(p=1.0)],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        strict=True,
    )

    with pytest.raises(ValueError, match="No image or volume"):
        transform(keypoints=np.array([[4.0, 5.0, 6.0]], dtype=np.float32))


def test_elastic_transform3d_replay_roundtrip_and_shape_guard() -> None:
    volume = np.random.default_rng(137).random((7, 11, 13, 2), dtype=np.float32)
    mask3d = np.arange(7 * 11 * 13, dtype=np.uint16).reshape(7, 11, 13)
    keypoints = np.array([[6.0, 5.0, 3.0, 17.0]], dtype=np.float32)
    transform = A.ReplayCompose(
        [A.ElasticTransform3D(displacement_range=(0.02, 0.05), p=1.0)],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        seed=137,
    )

    result = transform(volume=volume, mask3d=mask3d, keypoints=keypoints)
    replayed = A.ReplayCompose.replay(
        json.loads(json.dumps(result["replay"], allow_nan=False)),
        volume=volume,
        mask3d=mask3d,
        keypoints=keypoints,
    )

    for target in ("volume", "mask3d", "keypoints"):
        np.testing.assert_array_equal(result[target], replayed[target])
    with pytest.raises(ValueError, match="same spatial shape"):
        A.ReplayCompose.replay(result["replay"], volume=volume[:-1])


def test_elastic_transform3d_applied_config_fixes_magnitude_without_storing_coefficients() -> None:
    volume = np.random.default_rng(137).random((7, 11, 13, 1), dtype=np.float32)
    pipeline = A.Compose(
        [A.ElasticTransform3D(displacement_range=(0.02, 0.05), p=1.0)],
        save_applied_params=True,
        seed=137,
    )

    result = pipeline(volume=volume)
    _, applied_config = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))[0]

    assert applied_config["displacement_range"][0] == applied_config["displacement_range"][1]
    assert "control_coefficients" not in applied_config
    reconstructed = A.Compose.from_applied_transforms(result["applied_transforms"], seed=137)
    assert reconstructed(volume=volume)["volume"].shape == volume.shape


def test_elastic_transform3d_tensor_volume_and_mask_use_native_route(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = np.random.default_rng(137).integers(0, 256, (7, 11, 13, 1), dtype=np.uint8)
    mask3d = np.arange(7 * 11 * 13, dtype=np.uint16).reshape(7, 11, 13)
    tensor_volume = torch.from_numpy(volume).permute(3, 0, 1, 2)
    tensor_mask3d = torch.from_numpy(mask3d)
    transform_kwargs = {"displacement_range": (0.05, 0.05), "control_grid_shape": (7, 7), "p": 1.0}
    numpy_transform = A.Compose([A.ElasticTransform3D(**transform_kwargs)], seed=137, strict=True)
    tensor_transform = A.Compose([A.ElasticTransform3D(**transform_kwargs)], seed=137, strict=True)

    monkeypatch.setattr(
        A.Compose,
        "_bridge_tensor_data_to_numpy",
        lambda *_args, **_kwargs: pytest.fail("ElasticTransform3D unexpectedly selected Compose's NumPy bridge"),
    )
    assert tensor_transform.transforms[0].supports_cpu_tensor is True
    numpy_result = numpy_transform(volume=volume, mask3d=mask3d)
    tensor_result = tensor_transform(volume=tensor_volume, mask3d=tensor_mask3d)

    assert tensor_result["volume"].dtype == tensor_volume.dtype
    assert tensor_result["mask3d"].dtype == tensor_mask3d.dtype
    torch.testing.assert_close(
        tensor_result["volume"],
        torch.from_numpy(numpy_result["volume"]).permute(3, 0, 1, 2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(tensor_result["mask3d"], torch.from_numpy(numpy_result["mask3d"]), rtol=0, atol=0)


@pytest.mark.parametrize("channels", (3, 5))
def test_elastic_transform3d_multichannel_tensor_volume_uses_the_numpy_bridge(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    volume = np.random.default_rng(137).integers(0, 256, (7, 11, 13, channels), dtype=np.uint8)
    tensor_volume = torch.from_numpy(volume).permute(3, 0, 1, 2)
    transform = A.Compose([A.ElasticTransform3D(p=1.0)], seed=137, strict=True)
    bridge_calls = 0
    original_bridge = A.Compose._bridge_tensor_data_to_numpy

    def count_bridge(self: A.Compose, *args: object, **kwargs: object) -> None:
        nonlocal bridge_calls
        bridge_calls += 1
        original_bridge(self, *args, **kwargs)

    monkeypatch.setattr(A.Compose, "_bridge_tensor_data_to_numpy", count_bridge)

    result = transform(volume=tensor_volume)["volume"]

    assert bridge_calls == 1
    assert result.shape == tensor_volume.shape
    assert result.dtype == tensor_volume.dtype
