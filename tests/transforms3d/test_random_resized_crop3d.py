"""Focused contracts for RandomResizedCrop3D."""

import json
import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.augmentations.crops._sampling import sample_3d_crop_shape


@pytest.mark.parametrize("sampling_method", ("standard", "uniform_scale"))
@pytest.mark.parametrize("ratio", (1.0, 4.0 / 3.0, None))
def test_random_resized_crop3d_resizes_all_geometric_targets(
    sampling_method: str,
    ratio: float | None,
) -> None:
    volume = np.random.default_rng(137).random((64, 128, 128, 1), dtype=np.float32)
    mask3d = np.arange(64 * 128 * 128, dtype=np.uint16).reshape(64, 128, 128)
    transform = A.Compose(
        [
            A.RandomResizedCrop3D(
                size=(32, 64, 64),
                ratio=ratio,
                sampling_method=sampling_method,
                p=1.0,
            ),
        ],
        strict=True,
        seed=137,
    )

    result = transform(volume=volume, mask3d=mask3d)

    assert result["volume"].shape == (32, 64, 64, 1)
    assert result["mask3d"].shape == (32, 64, 64)
    assert result["volume"].dtype == np.float32
    assert result["mask3d"].dtype == np.uint16


@pytest.mark.parametrize("sampling_method", ("standard", "uniform_scale"))
def test_random_resized_crop3d_samples_a_feasible_shape_without_retries(sampling_method: str) -> None:
    random = Mock()
    random.random.side_effect = (0.2, 0.3, 0.4, 0.5)

    crop_shape = sample_3d_crop_shape(
        depth=128,
        height=128,
        width=128,
        output_size=(32, 64, 64),
        scale=(0.2, 0.8),
        ratio=4.0 / 3.0,
        sampling_method=sampling_method,
        py_random=random,
    )

    assert crop_shape is not None
    assert random.random.call_count == 4


def test_random_resized_crop3d_ratio_none_uses_output_proportions() -> None:
    crop_shape = sample_3d_crop_shape(
        depth=64,
        height=128,
        width=128,
        output_size=(32, 64, 64),
        scale=(0.125, 0.125),
        ratio=None,
        sampling_method="standard",
        py_random=np.random.default_rng(137),
    )

    assert crop_shape == (32, 64, 64)


@pytest.mark.parametrize("sampling_method", ("standard", "uniform_scale"))
def test_random_resized_crop3d_numeric_ratio_respects_the_pre_rounding_policy(sampling_method: str) -> None:
    random = np.random.default_rng(137)
    for _ in range(100):
        crop_shape = sample_3d_crop_shape(
            depth=128,
            height=128,
            width=128,
            output_size=(32, 64, 64),
            scale=(0.2, 0.8),
            ratio=4.0 / 3.0,
            sampling_method=sampling_method,
            py_random=random,
        )
        assert crop_shape is not None
        assert max(crop_shape) / min(crop_shape) <= 4.0 / 3.0 + 1.0 / min(crop_shape)


@pytest.mark.parametrize("ratio", ((1.0, 1.0), 0.5, float("inf"), float("nan")))
def test_random_resized_crop3d_rejects_invalid_ratio(ratio: object) -> None:
    with pytest.raises(ValueError):
        A.RandomResizedCrop3D(size=(32, 64, 64), ratio=ratio)  # type: ignore[arg-type]


def test_random_resized_crop3d_rejects_unknown_sampling_method() -> None:
    with pytest.raises(ValueError):
        A.RandomResizedCrop3D(size=(32, 64, 64), sampling_method="rejection")  # type: ignore[arg-type]


def test_random_resized_crop3d_center_fallback_is_used_for_an_empty_region() -> None:
    volume = np.zeros((32, 256, 256, 1), dtype=np.float32)
    transform = A.ReplayCompose([A.RandomResizedCrop3D(size=(32, 64, 64), p=1.0)], seed=137)

    result = transform(volume=volume)
    crop_coords = result["replay"]["transforms"][0]["params"]["params"]["crop_coords"]

    assert crop_coords == (0, 32, 106, 149, 106, 149)


def test_random_resized_crop3d_scales_xyz_keypoints_after_cropping() -> None:
    transform = A.RandomResizedCrop3D(size=(16, 32, 32), p=1.0)
    keypoints = np.array([[48.0, 64.0, 32.0, 17.0]], dtype=np.float32)

    result = transform.apply_to_keypoints(keypoints, crop_coords=(16, 48, 32, 96, 32, 96))

    np.testing.assert_array_equal(result, np.array([[8.0, 16.0, 8.0, 17.0]], dtype=np.float32))


def test_random_resized_crop3d_replay_and_applied_config_contracts() -> None:
    volume = np.random.default_rng(137).random((64, 128, 128, 1), dtype=np.float32)
    mask3d = np.arange(64 * 128 * 128, dtype=np.uint16).reshape(64, 128, 128)
    keypoints = np.array([[64.0, 64.0, 32.0]], dtype=np.float32)
    replay_transform = A.ReplayCompose(
        [A.RandomResizedCrop3D(size=(32, 64, 64), p=1.0)],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        seed=137,
    )

    replay_result = replay_transform(volume=volume, mask3d=mask3d, keypoints=keypoints)
    replayed = A.ReplayCompose.replay(
        json.loads(json.dumps(replay_result["replay"], allow_nan=False)),
        volume=volume,
        mask3d=mask3d,
        keypoints=keypoints,
    )
    for target in ("volume", "mask3d", "keypoints"):
        np.testing.assert_array_equal(replayed[target], replay_result[target])

    pipeline = A.Compose(
        [A.RandomResizedCrop3D(size=(32, 64, 64), p=1.0)],
        save_applied_params=True,
        seed=137,
        strict=True,
    )
    applied_result = pipeline(volume=volume)
    _, applied_config = json.loads(json.dumps(applied_result["applied_transforms"], allow_nan=False))[0]
    assert applied_config["ratio"] == pytest.approx(4.0 / 3.0)
    assert applied_config["sampling_method"] == "standard"
    assert applied_config["scale"][0] == applied_config["scale"][1]
    assert math.isfinite(applied_config["scale"][0])

    reconstructed = A.Compose.from_applied_transforms(applied_result["applied_transforms"], strict=True)
    assert reconstructed(volume=volume)["volume"].shape == (32, 64, 64, 1)


def test_random_resized_crop3d_preserves_cpu_tensor_layout() -> None:
    volume = torch.arange(2 * 64 * 128 * 128, dtype=torch.float32).reshape(2, 64, 128, 128)
    transform = A.Compose([A.RandomResizedCrop3D(size=(32, 64, 64), p=1.0)], strict=True, seed=137)

    result = transform(volume=volume)["volume"]

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 32, 64, 64)
    assert result.dtype == torch.float32
