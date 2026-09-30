import cv2
import numpy as np
import pytest
import torch

import albumentations as A
from tests.helpers import TestDataFactory

IDENTITY_ROTATE_RANGE = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
IDENTITY_SCALE_RANGE = {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)}
IDENTITY_TRANSLATE_RANGE = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}


@pytest.mark.parametrize(
    ("axis", "array_axes"),
    [("x", (0, 1)), ("y", (0, 2)), ("z", (1, 2))],
)
def test_affine3d_quarter_turns_match_numpy_rot90(axis: str, array_axes: tuple[int, int]) -> None:
    volume = np.arange(5**3, dtype=np.uint8).reshape(5, 5, 5, 1)
    mask3d = (volume[..., 0] % 7).astype(np.uint8)
    rotate_range = {**IDENTITY_ROTATE_RANGE, axis: (90.0, 90.0)}
    transform = A.Compose(
        [
            A.Affine3D(
                rotate_range=rotate_range,
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        strict=True,
    )

    result = transform(volume=volume, mask3d=mask3d)

    np.testing.assert_array_equal(result["volume"], np.rot90(volume, axes=array_axes))
    np.testing.assert_array_equal(result["mask3d"], np.rot90(mask3d, axes=array_axes))


def test_affine3d_scales_volume_mask_and_xyz_keypoints_with_one_matrix() -> None:
    volume = np.zeros((5, 5, 5, 1), dtype=np.uint8)
    mask3d = np.zeros((5, 5, 5), dtype=np.uint8)
    volume[2, 2, 3, 0] = 137
    mask3d[2, 2, 3] = 7
    keypoints = np.array([[3.0, 2.0, 2.0]], dtype=np.float32)
    transform = A.Compose(
        [
            A.Affine3D(
                scale_range={"x": (2.0, 2.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        strict=True,
    )

    result = transform(volume=volume, mask3d=mask3d, keypoints=keypoints)

    assert result["volume"][2, 2, 4, 0] == 137
    assert result["mask3d"][2, 2, 4] == 7
    np.testing.assert_array_equal(result["keypoints"], np.array([[4.0, 2.0, 2.0]], dtype=np.float32))


def test_affine3d_uses_separate_constant_fills_for_volume_and_channel_less_mask() -> None:
    volume = np.full((3, 4, 5, 2), 137, dtype=np.uint8)
    mask3d = np.full((3, 4, 5), 7, dtype=np.uint16)
    transform = A.Compose(
        [
            A.Affine3D(
                translate_percent_range={"x": (0.2, 0.2), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=(17, 23),
                fill_mask=31,
                p=1.0,
            ),
        ],
        strict=True,
    )

    result = transform(volume=volume, mask3d=mask3d)

    np.testing.assert_array_equal(
        result["volume"][:, :, 0],
        np.broadcast_to(np.array((17, 23), dtype=np.uint8), (3, 4, 2)),
    )
    np.testing.assert_array_equal(result["mask3d"][:, :, 0], np.full((3, 4), 31, dtype=np.uint16))
    assert result["mask3d"].dtype == np.uint16


def test_affine3d_preserves_non_cubic_shape_dtype_and_single_slice_rotation() -> None:
    volume = TestDataFactory.create_volume((1, 5, 5, 3), dtype=np.float32, seed=137)
    transform = A.Compose(
        [
            A.Affine3D(
                rotate_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (90.0, 90.0)},
                interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        strict=True,
    )

    result = transform(volume=volume)["volume"]

    assert result.shape == volume.shape
    assert result.dtype == volume.dtype
    np.testing.assert_array_equal(result, np.rot90(volume, axes=(1, 2)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rotate_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
        {"scale_range": {"x": (-1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)}},
        {"interpolation": cv2.INTER_CUBIC},
        {"border_mode": cv2.BORDER_REFLECT},
    ],
)
def test_affine3d_rejects_incomplete_ranges_reflections_and_unsupported_resampling_modes(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        A.Affine3D(**kwargs)


def test_affine3d_seeded_replay_reproduces_the_sampled_matrix() -> None:
    volume = TestDataFactory.create_volume((5, 11, 13, 2), dtype=np.float32, seed=137)
    transform = A.ReplayCompose(
        [
            A.Affine3D(
                rotate_range={"x": (-10.0, 10.0), "y": (-5.0, 5.0), "z": (-15.0, 15.0)},
                scale_range={"x": (0.9, 1.1), "y": (0.95, 1.05), "z": (0.85, 1.15)},
                translate_percent_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
                p=1.0,
            ),
        ],
        seed=137,
    )

    first_result = transform(volume=volume)
    replay_result = A.ReplayCompose.replay(first_result["replay"], volume=volume)

    np.testing.assert_array_equal(replay_result["volume"], first_result["volume"])


def test_affine3d_accepts_cpu_tensor_volume_and_int16_mask3d() -> None:
    volume = np.arange(3 * 5 * 5 * 3, dtype=np.uint8).reshape(3, 5, 5, 3)
    tensor_volume = torch.from_numpy(np.ascontiguousarray(volume.transpose(3, 0, 1, 2)))
    tensor_mask = torch.from_numpy((volume[..., 0] % 11).astype(np.int16))
    transform = A.Compose(
        [
            A.Affine3D(
                rotate_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (90.0, 90.0)},
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        strict=True,
    )

    result = transform(volume=tensor_volume, mask3d=tensor_mask)

    assert result["volume"].dtype == tensor_volume.dtype
    assert result["mask3d"].dtype == tensor_mask.dtype
    torch.testing.assert_close(result["volume"], torch.rot90(tensor_volume, dims=(2, 3)), rtol=0, atol=0)
    expected_mask = torch.from_numpy(np.rot90(tensor_mask.numpy(), axes=(1, 2)).copy())
    torch.testing.assert_close(result["mask3d"], expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_affine3d_batch_matches_individual_volumes_and_channel_less_masks(tensor: bool, dtype: np.dtype) -> None:
    volumes = (np.arange(3 * 4 * 5 * 7 * 3).reshape(3, 4, 5, 7, 3) % 251).astype(dtype)
    if dtype == np.float32:
        volumes /= 251
    masks3d = (np.arange(2 * 4 * 5 * 7).reshape(2, 4, 5, 7) % 17).astype(np.int16)
    if tensor:
        volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
        masks3d = torch.from_numpy(masks3d)
    transform = A.Compose(
        [
            A.Affine3D(
                rotate_range={"x": (7.0, 7.0), "y": (-5.0, -5.0), "z": (11.0, 11.0)},
                translate_percent_range={"x": (0.2, 0.2), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                fill=(17, 23, 31),
                fill_mask=19,
                p=1.0,
            ),
        ],
        strict=True,
        telemetry=False,
    )

    batch = transform(volumes=volumes, masks3d=masks3d)
    expected_volumes = [transform(volume=volume)["volume"] for volume in volumes]
    expected_masks = [transform(mask3d=mask3d)["mask3d"] for mask3d in masks3d]

    if tensor:
        torch.testing.assert_close(batch["volumes"], torch.stack(expected_volumes), rtol=0, atol=0)
        torch.testing.assert_close(batch["masks3d"], torch.stack(expected_masks), rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(batch["volumes"], np.stack(expected_volumes))
        np.testing.assert_array_equal(batch["masks3d"], np.stack(expected_masks))
    assert batch["masks3d"].dtype == masks3d.dtype


@pytest.mark.parametrize("tensor", [False, True])
def test_affine3d_batch_uses_one_kernel_call_per_target(monkeypatch: pytest.MonkeyPatch, tensor: bool) -> None:
    from albumentations.augmentations.transforms3d import functional as f3d

    volumes = np.arange(2 * 3 * 4 * 5 * 2, dtype=np.uint8).reshape(2, 3, 4, 5, 2)
    masks3d = (volumes % 7).astype(np.int16)
    if tensor:
        volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
        masks3d = torch.from_numpy(masks3d).permute(0, 4, 1, 2, 3)
    calls = []
    original_warp = f3d.warp_affine3d

    def record_warp(data, *args, **kwargs):
        calls.append(data.shape)
        return original_warp(data, *args, **kwargs)

    monkeypatch.setattr(f3d, "warp_affine3d", record_warp)
    transform = A.Compose(
        [A.Affine3D(translate_percent_range={"x": (0.2, 0.2), "y": (0.0, 0.0), "z": (0.0, 0.0)}, p=1)],
        strict=True,
        telemetry=False,
    )

    result = transform(volumes=volumes, masks3d=masks3d)

    assert calls == [(2, 2, 3, 4, 5) if tensor else (2, 3, 4, 5, 2)] * 2
    assert result["volumes"].shape == volumes.shape
    assert result["masks3d"].dtype == masks3d.dtype


@pytest.mark.parametrize("tensor", [False, True])
def test_affine3d_empty_batches_keep_layout_and_dtype(tensor: bool) -> None:
    volumes = np.empty((0, 3, 4, 5, 2), dtype=np.float32)
    masks3d = np.empty((0, 3, 4, 5), dtype=np.int16)
    if tensor:
        volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
        masks3d = torch.from_numpy(masks3d)
    transform = A.Compose([A.Affine3D(p=1)], strict=True, telemetry=False)

    result = transform(volumes=volumes, masks3d=masks3d)

    assert result["volumes"].shape == volumes.shape
    assert result["masks3d"].shape == masks3d.shape
    assert result["volumes"].dtype == volumes.dtype
    assert result["masks3d"].dtype == masks3d.dtype
