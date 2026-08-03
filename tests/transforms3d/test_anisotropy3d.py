import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.type_definitions import Targets
from tests.helpers import TestDataFactory


def _fixed_anisotropy(
    axes: tuple[int, ...] = (0, 2),
    antialias: bool = True,
) -> A.Anisotropy3D:
    return A.Anisotropy3D(
        axes=axes,
        num_axes_range=(len(axes), len(axes)),
        downscale_factor_range=(2.0, 2.0),
        antialias=antialias,
        p=1.0,
    )


def test_anisotropy3d_is_volume_only() -> None:
    transform = _fixed_anisotropy()

    assert isinstance(transform, A.VolumeOnlyTransform)
    assert transform._targets == (Targets.VOLUME,)
    assert set(transform.targets) == {"user_data", "volume"}


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_anisotropy3d_numpy_preserves_shape_dtype_range_and_mask(dtype: type[np.generic], channels: int) -> None:
    volume = TestDataFactory.create_volume((5, 11, 13, channels), dtype=dtype, seed=137)
    mask3d = TestDataFactory.create_volume((5, 11, 13, 1), dtype=np.uint8, seed=138)[..., 0]

    result = A.Compose([_fixed_anisotropy()], seed=137, strict=True)(volume=volume, mask3d=mask3d)

    transformed_volume = result["volume"]
    assert transformed_volume.shape == volume.shape
    assert transformed_volume.dtype == volume.dtype
    assert transformed_volume.min() >= volume.min()
    assert transformed_volume.max() <= volume.max()
    np.testing.assert_array_equal(result["mask3d"], mask3d)
    assert not np.array_equal(transformed_volume, volume)


@pytest.mark.parametrize(
    ("axis", "volume_shape"),
    [
        (0, (1, 11, 13, 2)),
        (1, (5, 1, 13, 2)),
        (2, (5, 11, 1, 2)),
    ],
)
def test_anisotropy3d_handles_unit_spatial_dimensions(axis: int, volume_shape: tuple[int, int, int, int]) -> None:
    volume = TestDataFactory.create_volume(volume_shape, seed=137)

    result = A.Compose([_fixed_anisotropy((axis,))], seed=137, strict=True)(volume=volume)["volume"]

    assert result.shape == volume.shape
    assert result.dtype == volume.dtype


def test_anisotropy3d_numpy_accepts_a_channel_less_volume() -> None:
    volume = TestDataFactory.create_volume((5, 11, 13, 1), seed=137)[..., 0]

    result = A.Compose([_fixed_anisotropy()], seed=137, strict=True)(volume=volume)["volume"]

    assert result.shape == volume.shape
    assert result.dtype == volume.dtype
    assert not np.array_equal(result, volume)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_anisotropy3d_tensor_preserves_channel_first_volume_and_mask3d(dtype: torch.dtype, channels: int) -> None:
    numpy_dtype = np.uint8 if dtype == torch.uint8 else np.float32
    volume = TestDataFactory.create_volume((5, 11, 13, channels), dtype=numpy_dtype, seed=137)
    tensor_volume = torch.from_numpy(np.ascontiguousarray(volume.transpose(3, 0, 1, 2)))
    mask3d = torch.from_numpy(TestDataFactory.create_volume((5, 11, 13, 1), seed=138)[..., 0])

    result = A.Compose([_fixed_anisotropy()], seed=137, strict=True)(volume=tensor_volume, mask3d=mask3d)

    transformed_volume = result["volume"]
    assert isinstance(transformed_volume, torch.Tensor)
    assert transformed_volume.shape == tensor_volume.shape
    assert transformed_volume.dtype == dtype
    assert torch.all(transformed_volume >= tensor_volume.min())
    assert torch.all(transformed_volume <= tensor_volume.max())
    torch.testing.assert_close(result["mask3d"], mask3d)
    assert not torch.equal(transformed_volume, tensor_volume)


def test_anisotropy3d_replay_reproduces_selected_axes_and_factor() -> None:
    volume = TestDataFactory.create_volume((7, 11, 13, 2), seed=137)
    transform = A.ReplayCompose(
        [
            A.Anisotropy3D(
                axes=(0, 1, 2),
                num_axes_range=(1, 3),
                downscale_factor_range=(1.5, 3.0),
                p=1.0,
            ),
        ],
        seed=137,
    )

    first_result = transform(volume=volume)
    replay_result = A.ReplayCompose.replay(first_result["replay"], volume=volume)

    np.testing.assert_array_equal(replay_result["volume"], first_result["volume"])


def test_anisotropy3d_keeps_keypoints_in_their_original_coordinate_system() -> None:
    volume = TestDataFactory.create_volume((5, 11, 13, 1), seed=137)
    keypoints = np.array([[2, 3, 1], [8, 10, 4]], dtype=np.float32)
    transform = A.Compose(
        [_fixed_anisotropy()],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        seed=137,
        strict=True,
    )

    result = transform(volume=volume, keypoints=keypoints)

    np.testing.assert_array_equal(result["keypoints"], keypoints)


def test_anisotropy3d_tensor_uses_current_non_antialiased_trilinear_fallback() -> None:
    volume = TestDataFactory.create_volume((5, 11, 13, 3), dtype=np.float32, seed=137)
    tensor_volume = torch.from_numpy(np.ascontiguousarray(volume.transpose(3, 0, 1, 2)))

    antialiased = A.Compose([_fixed_anisotropy(antialias=True)], seed=137, strict=True)(volume=tensor_volume)["volume"]
    non_antialiased = A.Compose([_fixed_anisotropy(antialias=False)], seed=137, strict=True)(volume=tensor_volume)[
        "volume"
    ]

    torch.testing.assert_close(antialiased, non_antialiased)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"axes": ()}, "at least one"),
        ({"axes": (0, 0)}, "duplicates"),
        ({"axes": (0,), "num_axes_range": (2, 2)}, "more axes"),
        ({"downscale_factor_range": (1.0, 2.0)}, "must be > 1"),
    ],
)
def test_anisotropy3d_validates_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        A.Anisotropy3D(**kwargs)
