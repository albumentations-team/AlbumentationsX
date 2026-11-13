import numpy as np
import pytest
import torch

import albumentations as A

pytestmark = pytest.mark.pytorch


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("batched", [False, True])
def test_to_tensor_3d_preserves_interleaved_volume_and_mask_bits(dtype, batched):
    rng = np.random.default_rng(137)
    shape = (2, 5, 43, 47, 4) if batched else (5, 43, 47, 4)
    bits_dtype = {np.uint8: np.uint8, np.float16: np.uint16, np.float32: np.uint32}[dtype]
    volume = rng.integers(0, np.iinfo(bits_dtype).max, shape, dtype=bits_dtype, endpoint=True).view(dtype)
    volume.setflags(write=False)
    volume_key, mask_key = ("volumes", "masks3d") if batched else ("volume", "mask3d")
    pipeline = A.Compose([A.ToTensor3D()], telemetry=False)

    result = pipeline(**{volume_key: volume, mask_key: volume})

    expected = np.moveaxis(volume, -1, 1 if batched else 0).view(bits_dtype)
    for key in (volume_key, mask_key):
        assert result[key].is_contiguous()
        assert result[key].numpy().dtype == dtype
        np.testing.assert_array_equal(result[key].numpy().view(bits_dtype), expected)


def test_to_tensor_3d_materializes_flipped_scalar_mask():
    mask = np.arange(4 * 37 * 43, dtype=np.uint8).reshape(4, 37, 43)
    pipeline = A.Compose([A.HorizontalFlip(p=1), A.ToTensor3D()], telemetry=False)

    result = pipeline(mask3d=mask)["mask3d"]

    assert result.is_contiguous()
    np.testing.assert_array_equal(result.numpy(), mask[None, ..., ::-1])


test_cases = [
    pytest.param(
        (64, 64, 64, 3),  # volume shape
        (64, 64, 64, 1),  # mask shape
        (3, 64, 64, 64),  # expected volume shape
        (1, 64, 64, 64),  # expected mask shape
        id="rgb_volume_single_channel_mask",
    ),
    pytest.param(
        (64, 64, 64),  # volume shape
        (64, 64, 64),  # mask shape
        (1, 64, 64, 64),  # expected volume shape
        (1, 64, 64, 64),  # expected mask shape
        id="grayscale_volume_and_mask",
    ),
    pytest.param(
        (64, 64, 64, 3),  # volume shape
        (64, 64, 64, 4),  # mask shape
        (3, 64, 64, 64),  # expected volume shape
        (4, 64, 64, 64),  # expected mask shape
        id="rgb_volume_multi_channel_mask",
    ),
]


@pytest.mark.parametrize(
    "volume_shape,mask_shape,expected_volume_shape,expected_mask_shape",
    test_cases,
)
def test_to_tensor_3d_shapes(
    volume_shape,
    mask_shape,
    expected_volume_shape,
    expected_mask_shape,
):
    transform = A.Compose([A.ToTensor3D(p=1)])
    volume = np.random.randint(0, 256, volume_shape, dtype=np.uint8)
    mask3d = np.random.randint(0, 2, mask_shape, dtype=np.uint8)

    transformed = transform(volume=volume, mask3d=mask3d)

    assert isinstance(transformed["volume"], torch.Tensor)
    assert isinstance(transformed["mask3d"], torch.Tensor)
    assert transformed["volume"].shape == expected_volume_shape
    assert transformed["mask3d"].shape == expected_mask_shape


error_test_cases = [
    pytest.param(
        (64, 64),  # 2D array
        TypeError,
        r"volume must have shape \(D, H, W\) or \(D, H, W, C\), got",
        id="2d_array",
    ),
    pytest.param(
        (64, 64, 64, 3, 1),  # 5D array
        TypeError,
        r"volume must have shape \(D, H, W\) or \(D, H, W, C\), got",
        id="5d_array",
    ),
]


@pytest.mark.parametrize(
    "volume_shape,expected_error,expected_message",
    error_test_cases,
)
def test_to_tensor_3d_errors(volume_shape, expected_error, expected_message):
    transform = A.Compose([A.ToTensor3D(p=1)])
    volume = np.random.rand(*volume_shape)

    with pytest.raises(expected_error, match=expected_message):
        transform(volume=volume)


def test_to_tensor_3d_converts_volume_and_mask_collections() -> None:
    volumes = np.arange(2 * 3 * 4 * 5 * 2, dtype=np.uint8).reshape(2, 3, 4, 5, 2)
    masks3d = np.arange(3 * 3 * 4 * 5, dtype=np.uint8).reshape(3, 3, 4, 5)

    result = A.Compose([A.ToTensor3D(p=1)])(volumes=volumes, masks3d=masks3d)

    torch.testing.assert_close(result["volumes"], torch.from_numpy(volumes.transpose(0, 4, 1, 2, 3)))
    torch.testing.assert_close(result["masks3d"], torch.from_numpy(masks3d[:, None, ...]))
