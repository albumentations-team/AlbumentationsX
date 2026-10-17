import numpy as np
import pytest
from albucore.utils import get_max_value

from albumentations import Compose
from albumentations.augmentations import FromFloat, ToFloat, ToGray, ToRGB


@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32"])
@pytest.mark.parametrize(
    "param, shape",
    [
        ("image", (8, 7, 6)),
        ("image", (8, 7)),
        ("images", (4, 8, 7, 6)),
        ("images", (4, 8, 7)),
        ("volume", (3, 8, 7, 6)),
        ("volume", (3, 8, 7)),
        ("volumes", (2, 3, 8, 7, 6)),
        ("volumes", (2, 3, 8, 7)),
    ],
)
def test_to_float(param, shape, dtype):
    rng = np.random.default_rng(137)
    data = rng.uniform(0, 10, size=shape).astype(dtype)

    aug = ToFloat()
    result = aug(**{param: data})[param]

    assert result.dtype == (np.float16 if dtype == "float16" else np.float32)
    assert result.shape == data.shape
    np.testing.assert_allclose(data, result * get_max_value(np.dtype(dtype)))


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
def test_to_float_preserves_half_bits_through_compose(target):
    spatial = (2, 3) if target in {"image", "images"} else (2, 2, 3)
    shape = ((2,) if target in {"images", "volumes"} else ()) + (*spatial, 1)
    values = np.array([0x8000, 0x3800, 0x3C00, 0x3E00, 0x7C00, 0x7C01], dtype=np.uint16).view(np.float16)
    image = np.resize(values, shape)
    original = image.view(np.uint16).copy()
    image.setflags(write=False)
    pipeline = Compose([ToFloat(max_value=7.5)], seed=137, strict=True)

    result = pipeline(**{target: image})[target]

    assert result.dtype == np.float16
    np.testing.assert_array_equal(result.view(np.uint16), original)
    np.testing.assert_array_equal(image.view(np.uint16), original)


@pytest.mark.parametrize("dtype", ["uint8"])
@pytest.mark.parametrize(
    "param, shape",
    [
        ("image", (8, 7, 6)),
        ("image", (8, 7)),
        ("images", (4, 8, 7, 6)),
        ("images", (4, 8, 7)),
    ],
)
def test_from_float(param, shape, dtype):
    rng = np.random.default_rng()
    data = rng.random(size=shape, dtype=np.float32)

    aug = FromFloat(dtype=dtype)
    result = aug(**{param: data})[param]

    assert result.dtype == np.dtype(dtype)
    assert result.shape == data.shape
    # Because FromFloat has to round to the nearest integer, we get an absolute difference up to 0.5
    np.testing.assert_allclose(data * get_max_value(np.dtype(dtype)), result, atol=0.5)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize(
    "target,shape",
    [
        ("image", (1, 7, 1)),
        ("image", (7, 1, 1)),
        ("image", (1, 1, 1)),
        ("images", (1, 5, 7, 1)),
        ("volume", (1, 5, 7, 1)),
        ("volumes", (1, 1, 3, 7, 1)),
    ],
)
def test_to_rgb_keeps_singleton_spatial_and_collection_axes(dtype, target, shape):
    source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    if dtype != np.uint8:
        source = (source.astype(np.float32) / 137).astype(dtype)
    source.flags.writeable = False
    original = source.copy()
    expected = np.broadcast_to(source, (*shape[:-1], 3))

    result = Compose([ToRGB(p=1)], strict=True)(**{target: source})[target]

    assert result.shape == expected.shape
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize(
    "method,dtype",
    [
        ("average", np.uint8),
        ("average", np.float16),
        ("average", np.float32),
        ("max", np.uint8),
        ("max", np.float16),
        ("max", np.float32),
        ("desaturation", np.uint8),
        ("desaturation", np.float32),
    ],
)
@pytest.mark.parametrize(
    "target,shape",
    [
        ("images", (2, 5, 1, 5)),
        ("volume", (2, 5, 1, 5)),
        ("volumes", (1, 2, 5, 1, 5)),
    ],
)
def test_to_gray_keeps_singleton_width_after_channel_reduction(method, dtype, target, shape):
    source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    if dtype != np.uint8:
        source = (source.astype(np.float32) / 137).astype(dtype)
    source.flags.writeable = False
    original = source.copy()
    if method == "average":
        gray = source.mean(axis=-1, dtype=np.float64).astype(dtype)
    elif method == "max":
        gray = source.max(axis=-1)
    else:
        floating = source.astype(np.float32)
        gray = ((floating.max(axis=-1) + floating.min(axis=-1)) * 0.5).astype(dtype)
    expected = np.broadcast_to(gray[..., np.newaxis], (*shape[:-1], 3))

    result = Compose([ToGray(method=method, p=1)], strict=True)(**{target: source})[target]

    assert result.shape == expected.shape
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("shape", [(1, 7), (7, 1), (1, 1)])
def test_to_rgb_normalizes_channel_less_input_at_compose_boundary(dtype, shape):
    source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape).astype(dtype)
    source.flags.writeable = False
    original = source.copy()

    result = Compose([ToRGB(p=1)], strict=True)(image=source)["image"]

    assert result.shape == (*shape, 3)
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result, np.broadcast_to(source[..., np.newaxis], result.shape))
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("method", ["weighted_average", "from_lab", "desaturation", "average", "max", "pca"])
def test_single_channel_gray_can_feed_another_transform(method, dtype):
    source = np.random.default_rng(137).integers(0, 256, (5, 1, 3), dtype=np.uint8)
    if dtype != np.uint8:
        source = source.astype(dtype) / 255
    expected = Compose([ToGray(method=method, num_output_channels=3, p=1)], strict=True)(image=source)["image"]

    result = Compose([ToGray(method=method, num_output_channels=1, p=1), ToRGB(p=1)], strict=True)(image=source)[
        "image"
    ]

    assert result.shape == source.shape
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result, expected)
