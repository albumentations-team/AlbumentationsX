import numpy as np
import pytest
from albucore.utils import get_max_value

from albumentations import Compose
from albumentations.augmentations import FromFloat, ToFloat, ToRGB


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
@pytest.mark.parametrize(
    "param, shape",
    [
        ("image", (8, 7, 6)),
        ("image", (8, 7)),
        ("images", (4, 8, 7, 6)),
        ("images", (4, 8, 7)),
    ],
)
def test_to_float(param, shape, dtype):
    rng = np.random.default_rng()
    data = rng.uniform(0, 10, size=shape).astype(dtype)

    aug = ToFloat()
    result = aug(**{param: data})[param]

    assert result.dtype == np.float32
    assert result.shape == data.shape
    np.testing.assert_allclose(data, result * get_max_value(np.dtype(dtype)))


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
