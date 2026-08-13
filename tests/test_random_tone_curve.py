import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel

FLOAT_SHAPE_PREFIXES = {
    "image": (2, 3),
    "images": (2, 2, 3),
    "volume": (3, 2, 3),
}


def _make_float_input(shape: tuple[int, ...]) -> np.ndarray:
    values = np.array([0.0, 0.123456, 0.50123, 0.876543, 1.0], dtype=np.float32)
    return np.resize(values, int(np.prod(shape))).reshape(shape)


def _cubic_tone_curve(
    image: np.ndarray,
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> np.ndarray:
    low_y_float32 = np.asarray(low_y, dtype=np.float32)
    high_y_float32 = np.asarray(high_y, dtype=np.float32)
    one_minus_image = np.float32(1) - image
    result = (
        np.float32(3) * one_minus_image**2 * image * low_y_float32
        + np.float32(3) * one_minus_image * image**2 * high_y_float32
        + image**3
    )
    return np.clip(result, np.float32(0), np.float32(1))


def _broadcast_horner_tone_curve(
    image: np.ndarray,
    low_y: np.ndarray,
    high_y: np.ndarray,
) -> np.ndarray:
    low_y_float32 = np.asarray(low_y, dtype=np.float32)
    high_y_float32 = np.asarray(high_y, dtype=np.float32)
    coefficient_1 = np.float32(3) * low_y_float32
    coefficient_2 = np.float32(3) * high_y_float32 - np.float32(6) * low_y_float32
    coefficient_3 = np.float32(3) * low_y_float32 - np.float32(3) * high_y_float32 + np.float32(1)

    result = np.empty_like(image)
    np.multiply(image, coefficient_3, out=result)
    np.add(result, coefficient_2, out=result)
    np.multiply(result, image, out=result)
    np.add(result, coefficient_1, out=result)
    np.multiply(result, image, out=result)
    np.clip(result, np.float32(0), np.float32(1), out=result)
    result[image == np.float32(1)] = np.float32(1)
    return result


@pytest.mark.parametrize(("rank_name", "shape_prefix"), FLOAT_SHAPE_PREFIXES.items())
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize("per_channel", [False, True])
def test_move_tone_curve_float32_matches_cubic_formula(
    rank_name: str,
    shape_prefix: tuple[int, ...],
    num_channels: int,
    per_channel: bool,
) -> None:
    image = _make_float_input((*shape_prefix, num_channels))
    original = image.copy()

    if per_channel:
        low_y: float | np.ndarray = np.linspace(0.08, 0.42, num_channels, dtype=np.float64)
        high_y: float | np.ndarray = np.linspace(0.53, 0.91, num_channels, dtype=np.float64)
    else:
        low_y = 0.173
        high_y = 0.731

    result = fpixel.move_tone_curve(image, low_y, high_y, num_channels)
    expected = _cubic_tone_curve(image, low_y, high_y)

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(result[image == 0], np.float32(0))
    np.testing.assert_array_equal(result[image == 1], np.float32(1))
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(result, image)
    assert result.dtype == np.float32
    assert result.shape == image.shape
    assert np.all((result >= 0) & (result <= 1)), rank_name


@pytest.mark.parametrize(("rank_name", "shape_prefix"), FLOAT_SHAPE_PREFIXES.items())
def test_move_tone_curve_float32_rgb_channel_loop_matches_broadcast(
    rank_name: str,
    shape_prefix: tuple[int, ...],
) -> None:
    expanded_shape = (*shape_prefix[:-1], shape_prefix[-1] * 2, 3)
    image = _make_float_input(expanded_shape)[..., ::2, :]
    original = image.copy()
    low_y = np.array([0.08, 0.25, 0.42], dtype=np.float64)
    high_y = np.array([0.53, 0.72, 0.91], dtype=np.float64)

    assert not image.flags.c_contiguous, rank_name
    result = fpixel.move_tone_curve(image, low_y, high_y, 3)
    expected = _broadcast_horner_tone_curve(image, low_y, high_y)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(result, image)


def test_move_tone_curve_uint8_shared_golden_vector() -> None:
    image = np.array([0, 1, 2, 17, 64, 127, 128, 191, 254, 255], dtype=np.uint8).reshape(2, 5, 1)
    expected = np.array([0, 0, 1, 7, 47, 127, 128, 208, 255, 255], dtype=np.uint8).reshape(image.shape)

    result = fpixel.move_tone_curve(image, 0.1, 0.9, 1)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("low_y", "high_y"),
    [
        (0.80094105, 0.27785528),
        (
            np.array([0.80094105, 0.173, 0.091], dtype=np.float64),
            np.array([0.27785528, 0.731, 0.643], dtype=np.float64),
        ),
    ],
)
def test_move_tone_curve_float32_exact_asymmetric_endpoints(
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> None:
    num_channels = low_y.size if isinstance(low_y, np.ndarray) else 1
    image = np.array([0, 1], dtype=np.float32).reshape(2, 1, 1)
    image = np.broadcast_to(image, (2, 1, num_channels)).copy()

    result = fpixel.move_tone_curve(image, low_y, high_y, num_channels)

    np.testing.assert_array_equal(result[0], np.float32(0))
    np.testing.assert_array_equal(result[1], np.float32(1))


def test_move_tone_curve_uint8_per_channel_golden_vector() -> None:
    values = np.array([0, 1, 2, 17, 64, 127, 128, 191, 254, 255], dtype=np.uint8).reshape(2, 5)
    image = np.stack([values] * 3, axis=-1)
    low_y = np.array([0.1, 0.25, 0.4], dtype=np.float64)
    high_y = np.array([0.6, 0.75, 0.9], dtype=np.float64)
    expected = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 2, 2],
            [6, 14, 21],
            [36, 58, 80],
            [98, 127, 156],
            [99, 128, 157],
            [175, 197, 219],
            [254, 254, 255],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    ).reshape(image.shape)

    result = fpixel.move_tone_curve(image, low_y, high_y, 3)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(8, 8, 1), (2, 8, 8, 1)])
def test_move_tone_curve_uint8_per_channel_preserves_channel_dimension(shape: tuple[int, ...]) -> None:
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    low_y = np.array([0.1], dtype=np.float64)
    high_y = np.array([0.9], dtype=np.float64)

    result = fpixel.move_tone_curve(image, low_y, high_y, num_channels=1)
    expected = fpixel.move_tone_curve(image, 0.1, 0.9, num_channels=1)

    np.testing.assert_array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    ("low_y", "high_y"),
    [
        (0.2, np.array([0.8], dtype=np.float32)),
        (np.array([0.2], dtype=np.float32), 0.8),
    ],
)
def test_move_tone_curve_rejects_mixed_control_types(
    dtype: type[np.generic],
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> None:
    image = np.zeros((2, 3, 1), dtype=dtype)

    with pytest.raises(TypeError, match=r"must both be of type float or np.ndarray"):
        fpixel.move_tone_curve(image, low_y, high_y, 1)


@pytest.mark.parametrize(("target", "shape_prefix"), FLOAT_SHAPE_PREFIXES.items())
@pytest.mark.parametrize("per_channel", [False, True])
def test_random_tone_curve_compose_float32_target_routing(
    target: str,
    shape_prefix: tuple[int, ...],
    per_channel: bool,
) -> None:
    num_channels = 3
    image = _make_float_input((*shape_prefix, num_channels))
    transform = A.Compose(
        [A.RandomToneCurve(scale=0.0, per_channel=per_channel, p=1.0)],
        seed=137,
        strict=True,
    )
    low_y: float | np.ndarray = np.full(num_channels, 0.25, dtype=np.float64) if per_channel else 0.25
    high_y: float | np.ndarray = np.full(num_channels, 0.75, dtype=np.float64) if per_channel else 0.75
    expected = fpixel.move_tone_curve(image, low_y, high_y, num_channels)

    result = transform(**{target: image})[target]

    np.testing.assert_array_equal(result, expected)
