import json
from typing import Any, cast

import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("clip_range", [(0.0, 1.0), (-1.0, 1.0)])
@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (60.0, 20.0),
        ((20.0, 40.0, 60.0, 80.0, 100.0), (5.0, 10.0, 15.0, 20.0, 25.0)),
    ],
    ids=["scalar-statistics", "per-channel-statistics"],
)
def test_normalize_standard_clips_output(
    dtype: type[np.generic],
    clip_range: tuple[float, float],
    mean: float | tuple[float, ...],
    std: float | tuple[float, ...],
) -> None:
    coefficients = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32).reshape(-1, 1, 1)
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    image = np.broadcast_to(mean_array + coefficients * std_array, (5, 1, 5)).astype(dtype).copy()
    original = image.copy()

    result = A.Compose(
        [
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=1.0,
                clip_range=clip_range,
                p=1.0,
            ),
        ],
    )(image=image)["image"]

    expected = (image.astype(np.float32) - mean_array) / std_array
    np.clip(expected, *clip_range, out=expected)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    ("target_name", "shape"),
    [
        ("image", (2, 3, 2)),
        ("images", (2, 2, 3, 2)),
        ("volume", (2, 2, 3, 2)),
    ],
)
def test_normalize_clip_range_applies_to_all_image_targets(target_name: str, shape: tuple[int, ...]) -> None:
    data = np.linspace(-2.0, 2.0, num=np.prod(shape), dtype=np.float32).reshape(shape)
    original = data.copy()
    transform = A.Compose(
        [A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0, clip_range=(-0.5, 0.75), p=1.0)],
    )

    result = transform(**{target_name: data})[target_name]

    np.testing.assert_array_equal(result, np.clip(original, -0.5, 0.75))
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("normalization", ["image", "image_per_channel", "min_max", "min_max_per_channel"])
def test_normalize_clip_range_applies_after_per_image_normalization(normalization: str) -> None:
    image = np.array(
        [
            [[0, 10, 100], [1, 30, 80], [2, 50, 60]],
            [[3, 70, 40], [4, 90, 20], [5, 110, 0]],
        ],
        dtype=np.float32,
    )
    baseline = A.Compose([A.Normalize(normalization=normalization, p=1.0)])(image=image)["image"]

    result = A.Compose([A.Normalize(normalization=normalization, clip_range=(0.25, 0.75), p=1.0)])(
        image=image,
    )["image"]

    np.testing.assert_array_equal(result, np.clip(baseline, 0.25, 0.75))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_normalize_explicit_none_preserves_existing_output(dtype: type[np.generic]) -> None:
    rng = np.random.default_rng(137)
    if dtype == np.uint8:
        image = rng.integers(0, 256, (16, 12, 5), dtype=dtype)
    else:
        image = rng.random((16, 12, 5), dtype=np.float32)
    kwargs = {
        "mean": (0.1, 0.2, 0.3, 0.4, 0.5),
        "std": (0.2, 0.3, 0.4, 0.5, 0.6),
        "max_pixel_value": 1.0,
        "p": 1.0,
    }

    expected = A.Compose([A.Normalize(**kwargs)])(image=image)["image"]
    result = A.Compose([A.Normalize(**kwargs, clip_range=None)])(image=image)["image"]

    np.testing.assert_array_equal(result, expected)


def test_normalize_clip_range_serialization_round_trip() -> None:
    image = np.array([[[-2.0], [0.5], [3.0]]], dtype=np.float32)
    transform = A.Normalize(
        mean=0.0,
        std=1.0,
        max_pixel_value=1.0,
        clip_range=(-1.0, 1.0),
        p=1.0,
    )
    serialized = json.loads(json.dumps(A.to_dict(transform), allow_nan=False))

    restored = A.from_dict(serialized)

    assert restored.clip_range == (-1.0, 1.0)
    np.testing.assert_array_equal(restored(image=image)["image"], transform(image=image)["image"])


@pytest.mark.parametrize(
    "clip_range",
    [
        (1.0, 0.0),
        (0.0,),
        (0.0, 0.5, 1.0),
        (np.nan, 1.0),
        (0.0, np.inf),
        (-np.inf, 1.0),
        ("low", "high"),
    ],
)
def test_normalize_rejects_invalid_clip_range(clip_range: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        A.Normalize(clip_range=cast("Any", clip_range))
