import json

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.core.transform_params import SampledParams

from .utils import get_resolved_applied_params


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_exposure_matching_preserves_shape_dtype_and_normalized_mean(
    dtype: type[np.generic],
    channels: int,
) -> None:
    value = 51 if dtype == np.uint8 else 0.2
    expected_value = 102 if dtype == np.uint8 else 0.4
    image = np.full((8, 12, channels), value, dtype=dtype)
    transform = A.Compose([A.ExposureMatching(target_mean_range=(0.4, 0.4), p=1.0)])

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_allclose(result, expected_value, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (np.uint8, [25, 51, 77], [50, 102, 154]),
        (np.float32, [0.1, 0.2, 0.3], [0.2, 0.4, 0.6]),
    ],
)
def test_exposure_matching_uses_global_mean_across_channels(
    dtype: type[np.generic],
    values: list[float],
    expected: list[float],
) -> None:
    image = np.asarray(values, dtype=dtype).reshape(1, 1, 3)
    transform = A.Compose([A.ExposureMatching(target_mean_range=(0.4, 0.4), p=1.0)])

    result = transform(image=image)["image"]

    np.testing.assert_allclose(result, np.asarray(expected, dtype=dtype).reshape(1, 1, 3), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("value", "gain_range", "expected_gain", "expected_value"),
    [
        (0.0, None, 400_000.0, 0.0),
        (1e-8, None, 400_000.0, 0.004),
        (1e-8, (1.0, 3.0), 3.0, 3e-8),
    ],
)
def test_exposure_matching_handles_zero_and_near_zero_means(
    value: float,
    gain_range: tuple[float, float] | None,
    expected_gain: float,
    expected_value: float,
) -> None:
    image = np.full((4, 6, 3), value, dtype=np.float32)
    transform = A.ExposureMatching(target_mean_range=(0.4, 0.4), gain_range=gain_range, p=1.0)

    result = transform(image=image)["image"]

    np.testing.assert_allclose(result, expected_value, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(get_resolved_applied_params(transform)["gain"], expected_gain, rtol=1e-6)


def test_exposure_matching_clips_saturated_pixels_without_compensation() -> None:
    image = np.array([[[0.1], [0.9]]], dtype=np.float32)
    transform = A.Compose([A.ExposureMatching(target_mean_range=(1.0, 1.0), p=1.0)])

    result = transform(image=image)["image"]

    np.testing.assert_allclose(result, np.array([[[0.2], [1.0]]], dtype=np.float32))
    assert float(result.mean()) < 1.0


@pytest.mark.parametrize(
    ("image_value", "target_mean", "gain_range", "expected_gain", "expected_value"),
    [
        (0.2, 0.8, (1.0, 3.0), 3.0, 0.6),
        (0.8, 0.2, (0.5, 3.0), 0.5, 0.4),
    ],
)
def test_exposure_matching_clips_derived_gain(
    image_value: float,
    target_mean: float,
    gain_range: tuple[float, float],
    expected_gain: float,
    expected_value: float,
) -> None:
    image = np.full((5, 7, 3), image_value, dtype=np.float32)
    transform = A.ExposureMatching(target_mean_range=(target_mean, target_mean), gain_range=gain_range, p=1.0)

    result = transform(image=image)["image"]

    np.testing.assert_allclose(result, expected_value, rtol=0, atol=1e-6)
    np.testing.assert_allclose(get_resolved_applied_params(transform)["gain"], expected_gain)


@pytest.mark.parametrize("target", ["images", "volume"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_exposure_matching_derives_one_gain_per_image_or_slice(
    target: str,
    dtype: type[np.generic],
) -> None:
    values = np.array([51, 102, 204], dtype=np.uint8) if dtype == np.uint8 else np.array([0.2, 0.4, 0.8])
    data = np.stack([np.full((4, 6, 2), value, dtype=dtype) for value in values])
    transform = A.ExposureMatching(target_mean_range=(0.4, 0.4), p=1.0)

    result = transform(**{target: data})[target]

    expected_value = 102 if dtype == np.uint8 else 0.4
    np.testing.assert_allclose(result, expected_value, rtol=0, atol=1e-6)
    np.testing.assert_allclose(get_resolved_applied_params(transform, target)["gain"], [2.0, 1.0, 0.5])


def test_exposure_matching_keeps_gains_separate_for_simultaneous_target_routes() -> None:
    image = np.full((3, 5, 2), 0.2, dtype=np.float32)
    images = np.stack([image, np.full_like(image, 0.4)])
    volume = np.stack([image, np.full_like(image, 0.8), np.full_like(image, 0.4)])
    transform = A.ExposureMatching(target_mean_range=(0.4, 0.4), p=1.0)

    result = transform(image=image, images=images, volume=volume)
    params = SampledParams.from_dict(transform.get_applied_params())

    for target in ("image", "images", "volume"):
        np.testing.assert_allclose(result[target], 0.4, rtol=0, atol=1e-6)
    np.testing.assert_allclose(params.params_for("image")["gain"], 2.0)
    np.testing.assert_allclose(params.params_for("images")["gain"], [2.0, 1.0])
    np.testing.assert_allclose(params.params_for("volume")["gain"], [2.0, 0.5, 1.0])


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_get_exposure_gains_vectorizes_across_all_leading_dimensions(dtype: type[np.generic]) -> None:
    values = (
        np.array([[0, 51, 102], [204, 255, 25]], dtype=np.uint8)
        if dtype == np.uint8
        else np.array([[0.0, 0.2, 0.4], [0.8, 1.0, 0.1]], dtype=np.float32)
    )
    images = np.empty((2, 3, 4, 5, 2), dtype=dtype)
    images[...] = values[..., None, None, None]

    gains = fpixel.get_exposure_gains(images, target_mean=0.4, gain_range=(0.5, 3.0))

    normalized_values = values.astype(np.float64) / (255 if dtype == np.uint8 else 1)
    expected = np.clip(0.4 / np.maximum(normalized_values, 1e-6), 0.5, 3.0)
    assert isinstance(gains, np.ndarray)
    assert gains.shape == values.shape
    np.testing.assert_allclose(gains, expected, rtol=0, atol=1e-15)


def test_exposure_match_batch_validates_gain_shape() -> None:
    images = np.zeros((2, 3, 4, 5, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Expected exposure gains with shape \(2, 3\), got \(2,\)"):
        fpixel.exposure_match_batch(images, np.ones(2, dtype=np.float32))


def test_exposure_matching_records_sampled_target_and_gains_for_replay() -> None:
    images = np.stack(
        [
            np.full((4, 6, 3), 0.2, dtype=np.float32),
            np.full((4, 6, 3), 0.8, dtype=np.float32),
        ],
    )
    transform = A.ExposureMatching(target_mean_range=(0.3, 0.5), gain_range=(0.25, 3.0), p=1.0)
    pipeline = A.Compose([transform], save_applied_params=True, seed=137)

    result = pipeline(images=images)
    params = SampledParams.from_dict(transform.get_applied_params())
    applied_transforms = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    _, applied_config = applied_transforms[0]

    assert 0.3 <= params.shared["target_mean"] <= 0.5
    np.testing.assert_allclose(
        params.params_for("images")["gain"],
        [params.shared["target_mean"] / 0.2, params.shared["target_mean"] / 0.8],
    )
    assert applied_config["target_mean_range"] == params.shared["target_mean"]
    assert applied_config["gain_range"] == [0.25, 3.0]

    replay = A.Compose.from_applied_transforms(applied_transforms)
    replayed = replay(images=images)
    np.testing.assert_array_equal(replayed["images"], result["images"])


@pytest.mark.parametrize("target", ["images", "volume"])
def test_exposure_matching_replay_compose_reuses_applied_gains(target: str) -> None:
    images = np.stack(
        [
            np.full((4, 6, 3), 0.2, dtype=np.float32),
            np.full((4, 6, 3), 0.8, dtype=np.float32),
        ],
    )
    data = images
    pipeline = A.ReplayCompose([A.ExposureMatching(target_mean_range=(0.3, 0.5), p=1.0)], seed=137)

    result = pipeline(**{target: data})
    replay_params = SampledParams.from_dict(result["replay"]["transforms"][0]["params"])
    replayed = A.ReplayCompose.replay(result["replay"], **{target: data})

    assert 0.3 <= replay_params.shared["target_mean"] <= 0.5
    assert len(replay_params.params_for(target)["gain"]) == len(data)
    np.testing.assert_array_equal(replayed[target], result[target])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_mean_range": (-0.1, 0.5)}, "must be >= 0"),
        ({"target_mean_range": (0.6, 0.5)}, "First value should be less than"),
        ({"target_mean_range": (0.5, 1.1)}, "must be >= 0 and <= 1"),
        ({"gain_range": (-0.1, 2.0)}, "must be >= 0"),
        ({"gain_range": (2.0, 1.0)}, "First value should be less than"),
    ],
)
def test_exposure_matching_validates_ranges(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        A.ExposureMatching(**kwargs)
