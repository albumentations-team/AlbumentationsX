"""Independent interpolation, gain, and public target contracts for MRI bias fields."""

import numpy as np
import pytest
from scipy import ndimage

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel


def _reference_bias_field(img: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    spatial_shape = img.shape[: coarse.ndim - 1]
    coordinates = np.meshgrid(
        *[
            (np.arange(size, dtype=np.float64) + 0.5) * coarse_size / size - 0.5
            for size, coarse_size in zip(spatial_shape, coarse.shape[:-1], strict=True)
        ],
        indexing="ij",
    )
    log_gain = np.stack(
        [
            ndimage.map_coordinates(coarse[..., channel].astype(np.float64), coordinates, order=1, mode="nearest")
            for channel in range(coarse.shape[-1])
        ],
        axis=-1,
    )
    signal = img.astype(np.float64)
    if img.dtype == np.uint8:
        signal /= 255
    result = np.clip(signal * np.exp(log_gain), 0, 1)
    return np.rint(result * 255).astype(np.uint8) if img.dtype == np.uint8 else result.astype(np.float32)


@pytest.mark.parametrize("shape", [(9, 13, 1), (8, 12, 5), (5, 9, 13, 3), (1, 9, 13, 1), (5, 1, 13, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("per_channel", [False, True])
def test_bias_field_matches_independent_half_pixel_interpolation(
    shape: tuple[int, ...],
    dtype: type[np.generic],
    per_channel: bool,
) -> None:
    rng = np.random.default_rng(137)
    raw = rng.integers(0, 256, shape, dtype=np.uint8)
    img = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    coarse_shape = (*[min(size, 2) for size in shape[:-1]], shape[-1] if per_channel else 1)
    coarse = rng.standard_normal(coarse_shape, dtype=np.float32) * 0.3
    original, original_coarse = img.copy(), coarse.copy()
    img.setflags(write=False)

    result = fpixel.bias_field(img, coarse)

    expected = _reference_bias_field(img, coarse)
    np.testing.assert_allclose(result, expected, atol=1 if dtype == np.uint8 else 2e-6, rtol=1e-5)
    assert result.dtype == dtype
    np.testing.assert_array_equal(img, original)
    np.testing.assert_array_equal(coarse, original_coarse)


@pytest.mark.parametrize("spatial_shape", [(3, 5), (3, 4, 5)])
def test_full_resolution_grid_is_exponentiated_without_mutating_coefficients(spatial_shape: tuple[int, ...]) -> None:
    img = np.full((*spatial_shape, 5), 0.25, dtype=np.float32)
    log_gains = np.log(np.array([1, 2, 0.5, 3, 0.25], dtype=np.float32))
    coarse = np.broadcast_to(log_gains, img.shape)
    original = coarse.copy()

    result = fpixel.bias_field(img, coarse)

    expected = np.broadcast_to(np.array([0.25, 0.5, 0.125, 0.75, 0.0625], dtype=np.float32), img.shape)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    np.testing.assert_array_equal(coarse, original)


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_zero_strength_skips_field_application_and_is_exact_identity(
    target: str,
    dtype: type[np.generic],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spatial_shape = (9, 13) if target in {"image", "images"} else (5, 9, 13)
    shape = ((2,) if target in {"images", "volumes"} else ()) + (*spatial_shape, 3)
    raw = np.random.default_rng(137).integers(0, 256, shape, dtype=np.uint8)
    data = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    original = data.copy()
    data.setflags(write=False)

    def unexpected_kernel(*args: object, **kwargs: object) -> None:
        raise AssertionError("A zero-strength field must not enter the reconstruction kernel")

    monkeypatch.setattr(fpixel, "bias_field", unexpected_kernel)
    pipeline = A.Compose([A.BiasField(std_range=(0, 0), p=1)], seed=137, strict=True)

    result = pipeline(**{target: data})[target]

    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("target", ["images", "volumes"])
@pytest.mark.parametrize("per_channel", [False, True])
def test_collection_shares_field_and_keeps_channel_semantics(target: str, per_channel: bool) -> None:
    spatial_shape = (9, 13) if target == "images" else (5, 9, 13)
    signal = np.full((*spatial_shape, 5), 0.2, dtype=np.float32)
    data = np.stack([signal, signal * 0.5])
    original = data.copy()
    pipeline = A.Compose(
        [A.BiasField(std_range=(0.3, 0.3), per_channel=per_channel, p=1)],
        seed=137,
        strict=True,
    )

    result = pipeline(**{target: data})[target]

    np.testing.assert_allclose(result[1], result[0] * 0.5, atol=1e-6)
    np.testing.assert_array_equal(data, original)
    if per_channel:
        assert not np.allclose(result[0, ..., 0], result[0, ..., 1])
    else:
        np.testing.assert_array_equal(result[0, ..., 0], result[0, ..., 1])
    if target == "volumes":
        assert not np.allclose(result[0, 0], result[0, -1])


@pytest.mark.parametrize("target", ["image", "volume"])
def test_replay_seed_and_annotations_preserve_the_realized_coarse_field(target: str) -> None:
    shape = (9, 13, 3) if target == "image" else (5, 9, 13, 3)
    data = np.full(shape, 0.25, dtype=np.float32)
    mask_key = "mask" if target == "image" else "mask3d"
    mask = np.random.default_rng(137).integers(0, 4, shape[:-1], dtype=np.uint8)
    inputs = {target: data, "other": data.copy(), mask_key: mask}
    pipeline = A.ReplayCompose([A.BiasField(p=1)], additional_targets={"other": target}, strict=True)
    pipeline.set_random_seed(137)

    first = pipeline(**inputs)
    replay = A.ReplayCompose.replay(first["replay"], **inputs)
    groups = first["replay"]["transforms"][0]["params"]["target_params"]
    assert len(groups) == 1
    coarse = groups[0]["params"]["coarse_field"]
    assert coarse.size < data.size / 10
    for key in inputs:
        np.testing.assert_array_equal(first[key], replay[key])
    np.testing.assert_array_equal(first[target], first["other"])
    np.testing.assert_array_equal(first[mask_key], mask)
    assert not np.allclose(first[target], pipeline(**inputs)[target])


def test_bias_field_preserves_geometric_annotations() -> None:
    image = np.full((12, 16, 1), 0.25, dtype=np.float32)
    bboxes = np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32)
    keypoints = np.array([[4.0, 3.0]], dtype=np.float32)
    pipeline = A.Compose(
        [A.BiasField(std_range=(0.1, 0.3), scale_range=(0.05, 0.1), p=1)],
        bbox_params=A.BboxParams(coord_format="albumentations"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        seed=137,
        strict=True,
    )

    result = pipeline(image=image, bboxes=bboxes, keypoints=keypoints)
    np.testing.assert_array_equal(result["bboxes"], bboxes)
    np.testing.assert_array_equal(result["keypoints"], keypoints)


@pytest.mark.parametrize("per_channel", [False, True])
def test_channel_count_controls_coefficient_sharing_between_targets(per_channel: bool) -> None:
    inputs = {
        "image": np.full((9, 13, 3), 0.25, dtype=np.float32),
        "other": np.full((9, 13, 1), 0.25, dtype=np.float32),
    }
    pipeline = A.ReplayCompose(
        [A.BiasField(std_range=(0.3, 0.3), per_channel=per_channel, p=1)],
        additional_targets={"other": "image"},
        strict=True,
    )
    pipeline.set_random_seed(137)

    result = pipeline(**inputs)
    groups = result["replay"]["transforms"][0]["params"]["target_params"]
    if per_channel:
        assert len(groups) == 2
        assert sorted(group["params"]["coarse_field"].shape[-1] for group in groups) == [1, 3]
        assert not np.allclose(result["image"][..., 0], result["other"][..., 0])
    else:
        assert len(groups) == 1
        assert groups[0]["params"]["coarse_field"].shape[-1] == 1
        np.testing.assert_array_equal(result["image"][..., 0], result["other"][..., 0])
    replay = A.ReplayCompose.replay(result["replay"], **inputs)
    for key in inputs:
        np.testing.assert_array_equal(result[key], replay[key])


@pytest.mark.parametrize("per_channel", [False, True])
def test_coarse_resolution_preserves_singleton_depth(per_channel: bool) -> None:
    pipeline = A.ReplayCompose(
        [
            A.BiasField(std_range=(0.3, 0.3), scale_range=(0.5, 0.5), per_channel=per_channel, p=1),
        ]
    )
    pipeline.set_random_seed(137)

    result = pipeline(volume=np.full((1, 9, 13, 5), 0.25, dtype=np.float32))

    groups = result["replay"]["transforms"][0]["params"]["target_params"]
    assert groups[0]["params"]["coarse_field"].shape == (1, 4, 6, 5 if per_channel else 1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_extreme_strength_clips_and_zero_signal_remains_zero(dtype: type[np.generic]) -> None:
    volume = np.zeros((5, 9, 13, 2), dtype=dtype)
    volume[..., 0] = 255 if dtype == np.uint8 else 1
    pipeline = A.Compose(
        [
            A.BiasField(std_range=(1, 1), scale_range=(1, 1), per_channel=True, p=1),
        ],
        seed=137,
        strict=True,
    )

    result = pipeline(volume=volume)["volume"]

    np.testing.assert_array_equal(result[..., 1], 0)
    assert np.isfinite(result).all()
    assert result.dtype == dtype
    assert result[..., 0].max() == (255 if dtype == np.uint8 else 1)
    assert result[..., 0].min() < (255 if dtype == np.uint8 else 1)


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.complex64])
def test_unsupported_dtype_is_rejected(dtype: type[np.generic]) -> None:
    with pytest.raises(ValueError, match="uint8 or float32"):
        A.BiasField(p=1)(image=np.ones((9, 13, 1), dtype=dtype))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"std_range": (-0.1, 0.3)},
        {"std_range": (0.3, 0.1)},
        {"std_range": (0, 1.1)},
        {"std_range": (0, float("nan"))},
        {"scale_range": (0, 0.1)},
        {"scale_range": (0.2, 0.1)},
        {"scale_range": (0.01, float("inf"))},
    ],
)
def test_invalid_bias_field_policy_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A.BiasField(**kwargs, strict=True)
