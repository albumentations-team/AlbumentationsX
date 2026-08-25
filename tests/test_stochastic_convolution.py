import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.core.transform_params import SampledParams, SampledParamsError


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_stochastic_convolution_preserves_shape_dtype_and_range(dtype: np.dtype, channels: int) -> None:
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (23, 19, channels), dtype=np.uint8)
        if dtype == np.uint8
        else rng.random((23, 19, channels), dtype=np.float32)
    )
    transform = A.Compose(
        [
            A.StochasticConvolution(
                kernel_range=(3, 3),
                strength_range=(0.15, 0.15),
                per_channel=channels != 1,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()
    if dtype == np.uint8:
        assert result.min() >= 0
        assert result.max() <= 255
    else:
        assert result.min() >= 0
        assert result.max() <= 1


@pytest.mark.parametrize("channels", [1, 3, 5])
def test_zero_strength_is_an_exact_identity(channels: int) -> None:
    image = np.random.default_rng(137).random((17, 13, channels), dtype=np.float32)
    transform = A.Compose(
        [
            A.StochasticConvolution(
                kernel_range=(3, 7),
                strength_range=(0.0, 0.0),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]

    np.testing.assert_array_equal(result, image)


def test_shared_and_per_channel_kernels_have_distinct_channel_semantics() -> None:
    image = np.zeros((31, 29, 3), dtype=np.float32)
    image[15, 14] = 1.0

    shared = A.Compose(
        [
            A.StochasticConvolution(
                kernel_range=(5, 5),
                strength_range=(0.25, 0.25),
                per_channel=False,
                p=1.0,
            ),
        ],
        seed=137,
    )(image=image)["image"]
    per_channel = A.Compose(
        [
            A.StochasticConvolution(
                kernel_range=(5, 5),
                strength_range=(0.25, 0.25),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )(image=image)["image"]

    np.testing.assert_array_equal(shared[..., 0], shared[..., 1])
    assert not np.array_equal(per_channel[..., 0], per_channel[..., 1])


def test_batch_and_volume_reuse_one_kernel_realization() -> None:
    image = np.random.default_rng(137).random((13, 17, 3), dtype=np.float32)
    images = np.stack([image, image], axis=0)
    volume = np.stack([image, image], axis=0)
    transform = A.Compose(
        [
            A.StochasticConvolution(
                kernel_range=(3, 3),
                strength_range=(0.2, 0.2),
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(images=images, volume=volume)

    np.testing.assert_array_equal(result["images"][0], result["images"][1])
    np.testing.assert_array_equal(result["volume"][0], result["volume"][1])


def test_per_channel_groups_compatible_targets_and_replays() -> None:
    image = np.random.default_rng(137).random((11, 13, 3), dtype=np.float32)
    image2 = image.copy()
    volume_image = np.concatenate((image, np.zeros((11, 13, 2), dtype=np.float32)), axis=-1)
    volume = np.stack((volume_image, volume_image), axis=0)
    pipeline = A.ReplayCompose(
        [
            A.StochasticConvolution(
                kernel_range=(3, 3),
                strength_range=(0.1, 0.1),
                per_channel=True,
                p=1.0,
            ),
        ],
        additional_targets={"image2": "image"},
        is_check_shapes=False,
        seed=137,
    )

    original = pipeline(image=image, image2=image2, volume=volume)
    sampled_params = SampledParams.from_dict(original["replay"]["transforms"][0]["params"])
    replayed = A.ReplayCompose.replay(original["replay"], image=image, image2=image2, volume=volume)

    assert {params.targets for params in sampled_params.target_params} == {("image", "image2"), ("volume",)}
    assert sampled_params.params_for("image")["kernel"].shape == (3, 3, 3)
    assert sampled_params.params_for("volume")["kernel"].shape == (5, 3, 3)
    np.testing.assert_array_equal(original["image"], original["image2"])
    for target in ("image", "image2", "volume"):
        np.testing.assert_array_equal(replayed[target], original[target])


def test_per_channel_replay_rejects_changed_channel_count() -> None:
    image = np.zeros((11, 13, 3), dtype=np.float32)
    pipeline = A.ReplayCompose(
        [A.StochasticConvolution(kernel_range=(3, 3), strength_range=(0.1, 0.1), per_channel=True, p=1.0)],
        additional_targets={"image2": "image"},
        is_check_shapes=False,
        seed=137,
    )

    recorded = pipeline(image=image, image2=image.copy())["replay"]

    with pytest.raises(SampledParamsError, match="requirements do not match target 'image2'"):
        A.ReplayCompose.replay(recorded, image=image, image2=np.zeros((11, 13, 5), dtype=np.float32))


def test_stochastic_convolution_is_seed_reproducible() -> None:
    image = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32)
    transform_kwargs = {
        "kernel_range": (3, 7),
        "strength_range": (0.1, 0.3),
        "per_channel": True,
        "p": 1.0,
    }

    first = A.Compose([A.StochasticConvolution(**transform_kwargs)], seed=137)(image=image)["image"]
    second = A.Compose([A.StochasticConvolution(**transform_kwargs)], seed=137)(image=image)["image"]

    np.testing.assert_array_equal(first, second)


def test_stochastic_convolution_replay_reuses_realized_kernel() -> None:
    image = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32)
    pipeline = A.ReplayCompose(
        [
            A.StochasticConvolution(
                kernel_range=(3, 3),
                strength_range=(0.2, 0.2),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )

    original = pipeline(image=image)
    replayed = A.ReplayCompose.replay(original["replay"], image=image)

    np.testing.assert_array_equal(replayed["image"], original["image"])


def test_stochastic_convolution_records_constructor_valid_realized_ranges() -> None:
    image = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32)
    transform = A.Compose(
        [
            A.StochasticConvolution(
                kernel_range=(3, 7),
                strength_range=(0.1, 0.3),
                per_channel=True,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0,
            ),
        ],
        save_applied_params=True,
        seed=137,
    )

    result = transform(image=image)
    applied = result["applied_transforms"][0][1]

    assert applied["kernel_range"] in {3, 5, 7}
    assert 0.1 <= applied["strength_range"] <= 0.3
    assert applied["border_mode"] == cv2.BORDER_REFLECT
    assert applied["per_channel"] is True


def test_convolve_accepts_per_channel_kernels_and_explicit_border_mode() -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5, 1) / 25
    random_field = np.zeros((1, 3, 3), dtype=np.float32)
    random_field[0, 1, 0] = 1.0
    kernel = fpixel.create_stochastic_convolution_kernel(random_field, strength=1.0)

    result = fpixel.convolve(image, kernel, border_mode=cv2.BORDER_CONSTANT)
    expected = cv2.filter2D(image[..., 0], -1, kernel[0], borderType=cv2.BORDER_CONSTANT)[..., None]

    np.testing.assert_array_equal(result, np.clip(expected, 0, 1))


def test_kernel_strength_is_normalized_by_side_length_without_dc_normalization() -> None:
    field_3 = np.ones((3, 3), dtype=np.float32)
    field_5 = np.ones((5, 5), dtype=np.float32)

    kernel_3 = fpixel.create_stochastic_convolution_kernel(field_3, strength=1.5)
    kernel_5 = fpixel.create_stochastic_convolution_kernel(field_5, strength=1.5)

    np.testing.assert_allclose(kernel_3[0, 0], 0.5)
    np.testing.assert_allclose(kernel_5[0, 0], 0.3)
    np.testing.assert_allclose(kernel_3.sum(), 1.0 + 1.5 * 3.0)
    np.testing.assert_allclose(kernel_5.sum(), 1.0 + 1.5 * 5.0)


def test_constant_input_follows_realized_kernel_dc_gain() -> None:
    field = np.zeros((3, 3), dtype=np.float32)
    field[0, 0] = 1.0
    field[1, 2] = -2.0
    kernel = fpixel.create_stochastic_convolution_kernel(field, strength=1.0)
    image = np.full((11, 13, 1), 0.25, dtype=np.float32)

    result = fpixel.convolve(image, kernel, border_mode=cv2.BORDER_REFLECT)

    expected_value = np.float32(0.25 * kernel.sum())
    np.testing.assert_allclose(result, expected_value)


def test_extreme_strength_stays_finite_and_clipped() -> None:
    image = np.full((17, 19, 5), 0.5, dtype=np.float32)
    transform = A.Compose(
        [
            A.StochasticConvolution(
                kernel_range=(7, 7),
                strength_range=(100.0, 100.0),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]

    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1


@pytest.mark.parametrize("invalid_range", [(2, 5), (3, 6), (4, 6)])
def test_stochastic_convolution_rejects_even_kernel_sizes(invalid_range: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        A.StochasticConvolution(kernel_range=invalid_range)


@pytest.mark.parametrize(
    "border_mode",
    [cv2.BORDER_WRAP, cv2.BORDER_TRANSPARENT, cv2.BORDER_ISOLATED],
)
def test_stochastic_convolution_rejects_unsupported_border_modes(border_mode: int) -> None:
    with pytest.raises(ValueError, match="border_mode"):
        A.StochasticConvolution(border_mode=border_mode)


@pytest.mark.parametrize("strength", [-0.1, float("inf"), float("nan")])
def test_stochastic_convolution_kernel_rejects_invalid_strength(strength: float) -> None:
    random_field = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="strength"):
        fpixel.create_stochastic_convolution_kernel(random_field, strength)
