import numpy as np

import albumentations as A


def test_audio_spectrogram_recipe_preserves_axes_shape_dtype_and_fill() -> None:
    height, width = 20, 40
    spectrogram = np.ones((height, width, 1), dtype=np.float32)
    transform = A.Compose(
        [
            A.XYMasking(
                num_masks_x_range=(1, 1),
                num_masks_y_range=(1, 1),
                mask_x_length_range=(0.25, 0.25),
                mask_y_length_range=(0.20, 0.20),
                fill=-1.0,
                p=1.0,
            ),
        ],
        seed=137,
        strict=True,
    )

    result = transform(image=spectrogram)["image"]

    assert result.shape == (height, width, 1)
    assert result.dtype == np.float32
    assert np.count_nonzero(np.all(result == -1.0, axis=(0, 2))) == 10
    assert np.count_nonzero(np.all(result == -1.0, axis=(1, 2))) == 4


def test_eeg_recipe_uses_one_mask_across_all_channels() -> None:
    eeg = np.ones((16, 32, 19), dtype=np.float32)
    transform = A.Compose(
        [
            A.XYMasking(
                num_masks_x_range=(1, 1),
                mask_x_length_range=(0.25, 0.25),
                fill=0.0,
                p=1.0,
            ),
        ],
        seed=137,
        strict=True,
    )

    result = transform(image=eeg)["image"]

    assert result.shape == eeg.shape
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.broadcast_to(result[..., :1], result.shape))


def test_uint8_recipe_preserves_range_and_fill() -> None:
    spectrogram = np.full((12, 24, 1), 137, dtype=np.uint8)
    transform = A.Compose(
        [
            A.XYMasking(
                num_masks_x_range=(1, 1),
                mask_x_length_range=(6, 6),
                fill=0,
                p=1.0,
            ),
        ],
        seed=137,
        strict=True,
    )

    result = transform(image=spectrogram)["image"]

    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 137


def test_time_frequency_recipe_is_reproducible_across_seeded_pipelines() -> None:
    spectrogram = np.ones((20, 40, 1), dtype=np.float32)

    def make_pipeline() -> A.Compose:
        return A.Compose(
            [
                A.XYMasking(
                    num_masks_x_range=(1, 3),
                    num_masks_y_range=(1, 2),
                    mask_x_length_range=(0.10, 0.30),
                    mask_y_length_range=(0.10, 0.25),
                    fill=0.0,
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )

    first = make_pipeline()(image=spectrogram)["image"]
    second = make_pipeline()(image=spectrogram)["image"]

    np.testing.assert_array_equal(first, second)


def test_time_frequency_recipe_serialization_and_replay() -> None:
    spectrogram = np.arange(20 * 40, dtype=np.float32).reshape(20, 40, 1)
    pipeline = A.ReplayCompose(
        [
            A.XYMasking(
                num_masks_x_range=(1, 2),
                num_masks_y_range=(1, 2),
                mask_x_length_range=(0.10, 0.20),
                mask_y_length_range=(0.10, 0.20),
                fill=0.0,
                p=1.0,
            ),
        ],
        seed=137,
        strict=True,
    )

    restored = A.from_dict(A.to_dict(pipeline))
    original = pipeline(image=spectrogram)
    replayed = A.ReplayCompose.replay(original["replay"], image=spectrogram)
    restored_result = restored(image=spectrogram)["image"]

    np.testing.assert_array_equal(original["image"], replayed["image"])
    np.testing.assert_array_equal(restored_result, original["image"])
