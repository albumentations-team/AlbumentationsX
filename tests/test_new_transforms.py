"""Tests for newly added transforms: Vignetting, ChannelSwap, FilmGrain,
Halftone, GridMask, LensFlare, WaterRefraction, AtmosphericFog.
"""

import numpy as np
import pytest

import albumentations as A
import albumentations.augmentations.pixel.functional as fpixel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(shape=(100, 100, 3), dtype=np.uint8, seed=137):
    rng = np.random.default_rng(seed)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    return rng.uniform(0, 1, shape).astype(np.float32)


def _make_gradient_image(height=100, width=100, channels=3, dtype=np.uint8):
    """Image with clear luminance gradient: top=dark, bottom=bright."""
    grad = np.linspace(0, 1, height, dtype=np.float32)[:, np.newaxis, np.newaxis]
    grad = np.broadcast_to(grad, (height, width, channels)).copy()
    if dtype == np.uint8:
        return (grad * 255).astype(np.uint8)
    return grad.astype(np.float32)


# ===========================================================================
# Vignetting
# ===========================================================================


class TestVignetting:
    def test_corners_darker_than_center(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = fpixel.apply_vignette(img, intensity=0.8, center_x=0.5, center_y=0.5)

        center_val = float(result[50, 50].mean())
        corner_vals = [
            float(result[0, 0].mean()),
            float(result[0, 99].mean()),
            float(result[99, 0].mean()),
            float(result[99, 99].mean()),
        ]
        for cv in corner_vals:
            assert cv < center_val, f"Corner {cv} should be darker than center {center_val}"

    def test_zero_intensity_is_identity(self):
        img = _make_image(seed=42)
        result = fpixel.apply_vignette(img, intensity=0.0, center_x=0.5, center_y=0.5)
        np.testing.assert_array_equal(result, img)

    def test_max_intensity_makes_corners_black(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = fpixel.apply_vignette(img, intensity=1.0, center_x=0.5, center_y=0.5)
        assert result[0, 0].max() < 10, "Corners should be nearly black with intensity=1.0"

    def test_center_offset_shifts_bright_region(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result_left = fpixel.apply_vignette(img, intensity=0.8, center_x=0.2, center_y=0.5)
        result_right = fpixel.apply_vignette(img, intensity=0.8, center_x=0.8, center_y=0.5)

        left_brightness = float(result_left[:, :25].mean())
        right_brightness = float(result_right[:, :25].mean())
        assert left_brightness > right_brightness, "Left-centered vignette should be brighter on left side"

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype)
        aug = A.Vignetting(intensity_range=(0.3, 0.5), p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_single_channel(self):
        img = _make_image(shape=(100, 100, 1), seed=10)
        result = fpixel.apply_vignette(img, intensity=0.5, center_x=0.5, center_y=0.5)
        assert result.shape == (100, 100, 1)
        assert float(result[50, 50].mean()) > float(result[0, 0].mean())

    def test_serialization_roundtrip(self):
        aug = A.Vignetting(intensity_range=(0.2, 0.6), center_range=(0.4, 0.6), p=0.8)
        serialized = aug.to_dict()
        restored = A.from_dict(serialized)
        assert restored.intensity_range == aug.intensity_range
        assert restored.center_range == aug.center_range
        assert restored.p == aug.p


# ===========================================================================
# ChannelSwap
# ===========================================================================


class TestChannelSwap:
    def test_rgb_to_bgr_swap(self):
        img = _make_image(seed=1)
        aug = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        result = aug(image=img)["image"]
        np.testing.assert_array_equal(result[:, :, 0], img[:, :, 2])
        np.testing.assert_array_equal(result[:, :, 1], img[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], img[:, :, 0])

    def test_double_application_is_identity(self):
        img = _make_image(seed=2)
        aug = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        once = aug(image=img)["image"]
        twice = aug(image=once)["image"]
        np.testing.assert_array_equal(twice, img)

    def test_cyclic_permutation(self):
        img = _make_image(seed=3)
        aug = A.ChannelSwap(channel_order=(1, 2, 0), p=1.0)
        result = aug(image=img)["image"]
        np.testing.assert_array_equal(result[:, :, 0], img[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], img[:, :, 2])
        np.testing.assert_array_equal(result[:, :, 2], img[:, :, 0])

    def test_channel_count_mismatch_returns_unchanged(self):
        img = _make_image(shape=(50, 50, 1), seed=4)
        aug = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        with pytest.warns(UserWarning, match="channel_order has 3 elements but image has 1"):
            result = aug(image=img)["image"]
        np.testing.assert_array_equal(result, img)

    def test_invalid_permutation_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            A.ChannelSwap(channel_order=(0, 0, 1))

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            A.ChannelSwap(channel_order=(0,))

    def test_four_channels(self):
        img = _make_image(shape=(50, 50, 4), seed=5)
        aug = A.ChannelSwap(channel_order=(3, 2, 1, 0), p=1.0)
        result = aug(image=img)["image"]
        np.testing.assert_array_equal(result[:, :, 0], img[:, :, 3])
        np.testing.assert_array_equal(result[:, :, 3], img[:, :, 0])

    def test_deterministic_across_calls(self):
        img = _make_image(seed=6)
        aug = A.ChannelSwap(channel_order=(2, 0, 1), p=1.0)
        r1 = aug(image=img)["image"]
        r2 = aug(image=img)["image"]
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype, seed=7)
        aug = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_batch_images_same_permutation(self):
        images = np.stack([_make_image(seed=i) for i in range(3)])
        aug = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        result = aug(images=images)["images"]
        assert result.shape == images.shape
        for i in range(3):
            np.testing.assert_array_equal(result[i, :, :, 0], images[i, :, :, 2])
            np.testing.assert_array_equal(result[i, :, :, 2], images[i, :, :, 0])

    def test_volume_channel_permutation(self):
        volume = _make_image(shape=(4, 50, 50, 3), seed=8)
        aug = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        result = aug(volume=volume)["volume"]
        assert result.shape == volume.shape
        np.testing.assert_array_equal(result[:, :, :, 0], volume[:, :, :, 2])
        np.testing.assert_array_equal(result[:, :, :, 2], volume[:, :, :, 0])

    def test_volume_channel_mismatch_unchanged(self):
        volume = _make_image(shape=(4, 50, 50, 1), seed=9)
        aug = A.ChannelSwap(channel_order=(2, 1, 0), p=1.0)
        with pytest.warns(UserWarning, match="channel_order has 3 elements but .* have 1"):
            result = aug(volume=volume)["volume"]
        np.testing.assert_array_equal(result, volume)


# ===========================================================================
# FilmGrain
# ===========================================================================


class TestFilmGrain:
    def test_dark_areas_get_more_grain(self):
        img = _make_gradient_image(height=200, width=200)
        rng = np.random.default_rng(137)
        grain = rng.standard_normal((200, 200), dtype=np.float32)
        result = fpixel.apply_film_grain(img, grain, intensity=0.5)

        dark_region = img[:50, :, :]
        bright_region = img[150:, :, :]
        dark_result = result[:50, :, :]
        bright_result = result[150:, :, :]

        dark_diff = np.abs(dark_result.astype(float) - dark_region.astype(float)).mean()
        bright_diff = np.abs(bright_result.astype(float) - bright_region.astype(float)).mean()

        assert dark_diff > bright_diff, (
            f"Dark region diff ({dark_diff:.1f}) should be larger than bright ({bright_diff:.1f})"
        )

    def test_zero_intensity_preserves_image(self):
        img = _make_image(seed=10)
        rng = np.random.default_rng(10)
        grain = rng.standard_normal(img.shape[:2], dtype=np.float32)
        result = fpixel.apply_film_grain(img, grain, intensity=0.0)
        np.testing.assert_array_equal(result, img)

    def test_coarse_grain_is_spatially_correlated(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        aug_fine = A.Compose([A.FilmGrain(intensity_range=(0.5, 0.5), grain_size_range=(1, 1), p=1.0)], seed=137)
        aug_coarse = A.Compose([A.FilmGrain(intensity_range=(0.5, 0.5), grain_size_range=(4, 4), p=1.0)], seed=137)

        result_fine = aug_fine(image=img)["image"]
        result_coarse = aug_coarse(image=img)["image"]

        def spatial_autocorrelation(arr):
            diff = np.abs(arr[:, 1:].astype(float) - arr[:, :-1].astype(float))
            return diff.mean()

        fine_corr = spatial_autocorrelation(result_fine)
        coarse_corr = spatial_autocorrelation(result_coarse)

        assert coarse_corr < fine_corr, (
            f"Coarse grain ({coarse_corr:.1f}) should have lower pixel-to-pixel variance "
            f"than fine grain ({fine_corr:.1f})"
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype, seed=11)
        aug = A.FilmGrain(intensity_range=(0.2, 0.3), p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_single_channel(self):
        img = _make_image(shape=(80, 80, 1), seed=12)
        aug = A.FilmGrain(intensity_range=(0.2, 0.3), p=1.0)
        result = aug(image=img)["image"]
        assert result.shape == (80, 80, 1)
        assert not np.array_equal(result, img)

    def test_output_clipped_to_valid_range(self):
        img = _make_image(seed=13)
        aug = A.FilmGrain(intensity_range=(0.9, 1.0), grain_size_range=(1, 1), p=1.0)
        result = aug(image=img)["image"]
        assert result.min() >= 0
        assert result.max() <= 255


# ===========================================================================
# Halftone
# ===========================================================================


class TestHalftone:
    def test_blend_one_returns_original(self):
        img = _make_image(seed=20)
        result = fpixel.apply_halftone(img, dot_size=6, blend=1.0)
        np.testing.assert_array_equal(result, img)

    def test_output_differs_from_input(self):
        img = _make_image(seed=21)
        result = fpixel.apply_halftone(img, dot_size=8, blend=0.0)
        assert not np.array_equal(result, img)

    def test_larger_dots_produce_coarser_pattern(self):
        img = _make_image(shape=(200, 200, 3), seed=22)
        small = fpixel.apply_halftone(img, dot_size=4, blend=0.0)
        large = fpixel.apply_halftone(img, dot_size=20, blend=0.0)

        def count_unique_regions(arr):
            return len(np.unique(arr[::2, ::2, 0]))

        assert count_unique_regions(large) < count_unique_regions(small), (
            "Larger dots should produce fewer unique values"
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype, seed=23)
        aug = A.Halftone(dot_size_range=(4, 8), blend_range=(0.0, 0.3), p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_shape_preserved(self):
        for shape in [(100, 100, 3), (50, 80, 1), (120, 90, 3)]:
            img = _make_image(shape=shape, seed=24)
            aug = A.Halftone(p=1.0)
            result = aug(image=img)["image"]
            assert result.shape == shape

    def test_pure_black_stays_mostly_black(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = fpixel.apply_halftone(img, dot_size=8, blend=0.0)
        assert result.mean() < 5, "Pure black image should produce near-black halftone"

    def test_pure_white_has_large_dots(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = fpixel.apply_halftone(img, dot_size=8, blend=0.0)
        assert result.mean() > 100, "Pure white image should produce large dots"


# ===========================================================================
# GridMask
# ===========================================================================


class TestGridMask:
    def test_some_pixels_are_zeroed(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        aug = A.GridMask(num_grid_range=(4, 4), line_width_range=(0.3, 0.3), fill=0, p=1.0)
        result = aug(image=img)["image"]
        assert result.min() == 0, "Some pixels should be filled with 0"
        assert result.max() == 200, "Non-masked pixels should be unchanged"

    def test_mask_not_all_zeroed(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        aug = A.GridMask(num_grid_range=(3, 3), line_width_range=(0.2, 0.2), fill=0, p=1.0)
        result = aug(image=img)["image"]
        assert result.max() > 0, "Not all pixels should be masked"

    def test_custom_fill_value(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        aug = A.GridMask(num_grid_range=(4, 4), line_width_range=(0.3, 0.3), fill=128, p=1.0)
        result = aug(image=img)["image"]
        unique = set(np.unique(result))
        assert 128 in unique, "Custom fill value should appear in masked regions"
        assert 200 in unique, "Original pixel value should remain in non-masked regions"

    def test_mask_target_modified(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)
        aug = A.GridMask(
            num_grid_range=(4, 4),
            line_width_range=(0.3, 0.3),
            fill=0,
            fill_mask=0,
            p=1.0,
        )
        result = aug(image=img, mask=mask)
        assert result["mask"].min() == 0

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype, seed=30)
        aug = A.GridMask(num_grid_range=(3, 5), p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_more_grids_more_dropout(self):
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        aug_few = A.Compose(
            [A.GridMask(num_grid_range=(2, 2), line_width_range=(0.3, 0.3), fill=0, p=1.0)],
            seed=137,
        )
        aug_many = A.Compose(
            [A.GridMask(num_grid_range=(8, 8), line_width_range=(0.3, 0.3), fill=0, p=1.0)],
            seed=137,
        )
        few_zeros = (aug_few(image=img)["image"] == 0).sum()
        many_zeros = (aug_many(image=img)["image"] == 0).sum()
        assert many_zeros > few_zeros, "More grid divisions should produce more dropout"

    def test_rotated_gridmask_affects_pixels_and_is_not_axis_aligned(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        aug = A.Compose(
            [
                A.GridMask(
                    num_grid_range=(4, 4),
                    line_width_range=(0.3, 0.3),
                    rotation_range=(np.pi / 4, np.pi / 4),
                    fill=0,
                    p=1.0,
                ),
            ],
            seed=137,
        )
        result = aug(image=img)["image"]
        dropped = (result == 0).all(axis=-1)
        num_dropped = int(dropped.sum())
        total = img.shape[0] * img.shape[1]

        assert 0 < num_dropped < total, "Some pixels dropped, not all"

        row_dropout = dropped.sum(axis=1)
        col_dropout = dropped.sum(axis=0)
        assert row_dropout.std() > 0 or col_dropout.std() > 0, "Rotated grid should have variability"


# ===========================================================================
# LensFlare
# ===========================================================================


class TestLensFlare:
    def test_output_brighter_than_input(self):
        img = _make_image(shape=(100, 100, 3), seed=40)
        aug = A.LensFlare(intensity_range=(0.5, 0.8), p=1.0)
        result = aug(image=img)["image"]
        assert result.astype(float).mean() >= img.astype(float).mean() - 1, (
            "Lens flare is additive, output should be at least as bright"
        )

    def test_flare_roi_respected(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        aug = A.Compose(
            [
                A.LensFlare(
                    flare_roi=(0.0, 0.0, 0.1, 0.1),
                    intensity_range=(0.8, 1.0),
                    bloom_range=(0.01, 0.02),
                    p=1.0,
                ),
            ],
            seed=137,
        )
        result = aug(image=img)["image"]
        top_left_brightness = float(result[:20, :20].mean())
        bottom_right_brightness = float(result[180:, 180:].mean())
        assert top_left_brightness > bottom_right_brightness, (
            "Flare source in top-left ROI should make that region brightest"
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype, seed=41)
        aug = A.LensFlare(p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_shape_preserved(self):
        img = _make_image(shape=(80, 120, 3), seed=42)
        aug = A.LensFlare(p=1.0)
        result = aug(image=img)["image"]
        assert result.shape == (80, 120, 3)

    def test_output_clipped_uint8(self):
        img = np.full((100, 100, 3), 250, dtype=np.uint8)
        aug = A.LensFlare(intensity_range=(0.8, 1.0), p=1.0)
        result = aug(image=img)["image"]
        assert result.max() <= 255
        assert result.min() >= 0

    def test_zero_ghosts(self):
        img = _make_image(seed=43)
        aug = A.LensFlare(num_ghosts_range=(0, 0), intensity_range=(0.3, 0.5), p=1.0)
        result = aug(image=img)["image"]
        assert result.shape == img.shape

    def test_non_rgb_raises(self):
        img = _make_image(shape=(50, 50, 1), seed=44)
        aug = A.LensFlare(p=1.0)
        with pytest.raises(ValueError, match="3-channel"):
            aug(image=img)


# ===========================================================================
# WaterRefraction
# ===========================================================================


class TestWaterRefraction:
    def test_deforms_image(self):
        img = _make_image(seed=50)
        aug = A.WaterRefraction(amplitude_range=(0.03, 0.05), p=1.0)
        result = aug(image=img)["image"]
        assert not np.array_equal(result, img)

    def test_deforms_mask(self):
        img = _make_image(seed=51)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1

        aug = A.WaterRefraction(amplitude_range=(0.03, 0.05), p=1.0)
        result = aug(image=img, mask=mask)
        assert not np.array_equal(result["mask"], mask), "Mask should be deformed"

    def test_tiny_amplitude_nearly_identity(self):
        img = _make_image(seed=52)
        aug = A.Compose(
            [A.WaterRefraction(amplitude_range=(0.0001, 0.0001), wavelength_range=(0.5, 0.5), p=1.0)],
            seed=137,
        )
        result = aug(image=img)["image"]
        diff = np.abs(result.astype(float) - img.astype(float)).mean()
        assert diff < 2, f"Tiny amplitude should barely change image, got diff={diff:.2f}"

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype, seed=53)
        aug = A.WaterRefraction(p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_shape_preserved(self):
        for shape in [(100, 100, 3), (80, 120, 3), (50, 50, 1)]:
            img = _make_image(shape=shape, seed=54)
            aug = A.WaterRefraction(p=1.0)
            result = aug(image=img)["image"]
            assert result.shape == shape

    def test_more_waves_more_complex(self):
        img = _make_image(shape=(100, 100, 3), seed=137)
        aug_few = A.Compose(
            [A.WaterRefraction(amplitude_range=(0.03, 0.03), num_waves_range=(1, 1), p=1.0)],
            seed=137,
        )
        aug_many = A.Compose(
            [A.WaterRefraction(amplitude_range=(0.03, 0.03), num_waves_range=(10, 10), p=1.0)],
            seed=137,
        )
        few = aug_few(image=img)["image"]
        many = aug_many(image=img)["image"]
        assert not np.array_equal(few, many)

    def test_bboxes_supported(self):
        img = _make_image(seed=55)
        bboxes = np.array([[20, 20, 80, 80]], dtype=np.float32)
        aug = A.Compose(
            [A.WaterRefraction(amplitude_range=(0.01, 0.02), p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
            seed=137,
            strict=False,
        )
        result = aug(image=img, bboxes=bboxes, labels=[1])
        assert "bboxes" in result
        bboxes_out = result["bboxes"]
        assert isinstance(bboxes_out, (list, tuple, np.ndarray))
        assert len(bboxes_out) == len(bboxes)


# ===========================================================================
# AtmosphericFog
# ===========================================================================


class TestAtmosphericFog:
    def test_zero_density_is_identity(self):
        img = _make_image(seed=60)
        depth = np.linspace(1.0, 0.0, 100, dtype=np.float32)[:, np.newaxis]
        depth = np.broadcast_to(depth, (100, 100)).copy()
        result = fpixel.apply_atmospheric_fog(img, density=0.0, fog_color=(200.0, 200.0, 200.0), depth_map=depth)
        np.testing.assert_array_equal(result, img)

    def test_linear_mode_top_foggier(self):
        img = np.full((200, 200, 3), 50, dtype=np.uint8)
        aug = A.Compose(
            [A.AtmosphericFog(density_range=(3.0, 3.0), fog_color=(255, 255, 255), depth_mode="linear", p=1.0)],
            seed=137,
        )
        result = aug(image=img)["image"]
        top_brightness = float(result[:30, :].mean())
        bottom_brightness = float(result[170:, :].mean())
        assert top_brightness > bottom_brightness, (
            f"Top ({top_brightness:.0f}) should be foggier (brighter with white fog) "
            f"than bottom ({bottom_brightness:.0f})"
        )

    def test_radial_mode_edges_foggier(self):
        img = np.full((200, 200, 3), 50, dtype=np.uint8)
        aug = A.Compose(
            [A.AtmosphericFog(density_range=(3.0, 3.0), fog_color=(255, 255, 255), depth_mode="radial", p=1.0)],
            seed=137,
        )
        result = aug(image=img)["image"]
        center_brightness = float(result[90:110, 90:110].mean())
        edge_brightness = float(result[:10, :10].mean())
        assert edge_brightness > center_brightness, "Edges should be foggier in radial mode"

    def test_high_density_washes_out_to_fog_color(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        fog_color = (200, 200, 200)
        aug = A.Compose(
            [A.AtmosphericFog(density_range=(50.0, 50.0), fog_color=fog_color, depth_mode="linear", p=1.0)],
            seed=137,
        )
        result = aug(image=img)["image"]
        top_region = result[:10, :]
        assert np.abs(top_region.astype(float).mean() - 200) < 10, (
            "Very high density should make distant regions converge to fog color"
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_dtype_preserved(self, dtype):
        img = _make_image(dtype=dtype, seed=61)
        aug = A.AtmosphericFog(p=1.0)
        result = aug(image=img)["image"]
        assert result.dtype == dtype

    def test_single_channel(self):
        img = _make_image(shape=(100, 100, 1), seed=62)
        aug = A.AtmosphericFog(density_range=(1.0, 2.0), fog_color=(200,), p=1.0)
        result = aug(image=img)["image"]
        assert result.shape == (100, 100, 1)
        assert not np.array_equal(result, img)

    @pytest.mark.parametrize("depth_mode", ["linear", "diagonal", "radial"])
    def test_all_depth_modes_produce_different_results(self, depth_mode):
        img = _make_image(seed=63)
        aug = A.AtmosphericFog(density_range=(2.0, 2.0), depth_mode=depth_mode, p=1.0)
        result = aug(image=img)["image"]
        assert not np.array_equal(result, img)

    def test_depth_modes_are_structurally_distinct(self):
        img = _make_image(seed=63)
        modes = {}
        for depth_mode in ["linear", "diagonal", "radial"]:
            aug = A.AtmosphericFog(density_range=(2.0, 2.0), depth_mode=depth_mode, p=1.0)
            modes[depth_mode] = aug(image=img)["image"]

        for depth_mode, result in modes.items():
            assert not np.array_equal(result, img), f"{depth_mode} depth mode did not change the image"

        assert not np.array_equal(
            modes["linear"],
            modes["radial"],
        ), "linear and radial depth modes produced identical outputs"

        h, w = img.shape[:2]
        linear = modes["linear"]
        top = linear[: h // 4].mean()
        bottom = linear[3 * h // 4 :].mean()
        assert top != pytest.approx(bottom), "linear depth should show top/bottom brightness difference"

        radial = modes["radial"]
        center = radial[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
        corners = np.concatenate(
            [
                radial[: h // 4, : w // 4].reshape(-1, 3),
                radial[: h // 4, 3 * w // 4 :].reshape(-1, 3),
                radial[3 * h // 4 :, : w // 4].reshape(-1, 3),
                radial[3 * h // 4 :, 3 * w // 4 :].reshape(-1, 3),
            ],
            axis=0,
        ).mean()
        assert center != pytest.approx(corners), "radial depth should show center/corners difference"

    def test_output_clipped(self):
        img = _make_image(seed=64)
        aug = A.AtmosphericFog(density_range=(5.0, 5.0), fog_color=(255, 255, 255), p=1.0)
        result = aug(image=img)["image"]
        assert result.min() >= 0
        assert result.max() <= 255

    def test_fog_color_affects_result(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        aug_white = A.Compose(
            [A.AtmosphericFog(density_range=(3.0, 3.0), fog_color=(255, 255, 255), p=1.0)],
            seed=137,
        )
        aug_dark = A.Compose(
            [A.AtmosphericFog(density_range=(3.0, 3.0), fog_color=(50, 50, 50), p=1.0)],
            seed=137,
        )
        white_result = aug_white(image=img)["image"]
        dark_result = aug_dark(image=img)["image"]
        assert float(white_result.mean()) > float(dark_result.mean()), (
            "White fog should produce brighter result than dark fog"
        )
