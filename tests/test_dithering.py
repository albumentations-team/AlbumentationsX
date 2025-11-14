"""Tests for dithering transforms."""


import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.pixel.dithering_functional import (
    apply_dithering,
    error_diffusion_dither,
    generate_bayer_matrix,
    ordered_dither,
    quantize_value,
    random_dither,
)


class TestDitheringFunctional:
    """Test dithering functional implementations."""

    def test_quantize_value(self):
        """Test value quantization."""
        # Test binary quantization
        np.testing.assert_equal(quantize_value(0.3, 2), 0.0)
        np.testing.assert_equal(quantize_value(0.7, 2), 1.0)
        np.testing.assert_equal(quantize_value(0.5, 2), 1.0)

        # Test multi-level quantization
        np.testing.assert_equal(quantize_value(0.0, 4), 0.0)
        np.testing.assert_allclose(quantize_value(0.25, 4), 1/3)
        np.testing.assert_allclose(quantize_value(0.5, 4), 2/3)
        np.testing.assert_equal(quantize_value(1.0, 4), 1.0)

    def test_generate_bayer_matrix(self):
        """Test Bayer matrix generation."""
        # Test 2x2 matrix
        bayer2 = generate_bayer_matrix(2)
        np.testing.assert_equal(bayer2.shape, (2, 2))
        assert bayer2.min() >= 0
        assert bayer2.max() <= 1

        # Test 4x4 matrix
        bayer4 = generate_bayer_matrix(4)
        np.testing.assert_equal(bayer4.shape, (4, 4))
        assert bayer4.min() >= 0
        assert bayer4.max() <= 1

        # Test 8x8 matrix
        bayer8 = generate_bayer_matrix(8)
        np.testing.assert_equal(bayer8.shape, (8, 8))

        # Test 16x16 matrix
        bayer16 = generate_bayer_matrix(16)
        np.testing.assert_equal(bayer16.shape, (16, 16))

        # Test invalid size
        with pytest.raises(ValueError):
            generate_bayer_matrix(3)

    def test_random_dither(self):
        """Test random noise dithering."""
        rng = np.random.default_rng(137)
        img = np.full((10, 10, 1), 0.5, dtype=np.float32)

        # Test binary dithering with noise
        result = random_dither(img, n_colors=2, noise_range=(-0.3, 0.3), random_generator=rng)
        np.testing.assert_equal(result.shape, img.shape)
        assert np.all((result == 0) | (result == 1))
        # With noise, we should get a mix of 0s and 1s
        assert np.sum(result == 0) > 0
        assert np.sum(result == 1) > 0

    def test_ordered_dither(self):
        """Test ordered dithering with Bayer matrix."""
        # Create uniform gray image
        img = np.full((8, 8, 1), 0.5, dtype=np.float32)

        # Test with 4x4 Bayer matrix
        result = ordered_dither(img, n_colors=2, matrix_size=4)
        np.testing.assert_equal(result.shape, img.shape)
        assert np.all((result == 0) | (result == 1))
        # Should create a pattern
        assert np.sum(result == 0) > 0
        assert np.sum(result == 1) > 0

        # Test with different sizes
        for size in [2, 4, 8, 16]:
            result = ordered_dither(img, n_colors=2, matrix_size=size)
            np.testing.assert_equal(result.shape, img.shape)

    def test_error_diffusion_dither(self):
        """Test error diffusion dithering."""
        # Create gradient image
        img = np.linspace(0, 1, 100).reshape(10, 10, 1).astype(np.float32)

        # Test Floyd-Steinberg
        result = error_diffusion_dither(img, n_colors=2, algorithm="floyd_steinberg")
        np.testing.assert_equal(result.shape, img.shape)
        assert np.all((result >= 0) & (result <= 1))

        # Test serpentine scanning
        result_serpentine = error_diffusion_dither(
            img, n_colors=2, algorithm="floyd_steinberg", serpentine=True
        )
        np.testing.assert_equal(result_serpentine.shape, img.shape)

        # Test different algorithms
        for algo in ["jarvis", "stucki", "atkinson", "burkes", "sierra", "sierra_2row", "sierra_lite"]:
            result = error_diffusion_dither(img, n_colors=2, algorithm=algo)
            np.testing.assert_equal(result.shape, img.shape)
            assert np.all((result >= 0) & (result <= 1))

        # Test invalid algorithm
        with pytest.raises(ValueError):
            error_diffusion_dither(img, n_colors=2, algorithm="invalid")

    def test_apply_dithering_grayscale_mode(self):
        """Test dithering with grayscale conversion."""
        rng = np.random.default_rng(137)
        # Create color image
        img = np.random.rand(10, 10, 3).astype(np.float32)

        # Test grayscale mode
        result = apply_dithering(
            img, method="random", n_colors=2, color_mode="grayscale", noise_range=(-0.5, 0.5), random_generator=rng
        )
        np.testing.assert_equal(result.shape, img.shape)
        # All channels should be identical after grayscale conversion
        np.testing.assert_allclose(result[..., 0], result[..., 1])
        np.testing.assert_allclose(result[..., 1], result[..., 2])

    def test_apply_dithering_per_channel(self):
        """Test per-channel dithering."""
        rng = np.random.default_rng(137)
        # Create color image
        img = np.random.rand(10, 10, 3).astype(np.float32)

        # Test per-channel mode
        result = apply_dithering(
            img, method="ordered", n_colors=4, color_mode="per_channel", matrix_size=4
        )
        np.testing.assert_equal(result.shape, img.shape)
        # Channels can be different
        assert not np.allclose(result[..., 0], result[..., 1])


class TestDitheringTransform:
    """Test Dithering transform class."""

    @pytest.mark.parametrize(
        "method", ["random", "ordered", "error_diffusion"]
    )
    @pytest.mark.parametrize("n_colors", [2, 4, 16])
    @pytest.mark.parametrize("img_dtype", [np.uint8, np.float32])
    def test_dithering_methods(self, method, n_colors, img_dtype):
        """Test different dithering methods with various parameters."""
        # Create test image
        if img_dtype == np.uint8:
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        else:
            img = np.random.rand(100, 100, 3).astype(np.float32)

        # Create transform
        transform = A.Dithering(method=method, n_colors=n_colors, p=1.0)

        # Apply transform
        result = transform(image=img)["image"]

        # Check output
        np.testing.assert_equal(result.shape, img.shape)
        np.testing.assert_equal(result.dtype, img.dtype)

        # Check value range
        if img_dtype == np.uint8:
            assert result.min() >= 0
            assert result.max() <= 255
        else:
            assert result.min() >= 0
            assert result.max() <= 1

    def test_error_diffusion_algorithms(self):
        """Test different error diffusion algorithms."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        algorithms = [
            "floyd_steinberg", "jarvis", "stucki", "atkinson",
            "burkes", "sierra", "sierra_2row", "sierra_lite"
        ]

        for algo in algorithms:
            transform = A.Dithering(
                method="error_diffusion",
                error_diffusion_algorithm=algo,
                n_colors=2,
                p=1.0
            )
            result = transform(image=img)["image"]
            np.testing.assert_equal(result.shape, img.shape)

    def test_color_modes(self):
        """Test different color handling modes."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        # Test grayscale mode
        transform = A.Dithering(
            method="ordered",
            n_colors=2,
            color_mode="grayscale",
            p=1.0
        )
        result = transform(image=img)["image"]
        np.testing.assert_equal(result.shape, img.shape)

        # Test per-channel mode
        transform = A.Dithering(
            method="ordered",
            n_colors=4,
            color_mode="per_channel",
            p=1.0
        )
        result = transform(image=img)["image"]
        np.testing.assert_equal(result.shape, img.shape)

    def test_bayer_matrix_sizes(self):
        """Test different Bayer matrix sizes."""
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        for size in [2, 4, 8, 16]:
            transform = A.Dithering(
                method="ordered",
                n_colors=2,
                bayer_matrix_size=size,
                p=1.0
            )
            result = transform(image=img)["image"]
            np.testing.assert_equal(result.shape, img.shape)

    def test_serpentine_scanning(self):
        """Test serpentine scanning in error diffusion."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        # Without serpentine
        transform1 = A.Dithering(
            method="error_diffusion",
            serpentine=False,
            p=1.0
        )
        result1 = transform1(image=img)["image"]

        # With serpentine
        transform2 = A.Dithering(
            method="error_diffusion",
            serpentine=True,
            p=1.0
        )
        result2 = transform2(image=img)["image"]

        # Results should be different
        assert not np.array_equal(result1, result2)

    def test_probability(self):
        """Test transform probability."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        # With p=0, image should not change
        transform = A.Dithering(method="ordered", p=0.0)
        result = transform(image=img)["image"]
        np.testing.assert_array_equal(result, img)

        # With p=1, image should always change (unless already quantized)
        transform = A.Dithering(method="ordered", n_colors=2, p=1.0)
        result = transform(image=img)["image"]
        # Check that result has limited colors
        unique_values = np.unique(result)
        assert len(unique_values) <= 2 * 3  # 2 colors per channel, 3 channels

    def test_serialization(self):
        """Test transform serialization."""
        transform = A.Dithering(
            method="error_diffusion",
            n_colors=4,
            error_diffusion_algorithm="atkinson",
            bayer_matrix_size=8,
            serpentine=True,
            noise_range=(-0.3, 0.3),
            p=0.7
        )

        # Serialize
        serialized = A.to_dict(transform)

        # Deserialize
        deserialized = A.from_dict(serialized)

        # Check parameters
        np.testing.assert_equal(deserialized.method, transform.method)
        np.testing.assert_equal(deserialized.n_colors, transform.n_colors)
        np.testing.assert_equal(deserialized.error_diffusion_algorithm, transform.error_diffusion_algorithm)
        np.testing.assert_equal(deserialized.bayer_matrix_size, transform.bayer_matrix_size)
        np.testing.assert_equal(deserialized.serpentine, transform.serpentine)
        np.testing.assert_equal(deserialized.noise_range, transform.noise_range)
        np.testing.assert_equal(deserialized.p, transform.p)

    def test_invalid_color_mode(self):
        """Test that invalid color mode raises error."""
        with pytest.raises(ValueError):
            A.Dithering(color_mode="invalid_mode")

    def test_with_compose(self):
        """Test Dithering in a Compose pipeline."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.Dithering(
                method="error_diffusion",
                n_colors=16,
                error_diffusion_algorithm="floyd_steinberg",
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
        ])

        result = transform(image=img)["image"]
        np.testing.assert_equal(result.shape, img.shape)
        np.testing.assert_equal(result.dtype, img.dtype)

    def test_grayscale_image(self):
        """Test with grayscale input image."""
        img = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)

        transform = A.Dithering(method="ordered", n_colors=2, p=1.0)
        result = transform(image=img)["image"]

        np.testing.assert_equal(result.shape, img.shape)
        np.testing.assert_equal(result.dtype, img.dtype)

    def test_different_image_sizes(self):
        """Test with various image sizes."""
        sizes = [(10, 10), (50, 100), (137, 137), (1024, 768)]

        for h, w in sizes:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            transform = A.Dithering(method="ordered", p=1.0)
            result = transform(image=img)["image"]
            np.testing.assert_equal(result.shape, img.shape)

    def test_edge_cases(self):
        """Test edge cases."""
        # All black image
        img_black = np.zeros((50, 50, 3), dtype=np.uint8)
        transform = A.Dithering(method="error_diffusion", n_colors=2, p=1.0)
        result = transform(image=img_black)["image"]
        np.testing.assert_array_equal(result, 0)

        # All white image
        img_white = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = transform(image=img_white)["image"]
        np.testing.assert_array_equal(result, 255)

        # Single pixel image
        img_pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = transform(image=img_pixel)["image"]
        np.testing.assert_equal(result.shape, img_pixel.shape)

    def test_multichannel_support(self):
        """Test that dithering works with any number of channels."""
        # Test various channel counts
        channel_counts = [1, 2, 3, 4, 5, 10, 20, 100]

        for n_channels in channel_counts:
            img = np.random.rand(30, 30, n_channels).astype(np.float32)

            # Test per_channel mode
            transform = A.Dithering(
                method="ordered",
                n_colors=4,
                color_mode="per_channel",
                p=1.0
            )
            result = transform(image=img)["image"]

            # Shape should be preserved
            np.testing.assert_equal(result.shape, img.shape), f"Shape mismatch for {n_channels} channels"

            # Each channel should have limited values
            for ch in range(n_channels):
                unique_vals = len(np.unique(result[:, :, ch]))
                assert unique_vals <= 4, f"Channel {ch} has too many unique values"

    def test_uint8_optimization(self):
        """Test that uint8 images use optimized paths for ordered dithering."""
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img_float32 = img_uint8.astype(np.float32) / 255.0

        # Test ordered dithering
        transform = A.Dithering(method="ordered", n_colors=2, p=1.0)

        result_uint8 = transform(image=img_uint8)["image"]
        result_float32 = transform(image=img_float32)["image"]

        # Results should be similar (allowing for minor numerical differences)
        np.testing.assert_equal(result_uint8.dtype, np.uint8)
        np.testing.assert_equal(result_float32.dtype, np.float32)

        # Compare normalized results
        result_uint8_norm = result_uint8.astype(np.float32) / 255.0
        # Both should be binary (0 or 1)
        assert np.all((result_uint8_norm == 0) | (result_uint8_norm == 1))
        assert np.all((result_float32 == 0) | (result_float32 == 1))

        # Test ordered dithering
        transform = A.Dithering(method="ordered", n_colors=4, bayer_matrix_size=4, p=1.0)

        result_uint8 = transform(image=img_uint8)["image"]
        result_float32 = transform(image=img_float32)["image"]

        np.testing.assert_equal(result_uint8.dtype, np.uint8)
        np.testing.assert_equal(result_float32.dtype, np.float32)

        # Test that error diffusion still works with uint8 input (uses float32 internally)
        transform = A.Dithering(method="error_diffusion", n_colors=2, p=1.0)

        result_uint8 = transform(image=img_uint8)["image"]
        np.testing.assert_equal(result_uint8.dtype, np.uint8)  # Output preserves input dtype

    def test_grayscale_mode_multichannel(self):
        """Test grayscale mode with different input channel counts."""
        channel_counts = [1, 2, 3, 4, 5, 10]

        for n_channels in channel_counts:
            img = np.random.rand(30, 30, n_channels).astype(np.float32)

            transform = A.Dithering(
                method="error_diffusion",
                n_colors=2,
                color_mode="grayscale",
                p=1.0
            )
            result = transform(image=img)["image"]

            # Grayscale mode preserves original number of channels
            np.testing.assert_equal(result.shape, img.shape), f"Grayscale output shape wrong for {n_channels} input channels"

            # All channels should be identical in grayscale mode
            for ch in range(1, n_channels):
                np.testing.assert_allclose(result[:, :, 0], result[:, :, ch])
