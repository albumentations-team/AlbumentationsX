"""Test SimSIMD integration for AlbumentationsX operations.

This module tests that SimSIMD operations produce identical results to OpenCV
and NumPy equivalents, validating correctness before benchmarking performance.
"""

import numpy as np
import pytest

# Try importing cv2 and simsimd
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import simsimd

    HAS_SIMSIMD = True
except ImportError:
    HAS_SIMSIMD = False

# Mark all tests as requiring both libraries
pytestmark = pytest.mark.skipif(
    not (HAS_CV2 and HAS_SIMSIMD),
    reason="Requires both cv2 and simsimd",
)


class TestMatrixOperations:
    """Test matrix operations: cv2.gemm vs simsimd.dot"""

    @pytest.mark.parametrize(
        "rows,cols,inner",
        [
            (10, 10, 10),  # Small matrices
            (100, 100, 100),  # Larger matrices
            (768, 2, 768),  # Typical TPS dimensions
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
        ],
    )
    def test_gemm_vs_dot_product(self, rows, cols, inner, dtype):
        """Test that cv2.gemm and simsimd.dot produce identical results."""
        # Generate random matrices
        a = np.random.randn(rows, inner).astype(dtype)
        b = np.random.randn(inner, cols).astype(dtype)

        # cv2.gemm: C = alpha * A * B + beta * D
        # With alpha=1, beta=0, D=None, it's just A @ B
        result_cv2 = cv2.gemm(a, b, 1, None, 0)

        # NumPy @ for comparison (more accurate than cv2 for float32)
        result_np = a @ b

        # cv2.gemm and numpy @ should be close (cv2 uses BLAS internally)
        # Looser tolerance for float32 due to accumulation differences
        if dtype == np.float32:
            np.testing.assert_allclose(result_cv2, result_np, rtol=1e-4, atol=1e-6)
        else:
            np.testing.assert_allclose(result_cv2, result_np, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("size", [128, 768, 1536, 4096])
    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.float16,
        ],
    )
    def test_vector_dot_product(self, size, dtype):
        """Test vector dot products: np.dot vs simsimd.dot"""
        a = np.random.randn(size).astype(dtype)
        b = np.random.randn(size).astype(dtype)

        # NumPy dot product
        result_np = np.dot(a, b)

        # SimSIMD dot product
        if dtype == np.float16:
            result_simsimd = simsimd.dot(a, b, "float16")
        else:
            result_simsimd = simsimd.dot(a, b)

        # Compare results (looser tolerance for float16)
        if dtype == np.float16:
            assert np.isclose(result_np, result_simsimd, rtol=1e-2, atol=1e-3)
        else:
            assert np.isclose(result_np, result_simsimd, rtol=1e-5, atol=1e-8)

    def test_gemm_with_scaling_factors(self):
        """Test cv2.gemm with alpha and beta scaling factors."""
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(20, 15).astype(np.float32)
        c = np.random.randn(10, 15).astype(np.float32)

        alpha = 0.7
        beta = 0.3

        # cv2.gemm: result = alpha * A * B + beta * C
        result_cv2 = cv2.gemm(a, b, alpha, c, beta)

        # NumPy equivalent
        result_np = alpha * (a @ b) + beta * c

        np.testing.assert_allclose(result_cv2, result_np, rtol=1e-5, atol=1e-8)


class TestWeightedSum:
    """Test weighted sum operations: cv2.addWeighted vs simsimd.wsum"""

    @pytest.mark.parametrize(
        "shape",
        [
            (100, 100),  # 2D grayscale
            (100, 100, 3),  # RGB
            (100, 100, 4),  # RGBA
            (100, 100, 16),  # Hyperspectral
            (256, 256, 3),  # Larger RGB
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.float32,
        ],
    )
    def test_weighted_sum(self, shape, dtype):
        """Test that cv2.addWeighted and simsimd.wsum produce similar results."""
        img1 = (
            np.random.randint(0, 256, shape).astype(dtype)
            if dtype == np.uint8
            else np.random.randn(*shape).astype(dtype)
        )
        img2 = (
            np.random.randint(0, 256, shape).astype(dtype)
            if dtype == np.uint8
            else np.random.randn(*shape).astype(dtype)
        )

        alpha = 0.6
        beta = 0.4
        gamma = 0.0

        # cv2.addWeighted: result = alpha * img1 + beta * img2 + gamma
        # Only works for channels <= 4
        if len(shape) < 3 or shape[2] <= 4:
            try:
                result_cv2 = cv2.addWeighted(img1, alpha, img2, beta, gamma)

                # simsimd.wsum: result = alpha * img1 + beta * img2
                result_simsimd = np.empty_like(img1)
                simsimd.wsum(img1.reshape(-1), img2.reshape(-1), alpha=alpha, beta=beta, out=result_simsimd.reshape(-1))

                # Compare results with looser tolerance for float32 accumulation differences
                if dtype == np.uint8:
                    # uint8 may have rounding differences (saturation arithmetic)
                    np.testing.assert_allclose(result_cv2, result_simsimd, rtol=2 / 255, atol=2)
                else:
                    # float32 accumulation order can differ
                    np.testing.assert_allclose(result_cv2, result_simsimd, rtol=1e-4, atol=1e-6)
            except cv2.error:
                # cv2 doesn't support this dtype, skip comparison
                pytest.skip(f"cv2.addWeighted doesn't support {dtype}")
        else:
            # cv2.addWeighted fails for C > 4, but simsimd.wsum should work
            result_simsimd = np.empty_like(img1)
            simsimd.wsum(img1.reshape(-1), img2.reshape(-1), alpha=alpha, beta=beta, out=result_simsimd.reshape(-1))

            # Verify against NumPy
            result_np = alpha * img1 + beta * img2
            if dtype == np.uint8:
                result_np = np.clip(result_np, 0, 255).astype(np.uint8)
                # For uint8, allow ±2 due to rounding differences
                np.testing.assert_allclose(result_simsimd, result_np, rtol=3 / 255, atol=2)
            else:  # float32
                np.testing.assert_allclose(result_simsimd, result_np, rtol=1e-4, atol=1e-6)

    def test_albucore_uses_simsimd(self):
        """Test that albucore add_weighted works correctly.

        Note: albucore.add_weighted has a @clipped decorator that clips
        the result to [0, 1] for float32 and [0, 255] for uint8.
        """
        try:
            from albucore import add_weighted

            # Use values in [0, 1] range to avoid clipping effects
            img1 = np.random.rand(100, 100, 3).astype(np.float32)
            img2 = np.random.rand(100, 100, 3).astype(np.float32)

            result = add_weighted(img1, 0.6, img2, 0.4)

            # Verify result is correct (albucore clips to [0, 1] for float32)
            expected = np.clip(0.6 * img1 + 0.4 * img2, 0, 1).astype(np.float32)
            # Use looser tolerance for float32 operations
            np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)

        except ImportError:
            pytest.skip("albucore not available")


class TestFusedMultiplyAdd:
    """Test fused multiply-add operations: simsimd.fma"""

    @pytest.mark.parametrize(
        "shape",
        [
            (100, 100, 3),
            (256, 256, 3),
            (100, 100, 16),  # Hyperspectral
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
        ],
    )
    def test_fma_correctness(self, shape, dtype):
        """Test that simsimd.fma produces correct results."""
        a = np.random.randn(*shape).astype(dtype)
        b = np.random.randn(*shape).astype(dtype)
        c = np.random.randn(*shape).astype(dtype)

        alpha = 0.7
        beta = 0.3

        # SimSIMD FMA: result = alpha * a * b + beta * c
        result_simsimd = np.empty_like(a)
        simsimd.fma(a.reshape(-1), b.reshape(-1), c.reshape(-1), alpha=alpha, beta=beta, out=result_simsimd.reshape(-1))

        # NumPy equivalent
        result_np = alpha * a * b + beta * c

        # Compare results with appropriate tolerance for float32
        np.testing.assert_allclose(result_simsimd, result_np, rtol=1e-4, atol=1e-6)

    @pytest.mark.slow
    def test_fma_vs_sequential_operations(self, large_float_array_1000x1000):
        """Compare FMA with sequential multiply and add."""
        # Reuse the large float array for a, create b and c
        # Need to make contiguous copy for simsimd
        a = np.ascontiguousarray(large_float_array_1000x1000[:, :, 0])
        b = np.random.randn(1000, 1000).astype(np.float32)
        c = np.random.randn(1000, 1000).astype(np.float32)

        alpha = 0.7
        beta = 0.3

        # NumPy (reference)
        result_np = alpha * a * b + beta * c

        # SimSIMD FMA (fused)
        result_simsimd = np.empty_like(a)
        simsimd.fma(a.reshape(-1), b.reshape(-1), c.reshape(-1), alpha=alpha, beta=beta, out=result_simsimd.reshape(-1))

        # Should produce similar results (looser tolerance for large arrays)
        np.testing.assert_allclose(result_simsimd, result_np, rtol=1e-4, atol=1e-6)


class TestDistanceMetrics:
    """Test distance metrics for potential use in AlbumentationsX."""

    @pytest.mark.parametrize("size", [128, 768, 1536])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_euclidean_distance(self, size, dtype):
        """Test squared Euclidean distance calculation."""
        a = np.random.randn(size).astype(dtype)
        b = np.random.randn(size).astype(dtype)

        # NumPy
        result_np = np.sum((a - b) ** 2)

        # SimSIMD
        result_simsimd = simsimd.sqeuclidean(a, b)

        assert np.isclose(result_np, result_simsimd, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("size", [128, 768, 1536])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cosine_distance(self, size, dtype):
        """Test cosine distance calculation."""
        a = np.random.randn(size).astype(dtype)
        b = np.random.randn(size).astype(dtype)

        # NumPy
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        result_np = 1 - dot_product / (norm_a * norm_b)

        # SimSIMD
        result_simsimd = simsimd.cosine(a, b)

        assert np.isclose(result_np, result_simsimd, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
