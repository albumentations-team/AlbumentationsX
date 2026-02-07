"""Benchmark SimSIMD operations vs OpenCV and NumPy.

This module benchmarks the performance of SimSIMD operations against their
OpenCV and NumPy equivalents to validate that SimSIMD provides speedups.

Usage:
    pytest tests/benchmark_simsimd.py -v --benchmark-only
    pytest tests/benchmark_simsimd.py -v --benchmark-compare
"""

import numpy as np
import pytest

# Try importing libraries
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

pytestmark = pytest.mark.skipif(
    not (HAS_CV2 and HAS_SIMSIMD),
    reason="Requires both cv2 and simsimd",
)


@pytest.mark.slow
class TestMatrixMultiplicationBenchmark:
    """Benchmark cv2.gemm vs simsimd.dot vs numpy.dot"""

    def test_benchmark_gemm_small(self, benchmark):
        """Benchmark small matrix multiplication (TPS use case)."""
        a = np.random.randn(10, 768).astype(np.float32)
        b = np.random.randn(768, 10).astype(np.float32)

        result = benchmark(cv2.gemm, a, b, 1, None, 0)
        assert result.shape == (10, 10)

    def test_benchmark_dot_small(self, benchmark):
        """Benchmark NumPy @ for small matrices."""
        a = np.random.randn(10, 768).astype(np.float32)
        b = np.random.randn(768, 10).astype(np.float32)

        def matmul():
            return a @ b

        result = benchmark(matmul)
        assert result.shape == (10, 10)

    def test_benchmark_gemm_medium(self, benchmark):
        """Benchmark medium matrix multiplication."""
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)

        result = benchmark(cv2.gemm, a, b, 1, None, 0)
        assert result.shape == (100, 100)

    def test_benchmark_dot_medium(self, benchmark):
        """Benchmark NumPy @ for medium matrices."""
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)

        def matmul():
            return a @ b

        result = benchmark(matmul)
        assert result.shape == (100, 100)


@pytest.mark.slow
class TestVectorDotProductBenchmark:
    """Benchmark vector dot products: np.dot vs simsimd.dot"""

    def test_benchmark_numpy_dot_1536(self, benchmark):
        """Benchmark NumPy dot product for 1536-dim vectors."""
        a = np.random.randn(1536).astype(np.float32)
        b = np.random.randn(1536).astype(np.float32)

        result = benchmark(np.dot, a, b)
        assert isinstance(result, (float, np.floating))

    def test_benchmark_simsimd_dot_1536(self, benchmark):
        """Benchmark SimSIMD dot product for 1536-dim vectors."""
        a = np.random.randn(1536).astype(np.float32)
        b = np.random.randn(1536).astype(np.float32)

        result = benchmark(simsimd.dot, a, b)
        assert isinstance(result, (float, np.floating))

    def test_benchmark_numpy_dot_768(self, benchmark):
        """Benchmark NumPy dot product for 768-dim vectors."""
        a = np.random.randn(768).astype(np.float32)
        b = np.random.randn(768).astype(np.float32)

        result = benchmark(np.dot, a, b)
        assert isinstance(result, (float, np.floating))

    def test_benchmark_simsimd_dot_768(self, benchmark):
        """Benchmark SimSIMD dot product for 768-dim vectors."""
        a = np.random.randn(768).astype(np.float32)
        b = np.random.randn(768).astype(np.float32)

        result = benchmark(simsimd.dot, a, b)
        assert isinstance(result, (float, np.floating))


@pytest.mark.slow
class TestWeightedSumBenchmark:
    """Benchmark cv2.addWeighted vs simsimd.wsum"""

    def test_benchmark_cv2_addweighted_rgb(self, benchmark):
        """Benchmark cv2.addWeighted for RGB images."""
        img1 = np.random.randn(512, 512, 3).astype(np.float32)
        img2 = np.random.randn(512, 512, 3).astype(np.float32)

        result = benchmark(cv2.addWeighted, img1, 0.6, img2, 0.4, 0.0)
        assert result.shape == (512, 512, 3)

    def test_benchmark_simsimd_wsum_rgb(self, benchmark):
        """Benchmark SimSIMD wsum for RGB images."""
        img1 = np.random.randn(512, 512, 3).astype(np.float32)
        img2 = np.random.randn(512, 512, 3).astype(np.float32)
        out = np.empty_like(img1)

        def wsum_wrapper():
            simsimd.wsum(img1.reshape(-1), img2.reshape(-1), alpha=0.6, beta=0.4, out=out.reshape(-1))
            return out

        result = benchmark(wsum_wrapper)
        assert result.shape == (512, 512, 3)

    def test_benchmark_numpy_weighted_sum_rgb(self, benchmark):
        """Benchmark NumPy weighted sum for RGB images."""
        img1 = np.random.randn(512, 512, 3).astype(np.float32)
        img2 = np.random.randn(512, 512, 3).astype(np.float32)

        def numpy_wsum():
            return 0.6 * img1 + 0.4 * img2

        result = benchmark(numpy_wsum)
        assert result.shape == (512, 512, 3)

    def test_benchmark_simsimd_wsum_hyperspectral(self, benchmark):
        """Benchmark SimSIMD wsum for hyperspectral (16 channels) - cv2 can't do this!"""
        img1 = np.random.randn(512, 512, 16).astype(np.float32)
        img2 = np.random.randn(512, 512, 16).astype(np.float32)
        out = np.empty_like(img1)

        def wsum_wrapper():
            simsimd.wsum(img1.reshape(-1), img2.reshape(-1), alpha=0.6, beta=0.4, out=out.reshape(-1))
            return out

        result = benchmark(wsum_wrapper)
        assert result.shape == (512, 512, 16)

    def test_benchmark_numpy_weighted_sum_hyperspectral(self, benchmark):
        """Benchmark NumPy weighted sum for hyperspectral (16 channels)."""
        img1 = np.random.randn(512, 512, 16).astype(np.float32)
        img2 = np.random.randn(512, 512, 16).astype(np.float32)

        def numpy_wsum():
            return 0.6 * img1 + 0.4 * img2

        result = benchmark(numpy_wsum)
        assert result.shape == (512, 512, 16)


@pytest.mark.slow
class TestFMABenchmark:
    """Benchmark fused multiply-add operations"""

    def test_benchmark_sequential_cv2(self, benchmark):
        """Benchmark sequential cv2.multiply + cv2.add."""
        a = np.random.randn(512, 512, 3).astype(np.float32)
        b = np.random.randn(512, 512, 3).astype(np.float32)
        c = np.random.randn(512, 512, 3).astype(np.float32)

        def sequential_ops():
            temp = cv2.multiply(a, b)
            temp = cv2.multiply(temp, np.array([0.7], dtype=np.float32))
            return cv2.add(temp, 0.3 * c)

        result = benchmark(sequential_ops)
        assert result.shape == (512, 512, 3)

    def test_benchmark_simsimd_fma(self, benchmark):
        """Benchmark SimSIMD fused multiply-add."""
        a = np.random.randn(512, 512, 3).astype(np.float32)
        b = np.random.randn(512, 512, 3).astype(np.float32)
        c = np.random.randn(512, 512, 3).astype(np.float32)
        out = np.empty_like(a)

        def fma_wrapper():
            simsimd.fma(a.reshape(-1), b.reshape(-1), c.reshape(-1), alpha=0.7, beta=0.3, out=out.reshape(-1))
            return out

        result = benchmark(fma_wrapper)
        assert result.shape == (512, 512, 3)

    def test_benchmark_numpy_fma(self, benchmark):
        """Benchmark NumPy fused multiply-add equivalent."""
        a = np.random.randn(512, 512, 3).astype(np.float32)
        b = np.random.randn(512, 512, 3).astype(np.float32)
        c = np.random.randn(512, 512, 3).astype(np.float32)

        def numpy_fma():
            return 0.7 * a * b + 0.3 * c

        result = benchmark(numpy_fma)
        assert result.shape == (512, 512, 3)


@pytest.mark.slow
class TestDistanceBenchmark:
    """Benchmark distance metrics"""

    def test_benchmark_numpy_euclidean(self, benchmark):
        """Benchmark NumPy squared Euclidean distance."""
        a = np.random.randn(1536).astype(np.float32)
        b = np.random.randn(1536).astype(np.float32)

        def numpy_sqeuclidean():
            return np.sum((a - b) ** 2)

        result = benchmark(numpy_sqeuclidean)
        assert isinstance(result, (float, np.floating))

    def test_benchmark_simsimd_euclidean(self, benchmark):
        """Benchmark SimSIMD squared Euclidean distance."""
        a = np.random.randn(1536).astype(np.float32)
        b = np.random.randn(1536).astype(np.float32)

        result = benchmark(simsimd.sqeuclidean, a, b)
        assert isinstance(result, (float, np.floating))

    def test_benchmark_numpy_cosine(self, benchmark):
        """Benchmark NumPy cosine distance."""
        a = np.random.randn(1536).astype(np.float32)
        b = np.random.randn(1536).astype(np.float32)

        def numpy_cosine():
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return 1 - dot / (norm_a * norm_b)

        result = benchmark(numpy_cosine)
        assert isinstance(result, (float, np.floating))

    def test_benchmark_simsimd_cosine(self, benchmark):
        """Benchmark SimSIMD cosine distance."""
        a = np.random.randn(1536).astype(np.float32)
        b = np.random.randn(1536).astype(np.float32)

        result = benchmark(simsimd.cosine, a, b)
        assert isinstance(result, (float, np.floating))


if __name__ == "__main__":
    # Run with: pytest tests/benchmark_simsimd.py -v --benchmark-only
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-sort=name"])
