import multiprocessing
import sys

import cv2
import numpy as np
import pytest

import albumentations as A

cv2.setRNGSeed(137)

np.random.seed(137)


@pytest.fixture
def mask():
    # Use independent RNG for this fixture to avoid order-dependent results
    rng = np.random.default_rng(137)
    return rng.integers(0, 2, (100, 100), dtype=np.uint8)


@pytest.fixture(scope="module")
def image():
    # Use independent RNG for this fixture to avoid order-dependent results
    rng = np.random.default_rng(137)
    return rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def bboxes():
    return np.array([[15, 12, 75, 30, 1], [55, 25, 90, 90, 2]])


@pytest.fixture
def volume():
    return np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)


@pytest.fixture
def mask3d():
    return np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)


@pytest.fixture
def albumentations_bboxes():
    return np.array([[0.15, 0.12, 0.75, 0.30, 1], [0.55, 0.25, 0.90, 0.90, 2]])


@pytest.fixture
def keypoints():
    return np.array([[30, 20, 0, 0.5, 1], [20, 30, 60, 2.5, 2]], dtype=np.float32)


@pytest.fixture
def template():
    return cv2.randu(np.zeros((100, 100, 3), dtype=np.uint8), 0, 255)


@pytest.fixture
def float_template():
    return cv2.randu(np.zeros((100, 100, 3), dtype=np.float32), 0, 1)


@pytest.fixture(scope="package")
def mp_pool():
    # Usage of `fork` as a start method for multiprocessing could lead to deadlocks on macOS.
    # Because `fork` was the default start method for macOS until Python 3.8
    # we had to manually set the start method to `spawn` to avoid those issues.
    if sys.platform == "darwin":
        method = "spawn"
    else:
        method = None
    return multiprocessing.get_context(method).Pool(4)


# Create independent RNGs for module-level constants to avoid order-dependent results
# Each constant gets its own seed derived from the constant name for reproducibility
def _make_rng(seed_offset: int) -> np.random.Generator:
    """Create an independent RNG with a unique seed."""
    return np.random.default_rng(137 + seed_offset)


# Optimized: Use np.random.default_rng for uint8 (2-3x faster than cv2.randu)
SQUARE_UINT8_IMAGE = _make_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
RECTANGULAR_UINT8_IMAGE = _make_rng(1).integers(0, 256, (101, 99, 3), dtype=np.uint8)

# Keep cv2.randu for float32 (2x faster than numpy)
SQUARE_FLOAT_IMAGE = cv2.randu(np.empty((100, 100, 3), dtype=np.float32), 0, 1)
RECTANGULAR_FLOAT_IMAGE = cv2.randu(np.empty((101, 99, 3), dtype=np.float32), 0, 1)

UINT8_IMAGES = [SQUARE_UINT8_IMAGE, RECTANGULAR_UINT8_IMAGE]

FLOAT32_IMAGES = [SQUARE_FLOAT_IMAGE, RECTANGULAR_FLOAT_IMAGE]

IMAGES = UINT8_IMAGES + FLOAT32_IMAGES
VOLUME = _make_rng(2).integers(0, 256, (4, 101, 99, 3), dtype=np.uint8)

SQUARE_IMAGES = [SQUARE_UINT8_IMAGE, SQUARE_FLOAT_IMAGE]
RECTANGULAR_IMAGES = [RECTANGULAR_UINT8_IMAGE, RECTANGULAR_FLOAT_IMAGE]

SQUARE_MULTI_UINT8_IMAGE = _make_rng(3).integers(0, 256, (100, 100, 5), dtype=np.uint8)
SQUARE_MULTI_FLOAT_IMAGE = _make_rng(4).uniform(0.0, 1.0, (100, 100, 5)).astype(np.float32)

MULTI_IMAGES = [SQUARE_MULTI_UINT8_IMAGE, SQUARE_MULTI_FLOAT_IMAGE]


@pytest.fixture
def image_float32():
    return cv2.randu(np.empty((100, 100, 3), dtype=np.float32), 0, 1)


# Module-scoped fixtures for large arrays (avoid recreation in parametrized tests)
# Each fixture uses its own independent RNG for reproducibility regardless of test order
@pytest.fixture(scope="module")
def large_image_1000x500():
    """Large uint8 image for resize and performance tests."""
    return _make_rng(10).integers(0, 256, (1000, 500, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def large_image_1000x800():
    """Large uint8 image for resize tests."""
    return _make_rng(11).integers(0, 256, (1000, 800, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def large_image_500x1000():
    """Large uint8 image for resize tests (portrait orientation)."""
    return _make_rng(12).integers(0, 256, (500, 1000, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def large_image_800x1000():
    """Large uint8 image for resize tests (portrait orientation)."""
    return _make_rng(13).integers(0, 256, (800, 1000, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def large_image_512x512():
    """512x512 uint8 image for benchmark and integration tests."""
    return _make_rng(14).integers(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def large_image_512x512_16ch():
    """512x512 16-channel uint8 image for multi-channel tests."""
    return _make_rng(15).integers(0, 256, (512, 512, 16), dtype=np.uint8)


@pytest.fixture(scope="module")
def large_float_array_1000x1000():
    """Large float32 array for SIMD integration tests."""
    return _make_rng(16).uniform(0, 1, (1000, 1000, 3)).astype(np.float32)


# Commonly used image sizes for functional tests
@pytest.fixture(scope="module")
def image_256x256_uint8():
    """256x256 uint8 image for equalize and other functional tests."""
    return _make_rng(20).integers(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def image_256x256_1ch_uint8():
    """256x256 single-channel uint8 image for grayscale tests."""
    return _make_rng(21).integers(0, 256, (256, 256, 1), dtype=np.uint8)


@pytest.fixture(scope="module")
def image_512x512_1ch_uint8():
    """512x512 single-channel uint8 image for mask tests."""
    return _make_rng(22).integers(0, 256, (512, 512, 1), dtype=np.uint8)


@pytest.fixture(scope="module")
def mask_256x256():
    """256x256 binary mask (uint8) for functional tests."""
    return _make_rng(23).integers(0, 2, (256, 256, 1), dtype=np.uint8)


@pytest.fixture
def compose_factory():
    """Factory for creating Compose instances with standard settings."""

    def _create(transforms, **kwargs):
        defaults = {"seed": 137, "strict": True}
        defaults.update(kwargs)
        return A.Compose(transforms, **defaults)

    return _create
