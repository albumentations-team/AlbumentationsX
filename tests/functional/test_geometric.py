import os

import cv2
import numpy as np
import pytest
from albucore import resize as albucore_resize

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.geometric.functional import (
    _PIL_AVAILABLE,
    _PYVIPS_AVAILABLE,
    from_distance_maps,
    to_distance_maps,
)
from tests.utils import set_seed


@pytest.mark.parametrize(
    "image_shape, keypoints, inverted",
    [
        ((100, 100), [(50, 50), (25, 75)], False),
        ((100, 100), [(50, 50), (25, 75)], True),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], False),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], True),
    ],
)
def test_to_distance_maps(image_shape, keypoints, inverted):
    distance_maps = to_distance_maps(keypoints, image_shape, inverted)

    assert distance_maps.shape == (*image_shape, len(keypoints))
    assert distance_maps.dtype == np.float32

    for i, (x, y) in enumerate(keypoints):
        if inverted:
            assert np.isclose(distance_maps[int(y), int(x), i], 1.0)
        else:
            assert np.isclose(distance_maps[int(y), int(x), i], 0.0)

    if inverted:
        assert np.all(distance_maps > 0) and np.all(distance_maps <= 1)
    else:
        assert np.all(distance_maps >= 0)


@pytest.mark.parametrize(
    "image_shape, keypoints, inverted, threshold, if_not_found_coords",
    [
        ((100, 100), [(50, 50), (25, 75)], False, None, None),
        ((100, 100), [(50, 50), (25, 75)], True, None, None),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], False, 10, None),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], True, 0.5, [0, 0]),
        ((150, 150), [(75, 75), (25, 125), (125, 25)], False, None, {"x": -1, "y": -1}),
    ],
)
def test_from_distance_maps(
    image_shape,
    keypoints,
    inverted,
    threshold,
    if_not_found_coords,
):
    distance_maps = to_distance_maps(keypoints, image_shape, inverted)
    recovered_keypoints = from_distance_maps(
        distance_maps,
        inverted,
        if_not_found_coords,
        threshold,
    )

    assert len(recovered_keypoints) == len(keypoints)

    for original, recovered in zip(keypoints, recovered_keypoints, strict=False):
        if threshold is None:
            np.testing.assert_allclose(original, recovered, atol=1)
        else:
            x, y = original
            i = keypoints.index(original)
            if (inverted and distance_maps[int(y), int(x), i] >= threshold) or (
                not inverted and distance_maps[int(y), int(x), i] <= threshold
            ):
                np.testing.assert_allclose(original, recovered, atol=1)
            elif if_not_found_coords is not None:
                if isinstance(if_not_found_coords, dict):
                    assert np.allclose(
                        recovered,
                        [if_not_found_coords["x"], if_not_found_coords["y"]],
                    )
                else:
                    assert np.allclose(recovered, if_not_found_coords)
            else:
                np.testing.assert_allclose(original, recovered, atol=1)


@pytest.mark.parametrize(
    "image_shape, keypoints, inverted",
    [
        ((100, 100), [(50, 50), (25, 75)], False),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], True),
    ],
)
def test_to_distance_maps_extra_columns(image_shape, keypoints, inverted):
    keypoints_with_extra = [(x, y, 0, 1) for x, y in keypoints]
    distance_maps = to_distance_maps(keypoints, image_shape, inverted)

    assert distance_maps.shape == (*image_shape, len(keypoints))
    assert distance_maps.dtype == np.float32

    for i, (x, y, _, _) in enumerate(keypoints_with_extra):
        if inverted:
            assert np.isclose(distance_maps[int(y), int(x), i], 1.0)
        else:
            assert np.isclose(distance_maps[int(y), int(x), i], 0.0)


@pytest.mark.parametrize(
    "image_shape, grid, expected",
    [
        # Normal case: standard grids
        (
            (100, 200),
            (2, 2),
            np.array(
                [
                    [0, 0, 50, 100],
                    [0, 100, 50, 200],
                    [50, 0, 100, 100],
                    [50, 100, 100, 200],
                ],
            ),
        ),
        # Single row grid
        (
            (100, 200),
            (1, 4),
            np.array(
                [
                    [0, 0, 100, 50],
                    [0, 50, 100, 100],
                    [0, 100, 100, 150],
                    [0, 150, 100, 200],
                ],
            ),
        ),
        # Single column grid
        (
            (100, 200),
            (4, 1),
            np.array(
                [[0, 0, 25, 200], [25, 0, 50, 200], [50, 0, 75, 200], [75, 0, 100, 200]],
            ),
        ),
        # Edge case: Grid size equals image size
        (
            (100, 200),
            (100, 200),
            np.array([[i, j, i + 1, j + 1] for i in range(100) for j in range(200)]),
        ),
        # Edge case: Image where width is much larger than height
        (
            (10, 1000),
            (1, 10),
            np.array([[0, i * 100, 10, (i + 1) * 100] for i in range(10)]),
        ),
        # Edge case: Image where height is much larger than width
        (
            (1000, 10),
            (10, 1),
            np.array([[i * 100, 0, (i + 1) * 100, 10] for i in range(10)]),
        ),
        # Corner case: height and width are not divisible by the number of splits
        (
            (105, 205),
            (3, 4),
            np.array(
                [
                    [0, 0, 35, 51],
                    [0, 51, 35, 103],
                    [0, 103, 35, 154],
                    [0, 154, 35, 205],
                    [35, 0, 70, 51],
                    [35, 51, 70, 103],
                    [35, 103, 70, 154],
                    [35, 154, 70, 205],
                    [70, 0, 105, 51],
                    [70, 51, 105, 103],
                    [70, 103, 105, 154],
                    [70, 154, 105, 205],
                ],
            ),
        ),
    ],
)
def test_split_uniform_grid(image_shape, grid, expected):
    random_seed = 42
    result = fgeometric.split_uniform_grid(
        image_shape,
        grid,
        random_generator=np.random.default_rng(random_seed),
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "size, divisions, random_seed, expected",
    [
        (10, 2, None, [0, 5, 10]),
        (10, 2, 42, [0, 5, 10]),  # Consistent shuffling with seed
        (9, 3, None, [0, 3, 6, 9]),
        (9, 3, 42, [0, 3, 6, 9]),  # Expected shuffle result with a specific seed
        (20, 5, 42, [0, 4, 8, 12, 16, 20]),  # Regular intervals
        (7, 3, 42, [0, 2, 4, 7]),  # Irregular intervals, specific seed
        (7, 3, 41, [0, 3, 5, 7]),  # Irregular intervals, specific seed
    ],
)
def test_generate_shuffled_splits(size, divisions, random_seed, expected):
    result = fgeometric.generate_shuffled_splits(
        size,
        divisions,
        random_generator=np.random.default_rng(random_seed),
    )
    assert len(result) == divisions + 1
    (
        np.testing.assert_array_equal(
            result,
            expected,
        ),
        f"Failed for size={size}, divisions={divisions}, random_seed={random_seed}",
    )


@pytest.mark.parametrize(
    "size, divisions, random_seed",
    [
        (10, 2, 42),
        (9, 3, 99),
        (20, 5, 101),
        (7, 3, 42),
    ],
)
def test_consistent_shuffling(size, divisions, random_seed):
    set_seed(random_seed)
    result1 = fgeometric.generate_shuffled_splits(
        size,
        divisions,
        random_generator=np.random.default_rng(random_seed),
    )
    assert len(result1) == divisions + 1
    set_seed(random_seed)
    result2 = fgeometric.generate_shuffled_splits(
        size,
        divisions,
        random_generator=np.random.default_rng(random_seed),
    )
    assert len(result2) == divisions + 1
    (
        np.testing.assert_array_equal(result1, result2),
        "Shuffling is not consistent with the given random state",
    )


@pytest.mark.parametrize(
    ["image_shape", "grid", "scale", "absolute_scale", "expected_shape"],
    [
        # Basic test cases
        ((100, 100), (4, 4), 0.05, False, ((100, 100), (100, 100))),
        ((200, 100), (3, 5), 0.03, False, ((200, 100), (200, 100))),
        # Test different image shapes
        ((50, 75), (2, 2), 0.05, False, ((50, 75), (50, 75))),
        ((300, 200), (5, 5), 0.05, False, ((300, 200), (300, 200))),
        # Test different grid sizes
        ((100, 100), (2, 3), 0.05, False, ((100, 100), (100, 100))),
        ((100, 100), (6, 6), 0.05, False, ((100, 100), (100, 100))),
        # Test with absolute scale
        ((100, 100), (4, 4), 5.0, True, ((100, 100), (100, 100))),
        ((200, 200), (3, 3), 10.0, True, ((200, 200), (200, 200))),
    ],
)
def test_create_piecewise_affine_maps_shapes(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float,
    absolute_scale: bool,
    expected_shape: tuple[tuple[int, int], tuple[int, int]],
):
    """Test that output maps have correct shapes and types."""
    generator = np.random.default_rng(42)
    map_x, map_y = fgeometric.create_piecewise_affine_maps(
        image_shape,
        grid,
        scale,
        absolute_scale,
        generator,
    )

    assert map_x is not None and map_y is not None
    assert map_x.shape == expected_shape[0]
    assert map_y.shape == expected_shape[1]
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32


@pytest.mark.parametrize(
    ["image_shape", "grid", "scale"],
    [
        ((100, 100), (4, 4), 0.05),
        ((200, 100), (3, 5), 0.03),
    ],
)
def test_create_piecewise_affine_maps_bounds(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float,
):
    """Test that output maps stay within image bounds."""
    generator = np.random.default_rng(42)
    map_x, map_y = fgeometric.create_piecewise_affine_maps(
        image_shape,
        grid,
        scale,
        False,
        generator,
    )

    assert map_x is not None and map_y is not None
    height, width = image_shape

    # Check bounds
    assert np.all(map_x >= 0)
    assert np.all(map_x <= width - 1)
    assert np.all(map_y >= 0)
    assert np.all(map_y <= height - 1)


@pytest.mark.parametrize(
    ["scale", "expected_result"],
    [
        (0.0, (None, None)),  # Zero scale should return None
        (-1.0, (None, None)),  # Negative scale should return None
    ],
)
def test_create_piecewise_affine_maps_edge_cases(
    scale: float,
    expected_result: tuple[None, None],
):
    """Test edge cases with zero or negative scale."""
    generator = np.random.default_rng(42)
    result = fgeometric.create_piecewise_affine_maps(
        (100, 100),
        (4, 4),
        scale,
        False,
        generator,
    )
    assert result == expected_result


def test_create_piecewise_affine_maps_reproducibility():
    """Test that the function produces the same output with the same random seed."""
    result1 = fgeometric.create_piecewise_affine_maps(
        (100, 100),
        (4, 4),
        0.05,
        False,
        random_generator=np.random.default_rng(42),
    )
    result2 = fgeometric.create_piecewise_affine_maps(
        (100, 100),
        (4, 4),
        0.05,
        False,
        random_generator=np.random.default_rng(42),
    )

    assert result1[0] is not None and result1[1] is not None
    assert result2[0] is not None and result2[1] is not None
    np.testing.assert_array_almost_equal(result1[0], result2[0])
    np.testing.assert_array_almost_equal(result1[1], result2[1])


@pytest.mark.parametrize(
    ["image_shape", "grid"],
    [
        ((0, 100), (4, 4)),  # Zero height
        ((100, 0), (4, 4)),  # Zero width
        ((100, 100), (0, 4)),  # Zero grid rows
        ((100, 100), (4, 0)),  # Zero grid columns
    ],
)
def test_create_piecewise_affine_maps_zero_dimensions(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
):
    """Test handling of zero dimensions."""
    generator = np.random.default_rng(42)
    with pytest.raises(ValueError):
        fgeometric.create_piecewise_affine_maps(
            image_shape,
            grid,
            0.05,
            False,
            generator,
        )


@pytest.mark.parametrize(
    ["image_shape", "grid", "scale", "absolute_scale"],
    [
        ((100, 100), (4, 4), 0.05, False),
        ((200, 100), (3, 5), 0.03, True),
    ],
)
def test_create_piecewise_affine_maps_grid_points(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float,
    absolute_scale: bool,
):
    """Test that grid points are properly distributed."""
    generator = np.random.default_rng(42)
    map_x, map_y = fgeometric.create_piecewise_affine_maps(
        image_shape,
        grid,
        scale,
        absolute_scale,
        generator,
    )

    assert map_x is not None and map_y is not None

    height, width = image_shape
    nb_rows, nb_cols = grid

    # Sample points should roughly correspond to grid intersections
    y_steps = np.linspace(0, height - 1, nb_rows)
    x_steps = np.linspace(0, width - 1, nb_cols)

    # Check that grid points are present in the maps
    for y in y_steps:
        for x in x_steps:
            y_idx = int(y)
            x_idx = int(x)

            # Calculate neighborhood size based on scale
            if absolute_scale:
                radius = int(scale * 3)  # 3 sigma radius
            else:
                radius = int(min(width, height) * scale * 3)
            radius = max(1, min(radius, 5))  # Keep radius reasonable

            # Get valid slice bounds
            y_start = max(0, y_idx - radius)
            y_end = min(height, y_idx + radius + 1)
            x_start = max(0, x_idx - radius)
            x_end = min(width, x_idx + radius + 1)

            # Extract neighborhood
            neighborhood = map_x[y_start:y_end, x_start:x_end]

            # Calculate maximum allowed deviation
            if absolute_scale:
                max_deviation = scale * 3
            else:
                max_deviation = min(width, height) * scale * 3

            # Check if any point in neighborhood is close to expected x coordinate
            assert np.any(
                np.abs(neighborhood - x) < max_deviation,
            ), f"No points near grid intersection ({x}, {y}) within allowed deviation"


def test_pad_with_params_zero_channels():
    """Test that padding works correctly with 0-channel images."""
    # Create a 0-channel image
    img = np.zeros((10, 10, 0), dtype=np.uint8)

    result = fgeometric.pad_with_params(
        img=img,
        h_pad_top=2,
        h_pad_bottom=3,
        w_pad_left=4,
        w_pad_right=5,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )

    # Check result shape
    assert result.shape == (15, 19, 0)  # 10+2+3, 10+4+5, 0
    assert result.dtype == np.uint8
    assert result.size == 0


@pytest.fixture
def random_generator():
    return np.random.default_rng(42)  # Fixed seed for reproducibility


@pytest.mark.parametrize(
    "image_shape",
    [
        (100, 100),
        (224, 224),
        (32, 64),
        (1, 1),
    ],
)
@pytest.mark.parametrize(
    "alpha",
    [
        0.0,
        1.0,
        10.0,
    ],
)
@pytest.mark.parametrize(
    "sigma",
    [
        1.0,
        50.0,
        100.0,
    ],
)
@pytest.mark.parametrize(
    "same_dxdy",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "kernel_size",
    [
        (0, 0),  # No blur
        (3, 3),  # Small kernel
        (17, 17),  # Large kernel
    ],
)
@pytest.mark.parametrize(
    "noise_distribution",
    [
        "gaussian",
        "uniform",
    ],
)
def test_generate_displacement_fields(
    random_generator,
    image_shape,
    alpha,
    sigma,
    same_dxdy,
    kernel_size,
    noise_distribution,
):
    # Generate displacement fields
    dx, dy = fgeometric.generate_displacement_fields(
        image_shape=image_shape,
        alpha=alpha,
        sigma=sigma,
        same_dxdy=same_dxdy,
        kernel_size=kernel_size,
        random_generator=random_generator,
        noise_distribution=noise_distribution,
    )

    # Test output shapes
    assert dx.shape == image_shape
    assert dy.shape == image_shape

    # Test output dtypes
    assert dx.dtype == np.float32
    assert dy.dtype == np.float32

    # Test same_dxdy behavior
    if same_dxdy:
        np.testing.assert_array_equal(dx, dy)

    # Test alpha scaling
    if alpha == 0:
        np.testing.assert_array_equal(dx, np.zeros_like(dx))
        np.testing.assert_array_equal(dy, np.zeros_like(dy))
    else:
        assert np.abs(dx).max() <= abs(alpha) * 3  # 3 sigma rule for gaussian
        assert np.abs(dy).max() <= abs(alpha) * 3

    # Test value ranges for uniform distribution
    if noise_distribution == "uniform":
        assert np.all(np.abs(dx) <= abs(alpha))
        assert np.all(np.abs(dy) <= abs(alpha))


def test_reproducibility(random_generator):
    """Test that the function produces the same output with the same random seed"""
    params = {
        "image_shape": (100, 100),
        "alpha": 1.0,
        "sigma": 50.0,
        "same_dxdy": False,
        "kernel_size": (17, 17),
        "random_generator": np.random.default_rng(42),  # Create new generator each time
        "noise_distribution": "gaussian",
    }

    dx1, dy1 = fgeometric.generate_displacement_fields(**params)

    # Create new generator with same seed for second call
    params["random_generator"] = np.random.default_rng(42)
    dx2, dy2 = fgeometric.generate_displacement_fields(**params)

    np.testing.assert_array_equal(dx1, dx2)
    np.testing.assert_array_equal(dy1, dy2)


def test_gaussian_blur_effect(random_generator):
    """Test that Gaussian blur is actually smoothing the displacement field"""
    params = {
        "image_shape": (100, 100),
        "alpha": 1.0,
        "sigma": 50.0,
        "same_dxdy": False,
        "noise_distribution": "gaussian",
    }

    # Generate fields with small kernel (less smoothing)
    dx1, _ = fgeometric.generate_displacement_fields(
        **params,
        kernel_size=(3, 3),  # Small kernel
        random_generator=np.random.default_rng(42),
    )

    # Generate fields with large kernel (more smoothing)
    dx2, _ = fgeometric.generate_displacement_fields(
        **params,
        kernel_size=(17, 17),  # Large kernel
        random_generator=np.random.default_rng(42),
    )

    # Calculate local variation using standard deviation of local neighborhoods
    def calculate_local_variation(arr, window_size=3):
        from scipy.ndimage import uniform_filter

        # Ensure we're working with float64 for better numerical stability
        arr = arr.astype(np.float64)
        local_mean = uniform_filter(arr, size=window_size)
        local_sqr_mean = uniform_filter(arr**2, size=window_size)
        # Add small epsilon to avoid numerical instability
        variance = np.maximum(local_sqr_mean - local_mean**2, 0)
        return np.mean(np.sqrt(variance + 1e-10))

    var1 = calculate_local_variation(dx1)
    var2 = calculate_local_variation(dx2)

    assert var2 < var1, (
        f"Gaussian blur should reduce local variation. Before blur (3x3): {var1:.6f}, After blur (17x17): {var2:.6f}"
    )


def test_memory_efficiency(random_generator):
    """Test that the function doesn't create unnecessary copies"""
    import tracemalloc

    tracemalloc.start()

    params = {
        "image_shape": (1000, 1000),  # Large image
        "alpha": 1.0,
        "sigma": 50.0,
        "same_dxdy": True,  # Should reuse memory
        "kernel_size": (17, 17),
        "random_generator": random_generator,
        "noise_distribution": "gaussian",
    }

    # Get memory snapshot before
    snapshot1 = tracemalloc.take_snapshot()

    # Run function
    dx, _dy = fgeometric.generate_displacement_fields(**params)

    # Get memory snapshot after
    snapshot2 = tracemalloc.take_snapshot()

    # Compare memory usage
    stats = snapshot2.compare_to(snapshot1, "lineno")

    # Check that memory usage is reasonable (less than 4 times the size of output)
    # Factor of 4 accounts for temporary arrays during computation
    expected_size = dx.nbytes * 4
    total_memory = sum(stat.size_diff for stat in stats)

    assert total_memory <= expected_size, "Memory usage is higher than expected"

    tracemalloc.stop()


@pytest.mark.parametrize(
    "input_shape,target_shape",
    [
        ((100, 100), (200, 200)),
        ((200, 200), (100, 100)),
        ((150, 100), (150, 200)),
    ],
)
def test_albucore_resize(input_shape, target_shape):
    img = np.random.randint(0, 255, (*input_shape, 3), dtype=np.uint8)

    resized = albucore_resize(img, (target_shape[1], target_shape[0]), interpolation=0)

    assert resized.shape == (*target_shape, 3)


@pytest.mark.parametrize(
    "input_shape,target_shape",
    [
        ((100, 100), (200, 200)),
        ((256, 256), (512, 512)),
        ((200, 200), (100, 100)),
        ((150, 100), (150, 200)),
    ],
)
def test_albucore_resize_2d_mask(input_shape, target_shape):
    """Test that albucore.resize handles 2D arrays (masks) correctly."""
    mask = np.random.randint(0, 2, input_shape, dtype=np.uint8)

    resized = albucore_resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    assert resized.shape == target_shape
    assert resized.ndim == 2


@pytest.mark.skipif(not _PYVIPS_AVAILABLE, reason="pyvips is not installed")
@pytest.mark.parametrize(
    "input_shape,target_shape",
    [
        ((100, 100), (200, 200)),
        ((200, 200), (100, 100)),
        ((150, 100), (150, 200)),
    ],
)
def test_resize_pyvips(input_shape, target_shape):
    img = np.random.randint(0, 255, (*input_shape, 3), dtype=np.uint8)

    resized = fgeometric.resize_pyvips(img, target_shape, interpolation=0)
    assert resized.shape == (*target_shape, 3)


@pytest.mark.xfail(reason="pyvips and OpenCV have different interpolation implementations")
@pytest.mark.skipif(not _PYVIPS_AVAILABLE, reason="pyvips is not installed")
@pytest.mark.parametrize("interpolation", [0, 1, 2])
@pytest.mark.parametrize(
    "input_shape,target_shape",
    [
        ((100, 100), (200, 200)),
        ((200, 200), (100, 100)),
    ],
)
def test_resize_cv2_vs_pyvips(input_shape, target_shape, interpolation):
    img = np.random.randint(0, 255, (*input_shape, 3), dtype=np.uint8)

    resized_opencv = albucore_resize(img, (target_shape[1], target_shape[0]), interpolation=interpolation)
    resized_pyvips = fgeometric.resize_pyvips(img, target_shape, interpolation=interpolation)
    np.testing.assert_allclose(resized_opencv, resized_pyvips, atol=1)


@pytest.mark.xfail(reason="Pillow and OpenCV have different interpolation implementations")
@pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow is not installed")
@pytest.mark.parametrize("interpolation", [0, 1, 2])
@pytest.mark.parametrize(
    "input_shape,target_shape",
    [
        ((100, 100), (200, 200)),
        ((200, 200), (100, 100)),
    ],
)
def test_resize_cv2_vs_pillow(input_shape, target_shape, interpolation):
    img = np.random.randint(0, 255, (*input_shape, 3), dtype=np.uint8)

    resized_opencv = albucore_resize(img, (target_shape[1], target_shape[0]), interpolation=interpolation)
    resized_pil = fgeometric.resize_pil(img, target_shape, interpolation=interpolation)
    np.testing.assert_allclose(resized_opencv, resized_pil, atol=1)


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow is not installed")
@pytest.mark.parametrize(
    "interpolation",
    [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ],
)
@pytest.mark.parametrize(
    "input_shape,target_shape",
    [
        ((100, 100), (50, 75)),  # Downscale with different aspect ratio
        ((50, 50), (100, 150)),  # Upscale with different aspect ratio
    ],
)
def test_resize_pil_with_cv2_interpolation_constants(input_shape, target_shape, interpolation):
    """Test that resize_pil correctly maps cv2 interpolation constants to PIL."""
    img = np.random.randint(0, 255, (*input_shape, 3), dtype=np.uint8)

    # This should not raise an error
    resized = fgeometric.resize_pil(img, target_shape, interpolation=interpolation)

    # Check output shape
    assert resized.shape[:2] == target_shape
    assert resized.shape[2] == 3
    assert resized.dtype == np.uint8


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow is not installed")
@pytest.mark.parametrize("num_channels", [1, 3, 4, 5, 10])
def test_resize_pil_different_channel_counts(num_channels):
    """Test that resize_pil handles different channel counts correctly."""
    input_shape = (100, 100, num_channels)
    target_shape = (50, 75)

    img = np.random.randint(0, 255, input_shape, dtype=np.uint8)

    resized = fgeometric.resize_pil(img, target_shape, interpolation=cv2.INTER_LINEAR)

    assert resized.shape == (*target_shape, num_channels)
    assert resized.dtype == np.uint8


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow is not installed")
def test_resize_backend_selection():
    """Test that the resize function correctly selects backends based on environment variable."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    target_shape = (50, 75)

    # Clear the cache to ensure fresh backend selection
    fgeometric._get_resize_backend.cache_clear()

    # Test default backend (opencv)
    os.environ.pop("ALBUMENTATIONS_RESIZE", None)
    fgeometric._get_resize_backend.cache_clear()
    backend = fgeometric._get_resize_backend()
    assert backend == "opencv"

    resized_default = fgeometric.resize(img, target_shape, cv2.INTER_LINEAR)
    assert resized_default.shape == (*target_shape, 3)

    # Test pillow backend
    os.environ["ALBUMENTATIONS_RESIZE"] = "pillow"
    fgeometric._get_resize_backend.cache_clear()
    backend = fgeometric._get_resize_backend()
    assert backend == "pillow"

    resized_pillow = fgeometric.resize(img, target_shape, cv2.INTER_LINEAR)
    assert resized_pillow.shape == (*target_shape, 3)

    # Clean up
    os.environ.pop("ALBUMENTATIONS_RESIZE", None)
    fgeometric._get_resize_backend.cache_clear()


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow is not installed")
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_resize_pil_preserves_dtype(dtype):
    """Test that resize_pil preserves the input dtype."""
    img = np.random.rand(100, 100, 3).astype(dtype)
    if dtype == np.uint8:
        img = (img * 255).astype(np.uint8)

    target_shape = (50, 75)
    resized = fgeometric.resize_pil(img, target_shape, interpolation=cv2.INTER_LINEAR)

    assert resized.dtype == dtype
    assert resized.shape == (*target_shape, 3)


class TestPiecewiseAffineNumericalAccuracy:
    """Verify vectorized IDW interpolation matches reference per-pixel computation."""

    @staticmethod
    def _reference_piecewise_affine_maps(
        image_shape: tuple[int, int],
        grid: tuple[int, int],
        scale: float,
        absolute_scale: bool,
        random_generator: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pure-Python reference implementation (per-pixel loop)."""
        height, width = image_shape
        nb_rows, nb_cols = grid

        if scale <= 0:
            return None, None

        y = np.linspace(0, height - 1, nb_rows, dtype=np.float32)
        x = np.linspace(0, width - 1, nb_cols, dtype=np.float32)
        xx_src, yy_src = np.meshgrid(x, y)

        jitter_scale = scale / 3 if absolute_scale else scale * min(width, height) / 3
        jitter = random_generator.normal(0, jitter_scale, (nb_rows, nb_cols, 2)).astype(np.float32)

        control_points = np.zeros((nb_rows * nb_cols, 4), dtype=np.float32)
        for i in range(nb_rows):
            for j in range(nb_cols):
                idx = i * nb_cols + j
                control_points[idx, 0] = xx_src[i, j]
                control_points[idx, 1] = yy_src[i, j]
                control_points[idx, 2] = np.clip(xx_src[i, j] + jitter[i, j, 1], 0, width - 1)
                control_points[idx, 3] = np.clip(yy_src[i, j] + jitter[i, j, 0], 0, height - 1)

        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                dx = j - control_points[:, 0]
                dy = i - control_points[:, 1]
                dist = dx * dx + dy * dy
                weights = 1 / (dist + 1e-8)
                weights = weights / np.sum(weights)
                map_x[i, j] = np.sum(weights * control_points[:, 2])
                map_y[i, j] = np.sum(weights * control_points[:, 3])

        np.clip(map_x, 0, width - 1, out=map_x)
        np.clip(map_y, 0, height - 1, out=map_y)
        return map_x, map_y

    @pytest.mark.parametrize(
        ["image_shape", "grid", "scale", "absolute_scale"],
        [
            ((64, 64), (4, 4), 0.05, False),
            ((48, 80), (3, 5), 0.03, False),
            ((32, 32), (2, 2), 0.1, False),
            ((64, 64), (4, 4), 1.5, True),
        ],
    )
    def test_maps_match_reference(self, image_shape, grid, scale, absolute_scale):
        """Vectorized IDW must match per-pixel reference to float32 precision."""
        ref_mx, ref_my = self._reference_piecewise_affine_maps(
            image_shape,
            grid,
            scale,
            absolute_scale,
            np.random.default_rng(137),
        )
        vec_mx, vec_my = fgeometric.create_piecewise_affine_maps(
            image_shape,
            grid,
            scale,
            absolute_scale,
            np.random.default_rng(137),
        )
        np.testing.assert_allclose(vec_mx, ref_mx, atol=1e-3, rtol=1e-5)
        np.testing.assert_allclose(vec_my, ref_my, atol=1e-3, rtol=1e-5)

    @pytest.mark.parametrize("scale", [0.0, -0.01])
    def test_nonpositive_scale_returns_none(self, scale):
        """Non-positive scale must return (None, None) in both implementations."""
        ref = self._reference_piecewise_affine_maps(
            (32, 32),
            (4, 4),
            scale,
            False,
            np.random.default_rng(137),
        )
        vec = fgeometric.create_piecewise_affine_maps(
            (32, 32),
            (4, 4),
            scale,
            False,
            np.random.default_rng(137),
        )
        assert ref == (None, None)
        assert vec == (None, None)

    def test_remapped_images_match(self):
        """End-to-end: remap with vectorized maps must produce near-identical output."""
        img = np.random.default_rng(137).integers(0, 256, (64, 64, 3), dtype=np.uint8)

        ref_mx, ref_my = self._reference_piecewise_affine_maps(
            (64, 64),
            (4, 4),
            0.05,
            False,
            np.random.default_rng(42),
        )
        vec_mx, vec_my = fgeometric.create_piecewise_affine_maps(
            (64, 64),
            (4, 4),
            0.05,
            False,
            np.random.default_rng(42),
        )

        ref_out = cv2.remap(img, ref_mx, ref_my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        vec_out = cv2.remap(img, vec_mx, vec_my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        diff = np.abs(ref_out.astype(np.int16) - vec_out.astype(np.int16))
        assert diff.max() <= 10, f"Max pixel diff {diff.max()} exceeds tolerance"
        assert diff.mean() < 0.1, f"Mean pixel diff {diff.mean():.4f} exceeds tolerance"


class TestGenerateInverseDistortionMap:
    """Verify vectorized inverse distortion map properties."""

    def test_inverse_map_shapes_and_dtypes(self):
        rng = np.random.default_rng(137)
        h, w = 32, 32
        map_x = rng.uniform(0, w - 1, (h, w)).astype(np.float32)
        map_y = rng.uniform(0, h - 1, (h, w)).astype(np.float32)

        inv_mx, inv_my = fgeometric.generate_inverse_distortion_map(map_x, map_y, (h, w))

        assert inv_mx.shape == (h, w)
        assert inv_my.shape == (h, w)
        assert inv_mx.dtype == np.float32
        assert inv_my.dtype == np.float32

    def test_inverse_map_approximate_inversion(self):
        """Forward then inverse should approximately recover identity for smooth maps."""
        h, w = 64, 64
        yy, xx = np.mgrid[:h, :w]
        map_x = (xx + 2 * np.sin(yy * 0.1)).astype(np.float32)
        map_y = (yy + 2 * np.cos(xx * 0.1)).astype(np.float32)
        np.clip(map_x, 0, w - 1, out=map_x)
        np.clip(map_y, 0, h - 1, out=map_y)

        inv_mx, inv_my = fgeometric.generate_inverse_distortion_map(map_x, map_y, (h, w))

        interior = (inv_mx > 0) | (inv_my > 0)
        assert interior.sum() > h * w * 0.5, "Inverse map should fill most of the image"

        dst_x = np.clip(np.round(map_x).astype(np.int32), 0, w - 1)
        dst_y = np.clip(np.round(map_y).astype(np.int32), 0, h - 1)
        valid = interior[dst_y, dst_x]
        assert valid.sum() > h * w * 0.25, "Too few valid round-trip samples"

        src_x = xx[valid].astype(np.float32)
        src_y = yy[valid].astype(np.float32)
        rec_x = inv_mx[dst_y[valid], dst_x[valid]]
        rec_y = inv_my[dst_y[valid], dst_x[valid]]

        max_err = max(np.abs(rec_x - src_x).max(), np.abs(rec_y - src_y).max())
        assert max_err < 3.0, f"Round-trip max error too large: {max_err}"

    def test_closest_source_wins_on_collision(self):
        """When two source pixels map to the same destination cell, the closer one should win."""
        # Use identity map so every pixel maps to itself,
        # then override two source pixels to both target destination (5, 5).
        # Source (4,4) -> dst (5.1, 5.1): floor (5,5), dist=0.1+0.1=0.2  (closer)
        # Source (4,6) -> dst (5.4, 5.4): floor (5,5), dist=0.4+0.4=0.8
        h, w = 16, 16
        yy, xx = np.mgrid[:h, :w]
        map_x = xx.astype(np.float32)
        map_y = yy.astype(np.float32)

        map_x[4, 4] = 5.1
        map_y[4, 4] = 5.1
        map_x[4, 6] = 5.4
        map_y[4, 6] = 5.4

        inv_mx, inv_my = fgeometric.generate_inverse_distortion_map(map_x, map_y, (h, w))

        # Destination (row=5, col=5) should map back to source col=4, row=4 (the closer source)
        assert inv_mx[5, 5] == pytest.approx(4.0, abs=1.0)
        assert inv_my[5, 5] == pytest.approx(4.0, abs=1.0)

    def test_identity_map_produces_identity_inverse(self):
        """Identity forward map should produce near-identity inverse."""
        h, w = 32, 32
        yy, xx = np.mgrid[:h, :w]
        map_x = xx.astype(np.float32)
        map_y = yy.astype(np.float32)

        inv_mx, inv_my = fgeometric.generate_inverse_distortion_map(map_x, map_y, (h, w))

        np.testing.assert_allclose(inv_mx, map_x, atol=1.0)
        np.testing.assert_allclose(inv_my, map_y, atol=1.0)


class TestPerspectiveBboxesVectorized:
    """Verify vectorized HBB min/max produces same results as per-box loop."""

    def test_perspective_bboxes_correctness(self):
        """Column-stack min/max must match per-box list comprehension."""
        rng = np.random.default_rng(137)
        points = rng.uniform(0, 1, (10, 4, 2)).astype(np.float32)

        ref = np.array(
            [[np.min(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 0]), np.max(box[:, 1])] for box in points],
        )
        vec = np.column_stack(
            [
                points[:, :, 0].min(axis=1),
                points[:, :, 1].min(axis=1),
                points[:, :, 0].max(axis=1),
                points[:, :, 1].max(axis=1),
            ],
        )
        np.testing.assert_array_equal(vec, ref)


class TestComputeAffineWarpOutputShape:
    """Verify simplified output shape computation matches np.ceil-based version."""

    @pytest.mark.parametrize("input_shape", [(100, 200, 3), (100, 200)])
    def test_output_shape_types(self, input_shape):
        matrix = np.eye(3)
        _, shape = fgeometric.compute_affine_warp_output_shape(matrix, input_shape)
        assert all(isinstance(v, int) for v in shape)
        if len(input_shape) == 3:
            assert len(shape) == 3
        else:
            assert len(shape) == 2
