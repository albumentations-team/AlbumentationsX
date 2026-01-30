"""Tests for apply_to_mask(s) methods in crop and flip transforms."""

import numpy as np
import pytest

import albumentations as A


# Test crops with single mask (empty and non-empty)
@pytest.mark.parametrize(
    "transform_class,init_params,expected_shape",
    [
        (A.RandomCrop, {"height": 50, "width": 60}, (50, 60)),
        (A.CenterCrop, {"height": 40, "width": 70}, (40, 70)),
        (A.Crop, {"x_min": 10, "y_min": 15, "x_max": 70, "y_max": 55}, (40, 60)),
    ],
)
@pytest.mark.parametrize("mask_shape", [(100, 120), (100, 120, 3)])
def test_crop_apply_to_mask_single(transform_class, init_params, expected_shape, mask_shape):
    """Test that apply_to_mask works correctly for crops with non-square dimensions."""
    transform = transform_class(**init_params, p=1.0)
    mask = np.random.randint(0, 2, mask_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 120, 3), dtype=np.uint8), mask=mask)

    # Check that mask was cropped
    assert result["mask"].shape[:2] == expected_shape
    if len(mask_shape) == 3:
        assert result["mask"].shape[2] == mask_shape[2]


def test_crop_apply_to_mask_empty_channels():
    """Test that apply_to_mask handles empty channel dimension correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create mask with 0 channels (empty)
    mask = np.empty((100, 100, 0), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask=mask)

    # Check that mask was cropped with correct shape
    assert result["mask"].shape == (50, 50, 0)
    assert result["mask"].dtype == np.uint8


# Test crops with masks batch (empty and non-empty) - using non-square dimensions
@pytest.mark.parametrize(
    "transform_class,init_params,expected_shape",
    [
        (A.RandomCrop, {"height": 50, "width": 60}, (50, 60)),
        (A.CenterCrop, {"height": 40, "width": 70}, (40, 70)),
        (A.Crop, {"x_min": 10, "y_min": 15, "x_max": 70, "y_max": 55}, (40, 60)),
    ],
)
@pytest.mark.parametrize("num_masks", [1, 3, 5])
@pytest.mark.parametrize("channels", [None, 1, 3])
def test_crop_apply_to_masks_batch(transform_class, init_params, expected_shape, num_masks, channels):
    """Test that apply_to_masks works correctly for batch processing with non-square dimensions."""
    transform = transform_class(**init_params, p=1.0)

    if channels is None:
        masks_shape = (num_masks, 100, 120)
    else:
        masks_shape = (num_masks, 100, 120, channels)

    masks = np.random.randint(0, 2, masks_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 120, 3), dtype=np.uint8), masks=masks)

    # Check that all masks were cropped
    assert result["masks"].shape[0] == num_masks
    assert result["masks"].shape[1:3] == expected_shape
    if channels is not None:
        assert result["masks"].shape[3] == channels


def test_crop_apply_to_masks_empty_batch():
    """Test that apply_to_masks handles empty batch correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create empty batch of masks
    masks = np.empty((0, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks=masks)

    # Check that empty batch returns correct cropped dimensions
    assert result["masks"].shape == (0, 50, 50)
    assert result["masks"].dtype == np.uint8


def test_crop_apply_to_masks_empty_batch_with_channels():
    """Test that apply_to_masks handles empty batch with channels correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create empty batch of masks with channels
    masks = np.empty((0, 100, 100, 3), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks=masks)

    # Check that empty batch returns correct cropped dimensions
    assert result["masks"].shape == (0, 50, 50, 3)
    assert result["masks"].dtype == np.uint8


# Test flips with single mask (empty and non-empty) - using non-square images
@pytest.mark.parametrize("transform_class", [A.HorizontalFlip, A.VerticalFlip, A.Transpose])
@pytest.mark.parametrize("mask_shape", [(80, 120), (80, 120, 3)])
def test_flip_apply_to_mask_single(transform_class, mask_shape):
    """Test that apply_to_mask works correctly for flips with non-square images."""
    transform = transform_class(p=1.0)
    mask = np.random.randint(0, 2, mask_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask)

    # Check that mask shape is handled correctly
    if transform_class == A.Transpose:
        # Transpose swaps H and W: (80, 120) -> (120, 80)
        assert result["mask"].shape[0] == mask_shape[1]
        assert result["mask"].shape[1] == mask_shape[0]
    else:
        # Other flips preserve dimensions
        assert result["mask"].shape[:2] == mask_shape[:2]

    if len(mask_shape) == 3:
        assert result["mask"].shape[2] == mask_shape[2]


def test_flip_apply_to_mask_empty_channels():
    """Test that HorizontalFlip and VerticalFlip handle empty channel dimension correctly."""
    for transform_class in [A.HorizontalFlip, A.VerticalFlip]:
        transform = transform_class(p=1.0)
        # Create mask with 0 channels (empty)
        mask = np.empty((80, 120, 0), dtype=np.uint8)

        # Apply the transform through Compose
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask)

        # Check correct shape (flips preserve spatial dimensions)
        assert result["mask"].shape == (80, 120, 0)
        assert result["mask"].dtype == np.uint8


def test_transpose_empty_mask_swaps_dimensions():
    """Test that Transpose swaps dimensions correctly for empty single mask."""
    transform = A.Transpose(p=1.0)
    aug = A.Compose([transform])

    # Test with 3D mask (H, W, 0) - 0 channels
    mask_empty_channels = np.empty((80, 120, 0), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask_empty_channels)
    # Transpose should swap H and W: (80, 120, 0) -> (120, 80, 0)
    assert result["mask"].shape == (120, 80, 0)
    assert result["mask"].dtype == np.uint8


# Test flips with masks batch (empty and non-empty) - using non-square images
@pytest.mark.parametrize("transform_class", [A.HorizontalFlip, A.VerticalFlip])
@pytest.mark.parametrize("num_masks", [1, 3, 5])
@pytest.mark.parametrize("channels", [None, 1, 3])
def test_flip_apply_to_masks_batch(transform_class, num_masks, channels):
    """Test that apply_to_masks works correctly for batch processing with non-square images."""
    transform = transform_class(p=1.0)

    if channels is None:
        masks_shape = (num_masks, 80, 120)
    else:
        masks_shape = (num_masks, 80, 120, channels)

    masks = np.random.randint(0, 2, masks_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks)

    # Check that all masks were processed
    assert result["masks"].shape[0] == num_masks

    # Check spatial dimensions (non-transpose flips preserve dimensions)
    assert result["masks"].shape[1:3] == (80, 120)

    if channels is not None:
        assert result["masks"].shape[3] == channels


def test_flip_apply_to_masks_empty_batch():
    """Test that HorizontalFlip and VerticalFlip handle empty batch correctly."""
    for transform_class in [A.HorizontalFlip, A.VerticalFlip]:
        transform = transform_class(p=1.0)
        # Create empty batch of masks
        masks = np.empty((0, 80, 120), dtype=np.uint8)

        # Apply the transform through Compose
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks)

        # Check that empty batch preserves dimensions
        assert result["masks"].shape == (0, 80, 120)
        assert result["masks"].dtype == np.uint8


def test_transpose_empty_masks_swaps_dimensions():
    """Test that Transpose swaps dimensions correctly for empty masks."""
    transform = A.Transpose(p=1.0)

    # Test with 2D masks (N, H, W)
    masks_2d = np.empty((0, 80, 120), dtype=np.uint8)
    aug = A.Compose([transform])
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks_2d)
    # Transpose should swap H and W: (0, 80, 120) -> (0, 120, 80)
    assert result["masks"].shape == (0, 120, 80)
    assert result["masks"].dtype == np.uint8

    # Test with 3D masks (N, H, W, C)
    masks_3d = np.empty((0, 80, 120, 3), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks_3d)
    # Transpose should swap H and W: (0, 80, 120, 3) -> (0, 120, 80, 3)
    assert result["masks"].shape == (0, 120, 80, 3)
    assert result["masks"].dtype == np.uint8


@pytest.mark.parametrize(
    "group_element,expected_swap",
    [
        ("e", False),  # identity - no swap
        ("r90", True),  # rotation 90 - swaps dimensions for non-square images
        ("r180", False),  # rotation 180 - no swap
        ("r270", True),  # rotation 270 - swaps dimensions for non-square images
        ("v", False),  # vertical flip - no swap
        ("h", False),  # horizontal flip - no swap
        ("t", True),  # transpose - swaps dimensions
        ("hvt", True),  # anti-diagonal transpose - swaps dimensions
    ],
)
def test_d4_empty_mask_dimension_handling(group_element, expected_swap):
    """Test that D4 correctly handles dimension swapping for empty masks based on group element."""
    # Create a D4 transform but we'll call apply_to_mask directly with specific group element
    transform = A.D4(p=1.0)

    # Test with single empty mask (H, W, C) - using non-square image
    mask = np.empty((80, 120, 3), dtype=np.uint8)
    result_mask = transform.apply_to_mask(mask, group_element=group_element)

    if expected_swap:
        # Group elements that swap dimensions: r90, r270, t, hvt
        # Should swap H and W: (80, 120, 3) -> (120, 80, 3)
        assert result_mask.shape == (120, 80, 3), f"Failed for group_element={group_element}"
    else:
        # Other group elements preserve dimensions: e, r180, v, h
        assert result_mask.shape == (80, 120, 3), f"Failed for group_element={group_element}"
    assert result_mask.dtype == np.uint8


@pytest.mark.parametrize(
    "group_element,expected_swap",
    [
        ("e", False),
        ("r90", True),
        ("r180", False),
        ("r270", True),
        ("v", False),
        ("h", False),
        ("t", True),
        ("hvt", True),
    ],
)
def test_d4_empty_masks_batch_dimension_handling(group_element, expected_swap):
    """Test that D4 correctly handles dimension swapping for empty mask batches."""
    transform = A.D4(p=1.0)

    # Test with batch of empty masks (N, H, W, C)
    masks = np.empty((0, 80, 120, 3), dtype=np.uint8)
    result_masks = transform.apply_to_masks(masks, group_element=group_element)

    if expected_swap:
        # Group elements that swap dimensions: r90, r270, t, hvt
        # Should swap H and W: (0, 80, 120, 3) -> (0, 120, 80, 3)
        assert result_masks.shape == (0, 120, 80, 3), f"Failed for group_element={group_element}"
    else:
        # Other group elements preserve dimensions: e, r180, v, h
        assert result_masks.shape == (0, 80, 120, 3), f"Failed for group_element={group_element}"
    assert result_masks.dtype == np.uint8


@pytest.mark.parametrize(
    "group_element,expected_swap",
    [
        ("e", False),
        ("r90", True),
        ("r270", True),
        ("t", True),
        ("hvt", True),
    ],
)
def test_d4_empty_mask3d_dimension_handling(group_element, expected_swap):
    """Test that D4 correctly handles dimension swapping for empty mask3d."""
    transform = A.D4(p=1.0)

    # Test with empty mask3d (D, H, W, C)
    mask3d = np.empty((10, 80, 120, 3), dtype=np.uint8)
    result_mask3d = transform.apply_to_mask3d(mask3d, group_element=group_element)

    if expected_swap:
        # Group elements that swap dimensions: r90, r270, t, hvt
        # Should swap H and W: (10, 80, 120, 3) -> (10, 120, 80, 3)
        assert result_mask3d.shape == (10, 120, 80, 3), f"Failed for group_element={group_element}"
    else:
        # Other group elements preserve dimensions: e, r180, v, h
        assert result_mask3d.shape == (10, 80, 120, 3), f"Failed for group_element={group_element}"
    assert result_mask3d.dtype == np.uint8


@pytest.mark.parametrize(
    "group_element,expected_swap",
    [
        ("e", False),
        ("r90", True),
        ("r270", True),
        ("t", True),
        ("hvt", True),
    ],
)
def test_d4_empty_masks3d_batch_dimension_handling(group_element, expected_swap):
    """Test that D4 correctly handles dimension swapping for empty masks3d batches."""
    transform = A.D4(p=1.0)

    # Test with batch of empty masks3d (N, D, H, W, C)
    masks3d = np.empty((0, 10, 80, 120, 3), dtype=np.uint8)
    result_masks3d = transform.apply_to_masks3d(masks3d, group_element=group_element)

    if expected_swap:
        # Group elements that swap dimensions: r90, r270, t, hvt
        # Should swap H and W: (0, 10, 80, 120, 3) -> (0, 10, 120, 80, 3)
        assert result_masks3d.shape == (0, 10, 120, 80, 3), f"Failed for group_element={group_element}"
    else:
        # Other group elements preserve dimensions: e, r180, v, h
        assert result_masks3d.shape == (0, 10, 80, 120, 3), f"Failed for group_element={group_element}"
    assert result_masks3d.dtype == np.uint8


# Test crops with mask3d (empty and non-empty)
@pytest.mark.parametrize(
    "transform_class,init_params",
    [
        (A.RandomCrop, {"height": 50, "width": 50}),
        (A.CenterCrop, {"height": 50, "width": 50}),
        (A.Crop, {"x_min": 10, "y_min": 10, "x_max": 60, "y_max": 60}),
    ],
)
def test_crop_apply_to_mask3d(transform_class, init_params):
    """Test that apply_to_mask3d works correctly for crops."""
    transform = transform_class(**init_params, p=1.0)
    # mask3d has shape (D, H, W) or (D, H, W, C)
    mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask3d=mask3d)

    # Check that mask3d was cropped (depth preserved, H and W cropped)
    assert result["mask3d"].shape == (10, 50, 50)


def test_crop_apply_to_mask3d_empty():
    """Test that apply_to_mask3d handles empty mask correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create empty mask3d (0 depth)
    mask3d = np.empty((0, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask3d=mask3d)

    # Check correct shape
    assert result["mask3d"].shape == (0, 50, 50)
    assert result["mask3d"].dtype == np.uint8


# Test crops with masks3d batch (empty and non-empty)
@pytest.mark.parametrize(
    "transform_class,init_params",
    [
        (A.RandomCrop, {"height": 50, "width": 50}),
        (A.CenterCrop, {"height": 50, "width": 50}),
        (A.Crop, {"x_min": 10, "y_min": 10, "x_max": 60, "y_max": 60}),
    ],
)
@pytest.mark.parametrize("num_masks3d", [1, 3])
def test_crop_apply_to_masks3d_batch(transform_class, init_params, num_masks3d):
    """Test that apply_to_masks3d works correctly for batch processing."""
    transform = transform_class(**init_params, p=1.0)
    # masks3d has shape (N, D, H, W)
    masks3d = np.random.randint(0, 2, (num_masks3d, 10, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks3d=masks3d)

    # Check that all masks3d were cropped
    assert result["masks3d"].shape == (num_masks3d, 10, 50, 50)


def test_crop_apply_to_masks3d_empty_batch():
    """Test that apply_to_masks3d handles empty batch correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create empty batch of masks3d
    masks3d = np.empty((0, 10, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks3d=masks3d)

    # Check correct shape
    assert result["masks3d"].shape == (0, 10, 50, 50)
    assert result["masks3d"].dtype == np.uint8


# Test flips with mask3d and masks3d
@pytest.mark.parametrize("transform_class", [A.HorizontalFlip, A.VerticalFlip, A.Transpose, A.D4])
def test_flip_apply_to_mask3d(transform_class):
    """Test that apply_to_mask3d works correctly for flips."""
    transform = transform_class(p=1.0)
    # mask3d has shape (D, H, W)
    mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask3d=mask3d)

    # Check shape (depth preserved)
    if transform_class == A.Transpose:
        assert result["mask3d"].shape == (10, 100, 100)  # Square, so same after transpose
    else:
        assert result["mask3d"].shape == (10, 100, 100)


def test_flip_apply_to_mask3d_empty():
    """Test that HorizontalFlip and VerticalFlip handle empty mask3d correctly."""
    for transform_class in [A.HorizontalFlip, A.VerticalFlip]:
        transform = transform_class(p=1.0)
        # Create empty mask3d (0 depth)
        mask3d = np.empty((0, 80, 120), dtype=np.uint8)

        # Apply the transform through Compose
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask3d=mask3d)

        # Check correct shape
        assert result["mask3d"].shape == (0, 80, 120)
        assert result["mask3d"].dtype == np.uint8


def test_transpose_empty_mask3d_swaps_dimensions():
    """Test that Transpose swaps dimensions correctly for empty mask3d."""
    transform = A.Transpose(p=1.0)
    aug = A.Compose([transform])

    # Test with 3D mask3d (D, H, W) where D=0
    mask3d_empty_depth = np.empty((0, 80, 120), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask3d=mask3d_empty_depth)
    # Transpose should swap H and W: (0, 80, 120) -> (0, 120, 80)
    assert result["mask3d"].shape == (0, 120, 80)
    assert result["mask3d"].dtype == np.uint8

    # Test with 4D mask3d (D, H, W, C) where D=0
    mask3d_empty_depth_4d = np.empty((0, 80, 120, 3), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask3d=mask3d_empty_depth_4d)
    # Transpose should swap H and W: (0, 80, 120, 3) -> (0, 120, 80, 3)
    assert result["mask3d"].shape == (0, 120, 80, 3)
    assert result["mask3d"].dtype == np.uint8


@pytest.mark.parametrize("transform_class", [A.HorizontalFlip, A.VerticalFlip, A.Transpose])
@pytest.mark.parametrize("num_masks3d", [1, 3])
def test_flip_apply_to_masks3d_batch(transform_class, num_masks3d):
    """Test that apply_to_masks3d works correctly for batch processing."""
    transform = transform_class(p=1.0)
    # masks3d has shape (N, D, H, W)
    masks3d = np.random.randint(0, 2, (num_masks3d, 10, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks3d=masks3d)

    # Check shape (square image, so dimensions preserved even for transpose)
    assert result["masks3d"].shape == (num_masks3d, 10, 100, 100)


def test_flip_apply_to_masks3d_empty_batch():
    """Test that HorizontalFlip and VerticalFlip handle empty masks3d batch correctly."""
    for transform_class in [A.HorizontalFlip, A.VerticalFlip]:
        transform = transform_class(p=1.0)
        # Create empty batch of masks3d
        masks3d = np.empty((0, 10, 80, 120), dtype=np.uint8)

        # Apply the transform through Compose
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks3d=masks3d)

        # Check correct shape
        assert result["masks3d"].shape == (0, 10, 80, 120)
        assert result["masks3d"].dtype == np.uint8


def test_transpose_empty_masks3d_swaps_dimensions():
    """Test that Transpose swaps dimensions correctly for empty masks3d batch."""
    transform = A.Transpose(p=1.0)
    aug = A.Compose([transform])

    # Test with 4D masks3d (N, D, H, W) where N=0
    masks3d_empty_batch = np.empty((0, 10, 80, 120), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks3d=masks3d_empty_batch)
    # Transpose should swap H and W: (0, 10, 80, 120) -> (0, 10, 120, 80)
    assert result["masks3d"].shape == (0, 10, 120, 80)
    assert result["masks3d"].dtype == np.uint8

    # Test with 5D masks3d (N, D, H, W, C) where N=0
    masks3d_empty_batch_5d = np.empty((0, 10, 80, 120, 3), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks3d=masks3d_empty_batch_5d)
    # Transpose should swap H and W: (0, 10, 80, 120, 3) -> (0, 10, 120, 80, 3)
    assert result["masks3d"].shape == (0, 10, 120, 80, 3)
    assert result["masks3d"].dtype == np.uint8


# Test that batch processing is actually faster than loop (integration test)
@pytest.mark.parametrize("transform_class,init_params", [(A.CenterCrop, {"height": 50, "width": 50})])
def test_crop_masks_batch_vs_loop(transform_class, init_params):
    """Test that batch processing gives same results as loop processing."""
    transform = transform_class(**init_params, p=1.0)
    num_masks = 5
    masks = np.random.randint(0, 2, (num_masks, 100, 100), dtype=np.uint8)

    # Apply using batch method
    aug = A.Compose([transform])
    result_batch = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks=masks)

    # Apply using loop (simulating old behavior)
    result_loop_masks = []
    for i in range(num_masks):
        mask_result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask=masks[i])
        result_loop_masks.append(mask_result["mask"])
    result_loop = np.stack(result_loop_masks)

    # Check that results are identical
    np.testing.assert_array_equal(result_batch["masks"], result_loop)
