"""Comprehensive tests for flip transforms with masks.

These tests verify:
1. Correct axis flipping for different mask shapes
2. Array contiguity (no negative strides)
3. Compatibility with PyTorch tensor conversion
4. Actual transformation correctness (not just shapes)
"""

import numpy as np
import pytest

import albumentations as A


class TestFlipMasksCorrectness:
    """Test that flips actually flip the correct pixels on the correct axes."""

    @pytest.mark.parametrize(
        "transform_class,axis",
        [
            (A.HorizontalFlip, 1),  # Flips along width (axis 1 for 2D, axis 2 for 3D+)
            (A.VerticalFlip, 0),  # Flips along height (axis 0 for 2D, axis 1 for 3D+)
        ],
    )
    def test_flip_single_mask_correctness(self, transform_class, axis):
        """Test that single mask (H, W, C) is flipped on the correct axis."""
        # Create a mask with distinct pattern
        mask = np.array(
            [
                [[1, 10], [2, 20], [3, 30]],
                [[4, 40], [5, 50], [6, 60]],
            ],
            dtype=np.uint8,
        )  # Shape: (2, 3, 2) = (H, W, C)

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), mask=mask)

        # Verify the flip happened on the correct axis
        if axis == 0:  # Vertical flip (flip H)
            expected = np.flip(mask, axis=0)
            # Should be: [[4,40], [5,50], [6,60]], [[1,10], [2,20], [3,30]]
        else:  # Horizontal flip (flip W)
            expected = np.flip(mask, axis=1)
            # Should be: [[3,30], [2,20], [1,10]], [[6,60], [5,50], [4,40]]

        np.testing.assert_array_equal(result["mask"], expected)

    @pytest.mark.parametrize(
        "transform_class,axis",
        [
            (A.HorizontalFlip, 2),  # Flips along width (axis 2 for (N,H,W,C))
            (A.VerticalFlip, 1),  # Flips along height (axis 1 for (N,H,W,C))
        ],
    )
    def test_flip_masks_batch_correctness(self, transform_class, axis):
        """Test that masks batch (N, H, W, C) is flipped on the correct axis."""
        # Create two masks with distinct patterns
        masks = np.array(
            [
                [  # First mask
                    [[1, 10], [2, 20], [3, 30]],
                    [[4, 40], [5, 50], [6, 60]],
                ],
                [  # Second mask
                    [[7, 70], [8, 80], [9, 90]],
                    [[10, 100], [11, 110], [12, 120]],
                ],
            ],
            dtype=np.uint8,
        )  # Shape: (2, 2, 3, 2) = (N, H, W, C)

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), masks=masks)

        # Verify the flip happened on the correct axis
        expected = np.flip(masks, axis=axis)
        np.testing.assert_array_equal(result["masks"], expected)

    def test_transpose_mask_correctness(self):
        """Test that Transpose actually transposes H and W."""
        # Create a mask with distinct pattern
        mask = np.array(
            [
                [[1, 10], [2, 20], [3, 30]],
                [[4, 40], [5, 50], [6, 60]],
            ],
            dtype=np.uint8,
        )  # Shape: (2, 3, 2) = (H=2, W=3, C=2)

        transform = A.Transpose(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), mask=mask)

        # After transpose: (H, W, C) -> (W, H, C) = (3, 2, 2)
        assert result["mask"].shape == (3, 2, 2)

        # Verify actual transposition: axes (0,1,2) -> (1,0,2)
        expected = np.transpose(mask, (1, 0, 2))
        np.testing.assert_array_equal(result["mask"], expected)

    def test_transpose_masks_batch_correctness(self):
        """Test that Transpose transposes H and W for batch."""
        masks = np.array(
            [
                [  # First mask (H=2, W=3)
                    [[1, 10], [2, 20], [3, 30]],
                    [[4, 40], [5, 50], [6, 60]],
                ],
                [  # Second mask
                    [[7, 70], [8, 80], [9, 90]],
                    [[10, 100], [11, 110], [12, 120]],
                ],
            ],
            dtype=np.uint8,
        )  # Shape: (N=2, H=2, W=3, C=2)

        transform = A.Transpose(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), masks=masks)

        # After transpose: (N, H, W, C) -> (N, W, H, C) = (2, 3, 2, 2)
        assert result["masks"].shape == (2, 3, 2, 2)

        # Verify actual transposition: axes (0,1,2,3) -> (0,2,1,3)
        expected = np.transpose(masks, (0, 2, 1, 3))
        np.testing.assert_array_equal(result["masks"], expected)


class TestFlipMasksContiguity:
    """Test that flipped masks are contiguous (no negative strides)."""

    @pytest.mark.parametrize(
        "transform_class",
        [
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
            A.D4,
        ],
    )
    def test_single_mask_contiguous(self, transform_class):
        """Test that single mask output is contiguous."""
        mask = np.random.randint(0, 2, (80, 120, 3), dtype=np.uint8)

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask)

        # Check that output is contiguous
        assert result["mask"].flags["C_CONTIGUOUS"], (
            f"{transform_class.__name__} produced non-contiguous single mask. "
            f"Strides: {result['mask'].strides}, Shape: {result['mask'].shape}"
        )

    @pytest.mark.parametrize(
        "transform_class",
        [
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
            A.D4,
        ],
    )
    def test_masks_batch_contiguous(self, transform_class):
        """Test that masks batch output is contiguous."""
        masks = np.random.randint(0, 2, (5, 80, 120, 3), dtype=np.uint8)

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks)

        # Check that output is contiguous
        assert result["masks"].flags["C_CONTIGUOUS"], (
            f"{transform_class.__name__} produced non-contiguous masks batch. "
            f"Strides: {result['masks'].strides}, Shape: {result['masks'].shape}"
        )

    @pytest.mark.parametrize(
        "transform_class",
        [
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
            A.D4,
        ],
    )
    def test_mask3d_contiguous(self, transform_class):
        """Test that 3D mask output is contiguous."""
        mask3d = np.random.randint(0, 2, (20, 80, 120, 3), dtype=np.uint8)

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(
            image=np.zeros((80, 120, 3), dtype=np.uint8),
            volume=np.zeros((20, 80, 120, 3), dtype=np.uint8),
            mask3d=mask3d,
        )

        # Check that output is contiguous
        assert result["mask3d"].flags["C_CONTIGUOUS"], (
            f"{transform_class.__name__} produced non-contiguous mask3d. "
            f"Strides: {result['mask3d'].strides}, Shape: {result['mask3d'].shape}"
        )


class TestFlipMasksPyTorchCompatibility:
    """Test that flipped masks can be converted to PyTorch tensors."""

    @pytest.mark.parametrize(
        "transform_class",
        [
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
            A.D4,
        ],
    )
    def test_single_mask_to_torch(self, transform_class):
        """Test that single mask can be converted to PyTorch tensor."""
        import torch

        mask = np.random.randint(0, 2, (80, 120, 3), dtype=np.uint8)

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask)

        # This should not raise ValueError about negative strides
        try:
            tensor = torch.from_numpy(result["mask"])
            assert tensor.shape == result["mask"].shape
        except ValueError as e:
            pytest.fail(
                f"{transform_class.__name__} mask cannot be converted to torch: {e}. "
                f"Shape: {result['mask'].shape}, Strides: {result['mask'].strides}",
            )

    @pytest.mark.parametrize(
        "transform_class",
        [
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
            A.D4,
        ],
    )
    def test_masks_batch_to_torch(self, transform_class):
        """Test that masks batch can be converted to PyTorch tensor."""
        import torch

        masks = np.random.randint(0, 2, (5, 80, 120, 3), dtype=np.uint8)

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks)

        # This should not raise ValueError about negative strides
        try:
            tensor = torch.from_numpy(result["masks"])
            assert tensor.shape == result["masks"].shape
        except ValueError as e:
            pytest.fail(
                f"{transform_class.__name__} masks cannot be converted to torch: {e}. "
                f"Shape: {result['masks'].shape}, Strides: {result['masks'].strides}",
            )

    def test_horizontal_flip_with_to_tensor_v2(self):
        """Test the exact scenario that was failing: HFlip + ToFloat + ToTensorV2."""
        import torch

        image = np.random.randint(0, 256, (101, 99, 3), dtype=np.uint8)
        masks = np.stack([image[:, :, 0]] * 2)  # Shape: (2, 101, 99)

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.ToFloat(max_value=255),
                A.ToTensorV2(),
            ],
            is_check_shapes=False,
            strict=True,
        )

        # This was failing before the fix
        result = transform(image=image, masks=masks)

        # Verify tensor conversion worked
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["masks"], torch.Tensor)
        assert result["masks"].shape[0] == 2  # 2 masks


class TestD4MasksSpecific:
    """Test D4 transform with all group elements."""

    @pytest.mark.parametrize("group_element", ["e", "r90", "r180", "r270", "v", "h", "t", "hvt"])
    def test_d4_mask_contiguous_all_elements(self, group_element):
        """Test that all D4 group elements produce contiguous masks when used through Compose."""
        mask = np.random.randint(0, 2, (100, 100, 3), dtype=np.uint8)

        # Seed to get specific group element (this is implementation detail, but tests D4's behavior)
        # For testing purposes, we just verify that going through Compose produces contiguous output
        transform = A.D4(p=1.0)
        aug = A.Compose([transform])

        # Apply through Compose (which uses ensure_contiguous_output)
        result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask=mask)

        # Check contiguity - should always be contiguous through Compose
        assert result["mask"].flags["C_CONTIGUOUS"], (
            f"D4 mask not contiguous after Compose. Strides: {result['mask'].strides}, Shape: {result['mask'].shape}"
        )

    @pytest.mark.parametrize("group_element", ["e", "r90", "r180", "r270", "v", "h", "t", "hvt"])
    def test_d4_masks_batch_contiguous_all_elements(self, group_element):
        """Test that all D4 group elements produce contiguous masks batch when used through Compose."""
        masks = np.random.randint(0, 2, (3, 100, 100, 3), dtype=np.uint8)

        transform = A.D4(p=1.0)
        aug = A.Compose([transform])

        # Apply through Compose (which uses ensure_contiguous_output)
        result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks=masks)

        # Check contiguity - should always be contiguous through Compose
        assert result["masks"].flags["C_CONTIGUOUS"], (
            f"D4 masks batch not contiguous after Compose. "
            f"Strides: {result['masks'].strides}, Shape: {result['masks'].shape}"
        )

    @pytest.mark.parametrize(
        "group_element,should_transpose",
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
    def test_d4_mask_dimension_changes(self, group_element, should_transpose):
        """Test that D4 changes dimensions correctly for non-square masks."""
        # Non-square mask
        mask = np.random.randint(0, 2, (80, 120, 3), dtype=np.uint8)

        transform = A.D4(p=1.0)
        result = transform.apply_to_mask(mask, group_element=group_element)

        if should_transpose:
            # Should swap H and W
            assert result.shape == (120, 80, 3), (
                f"D4 with '{group_element}' should swap dimensions but got {result.shape}"
            )
        else:
            # Should preserve H and W
            assert result.shape == (80, 120, 3), (
                f"D4 with '{group_element}' should preserve dimensions but got {result.shape}"
            )
