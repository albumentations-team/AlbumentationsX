"""Tests for keypoint label swapping functionality."""

import numpy as np
import pytest

import albumentations as A


class TestKeypointLabelSwapping:
    """Test keypoint label swapping functionality."""

    @pytest.mark.parametrize(
        "transform_class,expected_swaps",
        [
            (A.HorizontalFlip, {"left_eye": "right_eye", "right_eye": "left_eye"}),
            (A.VerticalFlip, {"top_lip": "bottom_lip", "bottom_lip": "top_lip"}),
            (A.Transpose, {"left_eye": "right_eye", "right_eye": "left_eye"}),
        ],
    )
    def test_basic_label_mapping_string_labels(self, transform_class, expected_swaps):
        """Test basic label mapping with string labels."""
        # Setup
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels = list(expected_swaps.keys())

        transform = A.Compose(
            [
                transform_class(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    transform_class.__name__: {
                        "keypoint_labels": expected_swaps,
                    },
                },
            ),
        )

        # Apply transform
        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        # Check that labels were swapped correctly
        expected_labels = [expected_swaps[label] for label in labels]
        assert result["keypoint_labels"] == expected_labels

        # Check that keypoints were transformed
        assert not np.array_equal(result["keypoints"], keypoints)

    @pytest.mark.parametrize(
        "transform_class,expected_swaps",
        [
            (A.HorizontalFlip, {0: 1, 1: 0}),  # left_eye <-> right_eye
            (A.VerticalFlip, {0: 1, 1: 0}),  # top_lip <-> bottom_lip
            (A.Transpose, {0: 1, 1: 0}),  # left_eye <-> right_eye
        ],
    )
    def test_basic_label_mapping_integer_labels(self, transform_class, expected_swaps):
        """Test basic label mapping with integer labels."""
        # Setup
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels = list(expected_swaps.keys())

        transform = A.Compose(
            [
                transform_class(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    transform_class.__name__: {
                        "keypoint_labels": expected_swaps,
                    },
                },
            ),
        )

        # Apply transform
        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        # Check that labels were swapped correctly
        expected_labels = [expected_swaps[label] for label in labels]
        assert result["keypoint_labels"] == expected_labels

    def test_no_label_mapping_preserves_labels(self):
        """Test that labels are preserved when no mapping is defined."""
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels = ["left_eye", "right_eye"]

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                # No label_mapping defined
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        # Labels should remain unchanged
        assert result["keypoint_labels"] == labels

    def test_partial_label_mapping(self):
        """Test that only mapped labels are swapped, others remain unchanged."""
        keypoints = np.array([[50, 25], [75, 30], [60, 35]], dtype=np.float32)
        labels = ["left_eye", "right_eye", "nose"]

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    "HorizontalFlip": {
                        "keypoint_labels": {
                            "left_eye": "right_eye",
                            "right_eye": "left_eye",
                            # 'nose' not mapped, should remain unchanged
                        },
                    },
                },
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        expected_labels = ["right_eye", "left_eye", "nose"]
        assert result["keypoint_labels"] == expected_labels

    def test_multiple_label_fields(self):
        """Test handling multiple label fields with selective mapping."""
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels1 = ["left_eye", "right_eye"]
        labels2 = ["visible", "visible"]

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels", "visibility"],
                label_mapping={
                    "HorizontalFlip": {
                        "keypoint_labels": {
                            "left_eye": "right_eye",
                            "right_eye": "left_eye",
                        },
                        # 'visibility' field not included - won't be transformed
                    },
                },
            ),
        )

        result = transform(
            image=np.ones((100, 100, 3), dtype=np.uint8),
            keypoints=keypoints,
            keypoint_labels=labels1,
            visibility=labels2,
        )

        expected_labels1 = ["right_eye", "left_eye"]
        expected_labels2 = ["visible", "visible"]  # Should remain unchanged

        assert result["keypoint_labels"] == expected_labels1
        assert result["visibility"] == expected_labels2

    def test_empty_keypoints_and_labels(self):
        """Test handling empty keypoints and labels."""
        keypoints = np.array([], dtype=np.float32).reshape(0, 2)
        labels = []

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    "HorizontalFlip": {
                        "keypoint_labels": {"left_eye": "right_eye"},
                    },
                },
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        assert result["keypoints"].shape == (0, 2)
        assert result["keypoint_labels"] == []

    def test_no_label_fields_no_error(self):
        """Test that transforms work normally when no label fields are specified."""
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                # No label_fields specified
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints)

        # Should work without error
        assert result["keypoints"].shape == (2, 2)

    def test_double_horizontal_flip_returns_original_labels(self):
        """Test that applying horizontal flip twice returns original labels."""
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels = ["left_eye", "right_eye"]

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    "HorizontalFlip": {
                        "keypoint_labels": {
                            "left_eye": "right_eye",
                            "right_eye": "left_eye",
                        },
                    },
                },
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        # Labels should return to original after double flip
        assert result["keypoint_labels"] == labels
        # Keypoints should also return to original (approximately)
        np.testing.assert_allclose(result["keypoints"], keypoints, atol=1e-6)

    def test_transform_without_label_mapping_support(self):
        """Test that transforms without label mapping support don't affect labels."""
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels = ["left_eye", "right_eye"]

        transform = A.Compose(
            [
                A.Rotate(limit=45, p=1.0),  # Rotation shouldn't affect semantic labels
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    "Rotate": {
                        "keypoint_labels": {"left_eye": "right_eye"},  # This mapping should be ignored
                    },
                },
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        # Labels should remain unchanged for non-parity-changing transforms
        assert result["keypoint_labels"] == labels

    @pytest.mark.parametrize(
        "transform_class",
        [
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
        ],
    )
    def test_parity_changing_transforms_support_label_swapping(self, transform_class):
        """Test that all parity-changing transforms support label swapping."""
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels = ["left_eye", "right_eye"]

        # This should not raise an error
        transform = A.Compose(
            [
                transform_class(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    transform_class.__name__: {
                        "keypoint_labels": {"left_eye": "right_eye", "right_eye": "left_eye"},
                    },
                },
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        expected_labels = ["right_eye", "left_eye"]
        assert result["keypoint_labels"] == expected_labels

    @pytest.mark.parametrize(
        "group_element,expected_transform",
        [
            ("h", "HorizontalFlip"),
            ("v", "VerticalFlip"),
            ("t", "Transpose"),
            ("hvt", "Transpose"),
            ("e", None),  # Identity - no swapping
            ("r90", None),  # Rotation - no swapping
            ("r180", None),  # Rotation - no swapping
            ("r270", None),  # Rotation - no swapping
        ],
    )
    def test_d4_label_mapping(self, group_element, expected_transform):
        """Test D4 transform label mapping for different group elements."""
        keypoints = np.array([[50, 25]], dtype=np.float32)
        labels = ["left_eye"]

        # Mock D4 transform to always return specific group element
        class MockD4(A.D4):
            def get_params(self):
                return {"group_element": group_element}

        transform = A.Compose(
            [
                MockD4(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={
                    "HorizontalFlip": {
                        "keypoint_labels": {"left_eye": "right_eye"},
                    },
                    "VerticalFlip": {
                        "keypoint_labels": {"left_eye": "top_eye"},
                    },
                    "Transpose": {
                        "keypoint_labels": {"left_eye": "transposed_eye"},
                    },
                },
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        if expected_transform == "HorizontalFlip":
            expected_labels = ["right_eye"]
        elif expected_transform == "VerticalFlip":
            expected_labels = ["top_eye"]
        elif expected_transform == "Transpose":
            expected_labels = ["transposed_eye"]
        else:
            # No swapping for identity and rotations
            expected_labels = ["left_eye"]

        assert result["keypoint_labels"] == expected_labels

    def test_user_defined_mapping_example(self):
        """Test example of user-defined mapping for their specific use case."""
        # User has their own keypoint format
        keypoints = np.array([[50, 25], [75, 30], [60, 35], [45, 40]], dtype=np.float32)
        labels = ["L_eye", "R_eye", "nose_tip", "L_ear"]  # User's custom labels

        # User defines their own mapping
        user_mapping = {
            "HorizontalFlip": {
                "keypoint_labels": {
                    "L_eye": "R_eye",
                    "R_eye": "L_eye",
                    "L_ear": "R_ear",
                    "R_ear": "L_ear",  # Add reverse mapping
                    "nose_tip": "nose_tip",  # unchanged
                },
            },
        }

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                label_mapping=user_mapping,
            ),
        )

        result = transform(image=np.ones((100, 100, 3), dtype=np.uint8), keypoints=keypoints, keypoint_labels=labels)

        expected_labels = ["R_eye", "L_eye", "nose_tip", "R_ear"]  # L_ear -> R_ear per mapping
        assert result["keypoint_labels"] == expected_labels

    def test_multiple_label_fields_selective_mapping(self):
        """Test that only specified label fields get mapped."""
        keypoints = np.array([[50, 25], [75, 30]], dtype=np.float32)
        labels1 = [0, 1]  # keypoint_labels
        labels2 = [2, 3]  # keypoint_types
        labels3 = ["visible", "visible"]  # visibility

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels", "keypoint_types", "visibility"],
                label_mapping={
                    "HorizontalFlip": {
                        "keypoint_labels": {0: 1, 1: 0},  # Swap labels 0<->1
                        "keypoint_types": {2: 3, 3: 2},  # Swap types 2<->3
                        # 'visibility' not included - won't be transformed
                    },
                },
            ),
        )

        result = transform(
            image=np.ones((100, 100, 3), dtype=np.uint8),
            keypoints=keypoints,
            keypoint_labels=labels1,
            keypoint_types=labels2,
            visibility=labels3,
        )

        expected_labels1 = [1, 0]  # Swapped
        expected_labels2 = [3, 2]  # Swapped
        expected_labels3 = ["visible", "visible"]  # Unchanged

        assert result["keypoint_labels"] == expected_labels1
        assert result["keypoint_types"] == expected_labels2
        assert result["visibility"] == expected_labels3
