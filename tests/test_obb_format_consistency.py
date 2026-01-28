"""Tests to verify OBB format preservation and OBB/HBB consistency.

This test file checks:
1. Input format == output format (YOLO in => YOLO out, etc.)
2. OBB with angle=0 should give same bbox coords as HBB for flips and D4
"""

import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize(
    "bbox_format,input_bbox",
    [
        pytest.param("pascal_voc", [10, 20, 50, 60, 0.0], id="pascal_voc"),
        pytest.param("coco", [10, 20, 40, 40, 0.0], id="coco"),
        pytest.param("yolo", [0.3, 0.4, 0.4, 0.4, 0.0], id="yolo"),
        pytest.param("albumentations", [0.1, 0.2, 0.5, 0.6, 0.0], id="albumentations"),
    ],
)
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(A.HorizontalFlip(p=1.0), id="HFlip"),
        pytest.param(A.VerticalFlip(p=1.0), id="VFlip"),
        pytest.param(A.Rotate(limit=(45, 45), p=1.0), id="Rotate"),
        pytest.param(A.RandomRotate90(p=1.0), id="Rotate90"),
    ],
)
@pytest.mark.obb
def test_obb_format_preservation(bbox_format: str, input_bbox: list, transform: A.BasicTransform) -> None:
    """Test that input format == output format for OBB."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    aug = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format=bbox_format, bbox_type="obb")
    )

    np.random.seed(137)  # For consistent RandomRotate90
    result = aug(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Verify format consistency
    if bbox_format in ["pascal_voc", "coco"]:
        # Pixel coordinates - should have values that could be > 1
        assert isinstance(output_bbox, (list, tuple, np.ndarray))
        # For pascal_voc, at least some coords should be > 1 (not normalized)
        if bbox_format == "pascal_voc":
            # Original had coords [10, 20, 50, 60], so output should also be in pixel space
            assert any(abs(v) > 1 for v in output_bbox[:4]), \
                f"Expected pixel coords for {bbox_format}, got {output_bbox}"
    else:
        # Normalized coordinates [0, 1]
        assert all(0 <= v <= 1.01 for v in output_bbox[:4]), \
            f"Expected normalized coords for {bbox_format}, got {output_bbox}"


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(A.HorizontalFlip(p=1.0), id="HFlip"),
        pytest.param(A.VerticalFlip(p=1.0), id="VFlip"),
    ],
)
@pytest.mark.obb
def test_obb_matches_hbb_for_axis_aligned_flips(
    transform: A.BasicTransform,
) -> None:
    """Test that OBB (angle=0) gives same result as HBB for flips."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox_coords = [0.2, 0.3, 0.6, 0.7]  # albumentations format

    # HBB
    aug_hbb = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="hbb")
    )
    result_hbb = aug_hbb(image=image, bboxes=[bbox_coords])
    bbox_hbb = result_hbb["bboxes"][0]

    # OBB with angle=0
    aug_obb = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )
    result_obb = aug_obb(image=image, bboxes=[bbox_coords + [0.0]])
    bbox_obb = result_obb["bboxes"][0]

    # Check that OBB coords match HBB coords (ignoring the angle)
    np.testing.assert_allclose(
        bbox_obb[:4],
        bbox_hbb,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"OBB coords don't match HBB for {transform.__class__.__name__}"
    )


@pytest.mark.parametrize(
    "group_member",
    ["e", "r90", "r180", "r270", "v", "h", "t", "hvt"],
)
@pytest.mark.obb
def test_obb_matches_hbb_for_axis_aligned_d4(group_member: str) -> None:
    """Test that OBB (angle=0) gives same result as HBB for all D4 operations."""
    from albumentations.augmentations.geometric.functional import bboxes_d4

    bbox_coords = [0.2, 0.3, 0.6, 0.7]  # albumentations format

    hbb_arr = np.array([bbox_coords], dtype=np.float32)
    obb_arr = np.array([bbox_coords + [0.0]], dtype=np.float32)

    hbb_result = bboxes_d4(hbb_arr, group_member, bbox_type="hbb")
    obb_result = bboxes_d4(obb_arr, group_member, bbox_type="obb")

    np.testing.assert_allclose(
        obb_result[0, :4],
        hbb_result[0],
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"OBB coords don't match HBB for D4 operation '{group_member}'"
    )


@pytest.mark.parametrize(
    "bbox_format,input_bbox",
    [
        pytest.param("pascal_voc", [10, 20, 50, 60, 30.0], id="pascal_voc"),
        pytest.param("coco", [10, 20, 40, 40, 30.0], id="coco"),
        pytest.param("yolo", [0.3, 0.4, 0.4, 0.4, 30.0], id="yolo"),
        pytest.param("albumentations", [0.1, 0.2, 0.5, 0.6, 30.0], id="albumentations"),
    ],
)
@pytest.mark.obb
def test_obb_format_roundtrip_with_angle(bbox_format: str, input_bbox: list) -> None:
    """Test that format is preserved through transform and back with non-zero angle."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Apply identity transform
    aug = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format=bbox_format, bbox_type="obb")
    )

    result = aug(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Should be very close to input
    np.testing.assert_allclose(
        output_bbox,
        input_bbox,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Format {bbox_format} not preserved through identity transform"
    )


@pytest.mark.parametrize(
    "bbox_format,input_bbox",
    [
        pytest.param("pascal_voc", [10, 20, 50, 60, 0.0], id="pascal_voc"),
        pytest.param("coco", [10, 20, 40, 40, 0.0], id="coco"),
        pytest.param("yolo", [0.3, 0.4, 0.4, 0.4, 0.0], id="yolo"),
        pytest.param("albumentations", [0.1, 0.2, 0.5, 0.6, 0.0], id="albumentations"),
    ],
)
@pytest.mark.obb
def test_obb_format_preserved_through_pipeline(bbox_format: str, input_bbox: list) -> None:
    """Test that format is preserved through multi-transform pipeline."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    aug = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Rotate(limit=(30, 30), p=1.0),
        ],
        bbox_params=A.BboxParams(format=bbox_format, bbox_type="obb")
    )

    result = aug(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Verify format consistency (same checks as test_obb_format_preservation)
    if bbox_format in ["pascal_voc", "coco"]:
        if bbox_format == "pascal_voc":
            assert any(abs(v) > 1 for v in output_bbox[:4]), \
                f"Expected pixel coords for {bbox_format}, got {output_bbox}"
    else:
        # Allow slightly over 1.0 due to rotation artifacts near borders
        assert all(-0.01 <= v <= 1.01 for v in output_bbox[:4]), \
            f"Expected normalized coords for {bbox_format}, got {output_bbox}"


@pytest.mark.obb
def test_obb_support_declaration():
    """Test that transforms correctly declare OBB support."""
    # Transforms that should support OBB
    assert "obb" in A.HorizontalFlip()._supported_bbox_types
    assert "obb" in A.VerticalFlip()._supported_bbox_types
    assert "obb" in A.Transpose()._supported_bbox_types
    assert "obb" in A.D4()._supported_bbox_types
    assert "obb" in A.Rotate()._supported_bbox_types
    assert "obb" in A.RandomRotate90()._supported_bbox_types
    assert "obb" in A.Affine()._supported_bbox_types
    assert "obb" in A.Perspective()._supported_bbox_types
    assert "obb" in A.ShiftScaleRotate()._supported_bbox_types
    assert "obb" in A.NoOp()._supported_bbox_types


@pytest.mark.obb
def test_compose_validates_obb_support_at_init():
    """Test that Compose rejects unsupported transforms with OBB at __init__ time."""
    # Create a custom transform that doesn't support OBB
    class UnsupportedTransform(A.DualTransform):
        _supported_bbox_types = frozenset({"hbb"})  # Explicitly only HBB

        def apply(self, img, **params):
            return img

        def apply_to_bboxes(self, bboxes, **params):
            return bboxes

    # This should fail at Compose.__init__, not at __call__
    with pytest.raises(ValueError, match="do not support OBB"):
        A.Compose(
            [UnsupportedTransform()],
            bbox_params=A.BboxParams(format="pascal_voc", bbox_type="obb")
        )


@pytest.mark.obb
def test_compose_allows_imageonly_with_obb():
    """ImageOnly transforms should not block OBB usage."""
    # Should NOT raise - ImageOnly transforms are skipped in validation
    compose = A.Compose(
        [A.Normalize(), A.HorizontalFlip()],
        bbox_params=A.BboxParams(format="pascal_voc", bbox_type="obb")
    )
    assert compose is not None

    # Verify it works at runtime
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = [[10, 20, 50, 60, 0.0]]
    result = compose(image=image, bboxes=bboxes)
    assert "bboxes" in result


@pytest.mark.obb
def test_no_runtime_obb_errors():
    """Verify that OBB errors happen at Compose init, not at runtime."""
    # This tests the principle: validation at init, not at call
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = [[0.1, 0.2, 0.5, 0.6, 0.0]]  # OBB format

    # If this passes __init__, it should work at __call__ time
    compose = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    # Should not raise at runtime
    result = compose(image=image, bboxes=bboxes)
    assert "bboxes" in result
    assert len(result["bboxes"]) == 1


@pytest.mark.obb
def test_nested_compose_obb_validation():
    """Test that nested Compose also validates OBB support."""
    class UnsupportedTransform(A.DualTransform):
        _supported_bbox_types = frozenset({"hbb"})

        def apply(self, img, **params):
            return img

        def apply_to_bboxes(self, bboxes, **params):
            return bboxes

    # Single-level nested compose should fail validation
    with pytest.raises(ValueError, match="do not support OBB"):
        A.Compose(
            [
                A.HorizontalFlip(),
                A.OneOf([
                    UnsupportedTransform(),
                    A.VerticalFlip()
                ])
            ],
            bbox_params=A.BboxParams(format="pascal_voc", bbox_type="obb")
        )


@pytest.mark.obb
def test_deeply_nested_compose_obb_validation():
    """Test that deeply nested unsupported transforms are caught at Compose init."""
    class UnsupportedTransform(A.DualTransform):
        _supported_bbox_types = frozenset({"hbb"})

        def apply(self, img, **params):
            return img

        def apply_to_bboxes(self, bboxes, **params):
            return bboxes

    # Multi-level nested compose (OneOf > SomeOf > Unsupported) should fail validation
    with pytest.raises(ValueError, match="do not support OBB"):
        A.Compose(
            [
                A.HorizontalFlip(),
                A.OneOf([
                    A.SomeOf([
                        UnsupportedTransform(),
                        A.VerticalFlip()
                    ], n=1)
                ])
            ],
            bbox_params=A.BboxParams(format="pascal_voc", bbox_type="obb")
        )


@pytest.mark.obb
def test_obb_affine_filters_out_of_bounds_boxes() -> None:
    """Test that OBB boxes going out of bounds during affine transforms are properly filtered.

    This is a regression test for the bug where validate_bboxes was called with pixel dimensions
    on normalized coordinates, causing boxes outside [0, 1] to not be filtered correctly.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create a bbox that will be pushed far out of bounds by rotation
    # When rotated 90 degrees with shift, this should go completely outside the image
    bboxes = [[0.8, 0.8, 0.95, 0.95, 0.0]]  # OBB near bottom-right corner

    transform = A.Compose([
        A.Affine(
            rotate=90,
            translate_percent={"x": 0.5, "y": 0.5},  # Large shift to push bbox out
            p=1.0
        )
    ], bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"))

    result = transform(image=image, bboxes=bboxes)

    # The bbox should be filtered out since it's outside the image bounds
    # Before the fix, validate_bboxes was comparing normalized coords against pixel dimensions,
    # so boxes with x_max > 1.0 weren't being filtered (since 1.0 < 100)
    assert len(result["bboxes"]) == 0, "Out of bounds OBB should have been filtered"


@pytest.mark.obb
def test_obb_affine_preserves_in_bounds_boxes() -> None:
    """Test that OBB boxes staying in bounds during affine transforms are preserved."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create a bbox in the center that will stay in bounds
    bboxes = [[0.4, 0.4, 0.6, 0.6, 0.0]]  # OBB in center

    transform = A.Compose([
        A.Affine(rotate=45, p=1.0)
    ], bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"))

    result = transform(image=image, bboxes=bboxes)

    # The bbox should be preserved since it stays in bounds
    assert len(result["bboxes"]) == 1, "In-bounds OBB should have been preserved"

    # Verify the bbox is still within [0, 1] range
    bbox = result["bboxes"][0]
    assert 0 <= bbox[0] <= 1, f"x_min {bbox[0]} out of bounds"
    assert 0 <= bbox[1] <= 1, f"y_min {bbox[1]} out of bounds"
    assert 0 <= bbox[2] <= 1, f"x_max {bbox[2]} out of bounds"
    assert 0 <= bbox[3] <= 1, f"y_max {bbox[3]} out of bounds"
