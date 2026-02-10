"""Tests for distortion transforms with OBB support."""

import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize(
    "transform_cls,params",
    [
        pytest.param(A.ElasticTransform, {"alpha": 1, "sigma": 50}, id="ElasticTransform"),
        pytest.param(A.GridDistortion, {"num_steps": 5, "distort_limit": 0.3}, id="GridDistortion"),
        pytest.param(A.OpticalDistortion, {"distort_limit": 0.05}, id="OpticalDistortion"),
        pytest.param(A.PiecewiseAffine, {"scale": (0.03, 0.05)}, id="PiecewiseAffine"),
        pytest.param(A.ThinPlateSpline, {"scale_range": (0.2, 0.4)}, id="ThinPlateSpline"),
    ],
)
@pytest.mark.obb
def test_distortion_transforms_declare_obb_support(transform_cls, params):
    """Test that distortion transforms declare OBB support in _supported_bbox_types."""
    transform = transform_cls(**params)
    assert hasattr(transform, "_supported_bbox_types"), (
        f"{transform_cls.__name__} should have _supported_bbox_types attribute"
    )
    assert "obb" in transform._supported_bbox_types, f"{transform_cls.__name__} should support OBB"
    assert "hbb" in transform._supported_bbox_types, f"{transform_cls.__name__} should support HBB"


@pytest.mark.parametrize(
    "transform_cls,params",
    [
        pytest.param(A.ElasticTransform, {"alpha": 1, "sigma": 50}, id="ElasticTransform"),
        pytest.param(A.GridDistortion, {"num_steps": 5, "distort_limit": 0.3}, id="GridDistortion"),
        pytest.param(A.OpticalDistortion, {"distort_limit": 0.05}, id="OpticalDistortion"),
        pytest.param(A.PiecewiseAffine, {"scale": (0.03, 0.05)}, id="PiecewiseAffine"),
    ],
)
@pytest.mark.obb
def test_distortion_transforms_preserve_obb_format(transform_cls, params):
    """Test that distortion transforms preserve OBB format (5 values)."""
    np.random.seed(137)
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    bboxes = np.array([[0.3, 0.3, 0.7, 0.7, 45]], dtype=np.float32)

    transform = A.Compose(
        [
            transform_cls(**params, p=1.0),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            label_fields=[],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )

    result = transform(image=image, bboxes=bboxes)

    # Check bbox wasn't filtered out
    assert len(result["bboxes"]) > 0, f"{transform_cls.__name__} filtered out all bboxes"

    # Check format preserved (5 values for OBB)
    assert result["bboxes"].shape[1] == 5, (
        f"{transform_cls.__name__} should preserve 5 values for OBB, got {result['bboxes'].shape[1]}"
    )


@pytest.mark.parametrize(
    "transform_cls,params",
    [
        pytest.param(A.ElasticTransform, {"alpha": 1, "sigma": 50}, id="ElasticTransform"),
        pytest.param(A.GridDistortion, {"num_steps": 5, "distort_limit": 0.3}, id="GridDistortion"),
    ],
)
@pytest.mark.obb
def test_distortion_transforms_obb_with_labels(transform_cls, params):
    """Test that distortion transforms preserve label fields with OBB."""
    np.random.seed(137)
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    bboxes = np.array([[0.3, 0.3, 0.7, 0.7, 45]], dtype=np.float32)
    labels = [1]

    transform = A.Compose(
        [
            transform_cls(**params, p=1.0),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            label_fields=["labels"],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )

    result = transform(image=image, bboxes=bboxes, labels=labels)

    if len(result["bboxes"]) > 0:
        assert len(result["labels"]) == len(result["bboxes"]), f"{transform_cls.__name__} bbox/label count mismatch"
        assert result["labels"][0] == 1, f"{transform_cls.__name__} didn't preserve label"


@pytest.mark.parametrize(
    "transform_cls,params,bbox_format,input_bbox",
    [
        pytest.param(
            A.ElasticTransform,
            {"alpha": 1, "sigma": 50},
            "albumentations",
            [0.3, 0.3, 0.7, 0.7, 45],
            id="ElasticTransform-albumentations",
        ),
        pytest.param(
            A.GridDistortion,
            {"num_steps": 5, "distort_limit": 0.3},
            "pascal_voc",
            [30, 30, 70, 70, 45],
            id="GridDistortion-pascal_voc",
        ),
        pytest.param(
            A.OpticalDistortion,
            {"distort_limit": 0.05},
            "coco",
            [30, 30, 40, 40, 45],
            id="OpticalDistortion-coco",
        ),
    ],
)
@pytest.mark.obb
def test_distortion_transforms_obb_format_consistency(transform_cls, params, bbox_format, input_bbox):
    """Test that input format == output format for distortion transforms with OBB."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    transform = A.Compose(
        [
            transform_cls(**params, p=1.0),
        ],
        bbox_params=A.BboxParams(
            coord_format=bbox_format,
            bbox_type="obb",
            label_fields=[],
        ),
    )

    np.random.seed(137)
    result = transform(image=image, bboxes=[input_bbox])

    if len(result["bboxes"]) > 0:
        output_bbox = result["bboxes"][0]

        # Verify format consistency
        if bbox_format in ["pascal_voc", "coco"]:
            # Pixel coordinates - should have values that could be > 1
            assert any(abs(v) > 1 for v in output_bbox[:4]), (
                f"Expected pixel coords for {bbox_format}, got {output_bbox}"
            )
        else:
            # Normalized coordinates [0, 1]
            assert all(0 <= v <= 1.01 for v in output_bbox[:4]), (
                f"Expected normalized coords for {bbox_format}, got {output_bbox}"
            )

        # Check OBB has angle
        assert len(output_bbox) == 5, f"Expected 5 values for OBB, got {len(output_bbox)}"


@pytest.mark.obb
def test_pixel_dropout_declares_obb_support():
    """Test that PixelDropout declares OBB support."""
    transform = A.PixelDropout()
    assert "obb" in transform._supported_bbox_types
    assert "hbb" in transform._supported_bbox_types


@pytest.mark.obb
def test_pixel_dropout_with_obb():
    """Test PixelDropout works with OBB bboxes (passes through unchanged)."""
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    bboxes = np.array([[0.3, 0.3, 0.7, 0.7, 45]], dtype=np.float32)

    transform = A.Compose(
        [
            A.PixelDropout(dropout_prob=0.1, p=1.0),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            label_fields=[],
        ),
    )

    result = transform(image=image, bboxes=bboxes)

    # PixelDropout doesn't transform coordinates, so bbox should pass through
    assert len(result["bboxes"]) == 1
    assert result["bboxes"].shape == (1, 5)
    np.testing.assert_array_equal(result["bboxes"], bboxes)
