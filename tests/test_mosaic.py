import numpy as np
import pytest

from albumentations.augmentations.mixing.transforms import Mosaic
from albumentations.core.bbox_utils import BboxParams
from albumentations.core.composition import Compose
from albumentations.core.keypoints_utils import KeypointParams


@pytest.mark.parametrize(
    "img_shape, target_size",
    [
        ((100, 80, 3), (100, 80)),  # Standard RGB
        ((64, 64, 1), (64, 64)),  # Grayscale
    ],
)
def test_mosaic_identity_single_image(img_shape: tuple[int, ...], target_size: tuple[int, int]) -> None:
    """Check Mosaic returns the original image when metadata is empty and target_size matches."""
    img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)

    # Set cell_shape = target_size for identity case
    transform = Mosaic(target_size=target_size, cell_shape=target_size, grid_yx=(1, 1), p=1.0)

    # Input data structure expects a list for metadata
    data = {"image": img, "mosaic_metadata": []}

    result = transform(**data)
    transformed_img = result["image"]

    assert transformed_img.shape == img.shape
    np.testing.assert_array_equal(transformed_img, img)


# Separate parametrize for shapes, sizes, and fill values
@pytest.mark.parametrize(
    "img_shape, target_size, fill, fill_mask",
    [
        # Matching sizes
        ((100, 80, 3), (100, 80), 128, 1),  # RGB
        ((64, 64, 1), (64, 64), 50, 2),  # Grayscale
        # Target smaller (cropping)
        ((100, 100, 3), (80, 80), 100, 4),
        # Target larger (padding)
        ((50, 50, 1), (70, 70), 200, 5),
    ],
)
# Separate parametrize for grid dimensions
@pytest.mark.parametrize(
    "grid_yx",
    [
        (1, 1),
        (2, 2),
        (1, 2),
        (3, 2),
        (1, 3),
    ],
)
def test_mosaic_identity_monochromatic(
    img_shape: tuple[int, ...],
    target_size: tuple[int, int],
    grid_yx: tuple[int, int],
    fill: int,
    fill_mask: int,
) -> None:
    """Check Mosaic returns a uniform image/mask if input is uniform (no metadata)."""
    # --- Image Setup ---
    if len(img_shape) == 2:
        img = np.full(img_shape, fill_value=fill, dtype=np.uint8)
        expected_output_shape_img = (*target_size,)
    else:
        img = np.full(img_shape, fill_value=fill, dtype=np.uint8)
        expected_output_shape_img = (*target_size, img_shape[-1])

    # --- Mask Setup ---
    mask_shape = (*img_shape[:2], 1)
    mask = np.full(mask_shape, fill_value=fill_mask, dtype=np.uint8)
    expected_output_shape_mask = (*target_size, 1)

    # --- Transform --- (Use 0 for padding values to test persistence)
    transform = Mosaic(
        target_size=target_size,
        grid_yx=grid_yx,
        p=1.0,
        fill=0,
        fill_mask=0,
    )

    # --- Apply ---
    data = {"image": img, "mask": mask, "mosaic_metadata": []}
    result = transform(**data)
    transformed_img = result["image"]
    transformed_mask = result["mask"]

    # --- Assertions (Image) ---
    assert transformed_img.shape == expected_output_shape_img
    assert transformed_img.dtype == img.dtype

    is_padded_h = target_size[0] > img_shape[0]
    is_padded_w = target_size[1] > img_shape[1]

    if not is_padded_h and not is_padded_w:
        expected_output_img = np.full(expected_output_shape_img, fill_value=fill, dtype=np.uint8)
        np.testing.assert_array_equal(transformed_img, expected_output_img)
    else:
        assert np.all((transformed_img == fill) | (transformed_img == 0))
        orig_h, orig_w = img_shape[:2]
        assert np.all(transformed_img[:orig_h, :orig_w] == fill)

    # --- Assertions (Mask) ---
    assert transformed_mask.shape == expected_output_shape_mask
    assert transformed_mask.dtype == mask.dtype

    if not is_padded_h and not is_padded_w:
        expected_output_mask = np.full(expected_output_shape_mask, fill_value=fill_mask, dtype=np.uint8)
        np.testing.assert_array_equal(transformed_mask, expected_output_mask)
    else:
        assert np.all((transformed_mask == fill_mask) | (transformed_mask == 0))
        orig_h, orig_w = mask_shape[:2]
        assert np.all(transformed_mask[:orig_h, :orig_w] == fill_mask)


def test_mosaic_identity_with_targets() -> None:
    """Check Mosaic returns original image, mask, and bboxes when grid is (1, 1) and no metadata."""
    img_size = (8, 6)
    img = np.random.randint(0, 256, size=(*img_size, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=img_size, dtype=np.uint8)
    # Bbox in albumentations format [x_min, y_min, x_max, y_max, class_id]
    bboxes = np.array(
        [
            [0.2, 0.3, 0.8, 0.7, 1],
            [0.1, 0.1, 0.5, 0.5, 2],
            [0.6, 0.2, 0.9, 0.4, 0],
        ],
        dtype=np.float32,
    )

    # Set cell_shape = target_size for identity case
    transform = Mosaic(target_size=img_size, cell_shape=img_size, grid_yx=(1, 1), p=1.0)

    # Use Compose to handle bbox processing
    pipeline = Compose(
        [
            transform,
        ],
        bbox_params=BboxParams(coord_format="albumentations", label_fields=["class_labels"]),
    )

    data = {
        "image": img.copy(),
        "mask": mask.copy(),
        "bboxes": bboxes.copy()[:, :4],  # Pass only coords
        "class_labels": bboxes[:, 4].tolist(),  # Pass labels separately
        "mosaic_metadata": [],
    }

    result = pipeline(**data)

    # Check image
    assert result["image"].shape == img.shape
    np.testing.assert_array_equal(result["image"], img)

    # Check mask
    assert result["mask"].shape == mask.shape
    np.testing.assert_array_equal(result["mask"], mask)

    # Check bboxes (should be identical after identity transform)
    # Need to reconstruct the expected format from Compose output
    expected_bboxes_with_labels = np.concatenate(
        (data["bboxes"], np.array(data["class_labels"])[..., np.newaxis]),
        axis=1,
    )
    result_bboxes_with_labels = np.concatenate(
        (result["bboxes"], np.array(result["class_labels"])[..., np.newaxis]),
        axis=1,
    )
    np.testing.assert_allclose(result_bboxes_with_labels, expected_bboxes_with_labels, atol=1e-6)


def test_mosaic_primary_mask_metadata_no_mask() -> None:
    """Test Mosaic behavior when primary has mask but metadata item doesn't.

    Expects the area corresponding to the metadata item without a mask to be
    filled with the fill_mask value in the output mask.
    """
    # Set a fixed seed for reproducibility of item selection and placement
    target_size = (100, 100)
    cell_shape = (100, 100)
    primary_image = np.zeros((*target_size, 3), dtype=np.uint8)
    primary_mask = np.ones((*target_size, 1), dtype=np.uint8) * 55  # Non-zero primary mask value

    # Metadata item with compatible image but NO mask
    metadata_item_no_mask = {"image": np.ones((80, 80, 3), dtype=np.uint8) * 10}
    # Metadata item with compatible image AND mask
    metadata_item_with_mask = {
        "image": np.ones((70, 70, 3), dtype=np.uint8) * 20,
        "mask": np.ones((70, 70, 1), dtype=np.uint8) * 77,  # Distinct mask value
    }

    metadata = [metadata_item_no_mask, metadata_item_with_mask, metadata_item_with_mask]
    fill_mask_value = 100  # Distinct fill value

    # Use a 2x2 grid to ensure all items (primary + 3 metadata) are potentially used
    transform = Compose(
        [
            Mosaic(
                grid_yx=(2, 2),
                center_range=(0.5, 0.5),
                cell_shape=cell_shape,
                target_size=target_size,
                fit_mode="contain",
                fill_mask=fill_mask_value,
                metadata_key="mosaic_input",
                p=1.0,
            ),
        ],
        seed=137,
        strict=True,
    )

    data = {
        "image": primary_image,
        "mask": primary_mask,
        "mosaic_input": metadata,
    }

    result = transform(**data)
    output_mask = result["mask"]

    # Basic shape check
    assert output_mask.shape == (*target_size, 1)
    assert output_mask.dtype == np.uint8

    # Check that all expected values are present in the mask
    unique_values = np.unique(output_mask)

    # Value from primary mask should be present (potentially cropped/transformed)
    # We check if *any* pixel has this value, as exact placement isn't tested here.
    assert 55 in unique_values

    # Value from metadata item *with* mask should be present
    assert 77 in unique_values

    # The fill_mask_value *must* be present, corresponding to the item without a mask
    assert fill_mask_value in unique_values

    # Optionally, check that the fill_value for the image (default 0) is NOT in the mask
    # unless the fill_mask_value itself was 0.
    if fill_mask_value != 0:
        assert 0 not in unique_values  # Assuming default image fill value 0 wasn't used for mask


def test_mosaic_obb_support() -> None:
    """Test that Mosaic declares OBB support."""
    transform = Mosaic()
    assert hasattr(transform, "_supported_bbox_types")
    assert "obb" in transform._supported_bbox_types
    assert "hbb" in transform._supported_bbox_types


def test_mosaic_obb_basic() -> None:
    """Test Mosaic with OBB bboxes - basic functionality."""
    target_size = (100, 100)
    grid_yx = (1, 1)
    cell_shape = (100, 100)

    # Primary data with OBB
    img_primary = np.ones((*target_size, 3), dtype=np.uint8) * 1
    # OBB format: [x_min, y_min, x_max, y_max, angle] in albumentations normalized format
    bboxes_primary = np.array([[0.2, 0.2, 0.8, 0.8, 45.0]], dtype=np.float32)

    transform = Compose(
        [
            Mosaic(
                target_size=target_size,
                cell_shape=cell_shape,
                grid_yx=grid_yx,
                p=1.0,
            ),
        ],
        bbox_params=BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            min_area=0.0,
            min_visibility=0.0,
        ),
        seed=137,
    )

    data = {
        "image": img_primary,
        "bboxes": bboxes_primary,
        "mosaic_metadata": [],
    }

    result = transform(**data)

    # Check OBB format is preserved (5 columns)
    assert result["bboxes"].shape[1] == 5


def test_mosaic_obb_with_metadata() -> None:
    """Test Mosaic with OBB bboxes from primary and metadata."""
    target_size = (100, 100)
    grid_yx = (2, 2)
    cell_shape = (60, 60)
    center_range = (0.5, 0.5)

    # Primary data with OBB
    img_primary = np.ones((*target_size, 3), dtype=np.uint8) * 1
    # OBB format: [x_min, y_min, x_max, y_max, angle]
    bboxes_primary = np.array([[0.3, 0.3, 0.7, 0.7, 30.0]], dtype=np.float32)
    bbox_classes_primary = [1]

    # Metadata with OBB
    img_meta1 = np.ones((80, 80, 3), dtype=np.uint8) * 2
    bboxes_meta1 = np.array([[0.25, 0.25, 0.75, 0.65, 60.0]], dtype=np.float32)
    bbox_classes_meta1 = [2]

    img_meta2 = np.ones((90, 90, 3), dtype=np.uint8) * 3
    bboxes_meta2 = np.array([[0.2, 0.2, 0.6, 0.6, 90.0]], dtype=np.float32)
    bbox_classes_meta2 = [3]

    metadata_list = [
        {"image": img_meta1, "bboxes": bboxes_meta1, "bbox_classes": bbox_classes_meta1},
        {"image": img_meta2, "bboxes": bboxes_meta2, "bbox_classes": bbox_classes_meta2},
    ]

    transform = Compose(
        [
            Mosaic(
                target_size=target_size,
                cell_shape=cell_shape,
                grid_yx=grid_yx,
                center_range=center_range,
                p=1.0,
            ),
        ],
        bbox_params=BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            label_fields=["bbox_classes"],
            min_area=0.0,
            min_visibility=0.0,
        ),
        seed=137,
    )

    data = {
        "image": img_primary,
        "bboxes": bboxes_primary,
        "bbox_classes": bbox_classes_primary,
        "mosaic_metadata": metadata_list,
    }

    result = transform(**data)

    # Check OBB format is preserved
    assert result["bboxes"].shape[1] == 5
    # Should have bboxes from multiple cells
    assert len(result["bboxes"]) > 0
    # Labels should be preserved
    assert "bbox_classes" in result
    assert len(result["bbox_classes"]) == len(result["bboxes"])


def test_mosaic_obb_empty_result() -> None:
    """Test Mosaic with OBB when all bboxes are filtered out."""
    target_size = (100, 100)
    grid_yx = (1, 1)
    cell_shape = (100, 100)

    img_primary = np.ones((*target_size, 3), dtype=np.uint8)
    # Empty bboxes array in OBB format
    bboxes_primary = np.empty((0, 5), dtype=np.float32)

    transform = Compose(
        [
            Mosaic(
                target_size=target_size,
                cell_shape=cell_shape,
                grid_yx=grid_yx,
                p=1.0,
            ),
        ],
        bbox_params=BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            min_area=0.0,
            min_visibility=0.0,
        ),
        seed=137,
    )

    data = {
        "image": img_primary,
        "bboxes": bboxes_primary,
        "mosaic_metadata": [],
    }

    result = transform(**data)

    # Even if empty, should preserve OBB format (5 columns)
    assert result["bboxes"].shape == (0, 5)


def test_mosaic_obb_empty_with_labels() -> None:
    """Test Mosaic with OBB preserves label columns when empty."""
    target_size = (100, 100)
    grid_yx = (1, 1)
    cell_shape = (100, 100)

    img_primary = np.ones((*target_size, 3), dtype=np.uint8)
    # Empty bboxes array in OBB format WITH label column (6 columns total)
    bboxes_primary = np.empty((0, 6), dtype=np.float32)
    bbox_classes_primary = []

    transform = Compose(
        [
            Mosaic(
                target_size=target_size,
                cell_shape=cell_shape,
                grid_yx=grid_yx,
                p=1.0,
            ),
        ],
        bbox_params=BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            label_fields=["bbox_classes"],
            min_area=0.0,
            min_visibility=0.0,
        ),
        seed=137,
    )

    data = {
        "image": img_primary,
        "bboxes": bboxes_primary,
        "bbox_classes": bbox_classes_primary,
        "mosaic_metadata": [],
    }

    result = transform(**data)

    # Should preserve 6 columns (5 for OBB + 1 for label)
    assert result["bboxes"].shape == (0, 6), f"Expected (0, 6), got {result['bboxes'].shape}"
    assert len(result["bbox_classes"]) == 0


def test_mosaic_hbb_empty_with_labels() -> None:
    """Test Mosaic with HBB preserves label columns when empty."""
    target_size = (100, 100)
    grid_yx = (1, 1)
    cell_shape = (100, 100)

    img_primary = np.ones((*target_size, 3), dtype=np.uint8)
    # Empty bboxes array in HBB format WITH label column (5 columns total)
    bboxes_primary = np.empty((0, 5), dtype=np.float32)
    bbox_classes_primary = []

    transform = Compose(
        [
            Mosaic(
                target_size=target_size,
                cell_shape=cell_shape,
                grid_yx=grid_yx,
                p=1.0,
            ),
        ],
        bbox_params=BboxParams(
            coord_format="albumentations",
            bbox_type="hbb",
            label_fields=["bbox_classes"],
            min_area=0.0,
            min_visibility=0.0,
        ),
        seed=137,
    )

    data = {
        "image": img_primary,
        "bboxes": bboxes_primary,
        "bbox_classes": bbox_classes_primary,
        "mosaic_metadata": [],
    }

    result = transform(**data)

    # Should preserve 5 columns (4 for HBB + 1 for label)
    assert result["bboxes"].shape == (0, 5), f"Expected (0, 5), got {result['bboxes'].shape}"
    assert len(result["bbox_classes"]) == 0


def test_mosaic_simplified_deterministic() -> None:
    """Test Mosaic with fixed parameters, albumentations format, no labels."""
    target_size = (100, 100)
    grid_yx = (1, 2)
    center_range = (0.5, 0.5)
    # Set cell_shape = target_size to match the deterministic calculation assumptions
    cell_shape = (100, 100)

    # --- Primary Data ---
    img_primary = np.ones((*target_size, 3), dtype=np.uint8) * 1
    mask_primary = np.ones((*target_size, 1), dtype=np.uint8) * 11
    # BBoxes: Albumentations format [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
    bboxes_primary = np.array([[0, 0, 1, 1]], dtype=np.float32)
    # Keypoints: Albumentations format [x, y, Z, angle, scale]
    keypoints_primary = np.array([[10, 10, 0, 0, 0], [50, 50, 0, 0, 0]], dtype=np.float32)

    # --- Metadata ---
    img_meta = np.ones((*cell_shape, 3), dtype=np.uint8) * 2  # Use cell_shape for meta consistency
    mask_meta = np.ones(cell_shape, dtype=np.uint8) * 22
    bboxes_meta = np.array([[0, 0, 1, 1]], dtype=np.float32)  # rel to meta_size
    keypoints_meta = np.array([[10, 10, 0, 0, 0], [90, 90, 0, 0, 0]], dtype=np.float32)  # rel to meta_size

    metadata_list = [
        {
            "image": img_meta,
            "mask": mask_meta,
            "bboxes": bboxes_meta,
            "keypoints": keypoints_meta,
        },
    ]

    # --- Transform ---
    transform = Mosaic(
        target_size=target_size,
        grid_yx=grid_yx,
        cell_shape=cell_shape,  # Use defined cell_shape
        center_range=center_range,
        p=1.0,
        fill=0,
        fill_mask=0,
        fit_mode="cover",  # Match the calculation trace
    )

    pipeline = Compose(
        [
            transform,
        ],
        bbox_params=BboxParams(coord_format="albumentations", min_visibility=0.0, min_area=0.0),
        keypoint_params=KeypointParams(coord_format="xy"),
    )

    # --- Input Data ---
    data = {
        "image": img_primary,
        "mask": mask_primary,
        "bboxes": bboxes_primary,
        "keypoints": keypoints_primary,
        "mosaic_metadata": metadata_list,
    }

    # --- Apply ---
    result = pipeline(**data)

    # --- Calculate Expected Annotations ---
    # Corrected expectation based on fit_mode="cover" calculation trace:
    expected_bboxes = np.array([[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]], dtype=np.float32)

    # --- Assertions ---
    # Image/Mask Shape Check
    assert result["image"].shape == (*target_size, 3)
    assert result["mask"].shape == (*target_size, 1)  # Mask should have channel dimension
    # Relaxed Image/Mask Content Check: Ensure the two halves are not just the fill value
    split_col = 50  # Based on center_range=(0.5, 0.5)
    assert not np.all(result["image"][:, :split_col] == 0)  # Check left half
    assert not np.all(result["image"][:, split_col:] == 0)  # Check right half
    assert not np.all(result["mask"][:, :split_col] == 0)
    assert not np.all(result["mask"][:, split_col:] == 0)

    # Check bboxes
    assert "bboxes" in result
    np.testing.assert_allclose(result["bboxes"], expected_bboxes, atol=1e-6)

    # Relaxed Keypoints check
    assert "keypoints" in result
    assert result["keypoints"].shape[0] > 0  # Expect some keypoints
    assert result["keypoints"].shape[1] == 5  # x, y, Z, angle, scale
