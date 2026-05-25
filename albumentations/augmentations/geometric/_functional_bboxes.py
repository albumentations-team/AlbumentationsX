"""Bounding box geometric functional helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ._functional_grid import (
    swap_tiles_on_image,
)
from ._functional_images import (
    C4_GROUP_ELEMENT_TO_K,
    ROT90_180_FACTOR,
    ROT90_270_FACTOR,
    apply_affine_to_points,
    calculate_affine_transform_padding,
    distort_image,
    get_pad_grid_dimensions,
    is_identity_matrix,
    morphology,
)
from ._functional_shared import (
    BBOX_OBB_MIN_COLUMNS,
    NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    REFLECT_BORDER_MODES,
    bboxes_from_masks,
    bboxes_to_mask,
    cv2,
    denormalize_bboxes,
    handle_empty_array,
    mask_to_bboxes,
    masks_from_bboxes,
    normalize_bboxes,
    np,
    obb_to_polygons,
    polygons_to_obb,
    reduce_sum,
    remap,
)


def _split_polygons_and_extras(bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    polygons = obb_to_polygons(bboxes)
    extras = bboxes[:, 5:] if bboxes.shape[1] > BBOX_OBB_MIN_COLUMNS else None
    return polygons, extras


def _split_obb_params(
    bboxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]
    center_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    center_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    angle = bboxes[:, 4]
    extras = bboxes[:, 5:] if bboxes.shape[1] > BBOX_OBB_MIN_COLUMNS else None
    return center_x, center_y, width, height, angle, extras


def _merge_obb_params(
    center_x: np.ndarray,
    center_y: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    angle: np.ndarray,
    extras: np.ndarray | None,
) -> np.ndarray:
    x_min = center_x - width * 0.5
    x_max = center_x + width * 0.5
    y_min = center_y - height * 0.5
    y_max = center_y + height * 0.5
    obb = np.stack([x_min, y_min, x_max, y_max, angle], axis=1)
    if extras is not None:
        obb = np.concatenate([obb, extras], axis=1)
    return obb


@handle_empty_array("bboxes")
def resize_bboxes(
    bboxes: np.ndarray,
    image_shape: tuple[int, int],
    output_shape: tuple[int, int],
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Resize bounding boxes according to image scaling. Params: image_shape,
    output_shape, bbox_type (hbb/obb). Normalized coords; OBB supports non-uniform scale.

    Args:
        bboxes (np.ndarray): Array of bboxes in normalized coords [x_min, y_min, x_max, y_max, (angle), ...]
        image_shape (tuple[int, int]): Original image shape (height, width)
        output_shape (tuple[int, int]): Target image shape (height, width)
        bbox_type (Literal['hbb', 'obb']): Type of bboxes - "hbb" or "obb"

    Returns:
        np.ndarray: Resized bboxes in normalized coordinates.

    """
    if bbox_type == "hbb":
        # HBB is scale-invariant in normalized coords
        return bboxes

    # OBB: check if uniform scaling
    height, width = image_shape[:2]
    output_height, output_width = output_shape[:2]
    scale_x = output_width / width
    scale_y = output_height / height

    if abs(scale_x - scale_y) < 1e-7:
        # Uniform scaling: OBB angle is preserved, coordinates are scale-invariant
        return bboxes

    # Non-uniform scaling: denormalize OBB, convert to polygons, scale, convert back
    bboxes_px = denormalize_bboxes(bboxes, image_shape)
    extras = bboxes_px[:, 5:] if bboxes_px.shape[1] > BBOX_OBB_MIN_COLUMNS else None

    # Convert to polygons in pixel space
    polygons_px = obb_to_polygons(bboxes_px)

    # Scale the polygons
    polygons_px[..., 0] *= scale_x
    polygons_px[..., 1] *= scale_y

    # Convert back to OBB in pixel space
    transformed_bboxes_px = polygons_to_obb(polygons_px, extra_fields=extras)

    # Normalize back to [0, 1]
    return normalize_bboxes(transformed_bboxes_px, output_shape)


@handle_empty_array("bboxes")
def bboxes_rot90(
    bboxes: np.ndarray,
    group_element: Literal["e", "r90", "r180", "r270"],
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Rotate bounding boxes by 90° CCW (see np.rot90). group_element: e, r90, r180,
    r270. Supports hbb and obb; OBB center/size/angle updated correctly.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        group_element (Literal['e', 'r90', 'r180', 'r270']): C4 group element to apply.
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB uses center/size/angle update.

    Returns:
        np.ndarray: Rotated bounding boxes

    """
    rot90_count = C4_GROUP_ELEMENT_TO_K[group_element]
    if rot90_count == 0:
        return bboxes

    if bbox_type == "obb":
        center_x, center_y, width, height, angle, extras = _split_obb_params(bboxes)
        if rot90_count == 1:
            new_center_x, new_center_y = center_y, 1 - center_x
        elif rot90_count == ROT90_180_FACTOR:
            new_center_x, new_center_y = 1 - center_x, 1 - center_y
        else:  # rot90_count == ROT90_270_FACTOR
            new_center_x, new_center_y = 1 - center_y, center_x

        if rot90_count % 2 == 1:
            width, height = height, width

        angle = angle + rot90_count * 90
        return _merge_obb_params(new_center_x, new_center_y, width, height, angle, extras)

    rotated_bboxes = bboxes.copy()
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    if rot90_count == 1:
        rotated_bboxes[:, 0] = y_min
        rotated_bboxes[:, 1] = 1 - x_max
        rotated_bboxes[:, 2] = y_max
        rotated_bboxes[:, 3] = 1 - x_min
    elif rot90_count == ROT90_180_FACTOR:
        rotated_bboxes[:, 0] = 1 - x_max
        rotated_bboxes[:, 1] = 1 - y_max
        rotated_bboxes[:, 2] = 1 - x_min
        rotated_bboxes[:, 3] = 1 - y_min
    elif rot90_count == ROT90_270_FACTOR:
        rotated_bboxes[:, 0] = 1 - y_max
        rotated_bboxes[:, 1] = x_min
        rotated_bboxes[:, 2] = 1 - y_min
        rotated_bboxes[:, 3] = x_max

    return rotated_bboxes


@handle_empty_array("bboxes")
def bboxes_d4(
    bboxes: np.ndarray,
    group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Apply D4 symmetry (rotations and reflections) to bounding boxes. group_member:
    e, r90, r180, r270, v, hvt, h, t. Supports hbb and obb.

    The function transforms a bounding box according to the specified group member from the `D_4` group.
    These transformations include rotations and reflections, specified to work on an image's bounding box given
    its dimensions.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
        Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
        group_member (Literal['e', 'r90', 'r180', 'r270', 'v', 'hvt', 'h', 't']): A string identifier for the
            `D_4` group transformation to apply.
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB uses center/size/angle update.

    Returns:
        np.ndarray: The transformed bounding box.

    Raises:
        ValueError: If an invalid group member is specified.

    """
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: bboxes_rot90(x, "r90", bbox_type=bbox_type),  # Rotate 90 degrees
        "r180": lambda x: bboxes_rot90(x, "r180", bbox_type=bbox_type),  # Rotate 180 degrees
        "r270": lambda x: bboxes_rot90(x, "r270", bbox_type=bbox_type),  # Rotate 270 degrees
        "v": lambda x: bboxes_vflip(x, bbox_type=bbox_type),  # Vertical flip
        "hvt": lambda x: bboxes_transpose(
            bboxes_rot90(x, "r180", bbox_type=bbox_type),
            bbox_type=bbox_type,
        ),  # Reflect over anti-diagonal
        "h": lambda x: bboxes_hflip(x, bbox_type=bbox_type),  # Horizontal flip
        "t": lambda x: bboxes_transpose(x, bbox_type=bbox_type),  # Transpose (reflect over main diagonal)
    }

    # Execute the appropriate transformation
    if group_member in transformations:
        return transformations[group_member](bboxes)

    raise ValueError(f"Invalid group member: {group_member}")


@handle_empty_array("bboxes")
def perspective_bboxes(
    bboxes: np.ndarray,
    image_shape: tuple[int, int],
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Apply perspective transformation to bounding boxes. matrix, image_shape,
    max_width, max_height, keep_size. HBB and OBB supported; OBB via corners.

    This function transforms bounding boxes using the given perspective transformation matrix.
    It handles bounding boxes with additional attributes beyond the standard coordinates.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (num_bboxes, 4+).
                             Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
                             Additional columns beyond the first 4 are preserved unchanged.
        image_shape (tuple[int, int]): The shape of the image (height, width).
        matrix (np.ndarray): The perspective transformation matrix.
        max_width (int): The maximum width of the output image.
        max_height (int): The maximum height of the output image.
        keep_size (bool): If True, maintains the original image size after transformation.
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB path uses polygons.

    Returns:
        np.ndarray: An array of transformed bounding boxes with the same shape as input.
                    The first 4 columns contain the transformed coordinates, and any
                    additional columns are preserved from the input.

    Note:
        - This function modifies only the coordinate columns (first 4) of the input bounding boxes.
        - Any additional attributes (columns beyond the first 4) are kept unchanged.
        - The function handles denormalization and renormalization of coordinates internally.

    Examples:
        >>> bboxes = np.array([[0.1, 0.1, 0.3, 0.3, 1], [0.5, 0.5, 0.8, 0.8, 2]])
        >>> image_shape = (100, 100)
        >>> matrix = np.array([[1.5, 0.2, -20], [-0.1, 1.3, -10], [0.002, 0.001, 1]])
        >>> transformed_bboxes = perspective_bboxes(bboxes, image_shape, matrix, 150, 150, False)

    """
    height, width = image_shape[:2]
    transformed_bboxes = bboxes.copy()
    if bbox_type == "obb":
        polygons, extras = _split_polygons_and_extras(bboxes)
        polygons = polygons.copy()
        polygons[..., 0] *= width
        polygons[..., 1] *= height
        transformed = cv2.perspectiveTransform(polygons.reshape(-1, 1, 2).astype(np.float32), matrix)
        transformed = transformed.reshape(polygons.shape)

        if keep_size:
            scale_x, scale_y = width / max_width, height / max_height
            transformed[..., 0] *= scale_x
            transformed[..., 1] *= scale_y
            output_shape = image_shape
        else:
            output_shape = (max_height, max_width)

        transformed[..., 0] /= output_shape[1]
        transformed[..., 1] /= output_shape[0]
        return polygons_to_obb(transformed, extra_fields=extras)

    denormalized_coords = denormalize_bboxes(bboxes[:, :4], image_shape)

    x_min, y_min, x_max, y_max = denormalized_coords.T
    points = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
    ).transpose(2, 0, 1)
    points_reshaped = points.reshape(-1, 1, 2)

    transformed_points = cv2.perspectiveTransform(
        points_reshaped.astype(np.float32),
        matrix,
    )
    transformed_points = transformed_points.reshape(-1, 4, 2)

    new_coords = np.column_stack(
        [
            transformed_points[:, :, 0].min(axis=1),
            transformed_points[:, :, 1].min(axis=1),
            transformed_points[:, :, 0].max(axis=1),
            transformed_points[:, :, 1].max(axis=1),
        ],
    )

    if keep_size:
        scale_x, scale_y = width / max_width, height / max_height
        new_coords[:, [0, 2]] *= scale_x
        new_coords[:, [1, 3]] *= scale_y
        output_shape = image_shape
    else:
        output_shape = (max_height, max_width)

    normalized_coords = normalize_bboxes(new_coords, output_shape)
    transformed_bboxes[:, :4] = normalized_coords

    return transformed_bboxes


@handle_empty_array("bboxes")
def bboxes_affine_largest_box(
    bboxes: np.ndarray,
    matrix: np.ndarray,
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Apply affine to bboxes and return largest enclosing axis-aligned boxes.
    matrix, image_shape, border_mode. For hbb type. Returns (N, 4+).

    This function transforms each corner of every bounding box using the given affine transformation
    matrix, then computes the new bounding boxes that fully enclose the transformed corners.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (N, 4+) where N is the number of
                             bounding boxes. Each row should contain [x_min, y_min, x_max, y_max]
                             followed by any additional attributes (e.g., class labels).
        matrix (np.ndarray): The 3x3 affine transformation matrix to apply.
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB path uses polygon transform.

    Returns:
        np.ndarray: An array of transformed bounding boxes with the same shape as the input.
                    Each row contains [new_x_min, new_y_min, new_x_max, new_y_max] followed by
                    any additional attributes from the input bounding boxes.

    Note:
        - This function assumes that the input bounding boxes are in the format [x_min, y_min, x_max, y_max].
        - The resulting bounding boxes are the smallest axis-aligned boxes that completely
          enclose the transformed original boxes. They may be larger than the minimal possible
          bounding box if the original box becomes rotated.
        - Any additional attributes beyond the first 4 coordinates are preserved unchanged.
        - This method is called "largest box" because it returns the largest axis-aligned box
          that encloses all corners of the transformed bounding box.

    Examples:
        >>> bboxes = np.array([[10, 10, 20, 20, 1], [30, 30, 40, 40, 2]])  # Two boxes with class labels
        >>> matrix = np.array([[2, 0, 5], [0, 2, 5], [0, 0, 1]])  # Scale by 2 and translate by (5, 5)
        >>> transformed_bboxes = bboxes_affine_largest_box(bboxes, matrix)
        >>> print(transformed_bboxes)
        [[ 25.  25.  45.  45.   1.]
         [ 65.  65.  85.  85.   2.]]

    """
    if bbox_type == "obb":
        polygons, extras = _split_polygons_and_extras(bboxes)
        transformed = apply_affine_to_points(polygons.reshape(-1, 2), matrix).reshape(polygons.shape)
        return polygons_to_obb(transformed, extra_fields=extras)

    # Extract corners of all bboxes
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    corners = (
        np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]).transpose(2, 0, 1).reshape(-1, 2)
    )

    # Transform all corners at once
    transformed_corners = apply_affine_to_points(corners, matrix).reshape(-1, 4, 2)

    # Compute new bounding boxes
    new_x_min = np.min(transformed_corners[:, :, 0], axis=1)
    new_x_max = np.max(transformed_corners[:, :, 0], axis=1)
    new_y_min = np.min(transformed_corners[:, :, 1], axis=1)
    new_y_max = np.max(transformed_corners[:, :, 1], axis=1)

    return np.column_stack([new_x_min, new_y_min, new_x_max, new_y_max, bboxes[:, 4:]])


@handle_empty_array("bboxes")
def bboxes_affine_ellipse(
    bboxes: np.ndarray,
    matrix: np.ndarray,
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Apply affine to bboxes via ellipse approximation (center, axes, angle).
    matrix, image_shape, border_mode. For obb type. Returns (N, 5+).

    This function transforms bounding boxes by approximating each box with an ellipse,
    transforming points along the ellipse's circumference, and then computing the
    new bounding box that encloses the transformed ellipse.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (N, 4+) where N is the number of
                             bounding boxes. Each row should contain [x_min, y_min, x_max, y_max]
                             followed by any additional attributes (e.g., class labels).
        matrix (np.ndarray): The 3x3 affine transformation matrix to apply.
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB path uses polygon transform.

    Returns:
        np.ndarray: An array of transformed bounding boxes with the same shape as the input.
                    Each row contains [new_x_min, new_y_min, new_x_max, new_y_max] followed by
                    any additional attributes from the input bounding boxes.

    Note:
        - This function assumes that the input bounding boxes are in the format [x_min, y_min, x_max, y_max].
        - The ellipse approximation method can provide a tighter bounding box compared to the
          largest box method, especially for rotations.
        - 360 points are used to approximate each ellipse, which provides a good balance between
          accuracy and computational efficiency.
        - Any additional attributes beyond the first 4 coordinates are preserved unchanged.
        - This method may be more suitable for objects that are roughly elliptical in shape.

    """
    if bbox_type == "obb":
        polygons, extras = _split_polygons_and_extras(bboxes)
        transformed = apply_affine_to_points(polygons.reshape(-1, 2), matrix).reshape(polygons.shape)
        return polygons_to_obb(transformed, extra_fields=extras)
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    bbox_width = (x_max - x_min) / 2
    bbox_height = (y_max - y_min) / 2
    center_x = x_min + bbox_width
    center_y = y_min + bbox_height

    angles = np.arange(0, 360, dtype=np.float32)
    cos_angles = np.cos(np.radians(angles))
    sin_angles = np.sin(np.radians(angles))

    # Generate points for all ellipses at once
    x = bbox_width[:, np.newaxis] * sin_angles + center_x[:, np.newaxis]
    y = bbox_height[:, np.newaxis] * cos_angles + center_y[:, np.newaxis]
    points = np.stack([x, y], axis=-1).reshape(-1, 2)

    # Transform all points at once using the helper function
    transformed_points = apply_affine_to_points(points, matrix)

    transformed_points = transformed_points.reshape(len(bboxes), -1, 2)

    # Compute new bounding boxes
    new_x_min = np.min(transformed_points[:, :, 0], axis=1)
    new_x_max = np.max(transformed_points[:, :, 0], axis=1)
    new_y_min = np.min(transformed_points[:, :, 1], axis=1)
    new_y_max = np.max(transformed_points[:, :, 1], axis=1)

    return np.column_stack([new_x_min, new_y_min, new_x_max, new_y_max, bboxes[:, 4:]])


@handle_empty_array("bboxes")
def bboxes_affine(
    bboxes: np.ndarray,
    matrix: np.ndarray,
    rotate_method: Literal["largest_box", "ellipse"],
    image_shape: tuple[int, int],
    border_mode: int,
    output_shape: tuple[int, int],
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Apply affine transformation to bounding boxes. matrix, image_shape,
    border_mode. Dispatches to largest-box (hbb) or ellipse (obb).

    For reflection border modes (cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT), this function:
    1. Calculates necessary padding to avoid information loss
    2. Applies padding to the bounding boxes
    3. Adjusts the transformation matrix to account for padding
    4. Applies the affine transformation
    5. Validates the transformed bounding boxes

    For other border modes, it directly applies the affine transformation without padding.

    Args:
        bboxes (np.ndarray): Input bounding boxes
        matrix (np.ndarray): Affine transformation matrix
        rotate_method (Literal['largest_box', 'ellipse']): Method for rotating bounding boxes
            ('largest_box' or 'ellipse').
            Only applies to HBB (axis-aligned) bounding boxes. Ignored for OBB.
        image_shape (tuple[int, int]): Shape of the input image
        border_mode (int): OpenCV border mode
        output_shape (tuple[int, int]): Shape of the output image
        bbox_type (Literal['hbb', 'obb']): Bounding box type. OBB uses polygon transformation
            regardless of rotate_method.

    Returns:
        np.ndarray: Transformed and normalized bounding boxes

    """
    if is_identity_matrix(matrix):
        return bboxes

    if bbox_type == "obb":
        # polygons → transform → polygons_to_obb (corner-based; pixel coords for accuracy)
        denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)
        polygons, extras = _split_polygons_and_extras(denormalized_bboxes)
        polygons = polygons.copy()
        transformed_polygons = apply_affine_to_points(polygons.reshape(-1, 2), matrix).reshape(polygons.shape)
        transformed_bboxes_px = polygons_to_obb(transformed_polygons, extra_fields=extras)
        transformed_bboxes_px[:, :4] = normalize_bboxes(transformed_bboxes_px[:, :4], output_shape)
        return validate_bboxes(transformed_bboxes_px, (1, 1))

    bboxes = denormalize_bboxes(bboxes, image_shape)

    if border_mode in REFLECT_BORDER_MODES:
        # Step 1: Compute affine transform padding
        pad_left, pad_right, pad_top, pad_bottom = calculate_affine_transform_padding(
            matrix,
            image_shape,
        )
        grid_dimensions = get_pad_grid_dimensions(
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            image_shape,
        )
        bboxes = generate_reflected_bboxes(
            bboxes,
            grid_dimensions,
            image_shape,
            center_in_origin=True,
        )

    # Apply affine transform
    if rotate_method == "largest_box":
        transformed_bboxes = bboxes_affine_largest_box(bboxes, matrix, bbox_type=bbox_type)
    elif rotate_method == "ellipse":
        transformed_bboxes = bboxes_affine_ellipse(bboxes, matrix, bbox_type=bbox_type)
    else:
        raise ValueError(f"Method {rotate_method} is not a valid rotation method.")

    # Validate and normalize bboxes
    validated_bboxes = validate_bboxes(transformed_bboxes, output_shape)

    return normalize_bboxes(validated_bboxes, output_shape)


@handle_empty_array("bboxes")
def bboxes_vflip(bboxes: np.ndarray, bbox_type: Literal["hbb", "obb"]) -> np.ndarray:
    """Flip bounding boxes vertically. Normalized coords; y_min, y_max swapped.
    Supports hbb and obb (angle adjusted). For VerticalFlip.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB uses center/size/angle update.

    Returns:
        np.ndarray: Vertically flipped bounding boxes

    """
    if bbox_type == "obb":
        center_x, center_y, width, height, angle, extras = _split_obb_params(bboxes)
        center_y = 1 - center_y
        angle = -angle
        return _merge_obb_params(center_x, center_y, width, height, angle, extras)
    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 1] = 1 - bboxes[:, 3]  # new y_min = 1 - y_max
    flipped_bboxes[:, 3] = 1 - bboxes[:, 1]  # new y_max = 1 - y_min

    return flipped_bboxes


@handle_empty_array("bboxes")
def bboxes_hflip(bboxes: np.ndarray, bbox_type: Literal["hbb", "obb"]) -> np.ndarray:
    """Flip bounding boxes horizontally. Normalized coords; x_min, x_max swapped.
    Supports hbb and obb (angle adjusted). For HorizontalFlip.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB uses center/size/angle update.

    Returns:
        np.ndarray: Horizontally flipped bounding boxes

    """
    if bbox_type == "obb":
        center_x, center_y, width, height, angle, extras = _split_obb_params(bboxes)
        center_x = 1 - center_x
        angle = 180.0 - angle
        return _merge_obb_params(center_x, center_y, width, height, angle, extras)
    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 0] = 1 - bboxes[:, 2]  # new x_min = 1 - x_max
    flipped_bboxes[:, 2] = 1 - bboxes[:, 0]  # new x_max = 1 - x_min

    return flipped_bboxes


@handle_empty_array("bboxes")
def bboxes_transpose(bboxes: np.ndarray, bbox_type: Literal["hbb", "obb"]) -> np.ndarray:
    """Transpose bounding boxes along the main diagonal. Swap x and y coords;
    for obb angle updated. Normalized coords. For Transpose transform.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB uses center/size/angle update.

    Returns:
        np.ndarray: Transposed bounding boxes

    """
    if bbox_type == "obb":
        center_x, center_y, width, height, angle, extras = _split_obb_params(bboxes)
        center_x, center_y = center_y, center_x
        width, height = height, width
        angle = 90.0 - angle
        return _merge_obb_params(center_x, center_y, width, height, angle, extras)
    transposed_bboxes = bboxes.copy()
    transposed_bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

    return transposed_bboxes


@handle_empty_array("bboxes")
def remap_bboxes(
    bboxes: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    image_shape: tuple[int, int],
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Remap bounding boxes using displacement maps. map_x, map_y; bbox_type hbb/obb.
    Converts bboxes to mask, remaps, converts back. For distortion transforms.

    Args:
        bboxes (np.ndarray): Bounding boxes array
        map_x (np.ndarray): X displacement map
        map_y (np.ndarray): Y displacement map
        image_shape (tuple[int, int]): Image shape (height, width)
        bbox_type (Literal['hbb', 'obb']): Type of bounding box - "hbb" for axis-aligned or "obb" for oriented

    Returns:
        np.ndarray: Remapped bounding boxes.

    """
    # Convert bboxes to mask
    bbox_masks = bboxes_to_mask(bboxes, image_shape)

    # Ensure maps are float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    transformed_masks = remap(
        bbox_masks,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    )

    # Convert masks back to bboxes
    return mask_to_bboxes(transformed_masks, bboxes, bbox_type=bbox_type)


@handle_empty_array("bboxes")
def pad_bboxes(
    bboxes: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Pad bounding boxes by a given amount (in normalized or pixel units). Params:
    pad_x, pad_y or pad_amount. Keeps boxes in [0,1] or image bounds.

    This function pads bounding boxes by a given amount.
    It handles both reflection and padding.

    Args:
        bboxes (np.ndarray): The bounding boxes to pad.
        pad_top (int): The amount to pad the top of the bounding boxes.
        pad_bottom (int): The amount to pad the bottom of the bounding boxes.
        pad_left (int): The amount to pad the left of the bounding boxes.
        pad_right (int): The amount to pad the right of the bounding boxes.
        border_mode (int): The border mode to use.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The padded bounding boxes.

    """
    if border_mode not in REFLECT_BORDER_MODES:
        shift_vector = np.array([pad_left, pad_top, pad_left, pad_top])
        return shift_bboxes(bboxes, shift_vector)

    grid_dimensions = get_pad_grid_dimensions(
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        image_shape,
    )

    bboxes = generate_reflected_bboxes(bboxes, grid_dimensions, image_shape)

    # Calculate the number of grid cells added on each side
    original_row, original_col = grid_dimensions["original_position"]

    image_height, image_width = image_shape[:2]

    # Subtract the offset based on the number of added grid cells
    left_shift = original_col * image_width - pad_left
    top_shift = original_row * image_height - pad_top

    shift_vector = np.array([-left_shift, -top_shift, -left_shift, -top_shift])

    bboxes = shift_bboxes(bboxes, shift_vector)

    new_height = pad_top + pad_bottom + image_height
    new_width = pad_left + pad_right + image_width

    return validate_bboxes(bboxes, (new_height, new_width))


def validate_bboxes(bboxes: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """Validate bounding boxes and remove invalid ones. Checks format, bounds;
    can remove empty or out-of-image boxes. Returns valid bboxes and mask.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (n, 4) where each row is [x_min, y_min, x_max, y_max].
        image_shape (Sequence[int]): Shape of the image as (height, width).

    Returns:
        np.ndarray: Array of valid bounding boxes, potentially with fewer boxes than the input.

    Examples:
        >>> bboxes = np.array([[10, 20, 30, 40], [-10, -10, 5, 5], [100, 100, 120, 120]])
        >>> valid_bboxes = validate_bboxes(bboxes, (100, 100))
        >>> print(valid_bboxes)
        [[10 20 30 40]]

    """
    rows, cols = image_shape[:2]

    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    valid_indices = (x_max > 0) & (y_max > 0) & (x_min < cols) & (y_min < rows)

    return bboxes[valid_indices]


def shift_bboxes(bboxes: np.ndarray, shift_vector: np.ndarray) -> np.ndarray:
    """Shift bounding boxes by a given (dx, dy) vector. Normalized or pixel;
    bbox_type hbb/obb. For crop/shift transforms. Keeps in bounds.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (n, m) where n is the number of bboxes
                             and m >= 4. The first 4 columns are [x_min, y_min, x_max, y_max].
        shift_vector (np.ndarray): Vector to shift the bounding boxes by, with shape (4,) for
                                   [shift_x, shift_y, shift_x, shift_y].

    Returns:
        np.ndarray: Shifted bounding boxes with the same shape as input.

    """
    # Create a copy of the input array to avoid modifying it in-place
    shifted_bboxes = bboxes.copy()

    # Add the shift vector to the first 4 columns
    shifted_bboxes[:, :4] += shift_vector

    return shifted_bboxes


def generate_reflected_bboxes(
    bboxes: np.ndarray,
    grid_dims: dict[str, tuple[int, int]],
    image_shape: tuple[int, int],
    center_in_origin: bool = False,
) -> np.ndarray:
    """Generate reflected bounding boxes for the entire reflection grid. From
    base bboxes and grid layout; for Mosaic and reflection-based crops.

    Args:
        bboxes (np.ndarray): Original bounding boxes.
        grid_dims (dict[str, tuple[int, int]]): Grid dimensions and original position.
        image_shape (tuple[int, int]): Shape of the original image as (height, width).
        center_in_origin (bool): If True, center the grid at the origin. Default is False.

    Returns:
        np.ndarray: Array of reflected and shifted bounding boxes for the entire grid.

    """
    rows, cols = image_shape[:2]
    grid_rows, grid_cols = grid_dims["grid_shape"]
    original_row, original_col = grid_dims["original_position"]

    # Prepare flipped versions of bboxes
    bboxes_hflipped = flip_bboxes(bboxes, flip_horizontal=True, image_shape=image_shape)
    bboxes_vflipped = flip_bboxes(bboxes, flip_vertical=True, image_shape=image_shape)
    bboxes_hvflipped = flip_bboxes(
        bboxes,
        flip_horizontal=True,
        flip_vertical=True,
        image_shape=image_shape,
    )

    # Shift all versions to the original position
    shift_vector = np.array(
        [
            original_col * cols,
            original_row * rows,
            original_col * cols,
            original_row * rows,
        ],
    )
    bboxes = shift_bboxes(bboxes, shift_vector)
    bboxes_hflipped = shift_bboxes(bboxes_hflipped, shift_vector)
    bboxes_vflipped = shift_bboxes(bboxes_vflipped, shift_vector)
    bboxes_hvflipped = shift_bboxes(bboxes_hvflipped, shift_vector)

    new_bboxes = []

    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            # Determine which version of bboxes to use based on grid position
            if (grid_row - original_row) % 2 == 0 and (grid_col - original_col) % 2 == 0:
                current_bboxes = bboxes
            elif (grid_row - original_row) % 2 == 0:
                current_bboxes = bboxes_hflipped
            elif (grid_col - original_col) % 2 == 0:
                current_bboxes = bboxes_vflipped
            else:
                current_bboxes = bboxes_hvflipped

            # Shift to the current grid cell
            cell_shift = np.array(
                [
                    (grid_col - original_col) * cols,
                    (grid_row - original_row) * rows,
                    (grid_col - original_col) * cols,
                    (grid_row - original_row) * rows,
                ],
            )
            shifted_bboxes = shift_bboxes(current_bboxes, cell_shift)

            new_bboxes.append(shifted_bboxes)

    result = np.vstack(new_bboxes)

    return shift_bboxes(result, -shift_vector) if center_in_origin else result


@handle_empty_array("bboxes")
def flip_bboxes(
    bboxes: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    image_shape: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Flip bounding boxes horizontally and/or vertically. direction: 'horizontal',
    'vertical', or 'both'. Normalized coords; hbb and obb. For flips.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (n, m) where each row is
            [x_min, y_min, x_max, y_max, ...].
        flip_horizontal (bool): Whether to flip horizontally.
        flip_vertical (bool): Whether to flip vertically.
        image_shape (tuple[int, int]): Shape of the image as (height, width).

    Returns:
        np.ndarray: Flipped bounding boxes.

    """
    rows, cols = image_shape[:2]
    flipped_bboxes = bboxes.copy()
    if flip_horizontal:
        flipped_bboxes[:, [0, 2]] = cols - flipped_bboxes[:, [2, 0]]
    if flip_vertical:
        flipped_bboxes[:, [1, 3]] = rows - flipped_bboxes[:, [3, 1]]
    return flipped_bboxes


@handle_empty_array("bboxes")
def bbox_distort_image(
    bboxes: np.ndarray,
    generated_mesh: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Distort bounding boxes based on a generated mesh. Each bbox warped per mesh
    cell; image_shape for clipping. For PiecewiseAffine with bboxes.

    This function applies a perspective transformation to each bounding box based on the provided generated mesh.
    It ensures that the bounding boxes are clipped to the image boundaries after transformation.

    Args:
        bboxes (np.ndarray): The bounding boxes to distort.
        generated_mesh (np.ndarray): The generated mesh to distort the bounding boxes with.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The distorted bounding boxes.

    """
    bboxes = bboxes.copy()
    masks = masks_from_bboxes(bboxes, image_shape)

    transformed_masks = cv2.merge(
        [distort_image(mask, generated_mesh, cv2.INTER_NEAREST) for mask in masks],
    )

    if transformed_masks.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        transformed_masks = transformed_masks.transpose(2, 0, 1)

    # Normalize the returned bboxes
    bboxes[:, :4] = bboxes_from_masks(transformed_masks)

    return bboxes


@handle_empty_array("bboxes")
def bboxes_piecewise_affine(
    bboxes: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Apply piecewise affine to bboxes via map_x, map_y. bbox->mask->remap->bbox.
    border_mode, image_shape. For PiecewiseAffine with bboxes.

    This function applies a piecewise affine transformation to the bounding boxes of an image.
    It first converts the bounding boxes to masks, then applies the transformation, and finally
    converts the transformed masks back to bounding boxes.

    Args:
        bboxes (np.ndarray): The bounding boxes to transform.
        map_x (np.ndarray): The x-coordinates of the transformation.
        map_y (np.ndarray): The y-coordinates of the transformation.
        border_mode (int): The border mode to use for the transformation.
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        np.ndarray: The transformed bounding boxes.

    """
    masks = masks_from_bboxes(bboxes, image_shape).transpose(1, 2, 0)

    transformed_masks = remap(
        masks,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        border_mode=border_mode,
        border_value=0,
    )

    if transformed_masks.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        transformed_masks = transformed_masks.transpose(2, 0, 1)

    # Normalize the returned bboxes
    bboxes[:, :4] = bboxes_from_masks(transformed_masks)

    return bboxes


def is_valid_component(
    component_area: float,
    original_area: float,
    min_area: float | None,
    min_visibility: float | None,
) -> bool:
    """Return True if component meets min_area and min_visibility. component_area,
    original_area; None thresholds pass. For GridShuffle bbox filtering.
    """
    visibility = component_area / original_area
    return (min_area is None or component_area >= min_area) and (min_visibility is None or visibility >= min_visibility)


@handle_empty_array("bboxes")
def bboxes_grid_shuffle(
    bboxes: np.ndarray,
    tiles: np.ndarray,
    mapping: list[int],
    image_shape: tuple[int, int],
    min_area: float,
    min_visibility: float,
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Shuffle bboxes according to grid tile mapping. bbox->mask->swap_tiles->components->bboxes.
    min_area, min_visibility, bbox_type. For GridShuffle with bboxes.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        tiles (np.ndarray): Array of grid tiles
        mapping (list[int]): Mapping of tile indices
        image_shape (tuple[int, int]): Shape of the image (height, width)
        min_area (float): Minimum area of a bounding box to keep
        min_visibility (float): Minimum visibility ratio of a bounding box to keep
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB is not supported here.

    Returns:
        np.ndarray: Shuffled bounding boxes

    """
    # Convert bboxes to masks
    masks = masks_from_bboxes(bboxes, image_shape)

    # Apply grid shuffle to each mask and handle split components
    all_component_masks = []
    extra_bbox_data = []  # Store additional bbox data for each component

    for idx, mask in enumerate(masks):
        original_area = float(reduce_sum(mask))  # Get original mask area

        # Shuffle the mask
        shuffled_mask = swap_tiles_on_image(mask, tiles, mapping)

        # Find connected components
        num_components, components = cv2.connectedComponents(
            shuffled_mask.astype(np.uint8),
        )

        # For each component, create a separate binary mask
        for comp_idx in range(1, num_components):  # Skip background (0)
            component_mask = (components == comp_idx).astype(np.uint8)

            # Calculate area and visibility ratio
            component_area = float(reduce_sum(component_mask))
            # Check if component meets minimum requirements
            if is_valid_component(
                component_area,
                original_area,
                min_area,
                min_visibility,
            ):
                all_component_masks.append(component_mask)
                # Append additional bbox data for this component
                if bboxes.shape[1] > NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS:
                    extra_bbox_data.append(bboxes[idx, 4:])

    # Convert all component masks to bboxes
    if all_component_masks:
        all_component_masks = np.array(all_component_masks)
        shuffled_bboxes = bboxes_from_masks(all_component_masks)

        # Add back additional bbox data if present
        if extra_bbox_data:
            extra_bbox_data = np.array(extra_bbox_data)
            return np.column_stack([shuffled_bboxes, extra_bbox_data])
    else:
        # Handle case where no valid components were found
        return np.zeros((0, bboxes.shape[1]), dtype=bboxes.dtype)

    return shuffled_bboxes


@handle_empty_array("bboxes")
def bboxes_morphology(
    bboxes: np.ndarray,
    kernel: np.ndarray,
    operation: Literal["dilation", "erosion"],
    image_shape: tuple[int, int],
    bbox_type: Literal["hbb", "obb"],
) -> np.ndarray:
    """Apply dilation or erosion to bboxes via mask. bbox->mask->morphology->bbox.
    kernel, operation, image_shape, bbox_type (hbb/obb). For BboxMorphology.

    This function applies morphology to bounding boxes by first converting the bounding
    boxes to a mask and then applying the morphology to the mask.

    Args:
        bboxes (np.ndarray): Bounding boxes as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.
        operation (Literal['dilation', 'erosion']): The operation to apply.
        image_shape (tuple[int, int]): The shape of the image.
        bbox_type (Literal['hbb', 'obb']): Bounding box type; OBB is not supported here.

    Returns:
        np.ndarray: The morphology applied to the bounding boxes.

    """
    bboxes = bboxes.copy()
    masks = masks_from_bboxes(bboxes, image_shape)
    transformed_masks = np.empty_like(masks)

    for index, mask in enumerate(masks):
        transformed_masks[index] = morphology(mask, kernel, operation)

    bboxes[:, :4] = bboxes_from_masks(transformed_masks)
    return bboxes


__all__ = [
    "_merge_obb_params",
    "_split_obb_params",
    "_split_polygons_and_extras",
    "bbox_distort_image",
    "bboxes_affine",
    "bboxes_affine_ellipse",
    "bboxes_affine_largest_box",
    "bboxes_d4",
    "bboxes_grid_shuffle",
    "bboxes_hflip",
    "bboxes_morphology",
    "bboxes_piecewise_affine",
    "bboxes_rot90",
    "bboxes_transpose",
    "bboxes_vflip",
    "flip_bboxes",
    "generate_reflected_bboxes",
    "is_valid_component",
    "pad_bboxes",
    "perspective_bboxes",
    "remap_bboxes",
    "resize_bboxes",
    "shift_bboxes",
    "validate_bboxes",
]
