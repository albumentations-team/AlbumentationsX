"""Keypoint geometric functional helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from ._functional_distortion import (
    generate_inverse_distortion_map,
)
from ._functional_images import (
    C4_GROUP_ELEMENT_TO_K,
    PAIR,
    ROT90_180_FACTOR,
    ROT90_270_FACTOR,
    apply_affine_to_points,
    calculate_affine_transform_padding,
    get_pad_grid_dimensions,
    is_identity_matrix,
    rotation2d_matrix_to_euler_angles,
)
from ._functional_shared import (
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    REFLECT_BORDER_MODES,
    angle_2pi_range,
    cv2,
    handle_empty_array,
    np,
    remap,
    warn,
)


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_rot90(
    keypoints: np.ndarray,
    group_element: Literal["e", "r90", "r180", "r270"],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rotate keypoints by 90° CCW a specified number of times. group_element: e,
    r90, r180, r270. Updates x, y, angle; image_shape for pixel coords.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
        group_element (Literal['e', 'r90', 'r180', 'r270']): C4 group element to apply.
        image_shape (tuple[int, int]): The shape of the image (height, width).

    Returns:
        np.ndarray: The rotated keypoints with the same shape as the input.

    """
    rot90_count = C4_GROUP_ELEMENT_TO_K[group_element]
    if rot90_count == 0:
        return keypoints

    height, width = image_shape[:2]
    rotated_keypoints = keypoints.copy().astype(np.float32)

    x, y, angle = keypoints[:, 0], keypoints[:, 1], keypoints[:, 3]

    if rot90_count == 1:
        rotated_keypoints[:, 0] = y
        rotated_keypoints[:, 1] = width - 1 - x
        rotated_keypoints[:, 3] = angle - np.pi / 2
    elif rot90_count == ROT90_180_FACTOR:
        rotated_keypoints[:, 0] = width - 1 - x
        rotated_keypoints[:, 1] = height - 1 - y
        rotated_keypoints[:, 3] = angle - np.pi
    elif rot90_count == ROT90_270_FACTOR:
        rotated_keypoints[:, 0] = height - 1 - y
        rotated_keypoints[:, 1] = x
        rotated_keypoints[:, 3] = angle + np.pi / 2

    return rotated_keypoints


@handle_empty_array("keypoints")
def keypoints_d4(
    keypoints: np.ndarray,
    group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
    image_shape: tuple[int, int],
    **params: Any,
) -> np.ndarray:
    """Apply D4 symmetry (rotations and reflections) to keypoints. group_member: e,
    r90, r180, r270, v, hvt, h, t. image_shape for pixel coords.

    This function adjusts a keypoint's coordinates according to the specified `D_4` group transformation,
    which includes rotations and reflections suitable for image processing tasks. These transformations account
    for the dimensions of the image to ensure the keypoint remains within its boundaries.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
        group_member (Literal['e', 'r90', 'r180', 'r270', 'v', 'hvt', 'h', 't']): A string identifier for
            the `D_4` group transformation to apply.
            Valid values are 'e', 'r90', 'r180', 'r270', 'v', 'hv', 'h', 't'.
        image_shape (tuple[int, int]): The shape of the image.
        params (Any): Not used.

    Returns:
        np.ndarray: The transformed keypoint.

    Raises:
        ValueError: If an invalid group member is specified, indicating that the specified transformation
            does not exist.

    """
    rows, cols = image_shape[:2]
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: keypoints_rot90(x, "r90", image_shape),  # Rotate 90 degrees
        "r180": lambda x: keypoints_rot90(x, "r180", image_shape),  # Rotate 180 degrees
        "r270": lambda x: keypoints_rot90(x, "r270", image_shape),  # Rotate 270 degrees
        "v": lambda x: keypoints_vflip(x, rows),  # Vertical flip
        "hvt": lambda x: keypoints_transpose(
            keypoints_rot90(x, "r180", image_shape),
        ),  # Reflect over anti diagonal
        "h": lambda x: keypoints_hflip(x, cols),  # Horizontal flip
        "t": keypoints_transpose,  # Transpose (reflect over main diagonal)
    }
    # Execute the appropriate transformation
    if group_member in transformations:
        return transformations[group_member](keypoints)

    raise ValueError(f"Invalid group member: {group_member}")


@handle_empty_array("keypoints")
def keypoints_scale(
    keypoints: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Scale keypoint x and y by scale_x and scale_y. Use when mapping keypoints after resize or
    crop. Angle and other extra columns are unchanged.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)
        scale_x (float): Scale factor for x coordinates
        scale_y (float): Scale factor for y coordinates

    Returns:
        np.ndarray: Scaled keypoints

    """
    # Extract x, y, z, angle, and scale
    x, y, z, angle, scale = (
        keypoints[:, 0],
        keypoints[:, 1],
        keypoints[:, 2],
        keypoints[:, 3],
        keypoints[:, 4],
    )

    # Scale x and y
    x_scaled = x * scale_x
    y_scaled = y * scale_y

    # Scale the keypoint scale by the maximum of scale_x and scale_y
    scale_scaled = scale * max(scale_x, scale_y)

    # Create the output array
    scaled_keypoints = np.column_stack([x_scaled, y_scaled, z, angle, scale_scaled])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        return np.column_stack(
            [scaled_keypoints, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]],
        )

    return scaled_keypoints


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_affine(
    keypoints: np.ndarray,
    matrix: np.ndarray,
    image_shape: tuple[int, int],
    scale: dict[str, float],
    border_mode: int,
) -> np.ndarray:
    """Apply affine transformation to keypoints. matrix, image_shape, scale dict,
    border_mode. Updates coordinates, angles, and scales; handles reflection.

    This function transforms keypoints using the given affine transformation matrix.
    It handles reflection padding if necessary, updates coordinates, angles, and scales.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 4+) where N is the number of keypoints.
                                Each keypoint is represented as [x, y, angle, scale, ...].
        matrix (np.ndarray): The 2x3 or 3x3 affine transformation matrix.
        image_shape (tuple[int, int]): Shape of the image (height, width).
        scale (dict[str, float]): Dictionary containing scale factors for x and y directions.
                                  Expected keys are 'x' and 'y'.
        border_mode (int): Border mode for handling keypoints near image edges.
                            Use cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT, etc.

    Returns:
        np.ndarray: Transformed keypoints array with the same shape as input.

    Notes:
        - The function applies reflection padding if the mode is in REFLECT_BORDER_MODES.
        - Coordinates (x, y) are transformed using the affine matrix.
        - Angles are adjusted based on the rotation component of the affine transformation.
        - Scales are multiplied by the maximum of x and y scale factors.
        - The @angle_2pi_range decorator ensures angles remain in the [0, 2π] range.

    Examples:
        >>> keypoints = np.array([[100, 100, 0, 1]])
        >>> matrix = np.array([[1.5, 0, 10], [0, 1.2, 20]])
        >>> scale = {'x': 1.5, 'y': 1.2}
        >>> transformed_keypoints = keypoints_affine(keypoints, matrix, (480, 640), scale, cv2.BORDER_REFLECT_101)

    """
    keypoints = keypoints.copy().astype(np.float32)

    if is_identity_matrix(matrix):
        return keypoints

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
        keypoints = generate_reflected_keypoints(
            keypoints,
            grid_dimensions,
            image_shape,
            center_in_origin=True,
        )

    # Extract x, y coordinates (z is preserved)
    xy = keypoints[:, :2]

    # Transform x, y coordinates (same code path as bboxes_affine OBB)
    xy_transformed = apply_affine_to_points(xy, matrix)

    # Calculate angle adjustment (always extract 2x2 rotation; 2x3 affine has rotation in [:2,:2])
    angle_adjustment = rotation2d_matrix_to_euler_angles(matrix[:2, :2], y_up=False)

    # Update angles (now at index 3)
    keypoints[:, 3] = keypoints[:, 3] + angle_adjustment

    # Update scales (now at index 4)
    max_scale = max(scale["x"], scale["y"])
    keypoints[:, 4] *= max_scale

    # Update x, y coordinates and preserve z
    keypoints[:, :2] = xy_transformed

    return keypoints


def to_distance_maps(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    inverted: bool = False,
) -> np.ndarray:
    """Generate (H,W,N) array of Euclidean distance maps to N keypoints.
    Helper for image-only augmentations that need keypoint info.

    Args:
        keypoints (np.ndarray): A numpy array of shape (N, 2+) where N is the number of keypoints.
                   Each row represents a keypoint's (x, y) coordinates.
        image_shape (tuple[int, int]): Shape of the image (height, width)
        inverted (bool): If `True`, inverted distance maps are returned where each
            distance value d is replaced by `d/(d+1)`, i.e. the distance
            maps have values in the range `(0.0, 1.0]` with `1.0` denoting
            exactly the position of the respective keypoint.

    Returns:
        np.ndarray: A float32 array of shape (H, W, N) containing `N` distance maps for `N`
            keypoints. Each location `(y, x, n)` in the array denotes the
            euclidean distance at `(y, x)` to the `n`-th keypoint.
            If `inverted` is `True`, the distance `d` is replaced
            by `d/(d+1)`. The height and width of the array match the
            height and width in `image_shape`.

    """
    height, width = image_shape[:2]
    if len(keypoints) == 0:
        return np.zeros((height, width, 0), dtype=np.float32)

    # Create coordinate grids
    yy, xx = np.mgrid[:height, :width]

    # Convert keypoints to numpy array
    keypoints_array = np.array(keypoints)

    # Compute distances for all keypoints at once
    distances = np.sqrt(
        (xx[..., np.newaxis] - keypoints_array[:, 0]) ** 2 + (yy[..., np.newaxis] - keypoints_array[:, 1]) ** 2,
    )

    if inverted:
        return (1 / (distances + 1)).astype(np.float32)
    return distances.astype(np.float32)


def validate_if_not_found_coords(
    if_not_found_coords: Sequence[int] | dict[str, Any] | None,
) -> tuple[bool, float, float]:
    """Validate and process if_not_found_coords parameter for keypoint transforms.
    Returns (fill_value, replace_mask). Raises on invalid input.
    """
    if if_not_found_coords is None:
        return True, -1, -1
    if isinstance(if_not_found_coords, (tuple, list)):
        if len(if_not_found_coords) != PAIR:
            msg = "Expected tuple/list 'if_not_found_coords' to contain exactly two entries."
            raise ValueError(msg)
        return False, if_not_found_coords[0], if_not_found_coords[1]
    if isinstance(if_not_found_coords, dict):
        return False, if_not_found_coords["x"], if_not_found_coords["y"]

    msg = "Expected if_not_found_coords to be None, tuple, list, or dict."
    raise ValueError(msg)


def from_distance_maps(
    distance_maps: np.ndarray,
    inverted: bool,
    if_not_found_coords: Sequence[int] | dict[str, Any] | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Convert distance maps (H, W, N) back to keypoint coordinates. Finds peaks;
    inverted=False: min distance = keypoint. Inverse of to_distance_maps.

    This function is the inverse of `to_distance_maps`. It takes distance maps generated for a set of keypoints
    and reconstructs the original keypoint coordinates. The function supports both regular and inverted distance maps,
    and can handle cases where keypoints are not found or fall outside a specified threshold.

    Args:
        distance_maps (np.ndarray): A 3D numpy array of shape (height, width, nb_keypoints) containing
            distance maps for each keypoint. Each channel represents the distance map for one keypoint.
        inverted (bool): If True, treats the distance maps as inverted (where higher values indicate
            closer proximity to keypoints). If False, treats them as regular distance maps (where lower
            values indicate closer proximity).
        if_not_found_coords (Sequence[int] | dict[str, Any] | None, optional): Coordinates to use for
            keypoints that are not found or fall outside the threshold. Can be:
            - None: Drop keypoints that are not found.
            - Sequence of two integers: Use these as (x, y) coordinates for not found keypoints.
            - Dict with 'x' and 'y' keys: Use these values for not found keypoints.
            Defaults to None.
        threshold (float | None, optional): A threshold value to determine valid keypoints. For inverted
            maps, values >= threshold are considered valid. For regular maps, values <= threshold are
            considered valid. If None, all keypoints are considered valid. Defaults to None.

    Returns:
        np.ndarray: A 2D numpy array of shape (nb_keypoints, 2) containing the (x, y) coordinates
        of the reconstructed keypoints. If `drop_if_not_found` is True (derived from if_not_found_coords),
        the output may have fewer rows than input keypoints.

    Raises:
        ValueError: If the input `distance_maps` is not a 3D array.

    Notes:
        - The function uses vectorized operations for improved performance, especially with large numbers of keypoints.
        - When `threshold` is None, all keypoints are considered valid, and `if_not_found_coords` is not used.
        - The function assumes that the input distance maps are properly normalized and scaled according to the
          original image dimensions.

    Examples:
        >>> distance_maps = np.random.rand(100, 100, 3)  # 3 keypoints
        >>> inverted = True
        >>> if_not_found_coords = [0, 0]
        >>> threshold = 0.5
        >>> keypoints = from_distance_maps(distance_maps, inverted, if_not_found_coords, threshold)
        >>> print(keypoints.shape)
        (3, 2)

    """
    if distance_maps.ndim != NUM_MULTI_CHANNEL_DIMENSIONS:
        msg = f"Expected three-dimensional input, got {distance_maps.ndim} dimensions and shape {distance_maps.shape}."
        raise ValueError(msg)
    height, width, nb_keypoints = distance_maps.shape

    drop_if_not_found, if_not_found_x, if_not_found_y = validate_if_not_found_coords(
        if_not_found_coords,
    )

    # Find the indices of max/min values for all keypoints at once
    if inverted:
        hitidx_flat = np.argmax(
            distance_maps.reshape(height * width, nb_keypoints),
            axis=0,
        )
    else:
        hitidx_flat = np.argmin(
            distance_maps.reshape(height * width, nb_keypoints),
            axis=0,
        )

    # Convert flat indices to 2D coordinates
    hitidx_y, hitidx_x = np.unravel_index(hitidx_flat, (height, width))

    # Create keypoints array
    keypoints = np.column_stack((hitidx_x, hitidx_y)).astype(float)

    if threshold is not None:
        # Check threshold condition
        if inverted:
            valid_mask = distance_maps[hitidx_y, hitidx_x, np.arange(nb_keypoints)] >= threshold
        else:
            valid_mask = distance_maps[hitidx_y, hitidx_x, np.arange(nb_keypoints)] <= threshold

        if not drop_if_not_found:
            # Replace invalid keypoints with if_not_found_coords
            keypoints[~valid_mask] = [if_not_found_x, if_not_found_y]
        else:
            # Keep only valid keypoints
            return keypoints[valid_mask]

    return keypoints


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_vflip(keypoints: np.ndarray, rows: int) -> np.ndarray:
    """Flip keypoints vertically. image_shape for pixel coords; y -> height-1-y.
    Angle and scale preserved. For VerticalFlip transform.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)
        rows (int): Number of rows in the image

    Returns:
        np.ndarray: Vertically flipped keypoints

    """
    flipped_keypoints = keypoints.copy().astype(np.float32)

    # Flip y-coordinates
    flipped_keypoints[:, 1] = (rows - 1) - keypoints[:, 1]

    # Negate angles
    flipped_keypoints[:, 3] = -keypoints[:, 3]

    return flipped_keypoints


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_hflip(keypoints: np.ndarray, cols: int) -> np.ndarray:
    """Flip keypoints horizontally. image_shape for pixel coords; x -> width-1-x.
    Angle and scale preserved. For HorizontalFlip.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)
        cols (int): Number of columns in the image

    Returns:
        np.ndarray: Horizontally flipped keypoints

    """
    flipped_keypoints = keypoints.copy().astype(np.float32)

    # Flip x-coordinates
    flipped_keypoints[:, 0] = (cols - 1) - keypoints[:, 0]

    # Adjust angles
    flipped_keypoints[:, 3] = np.pi - keypoints[:, 3]

    return flipped_keypoints


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_transpose(keypoints: np.ndarray) -> np.ndarray:
    """Transpose keypoints along the main diagonal. Swap x, y; image_shape for
    pixel coords. Angle updated. For Transpose transform.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)

    Returns:
        np.ndarray: Transposed keypoints

    """
    transposed_keypoints = keypoints.copy()

    # Swap x and y coordinates
    transposed_keypoints[:, [0, 1]] = keypoints[:, [1, 0]]

    # Adjust angles to reflect the coordinate swap
    angles = keypoints[:, 3]
    transposed_keypoints[:, 3] = np.where(
        angles <= np.pi,
        np.pi / 2 - angles,
        3 * np.pi / 2 - angles,
    )

    return transposed_keypoints


def remap_keypoints_via_mask(
    keypoints: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Remap keypoints using mask and cv2.remap. image_shape, mask (displacement);
    samples new (x,y) from map. For distortion transforms with keypoints.
    """
    height, width = image_shape[:2]

    # Handle empty keypoints array
    if len(keypoints) == 0:
        return np.zeros((0, 2 if keypoints.size == 0 else keypoints.shape[1]))

    # Create mask where each keypoint has unique index
    kp_mask = np.zeros((height, width), dtype=np.int16)
    for idx, kp in enumerate(keypoints, start=1):
        x, y = round(kp[0]), round(kp[1])
        if 0 <= x < width and 0 <= y < height:
            # Note: cv2.circle takes (x,y) coordinates
            cv2.circle(kp_mask, (x, y), 1, idx, -1)

    # Remap the mask
    transformed_kp_mask = remap(
        kp_mask,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
    )

    # Extract transformed keypoints
    new_points = []
    for idx, kp in enumerate(keypoints, start=1):
        # Find points with this index
        points = np.where(transformed_kp_mask == idx)
        if len(points[0]) > 0:
            # Convert back to (x,y) coordinates
            new_points.append(np.concatenate([[points[1][0], points[0][0]], kp[2:]]))

    return np.array(new_points) if new_points else np.zeros((0, keypoints.shape[1]))


@handle_empty_array("keypoints")
def remap_keypoints(
    keypoints: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Transform keypoints using coordinate mapping (map_x, map_y). Interpolates
    new (x, y) from maps; image_shape for bounds. For remap-based distortions.

    This function applies the inverse of the mapping defined by map_x and map_y
    to keypoint coordinates. The inverse mapping is necessary because the mapping
    functions define how pixels move from the source to the destination image,
    while keypoints need to be transformed from the destination back to the source.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 2+), where
            the first two columns are x and y coordinates.
        map_x (np.ndarray): Map of x-coordinates with shape equal to image_shape.
        map_y (np.ndarray): Map of y-coordinates with shape equal to image_shape.
        image_shape (tuple[int, int]): Shape (height, width) of the original image.

    Returns:
        np.ndarray: Transformed keypoints with the same shape as the input keypoints.
            Returns an empty array if input keypoints is empty.

    """
    height, width = image_shape[:2]

    # Extract x and y coordinates
    x, y = keypoints[:, 0], keypoints[:, 1]

    # Clip coordinates to image boundaries
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    # Convert to integer indices
    x_idx, y_idx = x.astype(int), y.astype(int)
    inv_map_x, inv_map_y = generate_inverse_distortion_map(map_x, map_y, image_shape[:2])
    # Apply the inverse mapping
    new_x = inv_map_x[y_idx, x_idx]
    new_y = inv_map_y[y_idx, x_idx]

    # Clip the new coordinates to ensure they're within the image bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)

    # Create the transformed keypoints array
    return np.column_stack([new_x, new_y, keypoints[:, 2:]])


@handle_empty_array("keypoints")
def distort_image_keypoints(
    keypoints: np.ndarray,
    generated_mesh: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Map keypoints through a piecewise-affine mesh; new (x,y) from mesh cells. Use with
    PiecewiseAffine. Angle and extra columns unchanged.

    This function applies a perspective transformation to each keypoint based on the provided generated mesh.
    It ensures that the keypoints are clipped to the image boundaries after transformation.

    Args:
        keypoints (np.ndarray): The keypoints to distort.
        generated_mesh (np.ndarray): The generated mesh to distort the keypoints with.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The distorted keypoints.

    """
    distorted_keypoints = keypoints.copy()
    height, width = image_shape[:2]

    for mesh in generated_mesh:
        x1, y1, x2, y2 = mesh[:4]  # Source rectangle
        dst_quad = mesh[4:].reshape(4, 2)  # Destination quadrilateral

        src_quad = np.array(
            [
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2],  # Bottom-left
            ],
            dtype=np.float32,
        )

        perspective_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)

        mask = (keypoints[:, 0] >= x1) & (keypoints[:, 0] < x2) & (keypoints[:, 1] >= y1) & (keypoints[:, 1] < y2)
        cell_keypoints = keypoints[mask]

        if len(cell_keypoints) > 0:
            # Convert to float32 before applying the transformation
            points_float32 = cell_keypoints[:, :2].astype(np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(
                points_float32,
                perspective_mat,
            ).reshape(-1, 2)

            # Update distorted keypoints
            distorted_keypoints[mask, :2] = transformed_points

    # Clip keypoints to image boundaries
    distorted_keypoints[:, 0] = np.clip(
        distorted_keypoints[:, 0],
        0,
        width - 1,
        out=distorted_keypoints[:, 0],
    )
    distorted_keypoints[:, 1] = np.clip(
        distorted_keypoints[:, 1],
        0,
        height - 1,
        out=distorted_keypoints[:, 1],
    )

    return distorted_keypoints


@handle_empty_array("keypoints")
def pad_keypoints(
    keypoints: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Pad keypoints by given pad_top, pad_bottom, pad_left, pad_right. border_mode
    and image_shape; reflection or shift. For Pad with keypoints.

    This function pads keypoints by a given amount.
    It handles both reflection and padding.

    Args:
        keypoints (np.ndarray): The keypoints to pad.
        pad_top (int): The amount to pad the top of the keypoints.
        pad_bottom (int): The amount to pad the bottom of the keypoints.
        pad_left (int): The amount to pad the left of the keypoints.
        pad_right (int): The amount to pad the right of the keypoints.
        border_mode (int): The border mode to use.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The padded keypoints.

    """
    if border_mode not in REFLECT_BORDER_MODES:
        shift_vector = np.array([pad_left, pad_top, 0])
        return shift_keypoints(keypoints, shift_vector)

    grid_dimensions = get_pad_grid_dimensions(
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        image_shape,
    )

    keypoints = generate_reflected_keypoints(keypoints, grid_dimensions, image_shape)

    rows, cols = image_shape[:2]

    # Calculate the number of grid cells added on each side
    original_row, original_col = grid_dimensions["original_position"]

    # Subtract the offset based on the number of added grid cells
    keypoints[:, 0] -= original_col * cols - pad_left  # x
    keypoints[:, 1] -= original_row * rows - pad_top  # y

    new_height = pad_top + pad_bottom + rows
    new_width = pad_left + pad_right + cols

    return validate_keypoints(keypoints, (new_height, new_width))


def validate_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Drop keypoints outside image bounds. image_shape (H,W). Keeps points with x in [0,W),
    y in [0,H). Use after transforms that may move points out of frame.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, M) where N is the number of keypoints
                                and M >= 2. The first two columns represent x and y coordinates.
        image_shape (tuple[int, int]): Shape of the image as (height, width).

    Returns:
        np.ndarray: Array of valid keypoints that fall within the image boundaries.

    Note:
        This function only checks the x and y coordinates (first two columns) of the keypoints.
        Any additional columns (e.g., angle, scale) are preserved for valid keypoints.

    """
    rows, cols = image_shape[:2]

    x, y = keypoints[:, 0], keypoints[:, 1]

    valid_indices = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)

    return keypoints[valid_indices]


def shift_keypoints(keypoints: np.ndarray, shift_vector: np.ndarray) -> np.ndarray:
    """Translate keypoints by shift_vector (dx, dy, dz). Use when mapping keypoints after crop or
    shift. Angle, scale, and other extra columns unchanged.

    This function shifts the keypoints by a given shift vector.
    It only shifts the x, y and z coordinates of the keypoints.

    Args:
        keypoints (np.ndarray): The keypoints to shift.
        shift_vector (np.ndarray): The shift vector to apply to the keypoints.

    Returns:
        np.ndarray: The shifted keypoints.

    """
    shifted_keypoints = keypoints.copy()
    shifted_keypoints[:, :3] += shift_vector[:3]  # Only shift x, y and z
    return shifted_keypoints


def generate_reflected_keypoints(
    keypoints: np.ndarray,
    grid_dims: dict[str, tuple[int, int]],
    image_shape: tuple[int, int],
    center_in_origin: bool = False,
) -> np.ndarray:
    """Generate reflected keypoints for the entire reflection grid. grid_dims,
    image_shape, center_in_origin. For Mosaic/reflection padding with keypoints.

    This function creates a grid of keypoints by reflecting and shifting the original keypoints.
    It handles both centered and non-centered grids based on the `center_in_origin` parameter.

    Args:
        keypoints (np.ndarray): Original keypoints array of shape (N, 4+), where N is the number of keypoints,
                                and each keypoint is represented by at least 4 values (x, y, angle, scale, ...).
        grid_dims (dict[str, tuple[int, int]]): A dictionary containing grid dimensions and original position.
            It should have the following keys:
            - "grid_shape": tuple[int, int] representing (grid_rows, grid_cols)
            - "original_position": tuple[int, int] representing (original_row, original_col)
        image_shape (tuple[int, int]): Shape of the original image as (height, width).
        center_in_origin (bool, optional): If True, center the grid at the origin. Default is False.

    Returns:
        np.ndarray: Array of reflected and shifted keypoints for the entire grid. The shape is
                    (N * grid_rows * grid_cols, 4+), where N is the number of original keypoints.

    Note:
        - The function handles keypoint flipping and shifting to create a grid of reflected keypoints.
        - It preserves the angle and scale information of the keypoints during transformations.
        - The resulting grid can be either centered at the origin or positioned based on the original grid.

    """
    grid_rows, grid_cols = grid_dims["grid_shape"]
    original_row, original_col = grid_dims["original_position"]

    # Prepare flipped versions of keypoints
    keypoints_hflipped = flip_keypoints(
        keypoints,
        flip_horizontal=True,
        image_shape=image_shape,
    )
    keypoints_vflipped = flip_keypoints(
        keypoints,
        flip_vertical=True,
        image_shape=image_shape,
    )
    keypoints_hvflipped = flip_keypoints(
        keypoints,
        flip_horizontal=True,
        flip_vertical=True,
        image_shape=image_shape,
    )

    rows, cols = image_shape[:2]

    # Shift all versions to the original position
    shift_vector = np.array(
        [original_col * cols, original_row * rows, 0, 0, 0],
    )  # Only shift x and y
    keypoints = shift_keypoints(keypoints, shift_vector)
    keypoints_hflipped = shift_keypoints(keypoints_hflipped, shift_vector)
    keypoints_vflipped = shift_keypoints(keypoints_vflipped, shift_vector)
    keypoints_hvflipped = shift_keypoints(keypoints_hvflipped, shift_vector)

    new_keypoints = []

    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            # Determine which version of keypoints to use based on grid position
            if (grid_row - original_row) % 2 == 0 and (grid_col - original_col) % 2 == 0:
                current_keypoints = keypoints
            elif (grid_row - original_row) % 2 == 0:
                current_keypoints = keypoints_hflipped
            elif (grid_col - original_col) % 2 == 0:
                current_keypoints = keypoints_vflipped
            else:
                current_keypoints = keypoints_hvflipped

            # Shift to the current grid cell
            cell_shift = np.array(
                [
                    (grid_col - original_col) * cols,
                    (grid_row - original_row) * rows,
                    0,
                    0,
                    0,
                ],
            )
            shifted_keypoints = shift_keypoints(current_keypoints, cell_shift)

            new_keypoints.append(shifted_keypoints)

    result = np.vstack(new_keypoints)

    return shift_keypoints(result, -shift_vector) if center_in_origin else result


@handle_empty_array("keypoints")
def flip_keypoints(
    keypoints: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    image_shape: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Flip keypoints horizontally or vertically. direction: 'horizontal' or
    'vertical'; image_shape for pixel coords. For HorizontalFlip/VerticalFlip.

    This function flips keypoints horizontally or vertically based on the provided parameters.
    It also flips the angle of the keypoints when flipping horizontally.

    Args:
        keypoints (np.ndarray): The keypoints to flip.
        flip_horizontal (bool): Whether to flip horizontally.
        flip_vertical (bool): Whether to flip vertically.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The flipped keypoints.

    """
    rows, cols = image_shape[:2]
    flipped_keypoints = keypoints.copy()
    if flip_horizontal:
        flipped_keypoints[:, 0] = cols - flipped_keypoints[:, 0]
        flipped_keypoints[:, 3] = -flipped_keypoints[:, 3]  # Flip angle
    if flip_vertical:
        flipped_keypoints[:, 1] = rows - flipped_keypoints[:, 1]
        flipped_keypoints[:, 3] = -flipped_keypoints[:, 3]  # Flip angle
    return flipped_keypoints


def swap_tiles_on_keypoints(
    keypoints: np.ndarray,
    tiles: np.ndarray,
    mapping: np.ndarray,
) -> np.ndarray:
    """Reposition keypoints by tile swap mapping. tiles (M, 4), mapping (M,).
    Keypoints in tile i move to tile mapping[i]. For GridShuffle.

    This function takes a set of keypoints and repositions them according to a mapping of tile swaps.
    Keypoints are moved from their original tiles to new positions in the swapped tiles.

    Args:
        keypoints (np.ndarray): A 2D numpy array of shape (N, 2) where N is the number of keypoints.
                                Each row represents a keypoint's (x, y) coordinates.
        tiles (np.ndarray): A 2D numpy array of shape (M, 4) where M is the number of tiles.
                            Each row represents a tile's (start_y, start_x, end_y, end_x) coordinates.
        mapping (np.ndarray): A 1D numpy array of shape (M,) where M is the number of tiles.
                              Each element i contains the index of the tile that tile i should be swapped with.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as the input keypoints, containing the new positions
                    of the keypoints after the tile swap.

    Raises:
        RuntimeWarning: If any keypoint is not found within any tile.

    Notes:
        - Keypoints that do not fall within any tile will remain unchanged.
        - The function assumes that the tiles do not overlap and cover the entire image space.

    """
    if not keypoints.size:
        return keypoints

    # Broadcast keypoints and tiles for vectorized comparison
    kp_x = keypoints[:, 0][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_y = keypoints[:, 1][:, np.newaxis]  # Shape: (num_keypoints, 1)

    start_y, start_x, end_y, end_x = tiles.T  # Each shape: (num_tiles,)

    # Check if each keypoint is inside each tile
    in_tile = (kp_y >= start_y) & (kp_y < end_y) & (kp_x >= start_x) & (kp_x < end_x)

    # Find which tile each keypoint belongs to
    tile_indices = np.argmax(in_tile, axis=1)

    # Check if any keypoint is not in any tile
    not_in_any_tile = ~np.any(in_tile, axis=1)
    if np.any(not_in_any_tile):
        warn(
            "Some keypoints are not in any tile. They will be returned unchanged. This is unexpected and should be "
            "investigated.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Get the new tile indices
    new_tile_indices = np.array(mapping)[tile_indices]

    # Calculate the offsets
    old_start_x = tiles[tile_indices, 1]
    old_start_y = tiles[tile_indices, 0]
    new_start_x = tiles[new_tile_indices, 1]
    new_start_y = tiles[new_tile_indices, 0]

    # Apply the transformation
    new_keypoints = keypoints.copy()
    new_keypoints[:, 0] = (keypoints[:, 0] - old_start_x) + new_start_x
    new_keypoints[:, 1] = (keypoints[:, 1] - old_start_y) + new_start_y

    # Keep original coordinates for keypoints not in any tile
    new_keypoints[not_in_any_tile] = keypoints[not_in_any_tile]

    return new_keypoints


__all__ = [
    "distort_image_keypoints",
    "flip_keypoints",
    "from_distance_maps",
    "generate_reflected_keypoints",
    "keypoints_affine",
    "keypoints_d4",
    "keypoints_hflip",
    "keypoints_rot90",
    "keypoints_scale",
    "keypoints_transpose",
    "keypoints_vflip",
    "pad_keypoints",
    "remap_keypoints",
    "remap_keypoints_via_mask",
    "shift_keypoints",
    "swap_tiles_on_keypoints",
    "to_distance_maps",
    "validate_if_not_found_coords",
    "validate_keypoints",
]
