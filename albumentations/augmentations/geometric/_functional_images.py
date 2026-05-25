"""Image, resize, affine, perspective, pad, flip, rotate, and morphology helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from operator import index
from typing import Literal, Protocol, cast

from ._functional_shared import (
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    ImageType,
    albucore_copy_make_border,
    albucore_resize,
    angle_2pi_range,
    cv2,
    from_float,
    handle_empty_array,
    hflip,
    lru_cache,
    math,
    np,
    os,
    preserve_channel_dim,
    to_float,
    vflip,
    warn,
    warp_perspective,
)

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import pyvips

    _PYVIPS_AVAILABLE = True
except ImportError:
    _PYVIPS_AVAILABLE = False

PAIR = 2

ROT90_180_FACTOR = 2

ROT90_270_FACTOR = 3

C4_GROUP_ELEMENT_TO_K: dict[str, int] = {"e": 0, "r90": 1, "r180": 2, "r270": 3}

_PadArray = Callable[..., np.ndarray]


class _VipsImage(Protocol):
    def resize(self, scale: float, *, vscale: float, kernel: object) -> _VipsImage:
        """Return a resized pyvips image using the selected scale, vertical scale, and interpolation
        kernel for the optional backend.
        """
        ...

    def numpy(self) -> np.ndarray:
        """Return image data as a NumPy array so the optional pyvips backend can rejoin the shared
        image array pipeline cleanly after resize.
        """
        ...


@lru_cache(maxsize=1)
def _get_resize_backend() -> str:
    env_backend = os.environ.get("ALBUMENTATIONS_RESIZE", "opencv").lower()
    if env_backend == "pyvips" and _PYVIPS_AVAILABLE:
        return env_backend
    if env_backend == "pillow" and _PIL_AVAILABLE:
        return env_backend
    return "opencv"


def resize(
    img: ImageType,
    target_shape: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    """Resize an image to the specified target shape using the backend
    chosen via the ALBUMENTATIONS_RESIZE environment variable.

    If the image is already the target size, it is returned unchanged.

    Args:
        img (ImageType): Input image.
        target_shape (tuple[int, int]): Target (height, width) dimensions.
        interpolation (int): Interpolation method.

    Returns:
        np.ndarray: Resized image with shape target_shape + original channel dimensions.

    Raises:
        NotImplementedError: If the selected backend is not supported.

    """
    if target_shape == img.shape[:2]:
        return img

    height, width = target_shape
    if img.ndim == 2:
        return albucore_resize(img[:, :, np.newaxis], (width, height), interpolation=interpolation)[:, :, 0]

    backend = _get_resize_backend()
    if backend == "opencv":
        return albucore_resize(img, (width, height), interpolation=interpolation)
    if backend == "pyvips":
        return resize_pyvips(img, target_shape, interpolation)
    if backend == "pillow":
        return resize_pil(img, target_shape, interpolation)

    raise NotImplementedError(f"The provided backend '{backend}' is not supported yet.")


def resize_pyvips(
    img: ImageType,
    target_shape: tuple[int, int],
    interpolation: int = 1,
) -> np.ndarray:
    """Resize an image to target shape using pyvips. Params: target_shape,
    interpolation (0=nearest, 1=bilinear, 2=bicubic). Returns same dtype.

    This function resizes an input image to the target shape using the specified interpolation method.

    Args:
        img (ImageType): The input image as a NumPy array.
        target_shape (tuple[int, int]): The desired output shape (height, width).
        interpolation (int): The interpolation method to use.
            0: Nearest-neighbor
            1: Bilinear
            2: Bicubic

    Returns:
        np.ndarray: The resized image as a NumPy array with the original dtype.

    """
    # At this stage, the library's installation and importability have already been verified.

    height, width = img.shape[:2]
    target_height, target_width = target_shape
    original_dtype = img.dtype

    img_vips = cast("_VipsImage", pyvips.Image.new_from_array(img))

    scale_x = target_width / width
    scale_y = target_height / height

    interpolation_map = {
        0: pyvips.Kernel.NEAREST,
        1: pyvips.Kernel.LINEAR,
        2: pyvips.Kernel.CUBIC,
    }
    interpolation_method = interpolation_map.get(interpolation)
    if interpolation_method is None:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")

    resized_img_vips = img_vips.resize(
        scale_x,
        vscale=scale_y,
        kernel=interpolation_method,
    )

    return resized_img_vips.numpy().astype(original_dtype)


def resize_pil(
    img: ImageType,
    target_shape: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    """Resize an image using PIL. target_shape (H, W), interpolation (cv2 flag
    mapped to PIL). Handles grayscale, RGB, RGBA, and multi-channel.

    This function resizes an input image to the target shape using the specified interpolation method.

    Args:
        img (ImageType): The input image as a NumPy array.
        target_shape (tuple[int, int]): The desired output shape (height, width).
        interpolation (int): The cv2 interpolation flag that will be mapped to PIL interpolation.
            Maps cv2 constants to PIL.Image.Resampling constants.

    Returns:
        np.ndarray: The resized image as a NumPy array.

    """
    target_height, target_width = target_shape
    original_dtype = img.dtype

    # PIL doesn't support float32 RGB images, convert to uint8 if needed
    needs_conversion = img.dtype == np.float32
    if needs_conversion:
        img = from_float(cast("np.ndarray", img), target_dtype=np.dtype(np.uint8))

    # Map cv2 interpolation constants to PIL.Image.Resampling constants
    cv2_to_pil_interpolation = {
        cv2.INTER_NEAREST: Image.Resampling.NEAREST,
        cv2.INTER_NEAREST_EXACT: Image.Resampling.NEAREST,  # PIL doesn't have exact variant
        cv2.INTER_LINEAR: Image.Resampling.BILINEAR,
        cv2.INTER_CUBIC: Image.Resampling.BICUBIC,
        cv2.INTER_AREA: Image.Resampling.BOX,  # BOX is similar to INTER_AREA for downscaling
        cv2.INTER_LANCZOS4: Image.Resampling.LANCZOS,
        cv2.INTER_LINEAR_EXACT: Image.Resampling.BILINEAR,  # PIL doesn't have exact variant
    }

    pil_interpolation = cv2_to_pil_interpolation.get(interpolation)
    if pil_interpolation is None:
        # Fallback to BILINEAR for unknown interpolation methods
        warn(f"Interpolation method {interpolation} is not supported by PIL backend, using BILINEAR", stacklevel=2)
        pil_interpolation = Image.Resampling.BILINEAR

    # Convert numpy array to PIL Image
    # Images always have ndim=3 in albumentations
    if img.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape: {img.shape}")

    num_channels = img.shape[2]

    if num_channels == 1:
        # Grayscale image (H, W, 1) -> squeeze to (H, W) for PIL
        pil_img = Image.fromarray(img[:, :, 0], mode="L")
        resized_pil_img = pil_img.resize(
            (target_width, target_height),
            resample=pil_interpolation,
        )
        # Convert back to (H, W, 1)
        result = np.array(resized_pil_img)[:, :, np.newaxis]
    elif num_channels == 3:
        # RGB image
        pil_img = Image.fromarray(img, mode="RGB")
        resized_pil_img = pil_img.resize(
            (target_width, target_height),
            resample=pil_interpolation,
        )
        result = np.array(resized_pil_img)
    elif num_channels == 4:
        # RGBA image
        pil_img = Image.fromarray(img, mode="RGBA")
        resized_pil_img = pil_img.resize(
            (target_width, target_height),
            resample=pil_interpolation,
        )
        result = np.array(resized_pil_img)
    else:
        # For other channel counts, process each channel separately
        channels = []
        for i in range(num_channels):
            channel_img = Image.fromarray(img[:, :, i], mode="L")
            resized_channel = channel_img.resize(
                (target_width, target_height),
                resample=pil_interpolation,
            )
            channels.append(np.array(resized_channel))
        result = np.stack(channels, axis=-1)

    # Convert back to original dtype
    if needs_conversion:
        result = to_float(result)
    elif result.dtype != original_dtype:
        result = result.astype(original_dtype)

    return result


@preserve_channel_dim
def scale(img: ImageType, scale: float, interpolation: int) -> ImageType:
    """Scale an image by a factor while preserving aspect ratio. scale > 1
    enlarges, scale < 1 shrinks. interpolation: cv2 flag. Calls resize internally.

    This function scales both height and width dimensions of the image by the same factor.

    Args:
        img (ImageType): Input image to scale.
        scale (float): Scale factor. Values > 1 will enlarge the image, values < 1 will shrink it.
        interpolation (int): Interpolation method to use (cv2 interpolation flag).

    Returns:
        ImageType: Scaled image.

    """
    height, width = img.shape[:2]
    new_size = int(height * scale), int(width * scale)
    return resize(img, new_size, interpolation)


@preserve_channel_dim
def perspective(
    img: ImageType,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    border_val: float | tuple[float, ...] | np.ndarray | None,
    border_mode: int,
    keep_size: bool,
    interpolation: int,
) -> np.ndarray:
    """Apply perspective transformation to an image. matrix (3x3), interpolation,
    border_mode. Same shape or keep_size. For Perspective transform.

    This function warps an image according to a perspective transformation matrix.
    It can either maintain the original dimensions or use the specified max dimensions.

    Args:
        img (ImageType): Input image to transform.
        matrix (np.ndarray): 3x3 perspective transformation matrix.
        max_width (int): Maximum width of the output image if keep_size is False.
        max_height (int): Maximum height of the output image if keep_size is False.
        border_val (float | tuple[float, ...] | np.ndarray | None): Border value(s) for transformed borders.
        border_mode (int): OpenCV border mode (e.g., cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT).
        keep_size (bool): If True, maintain the original image dimensions.
        interpolation (int): Interpolation method for resampling (cv2 interpolation flag).

    Returns:
        np.ndarray: Perspective-transformed image.

    """
    return perspective_images(
        np.expand_dims(img, axis=0),
        matrix,
        max_width,
        max_height,
        border_val,
        border_mode,
        keep_size,
        interpolation,
    )[0]


def perspective_images(
    images: np.ndarray,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    border_val: float | tuple[float, ...] | np.ndarray | None,
    border_mode: int,
    keep_size: bool,
    interpolation: int,
) -> np.ndarray:
    """Apply perspective transformation to a batch of images (N, H, W, C). matrix,
    keep_size, border_val, interpolation. Single warp when grayscale and small.

    Args:
        images (np.ndarray): Batch of images of shape (N, H, W, C).
        matrix (np.ndarray): 3x3 perspective transformation matrix.
        max_width (int): Maximum width of the output image if keep_size is False.
        max_height (int): Maximum height of the output image if keep_size is False.
        border_val (float | tuple[float, ...] | np.ndarray | None): Border value(s) for transformed borders.
        border_mode (int): OpenCV border mode (e.g., cv2.BORDER_CONSTANT).
        keep_size (bool): If True, maintain the original image dimensions.
        interpolation (int): Interpolation method for resampling (cv2 interpolation flag).

    Returns:
        np.ndarray: Batch of perspective-transformed images with the same shape as input
        when keep_size is True, or (N, max_height, max_width, C) when False.

    """
    height, width = images.shape[1], images.shape[2]
    n = images.shape[0]
    num_channels = 1 if images.ndim == 3 else images.shape[3]

    if keep_size:
        adjusted_matrix = np.array([[width / max_width, 0, 0], [0, height / max_height, 0], [0, 0, 1]]) @ matrix
        dsize = (width, height)
    else:
        adjusted_matrix = matrix
        dsize = (max_width, max_height)

    if num_channels == 1:
        # Small images: stack N frames → (H,W,N), one warp call (albucore C++ chunks avoid N crossings).
        # Large images: per-frame warp (transpose copy cost > per-call savings).
        _stack_px = 256 * 256
        flat = images if images.ndim == 3 else images[:, :, :, 0]  # (N,H,W)
        if height * width <= _stack_px:
            stacked = np.ascontiguousarray(flat.transpose(1, 2, 0))  # (H,W,N)
            border_scalar = border_val[0] if isinstance(border_val, (tuple, np.ndarray)) else border_val
            warped = warp_perspective(
                stacked,
                adjusted_matrix,
                dsize,
                flags=interpolation,
                border_mode=border_mode,
                border_value=border_scalar,
            )
            out = np.moveaxis(warped, -1, 0)  # view (N,H',W')
        else:
            out = np.empty((n, dsize[1], dsize[0]), dtype=images.dtype)
            for i in range(n):
                warp_perspective(
                    flat[i],
                    adjusted_matrix,
                    dsize,
                    flags=interpolation,
                    border_mode=border_mode,
                    border_value=border_val,
                    dst=out[i],
                )
        return out[:, :, :, np.newaxis] if images.ndim == 4 else out

    result = np.empty((n, dsize[1], dsize[0], *images.shape[3:]), dtype=images.dtype)
    for i in range(n):
        warp_perspective(
            images[i],
            adjusted_matrix,
            dsize,
            flags=interpolation,
            border_mode=border_mode,
            border_value=border_val,
            dst=result[i],
        )
    return result


def rotation2d_matrix_to_euler_angles(matrix: np.ndarray, y_up: bool) -> float:
    """Extract rotation angle from 2D rotation matrix. y_up: True if Y axis points
    up. Returns angle in radians. For perspective_keypoints angle update.

    Args:
        matrix (np.ndarray): 2x2 or 3x3 rotation matrix.
        y_up (bool): True if Y axis points up.

    Returns:
        float: Rotation angle in radians.

    """
    if y_up:
        return np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.arctan2(-matrix[1, 0], matrix[0, 0])


@handle_empty_array("keypoints")
@angle_2pi_range
def perspective_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> np.ndarray:
    """Apply perspective transformation to keypoints. matrix, image_shape,
    max_width, max_height, keep_size. Updates x, y, angle, scale.

    Args:
        keypoints (np.ndarray): Array of shape (N, 5+) in format [x, y, z, angle, scale, ...]
        image_shape (tuple[int, int]): Original image shape (height, width)
        matrix (np.ndarray): 3x3 perspective transformation matrix
        max_width (int): Maximum width after transformation
        max_height (int): Maximum height after transformation
        keep_size (bool): Whether to keep original size

    Returns:
        np.ndarray: Transformed keypoints array with same shape as input

    """
    keypoints = keypoints.copy().astype(np.float32)

    height, width = image_shape[:2]

    x, y, z, angle, scale = (
        keypoints[:, 0],
        keypoints[:, 1],
        keypoints[:, 2],
        keypoints[:, 3],
        keypoints[:, 4],
    )

    # Reshape keypoints for perspective transform
    keypoint_vector = np.column_stack((x, y)).astype(np.float32).reshape(-1, 1, 2)

    # Apply perspective transform
    transformed_points = cv2.perspectiveTransform(keypoint_vector, matrix).squeeze()

    # Unsqueeze if we have a single keypoint
    if transformed_points.ndim == 1:
        transformed_points = transformed_points[np.newaxis, :]

    x, y = transformed_points[:, 0], transformed_points[:, 1]

    # Update angles
    angle += rotation2d_matrix_to_euler_angles(matrix[:2, :2], y_up=True)

    # Calculate scale factors
    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale *= max(scale_x, scale_y)

    if keep_size:
        scale_x = width / max_width
        scale_y = height / max_height
        x *= scale_x
        y *= scale_y
        scale *= max(scale_x, scale_y)

    # Create the output array with unchanged z coordinate
    transformed_keypoints = np.column_stack([x, y, z, angle, scale])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        return np.column_stack(
            [
                transformed_keypoints,
                keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:],
            ],
        )

    return transformed_keypoints


def is_identity_matrix(matrix: np.ndarray) -> bool:
    """Check if the given matrix is an identity matrix (3x3). For skipping no-op
    affine. Returns True if np.allclose(matrix, eye(3)).

    Args:
        matrix (np.ndarray): A 3x3 affine transformation matrix.

    Returns:
        bool: True if the matrix is an identity matrix, False otherwise.

    """
    return np.allclose(matrix, np.eye(3, dtype=matrix.dtype))


@handle_empty_array("points")
def apply_affine_to_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply affine transformation to a set of (x, y) points. matrix (2x3 or 3x3);
    points shape (N, 2). Returns transformed points.

    This function handles potential division by zero by replacing zero values
    in the homogeneous coordinate with a small epsilon value.

    Args:
        points (np.ndarray): Array of points with shape (N, 2).
        matrix (np.ndarray): 3x3 affine transformation matrix.

    Returns:
        np.ndarray: Transformed points with shape (N, 2).

    """
    homogeneous_points = np.column_stack([points, np.ones(points.shape[0])])
    transformed_points = homogeneous_points @ matrix.T

    # Handle potential division by zero
    epsilon = np.finfo(transformed_points.dtype).eps
    transformed_points[:, 2] = np.where(
        np.abs(transformed_points[:, 2]) < epsilon,
        np.sign(transformed_points[:, 2]) * epsilon,
        transformed_points[:, 2],
    )

    return transformed_points[:, :2] / transformed_points[:, 2:]


def calculate_affine_transform_padding(
    matrix: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Calculate padding for affine transformation to avoid empty/cropped regions.
    Returns (pad_top, pad_bottom, pad_left, pad_right) from inverse affine corners.
    """
    height, width = image_shape[:2]

    # Check for identity transform
    if is_identity_matrix(matrix):
        return (0, 0, 0, 0)

    # Original corners
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # Transform corners
    transformed_corners = apply_affine_to_points(corners, matrix)

    # Ensure transformed_corners is 2D
    transformed_corners = transformed_corners.reshape(-1, 2)

    # Find box that includes both original and transformed corners
    all_corners = np.vstack((corners, transformed_corners))
    min_x, min_y = all_corners.min(axis=0)
    max_x, max_y = all_corners.max(axis=0)

    # Compute the inverse transform
    inverse_matrix = np.linalg.inv(matrix)

    # Apply inverse transform to all corners of the bounding box
    bbox_corners = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
    )
    inverse_corners = apply_affine_to_points(bbox_corners, inverse_matrix).reshape(
        -1,
        2,
    )

    min_x, min_y = inverse_corners.min(axis=0)
    max_x, max_y = inverse_corners.max(axis=0)

    pad_left = max(0, math.ceil(0 - min_x))
    pad_right = max(0, math.ceil(max_x - width))
    pad_top = max(0, math.ceil(0 - min_y))
    pad_bottom = max(0, math.ceil(max_y - height))

    return pad_left, pad_right, pad_top, pad_bottom


D4_GROUP_ELEMENTS = ["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]


def d4(img: ImageType, group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]) -> ImageType:
    """Apply D4 symmetry (rotations and reflections) to an image. group_member:
    e, r90, r180, r270, v, hvt, h, t. Square input; same shape output.

    This function manipulates an image using transformations such as rotations and flips,
    corresponding to the `D_4` dihedral group symmetry operations.
    Each transformation is identified by a unique group member code.

    Args:
        img (ImageType): The input image array to transform.
        group_member (Literal['e', 'r90', 'r180', 'r270', 'v', 'hvt', 'h', 't']): A string identifier indicating
            the specific transformation to apply. Valid codes include:
            - 'e': Identity (no transformation).
            - 'r90': Rotate 90 degrees counterclockwise.
            - 'r180': Rotate 180 degrees.
            - 'r270': Rotate 270 degrees counterclockwise.
            - 'v': Vertical flip.
            - 'hvt': Transpose over second diagonal
            - 'h': Horizontal flip.
            - 't': Transpose (reflect over the main diagonal).

    Returns:
        ImageType: The transformed image array.

    """
    # Execute the appropriate transformation
    return D4_TRANSFORMATIONS[group_member](img)


def transpose(img: ImageType) -> ImageType:
    """Transpose the first two dimensions (H, W) of an array. (H, W, ...) -> (W, H, ...).
    Retains the order of any additional dimensions. For image transpose.

    Args:
        img (ImageType): Input array.

    Returns:
        ImageType: Transposed array.

    """
    num_channels = img.shape[-1]
    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and num_channels in {1, 3, 4}:
        result = cast("ImageType", cv2.transpose(img if num_channels != 1 else img[..., 0]))
        return result[..., None] if num_channels == 1 else result

    # Generate the new axes order
    new_axes = list(range(img.ndim))
    new_axes[0], new_axes[1] = 1, 0  # Swap the first two dimensions

    # Transpose the array using the new axes order
    return img.transpose(new_axes)


D4_TRANSFORMATIONS = {
    "e": lambda x: x,  # Identity transformation
    "r90": lambda x: rot90(x, "r90"),  # Rotate 90 degrees
    "r180": lambda x: rot90(x, "r180"),  # Rotate 180 degrees
    "r270": lambda x: rot90(x, "r270"),  # Rotate 270 degrees
    "v": vflip,  # Vertical flip
    "hvt": lambda x: transpose(rot90(x, "r180")),  # Reflect over anti-diagonal
    "h": hflip,  # Horizontal flip
    "t": transpose,  # Transpose (reflect over main diagonal)
}


def transpose_images(images: ImageType) -> ImageType:
    """Transpose a batch of images (N, H, W, C). Swaps H and W per image.
    Same as transpose on each image along axes 0, 1. Returns same shape.

    Args:
        images (ImageType): Batch of images to transpose with shape:
            - (N, H, W) for grayscale images
            - (N, H, W, C) for multi-channel images
            where N is the batch size, H is height, W is width, C is channels

    Returns:
        ImageType: Transposed batch of images with shape:
            - (N, W, H) for grayscale images
            - (N, W, H, C) for multi-channel images

    """
    # Generate the new axes order
    new_axes = list(range(images.ndim))
    # Swap dimensions 1 and 2 (Height and Width), preserving batch dimension and channels
    new_axes[1], new_axes[2] = 2, 1

    # Transpose the array using the new axes order
    return images.transpose(new_axes)


def transpose_volumes(volumes: np.ndarray) -> np.ndarray:
    """Transpose a batch of volumes (N, D, H, W, C). Swaps D and H per volume.
    Same as transpose on each volume along axes 0, 1.

    Args:
        volumes (np.ndarray): Batch of volumes to transpose with shape:
            - (N, D, H, W) for grayscale volumes
            - (N, D, H, W, C) for multi-channel volumes
            where N is the batch size, D is depth, H is height, W is width, C is channels

    Returns:
        np.ndarray: Transposed batch of volumes with shape:
            - (N, D, W, H) for grayscale volumes
            - (N, D, W, H, C) for multi-channel volumes

    """
    # Generate the new axes order
    new_axes = list(range(volumes.ndim))
    # Swap dimensions 2 and 3 (Height and Width), preserving batch, depth and channels
    new_axes[2], new_axes[3] = 3, 2

    # Transpose the array using the new axes order
    return volumes.transpose(new_axes)


def rot90(img: ImageType, group_element: Literal["e", "r90", "r180", "r270"]) -> ImageType:
    """Rotate image 90° counterclockwise. group_element: e, r90, r180, r270. Same as np.rot90.
    Use for D4-style augmentation. Same dtype and shape.

    Args:
        img (ImageType): The input image to rotate.
        group_element (Literal['e', 'r90', 'r180', 'r270']): C4 group element to apply.

    Returns:
        ImageType: The rotated image.

    """
    rot90_count = C4_GROUP_ELEMENT_TO_K[group_element]
    if rot90_count == 0:
        return img

    num_channels = img.shape[-1]
    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and num_channels in {1, 3, 4}:
        if rot90_count == 1:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rot90_count == 2:
            rotate_code = cv2.ROTATE_180
        else:
            rotate_code = cv2.ROTATE_90_CLOCKWISE

        cv2_input = img if num_channels != 1 else img[..., 0]
        result = cast("ImageType", cv2.rotate(cv2_input, rotate_code))
        return result[..., None] if num_channels == 1 else result

    return cast("ImageType", np.rot90(img, rot90_count))


def rot90_images(images: ImageType, group_element: Literal["e", "r90", "r180", "r270"]) -> ImageType:
    """Rotate a batch of images 90° CCW. k per image or single k. Same as rot90
    on each image. Shape (N, H, W, C) preserved. Returns same dtype.

    Args:
        images (ImageType): Batch of images to rotate with shape:
            - (N, H, W) for grayscale images
            - (N, H, W, C) for multi-channel images
            where N is the batch size, H is height, W is width, C is channels
        group_element (Literal['e', 'r90', 'r180', 'r270']): C4 group element to apply.

    Returns:
        ImageType: Rotated batch of images with shape:
            - (N, W, H) for grayscale images when group_element is r90 or r270
            - (N, H, W) for grayscale images when group_element is e or r180
            - (N, W, H, C) for multi-channel images when group_element is r90 or r270
            - (N, H, W, C) for multi-channel images when group_element is e or r180

    """
    rot90_count = C4_GROUP_ELEMENT_TO_K[group_element]
    return cast("ImageType", np.rot90(images, k=rot90_count, axes=(1, 2)))


@preserve_channel_dim
def pad(
    img: ImageType,
    min_height: int,
    min_width: int,
    border_mode: int,
    value: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Pad an image to ensure minimum height and width. Params: min_height,
    min_width, border_mode, fill. Pads on right/bottom if needed.

    This function adds padding to an image if its dimensions are smaller than
    the specified minimum dimensions. Padding is added evenly on all sides.

    Args:
        img (ImageType): Input image to pad.
        min_height (int): Minimum height of the output image.
        min_width (int): Minimum width of the output image.
        border_mode (int): OpenCV border mode for padding.
        value (tuple[float, ...] | float | None): Value(s) to fill the border pixels.

    Returns:
        np.ndarray: Padded image with dimensions at least (min_height, min_width).

    """
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = pad_with_params(
        img,
        h_pad_top,
        h_pad_bottom,
        w_pad_left,
        w_pad_right,
        border_mode,
        value,
    )

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(
            f"Invalid result shape. Got: {img.shape[:2]}. Expected: {(max(min_height, height), max(min_width, width))}",
        )

    return img


@preserve_channel_dim
def pad_with_params(
    img: ImageType,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    border_mode: int,
    value: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Pad an image with explicit padding per side. Params: pad_top, pad_bottom,
    pad_left, pad_right, border_mode, fill. For Pad/PadIfNeeded.

    This function adds specified amounts of padding to each side of the image.

    Args:
        img (ImageType): Input image to pad.
        h_pad_top (int): Number of pixels to add at the top.
        h_pad_bottom (int): Number of pixels to add at the bottom.
        w_pad_left (int): Number of pixels to add on the left.
        w_pad_right (int): Number of pixels to add on the right.
        border_mode (int): OpenCV border mode for padding.
        value (tuple[float, ...] | float | None): Value(s) to fill the border pixels.

    Returns:
        np.ndarray: Padded image.

    """
    # For 0-channel images, return empty array of correct padded size
    if img.size == 0:
        height, width = img.shape[:2]
        return np.zeros(
            (height + h_pad_top + h_pad_bottom, width + w_pad_left + w_pad_right, 0),
            dtype=img.dtype,
        )

    num_channels = img.shape[-1] if img.ndim >= 3 else 1
    if value is not None and border_mode == cv2.BORDER_CONSTANT:
        if isinstance(value, (int, float)):
            value = (float(value),) * min(num_channels, 4)
        elif isinstance(value, (tuple, list)) and len(value) < num_channels:
            # Extend to match channels; use scalar for >4ch to avoid albucore chunked path
            val_list = list(value)
            if num_channels <= 4:
                last = val_list[-1] if val_list else 0.0
                value = tuple(val_list) + (last,) * (num_channels - len(val_list))
            else:
                value = (val_list[0],) * 4

    return albucore_copy_make_border(
        img,
        top=h_pad_top,
        bottom=h_pad_bottom,
        left=w_pad_left,
        right=w_pad_right,
        border_type=border_mode,
        value=value if value is not None else 0,
    )


def pad_images_with_params(
    images: ImageType,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    border_mode: int,
    value: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Pad a batch of images (N, H, W, C) with explicit padding per side. Same
    params as pad_with_params; applies to each image.

    This function adds specified amounts of padding to each side of the image for each
    image in the batch.

    Args:
        images (ImageType): Input batch of images to pad.
        h_pad_top (int): Number of pixels to add at the top.
        h_pad_bottom (int): Number of pixels to add at the bottom.
        w_pad_left (int): Number of pixels to add on the left.
        w_pad_right (int): Number of pixels to add on the right.
        border_mode (int): OpenCV border mode for padding.
        value (tuple[float, ...] | float | None): Value(s) to fill the border pixels.

    Returns:
        np.ndarray: Padded batch of images.

    """
    no_channel_dim = images.ndim == 3
    if no_channel_dim:
        images = images[..., np.newaxis]

    cv2np_border_modes = {
        cv2.BORDER_CONSTANT: "constant",
        cv2.BORDER_REPLICATE: "edge",
        cv2.BORDER_REFLECT: "symmetric",
        cv2.BORDER_WRAP: "wrap",
        cv2.BORDER_REFLECT_101: "reflect",
        cv2.BORDER_REFLECT101: "reflect",
        cv2.BORDER_DEFAULT: "reflect",  # same as cv2.BORDER_REFLECT_101
    }
    mode = cv2np_border_modes[border_mode]

    pad_width = ((0, 0), (h_pad_top, h_pad_bottom), (w_pad_left, w_pad_right), (0, 0))
    if mode == "constant":
        constant_values = np.array(((0, 0), (value, value), (value, value), (0, 0)), dtype=object)
        kwargs = {"constant_values": constant_values}
    else:
        kwargs = {}

    pad_array = cast("_PadArray", np.pad)
    images = cast("ImageType", pad_array(images, pad_width=pad_width, mode=mode, **kwargs))
    if no_channel_dim:
        images = images[..., 0]

    return images


def get_pad_grid_dimensions(
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    image_shape: tuple[int, int],
) -> dict[str, tuple[int, int]]:
    """Calculate grid dimensions and original image position for reflection padding.
    Returns (grid_rows, grid_cols, row_offset, col_offset). For reflection crops.

    Args:
        pad_top (int): Number of pixels to pad above the image.
        pad_bottom (int): Number of pixels to pad below the image.
        pad_left (int): Number of pixels to pad to the left of the image.
        pad_right (int): Number of pixels to pad to the right of the image.
        image_shape (tuple[int, int]): Shape of the original image as (height, width).

    Returns:
        dict[str, tuple[int, int]]: A dictionary containing:
            - 'grid_shape': A tuple (grid_rows, grid_cols) where:
                - grid_rows (int): Number of times the image needs to be repeated vertically.
                - grid_cols (int): Number of times the image needs to be repeated horizontally.
            - 'original_position': A tuple (original_row, original_col) where:
                - original_row (int): Row index of the original image in the grid.
                - original_col (int): Column index of the original image in the grid.

    """
    rows, cols = image_shape[:2]

    grid_rows = 1 + math.ceil(pad_top / rows) + math.ceil(pad_bottom / rows)
    grid_cols = 1 + math.ceil(pad_left / cols) + math.ceil(pad_right / cols)
    original_row = math.ceil(pad_top / rows)
    original_col = math.ceil(pad_left / cols)

    return {
        "grid_shape": (grid_rows, grid_cols),
        "original_position": (original_row, original_col),
    }


@preserve_channel_dim
def distort_image(
    image: np.ndarray,
    generated_mesh: np.ndarray,
    interpolation: int,
) -> np.ndarray:
    """Apply perspective distortion to an image from a generated mesh. Each mesh
    cell is warped; interpolation for resampling. For PiecewiseAffine-style transforms.

    This function applies a perspective transformation to each cell of the image defined by the
    generated mesh. The distortion is applied using OpenCV's perspective transformation and
    blending techniques.

    Args:
        image (np.ndarray): The input image to be distorted. Can be a 2D grayscale image or a
                            3D color image.
        generated_mesh (np.ndarray): A 2D array where each row represents a quadrilateral cell
                                    as [x1, y1, x2, y2, dst_x1, dst_y1, dst_x2, dst_y2, dst_x3, dst_y3, dst_x4, dst_y4].
                                    The first four values define the source rectangle, and the last eight values
                                    define the destination quadrilateral.
        interpolation (int): Interpolation method to be used in the perspective transformation.
                             Should be one of the OpenCV interpolation flags (e.g., cv2.INTER_LINEAR).

    Returns:
        np.ndarray: The distorted image with the same shape and dtype as the input image.

    Note:
        - The function preserves the channel dimension of the input image.
        - Each cell of the generated mesh is transformed independently and then blended into the output image.
        - The distortion is applied using perspective transformation, which allows for more complex
          distortions compared to affine transformations.

    Examples:
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> mesh = np.array([[0, 0, 50, 50, 5, 5, 45, 5, 45, 45, 5, 45]])
        >>> distorted = distort_image(image, mesh, cv2.INTER_LINEAR)
        >>> distorted.shape
        (100, 100, 3)

    """
    distorted_image = np.zeros(image.shape, dtype=image.dtype)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for mesh in generated_mesh:
        x1, y1, x2, y2 = mesh[:4]
        dst_quad = mesh[4:].reshape(4, 2)

        src_quad = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )

        perspective_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)

        warped = warp_perspective(
            image,
            perspective_mat,
            dsize=(image.shape[1], image.shape[0]),
            flags=interpolation,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0,
        )

        mask[:] = 0
        cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)

        distorted_image = cv2.copyTo(warped, mask, distorted_image)

    return distorted_image


def create_affine_transformation_matrix(
    translate: Mapping[str, float],
    shear: dict[str, float],
    scale: dict[str, float],
    rotate: float,
    shift: tuple[float, float],
) -> np.ndarray:
    """Build 3x3 affine matrix from translation, shear, scale, rotation, shift.
    Order: shift topleft, scale, rotate, shear, translate, shift center.

    Args:
        translate (Mapping[str, float]): Translation in x and y directions.
        shear (dict[str, float]): Shear in x and y directions (in degrees).
        scale (dict[str, float]): Scale factors for x and y directions.
        rotate (float): Rotation angle in degrees.
        shift (tuple[float, float]): Shift to apply before and after transformations.

    Returns:
        np.ndarray: The resulting 3x3 affine transformation matrix.

    """
    # Convert angles to radians
    rotate_rad = np.deg2rad(rotate % 360)

    shear_x_rad = np.deg2rad(shear["x"])
    shear_y_rad = np.deg2rad(shear["y"])

    # Create individual transformation matrices
    # 1. Shift to top-left
    m_shift_topleft = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]])

    # 2. Scale
    m_scale = np.array([[scale["x"], 0, 0], [0, scale["y"], 0], [0, 0, 1]])

    # 3. Rotation
    m_rotate = np.array(
        [
            [np.cos(rotate_rad), np.sin(rotate_rad), 0],
            [-np.sin(rotate_rad), np.cos(rotate_rad), 0],
            [0, 0, 1],
        ],
    )

    # 4. Shear
    m_shear = np.array(
        [[1, np.tan(shear_x_rad), 0], [np.tan(shear_y_rad), 1, 0], [0, 0, 1]],
    )

    # 5. Translation
    m_translate = np.array([[1, 0, translate["x"]], [0, 1, translate["y"]], [0, 0, 1]])

    # 6. Shift back to center
    m_shift_center = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])

    # Combine all transformations
    # The order is important: transformations are applied from right to left
    m = m_shift_center @ m_translate @ m_shear @ m_rotate @ m_scale @ m_shift_topleft

    # Ensure the last row is exactly [0, 0, 1]
    m[2] = [0, 0, 1]

    return m


def compute_transformed_image_bounds(
    matrix: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the bounds of an image after applying an affine transformation. matrix
    3x3, image_shape (H, W). Returns min_coords, max_coords of transformed corners.

    Args:
        matrix (np.ndarray): The 3x3 affine transformation matrix.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - min_coords: An array with the minimum x and y coordinates.
            - max_coords: An array with the maximum x and y coordinates.

    """
    height, width = image_shape[:2]

    # Define the corners of the image
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])

    # Transform the corners
    transformed_corners = corners @ matrix.T
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]

    # Calculate the bounding box of the transformed corners
    min_coords = np.floor(transformed_corners.min(axis=0)).astype(int)
    max_coords = np.ceil(transformed_corners.max(axis=0)).astype(int)

    return min_coords, max_coords


def compute_affine_warp_output_shape(
    matrix: np.ndarray,
    input_shape: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute the output shape of an affine warp. matrix 3x3, input_shape (H, W[, C]).
    Returns (adjusted_matrix, output_shape). For Affine keep_size=False.

    This function computes the output shape of an affine warp based on the input matrix and input shape.
    It calculates the transformed image bounds and then determines the output shape based on the input shape.

    Args:
        matrix (np.ndarray): The 3x3 affine transformation matrix.
        input_shape (tuple[int, ...]): The shape of the input image as (height, width, ...).

    Returns:
        tuple[np.ndarray, tuple[int, int]]: A tuple containing:
            - matrix: The 3x3 affine transformation matrix.
            - output_shape: The output shape of the affine warp.

    """
    height, width = input_shape[:2]

    if height == 0 or width == 0:
        return matrix, cast("tuple[int, int]", input_shape[:2])

    min_coords, max_coords = compute_transformed_image_bounds(matrix, (height, width))
    minc, minr = min_coords
    maxc, maxr = max_coords

    out_height = maxr - minr + 1
    out_width = maxc - minc + 1

    output_shape_tuple: tuple[int, ...] = (
        (index(out_height), index(out_width), input_shape[2])
        if len(input_shape) == NUM_MULTI_CHANNEL_DIMENSIONS
        else (index(out_height), index(out_width))
    )

    # fit output image in new shape
    translation = np.array([[1, 0, -minc], [0, 1, -minr], [0, 0, 1]])
    matrix = translation @ matrix

    return matrix, cast("tuple[int, int]", output_shape_tuple)


def center(image_shape: tuple[int, int]) -> tuple[float, float]:
    """Calculate the center coordinates of the image. (width/2 - 0.5, height/2 - 0.5).
    For rotation and affine center. image_shape (H, W). Returns (cx, cy).

    Args:
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[float, float]: center_x, center_y

    """
    height, width = image_shape[:2]
    return width / 2 - 0.5, height / 2 - 0.5


def center_bbox(image_shape: tuple[int, int]) -> tuple[float, float]:
    """Calculate the center coordinates of the image for bounding boxes. (width/2,
    height/2). For bbox center in OBB or crop. image_shape (H, W). Returns (cx, cy).

    Args:
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[float, float]: center_x, center_y

    """
    height, width = image_shape[:2]
    return width / 2, height / 2


def hflip_images(volume: np.ndarray) -> np.ndarray:
    """Perform horizontal flip on a single volume (D, H, W) or (D, H, W, C). Flips
    along width axis. For Transforms3D HorizontalFlip.

    Flips the volume along the width axis (axis=2). Handles inputs with
    shapes (D, H, W) or (D, H, W, C).

    Args:
        volume (np.ndarray): Input volume.

    Returns:
        np.ndarray: Horizontally flipped volume.

    """
    return np.flip(volume, axis=2)


def vflip_images(volume: np.ndarray) -> np.ndarray:
    """Perform vertical flip on a single volume (D, H, W) or (D, H, W, C). Flips
    along height axis. For Transforms3D VerticalFlip.

    Flips the volume along the height axis (axis=1). Handles inputs with
    shapes (D, H, W) or (D, H, W, C).

    Args:
        volume (np.ndarray): Input volume.

    Returns:
        np.ndarray: Vertically flipped volume.

    """
    return np.flip(volume, axis=1)


def hflip_volumes(volumes: np.ndarray) -> np.ndarray:
    """Perform horizontal flip on batch of volumes (B, D, H, W) or (B, D, H, W, C).
    Flips along width axis. For Transforms3D batch HorizontalFlip.

    Flips the volumes along the width axis (axis=3). Handles inputs with
    shapes (B, D, H, W) or (B, D, H, W, C).

    Args:
        volumes (np.ndarray): Input batch of volumes.

    Returns:
        np.ndarray: Horizontally flipped batch of volumes.

    """
    # Width axis is 3 for both (B, D, H, W) and (B, D, H, W, C)
    return np.flip(volumes, axis=3)


def vflip_volumes(volumes: np.ndarray) -> np.ndarray:
    """Perform vertical flip on batch of volumes (B, D, H, W) or (B, D, H, W, C).
    Flips along height axis. For Transforms3D batch VerticalFlip.

    Flips the volumes along the height axis (axis=2). Handles inputs with
    shapes (B, D, H, W) or (B, D, H, W, C).

    Args:
        volumes (np.ndarray): Input batch of volumes.

    Returns:
        np.ndarray: Vertically flipped batch of volumes.

    """
    # Height axis is 2 for both (B, D, H, W) and (B, D, H, W, C)
    return np.flip(volumes, axis=2)


def rot90_volumes(volumes: np.ndarray, group_element: Literal["e", "r90", "r180", "r270"]) -> np.ndarray:
    """Rotate batch of volumes 90° CCW in H-W plane. group_element: e, r90, r180, r270.
    Shape (B, D, H, W) or (B, D, H, W, C). For Transforms3D D4/C4.

    Rotates the volumes in the height-width plane (axes 2 and 3).
    Handles inputs with shapes (B, D, H, W) or (B, D, H, W, C).

    Args:
        volumes (np.ndarray): Input batch of volumes.
        group_element (Literal['e', 'r90', 'r180', 'r270']): C4 group element to apply.

    Returns:
        np.ndarray: Rotated batch of volumes.

    """
    rot90_count = C4_GROUP_ELEMENT_TO_K[group_element]
    return np.rot90(volumes, k=rot90_count, axes=(2, 3))


@preserve_channel_dim
def erode(img: ImageType, kernel: np.ndarray) -> ImageType:
    """One iteration of morphological erosion. Shrinks bright regions. Use for mask/bbox
    morphology. Same shape and channel count.

    This function applies erosion to an image using the cv2.erode function.

    Args:
        img (ImageType): Input image as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.

    Returns:
        ImageType: The eroded image.

    """
    return cast("ImageType", cv2.erode(img, kernel, iterations=1))


@preserve_channel_dim
def dilate(img: ImageType, kernel: np.ndarray) -> ImageType:
    """One iteration of morphological dilation. Expands bright regions. Use for mask/bbox
    morphology. Same shape and channel count.

    This function applies dilation to an image using the cv2.dilate function.

    Args:
        img (ImageType): Input image as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.

    Returns:
        ImageType: The dilated image.

    """
    return cast("ImageType", cv2.dilate(img, kernel, iterations=1))


def morphology(
    img: ImageType,
    kernel: np.ndarray,
    operation: Literal["dilation", "erosion"],
) -> np.ndarray:
    """Apply dilation or erosion to an image. operation: 'dilation' or 'erosion';
    kernel is structuring element. For BboxMorphology / mask cleanup.

    This function applies morphology to an image using the cv2.morphologyEx function.

    Args:
        img (ImageType): Input image as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.
        operation (Literal['dilation', 'erosion']): The operation to apply.

    Returns:
        np.ndarray: The morphology applied to the image.

    """
    if operation == "dilation":
        return dilate(img, kernel)
    if operation == "erosion":
        return erode(img, kernel)

    raise ValueError(f"Unsupported operation: {operation}")


D4_TRANSFORMATIONS_IMAGES = {
    "e": lambda x: x,  # Identity transformation
    "r90": lambda x: rot90_images(x, "r90"),  # Rotate 90 degrees
    "r180": lambda x: rot90_images(x, "r180"),  # Rotate 180 degrees
    "r270": lambda x: rot90_images(x, "r270"),  # Rotate 270 degrees
    "v": vflip_images,  # Vertical flip (already batch-aware)
    "hvt": lambda x: transpose_images(rot90_images(x, "r180")),  # Reflect over anti-diagonal
    "h": hflip_images,  # Horizontal flip (already batch-aware)
    "t": transpose_images,  # Transpose (reflect over main diagonal)
}


def d4_images(img: ImageType, group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]) -> np.ndarray:
    """Apply one of eight D4 square symmetries to a batch of images (N, H, W[, C]).
    group_member: e, r90, r180, r270, v, hvt, h, t. Rotations and flips.

    This function manipulates a batch of images using transformations such as rotations and flips,
    corresponding to the `D_4` dihedral group symmetry operations.
    Each transformation is identified by a unique group member code.

    Args:
        img (ImageType): The input batch of images to transform with shape:
            - (N, H, W) for grayscale images
            - (N, H, W, C) for multi-channel images
            where N is the batch size, H is height, W is width, C is channels
        group_member (Literal['e', 'r90', 'r180', 'r270', 'v', 'hvt', 'h', 't']): A string identifier indicating
            the specific transformation to apply. Valid codes include:
            - 'e': Identity (no transformation).
            - 'r90': Rotate 90 degrees counterclockwise.
            - 'r180': Rotate 180 degrees.
            - 'r270': Rotate 270 degrees counterclockwise.
            - 'v': Vertical flip.
            - 'hvt': Transpose over second diagonal
            - 'h': Horizontal flip.
            - 't': Transpose (reflect over the main diagonal).

    Returns:
        np.ndarray: The transformed batch of images.

    """
    # Execute the appropriate transformation
    return D4_TRANSFORMATIONS_IMAGES[group_member](img)


__all__ = [
    "C4_GROUP_ELEMENT_TO_K",
    "D4_GROUP_ELEMENTS",
    "D4_TRANSFORMATIONS",
    "D4_TRANSFORMATIONS_IMAGES",
    "PAIR",
    "ROT90_180_FACTOR",
    "ROT90_270_FACTOR",
    "_PIL_AVAILABLE",
    "_PYVIPS_AVAILABLE",
    "_get_resize_backend",
    "apply_affine_to_points",
    "calculate_affine_transform_padding",
    "center",
    "center_bbox",
    "compute_affine_warp_output_shape",
    "compute_transformed_image_bounds",
    "create_affine_transformation_matrix",
    "d4",
    "d4_images",
    "dilate",
    "distort_image",
    "erode",
    "get_pad_grid_dimensions",
    "hflip_images",
    "hflip_volumes",
    "is_identity_matrix",
    "morphology",
    "pad",
    "pad_images_with_params",
    "pad_with_params",
    "perspective",
    "perspective_images",
    "perspective_keypoints",
    "resize",
    "resize_pil",
    "resize_pyvips",
    "rot90",
    "rot90_images",
    "rot90_volumes",
    "rotation2d_matrix_to_euler_angles",
    "scale",
    "transpose",
    "transpose_images",
    "transpose_volumes",
    "vflip_images",
    "vflip_volumes",
]
