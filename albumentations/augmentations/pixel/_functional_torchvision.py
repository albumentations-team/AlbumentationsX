"""TorchVision-style color adjustment and superpixel functional helpers."""

from __future__ import annotations

from collections.abc import Sequence

from ._functional_color import (
    grayscale_to_multichannel,
    to_gray_average,
    to_gray_weighted_average,
)
from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    ImageType,
    ImageUInt8,
    add_weighted,
    clipped,
    cv2,
    fgeometric,
    float32_io,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    mean,
    multiply,
    multiply_add,
    np,
    preserve_channel_dim,
    reduce_sum,
    std,
    sz_lut,
    uint8_io,
)


@float32_io
@clipped
@preserve_channel_dim
def fancy_pca(img: ImageType, alpha_vector: np.ndarray) -> ImageType:
    """Fancy PCA color augmentation: add noise along principal components. alpha_vector; any channel
    count. float32 I/O, clipped.

    Args:
        img (ImageType): Input image
        alpha_vector (np.ndarray): Vector of scale factors for each principal component.
                                   Should have the same length as the number of channels in the image.

    Returns:
        ImageType: Augmented image of the same shape, type, and range as the input.

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This function generalizes the Fancy PCA augmentation to work with any number of channels.
        - It preserves the original range of the image ([0, 255] for uint8, [0, 1] for float32).
        - For single-channel images, the augmentation is applied as a simple scaling of pixel intensity variation.
        - For multi-channel images, PCA is performed on the entire image, treating each pixel
          as a point in N-dimensional space (where N is the number of channels).
        - The augmentation preserves the correlation between channels while adding controlled noise.
        - Computation time may increase significantly for images with a large number of channels.

    References:
        ImageNet classification with deep convolutional neural networks: Krizhevsky, A., Sutskever, I.,
            & Hinton, G. E. (2012): In Advances in neural information processing systems (pp. 1097-1105).

    """
    orig_shape = img.shape
    num_channels = get_num_channels(img)

    # Reshape image to 2D array of pixels
    img_reshaped = img.reshape(-1, num_channels)

    # Center the pixel values
    img_mean = mean(img_reshaped, axis=0, dtype=np.float32)
    img_centered = img_reshaped - img_mean

    if num_channels == 1:
        # For grayscale images, apply a simple scaling
        std_dev = std(img_centered, eps=0)
        noise = alpha_vector[0] * std_dev * img_centered
    else:
        # Compute covariance matrix
        img_cov = np.cov(img_centered, rowvar=False)

        # Compute eigenvectors & eigenvalues of the covariance matrix
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)

        # Sort eigenvectors by eigenvalues in descending order
        sort_perm = eig_vals[::-1].argsort()
        eig_vals = eig_vals[sort_perm]
        eig_vecs = eig_vecs[:, sort_perm]

        # Create noise vector
        noise = np.dot(
            np.dot(eig_vecs, np.diag(alpha_vector * eig_vals)),
            img_centered.T,
        ).T

    # Add noise to the image
    img_pca = img_reshaped + noise

    # Reshape back to original shape
    img_pca = img_pca.reshape(orig_shape)

    # Clip values to [0, 1] range
    return np.clip(img_pca, 0, 1, out=img_pca)


@preserve_channel_dim
def adjust_brightness_torchvision(img: ImageType, factor: np.ndarray) -> ImageType:
    """Adjust brightness by multiplying pixels by factor. Torchvision-compatible; uint8 and float32.
    factor scalar or per-channel array.

    This function adjusts the brightness of an image by multiplying each pixel value by a factor.
    The brightness is adjusted by multiplying the image by the factor.

    Args:
        img (ImageType): Input image as a numpy array.
        factor (np.ndarray): The factor to adjust the brightness by.

    Returns:
        ImageType: The adjusted image.

    """
    if factor == 0:
        return np.zeros_like(img)
    if factor == 1:
        return img

    return multiply(img, factor, inplace=False)


@preserve_channel_dim
def adjust_contrast_torchvision(img: ImageType, factor: float) -> ImageType:
    """Adjust contrast by multiplying by factor (relative to grayscale mean). Torchvision-compatible;
    uint8 and float32. factor 0 yields flat gray.

    This function adjusts the contrast of an image by multiplying each pixel value by a factor.
    The contrast is adjusted by multiplying the image by the factor.

    Args:
        img (ImageType): Input image as a numpy array.
        factor (float): The factor to adjust the contrast by.

    Returns:
        ImageType: The adjusted image.

    """
    if factor == 1:
        return img

    img_mean = mean(img) if is_grayscale_image(img) else mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    if factor == 0:
        if img.dtype != np.float32:
            img_mean = int(img_mean + 0.5)
        return np.full_like(img, img_mean, dtype=img.dtype)

    return multiply_add(img, factor, img_mean * (1 - factor), inplace=False)


@clipped
@preserve_channel_dim
def adjust_saturation_torchvision(
    img: ImageType,
    factor: float,
    gamma: float = 0,
) -> ImageType:
    """Adjust saturation by blending with grayscale. Factor in [0, inf]; uses
    to_gray (weighted_average for RGB). Torchvision-compatible.

    Uses to_gray for conversion: weighted_average for RGB (matches OpenCV), average for
    arbitrary channels. Works on batches (4D) and volumes (5D).

    Args:
        img (ImageType): Input image as a numpy array.
        factor (float): The factor to adjust the saturation by.
        gamma (float): Unused, kept for API compatibility.

    Returns:
        ImageType: The adjusted image.

    """
    if factor == 1 or is_grayscale_image(img):
        return img

    gray = to_gray_weighted_average(img) if is_rgb_image(img) else to_gray_average(img)
    gray_expanded = grayscale_to_multichannel(gray, img.shape[-1])

    return gray_expanded if factor == 0 else add_weighted(img, factor, gray_expanded, 1 - factor)


def _adjust_hue_torchvision_uint8(img: ImageUInt8, factor: float) -> ImageUInt8:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = sz_lut(img[..., 0], lut, inplace=False)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue_torchvision(img: ImageType, factor: float) -> ImageType:
    """Adjust hue by shifting in HSV. factor in [-0.5, 0.5]. Torchvision-compatible; RGB only.
    LUT for uint8; in-place mod for float32.

    This function adjusts the hue of an image by adding a factor to the hue value.

    Args:
        img (ImageType): Input image.
        factor (float): The factor to adjust the hue by.

    Returns:
        ImageType: The adjusted image.

    """
    if is_grayscale_image(img) or factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_torchvision_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def apply_brightness_contrast_torchvision(
    img: ImageType,
    brightness_factor: float,
    contrast_factor: float,
    brightness_first: bool,
) -> ImageType:
    """Fused brightness and contrast: single LUT (uint8) or two passes (float32). clip(a*x+b);
    torchvision-compatible; order via brightness_first.

    Both operations are `clip(a*x + b)`. The image grayscale mean is computed once and propagated
    analytically through the pipeline: if brightness comes first, `mean_at_contrast = mean * brightness_factor`
    (clipped to valid range). This avoids re-reading the image after brightness is applied.

    For uint8 images the composition is pre-computed over all 256 input values into a single
    256-entry LUT applied in one `cv2.LUT` call. For float32, two sequential clipped passes are used.

    Args:
        img (ImageType): Input image (uint8 or float32).
        brightness_factor (float): Brightness multiplicative factor.
        contrast_factor (float): Contrast multiplicative factor.
        brightness_first (bool): Whether brightness is applied before contrast.

    Returns:
        ImageType: Adjusted image with the same dtype as input.

    """
    # Compute original grayscale mean once, normalised to [0, 1].
    gray_for_mean = (
        img
        if is_grayscale_image(img)
        else (to_gray_weighted_average(img) if is_rgb_image(img) else to_gray_average(img))
    )
    img_mean = float(mean(gray_for_mean))
    if img.dtype == np.uint8:
        img_mean /= 255.0

    # Propagate mean analytically: brightness scales the mean, clipped to [0, 1].
    mean_at_contrast = float(np.clip(img_mean * brightness_factor, 0.0, 1.0)) if brightness_first else img_mean

    if img.dtype == np.uint8:
        lut = np.arange(256, dtype=np.float32)
        if brightness_first:
            lut = np.clip(lut * brightness_factor, 0.0, 255.0)
            lut = np.clip(lut * contrast_factor + mean_at_contrast * 255.0 * (1.0 - contrast_factor), 0.0, 255.0)
        else:
            lut = np.clip(lut * contrast_factor + mean_at_contrast * 255.0 * (1.0 - contrast_factor), 0.0, 255.0)
            lut = np.clip(lut * brightness_factor, 0.0, 255.0)
        return sz_lut(img, lut.astype(np.uint8), inplace=False)

    # float32: two clipped passes, single buffer, in-place ops
    offset = mean_at_contrast * (1.0 - contrast_factor)
    out = np.empty_like(img)
    if brightness_first:
        np.multiply(img, brightness_factor, out=out)
        np.clip(out, 0.0, 1.0, out=out)
        np.multiply(out, contrast_factor, out=out)
        np.add(out, offset, out=out)
        np.clip(out, 0.0, 1.0, out=out)
    else:
        np.multiply(img, contrast_factor, out=out)
        np.add(out, offset, out=out)
        np.clip(out, 0.0, 1.0, out=out)
        np.multiply(out, brightness_factor, out=out)
        np.clip(out, 0.0, 1.0, out=out)
    return out


@uint8_io
@preserve_channel_dim
def superpixels(
    image: np.ndarray,
    n_segments: int,
    replace_samples: Sequence[bool],
    max_size: int | None,
    interpolation: int,
) -> np.ndarray:
    """Apply superpixels using SLIC: replace pixels with segment mean. n_segments, replace_samples,
    max_size, interpolation. uint8 I/O.

    This function applies superpixels to an image using the SLIC algorithm.
    The superpixels are applied by replacing the pixels in the image with the mean intensity of the superpixel.

    Args:
        image (np.ndarray): Input image as a numpy array.
        n_segments (int): The number of segments to use for the superpixels.
        replace_samples (Sequence[bool]): The samples to replace.
        max_size (int | None): The maximum size of the superpixels.
        interpolation (int): The interpolation method to use.

    Returns:
        np.ndarray: The superpixels applied to the image.

    """
    if not np.any(replace_samples):
        return image

    orig_shape = image.shape
    if max_size is not None:
        size = max(image.shape[:2])
        if size > max_size:
            scale = max_size / size
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            image = fgeometric.resize(image, (new_height, new_width), interpolation)

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=10,
    )

    min_value = 0
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    image = np.copy(image)

    num_channels = get_num_channels(image)

    for c in range(num_channels):
        image_sp_c = image[..., c]
        # Get unique segment labels (skip 0 if it exists as it's typically background)
        unique_labels = np.unique(segments)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        # Calculate mean intensity for each segment
        for idx, label in enumerate(unique_labels):
            # with mod here, because slic can sometimes create more superpixel than requested.
            # replace_samples then does not have enough values, so we just start over with the first one again.
            if replace_samples[idx % len(replace_samples)]:
                mask = segments == label
                mean_intensity = mean(image_sp_c[mask])

                if image_sp_c.dtype.kind in ["i", "u", "b"]:
                    # After rounding the value can end up slightly outside of the value_range. Hence, we need to clip.
                    # We do clip via min(max(...)) instead of np.clip because
                    # the latter one does not seem to keep dtypes for dtypes with large itemsizes (e.g. uint64).
                    value: int | float
                    value = int(np.round(mean_intensity))
                    value = min(max(value, min_value), max_value)
                else:
                    value = mean_intensity

                image_sp_c[mask] = value

    return fgeometric.resize(image, orig_shape[:2], interpolation) if orig_shape != image.shape else image


def slic(
    image: np.ndarray,
    n_segments: int,
    compactness: float = 10.0,
    max_iterations: int = 10,
) -> np.ndarray:
    """SLIC superpixel segmentation. n_segments, compactness, max_iterations. Returns label mask for
    oversegmentation; used by superpixels() or standalone.

    Args:
        image (np.ndarray): Input image (3D numpy array with shape (H, W, C)).
        n_segments (int): Approximate number of superpixels to generate.
        compactness (float): Balance between color proximity and space proximity.
        max_iterations (int): Maximum number of iterations for k-means.

    Returns:
        np.ndarray: Segmentation mask where each superpixel has a unique label.

    """
    height, width = image.shape[:2]
    num_pixels = height * width

    # Normalize image to [0, 1] range
    max_val = np.float32(image.max())
    image_normalized = image.astype(np.float32) / (max_val + np.float32(1e-6))

    # Initialize cluster centers via meshgrid
    grid_step = int((num_pixels / n_segments) ** 0.5)
    x_range = np.arange(grid_step // 2, width, grid_step)
    y_range = np.arange(grid_step // 2, height, grid_step)
    xx_grid, yy_grid = np.meshgrid(x_range, y_range)
    centers = np.column_stack([xx_grid.ravel(), yy_grid.ravel()]).astype(np.float32)

    # Initialize labels and distances
    labels = np.full((height, width), -1, dtype=np.int32)
    distances = np.full((height, width), np.inf, dtype=np.float32)

    inv_grid_step_sq = np.float32(1.0 / (grid_step * grid_step))

    for _ in range(max_iterations):
        for i, center in enumerate(centers):
            y, x = int(center[1]), int(center[0])

            y_low, y_high = max(0, y - grid_step), min(height, y + grid_step + 1)
            x_low, x_high = max(0, x - grid_step), min(width, x + grid_step + 1)

            crop = image_normalized[y_low:y_high, x_low:x_high]
            color_diff = crop - image_normalized[y, x]
            color_distance = reduce_sum(color_diff**2, axis=-1)

            yy, xx = np.ogrid[y_low:y_high, x_low:x_high]
            spatial_distance = ((yy - y) ** 2 + (xx - x) ** 2) * inv_grid_step_sq

            distance = color_distance + compactness * spatial_distance

            dist_slice = distances[y_low:y_high, x_low:x_high]
            mask = distance < dist_slice
            dist_slice[mask] = distance[mask]
            labels[y_low:y_high, x_low:x_high][mask] = i

        for i in range(len(centers)):
            mask = labels == i
            if np.any(mask):
                ys, xs = np.where(mask)
                centers[i] = [mean(xs.astype(np.float32)), mean(ys.astype(np.float32))]

    return labels


__all__ = [
    "_adjust_hue_torchvision_uint8",
    "adjust_brightness_torchvision",
    "adjust_contrast_torchvision",
    "adjust_hue_torchvision",
    "adjust_saturation_torchvision",
    "apply_brightness_contrast_torchvision",
    "fancy_pca",
    "slic",
    "superpixels",
]
