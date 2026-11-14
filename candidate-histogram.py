"""Histogram equalization for channel-last uint8 images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from albucore.lut import apply_uint8_lut
from albucore.utils import NUM_RGB_CHANNELS, get_opencv_max_channels

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


__all__ = ["equalize_uint8"]

_OPENCV_MAX_CHANNELS = get_opencv_max_channels()
_MAX_EXACT_HISTOGRAM_PIXELS = 2**24


def _histogram(img: NDArray[np.uint8], mask: NDArray[np.uint8] | None = None) -> NDArray[np.int64]:
    if img.size <= _MAX_EXACT_HISTOGRAM_PIXELS:
        return np.asarray(cv2.calcHist([img], [0], mask, [256], (0, 256)), dtype=np.int64).ravel()

    # Sum integer histograms from tiles whose counts fit exactly in float32.
    rows_per_tile = max(1, _MAX_EXACT_HISTOGRAM_PIXELS // img.shape[1])
    cols_per_tile = min(img.shape[1], _MAX_EXACT_HISTOGRAM_PIXELS)
    counts = np.zeros(256, dtype=np.int64)
    for row in range(0, img.shape[0], rows_per_tile):
        for col in range(0, img.shape[1], cols_per_tile):
            tile = np.s_[row : row + rows_per_tile, col : col + cols_per_tile]
            counts += _histogram(img[tile], None if mask is None else mask[tile])
    return counts


def _pillow_lut(histogram: NDArray[np.int64]) -> NDArray[np.uint8] | None:
    nonzero = np.flatnonzero(histogram)
    if nonzero.size <= 1:
        return None
    # The final occupied bin does not contribute to Pillow's step.
    step = int(histogram.sum() - histogram[nonzero[-1]]) // 255
    if step == 0:
        return None

    before_bin = np.cumsum(histogram) - histogram
    return ((before_bin + step // 2) // step).clip(0, 255).astype(np.uint8)


def _opencv_lut(histogram: NDArray[np.int64]) -> NDArray[np.uint8] | None:
    nonzero = np.flatnonzero(histogram)
    if nonzero.size <= 1:
        return None

    first = nonzero[0]
    scale = np.float32(255) / np.float32(histogram.sum() - histogram[first])
    counts = np.cumsum(histogram) - histogram[first]
    scaled = counts.astype(np.float32) * scale
    return np.asarray(np.clip(scaled.round(), 0, 255), dtype=np.uint8)


def _equalize_cv_unmasked(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    channels = img.shape[-1]
    if channels == 1:
        return np.asarray(cv2.equalizeHist(img[..., 0]), dtype=np.uint8)[..., None]
    if channels <= _OPENCV_MAX_CHANNELS:
        planes = [cv2.equalizeHist(plane) for plane in cv2.split(img)]
        return np.asarray(cv2.merge(planes), dtype=np.uint8)

    result = np.empty_like(img)
    for channel in range(channels):
        result[..., channel] = cv2.equalizeHist(img[..., channel])
    return result


def _equalize_packed_channels(
    img: NDArray[np.uint8],
    lut_for: Callable[[NDArray[np.int64]], NDArray[np.uint8] | None],
    mask: NDArray[np.uint8] | None,
) -> NDArray[np.uint8]:
    result = np.empty_like(img, order="C")
    plane = np.empty(img.shape[:2], dtype=np.uint8)
    for channel in range(img.shape[-1]):
        cv2.extractChannel(img, channel, dst=plane)
        selection = mask[..., channel] if mask is not None and mask.ndim == 3 else mask
        lut = lut_for(_histogram(plane, selection))
        if lut is not None:
            apply_uint8_lut(plane, lut, out=plane)
        cv2.insertChannel(plane, result, channel)
    return result


def equalize_uint8(
    img: NDArray[np.uint8],
    *,
    mode: Literal["cv", "pil"] = "cv",
    mask: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Equalize each channel of a prevalidated HWC uint8 image independently.

    Args:
        img: Input image with an explicit channel axis.
        mode: OpenCV or Pillow histogram mapping.
        mask: Optional uint8 histogram selection of shape HW, HWC with one
            shared channel, or HWC with one mask per image channel. Nonzero
            values select pixels for the histogram. The mapping applies to
            every pixel in that image channel.

    Returns:
        An independent writable uint8 array with the input shape.

    Notes:
        A mask with no selected pixels leaves the channel values unchanged.
        Pillow mode reuses a single channel buffer for packed RGB images and
        row-strided RGB crops.

    Examples:
        >>> image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        >>> equalize_uint8(image, mode="cv").shape
        (16, 16, 1)
    """
    if not img.size:
        return img.copy()
    if mode == "cv" and mask is None:
        return _equalize_cv_unmasked(img)

    lut_for = _pillow_lut if mode == "pil" else _opencv_lut
    if mask is not None and mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    channels = img.shape[-1]
    if img.strides[1:] == (channels, 1) and img.strides[0] >= img.shape[1] * channels:
        if mask is None:
            if channels == NUM_RGB_CHANNELS:
                return _equalize_packed_channels(img, lut_for, None)
        elif 1 < channels <= _OPENCV_MAX_CHANNELS and img.size >= 4_096 * channels:
            return _equalize_packed_channels(img, lut_for, mask)

    result = np.empty_like(img)
    for channel in range(channels):
        plane = img[..., channel]
        selection = mask[..., channel] if mask is not None and mask.ndim == 3 else mask
        lut = lut_for(_histogram(plane, selection))
        result[..., channel] = plane if lut is None else apply_uint8_lut(plane, lut)
    return result
