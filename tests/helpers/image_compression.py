"""OpenCV and channel references shared by ImageCompression tests."""

import cv2
import numpy as np

from albumentations.augmentations.pixel import functional as fpixel


def opencv_compression(image, quality, image_type, read_mode):
    flag = cv2.IMWRITE_JPEG_QUALITY if image_type == ".jpg" else cv2.IMWRITE_WEBP_QUALITY
    ok, encoded = cv2.imencode(image_type, image, (flag, quality))
    assert ok
    return cv2.imdecode(encoded, read_mode)


def assert_rgba_compression(result, original, quality, image_type):
    assert result.shape == original.shape
    assert result.dtype == original.dtype
    expected_rgb = fpixel.image_compression(original[..., :3], quality, image_type)
    np.testing.assert_array_equal(result[..., :3], expected_rgb, strict=True)
    assert result[..., 3].tobytes() == original[..., 3].tobytes()
