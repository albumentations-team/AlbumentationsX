"""OpenCV and channel references shared by ImageCompression tests."""

import cv2
import numpy as np

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel


def opencv_compression(image, quality, image_type, read_mode):
    flag = cv2.IMWRITE_JPEG_QUALITY if image_type == ".jpg" else cv2.IMWRITE_WEBP_QUALITY
    ok, encoded = cv2.imencode(image_type, image, (flag, quality))
    assert ok
    return cv2.imdecode(encoded, read_mode)


def assert_rgba_compression(result, original, expected_rgb):
    assert result.shape == original.shape
    assert result.dtype == original.dtype
    np.testing.assert_array_equal(result[..., :3], expected_rgb, strict=True)
    assert result[..., 3].tobytes() == original[..., 3].tobytes()


def check_rgba_compression(image, quality, compression_type, route="functional"):
    image_type = ".jpg" if compression_type == "jpeg" else ".webp"
    original = image.copy()
    expected_rgb = fpixel.image_compression(image[..., :3], quality, image_type)
    if route == "functional":
        result = fpixel.image_compression(image, quality, image_type)
    else:
        transform = A.ImageCompression(compression_type=compression_type, quality_range=(quality, quality), p=1)
        pipeline = A.Compose([transform], seed=137, strict=True) if route == "compose" else transform
        result = pipeline(image=image)["image"]

    assert_rgba_compression(result, original, expected_rgb)
    assert not np.shares_memory(result, image)
    assert image.tobytes() == original.tobytes()
    return result
