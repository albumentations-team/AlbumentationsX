"""Test data factory for creating reproducible test data.

This module centralizes all test data creation patterns with consistent seeding
to eliminate RNG ordering dependencies.
"""

import cv2
import numpy as np


class TestDataFactory:
    """Factory for creating test data with reproducible seeding.

    All methods use independent RNGs to avoid test ordering dependencies
    when running tests in parallel with pytest-xdist and pytest-randomly.
    """

    DEFAULT_SEED = 137

    @staticmethod
    def create_rng(seed: int | None = None) -> np.random.Generator:
        """Create an independent RNG with specified seed.

        Args:
            seed: Seed for RNG. If None, uses DEFAULT_SEED.

        Returns:
            numpy.random.Generator instance

        """
        if seed is None:
            seed = TestDataFactory.DEFAULT_SEED
        return np.random.default_rng(seed)

    @staticmethod
    def create_image(
        shape: tuple[int, ...],
        dtype: type = np.uint8,
        seed: int | None = None,
    ) -> np.ndarray:
        """Create test image with independent RNG.

        Args:
            shape: Image shape (H, W) or (H, W, C)
            dtype: Data type (np.uint8 or np.float32)
            seed: RNG seed (if None, uses DEFAULT_SEED)

        Returns:
            Test image array

        """
        rng = TestDataFactory.create_rng(seed)

        if dtype == np.uint8:
            return rng.integers(0, 256, shape, dtype=np.uint8)
        if dtype == np.float32:
            # Use cv2.randu for float32 (2x faster than numpy)
            img = np.empty(shape, dtype=np.float32)
            # Set RNG seed for cv2 to match our seed
            cv2.setRNGSeed(seed if seed is not None else TestDataFactory.DEFAULT_SEED)
            cv2.randu(img, 0, 1)
            return img
        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def create_mask(
        shape: tuple[int, int],
        seed: int | None = None,
    ) -> np.ndarray:
        """Create binary mask for testing.

        Args:
            shape: Mask shape (H, W)
            seed: RNG seed (if None, uses DEFAULT_SEED)

        Returns:
            Binary mask (uint8, values 0 or 1)

        """
        rng = TestDataFactory.create_rng(seed)
        return rng.integers(0, 2, shape, dtype=np.uint8)

    @staticmethod
    def create_volume(
        shape: tuple[int, int, int, int],
        dtype: type = np.uint8,
        seed: int | None = None,
    ) -> np.ndarray:
        """Create 3D volume for testing.

        Args:
            shape: Volume shape (D, H, W, C)
            dtype: Data type
            seed: RNG seed (if None, uses DEFAULT_SEED)

        Returns:
            Volume array

        """
        rng = TestDataFactory.create_rng(seed)

        if dtype == np.uint8:
            return rng.integers(0, 256, shape, dtype=np.uint8)
        if dtype == np.float32:
            return rng.uniform(0, 1, shape).astype(np.float32)
        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def create_bboxes(
        num_boxes: int = 3,
        format: str = "pascal_voc",
        image_shape: tuple[int, int] = (100, 100),
        seed: int | None = None,
    ) -> np.ndarray:
        """Create test bounding boxes.

        Args:
            num_boxes: Number of boxes to create
            format: Bbox format ('pascal_voc', 'albumentations', 'coco', 'yolo')
            image_shape: Image dimensions (H, W) for normalized formats
            seed: RNG seed (if None, uses DEFAULT_SEED)

        Returns:
            Array of bboxes

        """
        rng = TestDataFactory.create_rng(seed)
        h, w = image_shape

        boxes = []
        for _ in range(num_boxes):
            if format in ("albumentations", "yolo"):
                # Normalized coordinates
                x1 = rng.uniform(0.1, 0.4)
                y1 = rng.uniform(0.1, 0.4)
                x2 = rng.uniform(0.6, 0.9)
                y2 = rng.uniform(0.6, 0.9)
            else:
                # Pixel coordinates
                x1 = rng.uniform(10, w * 0.4)
                y1 = rng.uniform(10, h * 0.4)
                x2 = rng.uniform(w * 0.6, w - 10)
                y2 = rng.uniform(h * 0.6, h - 10)

            if format == "coco":
                # x, y, width, height
                boxes.append([x1, y1, x2 - x1, y2 - y1])
            elif format == "yolo":
                # x_center, y_center, width, height (normalized)
                boxes.append(
                    [
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        x2 - x1,
                        y2 - y1,
                    ],
                )
            else:
                # pascal_voc or albumentations: x_min, y_min, x_max, y_max
                boxes.append([x1, y1, x2, y2])

        return np.array(boxes, dtype=np.float32)

    @staticmethod
    def create_keypoints(
        num_points: int = 5,
        format: str = "xy",
        image_shape: tuple[int, int] = (100, 100),
        seed: int | None = None,
    ) -> np.ndarray:
        """Create test keypoints.

        Args:
            num_points: Number of keypoints to create
            format: Keypoint format ('xy', 'yx', 'xya', 'xys', 'xyas', 'xysa')
            image_shape: Image dimensions (H, W)
            seed: RNG seed (if None, uses DEFAULT_SEED)

        Returns:
            Array of keypoints

        """
        rng = TestDataFactory.create_rng(seed)
        h, w = image_shape

        points = []
        for _ in range(num_points):
            x = rng.uniform(5, w - 5)
            y = rng.uniform(5, h - 5)

            if format in ("xy", "yx"):
                point = [x, y] if format == "xy" else [y, x]
            elif format in ("xya", "xys"):
                angle_or_scale = rng.uniform(0, 2 * np.pi if format == "xya" else 1.5)
                point = [x, y, angle_or_scale]
            elif format in ("xyas", "xysa"):
                angle = rng.uniform(0, 2 * np.pi)
                scale = rng.uniform(0.5, 1.5)
                point = [x, y, angle, scale] if format == "xyas" else [x, y, scale, angle]
            else:
                raise ValueError(f"Unsupported keypoint format: {format}")

            points.append(point)

        return np.array(points, dtype=np.float32)
