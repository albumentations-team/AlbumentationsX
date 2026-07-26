"""Tests for RandomScale transform — uniform and anisotropic scaling."""

import hashlib

import cv2
import numpy as np
import pytest

import albumentations as A
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.core.serialization import from_dict, to_dict

SCALE_RANGES = [(-0.5, -0.5), {"x": (-0.5, -0.5), "y": (-0.5, -0.5)}]


class TestRandomScale:
    @pytest.mark.parametrize("scale_range", SCALE_RANGES)
    def test_non_square_output(self, scale_range):
        """Fixed scale produces predictable output shape on non-square input."""
        image = np.random.randint(0, 256, (80, 120, 3), dtype=np.uint8)
        result = A.RandomScale(scale_range=scale_range, p=1.0)(image=image)
        assert result["image"].shape == (40, 60, 3)

    def test_legacy_tuple_regression(self):
        """Deterministic seeded regression test for the legacy tuple mode.

        Golden hash locks the exact pixel output for tuple scale_range
        with a fixed seed and deterministic input image.
        """
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (101, 97, 3), dtype=np.uint8)
        t = A.RandomScale(scale_range=(-0.25, 0.35), p=1.0)
        t.set_random_seed(42)
        result = t(image=image)["image"]
        assert (
            hashlib.sha256(result.tobytes()).hexdigest()
            == "79cd1a26f4d0c7740bc9f69ddd9f4153c235df9bf11ad7775d64b4e1029c6d73"
        )
        assert result.shape == (114, 109, 3)

    @pytest.mark.parametrize("scale_range", SCALE_RANGES)
    @pytest.mark.parametrize("seed", [42, 99])
    def test_reproducibility(self, scale_range, seed):
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        t = A.RandomScale(scale_range=scale_range, p=1.0)
        t.set_random_seed(seed)
        r1 = t(image=image)["image"]
        t.set_random_seed(seed)
        r2 = t(image=image)["image"]
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.parametrize("scale_range", SCALE_RANGES)
    def test_round_trip(self, scale_range):
        t = A.RandomScale(scale_range=scale_range, p=0.8)
        restored = from_dict(to_dict(t))
        assert restored.scale_range == t.scale_range

    @pytest.mark.parametrize("scale_range", SCALE_RANGES)
    def test_records_sampled_scale_range_in_applied_config(self, scale_range):
        image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        t = A.RandomScale(scale_range=scale_range, p=1.0)
        t.set_random_seed(137)
        result = t(image=image)["image"]
        applied_config = t.get_applied_config()
        assert "scale_range" in applied_config
        assert result.shape == (32, 32, 3)

    @pytest.mark.parametrize("scale_range", SCALE_RANGES)
    def test_serialization_preserves_behavior(self, scale_range):
        image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        t = A.RandomScale(scale_range=scale_range, p=1.0)
        t.set_random_seed(137)
        result = t(image=image)["image"]

        serialized = A.to_dict(t)
        deserialized = A.from_dict(serialized)
        assert deserialized is not None
        assert deserialized.scale_range == t.scale_range

        deserialized.set_random_seed(137)
        np.testing.assert_array_equal(deserialized(image=image)["image"], result)

    @pytest.mark.parametrize(
        "scale_range",
        [(-0.5, 0.5), {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}],
    )
    def test_samples_different_values(self, scale_range):
        """With low < high, repeated calls produce different sampled values."""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        t = A.RandomScale(scale_range=scale_range, p=1.0)
        shapes = {t(image=image)["image"].shape for _ in range(10)}
        assert len(shapes) > 1

    @pytest.mark.parametrize(
        "scale_range",
        [
            {"x": (-0.2, 0.3)},  # missing y
            {"y": (-0.1, 0.15)},  # missing x
            {"x": (-0.2, 0.3), "y": (-0.1, 0.15), "z": (0.0, 0.0)},  # extra key
        ],
    )
    def test_rejects_invalid_dict_keys(self, scale_range):
        with pytest.raises(ValueError):
            A.RandomScale(scale_range=scale_range, p=1.0)

    @pytest.mark.parametrize(
        "scale_range",
        [
            pytest.param((-0.5, -1.0), id="tuple reversed bounds"),
            pytest.param((0.2, -0.1), id="tuple reversed bounds (cross-zero)"),
            pytest.param((-1.0, 0.5), id="tuple low == -1"),
            pytest.param((-2.0, 0.5), id="tuple low < -1"),
            pytest.param((-1.0, -1.0), id="tuple both -1"),
            pytest.param({"x": (-0.3, -0.5), "y": (0.0, 0.2)}, id="dict x reversed"),
            pytest.param({"x": (0.0, 0.2), "y": (-0.1, -0.3)}, id="dict y reversed"),
            pytest.param({"x": (-1.0, 0.5), "y": (0.0, 0.2)}, id="dict x low == -1"),
            pytest.param({"x": (-2.0, 0.5), "y": (0.0, 0.2)}, id="dict x low < -1"),
            pytest.param({"x": (0.0, 0.2), "y": (-1.0, 0.5)}, id="dict y low == -1"),
            pytest.param({"x": (0.0, 0.2), "y": (-2.0, 0.5)}, id="dict y low < -1"),
            pytest.param((float("inf"), 0.5), id="tuple inf"),
            pytest.param((-0.5, float("nan")), id="tuple nan"),
            pytest.param({"x": (float("-inf"), 0.5), "y": (0.0, 0.2)}, id="dict x -inf"),
        ],
    )
    def test_rejects_invalid_scale_range_values(self, scale_range):
        with pytest.raises(ValueError):
            A.RandomScale(scale_range=scale_range, p=1.0)

    def test_mixed_upscale_downscale_selects_inter_area(self):
        """area_for_downscale='image_mask' uses INTER_AREA for both image and mask
        when either axis shrinks.
        """
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        t = A.Compose(
            [
                A.RandomScale(
                    scale_range={"x": (1.0, 1.0), "y": (-0.5, -0.5)},
                    area_for_downscale="image_mask",
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                )
            ],
        )
        result = t(image=image, mask=mask)
        np.testing.assert_array_equal(
            result["image"],
            fgeometric.scale_xy(image, 2.0, 0.5, cv2.INTER_AREA),
        )
        np.testing.assert_array_equal(
            result["mask"],
            fgeometric.scale_xy(mask, 2.0, 0.5, cv2.INTER_AREA),
        )

    def test_all_targets(self, image, mask, bboxes, keypoints):
        scale = {"x": (-0.5, -0.5), "y": (0.0, 0.0)}
        t = A.Compose(
            [A.RandomScale(scale_range=scale, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
            keypoint_params=A.KeypointParams(coord_format="xy", remove_invisible=False),
        )
        data = t(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints, labels=[1, 2])
        h, w = image.shape[:2]
        new_w = max(1, round(w * 0.5))

        assert data["image"].shape == (h, new_w, 3)
        assert data["mask"].shape == (h, new_w)

        # HBB is scale-invariant in normalized coords, so denormalizing to the
        # output shape (100, 50) applies scale_x=0.5, scale_y=1.0 to pixel coords
        expected_bboxes = np.array(
            [[7.5, 12, 37.5, 30, 1], [27.5, 25, 45, 90, 2]],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(data["bboxes"], expected_bboxes, decimal=5)

        # keypoints_scale with scale_x=0.5, scale_y=1.0
        expected_keypoints = np.array(
            [[15, 20, 0, 0.5, 1], [10, 30, 60, 2.5, 2]],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(data["keypoints"], expected_keypoints, decimal=5)

        # OBB with the same non-uniform scale
        obb_img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        obb = np.array([[0.2, 0.3, 0.4, 0.5, 45.0]], dtype=np.float32)
        t_obb = A.Compose(
            [A.RandomScale(scale_range=scale, p=1.0)],
            bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb", label_fields=["labels"]),
        )
        result_obb = t_obb(image=obb_img, bboxes=obb, labels=[1])
        expected = fgeometric.resize_bboxes(
            obb,
            image_shape=(100, 200),
            output_shape=(100, 100),
            bbox_type="obb",
        )
        np.testing.assert_array_almost_equal(result_obb["bboxes"], expected, decimal=5)
