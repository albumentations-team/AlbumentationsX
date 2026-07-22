"""Tests for RandomScale transform — uniform and anisotropic scaling."""

import numpy as np
import pytest

import albumentations as A
from albumentations.core.serialization import from_dict, to_dict

SCALE_RANGES = [(-0.5, -0.5), {"x": (-0.5, -0.5), "y": (-0.5, -0.5)}]


class TestRandomScale:
    @pytest.mark.parametrize("scale_range", SCALE_RANGES)
    def test_non_square_output(self, scale_range):
        """Fixed scale produces predictable output shape on non-square input."""
        image = np.random.randint(0, 256, (80, 120, 3), dtype=np.uint8)
        result = A.RandomScale(scale_range=scale_range, p=1.0)(image=image)
        assert result["image"].shape == (40, 60, 3)

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

    def test_mixed_upscale_downscale(self):
        """One axis upscales, the other downscales — output shape reflects both."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = A.RandomScale(
            scale_range={"x": (1.0, 1.0), "y": (-0.5, -0.5)},
            p=1.0,
        )(image=image)
        assert result["image"].shape == (50, 200, 3)

    def test_all_targets(self, image, mask, bboxes, keypoints):
        t = A.Compose(
            [A.RandomScale(scale_range={"x": (-0.5, -0.5), "y": (0.0, 0.0)}, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
            keypoint_params=A.KeypointParams(coord_format="xy", remove_invisible=False),
        )
        data = t(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints, labels=[1, 2])
        h, w = image.shape[:2]
        assert data["image"].shape == (h, max(1, round(w * 0.5)), 3)
        assert data["mask"].shape == (h, max(1, round(w * 0.5)))
        assert len(data["bboxes"]) == 2
        assert len(data["keypoints"]) == 2

    def test_obb_count_preserved(self):
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        obb = np.array([[0.2, 0.3, 0.4, 0.5, 45.0]], dtype=np.float32)
        t = A.Compose(
            [A.RandomScale(scale_range={"x": (-0.5, -0.5), "y": (0.0, 0.0)}, p=1.0)],
            bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb", label_fields=["labels"]),
        )
        result = t(image=image, bboxes=obb, labels=[1])
        assert len(result["bboxes"]) == 1
