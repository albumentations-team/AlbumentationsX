"""Tests for A.CustomTransformsApplyMixin combined with Base transform classes
- ImageOnlyTransform
- DualTransform
- Transform3D
"""

from typing import Any

import numpy as np
import pytest

import albumentations as A

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def uint8_image():
    rng = np.random.default_rng(137)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def uint8_mask():
    rng = np.random.default_rng(137)
    return rng.integers(0, 2, (64, 64), dtype=np.uint8)


@pytest.fixture
def volume():
    rng = np.random.default_rng(137)
    return rng.integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)


@pytest.fixture
def mask3d():
    rng = np.random.default_rng(137)
    return rng.integers(0, 2, (8, 64, 64), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Transformations combined with custom apply mixin class logic
# ─────────────────────────────────────────────────────────────────────────────


class BrightnessWithLabel(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
    """ImageOnlyTransform + one custom target: float label."""

    def get_params(self) -> dict[str, Any]:
        return {"factor": 0.5}

    def apply(self, img: np.ndarray, factor: float = 1.0, **p) -> np.ndarray:
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(img.dtype)

    def apply_to_label(self, label: float, factor: float = 1.0, **p) -> float:
        return float(min(1.0, max(0.0, label * factor)))


class FlipWithMetadata(A.CustomTransformsApplyMixin, A.DualTransform):
    """DualTransform + one custom target: metadata dict."""

    def apply(self, img: np.ndarray, **p) -> np.ndarray:
        return np.fliplr(img)

    def apply_to_mask(self, mask: np.ndarray, **p) -> np.ndarray:
        return np.fliplr(mask)

    def apply_to_metadata(self, metadata: dict, **p) -> dict:
        return {**metadata, "flipped": not metadata.get("flipped", False)}


class RotateWithLabel(A.CustomTransformsApplyMixin, A.DualTransform):
    """DualTransform + integer rotation label."""

    def get_params(self) -> dict[str, Any]:
        return {"factor": 1}  # fixed for deterministic tests

    def apply(self, img: np.ndarray, factor: int = 0, **p) -> np.ndarray:
        return np.rot90(img, factor)

    def apply_to_mask(self, mask: np.ndarray, factor: int = 0, **p) -> np.ndarray:
        return np.rot90(mask, factor)

    def apply_to_label(self, label: int, factor: int = 0, **p) -> int:
        return (label + factor) % 4


class MultiTargetDual(A.CustomTransformsApplyMixin, A.DualTransform):
    """DualTransform + two custom targets."""

    def get_params(self) -> dict[str, Any]:
        return {"factor": 2}

    def apply(self, img: np.ndarray, **p) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **p) -> np.ndarray:
        return mask

    def apply_to_label(self, label: int, factor: int = 1, **p) -> int:
        return label + factor

    def apply_to_weight(self, weight: float, factor: int = 1, **p) -> float:
        return weight / factor


class VolumeWithLabel(A.CustomTransformsApplyMixin, A.Transform3D):
    """Transform3D + integer label."""

    def get_params(self) -> dict[str, Any]:
        return {"factor": 1}

    def apply(self, img: np.ndarray, **p) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **p) -> np.ndarray:
        return mask

    def apply_to_mask3d(self, mask3d: np.ndarray, **p) -> np.ndarray:
        return mask3d

    def apply_to_volume(self, volume: np.ndarray, **p) -> np.ndarray:
        return volume

    def apply_to_label(self, label: int, factor: int = 0, **p) -> int:
        return label + factor


# ─────────────────────────────────────────────────────────────────────────────
# ImageOnlyTransform + CustomTransformsApplyMixin
# ─────────────────────────────────────────────────────────────────────────────


class TestImageOnlyTransformWithMixin:
    def test_custom_key_registered_in_key2func(self):
        t = BrightnessWithLabel(p=1.0)
        assert "label" in t._key2func

    def test_builtin_image_key_still_registered(self):
        t = BrightnessWithLabel(p=1.0)
        assert "image" in t._key2func

    def test_apply_to_label_called_with_correct_params(self, uint8_image):
        """factor=0.5 is fixed in get_params; label must be halved."""
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image, label=0.8)
        assert abs(out["label"] - 0.4) < 1e-6

    def test_label_clamped_to_zero_lower_bound(self, uint8_image):
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image, label=0.0)
        assert out["label"] == 0.0

    def test_label_clamped_to_one_upper_bound(self, uint8_image):
        """factor=0.5 cannot push label above 1.0, but test the clamp logic."""
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image, label=1.0)
        assert 0.0 <= out["label"] <= 1.0

    def test_p_zero_leaves_label_unchanged(self, uint8_image):
        t = BrightnessWithLabel(p=0.0)
        out = t(image=uint8_image, label=0.7)
        assert out["label"] == 0.7

    def test_p_zero_leaves_image_unchanged(self, uint8_image):
        t = BrightnessWithLabel(p=0.0)
        out = t(image=uint8_image, label=0.5)
        np.testing.assert_array_equal(out["image"], uint8_image)

    def test_missing_label_key_does_not_crash(self, uint8_image):
        """Pipeline call without label= must not raise."""
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image)
        assert "label" not in out

    def test_image_is_transformed(self, uint8_image):
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image, label=0.5)
        # factor=0.5 darkens the image
        assert out["image"].mean() < uint8_image.mean()

    def test_image_shape_preserved(self, uint8_image):
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image, label=0.5)
        assert out["image"].shape == uint8_image.shape

    def test_image_dtype_preserved(self, uint8_image):
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image, label=0.5)
        assert out["image"].dtype == uint8_image.dtype

    def test_works_inside_compose(self, uint8_image):
        pipeline = A.Compose([BrightnessWithLabel(p=1.0)])
        out = pipeline(image=uint8_image, label=0.8)
        assert "label" in out
        assert "image" in out
        assert abs(out["label"] - 0.4) < 1e-6

    @pytest.mark.parametrize("label", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_label_range_preserved(self, uint8_image, label):
        t = BrightnessWithLabel(p=1.0)
        out = t(image=uint8_image, label=label)
        assert 0.0 <= out["label"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# DualTransform + CustomTransformsApplyMixin
# ─────────────────────────────────────────────────────────────────────────────


class TestDualTransformWithMetadataParameter:
    def test_custom_key_registered_in_key2func(self):
        t = FlipWithMetadata(p=1.0)
        assert "metadata" in t._key2func

    def test_builtin_mask_key_still_registered(self):
        t = FlipWithMetadata(p=1.0)
        assert "mask" in t._key2func

    def test_metadata_updated_on_flip(self, uint8_image, uint8_mask):
        t = FlipWithMetadata(p=1.0)
        out = t(image=uint8_image, mask=uint8_mask, metadata={"source": "cam"})
        assert out["metadata"]["flipped"] is True
        assert out["metadata"]["source"] == "cam"

    def test_metadata_flipped_toggles_on_double_flip(self, uint8_image, uint8_mask):
        t = FlipWithMetadata(p=1.0)
        out1 = t(image=uint8_image, mask=uint8_mask, metadata={"flipped": False})
        out2 = t(image=out1["image"], mask=out1["mask"], metadata=out1["metadata"])
        assert out2["metadata"]["flipped"] is False

    def test_p_zero_leaves_metadata_unchanged(self, uint8_image, uint8_mask):
        t = FlipWithMetadata(p=0.0)
        meta = {"source": "cam", "id": 42}
        out = t(image=uint8_image, mask=uint8_mask, metadata=meta)
        assert out["metadata"] == meta

    def test_mask_flipped_in_sync_with_image(self, uint8_image, uint8_mask):
        t = FlipWithMetadata(p=1.0)
        out = t(image=uint8_image, mask=uint8_mask, metadata={})
        np.testing.assert_array_equal(out["image"], np.fliplr(uint8_image))
        np.testing.assert_array_equal(out["mask"], np.fliplr(uint8_mask))

    def test_missing_metadata_does_not_crash(self, uint8_image, uint8_mask):
        t = FlipWithMetadata(p=1.0)
        out = t(image=uint8_image, mask=uint8_mask)
        assert "metadata" not in out

    def test_original_metadata_dict_not_mutated(self, uint8_image, uint8_mask):
        t = FlipWithMetadata(p=1.0)
        original = {"source": "cam", "id": 1}
        original_copy = dict(original)
        t(image=uint8_image, mask=uint8_mask, metadata=original)
        assert original == original_copy

    def test_works_inside_compose(self, uint8_image, uint8_mask):
        pipeline = A.Compose([FlipWithMetadata(p=1.0)])
        out = pipeline(image=uint8_image, mask=uint8_mask, metadata={"id": 5})
        assert out["metadata"]["flipped"] is True


# ─────────────────────────────────────────────────────────────────────────────
# DualTransform + CustomTransformsApplyMixin (integer label)
# ─────────────────────────────────────────────────────────────────────────────


class TestDualTransformWithIntegerLabelParameter:
    def test_custom_key_registered_in_key2func(self):
        t = RotateWithLabel(p=1.0)
        assert "label" in t._key2func

    def test_label_updated_by_factor(self, uint8_image):
        """factor=1 is fixed; (label + 1) % 4."""
        t = RotateWithLabel(p=1.0)
        out = t(image=uint8_image, label=0)
        assert out["label"] == 1

    @pytest.mark.parametrize(
        ["initial_label", "expected_label"],
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # wraps around
        ],
    )
    def test_label_modular_arithmetic(self, uint8_image, initial_label, expected_label):
        t = RotateWithLabel(p=1.0)
        out = t(image=uint8_image, label=initial_label)
        assert out["label"] == expected_label

    def test_label_always_in_valid_range(self, uint8_image):
        """Run many times; label must stay in {0, 1, 2, 3}."""
        t = RotateWithLabel(p=1.0)
        for label in range(4):
            out = t(image=uint8_image, label=label)
            assert out["label"] in range(4)

    def test_p_zero_leaves_label_unchanged(self, uint8_image):
        t = RotateWithLabel(p=0.0)
        out = t(image=uint8_image, label=2)
        assert out["label"] == 2

    def test_mask_and_label_both_transformed(self, uint8_image, uint8_mask):
        t = RotateWithLabel(p=1.0)
        out = t(image=uint8_image, mask=uint8_mask, label=0)
        np.testing.assert_array_equal(out["image"], np.rot90(uint8_image, 1))
        np.testing.assert_array_equal(out["mask"], np.rot90(uint8_mask, 1))
        assert out["label"] == 1

    def test_image_shape_preserved_after_rotation(self, uint8_image):
        t = RotateWithLabel(p=1.0)
        out = t(image=uint8_image, label=0)
        assert out["image"].shape == uint8_image.shape

    def test_works_inside_compose(self, uint8_image, uint8_mask):
        pipeline = A.Compose([RotateWithLabel(p=1.0)])
        out = pipeline(image=uint8_image, mask=uint8_mask, label=3)
        assert out["label"] == 0  # (3+1)%4


# ─────────────────────────────────────────────────────────────────────────────
# DualTransform + CustomTransformsApplyMixin (multiple parameters)
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleCustomParameters:
    def test_all_custom_keys_registered(self):
        t = MultiTargetDual(p=1.0)
        assert "label" in t._key2func
        assert "weight" in t._key2func

    def test_builtin_keys_not_displaced(self):
        t = MultiTargetDual(p=1.0)
        assert "image" in t._key2func
        assert "mask" in t._key2func

    def test_both_custom_targets_transformed(self, uint8_image, uint8_mask):
        """factor=2: label += 2, weight /= 2."""
        t = MultiTargetDual(p=1.0)
        out = t(image=uint8_image, mask=uint8_mask, label=1, weight=0.8)
        assert out["label"] == 3
        assert abs(out["weight"] - 0.4) < 1e-6

    def test_only_present_keys_are_transformed(self, uint8_image):
        """Passing label but not weight must not crash."""
        t = MultiTargetDual(p=1.0)
        out = t(image=uint8_image, label=1)
        assert out["label"] == 3
        assert "weight" not in out

    def test_p_zero_leaves_all_custom_targets_unchanged(self, uint8_image, uint8_mask):
        t = MultiTargetDual(p=0.0)
        out = t(image=uint8_image, mask=uint8_mask, label=2, weight=0.6)
        assert out["label"] == 2
        assert out["weight"] == 0.6

    @pytest.mark.parametrize(
        ["label", "weight"],
        [
            (0, 1.0),
            (1, 0.5),
            (2, 0.25),
            (3, 0.0),
        ],
    )
    def test_parametrized_custom_targets(self, uint8_image, label, weight):
        t = MultiTargetDual(p=1.0)
        out = t(image=uint8_image, label=label, weight=weight)
        assert out["label"] == label + 2
        assert abs(out["weight"] - weight / 2) < 1e-6

    def test_works_inside_compose(self, uint8_image, uint8_mask):
        pipeline = A.Compose([MultiTargetDual(p=1.0)])
        out = pipeline(image=uint8_image, mask=uint8_mask, label=0, weight=1.0)
        assert out["label"] == 2
        assert abs(out["weight"] - 0.5) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Transform3D + CustomTransformsApplyMixin
# ─────────────────────────────────────────────────────────────────────────────


class TestTransform3DWithMixin:
    def test_custom_key_registered_in_key2func(self):
        t = VolumeWithLabel(p=1.0)
        assert "label" in t._key2func

    def test_volume_key_still_registered(self):
        t = VolumeWithLabel(p=1.0)
        assert "volume" in t._key2func

    def test_label_updated_when_volume_present(self, volume, mask3d):
        t = VolumeWithLabel(p=1.0)
        out = t(volume=volume, mask3d=mask3d, label=2)
        assert out["label"] == 3  # factor=1

    def test_p_zero_leaves_label_unchanged(self, volume, mask3d):
        t = VolumeWithLabel(p=0.0)
        out = t(volume=volume, mask3d=mask3d, label=1)
        assert out["label"] == 1

    def test_volume_shape_preserved(self, volume, mask3d):
        t = VolumeWithLabel(p=1.0)
        out = t(volume=volume, mask3d=mask3d, label=0)
        assert out["volume"].shape == volume.shape

    def test_mask3d_shape_preserved(self, volume, mask3d):
        t = VolumeWithLabel(p=1.0)
        out = t(volume=volume, mask3d=mask3d, label=0)
        assert out["mask3d"].shape == mask3d.shape

    def test_missing_label_does_not_crash(self, volume, mask3d):
        t = VolumeWithLabel(p=1.0)
        out = t(volume=volume, mask3d=mask3d)
        assert "label" not in out


# ─────────────────────────────────────────────────────────────────────────────
# params passthrough (shared params dict across all apply_* methods)
# ─────────────────────────────────────────────────────────────────────────────


class TestParamsPassthrough:
    """All apply_* methods — built-in and custom — must receive the same params."""

    def test_custom_apply_receives_get_params_values(self, uint8_image):
        """Factor from get_params must reach apply_to_label unchanged."""
        received = {}

        class _Capture(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            def get_params(self):
                return {"factor": 7}

            def apply(self, img, factor=0, **p):
                received["apply_factor"] = factor
                return img

            def apply_to_label(self, label, factor=0, **p):
                received["label_factor"] = factor
                return label

        t = _Capture(p=1.0)
        t(image=uint8_image, label=0)
        assert received["apply_factor"] == 7
        assert received["label_factor"] == 7

    def test_multiple_custom_targets_receive_same_params(self, uint8_image):
        received = {}

        class _CaptureMulti(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            def get_params(self):
                return {"alpha": 3, "beta": 9}

            def apply(self, img, **p):
                return img

            def apply_to_label(self, label, alpha=0, beta=0, **p):
                received["label"] = (alpha, beta)
                return label

            def apply_to_score(self, score, alpha=0, beta=0, **p):
                received["score"] = (alpha, beta)
                return score

        t = _CaptureMulti(p=1.0)
        t(image=uint8_image, label=0, score=1.0)
        assert received["label"] == (3, 9)
        assert received["score"] == (3, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Check MRO safety: built-in targets are never overwritten
# ─────────────────────────────────────────────────────────────────────────────


class TestBuiltinPriority:
    def test_apply_to_mask_not_overwritten_by_mixin(self, uint8_image, uint8_mask):
        """apply_to_mask is a built-in on DualTransform; mixin must not shadow it."""

        class _MaskTransform(A.CustomTransformsApplyMixin, A.DualTransform):
            def apply(self, img, **p):
                return np.zeros_like(img)

            def apply_to_mask(self, mask, **p):
                # This is override, but main function from base class should still be called
                return np.ones_like(mask) * 255

            def apply_to_label(self, label, **p):
                return label + 1

        t = _MaskTransform(p=1.0)
        out = t(image=uint8_image, mask=uint8_mask, label=0)
        np.testing.assert_array_equal(out["mask"], np.ones_like(uint8_mask) * 255)
        assert out["label"] == 1

    def test_user_data_key_not_duplicated(self):
        """user_data must appear exactly once in _key2func."""

        class _WithUserData(A.CustomTransformsApplyMixin, A.DualTransform):
            def apply(self, img, **p):
                return img

            def apply_to_mask(self, mask, **p):
                return mask

            def apply_to_user_data(self, data, **p):
                return data

        t = _WithUserData(p=1.0)
        # Routing must point to apply_to_user_data.
        assert t._key2func["user_data"] == t.apply_to_user_data


# ─────────────────────────────────────────────────────────────────────────────
# ReplayCompose integration
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayComposeIntegration:
    """Custom apply targets survive ReplayCompose record and replay."""

    def test_replay_compose_custom_target_record_and_replay(self, uint8_image):
        class FlipWithLabel(A.CustomTransformsApplyMixin, A.HorizontalFlip):
            def apply_to_label(self, label: int, **params: Any) -> int:
                return (label + 1) % 4

        transform = A.ReplayCompose([FlipWithLabel(p=1.0)], seed=137)
        result = transform(image=uint8_image, label=0)
        assert "replay" in result
        assert result["label"] == 1

        replayed = A.ReplayCompose.replay(result["replay"], image=uint8_image, label=0)
        assert replayed["label"] == 1

    def test_replay_compose_skips_custom_target_when_not_applied(self, uint8_image):
        """If transform was not applied when recording, replay also skips custom target."""

        class FlipWithLabel(A.CustomTransformsApplyMixin, A.HorizontalFlip):
            def apply_to_label(self, label: int, **params: Any) -> int:
                return label + 100  # obvious change

        transform = A.ReplayCompose([FlipWithLabel(p=0.5)], seed=137)
        # Run until we get one where transform was NOT applied (label unchanged)
        for _ in range(20):
            result = transform(image=uint8_image, label=5)
            if result["label"] == 5:  # not applied
                saved = result["replay"]
                break
        else:
            pytest.skip("RNG always applied transform in 20 tries")
        # On replay, recorded applied=False means label should still be passed through unchanged
        replayed = A.ReplayCompose.replay(saved, image=uint8_image, label=5)
        assert replayed["label"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# get_params_dependent_on_data with custom targets
# ─────────────────────────────────────────────────────────────────────────────


class TestGetParamsDependentOnData:
    """Custom targets can be used in targets_as_params and get_params_dependent_on_data."""

    def test_custom_target_in_targets_as_params(self, uint8_image):
        class DataAwareLabelTransform(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            targets_as_params = ("label",)

            def get_params(self) -> dict[str, Any]:
                return {"base": 10}

            def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
                label = data.get("label", 0)
                return {"offset": label * 2}

            def apply(self, img: np.ndarray, base: int = 0, offset: int = 0, **p) -> np.ndarray:
                return img

            def apply_to_label(self, label: int, base: int = 0, offset: int = 0, **p) -> int:
                return label + base + offset

        t = DataAwareLabelTransform(p=1.0)
        out = t(image=uint8_image, label=3)
        # offset = 3*2 = 6, base = 10, label_out = 3 + 10 + 6 = 19
        assert out["label"] == 19

    def test_missing_custom_target_in_targets_as_params_raises(self, uint8_image):
        class RequiresLabel(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            targets_as_params = ("label",)

            def get_params(self) -> dict[str, Any]:
                return {}

            def apply(self, img: np.ndarray, **p) -> np.ndarray:
                return img

        t = RequiresLabel(p=1.0)
        with pytest.raises(ValueError, match=r"requires.*label"):
            t(image=uint8_image)


# ─────────────────────────────────────────────────────────────────────────────
# add_targets with custom keys
# ─────────────────────────────────────────────────────────────────────────────


class TestAddTargetsWithCustomKeys:
    """additional_targets can alias custom apply keys."""

    def test_add_targets_aliases_custom_key(self, uint8_image):
        class TransformWithLabel(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            def get_params(self) -> dict[str, Any]:
                return {}

            def apply(self, img: np.ndarray, **p) -> np.ndarray:
                return img

            def apply_to_label(self, label: int, **p) -> int:
                return label + 1

        t = TransformWithLabel(p=1.0)
        t.add_targets({"rotation_label": "label"})
        assert "rotation_label" in t._available_keys
        assert "label" in t._available_keys
        assert t._key2func["rotation_label"] == t._key2func["label"]

        out = t(image=uint8_image, rotation_label=5)
        assert out["rotation_label"] == 6

    def test_compose_with_additional_targets_custom_key(self, uint8_image):
        class TransformWithLabel(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            def get_params(self) -> dict[str, Any]:
                return {}

            def apply(self, img: np.ndarray, **p) -> np.ndarray:
                return img

            def apply_to_label(self, label: int, **p) -> int:
                return label + 1

        pipeline = A.Compose(
            [TransformWithLabel(p=1.0)],
            additional_targets={"rotation_label": "label"},
        )
        out = pipeline(image=uint8_image, rotation_label=2)
        assert out["rotation_label"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Serialization (to_dict / from_dict)
# ─────────────────────────────────────────────────────────────────────────────


class TestSerialization:
    """Custom apply transforms serialize and deserialize correctly."""

    def test_to_dict_from_dict_preserves_behavior(self, uint8_image):
        t = BrightnessWithLabel(p=1.0)
        out_before = t(image=uint8_image, label=0.5)

        serialized = A.to_dict(t)
        restored = A.from_dict(serialized)
        out_after = restored(image=uint8_image, label=0.5)

        assert out_before["label"] == out_after["label"]
        np.testing.assert_array_equal(out_before["image"], out_after["image"])

    def test_compose_with_custom_apply_serializes(self, uint8_image):
        pipeline = A.Compose([BrightnessWithLabel(p=1.0), FlipWithMetadata(p=0.0)])
        out = pipeline(image=uint8_image, label=0.6, metadata={"x": 1})
        serialized = A.to_dict(pipeline)
        restored = A.from_dict(serialized)
        out2 = restored(image=uint8_image, label=0.6, metadata={"x": 1})
        assert out["label"] == out2["label"]
        assert out["metadata"] == out2["metadata"]


# ─────────────────────────────────────────────────────────────────────────────
# available_keys and multiple transforms
# ─────────────────────────────────────────────────────────────────────────────


class TestAvailableKeysAndComposition:
    """available_keys includes custom keys; Compose aggregates them."""

    def test_available_keys_includes_custom(self):
        t = BrightnessWithLabel(p=1.0)
        assert "label" in t.available_keys
        assert "image" in t.available_keys

    def test_multiple_transforms_different_custom_targets(self, uint8_image, uint8_mask):
        """Compose with transforms that have different custom targets."""
        pipeline = A.Compose(
            [
                BrightnessWithLabel(p=1.0),
                FlipWithMetadata(p=1.0),
            ],
        )
        out = pipeline(
            image=uint8_image,
            mask=uint8_mask,
            label=0.8,
            metadata={"id": 1},
        )
        assert "label" in out
        assert "metadata" in out
        assert abs(out["label"] - 0.4) < 1e-6
        assert out["metadata"]["flipped"] is True

    def test_multiple_transforms_same_custom_target_chained(self, uint8_image):
        """Two transforms both with apply_to_label; label flows through both in sequence."""

        # First: label += 1. Second: label *= 2. Input 5 -> 6 -> 12.
        class AddOne(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            def get_params(self) -> dict[str, Any]:
                return {}

            def apply(self, img: np.ndarray, **p) -> np.ndarray:
                return img

            def apply_to_label(self, label: int, **p) -> int:
                return label + 1

        class MulTwo(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            def get_params(self) -> dict[str, Any]:
                return {}

            def apply(self, img: np.ndarray, **p) -> np.ndarray:
                return img

            def apply_to_label(self, label: int, **p) -> int:
                return label * 2

        pipeline = A.Compose([AddOne(p=1.0), MulTwo(p=1.0)])
        out = pipeline(image=uint8_image, label=5)
        assert out["label"] == 12  # 5+1=6, 6*2=12

    def test_inheritance_adds_custom_apply(self, uint8_image):
        """Subclass can add another apply_to_ method."""

        class BaseWithLabel(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
            def get_params(self) -> dict[str, Any]:
                return {}

            def apply(self, img: np.ndarray, **p) -> np.ndarray:
                return img

            def apply_to_label(self, label: int, **p) -> int:
                return label + 1

        class ExtendedWithScore(BaseWithLabel):
            def apply_to_score(self, score: float, **p) -> float:
                return score * 2

        t = ExtendedWithScore(p=1.0)
        assert "label" in t._key2func
        assert "score" in t._key2func
        out = t(image=uint8_image, label=1, score=0.5)
        assert out["label"] == 2
        assert out["score"] == 1.0
