import numpy as np
import pytest

import albumentations as A
from albumentations.core.transform_params import (
    SampledParams,
    SampledParamsError,
    TargetParams,
    TargetRequirement,
    TargetSet,
)


def test_multiplicative_noise_groups_channel_vectors_by_actual_target_representation() -> None:
    image = np.ones((6, 7, 3), dtype=np.float32)
    volume = np.ones((2, 6, 7, 5), dtype=np.float32)
    transform = A.MultiplicativeNoise(multiplier=(0.5, 0.5), p=1.0)

    result = transform(image=image, volume=volume)
    sampled_params = SampledParams.from_dict(transform.get_applied_params())

    assert {tuple(group.targets) for group in sampled_params.groups} == {("image",), ("volume",)}
    assert result["image"].shape == image.shape
    assert result["volume"].shape == volume.shape
    np.testing.assert_allclose(result["image"], 0.5)
    np.testing.assert_allclose(result["volume"], 0.5)


def test_additive_noise_scales_materialized_maps_per_dtype_and_alias() -> None:
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    image2 = np.zeros((5, 6, 3), dtype=np.float32)
    transform = A.AdditiveNoise(
        noise_type="uniform",
        spatial_mode="per_pixel",
        noise_params={"ranges": [(0.1, 0.1)]},
        p=1.0,
    )
    transform.add_targets({"image2": "image"})

    result = transform(image=image, image2=image2)
    sampled_params = SampledParams.from_dict(transform.get_applied_params())

    assert {tuple(group.targets) for group in sampled_params.groups} == {("image",), ("image2",)}
    assert result["image"].dtype == image.dtype
    assert result["image2"].dtype == image2.dtype
    assert not np.array_equal(
        sampled_params.params_for("image")["noise_map"], sampled_params.params_for("image2")["noise_map"]
    )


def test_sampled_params_are_deterministically_ordered_and_schema_versioned() -> None:
    data = {
        "volume": np.zeros((2, 4, 5, 1), dtype=np.float32),
        "image2": np.zeros((4, 5, 1), dtype=np.float32),
        "image": np.zeros((4, 5, 1), dtype=np.float32),
    }
    targets = TargetSet.from_data(data, {"image": "image", "image2": "image", "volume": "volume"})
    assert [view.name for view in targets.ordered] == ["image", "image2", "volume"]

    sampled_params = SampledParams(
        shared={"shared": 1},
        groups=(
            TargetParams(
                targets=("image",),
                params={"specific": 2},
                requirements={"image": TargetRequirement()},
            ),
        ),
        target_schema=targets.schema(),
    )
    assert sampled_params.to_dict()["parameter_schema"] == 2
    assert sampled_params.params_for("image") == {"shared": 1, "specific": 2}


def test_legacy_flat_parameter_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported or legacy"):
        SampledParams.from_dict({"shape": (4, 5, 3), "noise_map": np.zeros((4, 5, 3))})


def test_replay_preserves_mixed_target_materialization() -> None:
    image = np.ones((6, 7, 3), dtype=np.float32)
    volume = np.ones((2, 6, 7, 5), dtype=np.float32)
    transform = A.ReplayCompose(
        [A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0)],
        seed=137,
    )
    first = transform(image=image, volume=volume)
    replayed = A.ReplayCompose.replay(first["replay"], image=image.copy(), volume=volume.copy())

    np.testing.assert_array_equal(first["image"], replayed["image"])
    np.testing.assert_array_equal(first["volume"], replayed["volume"])
    assert first["replay"]["transforms"][0]["params"]["parameter_schema"] == 2


def test_pixel_dropout_routes_mixed_alias_representations_by_actual_key() -> None:
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    image2 = np.full((4, 5, 5), 1.0, dtype=np.float32)
    transform = A.PixelDropout(dropout_prob=1.0, drop_value=0, p=1.0)
    transform.add_targets({"image2": "image"})

    result = transform(image=image, image2=image2)
    sampled_params = SampledParams.from_dict(transform.get_applied_params())

    assert {tuple(group.targets) for group in sampled_params.groups} == {("image",), ("image2",)}
    assert result["image"].shape == image.shape
    assert result["image2"].shape == image2.shape
    np.testing.assert_array_equal(result["image"], 0)
    np.testing.assert_array_equal(result["image2"], 0)


def test_common_spatial_shape_rejects_unaligned_targets_before_application() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image2 = np.zeros((3, 5, 3), dtype=np.uint8)
    transform = A.Rotate(limit=(0, 0), p=1.0)
    transform.add_targets({"image2": "image"})

    with pytest.raises(SampledParamsError, match="requires aligned spatial targets"):
        transform(image=image, image2=image2)


def test_sampled_params_reject_duplicate_keys_and_invalid_groups() -> None:
    targets = TargetSet.from_data({"image": np.zeros((2, 2, 1), dtype=np.uint8)}, {"image": "image"})

    with pytest.raises(SampledParamsError, match="cannot repeat"):
        TargetParams(
            targets=("image", "image"),
            params={"value": 1},
            requirements={"image": TargetRequirement()},
        )

    sampled_params = SampledParams(
        shared={"value": 1},
        groups=(
            TargetParams(
                targets=("image",),
                params={"value": 2},
                requirements={"image": TargetRequirement()},
            ),
        ),
    )
    with pytest.raises(SampledParamsError, match="duplicate keys"):
        sampled_params.validate(targets, {}, "Example")
