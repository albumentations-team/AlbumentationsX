import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import (
    SampledParams,
    SampledParamsError,
    TargetParams,
    TargetRequirement,
    TargetSet,
)
from tests.utils import make_sampling_args


def test_multiplicative_noise_shares_a_scalar_across_target_representations() -> None:
    image = np.ones((6, 7, 3), dtype=np.float32)
    volume = np.ones((2, 6, 7, 5), dtype=np.float32)
    transform = A.MultiplicativeNoise(multiplier=(0.5, 0.5), p=1.0)

    result = transform(image=image, volume=volume)
    sampled_params = SampledParams.from_dict(transform.get_applied_params())

    assert sampled_params.target_params == ()
    assert sampled_params.params["multiplier"] == 0.5
    assert result["image"].shape == image.shape
    assert result["volume"].shape == volume.shape
    np.testing.assert_allclose(result["image"], 0.5)
    np.testing.assert_allclose(result["volume"], 0.5)


def test_multiplicative_noise_elementwise_shared_channels_share_one_map() -> None:
    image = np.ones((6, 7, 3), dtype=np.float32)
    image2 = np.ones((6, 7, 5), dtype=np.float32)
    transform = A.MultiplicativeNoise(elementwise=True, per_channel=False, p=1.0)
    transform.add_targets({"image2": "image"})

    result = transform(image=image, image2=image2)
    sampled_params = SampledParams.from_dict(transform.get_applied_params())

    assert [target_params.targets for target_params in sampled_params.target_params] == [("image", "image2")]
    assert sampled_params.params_for("image")["multiplier"].shape == (6, 7, 1)
    np.testing.assert_array_equal(result["image"][:, :, 0], result["image"][:, :, 1])
    np.testing.assert_array_equal(result["image"][:, :, 0], result["image"][:, :, 2])
    np.testing.assert_array_equal(result["image2"][:, :, 0], result["image2"][:, :, 4])
    np.testing.assert_array_equal(result["image"][:, :, 0], result["image2"][:, :, 0])


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

    assert {tuple(target_params.targets) for target_params in sampled_params.target_params} == {("image",), ("image2",)}
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
        params={"value": 1},
        target_params=(
            TargetParams(
                targets=("image",),
                params={"specific": 2},
                requirements={"image": TargetRequirement()},
            ),
        ),
        target_schema=targets.schema(),
    )
    serialized = sampled_params.to_dict()
    assert set(serialized) == {"parameter_schema", "target_schema", "params", "target_params"}
    assert serialized["parameter_schema"] == 2
    assert sampled_params.params_for("image") == {"value": 1, "specific": 2}


def test_legacy_flat_parameter_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported or legacy"):
        SampledParams.from_dict({"shape": (4, 5, 3), "noise_map": np.zeros((4, 5, 3))})


def test_structured_payload_with_retired_field_names_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported or legacy"):
        SampledParams.from_dict(
            {"parameter_schema": 2, "target_schema": None, "common": {}, "target_params": []},
        )


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


def test_structured_payload_with_retired_requirement_fields_is_rejected() -> None:
    with pytest.raises(SampledParamsError, match="unsupported target parameter requirement"):
        SampledParams.from_dict(
            {
                "parameter_schema": 2,
                "target_schema": {"image": "image"},
                "params": {},
                "target_params": [
                    {
                        "targets": ["image"],
                        "params": {"noise_map": [1]},
                        "requirements": {
                            "image": {
                                "shape": None,
                                "spatial_shape": None,
                                "spatial_shape_suffix": None,
                                "channels": None,
                                "dtype": None,
                                "value_scale": None,
                                "layout": None,
                                "sampling_topology": None,
                                "dtype_scale": "uint8",
                            },
                        },
                    },
                ],
            },
        )


def test_pixel_dropout_routes_mixed_alias_representations_by_actual_key() -> None:
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    image2 = np.full((4, 5, 5), 1.0, dtype=np.float32)
    transform = A.PixelDropout(dropout_prob=1.0, drop_value=0, p=1.0)
    transform.add_targets({"image2": "image"})

    result = transform(image=image, image2=image2)
    sampled_params = SampledParams.from_dict(transform.get_applied_params())

    assert {tuple(target_params.targets) for target_params in sampled_params.target_params} == {("image",), ("image2",)}
    assert result["image"].shape == image.shape
    assert result["image2"].shape == image2.shape
    np.testing.assert_array_equal(result["image"], 0)
    np.testing.assert_array_equal(result["image2"], 0)


def test_equalize_materializes_callable_masks_for_each_alias() -> None:
    image = np.full((4, 5, 3), 137, dtype=np.uint8)
    image2 = np.full((6, 7, 3), 137, dtype=np.uint8)
    volume = np.full((2, 8, 9, 3), 137, dtype=np.uint8)
    sampled_shapes: list[tuple[int, ...]] = []

    def mask_for(image: np.ndarray) -> np.ndarray:
        sampled_shapes.append(image.shape)
        return np.ones((*image.shape[:2], 1), dtype=np.uint8)

    transform = A.Equalize(mask=mask_for, p=1.0)
    transform.add_targets({"image2": "image"})

    result = transform(image=image, image2=image2, volume=volume)
    sampled_params = SampledParams.from_dict(transform.get_applied_params())

    assert sampled_shapes == [image.shape, image2.shape, volume[0].shape]
    assert {target_params.targets for target_params in sampled_params.target_params} == {
        ("image",),
        ("image2",),
        ("volume",),
    }
    assert result["image"].shape == image.shape
    assert result["image2"].shape == image2.shape
    assert result["volume"].shape == volume.shape


def test_replay_rejects_target_specific_spatial_parameters_for_changed_alias_shape() -> None:
    transform = A.ReplayCompose(
        [A.PlasmaShadow(plasma_size=4, roughness=1.0, p=1.0)],
        additional_targets={"image2": "image"},
        is_check_shapes=False,
    )
    image = np.full((10, 12, 3), 120, dtype=np.uint8)
    image2 = np.full((8, 9, 3), 120, dtype=np.uint8)

    recorded = transform(image=image, image2=image2)["replay"]

    with pytest.raises(SampledParamsError, match="requirements do not match target 'image2'"):
        A.ReplayCompose.replay(
            recorded,
            image=image,
            image2=np.full((7, 9, 3), 120, dtype=np.uint8),
        )


def test_aligned_spatial_shape_rejects_unaligned_targets_before_application() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image2 = np.zeros((3, 5, 3), dtype=np.uint8)
    transform = A.Rotate(limit=(0, 0), p=1.0)
    transform.add_targets({"image2": "image"})

    with pytest.raises(SampledParamsError, match="requires aligned spatial targets"):
        transform(image=image, image2=image2)


def test_image_only_sampling_allows_unaligned_aliases() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image2 = np.zeros((3, 5, 3), dtype=np.uint8)
    transform = A.Compose(
        [
            A.AdditiveNoise(
                noise_type="uniform",
                spatial_mode="per_pixel",
                noise_params={"ranges": [(0.1, 0.1)]},
                p=1.0,
            ),
        ],
        additional_targets={"image2": "image"},
        is_check_shapes=False,
        strict=True,
    )

    result = transform(image=image, image2=image2)

    assert result["image"].shape == image.shape
    assert result["image2"].shape == image2.shape
    assert np.any(result["image"])
    assert np.any(result["image2"])


def test_channel_dropout_ignores_user_data_during_sampling() -> None:
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    user_data = {"source": "fixture"}
    transform = A.ChannelDropout(channel_drop_range=(1, 1), fill=0, p=1.0)

    result = transform(image=image, user_data=user_data)

    assert result["user_data"] == user_data
    assert sum(np.all(result["image"][:, :, channel] == 0) for channel in range(3)) == 1


def test_tensor_image_sequence_descriptor_uses_channel_first_sequence_layout() -> None:
    targets = TargetSet.from_data(
        {"images": torch.zeros((3, 5, 11, 13), dtype=torch.uint8)},
        {"images": "images"},
    )

    descriptor = targets.by_name("images").descriptor

    assert descriptor.layout == "images_clhw"
    assert descriptor.channels == 3
    assert descriptor.spatial_shape == (11, 13)
    assert descriptor.value_scale == 255


@pytest.mark.parametrize(
    ("transform", "parameter_name"),
    [
        (A.GaussNoise(p=1.0), "noise_map"),
        (A.AdditiveNoise(p=1.0), "noise_map"),
        (A.RGBShift(p=1.0), "noise_map"),
        (A.AtmosphericFog(p=1.0), "depth_map"),
    ],
)
def test_tensor_samplers_use_descriptor_value_scale(transform, parameter_name: str) -> None:
    data = {"image": torch.zeros((3, 5, 7), dtype=torch.uint8)}

    sampled_params = transform.sample_parameters(
        *make_sampling_args(transform, data),
        SamplingContext.from_owner(transform, {}),
    )

    assert parameter_name in sampled_params.params_for("image")


def test_rgb_shift_replay_rejects_changed_channel_count() -> None:
    recorded = A.ReplayCompose([A.RGBShift(p=1.0)], seed=137)(
        image=np.zeros((8, 9, 3), dtype=np.uint8),
    )["replay"]

    with pytest.raises(SampledParamsError, match="requirements do not match target 'image'"):
        A.ReplayCompose.replay(recorded, image=np.zeros((8, 9, 1), dtype=np.uint8))


def test_exposure_matching_replay_rejects_changed_batch_size() -> None:
    recorded = A.ReplayCompose([A.ExposureMatching(p=1.0)], seed=137)(
        images=np.zeros((2, 8, 9, 3), dtype=np.uint8),
    )["replay"]

    with pytest.raises(SampledParamsError, match="requirements do not match target 'images'"):
        A.ReplayCompose.replay(recorded, images=np.zeros((3, 8, 9, 3), dtype=np.uint8))


def test_pixel_dropout_attaches_bboxes_to_a_mask_group_without_an_image() -> None:
    transform = A.PixelDropout(mask_drop_value=0, p=1.0)
    data = {
        "mask": np.ones((5, 7), dtype=np.uint8),
        "bboxes": np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32),
    }

    sampled_params = transform.sample_parameters(
        *make_sampling_args(transform, data),
        SamplingContext.from_owner(transform, {}),
    )

    assert "drop_mask" in sampled_params.params_for("bboxes")


def test_sampled_params_reject_duplicate_keys_and_invalid_target_params() -> None:
    targets = TargetSet.from_data({"image": np.zeros((2, 2, 1), dtype=np.uint8)}, {"image": "image"})

    with pytest.raises(SampledParamsError, match="cannot repeat"):
        TargetParams(
            targets=("image", "image"),
            params={"value": 1},
            requirements={"image": TargetRequirement()},
        )

    sampled_params = SampledParams(
        params={"value": 1},
        target_params=(
            TargetParams(
                targets=("image",),
                params={"value": 2},
                requirements={"image": TargetRequirement()},
            ),
        ),
    )
    with pytest.raises(SampledParamsError, match="duplicate keys"):
        sampled_params.validate(targets, {}, "Example")
