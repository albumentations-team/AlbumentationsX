from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetSet
from albumentations.core.type_definitions import Targets
from tests.utils import get_primary_filtered_transform_params


@pytest.mark.parametrize(
    ("transform_cls", "params", "output_shape"),
    [
        (A.MaskDropout, {}, (8, 12, 3)),
        (A.ConstrainedCoarseDropout, {"mask_indices": [1]}, (8, 12, 3)),
        (A.CropNonEmptyMaskIfExists, {"height": 4, "width": 5}, (4, 5, 3)),
    ],
)
@pytest.mark.parametrize("tensor", [False, True])
def test_mask_dependent_transforms_leave_undeclared_collections_unchanged(transform_cls, params, output_shape, tensor):
    image = np.full((8, 12, 3), 137, dtype=np.uint8)
    mask = np.ones((8, 12), dtype=np.uint8)
    images = np.stack([image, image])
    masks = np.stack([mask, mask])
    if tensor:
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        masks = torch.from_numpy(masks)
    transform = transform_cls(p=1, **params)
    source_images, source_masks = images.clone() if tensor else images.copy(), masks.clone() if tensor else masks.copy()

    result = A.Compose([transform], telemetry=False)(image=image, mask=mask, images=images, masks=masks)

    assert "images" not in transform._key2func
    assert "masks" not in transform._key2func
    assert result["image"].shape == output_shape
    if tensor:
        torch.testing.assert_close(result["images"], source_images)
        torch.testing.assert_close(result["masks"], source_masks)
    else:
        np.testing.assert_array_equal(result["images"], source_images)
        np.testing.assert_array_equal(result["masks"], source_masks)


@pytest.mark.parametrize(
    ["transform_cls", "params"],
    get_primary_filtered_transform_params(
        base_classes=(A.BasicTransform,),
        except_augmentations={A.DualTransform, A.ImageOnlyTransform, A.Transform3D, A.VolumeOnlyTransform},
    ),
)
def test_declared_targets_have_active_handlers(transform_cls: type[A.BasicTransform], params: dict) -> None:
    transform = transform_cls(p=1, **params)
    raw_targets = transform._targets
    targets = raw_targets if isinstance(raw_targets, tuple) else (raw_targets,)
    target_names = {target.name.lower() if isinstance(target, Targets) else target for target in targets}
    stubs = {
        A.BasicTransform.apply,
        A.DualTransform.apply_to_bboxes,
        A.DualTransform.apply_to_keypoints,
        A.Transform3D.apply_to_volume,
        A.VolumeOnlyTransform.apply_to_volume,
    }
    inherited_dependencies = {
        A.BasicTransform.apply_to_images: "apply",
        A.BasicTransform.apply_to_volume: "apply_to_images",
        A.BasicTransform.apply_to_volumes: "apply_to_volume",
        A.DualTransform.apply_to_mask: "apply",
        A.DualTransform.apply_to_masks: "apply_to_mask",
        A.DualTransform.apply_to_mask3d: "apply_to_mask",
        A.DualTransform.apply_to_masks3d: "apply_to_mask3d",
        A.Transform3D.apply_to_mask3d: "apply_to_volume",
    }

    assert len(targets) == len(target_names)
    assert "user_data" not in target_names
    assert set(transform._key2func) == target_names
    for target_name, handler in transform._key2func.items():
        implementation = handler.__func__
        while True:
            assert implementation not in stubs, f"{transform_cls.__name__}.{target_name} resolves to a base-class stub"
            dependency = inherited_dependencies.get(implementation)
            if dependency is None:
                break
            implementation = getattr(transform, dependency).__func__


@pytest.mark.parametrize(
    ("base_cls", "target"),
    [
        (A.BasicTransform, Targets.IMAGES),
        (A.BasicTransform, Targets.VOLUME),
        (A.BasicTransform, Targets.VOLUMES),
        (A.DualTransform, Targets.MASK),
        (A.DualTransform, Targets.MASKS),
        (A.DualTransform, Targets.MASK3D),
        (A.DualTransform, Targets.MASKS3D),
        (A.Transform3D, Targets.MASK3D),
        (A.Transform3D, Targets.VOLUMES),
        (A.Transform3D, Targets.MASKS3D),
    ],
)
def test_declaration_check_rejects_inherited_wrapper_around_stub(base_cls, target) -> None:
    class BrokenTransform(base_cls):
        _targets = (target,)

    with pytest.raises(AssertionError, match="resolves to a base-class stub"):
        test_declared_targets_have_active_handlers(BrokenTransform, {})


@pytest.mark.parametrize("target", [Targets.IMAGE, Targets.IMAGES])
def test_image_and_collection_targets_use_independent_inherited_routes(target: Targets) -> None:
    class ImageRouteOnly(A.BasicTransform):
        _targets = (target,)

        def apply(self, image, **params):
            return 255 - image

    test_declared_targets_have_active_handlers(ImageRouteOnly, {})
    transform = ImageRouteOnly(p=1)
    images = np.arange(2 * 8 * 12 * 3, dtype=np.uint8).reshape(2, 8, 12, 3)
    name = target.name.lower()
    source = images[0] if target is Targets.IMAGE else images

    result = A.Compose([transform], strict=True, telemetry=False)(**{name: source})

    assert set(transform._key2func) == {name}
    np.testing.assert_array_equal(result[name], 255 - source)


def test_volume_collection_reuses_one_sampled_parameter_set_for_every_item() -> None:
    class SharedParameters3D(A.Transform3D):
        def __init__(self) -> None:
            super().__init__(p=1.0)
            self.sample_calls = 0
            self.applied_offsets: list[int] = []
            self.volume_shapes: list[tuple[int, ...]] = []

        def sample_parameters(
            self,
            params: dict[str, Any],
            data: dict[str, Any],
            targets: TargetSet,
            sampling: SamplingContext,
        ) -> SampledParams:
            self.sample_calls += 1
            return SampledParams(params={"offset": self.sample_calls})

        def apply_to_volume(self, volume: np.ndarray, offset: int, **params) -> np.ndarray:
            self.applied_offsets.append(offset)
            self.volume_shapes.append(volume.shape)
            return volume + offset

    transform = SharedParameters3D()
    volumes = np.zeros((2, 3, 4, 5, 1), dtype=np.uint8)
    masks3d = np.ones((3, 3, 4, 5), dtype=np.uint8)

    result = A.Compose([transform], strict=True, telemetry=False)(volumes=volumes, masks3d=masks3d)

    assert transform.sample_calls == 1
    assert transform.applied_offsets == [1] * 5
    assert transform.volume_shapes == [(3, 4, 5, 1)] * 5
    np.testing.assert_array_equal(result["volumes"], volumes + 1)
    np.testing.assert_array_equal(result["masks3d"], masks3d + 1)


def test_empty_volume_collections_keep_their_input_objects() -> None:
    volumes = np.empty((0, 3, 4, 5), dtype=np.uint8)
    masks3d = np.empty((0, 3, 4, 5), dtype=np.uint8)

    result = A.Compose([A.NoOp(p=1.0)], strict=True, telemetry=False)(volumes=volumes, masks3d=masks3d)

    assert result["volumes"] is volumes
    assert result["masks3d"] is masks3d


@pytest.mark.parametrize("target", ["volumes", "masks3d"])
def test_tensor_volume_collection_alias_uses_numpy_fallback(target: str) -> None:
    volumes = np.arange(2 * 3 * 5 * 7, dtype=np.uint8).reshape(2, 3, 5, 7, 1)
    extra = torch.from_numpy(volumes.copy()).permute(0, 4, 1, 2, 3)
    transform = A.Compose(
        [A.Flip3D(flip_axes=(0, 2), p=1)],
        additional_targets={"extra": target},
        strict=True,
        telemetry=False,
    )

    result = transform(**{target: volumes, "extra": extra})

    np.testing.assert_array_equal(result[target], volumes[:, ::-1, :, ::-1])
    torch.testing.assert_close(result["extra"], extra.flip((2, 4)))


@pytest.mark.parametrize("target", [Targets.VOLUMES, Targets.MASKS3D])
def test_tensor_collection_only_declaration_uses_native_single_handler(target: Targets) -> None:
    class CollectionOnly(A.Transform3D):
        _targets = (target,)

        def apply_to_volume(self, volume: np.ndarray | torch.Tensor, **params):
            assert isinstance(volume, torch.Tensor)
            return volume + 1

        def apply_to_mask3d(self, mask3d: np.ndarray | torch.Tensor, **params):
            return self.apply_to_volume(mask3d, **params)

    collection = torch.zeros((2, 1, 3, 5, 7), dtype=torch.uint8)
    name = target.name.lower()

    result = A.Compose([CollectionOnly(p=1)], strict=True, telemetry=False)(**{name: collection})

    torch.testing.assert_close(result[name], collection + 1)


def test_compose_checks_spatial_shapes_for_volume_collections() -> None:
    volumes = np.zeros((2, 3, 4, 5, 1), dtype=np.uint8)
    masks3d = np.zeros((3, 4, 4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="Depth, Height and Width"):
        A.Compose([A.NoOp(p=1.0)], telemetry=False)(volumes=volumes, masks3d=masks3d)


def test_dithering_reuses_random_noise_across_volume_collection() -> None:
    volume = np.arange(3 * 7 * 9 * 2, dtype=np.uint8).reshape(3, 7, 9, 2)
    volumes = np.stack([volume, volume])

    result = A.Compose([A.Dithering(method="random", n_colors=4, p=1.0)], seed=137, telemetry=False)(
        volumes=volumes,
    )

    np.testing.assert_array_equal(result["volumes"][0], result["volumes"][1])
    assert not np.array_equal(result["volumes"][0], volume)


def test_exposure_matching_uses_common_per_depth_gains_for_volume_collection() -> None:
    volumes = np.broadcast_to(
        np.array([[0.1, 0.2], [0.3, 0.6]], dtype=np.float32)[..., None, None, None], (2, 2, 3, 4, 1)
    ).copy()
    transform = A.ExposureMatching(target_mean_range=(0.4, 0.4), p=1.0)

    result = A.Compose([transform], strict=True, telemetry=False)(volumes=volumes)

    expected = np.broadcast_to(
        np.array([[0.2, 0.2], [0.6, 0.6]], dtype=np.float32)[..., None, None, None], volumes.shape
    )
    np.testing.assert_allclose(result["volumes"], expected)
