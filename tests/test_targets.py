import numpy as np
import pytest
import torch

import albumentations as A
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
        A.DualTransform.apply_to_mask: "apply",
        A.DualTransform.apply_to_masks: "apply_to_mask",
        A.DualTransform.apply_to_mask3d: "apply_to_mask",
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
        (A.DualTransform, Targets.MASK),
        (A.DualTransform, Targets.MASKS),
        (A.DualTransform, Targets.MASK3D),
        (A.Transform3D, Targets.MASK3D),
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
