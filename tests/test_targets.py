import pytest

import albumentations as A
from albumentations.core.type_definitions import Targets
from tests.utils import get_primary_public_transform_params


@pytest.mark.parametrize(
    ["transform_cls", "params"],
    [*get_primary_public_transform_params(), (A.ToTensorV2, {}), (A.ToTensor3D, {})],
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

    assert len(targets) == len(target_names)
    assert "user_data" not in target_names
    assert set(transform._key2func) == target_names
    for target_name, handler in transform._key2func.items():
        assert handler.__func__ not in stubs, f"{transform_cls.__name__}.{target_name} resolves to a base-class stub"
