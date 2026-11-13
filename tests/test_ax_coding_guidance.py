"""Contract tests for the unified AX coding-guidance hook."""

from __future__ import annotations

import pytest

from tools.ax_coding_guidance import run_sources


def rule_ids(sources: dict[str, str]) -> list[str]:
    return [diagnostic.rule for diagnostic in run_sources(sources)]


@pytest.mark.parametrize(
    ("source", "symbol", "replacement"),
    [
        ("import torch\nresult = torch.from_numpy(array)", "torch.from_numpy", "numpy_to_tensor_storage"),
        (
            "def convert():\n    import torch as pt\n    return pt.from_numpy(array)",
            "torch.from_numpy",
            "numpy_to_tensor_storage",
        ),
        ("from torch import from_numpy as wrap\nresult = wrap(array)", "torch.from_numpy", "numpy_to_tensor_storage"),
        ("result = tensor.detach().numpy(force=True)", ".numpy", "tensor_to_numpy_storage"),
        ("convert = tensor.numpy", ".numpy", "tensor_to_numpy_storage"),
    ],
)
def test_tensor_bridge_rejects_direct_pytorch_conversions(source: str, symbol: str, replacement: str) -> None:
    diagnostics = run_sources({"albumentations/core/example.py": source})

    (diagnostic,) = diagnostics
    assert diagnostic.rule == "AXG028"
    assert diagnostic.symbol == symbol
    assert f"albucore.tensor_bridge.{replacement}" in diagnostic.message


def test_tensor_bridge_allows_shared_bridge_and_non_package_sources() -> None:
    assert "AXG028" not in rule_ids(
        {
            "albumentations/core/example.py": """
import torch
from albucore.tensor_bridge import numpy_to_tensor_storage, tensor_to_numpy_storage
tensor = numpy_to_tensor_storage(array)
result = tensor_to_numpy_storage(tensor)
index = torch.as_tensor([0])
example = "tensor.numpy()"
""",
            "tests/test_example.py": "result = torch.from_numpy(array).numpy()",
            "benchmark/bench_example.py": "result = torch.from_numpy(array).numpy()",
        }
    )


def test_init_schema_defaults_are_checked_without_base_model_false_positives() -> None:
    assert rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class Example(DualTransform):
    class InitSchema:
        value: int = 1
""",
        },
    ) == ["AXG001", "AXG016"]


def test_init_schema_allows_validation_metadata_but_not_discriminator_defaults() -> None:
    assert "AXG001" not in rule_ids(
        {
            "albumentations/augmentations/example.py": """
from typing import ClassVar, Literal
class DualTransform: pass
class Example(DualTransform):
    class InitSchema:
        model_config = object()
        _private: ClassVar[int] = 1
        value: int = Field(gt=0)
""",
        },
    )
    assert "AXG001" in rule_ids(
        {
            "albumentations/augmentations/example.py": """
from typing import Literal
class DualTransform: pass
class Example(DualTransform):
    class InitSchema:
        kind: Literal['x'] = 'x'
""",
        },
    )


def test_transform_api_rules_cover_apply_sampling_random_and_names() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
import numpy as np
import random
class DualTransform: pass
class RandomNew(DualTransform):
    def apply(self, image, value=1):
        return sampling.py_random.random()
    def sample_parameters(self, self2, inputs, sampling):
        return {'x': np.random.uniform(0, 1), 'y': random.random(), 'z': sampling.py_random.random()}
""",
        },
    )
    assert {"AXG002", "AXG004", "AXG005", "AXG006", "AXG007"} <= set(ids)


@pytest.mark.parametrize("base", ["BaseCompose", "DualTransform"])
@pytest.mark.parametrize(
    "expression",
    [
        "self.py_random.random()",
        "self.random_generator",
        "np.random.default_rng()",
        "SamplingContext.from_seed(137)",
        "SamplingContext.from_owner(self)",
    ],
)
def test_builtin_sampling_cannot_open_an_independent_rng_path(base: str, expression: str) -> None:
    ids = rule_ids(
        {
            "albumentations/core/composition.py": f"""
import numpy as np
from albumentations.core.invocation import SamplingContext
class {base}: pass
class Example({base}):
    def sample_parameters(self, params, data, targets, sampling):
        rng = {expression}
        return rng
""",
        },
    )
    assert "AXG027" in ids


def test_rng_policy_forwarding_and_explicit_context_sampling_remain_allowed() -> None:
    ids = rule_ids(
        {
            "albumentations/core/composition.py": """
import numpy as np
class BaseCompose: pass
class Example(BaseCompose):
    def reconstruct(self, other):
        other.set_random_state(self.random_generator, self.py_random)
        other.set_random_state(random_generator=self.random_generator, py_random=self.py_random)
    def apply_in_invocation(self, invocation):
        return invocation.py_random.random(), invocation.random_generator.uniform()
    def sample_parameters(self, params, data, targets, sampling):
        return sampling.py_random.random(), sampling.random_generator.uniform()
class DualTransform: pass
class Dropout(DualTransform):
    def apply(self, img, seed):
        return np.random.default_rng(seed).random(img.shape)
""",
        },
    )
    assert "AXG027" not in ids


@pytest.mark.parametrize(
    ("method", "expression", "prohibited"),
    [
        pytest.param("apply", "np.random.default_rng(0)", True, id="constant-seed"),
        pytest.param("apply", "np.random.default_rng()", True, id="unseeded"),
        pytest.param("apply", "np.random.default_rng(seed=seed)", False, id="keyword-seed"),
        pytest.param("helper", "np.random.default_rng(seed)", True, id="seeded-helper"),
        pytest.param("apply", "random.Random(seed)", False, id="python-generator"),
        pytest.param("apply", "np.random.Generator(seed)", True, id="bit-generator-wrapper"),
        pytest.param("apply", "np.random.RandomState(seed)", True, id="legacy-generator"),
        pytest.param("apply", "SamplingContext.from_seed(seed)", True, id="replacement-sampling-context"),
        pytest.param("apply", "SamplingContext.from_owner(seed)", True, id="replacement-owner-context"),
    ],
)
def test_recorded_seed_exemption_only_allows_plain_generator_reconstruction(
    method: str, expression: str, prohibited: bool
) -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": f"""
import numpy as np
import random
from albumentations.core.invocation import SamplingContext
class DualTransform: pass
class Example(DualTransform):
    def {method}(self, img, seed):
        return {expression}
"""
        }
    )
    assert ("AXG027" in ids) == prohibited


def test_instance_rng_check_includes_non_transform_mixins() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class SamplingMixin:
    def sample_parameters(self, params, data, targets, sampling):
        return self.py_random.random()
class Example(SamplingMixin, DualTransform): pass
"""
        }
    )
    assert "AXG027" in ids


@pytest.mark.parametrize("method", ["__init__", "sample_parameters", "helper"])
@pytest.mark.parametrize("factory", ["np.random.default_rng()", "np.random.default_rng(137)", "random.Random(137)"])
def test_rng_factories_cannot_escape_through_constructors_or_sampling_helpers(method: str, factory: str) -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": f"""
import numpy as np
import random
class DualTransform: pass
class Example(DualTransform):
    def {method}(self):
        return {factory}
"""
        }
    )
    assert "AXG027" in ids


def test_rng_factory_check_includes_free_sampling_helpers() -> None:
    assert "AXG027" in rule_ids(
        {
            "albumentations/augmentations/example.py": """
from numpy.random import default_rng as generator
def draw():
    return generator().random()
"""
        }
    )


@pytest.mark.parametrize(
    ("imports", "expression"),
    [
        ("import numpy as arrays", "arrays.random.beta(1, 1)"),
        ("from numpy.random import uniform as draw", "draw()"),
        ("import random as stdlib", "stdlib.getrandbits(64)"),
        ("from random import choices as draw", "draw([1, 2])"),
    ],
)
def test_global_rng_checks_cover_aliases_and_distributions(imports: str, expression: str) -> None:
    assert "AXG005" in rule_ids({"albumentations/core/composition.py": f"{imports}\nvalue = {expression}\n"})


def test_apply_length_ignores_comments_and_docstrings_but_enforces_code_lines() -> None:
    body = "\n".join("        value = value" for _ in range(21))
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": f"""\nclass DualTransform: pass\nclass Example(DualTransform):\n    def apply(self, image):\n        "A long docstring\\n        with comments and prose"\n{body}\n        return value\n""",
        },
    )
    assert "AXG003" in ids


def test_sampled_dispatch_is_owned_by_framework_classes() -> None:
    diagnostics = run_sources(
        {
            "albumentations/core/transforms_interface.py": """
class BasicTransform:
    def _apply_with_params(self, params):
        pass

class DualTransform(BasicTransform):
    def _apply_with_params(self, params):
        pass
""",
            "albumentations/augmentations/example.py": """
from albumentations.core.transforms_interface import DualTransform as SpatialTransform

class _SharedResize(SpatialTransform):
    def _apply_with_params(self, params):
        pass

class Example(_SharedResize):
    def apply_with_params(self, params):
        pass

class Unrelated:
    def _apply_with_params(self, params):
        pass
""",
        },
    )

    assert {diagnostic.symbol for diagnostic in diagnostics if diagnostic.rule == "AXG026"} == {
        "_SharedResize._apply_with_params",
        "Example.apply_with_params",
    }


def test_transform_apply_methods_must_name_sampled_parameters_they_consume() -> None:
    diagnostics = rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class BaseExample(DualTransform):
    def apply(self, image, **params):
        return image + params["offset"]
    def apply_to_mask(self, mask, **params):
        return mask if params.get("keep_mask") else mask * 0
    def apply_to_keypoints(self, keypoints, **params):
        if "offset" in params:
            return keypoints
        for value in params:
            return value
        alias = params
        return alias["offset"]
    def apply_to_images(self, images, **params):
        return self._apply(images, lambda image: image + params["offset"])
    def apply_to_volume(self, volume, **params):
        return self._apply(volume, lambda image=params["offset"]: image)
""",
        },
    )
    assert diagnostics.count("AXG025") == 7
    assert "AXG025" not in rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class Example(DualTransform):
    def apply(self, image, offset, **params):
        return self._apply(image, offset, **params)
    def apply_to_mask(self, mask, keep_mask, **params):
        return mask if keep_mask else mask * 0
    def apply_to_images(self, images, **params):
        return self._apply(images, lambda params: params["offset"])
""",
        },
    )


def test_naming_ranges_removed_hooks_and_serialization_rules() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class Example(DualTransform):
    def __init__(self, blur_range: float, fill_value: int = 0): pass
    def get_params(self): return {}
    def get_transform_init_args_names(self): return ()
""",
        },
    )
    assert {"AXG008", "AXG009", "AXG010", "AXG011", "AXG020"} <= set(ids)


def test_cv2_bbox_and_scalar_rules_keep_the_public_exceptions_narrow() -> None:
    assert "AXG012" in rule_ids(
        {
            "albumentations/augmentations/example.py": """
import cv2
def helper(image):
    return cv2.resize(image, (4, 4))
""",
        },
    )
    assert "AXG012" not in rule_ids(
        {
            "albumentations/augmentations/geometric/_functional_distortion.py": """
from ._functional_shared import cv2
def upscale_distortion_maps(image):
    image = cv2.resize(image, (4, 4))
    return cv2.resize(image, (8, 8))
""",
        },
    )
    assert "AXG019" in rule_ids(
        {
            "albumentations/augmentations/geometric/_functional_distortion.py": """
from ._functional_shared import cv2
def upscale_distortion_maps(image):
    return cv2.resize(image, (4, 4))
""",
        },
    )
    ids = rule_ids(
        {
            "albumentations/core/bbox_utils.py": """
class BboxParams:
    def __init__(self, bbox_type='hbb'): pass
""",
            "albumentations/augmentations/example.py": """
import numpy as np
class DualTransform: pass
class Example(DualTransform):
    def __init__(self, bbox_type='hbb'): pass
    def value(self, value: float):
        return np.sqrt(value)
""",
        },
    )
    assert {"AXG013", "AXG014"} <= set(ids)


def test_guidance_resolves_direct_imports_and_positional_only_sampling() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
from cv2 import resize
from numpy.random import uniform
from random import randint

class DualTransform: pass
class Example(DualTransform):
    def sample_parameters(self, params, data, targets, sampling: SamplingContext, /):
        return {"x": uniform(), "y": randint(0, 1), "image": resize(data["image"], (4, 4))}
""",
        },
    )
    assert {"AXG005", "AXG012"} <= set(ids)
    assert "AXG004" not in ids


def test_scalar_math_tracks_local_dataflow_without_leaking_names_between_functions() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
import numpy as np

class DualTransform: pass
class Example(DualTransform):
    def value(self, angle: float):
        radians = np.deg2rad(angle)
        return np.sin(radians) + np.sqrt(angle * 2)

def array_value(angle):
    return np.sqrt(angle)
""",
        },
    )
    assert ids.count("AXG014") == 3


def test_docstring_rules_require_plural_examples_and_apply_docs_are_forbidden() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
class ImageOnlyTransform: pass
class Example(ImageOnlyTransform):
    \"\"\"Example.
    Example:
        text
    ---
    \"\"\"
    def apply(self, image):
        \"\"\"Do not document dispatch methods here.\"\"\"
        return image
    def public_method(self):
        return None
""",
        },
    )
    assert {"AXG015", "AXG016", "AXG017", "AXG018"} <= set(ids)


@pytest.mark.parametrize("method_name", ["apply_to_user_data", "apply_to_volumes", "apply_to_masks3d"])
@pytest.mark.parametrize(
    ("method_docstring", "expected_ids"),
    [("", []), ('"""Forward metadata."""', ["AXG018"])],
    ids=["without-docstring", "with-docstring"],
)
def test_dispatch_methods_have_the_same_docstring_contract_as_pixel_dispatch(
    method_name: str,
    method_docstring: str,
    expected_ids: list[str],
) -> None:
    assert (
        rule_ids(
            {
                "albumentations/augmentations/example.py": f"""
class Transform3D: pass
class Example(Transform3D):
    \"\"\"Forward metadata.

    Examples:
        >>> Example()
    \"\"\"
    def {method_name}(self, data, **params):
        {method_docstring}
        return dict(data)
""",
            },
        )
        == expected_ids
    )


def test_constructor_schema_rule_accepts_explicit_inherited_forwarding() -> None:
    assert "AXG020" not in rule_ids(
        {
            "albumentations/augmentations/example.py": """
class BasicTransform:
    def __init__(self, p: float = 1.0): pass
class Parent(BasicTransform):
    def __init__(self, p: float = 1.0): super().__init__(p=p)
class Child(Parent):
    def __init__(self, p: float = 0.5): super().__init__(p=p)
""",
        },
    )
    assert "AXG020" in rule_ids(
        {
            "albumentations/augmentations/example.py": """
class BasicTransform: pass
class Child(BasicTransform):
    def __init__(self, transpose: bool = False, p: float = 1.0): super().__init__(p=p)
""",
        },
    )


def test_sampling_plan_rules_reject_flat_returns_and_first_target_shape() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class Example(DualTransform):
    def sample_parameters(self, params, data, targets, sampling: SamplingContext):
        shape = params["shape"]
        alternate_shape = params.get("shape")
        return {"shape": alternate_shape}
""",
        },
    )
    assert {"AXG021", "AXG022"} <= set(ids)


def test_sampling_plan_rules_reject_direct_canonical_target_selection() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class Example(DualTransform):
    def sample_parameters(self, params, data, targets, sampling: SamplingContext):
        image = data["image"]
        images = data.get("images")
        has_volume = "volume" in data
        return SampledParams(params={"image": image, "images": images, "has_volume": has_volume})
""",
        },
    )
    assert ids.count("AXG024") == 1


def test_target_plan_rule_rejects_target_routing_parameter_names() -> None:
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class Example(DualTransform):
    def apply(self, image, volume_noise_map):
        return image
""",
        },
    )
    assert "AXG023" in ids
    assert "AXG023" not in rule_ids(
        {
            "albumentations/augmentations/example.py": """
class DualTransform: pass
class Example(DualTransform):
    def apply(self, image, image_type):
        return image
""",
        },
    )
