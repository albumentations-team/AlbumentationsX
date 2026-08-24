"""Contract tests for the unified AX coding-guidance hook."""

from __future__ import annotations

from tools.ax_coding_guidance import run_sources


def rule_ids(sources: dict[str, str]) -> list[str]:
    return [diagnostic.rule for diagnostic in run_sources(sources)]


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


def test_apply_length_ignores_comments_and_docstrings_but_enforces_code_lines() -> None:
    body = "\n".join("        value = value" for _ in range(21))
    ids = rule_ids(
        {
            "albumentations/augmentations/example.py": f"""\nclass DualTransform: pass\nclass Example(DualTransform):\n    def apply(self, image):\n        "A long docstring\\n        with comments and prose"\n{body}\n        return value\n""",
        },
    )
    assert "AXG003" in ids


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
        return {"shape": shape}
""",
        },
    )
    assert {"AXG021", "AXG022"} <= set(ids)


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
