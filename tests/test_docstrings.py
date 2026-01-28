"""Tests to validate that transform docstrings match their actual properties."""

import pytest
from google_docstring_parser import parse_google_docstring

from albumentations.core.type_definitions import Targets
from tests.utils import get_all_valid_transforms


def parse_targets_from_docstring(docstring_targets: str) -> set[str]:
    """Parse targets from docstring format to set of lowercase strings."""
    if not docstring_targets:
        return set()

    # Split by comma and clean up
    targets = [t.strip().lower() for t in docstring_targets.split(",")]
    return set(targets)


def get_class_targets(cls) -> set[str]:
    """Get targets from class _targets property as lowercase strings."""
    if not hasattr(cls, "_targets"):
        return set()

    targets = cls._targets
    if isinstance(targets, tuple):
        # Convert Targets enum to lowercase string names
        return {t.name.lower() for t in targets}
    elif isinstance(targets, Targets):
        return {targets.name.lower()}

    return set()


def parse_bbox_types_from_docstring(docstring_bbox_types: str | None) -> set[str]:
    """Parse bbox types from docstring format to set of strings."""
    if not docstring_bbox_types:
        return set()

    # Split by comma and clean up
    bbox_types = [t.strip().lower() for t in docstring_bbox_types.split(",")]
    return set(bbox_types)


def get_class_bbox_types(cls) -> set[str]:
    """Get bbox types from class _supported_bbox_types property."""
    if not hasattr(cls, "_supported_bbox_types"):
        return set()

    bbox_types = cls._supported_bbox_types
    if isinstance(bbox_types, frozenset):
        return set(bbox_types)
    elif isinstance(bbox_types, set):
        return bbox_types

    return set()


@pytest.mark.parametrize("transform_cls", get_all_valid_transforms())
def test_docstring_targets_match_class_property(transform_cls):
    """Test that 'Targets:' in docstring matches _targets class property."""
    transform_name = transform_cls.__name__
    docstring = transform_cls.__doc__

    if not docstring:
        pytest.skip(f"{transform_name} has no docstring")

    parsed = parse_google_docstring(docstring)
    docstring_targets_str = parsed.get("Targets")

    # Parse targets from docstring
    docstring_targets = parse_targets_from_docstring(docstring_targets_str)

    # Get targets from class property
    class_targets = get_class_targets(transform_cls)

    if not docstring_targets:
        pytest.skip(f"{transform_name} has no 'Targets:' section in docstring")

    if not class_targets:
        pytest.skip(f"{transform_name} has no _targets property")

    # Check they match
    assert docstring_targets == class_targets, (
        f"{transform_name}: Docstring targets {docstring_targets} "
        f"don't match class _targets {class_targets}"
    )


@pytest.mark.parametrize("transform_cls", get_all_valid_transforms())
def test_docstring_bbox_types_match_class_property(transform_cls):
    """Test that 'Supported bboxes:' in docstring matches _supported_bbox_types class property."""
    transform_name = transform_cls.__name__
    docstring = transform_cls.__doc__

    if not docstring:
        pytest.skip(f"{transform_name} has no docstring")

    parsed = parse_google_docstring(docstring)
    docstring_bbox_types_str = parsed.get("Supported bboxes")

    # Parse bbox types from docstring
    docstring_bbox_types = parse_bbox_types_from_docstring(docstring_bbox_types_str)

    # Get bbox types from class property
    class_bbox_types = get_class_bbox_types(transform_cls)

    # Only check if the transform supports bboxes
    class_targets = get_class_targets(transform_cls)
    supports_bboxes = "bboxes" in class_targets

    if supports_bboxes:
        # If transform supports bboxes, it should have bbox types defined
        if class_bbox_types:
            # If class has _supported_bbox_types, docstring should document it
            assert docstring_bbox_types, (
                f"{transform_name} supports bboxes and has _supported_bbox_types={class_bbox_types}, "
                f"but docstring has no 'Supported bboxes:' section"
            )

            # Check they match
            assert docstring_bbox_types == class_bbox_types, (
                f"{transform_name}: Docstring bbox types {docstring_bbox_types} "
                f"don't match class _supported_bbox_types {class_bbox_types}"
            )
    else:
        # If transform doesn't support bboxes, it shouldn't have bbox types in docstring
        assert not docstring_bbox_types, (
            f"{transform_name} doesn't support bboxes but has 'Supported bboxes: {docstring_bbox_types_str}' in docstring"
        )
