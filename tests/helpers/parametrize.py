"""Parametrization helpers for test decorators.

This module provides decorators and utilities to simplify pytest parametrization
with automatic param safety and category-based filtering.
"""

from typing import Any

from tests.helpers.transforms import TransformTestHelper
from tests.utils import (
    get_2d_transforms,
    get_dual_transforms,
    get_image_only_transforms,
    get_transforms,
)


class SafeParamsWrapper:
    """Wrapper that makes params dict copy-on-write.

    This prevents params mutation bugs by automatically copying
    the dict on first modification.
    """

    def __init__(self, params: dict[str, Any]):
        """Initialize wrapper.

        Args:
            params: Original params dict

        """
        self._params = params
        self._copied = False

    def get(self) -> dict[str, Any]:
        """Get params dict (will be copied if needed).

        Returns:
            Params dict

        """
        if not self._copied:
            self._params = TransformTestHelper.safe_copy_params(self._params)
            self._copied = True
        return self._params


def get_transforms_with_categories(
    transform_type: str = "2d",
    exclude_categories: list[str] | None = None,
    custom_arguments: dict | None = None,
) -> list[tuple[type, dict]]:
    """Get transforms filtered by category.

    Args:
        transform_type: Type of transforms ('2d', 'dual', 'image_only', 'all')
        exclude_categories: Categories to exclude:
            - 'metadata': Transforms requiring special metadata
            - 'mask_required': Transforms requiring mask input
            - 'rgb_only': Transforms that only work with RGB
            - 'special_setup': Transforms needing special setup
            - 'bbox_required': Transforms requiring bboxes
        custom_arguments: Custom arguments for specific transforms

    Returns:
        List of (transform_class, params) tuples

    """
    # Get base transforms
    if transform_type == "2d":
        transforms = get_2d_transforms(custom_arguments=custom_arguments)
    elif transform_type == "dual":
        transforms = get_dual_transforms(custom_arguments=custom_arguments)
    elif transform_type == "image_only":
        transforms = get_image_only_transforms(custom_arguments=custom_arguments)
    elif transform_type == "all":
        transforms = get_transforms(custom_arguments=custom_arguments)
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    # Filter by categories if specified
    if exclude_categories:
        filtered = []
        for transform_cls, params in transforms:
            should_exclude = False

            for category in exclude_categories:
                if (category == "metadata" and TransformTestHelper.requires_metadata(transform_cls)) or (
                    category == "mask_required" and TransformTestHelper.requires_mask(transform_cls)
                ):
                    should_exclude = True
                    break
                if (
                    (category == "rgb_only" and TransformTestHelper.is_rgb_only(transform_cls))
                    or (category == "special_setup" and TransformTestHelper.requires_special_setup(transform_cls))
                    or (category == "bbox_required" and TransformTestHelper.requires_bbox(transform_cls))
                ):
                    should_exclude = True
                    break

            if not should_exclude:
                filtered.append((transform_cls, params))

        return filtered

    return transforms


def wrap_params_safely(
    transform_params: list[tuple[type, dict]],
) -> list[tuple[type, SafeParamsWrapper]]:
    """Wrap params in SafeParamsWrapper for automatic copy-on-write.

    Args:
        transform_params: List of (transform_class, params) tuples

    Returns:
        List with params wrapped in SafeParamsWrapper

    """
    return [(transform_cls, SafeParamsWrapper(params)) for transform_cls, params in transform_params]


def build_exclude_set(*categories: str) -> set[type]:
    """Build a set of transform classes to exclude based on categories.

    Args:
        *categories: Category names to exclude

    Returns:
        Set of transform classes to exclude

    """
    exclude_set = set()

    for category in categories:
        if category == "metadata":
            exclude_set.update(TransformTestHelper.METADATA_TRANSFORMS)
        elif category == "mask_required":
            exclude_set.update(TransformTestHelper.MASK_REQUIRED_TRANSFORMS)
        elif category == "rgb_only":
            exclude_set.update(TransformTestHelper.RGB_ONLY_TRANSFORMS)
        elif category == "special_setup":
            exclude_set.update(TransformTestHelper.SPECIAL_SETUP_TRANSFORMS)
        elif category == "bbox_required":
            exclude_set.update(TransformTestHelper.BBOX_REQUIRED_TRANSFORMS)
        else:
            raise ValueError(f"Unknown category: {category}")

    return exclude_set
