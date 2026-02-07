"""Compose builder for test pipelines.

This module provides fluent builders for creating test Compose instances
with common configurations.
"""

from typing import Any

import albumentations as A


class ComposeBuilder:
    """Fluent builder for test Compose instances.

    Provides a clean API for building test pipelines with common configurations
    like bbox and keypoint parameters.

    Example:
        >>> builder = ComposeBuilder([A.HorizontalFlip(p=1.0)])
        >>> builder.with_bboxes('pascal_voc').with_keypoints('xy')
        >>> compose = builder.build()

    """

    def __init__(self, transforms: list | None = None):
        """Initialize builder.

        Args:
            transforms: List of transforms to include

        """
        self.transforms = transforms or []
        self.seed = 137
        self.strict = True
        self.bbox_params = None
        self.keypoint_params = None
        self.additional_targets = None
        self.p = 1.0

    def with_seed(self, seed: int) -> "ComposeBuilder":
        """Set random seed.

        Args:
            seed: Random seed

        Returns:
            Self for chaining

        """
        self.seed = seed
        return self

    def with_strict(self, strict: bool) -> "ComposeBuilder":
        """Set strict mode.

        Args:
            strict: Whether to enable strict validation

        Returns:
            Self for chaining

        """
        self.strict = strict
        return self

    def with_p(self, p: float) -> "ComposeBuilder":
        """Set probability.

        Args:
            p: Probability of applying the pipeline

        Returns:
            Self for chaining

        """
        self.p = p
        return self

    def with_bboxes(
        self,
        coord_format: str = "pascal_voc",
        label_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> "ComposeBuilder":
        """Add bbox parameters.

        Args:
            coord_format: Bbox coordinate format
            label_fields: Label field names
            **kwargs: Additional bbox params

        Returns:
            Self for chaining

        """
        if label_fields is None:
            label_fields = ["bbox_labels"]

        self.bbox_params = A.BboxParams(
            coord_format=coord_format,
            label_fields=label_fields,
            **kwargs,
        )
        return self

    def with_keypoints(
        self,
        coord_format: str = "xy",
        label_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> "ComposeBuilder":
        """Add keypoint parameters.

        Args:
            coord_format: Keypoint coordinate format
            label_fields: Label field names
            **kwargs: Additional keypoint params

        Returns:
            Self for chaining

        """
        if label_fields is None:
            label_fields = ["keypoint_labels"]

        self.keypoint_params = A.KeypointParams(
            coord_format=coord_format,
            label_fields=label_fields,
            **kwargs,
        )
        return self

    def with_additional_targets(
        self,
        additional_targets: dict[str, str],
    ) -> "ComposeBuilder":
        """Add additional target mappings.

        Args:
            additional_targets: Dict mapping target names to types

        Returns:
            Self for chaining

        """
        self.additional_targets = additional_targets
        return self

    def build(self) -> A.Compose:
        """Build the Compose instance.

        Returns:
            Configured Compose instance

        """
        kwargs = {
            "seed": self.seed,
            "strict": self.strict,
            "p": self.p,
        }

        if self.bbox_params is not None:
            kwargs["bbox_params"] = self.bbox_params

        if self.keypoint_params is not None:
            kwargs["keypoint_params"] = self.keypoint_params

        if self.additional_targets is not None:
            kwargs["additional_targets"] = self.additional_targets

        return A.Compose(self.transforms, **kwargs)


def create_compose(
    transforms: list,
    seed: int = 137,
    strict: bool = True,
    **kwargs: Any,
) -> A.Compose:
    """Quick factory function for creating Compose instances.

    This is a convenience wrapper around ComposeBuilder for simple cases.

    Args:
        transforms: List of transforms
        seed: Random seed
        strict: Enable strict validation
        **kwargs: Additional Compose parameters

    Returns:
        Compose instance

    """
    defaults = {"seed": seed, "strict": strict}
    defaults.update(kwargs)
    return A.Compose(transforms, **defaults)
