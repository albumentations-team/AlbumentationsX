"""Transform-specific test helpers.

This module centralizes transform categorization and provides utilities for
setting up transform-specific test data requirements.
"""

import copy
from typing import Any, ClassVar

import numpy as np

import albumentations as A
from tests.aug_definitions import transforms2metadata_key


class TransformTestHelper:
    """Helper for transform-specific test setup.

    Centralizes all transform categorization to eliminate duplication
    of exception lists across test files.
    """

    # Transforms that require special metadata
    METADATA_TRANSFORMS: ClassVar[set[type]] = {
        A.FDA,
        A.HistogramMatching,
        A.PixelDistributionAdaptation,
        A.Mosaic,
    }

    # Transforms that require a mask in the input data
    MASK_REQUIRED_TRANSFORMS: ClassVar[set[type]] = {
        A.MaskDropout,
        A.ConstrainedCoarseDropout,
    }

    # Transforms that only work with RGB images (not grayscale)
    RGB_ONLY_TRANSFORMS: ClassVar[set[type]] = {
        A.ChannelDropout,
        A.Spatter,
        A.ISONoise,
        A.RandomGravel,
        A.ChromaticAberration,
        A.PlanckianJitter,
        A.PixelDistributionAdaptation,
        A.MaskDropout,
        A.ConstrainedCoarseDropout,
        A.ChannelShuffle,
        A.ToRGB,
        A.RandomSunFlare,
        A.LensFlare,
        A.RandomFog,
        A.RandomSnow,
        A.RandomRain,
        A.HEStain,
    }

    # Transforms that require special setup (overlay, text, etc.)
    SPECIAL_SETUP_TRANSFORMS: ClassVar[set[type]] = {
        A.OverlayElements,
        A.TextImage,
        A.RandomCropNearBBox,
    }

    # Transforms that need bbox context
    BBOX_REQUIRED_TRANSFORMS: ClassVar[set[type]] = {
        A.RandomCropNearBBox,
        A.RandomSizedBBoxSafeCrop,
        A.BBoxSafeRandomCrop,
        A.AtLeastOneBBoxRandomCrop,
    }

    # Transforms that change image dimensions
    DIMENSION_CHANGING_TRANSFORMS: ClassVar[set[type]] = {
        A.RandomCrop,
        A.AtLeastOneBBoxRandomCrop,
        A.RandomResizedCrop,
        A.Resize,
        A.RandomSizedCrop,
        A.RandomSizedBBoxSafeCrop,
        A.BBoxSafeRandomCrop,
        A.Transpose,
        A.RandomCropNearBBox,
        A.CenterCrop,
        A.Crop,
        A.CropAndPad,
        A.LongestMaxSize,
        A.RandomScale,
        A.PadIfNeeded,
        A.SmallestMaxSize,
        A.RandomCropFromBorders,
        A.RandomRotate90,
        A.D4,
        A.SquareSymmetry,
    }

    # Transforms that don't support certain interpolations
    INTERPOLATION_RESTRICTED_TRANSFORMS: ClassVar[set[type]] = {
        A.Affine,
        A.GridElasticDeform,
        A.SafeRotate,
        A.ShiftScaleRotate,
        A.OpticalDistortion,
        A.ThinPlateSpline,
        A.Perspective,
        A.ElasticTransform,
        A.GridDistortion,
        A.PiecewiseAffine,
        A.CropAndPad,
        A.LongestMaxSize,
        A.SmallestMaxSize,
        A.RandomResizedCrop,
        A.RandomScale,
        A.Rotate,
    }

    @staticmethod
    def safe_copy_params(params: dict[str, Any]) -> dict[str, Any]:
        """Deep copy params dict to avoid mutation bugs.

        Args:
            params: Original params dict

        Returns:
            Deep copy of params

        """
        return copy.deepcopy(params)

    @staticmethod
    def is_rgb_only(transform_cls: type) -> bool:
        """Check if transform only works with RGB images.

        Args:
            transform_cls: Transform class

        Returns:
            True if transform requires RGB images

        """
        return transform_cls in TransformTestHelper.RGB_ONLY_TRANSFORMS

    @staticmethod
    def requires_metadata(transform_cls: type) -> bool:
        """Check if transform requires special metadata.

        Args:
            transform_cls: Transform class

        Returns:
            True if transform requires metadata

        """
        return transform_cls in TransformTestHelper.METADATA_TRANSFORMS

    @staticmethod
    def requires_mask(transform_cls: type) -> bool:
        """Check if transform requires mask in input data.

        Args:
            transform_cls: Transform class

        Returns:
            True if transform requires mask

        """
        return transform_cls in TransformTestHelper.MASK_REQUIRED_TRANSFORMS

    @staticmethod
    def requires_special_setup(transform_cls: type) -> bool:
        """Check if transform requires special setup.

        Args:
            transform_cls: Transform class

        Returns:
            True if transform requires special setup

        """
        return transform_cls in TransformTestHelper.SPECIAL_SETUP_TRANSFORMS

    @staticmethod
    def changes_dimensions(transform_cls: type) -> bool:
        """Check if transform changes image dimensions.

        Args:
            transform_cls: Transform class

        Returns:
            True if transform may change dimensions

        """
        return transform_cls in TransformTestHelper.DIMENSION_CHANGING_TRANSFORMS

    @staticmethod
    def prepare_test_data(
        transform_cls: type,
        base_image: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare data dict with all required metadata/masks for a transform.

        This replaces all the if/elif chains scattered across test files.

        Args:
            transform_cls: Transform class to prepare data for
            base_image: Base image to use
            **kwargs: Additional data to include (bboxes, keypoints, etc.)

        Returns:
            Data dict ready for transform application

        """
        data: dict[str, Any] = {"image": base_image}

        # Add any additional kwargs
        data.update(kwargs)

        # Add mask if required
        if TransformTestHelper.requires_mask(transform_cls):
            if "mask" not in data:
                mask = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.uint8)
                mask[:20, :20] = 1
                data["mask"] = mask

        # Add metadata for special transforms
        if transform_cls == A.OverlayElements:
            if "overlay_metadata" not in data:
                data["overlay_metadata"] = []

        elif transform_cls == A.TextImage:
            if "textimage_metadata" not in data:
                data["textimage_metadata"] = {
                    "text": "May the transformations be ever in your favor!",
                    "bbox": (0.1, 0.1, 0.9, 0.2),
                }

        elif transform_cls == A.Mosaic:
            if "mosaic_metadata" not in data:
                # Use mask from data if available
                mask = data.get("mask")
                mosaic_entry = {"image": base_image}
                if mask is not None:
                    mosaic_entry["mask"] = mask
                data["mosaic_metadata"] = [mosaic_entry]

        elif transform_cls in transforms2metadata_key:
            metadata_key = transforms2metadata_key[transform_cls]
            if metadata_key not in data:
                data[metadata_key] = [base_image]

        # Note: We don't auto-add bboxes for bbox-required transforms
        # because test functions handle those with specific formats

        return data

    @staticmethod
    def requires_bbox(transform_cls: type) -> bool:
        """Check if transform requires bboxes.

        Args:
            transform_cls: Transform class

        Returns:
            True if transform requires bboxes

        """
        return transform_cls in TransformTestHelper.BBOX_REQUIRED_TRANSFORMS

    @staticmethod
    def adjust_params_for_grayscale(
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Adjust params for grayscale images.

        Args:
            params: Original params

        Returns:
            Adjusted params (copied if modified)

        """
        if "fill" in params and not np.isscalar(params["fill"]):
            # Copy and adjust fill value for grayscale
            params = TransformTestHelper.safe_copy_params(params)
            params["fill"] = params["fill"][0]

        return params
