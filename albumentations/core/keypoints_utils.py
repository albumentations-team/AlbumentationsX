"""Module for handling keypoint operations during augmentation.

This module provides utilities for working with keypoints in various formats during
the augmentation process. It includes functions for converting between coordinate systems,
filtering keypoints based on visibility, validating keypoint data, and applying
transformations to keypoints. The module supports different keypoint formats including
xy, yx, and those with additional angle or size information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from albumentations.core.label_manager import LabelMetadata
from albumentations.core.type_definitions import NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS

from .utils import DataProcessor, Params

__all__ = [
    "KeypointParams",
    "KeypointsProcessor",
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "filter_keypoints",
]

keypoint_formats = {"xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"}


def angle_to_2pi_range(angles: np.ndarray) -> np.ndarray:
    """Convert angles to the range [0, 2π).

    This function takes an array of angles and ensures they are all within
    the range of 0 to 2π (exclusive) by applying modulo 2π.

    Args:
        angles (np.ndarray): Array of angle values in radians.

    Returns:
        np.ndarray: Array of the same shape as input with angles normalized to [0, 2π).

    """
    return np.mod(angles, 2 * np.pi)


class KeypointParams(Params):
    """Parameters of keypoints

    Args:
        format (str): format of keypoints. Should be 'xy', 'yx', 'xya', 'xys', 'xyas', 'xysa', 'xyz'.

            x - X coordinate,

            y - Y coordinate

            z - Z coordinate (for 3D keypoints)

            s - Keypoint scale

            a - Keypoint orientation in radians or degrees (depending on KeypointParams.angle_in_degrees)

        label_fields (list[str]): list of fields that are joined with keypoints, e.g labels.
            Should be same type as keypoints.
        remove_invisible (bool): to remove invisible points after transform or not
        angle_in_degrees (bool): angle in degrees or radians in 'xya', 'xyas', 'xysa' keypoints
        check_each_transform (bool): if `True`, then keypoints will be checked after each dual transform.
            Default: `True`
        label_mapping (dict[str, dict[str, dict[Any, Any]]] | None): Dictionary mapping transform names
            to label field mappings. Structure: {transform_name: {label_field: {from_label: to_label}}}.
            For example: {'HorizontalFlip': {'keypoint_labels': {'left_eye': 'right_eye', 'right_eye': 'left_eye'}}}
            or {'HorizontalFlip': {'keypoint_labels': {0: 1, 1: 0}}}. Works with any hashable label type.
            Can map multiple label fields per transform. Default: None.

    Note:
        The internal Albumentations format is [x, y, z, angle, scale]. For 2D formats (xy, yx, xya, xys, xyas, xysa),
        z coordinate is set to 0. For formats without angle or scale, these values are set to 0.

    """

    def __init__(
        self,
        format: str,  # noqa: A002
        label_fields: Sequence[str] | None = None,
        remove_invisible: bool = True,
        angle_in_degrees: bool = True,
        check_each_transform: bool = True,
        label_mapping: dict[str, dict[str, dict[Any, Any]]] | None = None,
    ):
        super().__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees
        self.check_each_transform = check_each_transform

        # Warn about potential misconfiguration
        if label_fields and label_mapping is None:
            import warnings

            msg = (
                "label_fields are set but label_mapping is not provided. "
                "If you don't need label swapping, remove label_fields. "
                "If you need label swapping, provide label_mapping."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        self.label_mapping = label_mapping if label_mapping is not None else {}

    def to_dict_private(self) -> dict[str, Any]:
        """Get the private dictionary representation of keypoint parameters.

        Returns:
            dict[str, Any]: Dictionary containing the keypoint parameters.

        """
        data = super().to_dict_private()
        data.update(
            {
                "remove_invisible": self.remove_invisible,
                "angle_in_degrees": self.angle_in_degrees,
                "check_each_transform": self.check_each_transform,
                "label_mapping": self.label_mapping,
            },
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the class is serializable.

        Returns:
            bool: Always returns True as KeypointParams is serializable.

        """
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        """Get the full class name for serialization.

        Returns:
            str: The string "KeypointParams" representing the class name.

        """
        return "KeypointParams"

    def __repr__(self) -> str:
        return (
            f"KeypointParams(format={self.format}, label_fields={self.label_fields},"
            f" remove_invisible={self.remove_invisible}, angle_in_degrees={self.angle_in_degrees},"
            f" check_each_transform={self.check_each_transform}, label_mapping={self.label_mapping})"
        )


class KeypointsProcessor(DataProcessor):
    """Processor for keypoint data transformation.

    This class handles the conversion, validation, and filtering of keypoints
    during transformations. It ensures keypoints are correctly formatted and
    processed according to the specified keypoint parameters.

    Args:
        params (KeypointParams): Parameters for keypoint processing.
        additional_targets (dict[str, str] | None): Dictionary mapping additional target names to their types.

    """

    def __init__(self, params: KeypointParams, additional_targets: dict[str, str] | None = None):
        super().__init__(params, additional_targets)
        # Store encoded mappings for transforms - will be populated during preprocessing
        self.encoded_label_mappings: dict[str, dict[str, dict[int, int]]] = {}

    @property
    def default_data_name(self) -> str:
        return "keypoints"

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        """Ensure the provided data dictionary contains all required label fields.

        Args:
            data (dict[str, Any]): The data dictionary to validate.

        Raises:
            ValueError: If any label field specified in params is missing from the data.

        """
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in 'keypoint_params' dict"
            raise ValueError(msg)

    def filter(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Filter keypoints based on visibility within given shape.

        Args:
            data (np.ndarray): Keypoints in [x, y, z, angle, scale] format
            shape (tuple[int, int] | tuple[int, int, int]): Shape to check against as (height, width) or
                (depth, height, width)

        Returns:
            np.ndarray: Filtered keypoints

        """
        self.params: KeypointParams
        return filter_keypoints(data, shape, remove_invisible=self.params.remove_invisible)

    def check(self, data: np.ndarray, shape: tuple[int, int] | tuple[int, int, int]) -> None:
        """Check if keypoints are valid within the given shape.

        Args:
            data (np.ndarray): Keypoints to validate.
            shape (tuple[int, int] | tuple[int, int, int]): Shape to check against.

        """
        check_keypoints(data, shape)

    def convert_label_mappings_to_encoded(self) -> None:
        """Convert string-based label mappings to encoded integer mappings.

        This should be called after labels are encoded during preprocessing.
        """
        if not self.params.label_mapping or not self.params.label_fields:
            return

        self.encoded_label_mappings = {}

        # First, update encoders with all labels from mappings
        self._update_encoders_with_mapping_labels()

        # Then convert mappings to encoded integers
        self._convert_mappings_to_encoded()

    def _update_encoders_with_mapping_labels(self) -> None:
        """Update encoders with all labels from mappings."""
        for field_mappings in self.params.label_mapping.values():
            for label_field, mapping in field_mappings.items():
                metadata = self.label_manager.metadata.get("keypoints", {}).get(label_field)
                if metadata and metadata.encoder is not None:
                    # Collect all labels (both from and to) from the mapping
                    all_mapping_labels = set(mapping.keys()) | set(mapping.values())
                    # Update encoder with all labels that might be needed
                    metadata.encoder.update(list(all_mapping_labels))

    def _convert_mappings_to_encoded(self) -> None:
        """Convert mappings to encoded integers."""
        for transform_name, field_mappings in self.params.label_mapping.items():
            encoded_mappings = {}

            for label_field, mapping in field_mappings.items():
                if metadata := self.label_manager.metadata.get("keypoints", {}).get(label_field):
                    encoded_mapping = self._convert_single_mapping(mapping, metadata)
                    encoded_mappings[label_field] = encoded_mapping

            self.encoded_label_mappings[transform_name] = encoded_mappings

    def _convert_single_mapping(self, mapping: dict[Any, Any], metadata: LabelMetadata) -> dict[int, int]:
        """Convert a single mapping to encoded integers."""
        encoded_mapping = {}

        if metadata.encoder is not None:
            # Convert string mapping to encoded integers
            # Pre-filter valid labels to avoid repeated lookups
            encoder_classes = set(metadata.encoder.classes_)
            valid_from_labels = set(mapping.keys()) & encoder_classes
            valid_to_labels = set(mapping.values()) & encoder_classes

            # Filter to only valid mappings where both from and to exist
            valid_mappings = {k: v for k, v in mapping.items() if k in valid_from_labels and v in valid_to_labels}

            # Convert valid mappings in batch
            if valid_mappings:
                from_labels = list(valid_mappings.keys())
                to_labels = list(valid_mappings.values())
                from_encoded = metadata.encoder.transform(from_labels)
                to_encoded = metadata.encoder.transform(to_labels)

                encoded_mapping.update(dict(zip(from_encoded, to_encoded)))

            # Track missing labels for warning
            missing_labels = []
            for from_label, to_label in mapping.items():
                if from_label not in encoder_classes:
                    missing_labels.append(from_label)
                if to_label not in encoder_classes:
                    missing_labels.append(to_label)

            # Warn about missing labels
            if missing_labels:
                import warnings

                unique_missing = list(set(missing_labels))
                warnings.warn(
                    f"Labels {unique_missing} in label_mapping are not found in the dataset. "
                    "These mappings will be ignored. Check your label_mapping configuration.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            # Numerical labels, use mapping as-is
            encoded_mapping |= mapping

        return encoded_mapping

    def add_label_fields_to_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add label fields to data arrays and convert label mappings to encoded form."""
        result = super().add_label_fields_to_data(data)
        # After labels are encoded, convert the mappings to work with encoded integers
        self.convert_label_mappings_to_encoded()
        return result

    def convert_from_albumentations(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Convert keypoints from internal Albumentations format to the specified format.

        Args:
            data (np.ndarray): Keypoints in Albumentations format.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information for validation.

        Returns:
            np.ndarray: Converted keypoints in the target format.

        """
        if not data.size:
            return data

        params = self.params
        return convert_keypoints_from_albumentations(
            data,
            params.format,
            shape,  # Pass full shape for proper 3D keypoint validation
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )

    def convert_to_albumentations(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Convert keypoints from the specified format to internal Albumentations format.

        Args:
            data (np.ndarray): Keypoints in source format.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information for validation.

        Returns:
            np.ndarray: Converted keypoints in Albumentations format.

        """
        if not data.size:
            return data
        params = self.params
        return convert_keypoints_to_albumentations(
            data,
            params.format,
            shape[:2],  # Only use height, width for conversion
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )


def check_keypoints(keypoints: np.ndarray, shape: tuple[int, int] | tuple[int, int, int]) -> None:
    """Check if keypoint coordinates are within valid ranges for the given shape.

    This function validates that:
    1. All x-coordinates are within [0, width)
    2. All y-coordinates are within [0, height)
    3. For 3D keypoints: All z-coordinates are within [0, depth)
    4. Angles are within the range [0, 2π)

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 3+) for 3D or (N, 2+) for 2D.
            - First 2 columns are always x, y
            - Column 3 (if present) is z for 3D or angle for 2D
            - Column 4 (if present) is angle for 3D or scale for 2D
            - Column 5+ (if present) are additional attributes
        shape (tuple[int, int] | tuple[int, int, int]): The shape of the image/volume
            - (height, width) for 2D
            - (depth, height, width) for 3D

    Raises:
        ValueError: If any keypoint coordinate is outside the valid range, or if angles are invalid.
                   The error message will detail which keypoints are invalid and why.

    Note:
        - The function assumes that keypoint coordinates are in absolute pixel values, not normalized
        - Angles are in radians

    """
    # Handle 3D case
    if len(shape) == 3:
        depth, height, width = shape
    else:
        height, width = shape
        depth = None

    # Check x and y coordinates (always present)
    x, y = keypoints[:, 0], keypoints[:, 1]
    invalid_x = np.where((x < 0) | (x >= width))[0]
    invalid_y = np.where((y < 0) | (y >= height))[0]

    error_messages = []

    # Handle x, y errors
    for idx in sorted(set(invalid_x) | set(invalid_y)):
        if idx in invalid_x:
            error_messages.append(
                f"Expected x for keypoint {keypoints[idx]} to be in range [0, {width}), got {x[idx]}",
            )
        if idx in invalid_y:
            error_messages.append(
                f"Expected y for keypoint {keypoints[idx]} to be in range [0, {height}), got {y[idx]}",
            )

    # For 3D keypoints, check z coordinates
    if depth is not None and keypoints.shape[1] > 2:
        z = keypoints[:, 2]
        invalid_z = np.where((z < 0) | (z >= depth))[0]
        error_messages.extend(
            f"Expected z for keypoint {keypoints[idx]} to be in range [0, {depth}), got {z[idx]}" for idx in invalid_z
        )

    # Check angles - for 2D it's column 3, for 3D it's column 4
    angle_col = 3 if depth is None else 4
    if keypoints.shape[1] > angle_col:
        angles = keypoints[:, angle_col]
        invalid_angles = np.where((angles < 0) | (angles >= 2 * math.pi))[0]
        error_messages.extend(
            f"Expected angle for keypoint {keypoints[idx]} to be in range [0, 2π), got {angles[idx]}"
            for idx in invalid_angles
        )

    if error_messages:
        raise ValueError("\n".join(error_messages))


def filter_keypoints(
    keypoints: np.ndarray,
    shape: tuple[int, int] | tuple[int, int, int],
    remove_invisible: bool,
) -> np.ndarray:
    """Filter keypoints to remove those outside the boundaries.

    Args:
        keypoints (np.ndarray): A numpy array of shape (N, 3+) where N is the number of keypoints.
                               Each row represents a keypoint (x, y, z, ...) for 3D or (x, y, ...) for 2D.
        shape (tuple[int, int] | tuple[int, int, int]): Shape to check against as (height, width) for 2D
                                                        or (depth, height, width) for 3D.
        remove_invisible (bool): If True, remove keypoints outside the boundaries.

    Returns:
        np.ndarray: Filtered keypoints.

    """
    if not remove_invisible:
        return keypoints

    if not keypoints.size:
        return keypoints

    # Handle 3D case (depth, height, width)
    if len(shape) == 3:
        depth, height, width = shape

        # Create boolean mask for visible keypoints
        x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
        visible = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z >= 0) & (z < depth)
    else:
        # Handle 2D case (height, width)
        height, width = shape

        # Create boolean mask for visible keypoints
        x, y = keypoints[:, 0], keypoints[:, 1]
        visible = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    # Apply the mask to filter keypoints
    return keypoints[visible]


def convert_keypoints_to_albumentations(
    keypoints: np.ndarray,
    source_format: Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"],
    shape: tuple[int, int] | tuple[int, int, int],
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> np.ndarray:
    """Convert keypoints from various formats to the Albumentations format.

    This function takes keypoints in different formats and converts them to the standard
    Albumentations format: [x, y, z, angle, scale]. For 2D formats, z is set to 0.
    For formats without angle or scale, these values are set to 0.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 2+), where N is the number of keypoints.
                                The number of columns depends on the source_format.
        source_format (Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"]): The format of the input keypoints.
            - "xy": [x, y]
            - "yx": [y, x]
            - "xya": [x, y, angle]
            - "xys": [x, y, scale]
            - "xyas": [x, y, angle, scale]
            - "xysa": [x, y, scale, angle]
            - "xyz": [x, y, z]
        shape (tuple[int, int] | tuple[int, int, int]): The shape of the image (height, width) or
            volume (depth, height, width).
        check_validity (bool, optional): If True, check if the converted keypoints are within the
            image/volume boundaries. Defaults to False.
        angle_in_degrees (bool, optional): If True, convert input angles from degrees to radians.
                                           Defaults to True.

    Returns:
        np.ndarray: Array of keypoints in Albumentations format [x, y, z, angle, scale] with shape (N, 5+).
                    Any additional columns from the input keypoints are preserved and appended after the
                    first 5 columns.

    Raises:
        ValueError: If the source_format is not one of the supported formats.

    Note:
        - For 2D formats (xy, yx, xya, xys, xyas, xysa), z coordinate is set to 0
        - Angles are converted to the range [0, 2π) radians
        - If the input keypoints have additional columns beyond what's specified in the source_format,
          these columns are preserved in the output

    """
    if source_format not in keypoint_formats:
        raise ValueError(f"Unknown source_format {source_format}. Supported formats are: {keypoint_formats}")

    format_to_indices: dict[str, list[int | None]] = {
        "xy": [0, 1, None, None, None],
        "yx": [1, 0, None, None, None],
        "xya": [0, 1, None, 2, None],
        "xys": [0, 1, None, None, 2],
        "xyas": [0, 1, None, 2, 3],
        "xysa": [0, 1, None, 3, 2],
        "xyz": [0, 1, 2, None, None],
    }

    indices: list[int | None] = format_to_indices[source_format]

    processed_keypoints = np.zeros((keypoints.shape[0], NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS), dtype=np.float32)

    for i, idx in enumerate(indices):
        if idx is not None:
            processed_keypoints[:, i] = keypoints[:, idx]

    if angle_in_degrees and indices[3] is not None:  # angle is now at index 3
        processed_keypoints[:, 3] = np.radians(processed_keypoints[:, 3])

    processed_keypoints[:, 3] = angle_to_2pi_range(processed_keypoints[:, 3])  # angle is now at index 3

    if keypoints.shape[1] > len(source_format):
        processed_keypoints = np.column_stack((processed_keypoints, keypoints[:, len(source_format) :]))

    if check_validity:
        check_keypoints(processed_keypoints, shape)

    return processed_keypoints


def convert_keypoints_from_albumentations(
    keypoints: np.ndarray,
    target_format: Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"],
    shape: tuple[int, int] | tuple[int, int, int],
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> np.ndarray:
    """Convert keypoints from Albumentations format to various other formats.

    This function takes keypoints in the standard Albumentations format [x, y, z, angle, scale]
    and converts them to the specified target format.

    Args:
        keypoints (np.ndarray): Array of keypoints in Albumentations format with shape (N, 5+),
                                where N is the number of keypoints. Each row represents a keypoint
                                [x, y, z, angle, scale, ...].
        target_format (Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"]): The desired output format.
            - "xy": [x, y]
            - "yx": [y, x]
            - "xya": [x, y, angle]
            - "xys": [x, y, scale]
            - "xyas": [x, y, angle, scale]
            - "xysa": [x, y, scale, angle]
            - "xyz": [x, y, z]
        shape (tuple[int, int] | tuple[int, int, int]): The shape of the image (height, width) or
            volume (depth, height, width).
        check_validity (bool, optional): If True, check if the keypoints are within the
            image/volume boundaries. Defaults to False.
        angle_in_degrees (bool, optional): If True, convert output angles to degrees.
                                           If False, angles remain in radians.
                                           Defaults to True.

    Returns:
        np.ndarray: Array of keypoints in the specified target format with shape (N, 2+).
                    Any additional columns from the input keypoints beyond the first 5
                    are preserved and appended after the converted columns.

    Raises:
        ValueError: If the target_format is not one of the supported formats.

    Note:
        - Input angles are assumed to be in the range [0, 2π) radians
        - If the input keypoints have additional columns beyond the first 5,
          these columns are preserved in the output

    """
    if target_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {target_format}. Supported formats are: {keypoint_formats}")

    x, y, z, angle, scale = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], keypoints[:, 3], keypoints[:, 4]
    angle = angle_to_2pi_range(angle)

    if check_validity:
        check_keypoints(np.column_stack((x, y, z, angle, scale)), shape)

    if angle_in_degrees:
        angle = np.degrees(angle)

    format_to_columns = {
        "xy": [x, y],
        "yx": [y, x],
        "xya": [x, y, angle],
        "xys": [x, y, scale],
        "xyas": [x, y, angle, scale],
        "xysa": [x, y, scale, angle],
        "xyz": [x, y, z],
    }

    result = np.column_stack(format_to_columns[target_format])

    # Add any additional columns from the original keypoints
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        return np.column_stack((result, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]))

    return result
