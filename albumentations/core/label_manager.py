"""Module for managing and transforming label data during augmentation.

This module provides utilities for encoding, decoding, and tracking metadata for labels
during the augmentation process. It includes classes for managing label transformations,
preserving data types, and ensuring consistent handling of categorical, numerical, and
mixed label types. The module supports automatic encoding of string labels to numerical
values and restoration of original data types after processing.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, TypedDict

import numpy as np


def custom_sort(item: Any) -> tuple[int, Real | str]:
    """Return a sort key (priority, value) so labels sort consistently: numbers first (priority 0),
    then strings (priority 1). Used for ordering.

    This function is used to sort labels in a consistent order, prioritizing numerical
    values before string values. All numerical values are given priority 0, while
    string values are given priority 1, ensuring numerical values are sorted first.

    Args:
        item (Any): Item to be sorted, can be either a numeric value or any other type.

    Returns:
        tuple[int, Real | str]: A tuple with sort priority (0 for numbers, 1 for others)
            and the value itself (or string representation for non-numeric values).

    """
    return (0, item) if isinstance(item, Real) else (1, str(item))


def _categorize_labels(labels: set[Any]) -> tuple[list[Real], list[str]]:
    numeric_labels: list[Real] = []
    string_labels: list[str] = []

    for label in labels:
        (numeric_labels if isinstance(label, Real) else string_labels).append(label)
    return numeric_labels, string_labels


class LabelEncoder:
    """Encode original label values (numeric or categorical) to integer indices and back. Keeps
    classes_ and inverse_classes_ mapping.

    This class handles the conversion between original label values and their
    numerical representations. It supports both numerical and categorical labels.

    Args:
        classes_ (dict[str | Real, int]): Mapping from original labels to encoded indices.
        inverse_classes_ (dict[int, str | Real]): Mapping from encoded indices to original labels.
        num_classes (int): Number of unique classes.
        is_numerical (bool): Whether the original labels are numerical.

    """

    def __init__(self) -> None:
        self.classes_: dict[str | Real, int] = {}
        self.inverse_classes_: dict[int, str | Real] = {}
        self.num_classes: int = 0
        self.is_numerical: bool = True

    def fit(self, y: Sequence[Any] | np.ndarray) -> "LabelEncoder":
        """Fit the encoder to the input labels (sequence or array). Builds the label-to-index
        mapping and returns self for chaining.

        Args:
            y (Sequence[Any] | np.ndarray): Input labels to fit the encoder.

        Returns:
            LabelEncoder: The fitted encoder instance.

        """
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        # If input is empty, default to non-numerical to allow potential updates later
        if not y:
            self.is_numerical = False
            return self

        self.is_numerical = all(isinstance(label, Real) for label in y)

        if self.is_numerical:
            return self

        unique_labels = sorted(set(y), key=custom_sort)
        for label in unique_labels:
            if label not in self.classes_:
                self.classes_[label] = self.num_classes
                self.inverse_classes_[self.num_classes] = label
                self.num_classes += 1
        return self

    def transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        """Transform labels to encoded integer indices using the fitted encoder. Input: sequence
        or array; returns 1D ndarray of indices.

        Args:
            y (Sequence[Any] | np.ndarray): Input labels to transform.

        Returns:
            np.ndarray: Encoded integer indices.

        """
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

        return np.array([self.classes_[label] for label in y])

    def fit_transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        """Fit the encoder to the input labels and return their encoded integer indices in one
        step. Convenience method for fit+transform.

        Args:
            y (Sequence[Any] | np.ndarray): Input labels to fit and transform.

        Returns:
            np.ndarray: Encoded integer indices.

        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        """Transform encoded integer indices back to original label values. Input: sequence or
        array of indices; returns ndarray of labels.

        Args:
            y (Sequence[Any] | np.ndarray): Encoded integer indices.

        Returns:
            np.ndarray: Original labels.

        """
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

        return np.array([self.inverse_classes_[label] for label in y], dtype=object)

    def update(self, y: Sequence[Any] | np.ndarray) -> "LabelEncoder":
        """Update the encoder with new labels not seen during fit. Adds new mappings only; does
        not change existing encodings. Idempotent for existing labels.

        This method identifies labels in the input sequence that are not already
        known to the encoder and adds them to the internal mapping. It does not
        change the encoding of previously seen labels.

        Args:
            y (Sequence[Any] | np.ndarray): A sequence or array of potentially new labels.

        Returns:
            LabelEncoder: The updated encoder instance.

        """
        if self.is_numerical:
            # Do not update if the original data was purely numerical
            return self

        # Standardize input type to list for easier processing
        if isinstance(y, np.ndarray):
            input_labels = y.flatten().tolist()
        elif isinstance(y, Sequence) and not isinstance(y, str):
            input_labels = list(y)
        elif y is None:
            # Handle cases where a label field might be None or empty
            return self
        else:
            # Handle single item case or string (treat string as single label)
            input_labels = [y]

        # Find labels not already in the encoder efficiently using sets
        current_labels_set = set(self.classes_.keys())
        new_unique_labels = set(input_labels) - current_labels_set

        if not new_unique_labels:
            # No new labels to add
            return self

        # Separate and sort new labels for deterministic encoding order
        numeric_labels, string_labels = _categorize_labels(new_unique_labels)
        sorted_new_labels = sorted(numeric_labels) + sorted(string_labels, key=str)

        for label in sorted_new_labels:
            new_id = self.num_classes
            self.classes_[label] = new_id
            self.inverse_classes_[new_id] = label
            self.num_classes += 1

        return self


@dataclass
class LabelMetadata:
    """Stores metadata for a label field: input_type, is_numerical, dtype, and optional
    LabelEncoder for encoding and decoding.
    """

    input_type: type
    is_numerical: bool
    dtype: np.dtype | None = None
    encoder: LabelEncoder | None = None


class LabelReplayMetadata(TypedDict):
    """Recorded interpretation of one encoded annotation field."""

    is_numerical: bool
    is_array: bool
    dtype: str | None
    labels: list[Any] | None


LabelReplayState = dict[str, dict[str, LabelReplayMetadata]]


class LabelReplayRecord(TypedDict):
    """Label state before and after transforms add donor annotations."""

    initial: LabelReplayState
    final: LabelReplayState


class LabelManager:
    """Manages encoding, decoding, and type handling for label fields across data types.
    Maintains per-field metadata and encoders.

    This class handles the encoding, decoding, and type management for label fields.
    It maintains metadata about each field to ensure proper conversion between
    original and encoded representations.

    Args:
        metadata (dict[str, dict[str, LabelMetadata]]): Dictionary mapping data types
            and label fields to their metadata.

    """

    def __init__(self) -> None:
        self.metadata: dict[str, dict[str, LabelMetadata]] = defaultdict(dict)

    def process_field(self, data_name: str, label_field: str, field_data: Any) -> np.ndarray:
        """Process a label field: analyze or update metadata, fit or update encoder, return
        encoded data as ndarray. Used internally by Compose.

        If the field has been processed before (metadata exists), this will update
        the existing LabelEncoder with any new labels found in `field_data` before encoding.
        Otherwise, it analyzes the input, creates metadata, and fits the encoder.

        Args:
            data_name (str): The name of the main data type (e.g., 'bboxes', 'keypoints').
            label_field (str): The specific label field being processed (e.g., 'class_labels').
            field_data (Any): The actual label data for this field.

        Returns:
            np.ndarray: The encoded label data as a numpy array.

        """
        if data_name in self.metadata and label_field in self.metadata[data_name]:
            # Metadata exists, potentially update encoder
            metadata = self.metadata[data_name][label_field]
            if not metadata.is_numerical and metadata.encoder:
                metadata.encoder.update(field_data)
        else:
            # First time seeing this field, analyze and create metadata
            metadata = self._analyze_input(field_data)
            self.metadata[data_name][label_field] = metadata

        # Encode data using the (potentially updated) metadata/encoder
        return self._encode_data(field_data, metadata)

    def restore_field(self, data_name: str, label_field: str, encoded_data: np.ndarray) -> Any:
        """Restore a label field from encoded array to its original format and type using stored
        metadata and encoder. Inverse of process_field.
        """
        metadata = self.metadata[data_name][label_field]
        decoded_data = self._decode_data(encoded_data, metadata)
        return self._restore_type(decoded_data, metadata)

    def _analyze_input(self, field_data: Any) -> LabelMetadata:
        """Analyze input label data and create LabelMetadata (input_type, is_numerical, dtype,
        optional LabelEncoder). Used when processing a field.
        """
        input_type = type(field_data)
        dtype = field_data.dtype if isinstance(field_data, np.ndarray) else None

        # Determine if input is numerical. Handle empty case explicitly.
        if isinstance(field_data, np.ndarray) and field_data.size > 0:
            is_numerical = np.issubdtype(field_data.dtype, np.number)
        elif isinstance(field_data, Sequence) and not isinstance(field_data, str) and field_data:
            is_numerical = all(isinstance(label, (int, float)) for label in field_data)
        elif isinstance(field_data, (int, float)):
            is_numerical = True  # Handle single numeric item
        else:
            # Default to non-numerical for empty sequences, single strings, or other types
            is_numerical = False

        metadata = LabelMetadata(
            input_type=input_type,
            is_numerical=is_numerical,
            dtype=dtype,
        )

        if not is_numerical:
            # A categorical field keeps one index space even when another source contains only numbers.
            metadata.encoder = LabelEncoder().fit([])
            metadata.encoder.update(field_data)

        return metadata

    def _encode_data(self, field_data: Any, metadata: LabelMetadata) -> np.ndarray:
        """Encode field data for processing: numeric to float32, categorical via LabelEncoder.
        Returns ndarray. Used before applying transforms.
        """
        if metadata.is_numerical:
            # For numerical values, convert to float32 for processing
            if isinstance(field_data, np.ndarray):
                return field_data.reshape(-1, 1).astype(np.float32)
            return np.array(field_data, dtype=np.float32).reshape(-1, 1)

        # For non-numerical values, use LabelEncoder
        if metadata.encoder is None:
            raise ValueError("Encoder not initialized for non-numerical data")
        return metadata.encoder.transform(field_data).reshape(-1, 1)

    def _decode_data(self, encoded_data: np.ndarray, metadata: LabelMetadata) -> np.ndarray:
        """Decode encoded array back to labels using metadata (dtype or encoder inverse_transform).
        Returns ndarray. Used when restoring fields.
        """
        if metadata.is_numerical:
            if metadata.dtype is not None:
                return encoded_data.astype(metadata.dtype)
            return encoded_data.flatten()  # Flatten for list conversion

        if metadata.encoder is None:
            raise ValueError("Encoder not found for non-numerical data")

        decoded = metadata.encoder.inverse_transform(encoded_data.astype(int))
        return decoded.reshape(-1)  # Ensure 1D array

    def _restore_type(self, decoded_data: np.ndarray, metadata: LabelMetadata) -> Any:
        """Restore the label container without truncating categorical values."""
        if isinstance(metadata.input_type, type) and issubclass(metadata.input_type, (list, Sequence)):
            return decoded_data.tolist()

        if isinstance(metadata.input_type, type) and issubclass(metadata.input_type, np.ndarray):
            if metadata.dtype is not None:
                if metadata.dtype.kind in {"U", "S"} and decoded_data.size:
                    scalar_type = str if metadata.dtype.kind == "U" else bytes
                    if not all(isinstance(label, scalar_type) for label in decoded_data):
                        return decoded_data
                    restored = np.asarray(decoded_data.tolist())
                    return restored.astype(np.result_type(metadata.dtype, restored.dtype), copy=False)
                return decoded_data.astype(metadata.dtype)
            return decoded_data

        return decoded_data.tolist()

    def handle_empty_data(self) -> list[Any]:
        """Handle the case when label data is empty or missing by returning an empty
        list. Gives callers a consistent return type. For process_field.
        """
        return []

    def get_replay_state(self) -> LabelReplayState:
        """Snapshot label interpretation without retaining live processors or encoders."""
        return {
            data_name: {
                label_field: {
                    "is_numerical": metadata.is_numerical,
                    "is_array": issubclass(metadata.input_type, np.ndarray),
                    "dtype": None if metadata.dtype is None else metadata.dtype.str,
                    "labels": None if metadata.encoder is None else list(metadata.encoder.inverse_classes_.values()),
                }
                for label_field, metadata in fields.items()
            }
            for data_name, fields in self.metadata.items()
        }

    def get_replay_changes(self, initial: LabelReplayState) -> LabelReplayState:
        """Return fields whose interpretation changed after preprocessing."""
        changed: LabelReplayState = {}
        for data_name, fields in self.get_replay_state().items():
            for label_field, metadata in fields.items():
                if metadata != initial.get(data_name, {}).get(label_field):
                    changed.setdefault(data_name, {})[label_field] = metadata
        return changed

    def restore_replay_state(self, record: LabelReplayRecord) -> None:
        """Restore donor label interpretation into the current invocation's processors."""
        current = self.get_replay_state()
        for data_name, fields in record["final"].items():
            for label_field, saved in fields.items():
                initial = record["initial"].get(data_name, {}).get(label_field)
                actual = current.get(data_name, {}).get(label_field)
                if not self._compatible_replay_labels(actual, initial):
                    raise ValueError(
                        f"Replayed donor annotations require the recorded primary label encoding "
                        f"for '{data_name}.{label_field}'.",
                    )
                metadata = self.metadata[data_name].get(label_field)
                if metadata is None:
                    metadata = LabelMetadata(
                        input_type=np.ndarray if saved["is_array"] else list,
                        is_numerical=saved["is_numerical"],
                        dtype=None if saved["dtype"] is None else np.dtype(saved["dtype"]),
                    )
                    self.metadata[data_name][label_field] = metadata
                if (labels := saved["labels"]) is not None:
                    encoder = LabelEncoder().fit([])
                    encoder.classes_ = {label: index for index, label in enumerate(labels)}
                    encoder.inverse_classes_ = dict(enumerate(labels))
                    encoder.num_classes = len(labels)
                    metadata.encoder = encoder

    @staticmethod
    def _compatible_replay_labels(actual: LabelReplayMetadata | None, initial: LabelReplayMetadata | None) -> bool:
        if actual is None or initial is None:
            return actual is initial
        return actual["is_numerical"] == initial["is_numerical"] and actual["labels"] == initial["labels"]

    def get_encoder(self, data_name: str, label_field: str) -> LabelEncoder | None:
        """Return the fitted LabelEncoder for the given data name and label field, or None if not
        processed yet. Use to inspect encodings.
        """
        if data_name in self.metadata and label_field in self.metadata[data_name]:
            encoder = self.metadata[data_name][label_field].encoder
            if isinstance(encoder, LabelEncoder):
                return encoder
        return None
