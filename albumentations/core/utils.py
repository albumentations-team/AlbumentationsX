"""Module containing utility functions and classes for the core Albumentations framework.

This module provides a collection of helper functions and base classes used throughout
the Albumentations library. It includes utilities for shape handling, parameter processing,
data conversion, and serialization. The module defines abstract base classes for data
processors that implement the conversion logic between different data formats used in
the transformation pipeline.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np
import torch

from albumentations.core.label_manager import LabelManager

from .serialization import Serializable
from .type_definitions import NUM_VOLUME_DIMENSIONS


def get_shape(data: dict[str, Any]) -> tuple[int, int]:
    """Extract (height, width) from data dict. Keys: image, images, volume.
    Raises if no image/volume present. Call for spatial checks during pipeline.

    After grayscale preprocessing, all data has channel dimension at the end.

    Args:
        data (dict[str, Any]): Dictionary containing image or volume data with one of:
            - 'volume': 3D array of shape (D, H, W, C)
            - 'image': 2D array of shape (H, W, C)
            - 'images': Batch of arrays of shape (N, H, W, C)

    Returns:
        tuple[int, int]: (height, width) dimensions

    """
    if "image" in data:
        return _get_shape_from_image(data["image"])
    if "images" in data:
        return _get_shape_from_images(data["images"])
    if "volume" in data:
        return _get_shape_from_volume(data["volume"])
    raise ValueError("No image or volume found in data", data.keys())


def get_image_data(data: dict[str, Any]) -> dict[str, Any]:
    """Extract dtype, spatial dimensions, and channel count from the first canonical image, image batch, or
    volume in core pipelines.

    NumPy inputs use HWC, NHWC, or DHWC layout. Tensor inputs use CHW, NCHW, or CDHW layout.

    Args:
        data (dict[str, Any]): Dictionary potentially containing image/volume arrays.

    Returns:
        dict[str, Any]: Dictionary with 'dtype', 'height', 'width', 'num_channels' keys.

    Raises:
        ValueError: If no valid image/volume data keys are found in the dictionary.

    """
    for target in ("image", "images", "volume"):
        array = data.get(target)
        if array is None:
            continue
        shape = array.shape
        if isinstance(array, torch.Tensor):
            if target == "image":
                height, width = shape[1], shape[2]
                num_channels = int(shape[0])
            elif target == "images":
                height, width = shape[2], shape[3]
                num_channels = int(shape[1])
            else:
                _, height, width = get_volume_shape(array)
                num_channels = int(shape[0])
        elif target == "image":
            height, width = shape[0], shape[1]
            num_channels = shape[-1]
        elif target == "images":
            height, width = shape[1], shape[2]
            num_channels = shape[-1]
        else:
            _, height, width = get_volume_shape(array)
            num_channels = shape[-1]
        return {
            "dtype": array.dtype,
            "height": height,
            "width": width,
            "num_channels": num_channels,
        }
    raise ValueError("No valid image/volume data found in data dict")


def get_volume_shape(volume: np.ndarray | torch.Tensor) -> tuple[int, int, int]:
    """Return `(D, H, W)` from a volume in its canonical container layout.

    NumPy volumes are `(D, H, W, C)` and Tensor volumes are `(C, D, H, W)` inside Compose. Channel-less `(D, H, W)`
    arrays are retained for direct functional calls.
    """
    if volume.ndim == NUM_VOLUME_DIMENSIONS - 1:
        return cast("tuple[int, int, int]", tuple(volume.shape))
    return cast(
        "tuple[int, int, int]",
        tuple(volume.shape[1:] if isinstance(volume, torch.Tensor) else volume.shape[:-1]),
    )


def _get_shape_from_image(img: np.ndarray | torch.Tensor) -> tuple[int, int]:
    """Extract (height, width) from a single image. Handles numpy HWC or PyTorch CHW. Private
    helper for get_shape when data has 'image' key.
    """
    if isinstance(img, torch.Tensor):
        return img.shape[1], img.shape[2]
    # Regular numpy array in HWC format
    return img.shape[0], img.shape[1]


def _get_shape_from_images(imgs: np.ndarray | torch.Tensor) -> tuple[int, int]:
    """Extract height and width from a NumPy NHWC batch or CPU Tensor NCHW sequence."""
    if isinstance(imgs, torch.Tensor):
        return imgs.shape[-2], imgs.shape[-1]
    # Regular numpy array batch in NHWC format - take first image
    return imgs[0].shape[0], imgs[0].shape[1]


def _get_shape_from_volume(vol: np.ndarray | torch.Tensor) -> tuple[int, int]:
    """Extract (height, width) from a single volume (D,H,W or D,H,W,C). Private helper for
    get_shape when data has 'volume' key.
    """
    _, height, width = get_volume_shape(vol)
    return height, width


def format_args(args_dict: dict[str, Any]) -> str:
    """Format a dict of argument names and values as "key1='val1', key2=val2" for repr.
    Strings are quoted; other values passed through str(). For transform __repr__.

    Args:
        args_dict (dict[str, Any]): Dictionary of argument names and values.

    Returns:
        str: Formatted string of arguments in the form "key1='value1', key2=value2".

    """
    formatted_args = []
    for k, v in args_dict.items():
        v_formatted = f"'{v}'" if isinstance(v, str) else str(v)
        formatted_args.append(f"{k}={v_formatted}")
    return ", ".join(formatted_args)


class Params(Serializable, ABC):
    """Base class for transform data params: coord_format and label_fields.
    BboxParams and KeypointParams subclass this. Serializable.

    Args:
        coord_format (Any): The coordinate format of the data this parameter object will process.
        label_fields (Sequence[str] | None): List of fields that are joined with the data, such as labels.

    """

    def __init__(self, coord_format: Any, label_fields: Sequence[str] | None):
        self.coord_format = coord_format
        self.label_fields = label_fields

    def to_dict_private(self) -> dict[str, Any]:
        """Return dict of private params (coord_format, label_fields) for serialization.
        BboxParams/KeypointParams override; not part of public API.

        Returns:
            dict[str, Any]: Dictionary with coord_format and label_fields parameters.

        """
        return {"coord_format": self.coord_format, "label_fields": self.label_fields}


ParamsT = TypeVar("ParamsT", bound=Params)


class DataProcessor(ABC, Generic[ParamsT]):
    """Abstract base for data processors: convert, validate, filter. Subclasses: BboxProcessor,
    KeypointsProcessor. Uses Params.

    Data processors handle the conversion, validation, and filtering of data
    during transformations.

    Args:
        params (Params): Parameters for data processing.
        additional_targets (dict[str, str] | None): Dictionary mapping additional target names to their types.

    """

    def __init__(self, params: ParamsT, additional_targets: dict[str, str] | None = None):
        self.params = params
        self.data_fields = [self.default_data_name]
        self.label_manager = LabelManager()
        # Cached so get_shape lookups can resolve aliased image/mask/volume keys
        # (e.g. {'custom_image_key': 'image'}). Updated by add_targets.
        self._additional_targets: dict[str, str] = {}

        if additional_targets is not None:
            self.add_targets(additional_targets)

    @property
    @abstractmethod
    def default_data_name(self) -> str:
        """Return the default key for this processor's data (e.g. 'bboxes', 'keypoints').
        Used to resolve additional_targets and data_fields. Abstract.

        Returns:
            str: Default data field name.

        """
        raise NotImplementedError

    def add_targets(self, additional_targets: dict[str, str]) -> None:
        """Register additional target keys processed like `default_data_name`, and cache
        the full alias map so shape-helper lookups resolve aliased image/mask/volume keys.

        Maps name to type; type must match `default_data_name`. Caching the full alias
        map lets `get_shape`/`get_volume_shape` resolve aliased image/mask/volume keys.
        Compose calls this when building the pipeline.
        """
        self._additional_targets.update(additional_targets)
        for k, v in additional_targets.items():
            if v == self.default_data_name and k not in self.data_fields:
                self.data_fields.append(k)

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        """Validate that input data dict has required keys and structure before processing.
        Override in subclasses. Called at pipeline apply time.

        Args:
            data (dict[str, Any]): Input data dictionary to validate.

        """

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        """Validate that the transform list is compatible with this processor (e.g. bbox_type).
        Override in BboxProcessor. Called at Compose init.

        Args:
            transforms (Sequence[object]): Sequence of transforms to validate.

        """

    def postprocess(self, data: dict[str, Any], *, filter_already_applied: bool = False) -> dict[str, Any]:
        """Convert data from Albumentations format back to user format and remove label fields.
        Uses shape from get_shape(data). Called after all transforms applied.

        Args:
            data (dict[str, Any]): Data dictionary after transformation.
            filter_already_applied (bool): Skip the final filter when target-aware execution already produced the
                current annotation rows.

        Returns:
            dict[str, Any]: Processed data dictionary.

        """
        shape: tuple[int, int] | tuple[int, int, int] = get_shape(data)

        # For xyz keypoints, get full 3D shape if available
        if hasattr(self.params, "coord_format") and self.params.coord_format == "xyz":
            volume = data.get("volume")
            if volume is not None:
                shape = get_volume_shape(volume)

        data = self._process_data_fields(data, shape, filter_already_applied=filter_already_applied)
        return self.remove_label_fields_from_data(data)

    def _process_data_fields(
        self,
        data: dict[str, Any],
        shape: tuple[int, int] | tuple[int, int, int],
        *,
        filter_already_applied: bool,
    ) -> dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self._process_single_field(
                data_name,
                data[data_name],
                shape,
                filter_already_applied=filter_already_applied,
            )
        return data

    def _process_single_field(
        self,
        data_name: str,
        field_data: Any,
        shape: tuple[int, int] | tuple[int, int, int],
        *,
        filter_already_applied: bool,
    ) -> Any:
        if not filter_already_applied:
            field_data = self.filter(field_data, shape)

        if data_name == "keypoints" and len(field_data) == 0:
            field_data = self._create_empty_keypoints_array()

        return self.check_and_convert(field_data, shape, direction="from")

    def _create_empty_keypoints_array(self) -> np.ndarray:
        return np.array([], dtype=np.float32).reshape(0, len(self.params.coord_format))

    def preprocess(self, data: dict[str, Any]) -> None:
        """Convert data to Albumentations format and add label fields. Mutates data in place.
        Uses get_shape(data). Called before transforms applied.

        Args:
            data (dict[str, Any]): Data dictionary to preprocess.

        """
        shape = get_shape(data)

        for data_name in set(self.data_fields) & set(data.keys()):
            if isinstance(data[data_name], Sequence) and not isinstance(data[data_name], np.ndarray):
                if len(data[data_name]) > 0:
                    data[data_name] = np.array(data[data_name], dtype=np.float32)
                else:
                    data[data_name] = self._create_empty_array()

        data = self.add_label_fields_to_data(data)
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self.check_and_convert(data[data_name], shape, direction="to")

    def check_and_convert(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
        direction: Literal["to", "from"] = "to",
    ) -> np.ndarray:
        """Validate and convert data to/from Albumentations format. direction 'to' for input, 'from'
        for output. Uses coord_format.

        Args:
            data (np.ndarray): Input data array.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.
            direction (Literal['to', 'from']): Conversion direction.
                "to" converts to Albumentations format, "from" converts from it.
                Defaults to "to".

        Returns:
            np.ndarray: Converted data array.

        """
        if self.params.coord_format == "albumentations":
            self.check(data, shape)
            return data

        process_func = self.convert_to_albumentations if direction == "to" else self.convert_from_albumentations
        return process_func(data, shape)

    def _create_empty_array(self) -> np.ndarray:
        """Create an empty array with shape (0, num_coords+) for this processor. Call when the
        user passes an empty list for bboxes/keypoints. Default: (0,) float32.

        Returns:
            np.ndarray: Empty array with correct shape.

        """
        # Default implementation - subclasses can override
        return np.array([], dtype=np.float32)

    @abstractmethod
    def filter(self, data: np.ndarray, shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
        """Remove rows outside image/volume bounds. shape is (H, W) or (D, H, W). Abstract;
        BboxProcessor/KeypointsProcessor implement. Call during pipeline postprocess.

        Args:
            data (np.ndarray): Data to filter.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        Returns:
            np.ndarray: Filtered data.

        """

    @abstractmethod
    def check(self, data: np.ndarray, shape: tuple[int, int] | tuple[int, int, int]) -> None:
        """Validate data array shape and value ranges for given image/volume shape.
        Raises on invalid. Abstract; call during check_and_convert.

        Args:
            data (np.ndarray): Data to validate.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        """

    @abstractmethod
    def convert_to_albumentations(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Convert from user coord format to internal normalized format. shape (H, W) or (D, H, W).
        Abstract. Called during pipeline preprocess.

        Args:
            data (np.ndarray): Data in external format.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        Returns:
            np.ndarray: Data in Albumentations format.

        """

    @abstractmethod
    def convert_from_albumentations(
        self,
        data: np.ndarray,
        shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Convert from internal format back to user coord format. shape (H, W) or (D, H, W).
        Abstract. Called during pipeline postprocess.

        Args:
            data (np.ndarray): Data in Albumentations format.
            shape (tuple[int, int] | tuple[int, int, int]): Shape information containing dimensions.

        Returns:
            np.ndarray: Data in external format.

        """

    def add_label_fields_to_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append encoded label columns to bbox/keypoint arrays and remove separate label keys. Uses
        params.label_fields. Called during pipeline preprocess.

        This method processes label fields and joins them with the corresponding data arrays.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Data with label fields added.

        """
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            # Skip empty sequences (will be converted to proper empty arrays in check_and_convert)
            if isinstance(data[data_name], Sequence) and len(data[data_name]) == 0:
                continue
            if isinstance(data[data_name], np.ndarray) and not data[data_name].size:
                continue
            data[data_name] = self._process_label_fields(data, data_name)

        return data

    def _process_label_fields(self, data: dict[str, Any], data_name: str) -> np.ndarray:
        data_array = data[data_name]
        label_fields = self.params.label_fields
        if not label_fields:
            return data_array

        encoded_columns: list[np.ndarray] = []
        for label_field in label_fields:
            self._validate_label_field_length(data, data_name, label_field)
            encoded_columns.append(self.label_manager.process_field(data_name, label_field, data[label_field]))

        result = np.empty(
            (data_array.shape[0], data_array.shape[1] + len(encoded_columns)),
            dtype=np.result_type(data_array, *encoded_columns),
        )
        result[:, : data_array.shape[1]] = data_array
        for index, encoded_column in enumerate(encoded_columns, start=data_array.shape[1]):
            result[:, index] = encoded_column[:, 0]
        for label_field in label_fields:
            del data[label_field]
        return result

    def _validate_label_field_length(self, data: dict[str, Any], data_name: str, label_field: str) -> None:
        if len(data[data_name]) != len(data[label_field]):
            raise ValueError(
                f"The lengths of {data_name} and {label_field} do not match. "
                f"Got {len(data[data_name])} and {len(data[label_field])} respectively.",
            )

    def remove_label_fields_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Split encoded label columns off data arrays and restore as separate dict keys.
        Inverse of add_label_fields_to_data. Call during postprocess. Mutates data.

        Args:
            data (dict[str, Any]): Input data dictionary with combined label fields.

        Returns:
            dict[str, Any]: Data with label fields extracted as separate entries.

        """
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            if not data[data_name].size:
                self._handle_empty_data_array(data)
                continue
            self._remove_label_fields(data, data_name)

        return data

    def _handle_empty_data_array(self, data: dict[str, Any]) -> None:
        if self.params.label_fields is not None:
            for label_field in self.params.label_fields:
                data[label_field] = self.label_manager.handle_empty_data()

    def _remove_label_fields(self, data: dict[str, Any], data_name: str) -> None:
        if self.params.label_fields is None:
            return

        data_array = data[data_name]
        num_label_fields = len(self.params.label_fields)
        non_label_columns = data_array.shape[1] - num_label_fields

        for idx, label_field in enumerate(self.params.label_fields):
            encoded_labels = data_array[:, non_label_columns + idx]
            data[label_field] = self.label_manager.restore_field(data_name, label_field, encoded_labels)

        data[data_name] = data_array[:, :non_label_columns]
