"""Module for serialization and deserialization of Albumentations transforms.

This module provides functionality to serialize transforms to JSON or YAML format and
deserialize them back. It implements the Serializable interface that allows transforms
to be converted to and from dictionaries, which can then be saved to disk or transmitted
over a network. This is particularly useful for saving augmentation pipelines and
restoring them later with the exact same configuration.
"""

import importlib.util
import json
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TextIO
from warnings import warn

try:
    import yaml

    yaml_available = True
except ImportError:
    yaml_available = False


from albumentations._version import __version__

__all__ = ["from_dict", "load", "save", "to_dict"]


SERIALIZABLE_REGISTRY: dict[str, "SerializableMeta"] = {}
NON_SERIALIZABLE_REGISTRY: dict[str, "SerializableMeta"] = {}

# Cache for default p values to avoid repeated inspect.signature calls
_default_p_cache: dict[type, float] = {}


def shorten_class_name(class_fullname: str) -> str:
    # Split the class_fullname once at the last '.' to separate the class name
    split_index = class_fullname.rfind(".")

    # If there's no '.' or the top module is not 'albumentations', return the full name
    if split_index == -1 or not class_fullname.startswith("albumentations."):
        return class_fullname

    # Extract the class name after the last '.'
    return class_fullname[split_index + 1 :]


class SerializableMeta(ABCMeta):
    """Metaclass that registers transform classes for lookup by full name during deserialization.
    Uses SERIALIZABLE_REGISTRY / NON_SERIALIZABLE_REGISTRY.
    """

    def __new__(cls, name: str, bases: tuple[type, ...], *args: Any, **kwargs: Any) -> "SerializableMeta":
        cls_obj = super().__new__(cls, name, bases, *args, **kwargs)
        if name != "Serializable" and ABC not in bases:
            if cls_obj.is_serializable():
                SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
            else:
                NON_SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
        return cls_obj

    @classmethod
    def is_serializable(cls) -> bool:
        """Return whether the class is registered for serialization. Subclasses override to True;
        default is False. Check this when saving or loading pipelines.

        Returns:
            bool: False by default. Subclasses override this to return True if they
                support serialization.

        """
        return False

    @classmethod
    def get_class_fullname(cls) -> str:
        """Return shortest full class name used in serialization registry (e.g.
        module.ClassName or alias). Uniquely identifies the class.

        Returns:
            str: The shortened class name that uniquely identifies this class
                in the serialization registry.

        """
        return get_shortest_class_fullname(cls)

    @classmethod
    def _to_dict(cls) -> dict[str, Any]:
        return {}


class Serializable(metaclass=SerializableMeta):
    @classmethod
    @abstractmethod
    def is_serializable(cls) -> bool:
        """Return True if the class supports serialization (used for registry). Subclasses must
        implement this method. Check when saving or loading pipelines.

        Subclasses must implement this method to indicate whether they support
        serialization. Classes that return True will be registered in SERIALIZABLE_REGISTRY.

        Returns:
            bool: True if the class supports serialization, False otherwise.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_class_fullname(cls) -> str:
        """Return unique class name for serialization (e.g. module.ClassName). Required when
        saving or loading pipelines. Subclasses must implement.

        This method returns a unique identifier for the class that is used when
        serializing and deserializing. The name must be unique across all
        serializable classes.

        Returns:
            str: The full class name (typically module.ClassName format).

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.

        """
        raise NotImplementedError

    @abstractmethod
    def to_dict_private(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self, on_not_implemented_error: str = "raise") -> dict[str, Any]:
        """Convert this transform to a serializable dict (dict, list, str, int, float).
        Use on_not_implemented_error to raise or warn.

        Args:
            self (Serializable): A transform that should be serialized. If the transform doesn't implement the `to_dict`
                method and `on_not_implemented_error` equals to 'raise' then `NotImplementedError` is raised.
                If `on_not_implemented_error` equals to 'warn' then `NotImplementedError` will be ignored
                but no transform parameters will be serialized.
            on_not_implemented_error (str): `raise` or `warn`.

        """
        if on_not_implemented_error not in {"raise", "warn"}:
            msg = f"Unknown on_not_implemented_error value: {on_not_implemented_error}. Supported values are: 'raise' "
            "and 'warn'"
            raise ValueError(msg)
        try:
            transform_dict = self.to_dict_private()
        except NotImplementedError:
            if on_not_implemented_error == "raise":
                raise

            transform_dict = {}
            warnings.warn(
                f"Got NotImplementedError while trying to serialize {self}. Object arguments are not preserved. "
                f"The transform class '{self.__class__.__name__}' needs to implement 'to_dict_private' or inherit from "
                f"BasicTransform to be properly serialized.",
                stacklevel=2,
            )
        return {"__version__": __version__, "transform": transform_dict}


def to_dict(transform: Serializable, on_not_implemented_error: str = "raise") -> dict[str, Any]:
    """Convert a transform to a serializable dict of standard Python types.
    Delegates to transform.to_dict; on_not_implemented_error: raise or warn.

    Args:
        transform (Serializable): A transform that should be serialized. If the transform doesn't implement
            the `to_dict` method and `on_not_implemented_error` equals to 'raise' then `NotImplementedError` is raised.
            If `on_not_implemented_error` equals to 'warn' then `NotImplementedError` will be ignored
            but no transform parameters will be serialized.
        on_not_implemented_error (str): `raise` or `warn`.

    """
    return transform.to_dict(on_not_implemented_error)


def instantiate_nonserializable(
    transform: dict[str, Any],
    nonserializable: dict[str, Any] | None = None,
) -> Serializable | None:
    if transform.get("__class_fullname__") in NON_SERIALIZABLE_REGISTRY:
        name = transform["__name__"]
        if nonserializable is None:
            msg = f"To deserialize a non-serializable transform with name {name} you need to pass a dict with"
            "this transform as the `lambda_transforms` argument"
            raise ValueError(msg)
        result_transform = nonserializable.get(name)
        if transform is None:
            raise ValueError(f"Non-serializable transform with {name} was not found in `nonserializable`")
        return result_transform
    return None


def from_dict(
    transform_dict: dict[str, Any],
    nonserializable: dict[str, Any] | None = None,
) -> Serializable | None:
    """Restore a transform (or pipeline) from a serialized dict. Pass nonserializable
    for Lambda/custom transforms keyed by name.

    Args:
        transform_dict (dict[str, Any]): Serialized transform pipeline.
        nonserializable (dict[str, Any] | None): Optional dict of non-serializable transforms keyed by name.

    """
    register_additional_transforms()
    transform = transform_dict["transform"]
    lmbd = instantiate_nonserializable(transform, nonserializable)
    if lmbd:
        return lmbd
    name = transform["__class_fullname__"]
    args = {k: v for k, v in transform.items() if k != "__class_fullname__"}

    # Get the transform class from registry
    cls = SERIALIZABLE_REGISTRY[shorten_class_name(name)]

    # Handle missing 'p' parameter for backward compatibility
    if "p" not in args:
        # Import here to avoid circular imports
        from albumentations.core.composition import BaseCompose

        # Check if it's a composition class by verifying if it is a subclass of BaseCompose
        if not issubclass(cls, BaseCompose):
            # Check if default 'p' value is cached
            if cls not in _default_p_cache:
                # Use inspect to get the default value of p from __init__
                import inspect

                sig = inspect.signature(cls.__init__)
                p_param = sig.parameters.get("p")
                default_p = p_param.default if p_param and p_param.default != inspect.Parameter.empty else 0.5
                _default_p_cache[cls] = default_p
            else:
                default_p = _default_p_cache[cls]

            warn(
                f"Transform {cls.__name__} has no 'p' parameter in serialized data, defaulting to {default_p}",
                stacklevel=2,
            )
            args["p"] = default_p

    # Handle nested transforms
    if "transforms" in args:
        args["transforms"] = [from_dict({"transform": t}, nonserializable=nonserializable) for t in args["transforms"]]

    return cls(**args)


def check_data_format(data_format: Literal["json", "yaml"]) -> None:
    if data_format not in {"json", "yaml"}:
        raise ValueError(f"Unknown data_format {data_format}. Supported formats are: 'json' and 'yaml'")


def serialize_enum(obj: Any) -> Any:
    """Recursively replace Enum instances with their value; traverse Mappings and
    Sequences. Call before saving pipeline to JSON/YAML.
    """
    if isinstance(obj, Mapping):
        return {k: serialize_enum(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, str):  # exclude strings since they're also sequences
        return [serialize_enum(v) for v in obj]
    return obj.value if isinstance(obj, Enum) else obj


def save(
    transform: Serializable,
    filepath_or_buffer: str | Path | TextIO,
    data_format: Literal["json", "yaml"] = "json",
    on_not_implemented_error: Literal["raise", "warn"] = "raise",
) -> None:
    """Serialize a transform pipeline to a file or file-like object in JSON or YAML.
    Use on_not_implemented_error to raise or warn if a transform lacks to_dict.

    Args:
        transform (Serializable): The transform pipeline to serialize.
        filepath_or_buffer (str | Path | TextIO): The file path or file-like object to write the serialized
            data to. String is interpreted as a path; file-like object is written to directly.
        data_format (Literal['json', 'yaml']): The format to serialize the data in. Defaults to 'json'.
        on_not_implemented_error (Literal['raise', 'warn']): If a transform does not implement to_dict:
            'raise' raises NotImplementedError; 'warn' ignores and omits transform arguments. Defaults to 'raise'.

    Raises:
        ValueError: If `data_format` is 'yaml' but PyYAML is not installed.

    """
    check_data_format(data_format)
    transform_dict = transform.to_dict(on_not_implemented_error=on_not_implemented_error)
    transform_dict = serialize_enum(transform_dict)

    # Determine whether to write to a file or a file-like object
    if isinstance(filepath_or_buffer, (str, Path)):  # It's a filepath
        with Path(filepath_or_buffer).open("w") as f:
            if data_format == "yaml":
                if not yaml_available:
                    msg = "You need to install PyYAML to save a pipeline in YAML format"
                    raise ValueError(msg)
                yaml.safe_dump(transform_dict, f, default_flow_style=False)
            elif data_format == "json":
                json.dump(transform_dict, f)
    elif data_format == "yaml":
        if not yaml_available:
            msg = "You need to install PyYAML to save a pipeline in YAML format"
            raise ValueError(msg)
        yaml.safe_dump(transform_dict, filepath_or_buffer, default_flow_style=False)
    elif data_format == "json":
        json.dump(transform_dict, filepath_or_buffer, indent=2)


def load(
    filepath_or_buffer: str | Path | TextIO,
    data_format: Literal["json", "yaml"] = "json",
    nonserializable: dict[str, Any] | None = None,
) -> object:
    """Load a serialized transform pipeline from file or file-like object (JSON or YAML).
    Pass nonserializable for Lambda/custom.

    Args:
        filepath_or_buffer (str | Path | TextIO): The file path or file-like object to read the serialized
            data from. String is interpreted as a path; file-like object is read from directly.
        data_format (Literal['json', 'yaml']): The format of the serialized data.
            Defaults to 'json'.
        nonserializable (dict[str, Any] | None): A dictionary that contains non-serializable transforms.
            This dictionary is required when restoring a pipeline that contains non-serializable transforms.
            Keys in the dictionary should be named the same as the `name` arguments in respective transforms
            from the serialized pipeline. Defaults to None.

    Returns:
        object: The deserialized transform pipeline.

    Raises:
        ValueError: If `data_format` is 'yaml' but PyYAML is not installed.

    """
    check_data_format(data_format)

    if isinstance(filepath_or_buffer, (str, Path)):  # Assume it's a filepath
        with Path(filepath_or_buffer).open() as f:
            if data_format == "json":
                transform_dict = json.load(f)
            else:
                if not yaml_available:
                    msg = "You need to install PyYAML to load a pipeline in yaml format"
                    raise ValueError(msg)
                transform_dict = yaml.safe_load(f)
    elif data_format == "json":
        transform_dict = json.load(filepath_or_buffer)
    else:
        if not yaml_available:
            msg = "You need to install PyYAML to load a pipeline in yaml format"
            raise ValueError(msg)
        transform_dict = yaml.safe_load(filepath_or_buffer)

    return from_dict(transform_dict, nonserializable=nonserializable)


def register_additional_transforms() -> None:
    """Register transforms that are not imported directly into the `albumentations` module by checking
    the availability of optional dependencies.
    """
    if importlib.util.find_spec("torch") is not None:
        try:
            # Import `albumentations.pytorch` only if `torch` is installed.
            import albumentations.pytorch

            # Use a dummy operation to acknowledge the use of the imported module and avoid linting errors.
            _ = albumentations.pytorch.ToTensorV2
        except ImportError:
            pass


def get_shortest_class_fullname(cls: type[Any]) -> str:
    """Return the shortest full class name for a class (e.g. module.ClassName or alias).
    Used for serialization registry lookup.

    Args:
        cls (type[Any]): Class (e.g. a transform or Compose subclass).

    Returns:
        str: Shortened full class name.

    """
    class_fullname = f"{cls.__module__}.{cls.__name__}"
    return shorten_class_name(class_fullname)
