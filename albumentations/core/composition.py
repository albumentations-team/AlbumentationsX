"""Module for composing multiple transforms into augmentation pipelines.
This module provides classes for combining multiple transformations into cohesive
augmentation pipelines. It includes various composition strategies such as sequential
application, random selection, and conditional application of transforms. These
composition classes handle the coordination between different transforms, ensuring
proper data flow and maintaining consistent behavior across the augmentation pipeline.
"""

import contextlib
import copy
import inspect
import random
import types
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Union, cast, get_args, get_origin

import cv2
import numpy as np
from numpy.typing import NDArray

from .analytics.collectors import collect_pipeline_info, get_environment_info

# Telemetry imports
from .analytics.settings import settings
from .analytics.telemetry import get_telemetry_client
from .bbox_utils import BboxParams, BboxProcessor
from .hub_mixin import HubMixin
from .keypoints_utils import KeypointParams, KeypointsProcessor
from .serialization import (
    SERIALIZABLE_REGISTRY,
    Serializable,
    get_shortest_class_fullname,
    instantiate_nonserializable,
    register_additional_transforms,
)
from .transforms_interface import BasicTransform
from .type_definitions import StackedMasks4D
from .utils import DataProcessor, format_args, get_shape

__all__ = [
    "BaseCompose",
    "BboxParams",
    "Compose",
    "KeypointParams",
    "OneOf",
    "OneOrOther",
    "RandomOrder",
    "ReplayCompose",
    "SelectiveChannelTransform",
    "Sequential",
    "SomeOf",
]

NUM_ONEOF_TRANSFORMS = 2

_RANGE_PARAMS_CACHE: dict[type, frozenset[str]] = {}


def _get_range_param_names(cls: type) -> frozenset[str]:
    """Return constructor parameter names whose type annotation accepts a tuple range value
    (e.g. tuple[int, int] | int). Results are cached per class for performance.
    """
    if cls in _RANGE_PARAMS_CACHE:
        return _RANGE_PARAMS_CACHE[cls]

    range_params: set[str] = set()
    for klass in cls.__mro__:
        if klass is object or "__init__" not in klass.__dict__:
            continue
        try:
            sig = inspect.signature(klass.__dict__["__init__"])
            for name, param in sig.parameters.items():
                if name == "self" or param.annotation is inspect.Parameter.empty:
                    continue
                if _annotation_accepts_tuple(param.annotation):
                    range_params.add(name)
        except (ValueError, TypeError):
            continue

    result = frozenset(range_params)
    _RANGE_PARAMS_CACHE[cls] = result
    return result


def _annotation_accepts_tuple(annotation: Any) -> bool:
    """Return True if the annotation includes tuple as an accepted type
    (e.g. tuple[int,int]|int). Handles typing.Union, types.UnionType (X|Y), and str annotations.
    """
    origin = get_origin(annotation)

    if origin is tuple:
        return True

    if origin is Union or isinstance(annotation, types.UnionType):
        return any(_annotation_accepts_tuple(arg) for arg in get_args(annotation))

    return isinstance(annotation, str) and "tuple" in annotation.lower()


def _wrap_scalars_for_replay(cls: type, config: dict[str, Any]) -> dict[str, Any]:
    """Convert scalar values in config to (v,v) degenerate range tuples for params
    that expect ranges (e.g. blur_range=5 becomes (5,5)) to skip symmetric expansion.
    """
    range_params = _get_range_param_names(cls)
    result = {}
    for key, value in config.items():
        if key in range_params and not isinstance(value, (tuple, list)):
            result[key] = (value, value)
        else:
            result[key] = value
    return result


REPR_INDENT_STEP = 2

TransformType = Union[BasicTransform, "BaseCompose"]
TransformsSeqType = list[TransformType]

AVAILABLE_KEYS = (
    "image",
    "mask",
    "masks",
    "bboxes",
    "keypoints",
    "volume",
    "volumes",
    "mask3d",
    "masks3d",
    "user_data",
)

MASK_KEYS = (
    "mask",  # 2D mask
    "masks",  # Multiple 2D masks
    "mask3d",  # 3D mask
    "masks3d",  # Multiple 3D masks
)

# Keys related to image data
IMAGE_KEYS = {"image", "images"}
CHECK_BBOX_PARAM = {"bboxes"}
CHECK_KEYPOINTS_PARAM = {"keypoints"}
VOLUME_KEYS = {"volume", "volumes"}

_VALID_INSTANCE_BINDING_TARGETS = frozenset({"mask", "masks", "bboxes", "keypoints"})
_BBOX_INSTANCE_ID = "_bbox_instance_id"
_KP_INSTANCE_ID = "_kp_instance_id"


def _make_stacked_masks(rows: list[np.ndarray]) -> StackedMasks4D:
    """Sole construction site for stacked instance masks; only place the canonical 4-D
    `(N, H, W, C)` shape brand is minted from raw per-instance arrays.

    Input rows may be `(H, W)` or `(H, W, C)`; output is always `(N, H, W, C)` with the canonical
    trailing channel dim added here so every consumer can index `masks.shape[3]` without rank
    checks.

    Empty `rows` returns a zero-row 4-D placeholder with `C=1` since no per-instance shape is known.
    """
    if not rows:
        return StackedMasks4D(np.empty((0, 0, 0, 1), dtype=np.uint8))
    arr = np.stack(rows, axis=0)
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    return StackedMasks4D(arr)


class BaseCompose(Serializable):
    """Base class for composing multiple transforms. Supports +, __radd__, - for pipeline
    modification; serialization; add_targets, set_deterministic.

    This class serves as a foundation for creating compositions of transforms
    in the Albumentations library. It provides basic functionality for
    managing a sequence of transforms and applying them to data.

    The class supports dynamic pipeline modification after initialization using
    mathematical operators:
    - Addition (`+`): Add transforms to the end of the pipeline
    - Right addition (`__radd__`): Add transforms to the beginning of the pipeline
    - Subtraction (`-`): Remove transforms by class from the pipeline

    Attributes:
        transforms (List[TransformType]): A list of transforms to be applied.
        p (float): Probability of applying the compose. Should be in the range [0, 1].
        replay_mode (bool): If True, the compose is in replay mode.
        _additional_targets (Dict[str, str]): Additional targets for transforms.
        _available_keys (Set[str]): Set of available keys for data.
        processors (Dict[str, Union[BboxProcessor, KeypointsProcessor]]): Processors for specific data types.

    Args:
        transforms (TransformsSeqType): A sequence of transforms to compose.
        p (float): Probability of applying the compose.

    Raises:
        ValueError: If an invalid additional target is specified.

    Note:
        - Subclasses should implement the __call__ method to define how
          the composition is applied to data.
        - The class supports serialization and deserialization of transforms.
        - It provides methods for adding targets, setting deterministic behavior,
          and checking data validity post-transform.
        - All compose classes support pipeline modification operators:
          - `compose + transform` adds individual transform(s) to the end
          - `transform + compose` adds individual transform(s) to the beginning
          - `compose - TransformClass` removes transforms by class type
          - Only BasicTransform instances (not BaseCompose) can be added
        - All operator operations return new instances without modifying the original.

    Examples:
        >>> import albumentations as A
        >>> # Create base pipeline
        >>> compose = A.Compose([A.HorizontalFlip(p=1.0)])
        >>>
        >>> # Add transforms using operators
        >>> extended = compose + A.VerticalFlip(p=1.0)  # Append
        >>> extended = compose + [A.Blur(), A.Rotate()]  # Append multiple
        >>> extended = A.RandomCrop(256, 256) + compose  # Prepend
        >>>
        >>> # Remove transforms by class
        >>> compose = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=1.0)])
        >>> reduced = compose - A.HorizontalFlip  # Remove by class

    """

    _transforms_dict: dict[int, BasicTransform] | None = None
    check_each_transform: tuple[DataProcessor, ...] | None = None
    main_compose: bool = True

    def __init__(
        self,
        transforms: TransformsSeqType,
        p: float,
        mask_interpolation: int | None = None,
        seed: int | None = None,
        save_applied_params: bool = False,
        **kwargs: Any,
    ):
        if isinstance(transforms, (BaseCompose, BasicTransform)):
            warnings.warn(
                "transforms is single transform, but a sequence is expected! Transform will be wrapped into list.",
                stacklevel=2,
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

        self.replay_mode = False
        self._additional_targets: dict[str, str] = {}
        self._available_keys: set[str] = set()
        self.processors: dict[str, BboxProcessor | KeypointsProcessor] = {}
        self._set_keys()
        self.set_mask_interpolation(mask_interpolation)
        self.set_random_seed(seed)
        self.save_applied_params = save_applied_params

    def _track_transform_params(self, transform: TransformType, data: dict[str, Any]) -> None:
        """Append a (class_fullname, applied_config) tuple to applied_transforms when
        save_applied_params=True. Skipped transforms (empty applied_config) are not recorded.
        """
        if "applied_transforms" in data and hasattr(transform, "applied_config") and transform.applied_config:
            data["applied_transforms"].append(
                (transform.get_class_fullname(), transform.applied_config.copy()),
            )

    def set_random_state(
        self,
        random_generator: np.random.Generator,
        py_random: random.Random,
    ) -> None:
        """Set random state directly from numpy and Python random generators. Propagates to all
        child transforms. Used for reproducibility.

        Args:
            random_generator (np.random.Generator): numpy random generator to use
            py_random (random.Random): python random generator to use

        """
        self.random_generator = random_generator
        self.py_random = py_random

        # Propagate both random states to all transforms
        for transform in self.transforms:
            if isinstance(transform, (BasicTransform, BaseCompose)):
                transform.set_random_state(random_generator, py_random)

    def set_random_seed(self, seed: int | None) -> None:
        """Set random state from a single integer seed. Propagates to all child transforms.
        Used for reproducibility; stored as self.seed.

        Args:
            seed (int | None): Random seed to use

        """
        # Store the original seed
        self.seed = seed

        # Use base seed directly (subclasses like Compose can override this)
        self.random_generator = np.random.default_rng(seed)
        self.py_random = random.Random(seed)

        # Propagate seed to all transforms
        for transform in self.transforms:
            if isinstance(transform, (BasicTransform, BaseCompose)):
                transform.set_random_seed(seed)

    def set_mask_interpolation(self, mask_interpolation: int | None) -> None:
        """Set interpolation mode for mask resizing operations. Propagates recursively to all
        transforms; overrides mask_interpolation on each. Use OpenCV flags.

        Args:
            mask_interpolation (int | None): OpenCV interpolation flag to use for mask transforms.
                If None, default interpolation for masks will be used.

        """
        self.mask_interpolation = mask_interpolation
        self._set_mask_interpolation_recursive(self.transforms)

    def _set_mask_interpolation_recursive(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BasicTransform):
                if hasattr(transform, "mask_interpolation") and self.mask_interpolation is not None:
                    transform.mask_interpolation = self.mask_interpolation
            elif isinstance(transform, BaseCompose):
                transform.set_mask_interpolation(self.mask_interpolation)

    def __iter__(self) -> Iterator[TransformType]:
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, *args: Any, **data: Any) -> dict[str, Any]:
        """Apply transforms. Abstract; subclasses (Compose, OneOf, etc.) implement the actual
        application logic. Accepts named data (image, mask, bboxes, etc.).

        Args:
            *args (Any): Positional arguments are not supported.
            **data (Any): Named parameters with data to transform.

        Returns:
            dict[str, Any]: Transformed data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError

    def __getitem__(self, item: int) -> TransformType:
        return self.transforms[item]

    def __repr__(self) -> str:
        return self.indented_repr()

    @property
    def additional_targets(self) -> dict[str, str]:
        """Get additional targets dictionary. Maps custom target names to built-in types
        (e.g. {'image2': 'image'}). Used when adding targets via add_targets.

        Returns:
            dict[str, str]: Dictionary containing additional targets mapping.

        """
        return self._additional_targets

    @property
    def available_keys(self) -> set[str]:
        """Get set of available keys. Union of all transform keys plus additional_targets and
        processor data_fields. Used to validate input data keys.

        Returns:
            set[str]: Set of string keys available for transforms.

        """
        return self._available_keys

    def indented_repr(self, indent: int = REPR_INDENT_STEP) -> str:
        """Get an indented string representation of the composition. Includes
        to_dict_private args; each transform shown with indent. For __repr__.

        Args:
            indent (int): Indentation level. Default: REPR_INDENT_STEP.

        Returns:
            str: Formatted string representation with proper indentation.

        """
        args = {k: v for k, v in self.to_dict_private().items() if not (k.startswith("__") or k == "transforms")}
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            t_repr = t.indented_repr(indent + REPR_INDENT_STEP) if hasattr(t, "indented_repr") else repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += "\n" + " " * (indent - REPR_INDENT_STEP) + f"], {format_args(args)})"
        return repr_string

    @classmethod
    def get_class_fullname(cls) -> str:
        """Get the full qualified name of the class. Returns shortest fullname for
        serialization (e.g. albumentations.Compose). For to_dict and replay.

        Returns:
            str: The shortest class fullname.

        """
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the class is serializable. True for all compose classes; for
        serialization to skip non-serializable types. Always True here.

        Returns:
            bool: True if the class is serializable, False otherwise.

        """
        return True

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the composition to a dictionary for serialization. Contains
        __class_fullname__, p, and list of transform dicts. For save/replay.

        Returns:
            dict[str, Any]: Dictionary representation of the composition.

        """
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t.to_dict_private() for t in self.transforms],
        }

    def get_dict_with_id(self) -> dict[str, Any]:
        """Get a dictionary representation with object IDs for replay mode. Includes
        id(self) and per-transform get_dict_with_id. For ReplayCompose.

        Returns:
            dict[str, Any]: Dictionary with composition data and object IDs.

        """
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(self, additional_targets: dict[str, str] | None) -> None:
        """Add additional targets to all transforms. Updates _additional_targets and
        propagates to every child transform and processor. Call _set_keys after.

        Args:
            additional_targets (dict[str, str] | None): Dict of name -> type mapping for additional targets.
                If None, no additional targets will be added.

        """
        if additional_targets:
            for k, v in additional_targets.items():
                if k in self._additional_targets and v != self._additional_targets[k]:
                    raise ValueError(
                        f"Trying to overwrite existed additional targets. "
                        f"Key={k} Exists={self._additional_targets[k]} New value: {v}",
                    )
            self._additional_targets.update(additional_targets)
            for t in self.transforms:
                t.add_targets(additional_targets)
            for proc in self.processors.values():
                proc.add_targets(additional_targets)
        self._set_keys()

    def _set_keys(self) -> None:
        """Set _available_keys from additional_targets and child transforms and targets_as_params.
        Updates processor data_fields; warns if processor has no transform.
        """
        self._available_keys.update(self._additional_targets.keys())
        for t in self.transforms:
            self._available_keys.update(t.available_keys)
            if hasattr(t, "targets_as_params"):
                self._available_keys.update(t.targets_as_params)
        if self.processors:
            self._available_keys.update(["labels"])
            for proc in self.processors.values():
                if proc.default_data_name not in self._available_keys:  # if no transform to process this data
                    warnings.warn(
                        f"Got processor for {proc.default_data_name}, but no transform to process it.",
                        stacklevel=2,
                    )
                self._available_keys.update(proc.data_fields)
                if proc.params.label_fields:
                    self._available_keys.update(proc.params.label_fields)

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> None:
        """Set deterministic mode for all transforms. Propagates to every child; when True,
        params are saved under save_key for replay (e.g. TTA).

        Args:
            flag (bool): Whether to enable deterministic mode.
            save_key (str): Key to save replay parameters. Default: "replay".

        """
        for t in self.transforms:
            t.set_deterministic(flag, save_key)

    def check_data_post_transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """Check and filter data after transformation. Runs each check_each_transform
        processor (e.g. bbox filter) on matching data keys. Returns filtered data dict.

        Args:
            data (dict[str, Any]): Dictionary containing transformed data

        Returns:
            dict[str, Any]: Filtered data dictionary

        """
        if self.check_each_transform:
            shape = get_shape(data, self._additional_targets)

            for proc in self.check_each_transform:
                for data_name, data_value in data.items():
                    if data_name in proc.data_fields or (
                        data_name in self._additional_targets
                        and self._additional_targets[data_name] in proc.data_fields
                    ):
                        data[data_name] = proc.filter(data_value, shape)
        return data

    def _validate_transforms(self, transforms: list[Any]) -> None:
        """Validate that all elements are BasicTransform instances. Raises TypeError if any
        element is not. Used before __add__/__radd__ and in __init__.

        Args:
            transforms (list[Any]): List of objects to validate

        Raises:
            TypeError: If any element is not a BasicTransform instance

        """
        for t in transforms:
            if not isinstance(t, BasicTransform):
                raise TypeError(
                    f"All elements must be instances of BasicTransform, got {type(t).__name__}",
                )

    def _combine_transforms(self, other: TransformType | TransformsSeqType, *, prepend: bool = False) -> "BaseCompose":
        """Combine transforms with the current compose. Prepends or appends other; returns new
        instance via _create_new_instance. Validates with _validate_transforms.

        Args:
            other (TransformType | TransformsSeqType): Transform or sequence of transforms to combine
            prepend (bool): If True, prepend other to the beginning; if False, append to the end

        Returns:
            BaseCompose: New compose instance with combined transforms

        Raises:
            TypeError: If other is not a valid transform or sequence of transforms

        """
        if isinstance(other, (list, tuple)):
            self._validate_transforms(other)
            other_list = list(other)
        else:
            self._validate_transforms([other])
            other_list = [other]

        new_transforms = [*other_list, *list(self.transforms)] if prepend else [*list(self.transforms), *other_list]

        return self._create_new_instance(new_transforms)

    def __add__(self, other: TransformType | TransformsSeqType) -> "BaseCompose":
        """Add transform(s) to the end of this compose. Returns new instance. Use +
        (e.g. compose + A.HorizontalFlip() or compose + [A.Blur(), A.Rotate()]).

        Args:
            other (TransformType | TransformsSeqType): Transform or sequence of transforms to append

        Returns:
            BaseCompose: New compose instance with transforms appended

        Raises:
            TypeError: If other is not a valid transform or sequence of transforms

        Examples:
            >>> new_compose = compose + A.HorizontalFlip()
            >>> new_compose = compose + [A.HorizontalFlip(), A.VerticalFlip()]

        """
        return self._combine_transforms(other, prepend=False)

    def __radd__(self, other: TransformType | TransformsSeqType) -> "BaseCompose":
        """Add transform(s) to the beginning of this compose. Returns new instance. Use +
        with transform on left (e.g. A.RandomCrop(256,256) + compose).

        Args:
            other (TransformType | TransformsSeqType): Transform or sequence of transforms to prepend

        Returns:
            BaseCompose: New compose instance with transforms prepended

        Raises:
            TypeError: If other is not a valid transform or sequence of transforms

        Examples:
            >>> new_compose = A.HorizontalFlip() + compose
            >>> new_compose = [A.HorizontalFlip(), A.VerticalFlip()] + compose

        """
        return self._combine_transforms(other, prepend=True)

    def __sub__(self, other: type[BasicTransform]) -> "BaseCompose | type[NotImplemented]":
        """Remove transform by class type. Removes first matching; returns new instance.
        Use - (e.g. compose - A.HorizontalFlip). Returns NotImplemented for other types.

        Removes the first transform in the compose that matches the provided transform class.

        Args:
            other (type[BasicTransform]): Transform class to remove (e.g., A.HorizontalFlip)

        Returns:
            BaseCompose | type[NotImplemented]: New compose instance with transform removed, or NotImplemented.

        Raises:
            ValueError: If no transform of that type is found in the compose

        Note:
            If multiple transforms of the same type exist in the compose,
            only the first occurrence will be removed.

        Examples:
            >>> # Remove by transform class
            >>> new_compose = compose - A.HorizontalFlip
            >>>
            >>> # With duplicates - only first occurrence removed
            >>> compose = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(), A.HorizontalFlip(p=1.0)])
            >>> result = compose - A.HorizontalFlip  # Removes first HorizontalFlip (p=0.5)
            >>> len(result.transforms)  # 2 (VerticalFlip and second HorizontalFlip remain)

        """
        # Return NotImplemented for unsupported operand types (Python data model convention)
        if not (isinstance(other, type) and issubclass(other, BasicTransform)):
            return NotImplemented

        # Find first transform of matching class
        new_transforms = list(self.transforms)
        for i, transform in enumerate(new_transforms):
            if type(transform) is other:
                new_transforms.pop(i)
                return self._create_new_instance(new_transforms)

        # No matching transform found
        class_name = other.__name__
        raise ValueError(f"No transform of type {class_name} found in the compose pipeline")

    def _create_new_instance(self, new_transforms: TransformsSeqType) -> "BaseCompose":
        """Create new instance of same class with new transforms. Copies init params
        and random state from self. Called by __add__, __radd__, __sub__.

        Args:
            new_transforms (TransformsSeqType): List of transforms for the new instance

        Returns:
            BaseCompose: New instance of the same class

        """
        # Get current instance parameters
        init_params = self._get_init_params()
        init_params["transforms"] = new_transforms

        # Create new instance
        new_instance = self.__class__(**init_params)

        # Copy random state from original instance to new instance
        if hasattr(self, "random_generator") and hasattr(self, "py_random"):
            new_instance.set_random_state(self.random_generator, self.py_random)

        return new_instance

    def _get_init_params(self) -> dict[str, Any]:
        """Get parameters needed to recreate this instance. Subclasses add their params.
        For _create_new_instance and serialization; no defaults in InitSchema.

        Note:
            Subclasses that add new initialization parameters (other than 'transforms',
            which is set separately in _create_new_instance) should override this method
            to include those parameters in the returned dictionary.

        Returns:
            dict[str, Any]: Dictionary of initialization parameters

        """
        return {
            "p": self.p,
        }

    def _get_effective_seed(self, base_seed: int | None) -> int | None:
        """Get effective seed considering worker context. In PyTorch DataLoader workers,
        combines base_seed with torch.initial_seed() for per-worker reproducibility.

        Args:
            base_seed (int | None): Base seed value

        Returns:
            int | None: Effective seed after considering worker context

        """
        if base_seed is None:
            return base_seed

        try:
            import torch
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # We're in a DataLoader worker process
                # Use torch.initial_seed() which is unique per worker and changes on respawn
                torch_seed = torch.initial_seed() % (2**32)
                return (base_seed + torch_seed) % (2**32)
        except (ImportError, AttributeError):
            # PyTorch not available or not in worker context
            pass

        return base_seed


class Compose(BaseCompose, HubMixin):
    """Compose multiple transforms sequentially. Supports bbox_params, keypoint_params,
    additional_targets, strict, seed; supports +, -, __radd__.

    This class allows you to chain multiple image augmentation transforms and apply them
    in a specified order. It also handles bounding box and keypoint transformations if
    the appropriate parameters are provided.

    The Compose class supports dynamic pipeline modification after initialization using
    mathematical operators. All parameters (bbox_params, keypoint_params, additional_targets,
    etc.) are preserved when using operators to modify the pipeline.

    Args:
        transforms (list[BasicTransform | BaseCompose]): A list of transforms to apply.
        bbox_params (dict[str, Any] | BboxParams | None): Parameters for bounding box transforms.
            Can be a dict of params or a BboxParams object. Default is None.
        keypoint_params (dict[str, Any] | KeypointParams | None): Parameters for keypoint transforms.
            Can be a dict of params or a KeypointParams object. Default is None.
        additional_targets (dict[str, str] | None): A dictionary mapping additional target names
            to their types. For example, {'image2': 'image'}. Default is None.
        p (float): Probability of applying all transforms. Should be in range [0, 1]. Default is 1.0.
        is_check_shapes (bool): If True, checks consistency of shapes for image/mask/masks on each call.
            Disable only if you are sure about your data consistency. Default is True.
        strict (bool): If True, enables strict mode which:
            1. Validates that all input keys are known/expected
            2. Validates that no transforms have invalid arguments
            3. Raises ValueError if any validation fails
            If False, these validations are skipped. Default is False.
        mask_interpolation (int | None): Interpolation method for mask transforms. When defined,
            it overrides the interpolation method specified in individual transforms. Default is None.
        seed (int | None): Controls reproducibility of random augmentations. Compose uses
            its own internal random state, completely independent from global random seeds.

            When seed is set (int):
            - Creates a fixed internal random state
            - Two Compose instances with the same seed and transforms will produce identical
              sequences of augmentations
            - Each call to the same Compose instance still produces random augmentations,
              but these sequences are reproducible between different Compose instances
            - Example: transform1 = A.Compose([...], seed=137) and
                      transform2 = A.Compose([...], seed=137) will produce identical sequences

            When seed is None (default):
            - Generates a new internal random state on each Compose creation
            - Different Compose instances will produce different sequences of augmentations
            - Example: transform = A.Compose([...])  # random results

            Important: Setting random seeds outside of Compose (like np.random.seed() or
            random.seed()) has no effect on augmentations as Compose uses its own internal
            random state.
        save_applied_params (bool): If True, saves the applied parameters of each transform. Default is False.
            You will need to use the `applied_transforms` key in the output dictionary to access the parameters.
        telemetry (bool): If True, enables telemetry collection to help improve AlbumentationsX.
            This collects anonymous usage data including pipeline configuration, environment info,
            and common parameter patterns. No image data or personal information is collected.
            Telemetry can be disabled globally via settings.telemetry_enabled = False or by
            setting the environment variable ALBUMENTATIONS_NO_TELEMETRY=1. Default is True.

    Examples:
        >>> # Basic usage:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.RandomCrop(width=256, height=256),
        ...     A.HorizontalFlip(p=0.5),
        ...     A.RandomBrightnessContrast(p=0.2),
        ... ], seed=137)
        >>> transformed = transform(image=image)

        >>> # Pipeline modification after initialization:
        >>> # Create initial pipeline with bbox support
        >>> base_transform = A.Compose([
        ...     A.HorizontalFlip(p=0.5),
        ...     A.RandomCrop(width=512, height=512)
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['labels']))
        >>>
        >>> # Add transforms using operators (bbox_params preserved)
        >>> extended = base_transform + A.RandomBrightnessContrast(p=0.3)
        >>> extended = base_transform + [A.Blur(), A.GaussNoise()]
        >>> extended = A.Resize(height=1024, width=1024) + base_transform
        >>>
        >>> # Remove transforms by class
        >>> pipeline = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(), A.Rotate()])
        >>> without_flip = pipeline - A.HorizontalFlip  # Remove by class

    Note:
        - The class checks the validity of input data and shapes if is_check_args and is_check_shapes are True.
        - When bbox_params or keypoint_params are provided, it sets up the corresponding processors.
        - The transform can handle additional targets specified in the additional_targets dictionary.
        - When strict mode is enabled, it performs additional validation to ensure data and transform
          configuration correctness.
        - Pipeline modification operators (+, -, __radd__) preserve all Compose parameters including
          bbox_params, keypoint_params, additional_targets, and other configuration settings.
        - All operators return new Compose instances without modifying the original pipeline.

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        strict: bool = False,
        mask_interpolation: int | None = None,
        seed: int | None = None,
        save_applied_params: bool = False,
        telemetry: bool = True,
        instance_binding: Sequence[str] | None = None,
    ):
        self._base_seed = seed
        super().__init__(
            transforms=transforms,
            p=p,
            mask_interpolation=mask_interpolation,
            seed=self._get_effective_seed(seed),
            save_applied_params=save_applied_params,
        )

        self.telemetry = telemetry
        self._resolve_processors(bbox_params, keypoint_params)

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self._instance_binding = self._setup_instance_binding(instance_binding)

        self.add_targets(additional_targets)
        if not self.transforms:  # if no transforms -> do nothing, all keys will be available
            self._available_keys.update(AVAILABLE_KEYS)
        if self._instance_binding:
            self._available_keys.add("instances")

        self.is_check_args = True
        self.strict = strict
        self.is_check_shapes = is_check_shapes
        self.check_each_transform = tuple(  # processors that check after each transform
            proc for proc in self.processors.values() if getattr(proc.params, "check_each_transform", False)
        )
        self._set_check_args_for_transforms(self.transforms)
        self._set_processors_for_transforms(self.transforms)

        self.save_applied_params = save_applied_params
        self._images_was_list = False
        self._masks_was_list = False
        self._last_torch_seed: int | None = None

        # Telemetry runs after nested composes so main_compose=False is already set on them.
        self._maybe_send_telemetry(telemetry)

    def _resolve_processors(
        self,
        bbox_params: dict[str, Any] | BboxParams | None,
        keypoint_params: dict[str, Any] | KeypointParams | None,
    ) -> None:
        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                msg = "unknown format of bbox_params, please use `dict` or `BboxParams`"
                raise ValueError(msg)
            self.processors["bboxes"] = BboxProcessor(b_params)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                k_params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                k_params = keypoint_params
            else:
                msg = "unknown format of keypoint_params, please use `dict` or `KeypointParams`"
                raise ValueError(msg)
            self.processors["keypoints"] = KeypointsProcessor(k_params)

    def _maybe_send_telemetry(self, telemetry: bool) -> None:
        if not (self.main_compose and settings.telemetry_enabled):
            return
        with contextlib.suppress(Exception):
            client = get_telemetry_client()
            telemetry_data = {**get_environment_info(), **collect_pipeline_info(self)}
            client.track_compose_init(telemetry_data, telemetry=telemetry)

    @property
    def strict(self) -> bool:
        """Get the current strict mode setting. When True, validates input keys and transform
        arguments; raises ValueError on invalid args. Read-only.

        Returns:
            bool: True if strict mode is enabled, False otherwise.

        """
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        # if value and not self._strict:
        if value:
            # Only validate when enabling strict mode
            self._validate_strict()
        self._strict = value

    def _validate_strict(self) -> None:
        """Validate no transforms have invalid arguments when strict is enabled. Recursively
        checks invalid_args; raises ValueError if any non-empty.
        """

        def check_transform(transform: TransformType) -> None:
            if hasattr(transform, "invalid_args") and transform.invalid_args:
                message = (
                    f"Argument(s) '{', '.join(transform.invalid_args)}' "
                    f"are not valid for transform {transform.__class__.__name__}"
                )
                raise ValueError(message)
            if isinstance(transform, BaseCompose):
                for t in transform.transforms:
                    check_transform(t)

        for transform in self.transforms:
            check_transform(transform)

    def _setup_instance_binding(self, instance_binding: Sequence[str] | None) -> frozenset[str] | None:
        self._bbox_label_map: dict[str, str] = {}
        self._kp_label_map: dict[str, str] = {}
        if instance_binding is None:
            return None
        targets = frozenset(instance_binding)
        self._validate_instance_binding_targets(targets)
        self._apply_bbox_instance_binding(targets)
        self._apply_keypoints_instance_binding(targets)
        return targets

    def _validate_instance_binding_targets(self, targets: frozenset[str]) -> None:
        if len(targets) < 2:
            raise ValueError("instance_binding must contain at least 2 targets")
        invalid = targets - _VALID_INSTANCE_BINDING_TARGETS
        if invalid:
            raise ValueError(
                f"Invalid instance_binding targets: {invalid}. "
                f"Valid targets: {sorted(_VALID_INSTANCE_BINDING_TARGETS)}",
            )
        if "mask" in targets and "masks" in targets:
            raise ValueError("instance_binding cannot contain both 'mask' and 'masks'")
        if "bboxes" in targets and "bboxes" not in self.processors:
            raise ValueError("bbox_params must be set when 'bboxes' is in instance_binding")
        if "keypoints" in targets and "keypoints" not in self.processors:
            raise ValueError("keypoint_params must be set when 'keypoints' is in instance_binding")

    def _apply_bbox_instance_binding(self, targets: frozenset[str]) -> None:
        if "bboxes" not in targets:
            return
        bbox_proc = self.processors["bboxes"]
        if not isinstance(bbox_proc, BboxProcessor):
            msg = "expected bbox processor"
            raise TypeError(msg)
        bbox_proc.params = copy.deepcopy(bbox_proc.params)
        bbox_params = bbox_proc.params
        user_fields = list(bbox_params.label_fields or [])
        internal_fields = [f"_ibl_bbox_{f}" for f in user_fields]
        self._bbox_label_map = dict(zip(internal_fields, user_fields, strict=True))
        internal_fields.append(_BBOX_INSTANCE_ID)
        bbox_params.label_fields = internal_fields

    def _apply_keypoints_instance_binding(self, targets: frozenset[str]) -> None:
        if "keypoints" not in targets:
            return
        kp_proc = self.processors["keypoints"]
        if not isinstance(kp_proc, KeypointsProcessor):
            msg = "expected keypoints processor"
            raise TypeError(msg)
        kp_proc.params = copy.deepcopy(kp_proc.params)
        kp_params = kp_proc.params
        user_fields = list(kp_params.label_fields or [])
        internal_fields = [f"_ibl_kp_{f}" for f in user_fields]
        self._kp_label_map = dict(zip(internal_fields, user_fields, strict=True))
        internal_fields.append(_KP_INSTANCE_ID)
        kp_params.label_fields = internal_fields
        kp_params.remove_invisible = False
        kp_params.check_each_transform = False

    def _set_processors_for_transforms(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BasicTransform):
                if hasattr(transform, "set_processors"):
                    transform.set_processors(self.processors)
            elif isinstance(transform, BaseCompose):
                self._set_processors_for_transforms(transform.transforms)

    def _set_check_args_for_transforms(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                self._set_check_args_for_transforms(transform.transforms)
                transform.check_each_transform = self.check_each_transform
                transform.processors = self.processors
            if isinstance(transform, Compose):
                transform.disable_check_args_private()

    def disable_check_args_private(self) -> None:
        """Disable argument checking. Sets is_check_args=False, strict=False, main_compose=False.
        Called for nested Compose so only top-level validates.
        """
        self.is_check_args = False
        self.strict = False
        self.main_compose = False

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply transformations with worker seed sync. Runs preprocess, each transform in
        order, check_data_post_transform, postprocess.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        Raises:
            KeyError: If positional arguments are provided.

        """
        # Check and sync worker seed if needed
        self._check_worker_seed()

        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)

        # Initialize applied_transforms only in top-level Compose if requested
        if self.save_applied_params and self.main_compose:
            data["applied_transforms"] = []

        need_to_run = force_apply or self.py_random.random() < self.p
        if not need_to_run:
            return data

        try:
            self.preprocess(data)
            resync = self._resync_masks_to_bboxes if self.main_compose and self._instance_binding else None
            for t in self.transforms:
                data = t(**data)
                self._track_transform_params(t, data)
                data = self.check_data_post_transform(data)
                if resync is not None:
                    resync(data)

            return self.postprocess(data)
        finally:
            # Clear per-call unpack/repack flags if preprocess or a transform raised mid-call.
            if self.main_compose and self._instance_binding:
                self._clear_instance_binding_call_state_if_pending()

    def _clear_instance_binding_call_state_if_pending(self) -> None:
        if getattr(self, "_repack_after_processors", False):
            del self._repack_after_processors
        if hasattr(self, "_instance_count"):
            delattr(self, "_instance_count")

    @staticmethod
    def from_applied_transforms(
        applied_transforms: list[tuple[str, dict[str, Any]]],
    ) -> "Compose":
        """Reconstruct a Compose pipeline from the applied_transforms list
        captured in a previous run; each entry is instantiated with p=1.0 for replay.

        Each (class_fullname, applied_config) pair is instantiated with p=1.0. Range params
        resolved to scalars during the original run are wrapped as (v, v) degenerate tuples so
        the constructor's InitSchema validator accepts them without symmetric expansion.
        This fixes constructor-level randomness only — transforms with internal randomness
        (random crop positions, dropout masks, etc.) may still vary between runs.

        Args:
            applied_transforms (list[tuple[str, dict[str, Any]]]): List of (class_fullname, applied_config)
                tuples as produced by Compose when save_applied_params=True.

        Returns:
            Compose: A pipeline with p=1.0 for all transforms and constructor params
                fixed to the values sampled in the original run.

        """
        register_additional_transforms()
        transforms = []
        for class_name, config in applied_transforms:
            cls = SERIALIZABLE_REGISTRY[class_name]

            replay_config = _wrap_scalars_for_replay(cls, config)
            replay_config["p"] = 1.0

            transforms.append(cls(**replay_config))

        return Compose(transforms, p=1.0)

    def _check_worker_seed(self) -> None:
        """Check and update random seed in worker context. Recalculates effective seed and
        propagates to all transforms for reproducibility.
        """
        if not hasattr(self, "_base_seed") or self._base_seed is None:
            return

        # Check if we're in a worker and need to update the seed
        try:
            import torch
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # Get the current torch initial seed
                current_torch_seed = torch.initial_seed()

                # Check if we've already synchronized for this seed
                if hasattr(self, "_last_torch_seed") and self._last_torch_seed == current_torch_seed:
                    return

                # Update the seed and mark as synchronized
                self._last_torch_seed = current_torch_seed
                effective_seed = self._get_effective_seed(self._base_seed)

                # Update our own random state
                self.random_generator = np.random.default_rng(effective_seed)
                self.py_random = random.Random(effective_seed)

                # Propagate to all transforms
                for transform in self.transforms:
                    if hasattr(transform, "set_random_state"):
                        transform.set_random_state(self.random_generator, self.py_random)
                    elif hasattr(transform, "set_random_seed"):
                        # For transforms that don't have set_random_state, use set_random_seed
                        transform.set_random_seed(effective_seed)
        except (ImportError, AttributeError):
            pass

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state from unpickling and handle worker seed. Resets _last_torch_seed and
        recalculates effective seed so worker sync runs again after unpickling.
        """
        self.__dict__.update(state)
        # If we have a base seed, recalculate effective seed in worker context
        if hasattr(self, "_base_seed") and self._base_seed is not None:
            # Reset _last_torch_seed to ensure worker-seed sync runs after unpickling
            self._last_torch_seed = None
            # Recalculate effective seed in worker context
            self.set_random_seed(self._base_seed)
        elif hasattr(self, "seed") and self.seed is not None:
            # For backward compatibility, if no base seed but seed exists
            self._base_seed = self.seed
            self._last_torch_seed = None
            self.set_random_seed(self.seed)

    def set_random_seed(self, seed: int | None) -> None:
        """Override for worker-aware seed. Stores _base_seed, computes effective seed via
        _get_effective_seed and propagates to all transforms.

        Args:
            seed (int | None): Random seed to use

        """
        # Store the original base seed
        self._base_seed = seed
        self.seed = seed

        # Get effective seed considering worker context
        effective_seed = self._get_effective_seed(seed)

        # Initialize random generators with effective seed
        self.random_generator = np.random.default_rng(effective_seed)
        self.py_random = random.Random(effective_seed)

        # Propagate to all transforms
        for transform in self.transforms:
            if hasattr(transform, "set_random_state"):
                transform.set_random_state(self.random_generator, self.py_random)
            elif hasattr(transform, "set_random_seed"):
                # For transforms that don't have set_random_state, use set_random_seed
                transform.set_random_seed(effective_seed)

    def preprocess(self, data: Any) -> None:
        """Preprocess input data before applying transforms. Validates shapes (if
        is_check_shapes), validates data keys (if strict), ensures contiguous, adds channels.
        """
        if self._instance_binding and "instances" in data and self.main_compose:
            self._unpack_instances(data)
        elif self._instance_binding and self.main_compose and isinstance(data, dict):
            self._require_instance_binding_data_present(data)

        # Always validate shapes if is_check_shapes is True, regardless of strict mode
        if self.is_check_shapes:
            shapes, volume_shapes = self._gather_shapes_from_data(data)
            self._check_shape_consistency(shapes, volume_shapes)

        # Do strict validation only if enabled
        if self.strict:
            self._validate_data(data)

        # Add channel dimensions first, before processors run
        self._preprocess_arrays(data)
        self._preprocess_processors(data)

    def _gather_shapes_from_data(self, data: dict[str, Any]) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Gather shapes from data for validation. Collects (H,W) or (D,H,W) from
        image, mask, images, volume. For preprocess shape check.

        Args:
            data (dict[str, Any]): Data dictionary containing various arrays

        Returns:
            tuple[list[tuple[int, ...]], list[tuple[int, ...]]]: Tuple of (2D shapes list, 3D shapes list).

        """
        shapes: list[tuple[int, ...]] = []  # For H,W checks
        volume_shapes: list[tuple[int, ...]] = []  # For D,H,W checks

        # List of targets to check shapes for
        shape_check_targets = {"image", "mask", "images", "volume", "volumes", "mask3d", "masks", "masks3d"}

        for data_name, data_value in data.items():
            # Resolve aliases via additional_targets so e.g. {'custom_image_key': 'image'}
            # gets the same shape-consistency check as the canonical 'image' key.
            canonical = self._additional_targets.get(data_name, data_name)
            if canonical not in shape_check_targets:
                continue

            # Skip empty data
            if data_value is None or not isinstance(data_value, np.ndarray):
                continue

            # Skip arrays with size 0 (empty arrays)
            if data_value.size == 0:
                continue

            self._process_data_shape(canonical, data_value, shapes, volume_shapes)

        return shapes, volume_shapes

    def _process_data_shape(
        self,
        data_name: str,
        data_value: np.ndarray,
        shapes: list[tuple[int, ...]],
        volume_shapes: list[tuple[int, ...]],
    ) -> None:
        """Process shape of a single data item. Appends (H,W) or (D,H,W) to shapes or
        volume_shapes depending on data_name (image, mask, images, volume, etc.).
        """
        # Handle 2D single data
        if data_name in {"image", "mask"}:
            shapes.append(data_value.shape[:2])  # H,W

        # Handle 2D batch data
        elif data_name in {"images", "masks"}:
            if data_value.ndim not in {3, 4}:  # (N,H,W) or (N,H,W,C)
                raise TypeError(f"{data_name} must be 3D or 4D array")
            shapes.append(data_value.shape[1:3])  # H,W from (N,H,W)

        # Handle 3D single data
        elif data_name in {"volume", "mask3d"}:
            if data_value.ndim not in {3, 4}:  # (D,H,W) or (D,H,W,C)
                raise TypeError(f"{data_name} must be 3D or 4D array")
            shapes.append(data_value.shape[1:3])  # H,W
            volume_shapes.append(data_value.shape[:3])  # D,H,W

        # Handle 3D batch data
        elif data_name in {"volumes", "masks3d"}:
            if data_value.ndim not in {4, 5}:  # (N,D,H,W) or (N,D,H,W,C)
                raise TypeError(f"{data_name} must be 4D or 5D array")
            shapes.append(data_value.shape[2:4])  # H,W from (N,D,H,W)
            volume_shapes.append(data_value.shape[1:4])  # D,H,W from (N,D,H,W)

    def _validate_data(self, data: dict[str, Any]) -> None:
        """Validate input data keys and arguments. When strict, checks every key is in
        _available_keys and runs _check_args. Raises ValueError on invalid key.
        """
        if not self.strict:
            return

        for data_name in data:
            if not self._is_valid_key(data_name):
                raise ValueError(f"Key {data_name} is not in available keys.")

        if self.is_check_args:
            self._check_args(**data)

    def _is_valid_key(self, key: str) -> bool:
        """Check if the key is valid for processing. True if key is in _available_keys,
        MASK_KEYS, IMAGE_KEYS, or 'applied_transforms'.
        """
        return key in self._available_keys or key in MASK_KEYS or key in IMAGE_KEYS or key == "applied_transforms"

    def _preprocess_processors(self, data: dict[str, Any]) -> None:
        """Run preprocessors if this is the main compose. Calls ensure_data_valid and
        preprocess on each processor (bboxes, keypoints). No-op when main_compose is False.
        """
        if not self.main_compose:
            return

        for processor in self.processors.values():
            processor.ensure_data_valid(data)
        for processor in self.processors.values():
            processor.preprocess(data)

    def _preprocess_arrays(self, data: dict[str, Any]) -> None:
        """Ensure arrays are contiguous and add channel dims to grayscale data.
        Calls _ensure_contiguous then _add_grayscale_channels. Call during preprocess.
        """
        self._ensure_contiguous(data)
        self._add_grayscale_channels(data)

    def _ensure_contiguous(self, data: dict[str, Any]) -> None:
        """Ensure all numpy arrays are contiguous. Replaces non-C-contiguous arrays in data
        with np.ascontiguousarray copies. Called by _preprocess_arrays.
        """
        for key, value in data.items():
            if isinstance(value, np.ndarray) and not value.flags["C_CONTIGUOUS"]:
                data[key] = np.ascontiguousarray(value)

    # Maps canonical grayscale-bearing target -> expected ndim *without* the channel dim.
    _GRAYSCALE_KEYS: ClassVar[dict[str, int]] = {
        "image": 2,  # (H, W) => (H, W, 1)
        "images": 3,  # (N, H, W) => (N, H, W, 1)
        "mask": 2,  # (H, W) => (H, W, 1)
        "masks": 3,  # (N, H, W) => (N, H, W, 1)
        "volume": 3,  # (D, H, W) => (D, H, W, 1)
        "volumes": 4,  # (N, D, H, W) => (N, D, H, W, 1)
        "mask3d": 3,  # (D, H, W) => (D, H, W, 1)
        "masks3d": 4,  # (N, D, H, W) => (N, D, H, W, 1)
    }

    def _add_grayscale_channels(self, data: dict[str, Any]) -> None:
        """Add a trailing channel dimension to grayscale image/mask/volume entries,
        resolving `_additional_targets` so aliased keys are handled like canonical ones.

        Expands `(H, W)` to `(H, W, 1)` (and the equivalent for batches/volumes). Tracks
        expansion in `_added_channel_dim` (keyed by user key) and `_added_channel_canonical`
        (user_key -> canonical name) so postprocess can strip only what we added.
        """
        self._added_channel_dim = {}
        self._added_channel_canonical = {}

        for key, value in data.items():
            canonical = self._additional_targets.get(key, key)
            expected_ndim = self._GRAYSCALE_KEYS.get(canonical)
            if expected_ndim is None:
                continue
            if not isinstance(value, np.ndarray):
                continue
            self._added_channel_canonical[key] = canonical
            if value.ndim == expected_ndim:
                data[key] = np.expand_dims(value, axis=-1)
                self._added_channel_dim[key] = True
            else:
                self._added_channel_dim[key] = False

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply post-processing after all transforms. Runs processor postprocess and
        _remove_grayscale_channels when main_compose. Returns data dict.

        Args:
            data (dict[str, Any]): Data after transformation.

        Returns:
            dict[str, Any]: Post-processed data.

        """
        if self.main_compose:
            for p in self.processors.values():
                p.postprocess(data)

            if self._instance_binding and getattr(self, "_repack_after_processors", False):
                try:
                    self._repack_instances(data)
                finally:
                    del self._repack_after_processors
                    if hasattr(self, "_instance_count"):
                        delattr(self, "_instance_count")

            # Remove channel dimensions that were added during preprocessing
            self._remove_grayscale_channels(data)

        return data

    def _remove_grayscale_channels(self, data: dict[str, Any]) -> None:
        """Strip the trailing channel dimension that `_add_grayscale_channels` added,
        for both numpy arrays and torch tensors, using the bookkeeping from preprocess.

        Uses `_added_channel_dim` to squeeze only where we added a dim, and
        `_added_channel_canonical` to dispatch torch logic by canonical role so aliased
        keys are handled the same as their canonical counterparts.
        """
        if not hasattr(self, "_added_channel_dim"):
            return

        canonical_map = getattr(self, "_added_channel_canonical", {})

        for key, was_added in self._added_channel_dim.items():
            if was_added and key in data:
                value = data[key]
                canonical = canonical_map.get(key, key)

                # Handle numpy arrays
                if isinstance(value, np.ndarray):
                    if value.shape[-1] == 1:
                        data[key] = np.squeeze(value, axis=-1)

                # Handle torch tensors
                elif hasattr(value, "__module__") and "torch" in value.__module__:
                    # Import torch only if we have a torch tensor
                    import torch

                    if isinstance(value, torch.Tensor):
                        # For torch tensors, we need to handle different cases
                        # ToTensorV2 transposes image tensors but not mask tensors
                        if canonical in {"image", "images"} and len(value.shape) >= 3 and value.shape[0] == 1:
                            # Image tensor with shape (1, H, W) -> (H, W) is not typical, skip
                            pass
                        elif canonical in {"mask", "masks", "mask3d", "masks3d"} and value.shape[-1] == 1:
                            # Mask tensor with shape (..., H, W, 1) -> (..., H, W)
                            data[key] = torch.squeeze(value, dim=-1)

    def _get_user_bbox_label_fields(self) -> list[str]:
        return list(self._bbox_label_map.values())

    def _get_user_kp_label_fields(self) -> list[str]:
        return list(self._kp_label_map.values())

    def _require_instance_binding_data_present(self, data: dict[str, Any]) -> None:
        """Ensure instance_binding calls pass `instances` for unpack or already-unpacked data with mask tensors
        and internal instance-id columns for nested preprocess.
        """
        binding = self._instance_binding
        if binding is None:
            return
        if "masks" in binding and "masks" not in data:
            msg = "`instances` must be provided when using instance_binding with `masks`."
            raise ValueError(msg)
        if "mask" in binding and "mask" not in data:
            msg = "`instances` must be provided when using instance_binding with `mask`."
            raise ValueError(msg)
        if "bboxes" in binding and _BBOX_INSTANCE_ID not in data:
            msg = "`instances` must be provided when using instance_binding with `bboxes`."
            raise ValueError(msg)
        if "keypoints" in binding and _KP_INSTANCE_ID not in data:
            msg = "`instances` must be provided when using instance_binding with `keypoints`."
            raise ValueError(msg)

    def _reserved_keys_for_instance_unpack(self, binding: frozenset[str]) -> frozenset[str]:
        """Return keys instance unpack assigns to pipeline data: mask targets, bboxes, keypoints, and
        internal label columns used when repacking instances.
        """
        keys: set[str] = set()
        if "masks" in binding:
            keys.add("masks")
        elif "mask" in binding:
            keys.add("mask")
        if "bboxes" in binding:
            keys.update({_BBOX_INSTANCE_ID, "bboxes"})
        if "keypoints" in binding:
            keys.update({_KP_INSTANCE_ID, "keypoints"})
        keys.update(self._bbox_label_map)
        keys.update(self._kp_label_map)
        return frozenset(keys)

    def _reject_instance_unpack_key_collisions(self, data: dict[str, Any], binding: frozenset[str]) -> None:
        reserved = self._reserved_keys_for_instance_unpack(binding)
        collisions = sorted(reserved & data.keys())
        if not collisions:
            return
        joined = ", ".join(collisions)
        msg = (
            f"Passing `instances` would overwrite existing data keys: {joined}. "
            "Omit those keys from the input when using instance_binding."
        )
        raise ValueError(msg)

    def _resync_masks_to_bboxes(self, data: dict[str, Any]) -> None:
        """Re-establish the row-alignment invariant after each transform so the next transform
        sees `_bbox_instance_id == range(N)` and aligned masks.

        When bboxes are bound, ensures `_bbox_instance_id == range(N)` and (when masks are also
        bound) `len(data["masks"]) == N` going into the next transform.

        Mid-pipeline the instance id lives as the LAST label column of `data["bboxes"]` (and
        `data["keypoints"]`) — `_apply_bbox_instance_binding` appends `_BBOX_INSTANCE_ID` last, so
        the column index is always `-1`. This is the structural chokepoint that makes the
        invariant a property of the pipeline rather than a per-transform concern. The fast path
        (non-mixing transforms that didn't reshuffle bbox order) early-outs at the
        `np.array_equal` check in microseconds.
        """
        binding = self._instance_binding
        if binding is None or "bboxes" not in binding:
            return
        bboxes_arr = data.get("bboxes")
        if not isinstance(bboxes_arr, np.ndarray) or bboxes_arr.shape[0] == 0:
            return
        bbox_ids_col = bboxes_arr[:, -1].astype(np.int64)
        n = bbox_ids_col.shape[0]
        new_ids = np.arange(n, dtype=np.int64)
        if np.array_equal(bbox_ids_col, new_ids):
            return
        if "keypoints" in binding:
            kp_arr = data.get("keypoints")
            if isinstance(kp_arr, np.ndarray) and kp_arr.shape[0] > 0:
                old_to_new = {int(old): new for new, old in enumerate(bbox_ids_col.tolist())}
                kp_ids_col = kp_arr[:, -1].astype(np.int64)
                kp_arr[:, -1] = np.array(
                    [old_to_new.get(int(k), int(k)) for k in kp_ids_col],
                    dtype=kp_arr.dtype,
                )
        if "masks" in binding:
            masks = data.get("masks")
            # When `len(masks) == n` the mixing transform produced row-aligned masks (Mosaic,
            # CopyAndPaste post-concat), so just rebase ids. When lengths differ, the surviving
            # ids point into the previous masks tensor — fancy-index to compact + reorder rows.
            if isinstance(masks, np.ndarray) and len(masks) != n:
                data["masks"] = masks[bbox_ids_col]
        bboxes_arr[:, -1] = new_ids.astype(bboxes_arr.dtype)

    def _unpack_instances(self, data: dict[str, Any]) -> None:
        binding = self._instance_binding
        if binding is None:
            msg = "_unpack_instances requires instance_binding"
            raise RuntimeError(msg)

        instances = data.pop("instances")
        if not isinstance(instances, (list, tuple)):
            raise TypeError("instances must be a list of dicts")

        self._reject_instance_unpack_key_collisions(data, binding)

        num_instances = len(instances)
        self._instance_count = num_instances

        if num_instances == 0:
            self._init_empty_instance_data(data, binding)
            self._repack_after_processors = True
            return

        instance_dicts = self._validate_instances(instances)
        self._unpack_masks(data, binding, instance_dicts)
        self._unpack_bboxes(data, binding, instance_dicts, num_instances)
        self._unpack_keypoints(data, binding, instance_dicts)
        self._unpack_bbox_labels(data, instance_dicts)
        self._unpack_kp_labels(data, instance_dicts)
        self._repack_after_processors = True

    def _init_empty_instance_data(self, data: dict[str, Any], binding: frozenset[str]) -> None:
        if "masks" in binding:
            data["masks"] = _make_stacked_masks([])
        if "bboxes" in binding:
            bbox_proc = self.processors["bboxes"]
            if isinstance(bbox_proc, BboxProcessor):
                data["bboxes"] = bbox_proc.params.make_empty_bboxes_array()
            else:
                data["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            data[_BBOX_INSTANCE_ID] = []
        if "keypoints" in binding:
            kp_proc_init = self.processors["keypoints"]
            if not isinstance(kp_proc_init, KeypointsProcessor):
                msg = "expected keypoints processor"
                raise TypeError(msg)
            data["keypoints"] = kp_proc_init.params.make_empty_keypoints_array()
            data[_KP_INSTANCE_ID] = []
        for internal_name in self._bbox_label_map:
            data[internal_name] = []
        for internal_name in self._kp_label_map:
            data[internal_name] = []

    def _unpack_masks(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        instance_dicts: list[dict[str, Any]],
    ) -> None:
        if "masks" in binding:
            # Stack as (N, H, W); `_add_grayscale_channels` will expand to canonical (N, H, W, 1)
            # in the same preprocess pass and set `_added_channel_dim["masks"] = True` so the
            # repack path strips the trailing singleton on the way back out.
            data["masks"] = np.stack([inst["mask"] for inst in instance_dicts])
        elif "mask" in binding:
            data["mask"] = np.stack([inst["mask"] for inst in instance_dicts], axis=-1)

    def _unpack_bboxes(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        instance_dicts: list[dict[str, Any]],
        num_instances: int,
    ) -> None:
        if "bboxes" not in binding:
            return
        data["bboxes"] = np.array([inst["bbox"] for inst in instance_dicts], dtype=np.float32)
        data[_BBOX_INSTANCE_ID] = list(range(num_instances))

    def _unpack_keypoints(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        instance_dicts: list[dict[str, Any]],
    ) -> None:
        if "keypoints" not in binding:
            return
        kp_proc_unpack = self.processors["keypoints"]
        if not isinstance(kp_proc_unpack, KeypointsProcessor):
            msg = "expected keypoints processor"
            raise TypeError(msg)
        kp_params = kp_proc_unpack.params
        all_kps: list[np.ndarray] = []
        all_ids: list[int] = []
        for idx, inst in enumerate(instance_dicts):
            kps = inst["keypoints"]
            count = kps.shape[0] if isinstance(kps, np.ndarray) else len(kps)
            if count > 0:
                all_kps.append(np.asarray(kps, dtype=np.float32))
                all_ids.extend([idx] * count)
        data["keypoints"] = np.concatenate(all_kps) if all_kps else kp_params.make_empty_keypoints_array()
        data[_KP_INSTANCE_ID] = all_ids

    def _unpack_bbox_labels(self, data: dict[str, Any], instance_dicts: list[dict[str, Any]]) -> None:
        for internal_name, user_name in self._bbox_label_map.items():
            data[internal_name] = [inst.get("bbox_labels", {})[user_name] for inst in instance_dicts]

    def _unpack_kp_labels(self, data: dict[str, Any], instance_dicts: list[dict[str, Any]]) -> None:
        for internal_name, user_name in self._kp_label_map.items():
            flat: list[Any] = []
            for inst in instance_dicts:
                flat.extend(inst.get("keypoint_labels", {}).get(user_name, []))
            data[internal_name] = flat

    def _validate_instances(self, instances: Sequence[Any]) -> list[dict[str, Any]]:
        binding = self._instance_binding
        if binding is None:
            msg = "_validate_instances requires instance_binding"
            raise RuntimeError(msg)

        bbox_label_fields = self._get_user_bbox_label_fields()
        kp_label_fields = self._get_user_kp_label_fields()
        normalized: list[dict[str, Any]] = []

        for idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                raise TypeError(f"instances[{idx}] must be a dict, got {type(inst).__name__}")
            self._validate_instance_mask(inst, idx, binding)
            self._validate_instance_bbox(inst, idx, binding, bbox_label_fields)
            self._validate_instance_keypoints(inst, idx, binding, kp_label_fields)
            normalized.append(inst)

        return normalized

    def _validate_instance_mask(self, inst: dict[str, Any], idx: int, binding: frozenset[str]) -> None:
        has_mask_binding = "masks" in binding or "mask" in binding
        if has_mask_binding and "mask" not in inst:
            raise ValueError(f"instances[{idx}] missing required key 'mask'")

    def _validate_instance_bbox(
        self,
        inst: dict[str, Any],
        idx: int,
        binding: frozenset[str],
        bbox_label_fields: list[str],
    ) -> None:
        if "bboxes" not in binding:
            return
        if "bbox" not in inst:
            raise ValueError(f"instances[{idx}] missing required key 'bbox'")
        if not bbox_label_fields:
            return
        inst_labels = inst.get("bbox_labels")
        if inst_labels is None:
            raise ValueError(f"instances[{idx}] missing 'bbox_labels'")
        missing = set(bbox_label_fields) - set(inst_labels)
        if missing:
            raise ValueError(
                f"instances[{idx}]['bbox_labels'] missing keys: {missing}. Expected: {bbox_label_fields}",
            )

    def _validate_instance_keypoints(
        self,
        inst: dict[str, Any],
        idx: int,
        binding: frozenset[str],
        kp_label_fields: list[str],
    ) -> None:
        if "keypoints" not in binding:
            return
        if "keypoints" not in inst:
            raise ValueError(f"instances[{idx}] missing required key 'keypoints'")
        kps = inst["keypoints"]
        num_kps = kps.shape[0] if isinstance(kps, np.ndarray) else len(kps)
        if not (kp_label_fields and num_kps > 0):
            return
        kp_labels = inst.get("keypoint_labels")
        if kp_labels is None:
            raise ValueError(f"instances[{idx}] missing 'keypoint_labels'")
        missing = set(kp_label_fields) - set(kp_labels)
        if missing:
            raise ValueError(
                f"instances[{idx}]['keypoint_labels'] missing keys: {missing}. Expected: {kp_label_fields}",
            )
        for field in kp_label_fields:
            if len(kp_labels[field]) != num_kps:
                raise ValueError(
                    f"instances[{idx}]['keypoint_labels']['{field}'] has "
                    f"{len(kp_labels[field])} values but keypoints has {num_kps} rows",
                )

    def _repack_instances(self, data: dict[str, Any]) -> None:
        """Reconstitute per-instance dicts from flat arrays via a single row-aligned pass; relies
        on the post-transform `_resync_masks_to_bboxes` invariant being in place.

        `_resync_masks_to_bboxes` runs every iteration of the run loop, so the row-alignment
        invariant holds here: when `bboxes` is bound, `_bbox_instance_id == range(N)` and (when
        `masks` is also bound) `len(data["masks"]) == N`. So bbox/mask/kp row indices all coincide
        and a single linear `for row_idx in range(n)` rebuilds the instance dicts. The two old
        fallback branches (id-as-position drift, no-bbox iteration over `_instance_count`) are no
        longer reachable for the bboxes case.
        """
        binding = self._instance_binding
        if binding is None:
            msg = "_repack_instances requires instance_binding"
            raise RuntimeError(msg)

        kp_ids = np.array(data.pop(_KP_INSTANCE_ID, []))
        bbox_ids = data.pop(_BBOX_INSTANCE_ID, [])

        # When bboxes is bound, `bbox_ids` length is the surviving instance count (already rebased
        # to range(N) by the resync hook). For masks-or-keypoints-only bindings (no bbox-driven
        # filter exists), fall back to the unpack-time count.
        n = len(bbox_ids) if "bboxes" in binding else self._instance_count

        data["instances"] = [
            self._repack_one_instance(
                data,
                binding,
                bbox_row_idx=row_idx,
                mask_row_idx=row_idx,
                kp_group_id=row_idx,
                kp_ids=kp_ids,
            )
            for row_idx in range(n)
        ]

        self._cleanup_instance_data(data, binding)

    def _repack_one_instance(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        bbox_row_idx: int,
        mask_row_idx: int,
        kp_group_id: int,
        kp_ids: np.ndarray,
    ) -> dict[str, Any]:
        inst: dict[str, Any] = {}
        self._repack_mask_into(inst, data, binding, int(mask_row_idx))
        self._repack_bbox_into(inst, data, binding, bbox_row_idx)
        self._repack_keypoints_into(inst, data, binding, kp_group_id, kp_ids)
        self._repack_bbox_labels_into(inst, data, bbox_row_idx)
        self._repack_kp_labels_into(inst, data, binding, kp_group_id, kp_ids)
        return inst

    def _repack_mask_into(
        self,
        inst: dict[str, Any],
        data: dict[str, Any],
        binding: frozenset[str],
        original_instance_idx: int,
    ) -> None:
        if "masks" in binding and "masks" in data:
            mask = data["masks"][original_instance_idx]
            added = hasattr(self, "_added_channel_dim") and self._added_channel_dim.get("masks")
            if added and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)
            inst["mask"] = mask
        elif "mask" in binding and "mask" in data:
            inst["mask"] = data["mask"][:, :, original_instance_idx]

    def _repack_bbox_into(
        self,
        inst: dict[str, Any],
        data: dict[str, Any],
        binding: frozenset[str],
        new_idx: int,
    ) -> None:
        if "bboxes" in binding and "bboxes" in data:
            inst["bbox"] = data["bboxes"][new_idx]

    def _repack_keypoints_into(
        self,
        inst: dict[str, Any],
        data: dict[str, Any],
        binding: frozenset[str],
        old_idx: int,
        kp_ids: np.ndarray,
    ) -> None:
        if "keypoints" not in binding or "keypoints" not in data:
            return
        if kp_ids.size > 0:
            inst["keypoints"] = data["keypoints"][kp_ids == old_idx]
        else:
            kp_proc = self.processors.get("keypoints")
            if isinstance(kp_proc, KeypointsProcessor):
                inst["keypoints"] = kp_proc.params.make_empty_keypoints_array()
            else:
                inst["keypoints"] = np.zeros((0, 2), dtype=np.float32)

    def _repack_bbox_labels_into(self, inst: dict[str, Any], data: dict[str, Any], new_idx: int) -> None:
        if not self._bbox_label_map:
            return
        inst["bbox_labels"] = {
            user_name: data[internal_name][new_idx]
            for internal_name, user_name in self._bbox_label_map.items()
            if internal_name in data
        }

    def _repack_kp_labels_into(
        self,
        inst: dict[str, Any],
        data: dict[str, Any],
        binding: frozenset[str],
        old_idx: int,
        kp_ids: np.ndarray,
    ) -> None:
        if not (self._kp_label_map and "keypoints" in binding):
            return
        inst["keypoint_labels"] = {}
        for internal_name, user_name in self._kp_label_map.items():
            if internal_name not in data:
                continue
            field_values = data[internal_name]
            if kp_ids.size > 0:
                kp_mask = kp_ids == old_idx
                inst["keypoint_labels"][user_name] = [field_values[i] for i, keep in enumerate(kp_mask) if keep]
            else:
                inst["keypoint_labels"][user_name] = []

    def _cleanup_instance_data(self, data: dict[str, Any], binding: frozenset[str]) -> None:
        for key in ("mask", "masks", "bboxes", "keypoints"):
            if key in binding:
                data.pop(key, None)
        for internal_name in self._bbox_label_map:
            data.pop(internal_name, None)
        for internal_name in self._kp_label_map:
            data.pop(internal_name, None)

    def _clean_params_dict(
        self,
        params_dict: dict[str, Any] | None,
        label_map: dict[str, str],
    ) -> dict[str, Any] | None:
        if params_dict is None or not self._instance_binding:
            return params_dict
        label_fields = params_dict.get("label_fields")
        if label_fields:
            user_fields = [label_map.get(f, f) for f in label_fields if f not in {_BBOX_INSTANCE_ID, _KP_INSTANCE_ID}]
            params_dict = {**params_dict, "label_fields": user_fields}
        return params_dict

    def to_dict_private(self) -> dict[str, Any]:
        dictionary = super().to_dict_private()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": self._clean_params_dict(
                    bbox_processor.params.to_dict_private() if bbox_processor else None,
                    self._bbox_label_map,
                ),
                "keypoint_params": self._clean_params_dict(
                    keypoints_processor.params.to_dict_private() if keypoints_processor else None,
                    self._kp_label_map,
                ),
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
                "seed": getattr(self, "_base_seed", None),
            },
        )
        if self._instance_binding:
            dictionary["instance_binding"] = sorted(self._instance_binding)
        return dictionary

    def get_dict_with_id(self) -> dict[str, Any]:
        """Get dict with object IDs for replay. Extends super with bbox_params,
        keypoint_params, additional_targets, params, is_check_shapes.

        Returns:
            dict[str, Any]: Dictionary with composition data and object IDs.

        """
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": self._clean_params_dict(
                    bbox_processor.params.to_dict_private() if bbox_processor else None,
                    self._bbox_label_map,
                ),
                "keypoint_params": self._clean_params_dict(
                    keypoints_processor.params.to_dict_private() if keypoints_processor else None,
                    self._kp_label_map,
                ),
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        if self._instance_binding:
            dictionary["instance_binding"] = sorted(self._instance_binding)
        return dictionary

    @staticmethod
    def _check_single_data(data_name: str, data: Any) -> tuple[int, int]:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{data_name} must be numpy array type")
        return data.shape[:2]

    @staticmethod
    def _check_multi_data(data_name: str, data: Any) -> tuple[int, int]:
        """Check multi-item data format and return shape. Validates (N,H,W) or (N,H,W,C);
        returns (H,W) of first item. Raises TypeError if not ndarray or wrong ndim.

        Args:
            data_name (str): Name of the data field being checked
            data (Any): Input numpy array of shape (N, H, W, C) or (N, H, W)

        Returns:
            tuple[int, int]: (height, width) of the first item
        Raises:
            TypeError: If data format is invalid

        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{data_name} must be numpy array type")
        if data.ndim not in {3, 4}:  # (N,H,W) or (N,H,W,C)
            raise TypeError(f"{data_name} must be 3D or 4D array")
        return data.shape[1:3]  # Return (H,W)

    @staticmethod
    def _check_bbox_keypoint_params(internal_data_name: str, processors: dict[str, Any]) -> None:
        if internal_data_name in CHECK_BBOX_PARAM and processors.get("bboxes") is None:
            raise ValueError("bbox_params must be specified for bbox transformations")
        if internal_data_name in CHECK_KEYPOINTS_PARAM and processors.get("keypoints") is None:
            raise ValueError("keypoints_params must be specified for keypoint transformations")

    @staticmethod
    def _check_shapes(shapes: list[tuple[int, ...]], is_check_shapes: bool) -> None:
        if is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            raise ValueError(
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency).",
            )

    def _check_args(self, **kwargs: Any) -> None:
        shapes: list[tuple[int, ...]] = []  # For H,W checks
        volume_shapes: list[tuple[int, ...]] = []  # For D,H,W checks

        for data_name, data in kwargs.items():
            # Get internal name for additional targets
            internal_name = self._additional_targets.get(data_name, data_name)

            # Always check bbox/keypoint params for all data items
            self._check_bbox_keypoint_params(internal_name, self.processors)

            # Process and validate the data
            self._check_and_process_single_arg(data_name, internal_name, data, shapes, volume_shapes)

        self._check_shape_consistency(shapes, volume_shapes)

    def _check_and_process_single_arg(
        self,
        data_name: str,
        internal_name: str,
        data: Any,
        shapes: list[tuple[int, ...]],
        volume_shapes: list[tuple[int, ...]],
    ) -> None:
        """Check and process a single argument from _check_args. Validates type and shape
        for image, mask, images, volume, etc.; appends to shapes/volume_shapes.
        """
        # For single items (image, mask), we must validate even if None
        if internal_name in {"image", "mask"}:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{data_name} must be numpy array type")
            shapes.append(data.shape[:2])
            return

        # List of targets to check shapes for
        shape_check_targets = {"image", "mask", "images", "volume", "volumes", "mask3d", "masks", "masks3d"}

        # Skip if not in our check list
        if data_name not in shape_check_targets:
            return

        # Skip empty data or non-array inputs
        if data is None or not isinstance(data, np.ndarray):
            return

        # Skip arrays with size 0 (empty arrays)
        if data.size == 0:
            return

        # Process the shape based on data type
        self._process_data_shape(data_name, data, shapes, volume_shapes)

    def _check_shape_consistency(self, shapes: list[tuple[int, ...]], volume_shapes: list[tuple[int, ...]]) -> None:
        """Check consistency of shapes. When is_check_shapes, ensures all 2D shapes match
        and all 3D shapes match. Raises ValueError if inconsistent.
        """
        # Check H,W consistency
        self._check_shapes(shapes, self.is_check_shapes)

        # Check D,H,W consistency for volumes and 3D masks
        if self.is_check_shapes and volume_shapes and volume_shapes.count(volume_shapes[0]) != len(volume_shapes):
            raise ValueError(
                "Depth, Height and Width of volume, mask3d, volumes and masks3d should be equal. "
                "You can disable shapes check by setting is_check_shapes=False.",
            )

    def _get_init_params(self) -> dict[str, Any]:
        """Get parameters needed to recreate this Compose instance. Includes bbox_params,
        keypoint_params, additional_targets, p, is_check_shapes, strict, seed, etc.

        Returns:
            dict[str, Any]: Dictionary of initialization parameters

        """
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")

        bbox_params: BboxParams | None = None
        if bbox_processor:
            bp = cast("BboxParams", bbox_processor.params)
            if self._instance_binding and "bboxes" in self._instance_binding:
                user_fields = list(self._bbox_label_map.values()) or None
                bbox_params = BboxParams(
                    coord_format=bp.coord_format,
                    label_fields=user_fields,
                    bbox_type=bp.bbox_type,
                    min_area=bp.min_area,
                    min_visibility=bp.min_visibility,
                    min_width=bp.min_width,
                    min_height=bp.min_height,
                    check_each_transform=bp.check_each_transform,
                    clip_bboxes_on_input=bp.clip_bboxes_on_input,
                    filter_invalid_bboxes=bp.filter_invalid_bboxes,
                    max_accept_ratio=bp.max_accept_ratio,
                    clip_after_transform=bp.clip_after_transform,
                )
            else:
                bbox_params = bp

        kp_params: KeypointParams | None = None
        if keypoints_processor:
            kp = cast("KeypointParams", keypoints_processor.params)
            if self._instance_binding and "keypoints" in self._instance_binding:
                user_fields = list(self._kp_label_map.values()) or None
                kp_params = KeypointParams(
                    coord_format=kp.coord_format,
                    label_fields=user_fields,
                    remove_invisible=kp.remove_invisible,
                    angle_in_degrees=kp.angle_in_degrees,
                    check_each_transform=kp.check_each_transform,
                    label_mapping=kp.label_mapping or None,
                )
            else:
                kp_params = kp

        return {
            "bbox_params": bbox_params,
            "keypoint_params": kp_params,
            "additional_targets": self.additional_targets,
            "p": self.p,
            "is_check_shapes": self.is_check_shapes,
            "strict": self.strict,
            "mask_interpolation": getattr(self, "mask_interpolation", None),
            "seed": getattr(self, "_base_seed", None),
            "save_applied_params": getattr(self, "save_applied_params", False),
            "telemetry": getattr(self, "telemetry", True),
            "instance_binding": sorted(self._instance_binding) if self._instance_binding else None,
        }


class OneOf(BaseCompose):
    """Apply one of the child transforms at random; probabilities normalized as weights.
    Selected transform runs with force_apply=True.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.

    """

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super().__init__(transforms=transforms, p=p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply the OneOf composition to the input data. Selects one transform by weight,
        runs it with force_apply=True. In replay mode runs all in order.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        Raises:
            KeyError: If positional arguments are provided.

        """
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or self.py_random.random() < self.p):
            idx: int = self.random_generator.choice(len(self.transforms), p=self.transforms_ps)
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
            self._track_transform_params(t, data)
        return data


class SomeOf(BaseCompose):
    """Select exactly n transforms from the list and apply them. Selection uniform; each
    runs with its own p. Use replace=True for sampling with replacement.

    The selection of which `n` transforms to apply is done **uniformly at random**
    from the provided list. Each transform in the list has an equal chance of being selected.

    Once the `n` transforms are selected, each one is applied **based on its
    individual probability** `p`.

    Args:
        transforms (list[BasicTransform | BaseCompose]): A list of transforms to choose from.
        n (int): The exact number of transforms to select and potentially apply.
                 If `replace=False` and `n` is greater than the number of available transforms,
                 `n` will be capped at the number of transforms.
        replace (bool): Whether to sample transforms with replacement. If True, the same
                        transform can be selected multiple times (up to `n` times).
                        Default is False.
        p (float): The probability that this `SomeOf` composition will be applied.
                   If applied, it will select `n` transforms and attempt to apply them.
                   Default is 1.0.

    Note:
        - The overall probability `p` of the `SomeOf` block determines if *any* selection
          and application occurs.
        - The individual probability `p` of each transform inside the list determines if
          that specific transform runs *if it is selected*.
        - If `replace` is True, the same transform might be selected multiple times, and
          its individual probability `p` will be checked each time it's encountered.
        - When using pipeline modification operators (+, -, __radd__), the `n` parameter
          is preserved while the pool of available transforms changes:
          - `SomeOf([A, B], n=2) + C` → `SomeOf([A, B, C], n=2)` (selects 2 from 3 transforms)
          - This allows for dynamic adjustment of the transform pool without changing selection count.

    Examples:
        >>> import albumentations as A
        >>> transform = A.SomeOf([
        ...     A.HorizontalFlip(p=0.5),  # 50% chance to apply if selected
        ...     A.VerticalFlip(p=0.8),    # 80% chance to apply if selected
        ...     A.RandomRotate90(p=1.0), # 100% chance to apply if selected
        ... ], n=2, replace=False, p=1.0) # Always select 2 transforms uniformly

        # In each call, 2 transforms out of 3 are chosen uniformly.
        # For example, if HFlip and VFlip are chosen:
        # - HFlip runs if random() < 0.5
        # - VFlip runs if random() < 0.8
        # If VFlip and Rotate90 are chosen:
        # - VFlip runs if random() < 0.8
        # - Rotate90 runs if random() < 1.0 (always)

        >>> # Pipeline modification example:
        >>> # Add more transforms to the pool while keeping n=2
        >>> extended = transform + [A.Blur(p=1.0), A.RandomBrightnessContrast(p=0.7)]
        >>> # Now selects 2 transforms from 5 available transforms uniformly

    """

    def __init__(self, transforms: TransformsSeqType, n: int = 1, replace: bool = False, p: float = 1):
        super().__init__(transforms, p)
        self.n = n
        if not replace and n > len(self.transforms):
            self.n = len(self.transforms)
            warnings.warn(
                f"`n` is greater than number of transforms. `n` will be set to {self.n}.",
                UserWarning,
                stacklevel=2,
            )
        self.replace = replace

    def __call__(self, *arg: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply n randomly selected transforms from the list of transforms. Selection
        uniform; order fixed (sorted indices). Each transform applied with its own p.

        Args:
            *arg (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
                data = self.check_data_post_transform(data)
            return data

        if self.py_random.random() < self.p:  # Check overall SomeOf probability
            # Get indices uniformly
            indices_to_consider = self._get_idx()
            for i in indices_to_consider:
                t = self.transforms[i]
                # Apply the transform respecting its own probability `t.p`
                data = t(**data)
                self._track_transform_params(t, data)
                data = self.check_data_post_transform(data)
        return data

    def _get_idx(self) -> NDArray[np.int_]:
        # Use uniform probability for selection, ignore individual p values here
        idx = self.random_generator.choice(
            len(self.transforms),
            size=self.n,
            replace=self.replace,
        )
        idx.sort()
        return idx

    def to_dict_private(self) -> dict[str, Any]:
        dictionary = super().to_dict_private()
        dictionary.update({"n": self.n, "replace": self.replace})
        return dictionary

    def _get_init_params(self) -> dict[str, Any]:
        base_params = super()._get_init_params()
        base_params.update(
            {
                "n": self.n,
                "replace": self.replace,
            },
        )
        return base_params


class RandomOrder(SomeOf):
    """Apply a random subset of transforms in random order. Subclass of SomeOf; selection
    uniform, order random. Use n, replace, p.

    Selects exactly `n` transforms uniformly at random from the list, and then applies
    the selected transforms in a random order. Each selected transform is applied
    based on its individual probability `p`.

    Attributes:
        transforms (TransformsSeqType): A list of transformations to choose from.
        n (int): The number of transforms to apply. If `n` is greater than the number of available transforms
                 and `replace` is False, `n` will be set to the number of available transforms.
        replace (bool): Whether to sample transforms with replacement. If True, the same transform can be
                        selected multiple times. Default is False.
        p (float): Probability of applying the selected transforms. Should be in the range [0, 1]. Default is 1.0.

    Examples:
        >>> import albumentations as A
        >>> transform = A.RandomOrder([
        ...     A.HorizontalFlip(p=0.5),
        ...     A.VerticalFlip(p=1.0),
        ...     A.RandomBrightnessContrast(p=0.8),
        ... ], n=2, replace=False, p=1.0)
        >>> # This will uniformly select 2 transforms and apply them in a random order,
        >>> # respecting their individual probabilities (0.5, 1.0, 0.8).

    Note:
        - Inherits from SomeOf, but overrides `_get_idx` to ensure random order without sorting.
        - Selection is uniform; application depends on individual transform probabilities.

    """

    def __init__(self, transforms: TransformsSeqType, n: int = 1, replace: bool = False, p: float = 1):
        # Initialize using SomeOf's logic (which now does uniform selection setup)
        super().__init__(transforms=transforms, n=n, replace=replace, p=p)

    def _get_idx(self) -> NDArray[np.int_]:
        # Perform uniform random selection without replacement, like SomeOf
        # Crucially, DO NOT sort the indices here to maintain random order.
        return self.random_generator.choice(
            len(self.transforms),
            size=self.n,
            replace=self.replace,
        )


class OneOrOther(BaseCompose):
    """Select one or the other transform. Selected runs with force_apply=True. Exactly two
    transforms; p chooses first vs second. Like OneOf n=2 but binary choice.
    """

    def __init__(
        self,
        first: TransformType | None = None,
        second: TransformType | None = None,
        transforms: TransformsSeqType | None = None,
        p: float = 0.5,
    ):
        if transforms is None:
            if first is None or second is None:
                msg = "You must set both first and second or set transforms argument."
                raise ValueError(msg)
            transforms = [first, second]
        super().__init__(transforms=transforms, p=p)
        if len(self.transforms) != NUM_ONEOF_TRANSFORMS:
            warnings.warn("Length of transforms is not equal to 2.", stacklevel=2)

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply one or another transform to the input data. With probability p applies
        first transform, else second; both called with force_apply=True.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
                self._track_transform_params(t, data)
            return data

        if self.py_random.random() < self.p:
            return self.transforms[0](force_apply=True, **data)

        return self.transforms[-1](force_apply=True, **data)


class SelectiveChannelTransform(BaseCompose):
    """Apply transforms to selected image channels. Extracts channels, runs compose,
    writes back. Use channels=(0,1,2) for RGB. Supports +, -, __radd__.

    This class extends BaseCompose to allow selective application of transformations to
    specified image channels. It extracts the selected channels, applies the transformations,
    and then reinserts the transformed channels back into their original positions in the image.

    Args:
        transforms (TransformsSeqType):
            A sequence of transformations (from Albumentations) to be applied to the specified channels.
        channels (Sequence[int]):
            A sequence of integers specifying the indices of the channels to which the transforms should be applied.
        p (float): Probability that the transform will be applied; the default is 1.0 (always apply).

    Returns:
        dict[str, Any]: The transformed data dictionary, which includes the transformed 'image' key.

    Note:
        - When using pipeline modification operators (+, -, __radd__), the `channels` parameter
          is preserved in the resulting SelectiveChannelTransform instance.
        - Only the transform list is modified while maintaining the same channel selection behavior.

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        channels: Sequence[int] = (0, 1, 2),
        p: float = 1.0,
    ) -> None:
        super().__init__(transforms=transforms, p=p)
        self.channels = channels

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply transforms to specific channels of the image. Extracts self.channels,
        runs child transforms on sub-image, merges back. Other keys in data pass through.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if force_apply or self.py_random.random() < self.p:
            image = data["image"]

            selected_channels = image[:, :, self.channels]
            sub_image = np.ascontiguousarray(selected_channels)

            for t in self.transforms:
                sub_data = {"image": sub_image}
                sub_image = t(force_apply=False, **sub_data)["image"]
                self._track_transform_params(t, sub_data)

            transformed_channels = cv2.split(sub_image)
            output_img = image.copy()

            for idx, channel in zip(self.channels, transformed_channels, strict=True):
                output_img[:, :, idx] = channel

            data["image"] = np.ascontiguousarray(output_img)

        return data

    def _get_init_params(self) -> dict[str, Any]:
        """Get parameters needed to recreate this SelectiveChannelTransform instance.
        Extends base with channels. For _create_new_instance and serialization.

        Returns:
            dict[str, Any]: Dictionary of initialization parameters

        """
        base_params = super()._get_init_params()
        base_params.update(
            {
                "channels": self.channels,
            },
        )
        return base_params


class ReplayCompose(Compose):
    """Compose with replay: records params per call in save_key; use replay() to reapply
    same augmentations. Set save_key, deterministic=True.

    This class extends the Compose class with the ability to record and replay
    transformations. This is useful for applying the same sequence of random
    transformations to different data.

    Args:
        transforms (TransformsSeqType):
            List of transformations to compose.
        bbox_params (dict[str, Any] | BboxParams | None):
            Parameters for bounding box transforms.
        keypoint_params (dict[str, Any] | KeypointParams | None):
            Parameters for keypoint transforms.
        additional_targets (dict[str, str] | None):
            Dictionary of additional targets.
        p (float):
            Probability of applying the compose.
        is_check_shapes (bool):
            Whether to check shapes of different targets.
        save_key (str):
            Key for storing the applied transformations.
        seed (int | None):
            Controls reproducibility of random augmentations.
            See superclass documentation for further information.

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        save_key: str = "replay",
        seed: int | None = None,
        instance_binding: Sequence[str] | None = None,
    ):
        super().__init__(
            transforms,
            bbox_params,
            keypoint_params,
            additional_targets,
            p,
            is_check_shapes,
            seed=seed,
            instance_binding=instance_binding,
        )
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key
        self._available_keys.add(save_key)

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Apply transforms and record params for replay. Stores in save_key; fill_with_params
        and fill_applied complete serialized form for replay().

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **kwargs (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data and replay information.

        """
        kwargs[self.save_key] = defaultdict(dict)
        result = super().__call__(force_apply=force_apply, **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Replay saved augmentations. Restores pipeline from saved_augmentations via
        _restore_for_replay; runs with force_apply=True. Use for TTA or reproducibility.

        Args:
            saved_augmentations (dict[str, Any]): Previously saved augmentation parameters.
            **kwargs (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data using saved parameters.

        """
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(
        transform_dict: dict[str, Any],
        lambda_transforms: dict[str, Any] | None = None,
    ) -> TransformType:
        """Restore transform from replay dict; pass lambda_transforms for Lambda in pipeline.
        Recursively restores nested composes; sets replay_mode, params.

        Args:
            transform_dict (dict[str, Any]): A dictionary that contains transform data.
            lambda_transforms (dict[str, Any] | None): Optional dict of Lambda instances keyed by transform name.

        """
        applied = transform_dict["applied"]
        params = transform_dict["params"]
        lmbd = instantiate_nonserializable(transform_dict, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform_dict["__class_fullname__"]
            args = {k: v for k, v in transform_dict.items() if k not in ["__class_fullname__", "applied", "params"]}
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    ReplayCompose._restore_for_replay(t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        transform = cast("BasicTransform", transform)
        if isinstance(transform, BasicTransform):
            transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized: dict[str, Any], all_params: Any) -> None:
        """Fill serialized transform data with params for replay. Copies from all_params by
        id into serialized['params']; recurses into transforms. Mutates serialized.

        Args:
            serialized (dict[str, Any]): Serialized transform data.
            all_params (Any): Parameters to fill in.

        """
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized: dict[str, Any]) -> bool:
        """Set 'applied' flag for transforms based on parameters. Recurses; leaf applied =
        params is not None. Returns True if any transform was applied.

        Args:
            serialized (dict[str, Any]): Serialized transform data.

        Returns:
            bool: True if any transform was applied, False otherwise.

        """
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def to_dict_private(self) -> dict[str, Any]:
        dictionary = super().to_dict_private()
        dictionary.update({"save_key": self.save_key})
        return dictionary

    def _get_init_params(self) -> dict[str, Any]:
        base_params = super()._get_init_params()
        base_params.update(
            {
                "save_key": self.save_key,
            },
        )
        return base_params


class Sequential(BaseCompose):
    """Apply all transforms to targets in order. Use inside Compose with OneOf (e.g.
    OneOf([Sequential([A,B]), Sequential([C,D])])). Each runs with its own p.

    Note:
        This transform is not intended to be a replacement for `Compose`. Instead, it should be used inside `Compose`
        the same way `OneOf` or `OneOrOther` are used. For instance, you can combine `OneOf` with `Sequential` to
        create an augmentation pipeline that contains multiple sequences of augmentations and applies one randomly
        chose sequence to input data (see the `Example` section for an example definition of such pipeline).

    Examples:
        >>> import albumentations as A
        >>> transform = A.Compose([
        >>>    A.OneOf([
        >>>        A.Sequential([
        >>>            A.HorizontalFlip(p=0.5),
        >>>            A.ShiftScaleRotate(p=0.5),
        >>>        ]),
        >>>        A.Sequential([
        >>>            A.VerticalFlip(p=0.5),
        >>>            A.RandomBrightnessContrast(p=0.5),
        >>>        ]),
        >>>    ], p=1)
        >>> ])

    """

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super().__init__(transforms=transforms, p=p)

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply every transform in order to the data. No random choice between branches;
        all transforms in the list run one after another with their own p.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if self.replay_mode or force_apply or self.py_random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
                self._track_transform_params(t, data)
                data = self.check_data_post_transform(data)
        return data
