"""Module for composing multiple transforms into augmentation pipelines.

This module provides classes for combining multiple transformations into cohesive
augmentation pipelines. It includes various composition strategies such as sequential
application, random selection, and conditional application of transforms. These
composition classes handle the coordination between different transforms, ensuring
proper data flow and maintaining consistent behavior across the augmentation pipeline.
"""

import contextlib
import random
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, Union, cast

import cv2
import numpy as np

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
)
from .transforms_interface import BasicTransform
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
REPR_INDENT_STEP = 2

TransformType = Union[BasicTransform, "BaseCompose"]
TransformsSeqType = list[TransformType]

AVAILABLE_KEYS = ("image", "mask", "masks", "bboxes", "keypoints", "volume", "volumes", "mask3d", "masks3d")

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


class BaseCompose(Serializable):
    """Base class for composing multiple transforms together.

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
        """Track transform parameters if tracking is enabled."""
        if "applied_transforms" in data and hasattr(transform, "params") and transform.params:
            data["applied_transforms"].append((transform.__class__.__name__, transform.params.copy()))

    def set_random_state(
        self,
        random_generator: np.random.Generator,
        py_random: random.Random,
    ) -> None:
        """Set random state directly from generators.

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
        """Set random state from seed.

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
        """Set interpolation mode for mask resizing operations.

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
        """Apply transforms.

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
        """Get additional targets dictionary.

        Returns:
            dict[str, str]: Dictionary containing additional targets mapping.

        """
        return self._additional_targets

    @property
    def available_keys(self) -> set[str]:
        """Get set of available keys.

        Returns:
            set[str]: Set of string keys available for transforms.

        """
        return self._available_keys

    def indented_repr(self, indent: int = REPR_INDENT_STEP) -> str:
        """Get an indented string representation of the composition.

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
        """Get the full qualified name of the class.

        Returns:
            str: The shortest class fullname.

        """
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the class is serializable.

        Returns:
            bool: True if the class is serializable, False otherwise.

        """
        return True

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the composition to a dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the composition.

        """
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t.to_dict_private() for t in self.transforms],
        }

    def get_dict_with_id(self) -> dict[str, Any]:
        """Get a dictionary representation with object IDs for replay mode.

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
        """Add additional targets to all transforms.

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
        """Set _available_keys"""
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
        """Set deterministic mode for all transforms.

        Args:
            flag (bool): Whether to enable deterministic mode.
            save_key (str): Key to save replay parameters. Default: "replay".

        """
        for t in self.transforms:
            t.set_deterministic(flag, save_key)

    def check_data_post_transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """Check and filter data after transformation.

        Args:
            data (dict[str, Any]): Dictionary containing transformed data

        Returns:
            dict[str, Any]: Filtered data dictionary

        """
        if self.check_each_transform:
            shape = get_shape(data)

            for proc in self.check_each_transform:
                for data_name, data_value in data.items():
                    if data_name in proc.data_fields or (
                        data_name in self._additional_targets
                        and self._additional_targets[data_name] in proc.data_fields
                    ):
                        data[data_name] = proc.filter(data_value, shape)
        return data

    def _validate_transforms(self, transforms: list[Any]) -> None:
        """Validate that all elements are BasicTransform instances.

        Args:
            transforms: List of objects to validate

        Raises:
            TypeError: If any element is not a BasicTransform instance

        """
        for t in transforms:
            if not isinstance(t, BasicTransform):
                raise TypeError(
                    f"All elements must be instances of BasicTransform, got {type(t).__name__}",
                )

    def _combine_transforms(self, other: TransformType | TransformsSeqType, *, prepend: bool = False) -> "BaseCompose":
        """Combine transforms with the current compose.

        Args:
            other: Transform or sequence of transforms to combine
            prepend: If True, prepend other to the beginning; if False, append to the end

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
        """Add transform(s) to the end of this compose.

        Args:
            other: Transform or sequence of transforms to append

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
        """Add transform(s) to the beginning of this compose.

        Args:
            other: Transform or sequence of transforms to prepend

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
        """Remove transform from this compose by class type.

        Removes the first transform in the compose that matches the provided transform class.

        Args:
            other: Transform class to remove (e.g., A.HorizontalFlip)

        Returns:
            BaseCompose: New compose instance with transform removed
            NotImplemented: If other is not a BasicTransform class

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
        """Create a new instance of the same class with new transforms.

        Args:
            new_transforms: List of transforms for the new instance

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
        """Get parameters needed to recreate this instance.

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
        """Get effective seed considering worker context.

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
    """Compose multiple transforms together and apply them sequentially to input data.

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
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
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
    ):
        # Store the original base seed for worker context recalculation
        self._base_seed = seed

        # Get effective seed considering worker context
        effective_seed = self._get_effective_seed(seed)

        super().__init__(
            transforms=transforms,
            p=p,
            mask_interpolation=mask_interpolation,
            seed=effective_seed,
            save_applied_params=save_applied_params,
        )

        # Store telemetry parameter
        self.telemetry = telemetry

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

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)
        if not self.transforms:  # if no transforms -> do nothing, all keys will be available
            self._available_keys.update(AVAILABLE_KEYS)

        self.is_check_args = True
        self.strict = strict

        self.is_check_shapes = is_check_shapes
        self.check_each_transform = tuple(  # processors that checks after each transform
            proc for proc in self.processors.values() if getattr(proc.params, "check_each_transform", False)
        )
        self._set_check_args_for_transforms(self.transforms)

        self._set_processors_for_transforms(self.transforms)

        self.save_applied_params = save_applied_params
        self._images_was_list = False
        self._masks_was_list = False
        self._last_torch_seed: int | None = None

        # Track telemetry after nested composes are processed
        # This ensures nested composes have main_compose=False from disable_check_args_private
        if self.main_compose and settings.telemetry_enabled:
            with contextlib.suppress(Exception):
                client = get_telemetry_client()

                # Collect telemetry data
                env_info = get_environment_info()
                pipeline_info = collect_pipeline_info(self)

                # Combine all data
                telemetry_data = {
                    **env_info,
                    **pipeline_info,
                }

                # Always call the client, let it decide based on telemetry parameter
                client.track_compose_init(telemetry_data, telemetry=telemetry)

    @property
    def strict(self) -> bool:
        """Get the current strict mode setting.

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
        """Validate that no transforms have invalid arguments when strict mode is enabled."""

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
        """Disable argument checking for transforms.

        This method disables strict mode and argument checking for all transforms in the composition.
        """
        self.is_check_args = False
        self.strict = False
        self.main_compose = False

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply transformations to data with automatic worker seed synchronization.

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

        self.preprocess(data)

        for t in self.transforms:
            data = t(**data)
            self._track_transform_params(t, data)
            data = self.check_data_post_transform(data)

        return self.postprocess(data)

    def _check_worker_seed(self) -> None:
        """Check and update random seed if in worker context."""
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
        """Set state from unpickling and handle worker seed."""
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
        """Override to use worker-aware seed functionality.

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
        """Preprocess input data before applying transforms."""
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
        """Gather shapes from various data types for validation.

        Args:
            data: Data dictionary containing various arrays

        Returns:
            Tuple of (2D shapes list, 3D shapes list)

        """
        shapes: list[tuple[int, ...]] = []  # For H,W checks
        volume_shapes: list[tuple[int, ...]] = []  # For D,H,W checks

        # List of targets to check shapes for
        shape_check_targets = {"image", "mask", "images", "volume", "volumes", "mask3d", "masks", "masks3d"}

        for data_name, data_value in data.items():
            # Skip if not in our check list
            if data_name not in shape_check_targets:
                continue

            # Skip empty data
            if data_value is None or not isinstance(data_value, np.ndarray):
                continue

            # Skip arrays with size 0 (empty arrays)
            if data_value.size == 0:
                continue

            self._process_data_shape(data_name, data_value, shapes, volume_shapes)

        return shapes, volume_shapes

    def _process_data_shape(
        self,
        data_name: str,
        data_value: np.ndarray,
        shapes: list[tuple[int, ...]],
        volume_shapes: list[tuple[int, ...]],
    ) -> None:
        """Process shape of a single data item."""
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
        """Validate input data keys and arguments."""
        if not self.strict:
            return

        for data_name in data:
            if not self._is_valid_key(data_name):
                raise ValueError(f"Key {data_name} is not in available keys.")

        if self.is_check_args:
            self._check_args(**data)

    def _is_valid_key(self, key: str) -> bool:
        """Check if the key is valid for processing."""
        return key in self._available_keys or key in MASK_KEYS or key in IMAGE_KEYS or key == "applied_transforms"

    def _preprocess_processors(self, data: dict[str, Any]) -> None:
        """Run preprocessors if this is the main compose."""
        if not self.main_compose:
            return

        for processor in self.processors.values():
            processor.ensure_data_valid(data)
        for processor in self.processors.values():
            processor.preprocess(data)

    def _preprocess_arrays(self, data: dict[str, Any]) -> None:
        """Ensure all arrays are contiguous and add channel dimensions to grayscale data."""
        self._ensure_contiguous(data)
        self._add_grayscale_channels(data)

    def _ensure_contiguous(self, data: dict[str, Any]) -> None:
        """Ensure all numpy arrays are contiguous."""
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = np.ascontiguousarray(value)

    def _add_grayscale_channels(self, data: dict[str, Any]) -> None:
        """Add channel dimension to grayscale data if missing."""
        # Track which data had channel dimensions added
        self._added_channel_dim = {}

        # Keys that should have channel dimension added when grayscale
        grayscale_keys = {
            "image": 2,  # (H, W) => (H, W, 1)
            "images": 3,  # (N, H, W) => (N, H, W, 1)
            "mask": 2,  # (H, W) => (H, W, 1)
            "masks": 3,  # (N, H, W) => (N, H, W, 1)
            "volume": 3,  # (D, H, W) => (D, H, W, 1)
            "volumes": 4,  # (N, D, H, W) => (N, D, H, W, 1)
            "mask3d": 3,  # (D, H, W) => (D, H, W, 1)
            "masks3d": 4,  # (N, D, H, W) => (N, D, H, W, 1)
        }

        for key, expected_ndim in grayscale_keys.items():
            if key in data and isinstance(data[key], np.ndarray):
                if data[key].ndim == expected_ndim:
                    data[key] = np.expand_dims(data[key], axis=-1)
                    self._added_channel_dim[key] = True
                else:
                    self._added_channel_dim[key] = False

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply post-processing to data after all transforms have been applied.

        Args:
            data (dict[str, Any]): Data after transformation.

        Returns:
            dict[str, Any]: Post-processed data.

        """
        if self.main_compose:
            for p in self.processors.values():
                p.postprocess(data)

            # Remove channel dimensions that were added during preprocessing
            self._remove_grayscale_channels(data)

        return data

    def _remove_grayscale_channels(self, data: dict[str, Any]) -> None:
        """Remove channel dimensions that were added during preprocessing."""
        if not hasattr(self, "_added_channel_dim"):
            return

        for key, was_added in self._added_channel_dim.items():
            if was_added and key in data:
                value = data[key]

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
                        if key in {"image", "images"} and len(value.shape) >= 3 and value.shape[0] == 1:
                            # Image tensor with shape (1, H, W) -> (H, W) is not typical, skip
                            pass
                        elif key in {"mask", "masks", "mask3d", "masks3d"} and value.shape[-1] == 1:
                            # Mask tensor with shape (..., H, W, 1) -> (..., H, W)
                            data[key] = torch.squeeze(value, dim=-1)

    def to_dict_private(self) -> dict[str, Any]:
        dictionary = super().to_dict_private()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
                "seed": getattr(self, "_base_seed", None),
            },
        )
        return dictionary

    def get_dict_with_id(self) -> dict[str, Any]:
        """Get a dictionary representation with object IDs for replay mode.

        Returns:
            dict[str, Any]: Dictionary with composition data and object IDs.

        """
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        return dictionary

    @staticmethod
    def _check_single_data(data_name: str, data: Any) -> tuple[int, int]:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{data_name} must be numpy array type")
        return data.shape[:2]

    @staticmethod
    def _check_multi_data(data_name: str, data: Any) -> tuple[int, int]:
        """Check multi-item data format and return shape.

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
        """Check and process a single argument from _check_args."""
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
        """Check consistency of shapes."""
        # Check H,W consistency
        self._check_shapes(shapes, self.is_check_shapes)

        # Check D,H,W consistency for volumes and 3D masks
        if self.is_check_shapes and volume_shapes and volume_shapes.count(volume_shapes[0]) != len(volume_shapes):
            raise ValueError(
                "Depth, Height and Width of volume, mask3d, volumes and masks3d should be equal. "
                "You can disable shapes check by setting is_check_shapes=False.",
            )

    def _get_init_params(self) -> dict[str, Any]:
        """Get parameters needed to recreate this Compose instance.

        Returns:
            dict[str, Any]: Dictionary of initialization parameters

        """
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")

        return {
            "bbox_params": bbox_processor.params if bbox_processor else None,
            "keypoint_params": keypoints_processor.params if keypoints_processor else None,
            "additional_targets": self.additional_targets,
            "p": self.p,
            "is_check_shapes": self.is_check_shapes,
            "strict": self.strict,
            "mask_interpolation": getattr(self, "mask_interpolation", None),
            "seed": getattr(self, "_base_seed", None),
            "save_applied_params": getattr(self, "save_applied_params", False),
            "telemetry": getattr(self, "telemetry", True),
        }


class OneOf(BaseCompose):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

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
        """Apply the OneOf composition to the input data.

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
    """Selects exactly `n` transforms from the given list and applies them.

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
        """Apply n randomly selected transforms from the list of transforms.

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

    def _get_idx(self) -> np.ndarray[np.int_]:
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
    """Apply a random subset of transforms from the given list in a random order.

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

    def _get_idx(self) -> np.ndarray[np.int_]:
        # Perform uniform random selection without replacement, like SomeOf
        # Crucially, DO NOT sort the indices here to maintain random order.
        return self.random_generator.choice(
            len(self.transforms),
            size=self.n,
            replace=self.replace,
        )


class OneOrOther(BaseCompose):
    """Select one or another transform to apply. Selected transform will be called with `force_apply=True`."""

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
        """Apply one or another transform to the input data.

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
    """A transformation class to apply specified transforms to selected channels of an image.

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
        """Apply transforms to specific channels of the image.

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
                sub_image = t(**sub_data)["image"]
                self._track_transform_params(t, sub_data)

            transformed_channels = cv2.split(sub_image)
            output_img = image.copy()

            for idx, channel in zip(self.channels, transformed_channels, strict=True):
                output_img[:, :, idx] = channel

            data["image"] = np.ascontiguousarray(output_img)

        return data

    def _get_init_params(self) -> dict[str, Any]:
        """Get parameters needed to recreate this SelectiveChannelTransform instance.

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
    """Composition class that enables transform replay functionality.

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
    ):
        super().__init__(transforms, bbox_params, keypoint_params, additional_targets, p, is_check_shapes, seed=seed)
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key
        self._available_keys.add(save_key)

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Apply transforms and record parameters for future replay.

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
        """Replay previously saved augmentations.

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
        """Args:
        transform_dict (dict[str, Any]): A dictionary that contains transform data.
        lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
        This dictionary is required when you are restoring a pipeline that contains lambda transforms.
        Keys in that dictionary should be named same as `name` arguments in respective lambda transforms
        from a serialized pipeline.

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
        """Fill serialized transform data with parameters for replay.

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
        """Set 'applied' flag for transforms based on parameters.

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
    """Sequentially applies all transforms to targets.

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
        """Apply all transforms in sequential order.

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
