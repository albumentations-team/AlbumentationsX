"""Module containing base interfaces for all transform implementations.

This module defines the fundamental transform interfaces that form the base hierarchy for
all transformation classes in Albumentations. It provides abstract classes that define
common behavior for image, keypoint, bounding box, and volumetric transformations.
The interfaces handle parameter validation, random state management, target type checking,
and serialization capabilities that are inherited by concrete transform implementations.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, ClassVar, cast, get_args, get_type_hints
from warnings import warn

import cv2
import numpy as np
import torch
from albucore import sz_lut
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from albumentations.core.bbox_utils import BboxProcessor
from albumentations.core.binding import FrameBinding, TargetResult
from albumentations.core.invocation import (
    InvocationContext,
    InvocationRngOwner,
    SamplingContext,
    TransformInvocationState,
    get_completed_transform_state,
    get_current_invocation,
    publish_completed_transform_state,
)
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.tensor import (
    TENSOR_ANNOTATION_TARGETS,
    TENSOR_CANONICAL_RANKS,
    TENSOR_CANONICAL_SHAPE_DESCRIPTIONS,
    TENSOR_TARGETS,
    numpy_to_tensor_annotation,
    numpy_to_tensor_spatial,
    tensor_metadata_field_target,
    tensor_metadata_to_numpy,
    tensor_to_numpy_annotation,
    tensor_to_numpy_spatial,
    validate_tensor_input,
    validate_tensor_metadata_input,
)
from albumentations.core.transform_params import (
    SampledParams,
    SampledParamsError,
    TargetParams,
    TargetSet,
    required_parameter_names,
)
from albumentations.core.validation import ValidatedTransformMeta

from .serialization import Serializable, SerializableMeta, get_shortest_class_fullname
from .type_definitions import (
    ALL_TARGETS,
    ImageType,
    StackedMasks4D,
    Targets,
    VolumeType,
)
from .utils import format_args, get_volume_shape, get_volumes_shape

__all__ = [
    "BasicTransform",
    "DualTransform",
    "ImageOnlyTransform",
    "NoOp",
    "SampledParams",
    "SampledParamsError",
    "SamplingContext",
    "TargetParams",
    "TargetSet",
    "Transform3D",
    "VolumeOnlyTransform",
]

_TARGET_APPLY_METHODS = {
    "image": "apply",
    "images": "apply_to_images",
    "mask": "apply_to_mask",
    "masks": "apply_to_masks",
    "bboxes": "apply_to_bboxes",
    "keypoints": "apply_to_keypoints",
    "volume": "apply_to_volume",
    "volumes": "apply_to_volumes",
    "mask3d": "apply_to_mask3d",
    "masks3d": "apply_to_masks3d",
    "user_data": "apply_to_user_data",
}

_SHARED_SHAPE_INDICES = {
    "image_chw": (1, 2, 0),
    "mask_chw": (1, 2, 0),
    "images_nchw": (2, 3, 1),
    "masks_nchw": (2, 3, 1),
    "volume_cdhw": (2, 3, 0),
    "mask3d_cdhw": (2, 3, 0),
    "volumes_ncdhw": (3, 4, 1),
    "masks3d_ncdhw": (3, 4, 1),
    "volumes_ndhwc": (2, 3, 4),
    "masks3d_ndhwc": (2, 3, 4),
}
_BATCH_SHARED_SHAPE_TARGETS = {"images", "volume", "volumes", "masks", "mask3d", "masks3d"}

_TARGET_NAMES = {target: target.name.lower() for target in Targets}


class Interpolation:
    def __init__(self, downscale: int = cv2.INTER_NEAREST, upscale: int = cv2.INTER_NEAREST):
        self.downscale = downscale
        self.upscale = upscale


class BaseTransformInitSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    p: float = Field(ge=0, le=1)
    strict: bool


class _BasicTransformInitSchema(BaseTransformInitSchema):
    pass


class CombinedMeta(SerializableMeta, ValidatedTransformMeta):
    pass


class _DiscardedAppliedOverrides:
    """Discards realized policy when no replay, trace, or observation consumer exists, avoiding a temporary dictionary
    in ordinary Compose calls.

    This is deliberately not a `dict` subclass. A no-op mapping inherits mutation
    methods such as `setdefault` and `|=` that can retain data globally even when
    `__setitem__` is overridden. Sampling supports only assignment and `update`.
    """

    def __setitem__(self, key: str, value: Any) -> None:
        return

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Discards bulk realized-policy writes for non-observing calls, preserving dictionary-update compatibility
        without retaining replay data.
        """
        return


_DISCARDED_APPLIED_OVERRIDES = _DiscardedAppliedOverrides()
_EMPTY_APPLIED_OVERRIDES: Mapping[str, Any] = {}


@dataclass(frozen=True, slots=True)
class _TensorFallbackRoute:
    """Describe one leaf-local NumPy fallback without retaining data on the transform."""

    targets: tuple[tuple[str, str], ...]
    metadata: tuple[tuple[str, Any], ...]


def _annotation_includes_tensor(annotation: object) -> bool:
    return annotation is torch.Tensor or any(_annotation_includes_tensor(argument) for argument in get_args(annotation))


@cache
def _handler_accepts_tensor(handler: Callable[..., Any]) -> bool:
    try:
        type_hints = get_type_hints(handler, include_extras=True)
        parameters = tuple(inspect.signature(handler).parameters.values())
    except (NameError, TypeError, ValueError):
        return False

    if parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    input_parameter = next(
        (
            parameter
            for parameter in parameters
            if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
        ),
        None,
    )
    if input_parameter is None:
        return False
    return _annotation_includes_tensor(type_hints.get(input_parameter.name))


def _handler_has_tensor_path(handler: Callable[..., Any]) -> bool:
    return _handler_accepts_tensor(inspect.unwrap(getattr(handler, "__func__", handler)))


def _sequence_metadata_target(value: torch.Tensor) -> str | None:
    return "image" if value.ndim in {2, 3} else None


def _iter_declared_metadata_tensors(
    value: Any,
    *,
    path: str,
    target: str | None = None,
) -> tuple[tuple[str, str | None, torch.Tensor], ...]:
    """Find Tensor values read through one `targets_as_params` key."""
    if isinstance(value, torch.Tensor):
        return ((path, target, value),)
    if isinstance(value, Mapping):
        mapping_tensors: list[tuple[str, str | None, torch.Tensor]] = []
        for key, nested in value.items():
            nested_target = tensor_metadata_field_target(key)
            if nested_target is not None or isinstance(nested, torch.Tensor):
                mapping_tensors.extend(
                    _iter_declared_metadata_tensors(nested, path=f"{path}[{key!r}]", target=nested_target),
                )
        return tuple(mapping_tensors)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        sequence_tensors: list[tuple[str, str | None, torch.Tensor]] = []
        for index, nested in enumerate(value):
            nested_target = target
            if nested_target is None and isinstance(nested, torch.Tensor):
                nested_target = _sequence_metadata_target(nested)
            if nested_target is not None or isinstance(nested, (Mapping, torch.Tensor)):
                sequence_tensors.extend(
                    _iter_declared_metadata_tensors(nested, path=f"{path}[{index}]", target=nested_target),
                )
        return tuple(sequence_tensors)
    return ()


def _metadata_to_numpy(value: Any, target: str | None = None) -> Any:
    """Build the NumPy view of one value read through `targets_as_params`."""
    if isinstance(value, torch.Tensor):
        return tensor_metadata_to_numpy(value, target)
    if isinstance(value, Mapping):
        converted = dict(value)
        for key, nested in value.items():
            nested_target = tensor_metadata_field_target(key)
            if nested_target is not None or isinstance(nested, torch.Tensor):
                converted[key] = _metadata_to_numpy(nested, nested_target)
        return converted
    if isinstance(value, list):
        return [_metadata_sequence_item_to_numpy(item, target) for item in value]
    if isinstance(value, tuple):
        return tuple(_metadata_sequence_item_to_numpy(item, target) for item in value)
    return value


def _metadata_sequence_item_to_numpy(value: Any, target: str | None) -> Any:
    if target is not None or isinstance(value, Mapping):
        return _metadata_to_numpy(value, target)
    if isinstance(value, torch.Tensor):
        return _metadata_to_numpy(value, _sequence_metadata_target(value))
    return value


class BasicTransform(InvocationRngOwner, Serializable, metaclass=CombinedMeta):
    """Base class for all transforms in Albumentations. Provides core functionality for application,
    serialization, and params.

    This class provides core functionality for transform application, serialization,
    and parameter handling. It defines the interface that all transforms must follow
    and implements common methods used across different transform types.

    Class Attributes:
        _targets (tuple[Targets, ...] | Targets): Target types this transform can work with.
        _available_keys (set[str]): String representations of valid target keys.
        _key2func (dict[str, Callable[..., Any]]): Mapping between target keys and their processing functions.
        _preserves_input_image_range (bool): Whether image targets retain the normalized range of their input dtype.

    Args:
        interpolation (int): Interpolation method for image transforms.
        fill (int | float | list[int] | list[float]): Fill value for image padding.
        fill_mask (int | float | list[int] | list[float]): Fill value for mask padding.
        deterministic (bool, optional): Whether the transform is deterministic.
        save_key (str, optional): Key for saving transform parameters.
        replay_mode (bool, optional): Whether the transform is in replay mode.
        applied_in_replay (bool, optional): Whether the transform was applied in replay.
        p (float): Probability of applying the transform.

    Note:
        The base class methods use *args to allow subclasses to add specific named parameters
        (e.g., def apply(self, img, gamma, **params) is a valid override of apply(self, img, *args, **params)).

    """

    _targets: tuple[Targets | str, ...] | Targets | str = ()
    _target_apply_methods: ClassVar[Mapping[str, str]] = {}
    _available_keys: set[str]  # targets that this transform, as string, lower-cased
    _key2func: dict[
        str,
        Callable[..., Any],
    ]  # mapping for targets (plus additional targets) and methods for which they depend
    _transform_init_args_names_cache: ClassVar[tuple[str, ...] | None] = None
    call_backup = None
    interpolation: int
    fill: tuple[float, ...] | float
    fill_mask: tuple[float, ...] | float | None
    # replay mode params
    deterministic: bool = False
    save_key = "replay"
    replay_mode = False
    applied_in_replay = False

    InitSchema: ClassVar[type[BaseTransformInitSchema]] = _BasicTransformInitSchema
    _valid_applied_config_keys_cache: ClassVar[frozenset[str] | None] = None
    _applied_replay_class: ClassVar[type["BasicTransform"] | None] = None
    _sampling_spatial_rank: ClassVar[int | None] = None
    _runtime_generated_params: ClassVar[frozenset[str]] = frozenset()
    _runtime_binding_params: ClassVar[Mapping[str, None]] = {}
    _preserves_input_image_range: ClassVar[bool] = True  # image targets retain the input dtype's normalized range
    _removed_sampling_hooks: ClassVar[frozenset[str]] = frozenset({"get_params", "get_params_dependent_on_data"})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject removed sampling hooks when a transform subclass is declared, keeping the execution hot path free
        from legacy compatibility checks.

        Only methods declared directly on the new class are considered. This lets
        an unrelated base class retain a same-named helper while making a former
        Albumentations sampling override fail at import or class-definition time.
        """
        super().__init_subclass__(**kwargs)
        cls._runtime_binding_params = FrameBinding.empty_parameters(cls._runtime_generated_params)
        removed_hooks = sorted(cls._removed_sampling_hooks.intersection(cls.__dict__))
        if removed_hooks:
            names = ", ".join(removed_hooks)
            raise TypeError(
                f"{cls.__name__} defines removed sampling hook(s): {names}. "
                "Implement sample_parameters(params, data, targets, sampling) instead.",
            )

    def _tensor_fallback_route(self, data: Mapping[str, Any]) -> _TensorFallbackRoute:
        """Collect visible Tensor targets and declared Tensor metadata for one leaf."""
        is_tensor_terminal = getattr(self, "_is_tensor_terminal", False)
        functions = self._dispatch_functions()
        invocation = get_current_invocation()
        aliases = self._additional_targets
        if invocation is not None and invocation.frame_state is not None:
            aliases = {**aliases, **invocation.frame_state.routes}
        targets = tuple(
            (data_name, aliases.get(data_name, data_name))
            for data_name, value in data.items()
            if isinstance(value, torch.Tensor)
            and aliases.get(data_name, data_name) in TENSOR_TARGETS
            and (is_tensor_terminal or data_name in functions or data_name in self.targets_as_params)
        )
        for data_name, target in targets:
            validate_tensor_input(data[data_name], data_name, target)

        metadata: list[tuple[str, Any]] = []
        for data_name in self.get_tensor_metadata_keys():
            if data_name not in data:
                continue
            metadata_value = data[data_name]
            metadata_tensors = _iter_declared_metadata_tensors(metadata_value, path=data_name)
            if not metadata_tensors:
                continue
            for path, metadata_target, value in metadata_tensors:
                validate_tensor_metadata_input(value, path, metadata_target)
            metadata.append((data_name, metadata_value))
        if (targets or metadata) and is_tensor_terminal:
            raise TypeError(
                "ToTensorV2 and ToTensor3D accept NumPy input only; remove this transform from the pipeline",
            )
        for data_name, target in targets:
            if data[data_name].ndim != TENSOR_CANONICAL_RANKS[target]:
                raise TypeError(
                    f"{data_name} must have canonical shape {TENSOR_CANONICAL_SHAPE_DESCRIPTIONS[target]} "
                    "before transform dispatch; pass optional-channel inputs through Compose",
                )
        return _TensorFallbackRoute(targets=targets, metadata=tuple(metadata))

    def get_tensor_metadata_keys(self) -> frozenset[str]:
        """Return `targets_as_params` keys that may contain Tensor metadata."""
        return frozenset(
            data_name
            for data_name in self.targets_as_params
            if self._additional_targets.get(data_name, data_name) not in TENSOR_TARGETS
        )

    def iter_tensor_metadata_inputs(
        self,
        data: Mapping[str, Any],
    ) -> tuple[tuple[str, str | None, torch.Tensor], ...]:
        """Return Tensor values nested under declared metadata keys."""
        result: list[tuple[str, str | None, torch.Tensor]] = []
        for data_name in self.get_tensor_metadata_keys():
            if data_name in data:
                result.extend(_iter_declared_metadata_tensors(data[data_name], path=data_name))
        return tuple(result)

    def _can_bypass_tensor_route(
        self,
        data: Mapping[str, Any],
        invocation: InvocationContext | None,
    ) -> bool:
        if invocation is not None and invocation.has_tensor_inputs is not None:
            return not invocation.has_tensor_inputs
        return not self.targets_as_params and not any(isinstance(value, torch.Tensor) for value in data.values())

    def _uses_tensor_fallback(self, route: _TensorFallbackRoute) -> bool:
        """Return whether the complete leaf invocation must use its NumPy path."""
        functions = self._dispatch_functions()
        if route.metadata:
            return True
        for data_name, _ in route.targets:
            handler = functions.get(data_name)
            if handler is None:
                return True
            implementation = getattr(handler, "__func__", handler)
            if implementation is BasicTransform.apply_to_volumes:
                handler = getattr(self, self._target_apply_methods.get("volume", _TARGET_APPLY_METHODS["volume"]))
            elif implementation is DualTransform.apply_to_masks3d:
                handler = getattr(self, self._target_apply_methods.get("mask3d", _TARGET_APPLY_METHODS["mask3d"]))
            if not _handler_has_tensor_path(handler):
                return True
        return False

    @staticmethod
    def _enter_tensor_fallback(
        data: Mapping[str, Any],
        route: _TensorFallbackRoute,
    ) -> dict[str, Any]:
        numpy_data = dict(data)
        for data_name, target in route.targets:
            value = data[data_name]
            numpy_data[data_name] = (
                tensor_to_numpy_annotation(value, target)
                if target in TENSOR_ANNOTATION_TARGETS
                else tensor_to_numpy_spatial(value, target)
            )
        for data_name, value in route.metadata:
            numpy_data[data_name] = _metadata_to_numpy(value)
        return numpy_data

    @staticmethod
    def _restore_tensor_fallback(
        data: dict[str, Any],
        route: _TensorFallbackRoute,
    ) -> dict[str, Any]:
        for data_name, target in route.targets:
            value = data.get(data_name)
            if not isinstance(value, np.ndarray):
                continue
            data[data_name] = (
                numpy_to_tensor_annotation(value, target)
                if target in TENSOR_ANNOTATION_TARGETS
                else numpy_to_tensor_spatial(value, target)
            )
        data.update(route.metadata)
        return data

    def __init__(self, p: float = 0.5):
        self.p = p
        self.invocation_key = object()
        self._additional_targets: dict[str, str] = {}
        self._replay_params: dict[Any, Any] = {}
        self._key2func = {}
        self._set_keys()
        self._initialize_invocation_rng(None)
        self._strict = False  # Use private attribute
        self.invalid_args: list[str] = []  # Store invalid args found during init

    @property
    def strict(self) -> bool:
        """Get the current strict mode setting. Returns True if strict validation of init arguments
        is enabled, False otherwise. Read-only.

        Returns:
            bool: True if strict mode is enabled, False otherwise.

        """
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        """Set strict mode and validate for invalid arguments if enabled. When True, invalid
        __init__ args raise ValueError. Use at init or before apply.
        """
        if value == self._strict:
            return  # No change needed

        # Only validate if strict is being set to True and we have stored init args
        if value and hasattr(self, "_init_args"):
            valid_args = {"p", "strict"}  # Base valid args
            if hasattr(self, "InitSchema"):
                valid_args.update(self.InitSchema.model_fields.keys())

            invalid_args = [name_arg for name_arg in self._init_args if name_arg not in valid_args]

            if invalid_args:
                message = (
                    f"Argument(s) '{', '.join(invalid_args)}' are not valid for transform {self.__class__.__name__}"
                )
                if value:  # In strict mode
                    raise ValueError(message)
                warn(message, stacklevel=2)

        self._strict = value

    @property
    def params(self) -> dict[Any, Any]:
        """Returns active parameters or caller-local observations. Normal Compose runs expose no child state, avoiding
        stale or shared data.
        """
        invocation = get_current_invocation()
        if invocation is None:
            state = get_completed_transform_state(self)
            return {} if state is None or state.params is None else state.params
        if not invocation.collect_applied:
            return {}
        state = invocation.get_transform_state(self)
        return {} if state is None or state.params is None else state.params

    @params.setter
    def params(self, value: dict[Any, Any]) -> None:
        invocation = get_current_invocation()
        if invocation is None:
            self._replay_params = value
            return
        if not invocation.collect_applied:
            msg = "Transform parameters are available only for direct calls, save_applied_params, or tracing"
            raise RuntimeError(msg)
        invocation.transform_state(self).params = value

    @property
    def applied_config(self) -> dict[str, Any]:
        """Returns caller-local realized configuration observations. Normal Compose runs expose no child configuration,
        avoiding stale or shared data.
        """
        invocation = get_current_invocation()
        if invocation is None:
            state = get_completed_transform_state(self)
            return {} if state is None or state.applied_config is None else state.applied_config
        if not invocation.collect_applied:
            return {}
        state = invocation.get_transform_state(self)
        return {} if state is None or state.applied_config is None else state.applied_config

    def _new_invocation_context(self) -> InvocationContext:
        """Creates a call-local observing context for direct execution, exposing parameters without storing sampled
        values or generators on the transform instance.
        """
        return super()._create_invocation_context(collect_applied=True)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickled transforms and clear runtime worker context so the first worker call
        can resynchronize against the active DataLoader seed.
        """
        self._restore_invocation_pickle_state(state)

    def __getstate__(self) -> dict[str, Any]:
        """Returns pickle-safe configuration, omitting runtime locks and thread reservations so workers create local
        execution machinery in their receiving process.
        """
        return self._get_invocation_pickle_state()

    def get_dict_with_id(self) -> dict[str, Any]:
        """Return a dictionary representation of the transform with its ID. Used for replay and
        debugging; includes id(self). Same as to_dict plus id.

        Returns:
            dict[str, Any]: Dictionary containing transform parameters and ID.

        """
        d = self.to_dict_private()
        d.update({"id": id(self)})
        return d

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Inspect the transform constructor and return its serializable public argument names, keeping
        inherited implementation details out of persisted configurations.
        """
        transform_cls = type(self)
        cache = transform_cls.__dict__.get("_transform_init_args_names_cache")
        if cache is not None:
            return cache

        signature = inspect.signature(transform_cls.__init__)
        result = tuple(
            sorted(
                name
                for name, parameter in signature.parameters.items()
                if name not in {"self", "strict"}
                and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
            ),
        )
        type.__setattr__(transform_cls, "_transform_init_args_names_cache", result)
        return result

    def get_processor(self, key: str) -> BboxProcessor | KeypointsProcessor | None:
        """Return the active annotation session for this invocation, keeping a leaf detached from
        root configuration and mutable processor state owned by other callers.
        """
        invocation = get_current_invocation()
        return None if invocation is None else invocation.get_processor(key)

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Any:
        """Apply the transform to the input data. Accepts named kwargs (image, mask, bboxes, etc.);
        returns dict of transformed data.

        Args:
            *args (Any): Positional arguments are not supported and will raise an error.
            force_apply (bool, optional): If True, the transform will be applied regardless of probability.
            **kwargs (Any): Input data to transform as named arguments.

        Returns:
            Any: Transformed data (dict of transformed inputs).

        Raises:
            KeyError: If positional arguments are provided.

        """
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)
        invocation = get_current_invocation()
        if invocation is None:
            if self.can_apply_without_invocation(force_apply=force_apply):
                return self.apply_without_invocation(force_apply=force_apply, publish_observation=True, **kwargs)
            context = self._new_invocation_context()
            with context:
                return self.apply_in_invocation(context, *args, force_apply=force_apply, **kwargs)
        return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **kwargs)

    def can_apply_without_invocation(self, *, force_apply: bool) -> bool:
        """Recognizes a pure direct leaf that needs no active state, preserving fast deterministic calls while all
        other leaves retain full invocation isolation.

        The base sampler deliberately has no data-dependent or random behavior. Concrete samplers, probabilistic
        leaves, replay, deterministic replay recording, and processor-backed leaves retain the full invocation
        context so custom code always sees isolated state.
        """
        return (
            (force_apply or self.p >= 1.0)
            and not self.replay_mode
            and not self.deterministic
            and type(self).sample_parameters is BasicTransform.sample_parameters
        )

    def apply_without_invocation(
        self,
        *,
        force_apply: bool,
        publish_observation: bool,
        **kwargs: Any,
    ) -> Any:
        """Run a deterministic leaf without invocation state so Compose skips ContextVar and stream
        setup while direct calls still publish their caller-local observations.
        """
        if not self.can_apply_without_invocation(force_apply=force_apply):
            msg = "Transform requires an active invocation"
            raise RuntimeError(msg)
        state = TransformInvocationState() if publish_observation else None
        result = self._apply_sampled(state, publish_observation, None, **kwargs)
        if state is not None:
            publish_completed_transform_state(self, state)
        return result

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **kwargs: Any,
    ) -> Any:
        """Applies one leaf through the active root invocation, returning early for skipped leaves without allocating
        child observations on ordinary Compose calls.
        """
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)
        if getattr(self, "_is_tensor_terminal", False):
            self._tensor_fallback_route(kwargs)
        if not self.replay_mode and not force_apply and self.p <= 0.0:
            return kwargs
        if self.replay_mode:
            state = invocation.transform_state(self) if invocation.collect_applied else None
            return self._apply_replay(state, invocation=invocation, **kwargs)

        state = invocation.get_transform_state(self) if invocation.collect_applied else None
        if state is not None:
            state.params = None
            state.applied_config = None

        if not self._should_apply_in_invocation(invocation, force_apply=force_apply):
            return kwargs
        state = invocation.transform_state(self) if invocation.collect_applied else None
        return self._apply_sampled(state, invocation.collect_applied, invocation, **kwargs)

    def _apply_sampled(
        self,
        state: TransformInvocationState | None,
        collect_applied: bool,
        invocation: InvocationContext | None,
        **kwargs: Any,
    ) -> Any:
        """Run one applied leaf through its Tensor-aware or NumPy fallback route."""
        if self._can_bypass_tensor_route(kwargs, invocation):
            return self._apply_sampled_in_route(state, collect_applied, invocation, **kwargs)
        route = self._tensor_fallback_route(kwargs)
        if self._uses_tensor_fallback(route):
            result = self._apply_sampled_in_route(
                state,
                collect_applied,
                invocation,
                **self._enter_tensor_fallback(kwargs, route),
            )
            return self._restore_tensor_fallback(result, route)
        return self._apply_sampled_in_route(state, collect_applied, invocation, **kwargs)

    def _apply_sampled_in_route(
        self,
        state: TransformInvocationState | None,
        collect_applied: bool,
        invocation: InvocationContext | None,
        **kwargs: Any,
    ) -> Any:
        """Samples parameters after probability succeeds and records policy only for replay, trace, or explicit
        observation that needs the durable artifact.
        """
        targets = (
            None if type(self).sample_parameters is BasicTransform.sample_parameters else self._build_target_set(kwargs)
        )
        params = self.update_transform_params(params={}, data=kwargs, invocation=invocation, targets=targets)

        if self.targets_as_params:
            missing_keys = set(self.targets_as_params).difference(kwargs.keys())
            if missing_keys and not (missing_keys == {"image"} and "images" in kwargs):
                msg = f"{self.__class__.__name__} requires {self.targets_as_params} missing keys: {missing_keys}"
                raise ValueError(msg)

        if (
            targets is None
            and state is None
            and not self.deterministic
            and type(self).apply_with_params in {BasicTransform.apply_with_params, DualTransform.apply_with_params}
        ):
            if invocation is not None and invocation.frame_state is not None:
                return self.apply_with_params(SampledParams(params=params), **kwargs)
            return self.apply_with_uniform_params(params, **kwargs)

        applied_overrides, sampled_params = self._sample_parameters(
            params=params,
            data=kwargs,
            targets=targets,
            invocation=invocation,
            collect_applied=collect_applied,
        )

        effective_params = SampledParams(
            params={**params, **sampled_params.params},
            target_params=sampled_params.target_params,
            target_schema=targets.schema() if targets is not None and sampled_params.target_params else None,
        )
        self._validate_sampled_params(effective_params, targets, kwargs)

        if state is not None:
            state.params = effective_params.to_dict()
            self._build_applied_config(state=state, overrides=applied_overrides)

        if self.deterministic:
            saved_params = kwargs[self.save_key]
            transform_id = id(self)
            existing = saved_params.get(transform_id)
            if existing is None:
                saved_params[transform_id] = [deepcopy(effective_params.to_dict())]
            elif isinstance(existing, list):
                existing.append(deepcopy(effective_params.to_dict()))
            else:
                saved_params[transform_id] = [existing, deepcopy(effective_params.to_dict())]
        return self.apply_with_params(effective_params, **kwargs)

    def _sample_parameters(
        self,
        *,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet | None,
        invocation: InvocationContext | None,
        collect_applied: bool,
    ) -> tuple[Any, SampledParams]:
        if targets is None:
            return _EMPTY_APPLIED_OVERRIDES, SampledParams(params={})
        if invocation is None:
            msg = "sampling transforms require an active sampling context"
            raise RuntimeError(msg)

        applied_overrides = {} if collect_applied else cast("dict[str, Any]", _DISCARDED_APPLIED_OVERRIDES)
        self._validate_spatial_targets(targets)
        sampled_params = self.sample_parameters(
            params=params,
            data=data,
            targets=targets,
            sampling=invocation.sampling_context(applied_overrides),
        )
        if not isinstance(sampled_params, SampledParams):
            raise TypeError(f"{self.__class__.__name__}.sample_parameters must return SampledParams")
        return applied_overrides, sampled_params

    def _validate_sampled_params(
        self,
        sampled_params: SampledParams,
        targets: TargetSet | None,
        data: Mapping[str, Any],
    ) -> None:
        if targets is None:
            self._validate_uniform_params(sampled_params, data)
            return

        sampled_params.validate(
            targets,
            {
                name: required.difference(self._runtime_generated_params)
                for name, required in self._get_required_parameters_by_target().items()
                if name in targets.names
            },
            self.__class__.__name__,
        )

    def _validate_uniform_params(self, sampled_params: SampledParams, data: Mapping[str, Any]) -> None:
        """Validate required parameters when sampling returns no target-specific parameters."""
        required_by_target = self._get_required_parameters_by_target()
        if not required_by_target:
            return

        missing_by_target = {
            name: required.difference(self._runtime_generated_params).difference(sampled_params.params)
            for name, required in required_by_target.items()
            if name in data and data[name] is not None
        }
        missing_by_target = {name: missing for name, missing in missing_by_target.items() if missing}
        if missing_by_target:
            raise ValueError(f"{self.__class__.__name__} missing required parameters: {missing_by_target}")

    def _get_required_parameters_by_target(self) -> dict[str, frozenset[str]]:
        transform_cls = type(self)
        target_names = tuple(self._key2func)
        cached = transform_cls.__dict__.get("_required_parameters_by_target_cache")
        if cached is None or cached[0] != target_names:
            required_by_target: dict[str, frozenset[str]] = {}
            for name, function in self._key2func.items():
                required = required_parameter_names(function)
                if required:
                    required_by_target[name] = required
            cached = (target_names, required_by_target)
            type.__setattr__(transform_cls, "_required_parameters_by_target_cache", cached)
        return cached[1]

    def _should_apply_in_invocation(self, invocation: InvocationContext, *, force_apply: bool) -> bool:
        """Evaluates this leaf's probability against the root Python stream, avoiding configured mutable generators
        while concurrent calls execute on the same graph.
        """
        return force_apply or self.p >= 1.0 or invocation.py_random.random() < self.p

    def _apply_replay(
        self,
        state: TransformInvocationState | None,
        *,
        invocation: InvocationContext | None,
        **kwargs: Any,
    ) -> Any:
        """Applies recorded replay parameters without new sampling, preserving the optional caller-local observation
        behavior of sampled leaves.
        """
        if not self.applied_in_replay:
            return kwargs
        if self._can_bypass_tensor_route(kwargs, invocation):
            return self._apply_replay_in_route(state, **kwargs)
        route = self._tensor_fallback_route(kwargs)
        if self._uses_tensor_fallback(route):
            result = self._apply_replay_in_route(state, **self._enter_tensor_fallback(kwargs, route))
            return self._restore_tensor_fallback(result, route)
        return self._apply_replay_in_route(state, **kwargs)

    def _apply_replay_in_route(self, state: TransformInvocationState | None, **kwargs: Any) -> Any:
        sampled_params = SampledParams.from_dict(deepcopy(self._replay_params))
        targets = self._build_target_set(kwargs)
        self._validate_spatial_targets(targets)
        sampled_params.validate(
            targets,
            {
                name: required_parameter_names(function).difference(self._runtime_generated_params)
                for name, function in self._key2func.items()
                if any(view.name == name for view in targets.ordered)
            },
            self.__class__.__name__,
        )
        if state is not None:
            state.params = sampled_params.to_dict()
        return self.apply_with_params(sampled_params, **kwargs)

    def get_applied_params(self) -> dict[str, Any]:
        """Returns the parameters that were used in the last transform application; returns empty
        dict if transform was not applied.
        """
        return self.params

    def get_applied_config(self) -> dict[str, Any]:
        """Return the constructor-valid configuration captured by the latest successful application, for JSON
        transport and public pipeline reconstruction.

        The result is empty when the transform was not applied. Realized values written by
        sample_parameters replaces its source constructor policy,
        and aliases expose the fields of their canonical replay class. Values are JSON-safe.
        """
        return self.applied_config

    def get_applied_replay_class(self) -> "type[BasicTransform]":
        """Select the public constructor represented by this transform's applied record, allowing semantic aliases
        to replay through canonical implementations.

        Most transforms replay as their own class. Semantic aliases declare their canonical
        implementation through `_applied_replay_class` so replay does not re-enter deprecated
        constructors.
        """
        replay_cls = self._applied_replay_class
        return type(self) if replay_cls is None else replay_cls

    @classmethod
    def _get_valid_config_keys(cls) -> frozenset[str]:
        if (
            "_valid_applied_config_keys_cache" not in cls.__dict__
            or cls.__dict__["_valid_applied_config_keys_cache"] is None
        ):
            signature = inspect.signature(cls.__init__)
            valid_keys = frozenset(
                name
                for name, parameter in signature.parameters.items()
                if name not in {"self", "strict"}
                and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
            )
            cls._valid_applied_config_keys_cache = valid_keys
            return valid_keys

        cached_keys = cls._valid_applied_config_keys_cache
        if cached_keys is None:
            msg = f"Valid applied config key cache was not initialized for {cls.__name__}"
            raise RuntimeError(msg)
        return cached_keys

    def _build_applied_config(self, *, state: TransformInvocationState, overrides: Mapping[str, Any]) -> None:
        """Merge constructor state with values realized by the latest application, then retain only fields accepted
        by the selected replay class.

        Merge base and public transform state with realized overrides, validate against the
        selected replay class, and discard fields that are not part of that class's public
        constructor.
        """
        replay_cls = self.get_applied_replay_class()
        valid_keys = replay_cls._get_valid_config_keys()  # noqa: SLF001 - replay classes share this base contract.

        if overrides:
            invalid = set(overrides) - valid_keys
            if invalid:
                msg = (
                    f"{self.__class__.__name__}.applied_config has keys {invalid} "
                    f"that are not constructor params for {replay_cls.__name__}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
                raise ValueError(msg)

        config = self.get_base_init_args()
        config.update(self.get_transform_init_args())
        config.update(overrides)

        state.applied_config = {key: value for key, value in config.items() if key in valid_keys}

    def inverse(self) -> "BasicTransform":
        """Return a new transform that is the mathematical inverse of this one. Useful for TTA to
        revert deterministic transforms. Override in subclasses.

        Useful for TTA (Test-Time Augmentation): apply a deterministic transform to an image
        before inference, then apply its inverse to the predicted mask to bring it back to
        the original image space.

        Only transforms that override `inverse()` support this operation, typically
        group-based transforms with a fixed `group_element` (e.g., D4, RandomRotate90,
        HorizontalFlip, VerticalFlip, Transpose).

        Raises:
            NotImplementedError: If the transform does not support inversion.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse(). "
            "Only transforms that override `inverse()` can be used for TTA inversion.",
        )

    def _dispatch_functions(self) -> dict[str, Callable[..., Any]]:
        invocation = get_current_invocation()
        if invocation is None or invocation.frame_state is None:
            return self._key2func
        return invocation.frame_state.functions(self.invocation_key, self._key2func)

    def apply_with_uniform_params(self, params: Mapping[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Apply one parameter mapping to every target without target-specific values."""
        return {
            key: self._key2func[key](arg, **params) if key in self._key2func and arg is not None else arg
            for key, arg in kwargs.items()
        }

    def apply_with_params(
        self,
        sampled_params: SampledParams,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dispatch sampled parameters to target handlers and retain explicitly returned binding IDs."""
        invocation = get_current_invocation()
        binding = None if invocation is None else invocation.frame_state
        functions = self._key2func if binding is None else binding.functions(self.invocation_key, self._key2func)
        if not self._runtime_binding_params:
            return {
                key: functions[key](arg, **sampled_params.params_for(key))
                if key in functions and arg is not None
                else arg
                for key, arg in kwargs.items()
            }
        result: dict[str, Any] = {}
        ownership: dict[str, TargetResult[Any]] = {}
        for key, arg in kwargs.items():
            handler = functions.get(key)
            if handler is None or arg is None:
                result[key] = arg
                continue
            params = sampled_params.params_for(key)
            metadata = self._runtime_binding_params if binding is None else binding.parameters(key)
            params = {**params, **metadata}
            value = handler(arg, **params)
            if isinstance(value, TargetResult):
                result[key] = value.values
                ownership[key] = value
            else:
                result[key] = value
        if ownership and binding is not None:
            binding.accept(ownership)
        return result

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> "BasicTransform":
        """Set transform to be deterministic. When True, params are saved under save_key for
        replay (e.g. TTA). Returns self for chaining.
        """
        if save_key == "params":
            msg = "params save_key is reserved"
            raise KeyError(msg)

        self.deterministic = flag
        if self.deterministic and self.targets_as_params:
            warn(
                self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                " because its' params depend on targets.",
                stacklevel=2,
            )
        self.save_key = save_key
        return self

    def __repr__(self) -> str:
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return f"{self.__class__.__name__}({format_args(state)})"

    def apply(self, img: ImageType, *args: Any, **params: Any) -> ImageType:
        """Applies an image with invocation-supplied parameters. Subclasses implement pixel kernels while sampling stays
        outside execution.
        """
        raise NotImplementedError

    @staticmethod
    def _apply_to_batch(
        batch: np.ndarray,
        apply_fn: Callable[[np.ndarray], np.ndarray],
        *,
        ensure_contiguous: bool = False,
    ) -> np.ndarray:
        """Apply a function to each element in a batch with pre-allocation. Uses first element to
        determine output shape; avoids per-call allocation.

        Args:
            batch (np.ndarray): Input batch array of shape (N, ...)
            apply_fn (Callable[[np.ndarray], np.ndarray]): Function to apply to each element
            ensure_contiguous (bool): Whether to ensure C-contiguous output

        Returns:
            np.ndarray: Transformed batch array.

        """
        if len(batch) == 0:
            return np.require(batch, requirements=["C_CONTIGUOUS"]) if ensure_contiguous else batch

        # Process first element to determine output shape
        first_result = apply_fn(batch[0])

        # Single element case
        if len(batch) == 1:
            result = first_result[np.newaxis]
            return np.require(result, requirements=["C_CONTIGUOUS"]) if ensure_contiguous else result

        # Pre-allocate for remaining elements based on first result
        result_shape = (len(batch), *first_result.shape)
        result = np.empty(result_shape, dtype=first_result.dtype)
        result[0] = first_result

        for i in range(1, len(batch)):
            result[i] = apply_fn(batch[i])

        return np.require(result, requirements=["C_CONTIGUOUS"]) if ensure_contiguous else result

    @staticmethod
    def _apply_to_tensor_batch(
        batch: torch.Tensor,
        apply_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if len(batch) == 0:
            return batch

        first = apply_fn(batch[0])
        if len(batch) == 1:
            return first.unsqueeze(0)

        result = torch.empty((len(batch), *first.shape), dtype=first.dtype, device=first.device)
        result[0].copy_(first)
        for index in range(1, len(batch)):
            result[index].copy_(apply_fn(batch[index]))
        return result

    @staticmethod
    def _apply_to_batch_same_shape(
        batch: np.ndarray,
        apply_fn: Callable[[np.ndarray], np.ndarray],
        *,
        ensure_contiguous: bool = False,
    ) -> np.ndarray:
        """Apply a function to each batch element with pre-allocation when every output preserves
        the input element shape and dtype.

        Args:
            batch (np.ndarray): Input batch array of shape (N, ...)
            apply_fn (Callable[[np.ndarray], np.ndarray]): Function to apply to each element
            ensure_contiguous (bool): Whether to ensure C-contiguous output

        Returns:
            np.ndarray: Transformed batch array.

        """
        if len(batch) == 0:
            return np.require(batch, requirements=["C_CONTIGUOUS"]) if ensure_contiguous else batch

        result = np.empty_like(batch)

        for i, item in enumerate(batch):
            result[i] = apply_fn(item)

        return np.require(result, requirements=["C_CONTIGUOUS"]) if ensure_contiguous else result

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType | TargetResult[ImageType]:
        """Apply transform on images. Input shape (N, H, W, C); uses _apply_to_batch with per-image
        apply. Returns same format. Batch API.

        Args:
            images (ImageType): Input images as numpy array of shape:
                - (num_images, height, width, channels)
                - (num_images, height, width) for grayscale
            *args (Any): Additional positional arguments
            **params (Any): Additional parameters specific to the transform

        Returns:
            ImageType | TargetResult[ImageType]: Transformed array, optionally with explicitly updated binding IDs.

        """
        return self._apply_to_batch(images, lambda img: self.apply(img, *args, **params))

    def apply_to_volume(self, volume: VolumeType, *args: Any, **params: Any) -> VolumeType:
        """Apply transform slice by slice to a volume. Delegates to apply_to_images so each slice
        is transformed consistently. Single volume.

        Args:
            volume (VolumeType): Input volume of shape (depth, height, width) or (depth, height, width, channels)
            *args (Any): Additional positional arguments
            **params (Any): Additional parameters specific to the transform

        Returns:
            VolumeType: Transformed volume as numpy array in the same format as input

        """
        return cast("VolumeType", self.apply_to_images(volume, *args, **params))

    def apply_to_volumes(
        self,
        volumes: Annotated[NDArray[np.generic], torch.Tensor],
        *args: Any,
        **params: Any,
    ) -> NDArray[np.generic] | torch.Tensor:
        """Apply the existing single-volume route to each item of an NDHWC or NCDHW collection."""
        volume_method = self._target_apply_methods.get("volume", _TARGET_APPLY_METHODS["volume"])
        apply_fn = getattr(self, volume_method)
        if isinstance(volumes, np.ndarray):
            return self._apply_to_batch(volumes, lambda volume: apply_fn(volume, *args, **params))
        return self._apply_to_tensor_batch(volumes, lambda volume: apply_fn(volume, *args, **params))

    def update_transform_params(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        invocation: InvocationContext | None = None,
        targets: TargetSet | None = None,
    ) -> dict[str, Any]:
        """Update parameters with input shape and transform-specific settings (interpolation, fill,
        fill_mask, bbox type) before data-aware parameter sampling.

        Args:
            params (dict[str, Any]): Parameters to be updated
            data (dict[str, Any]): Input data dictionary containing images and volume data
            invocation (InvocationContext | None): Active call state, when this transform runs in a Compose graph.
            targets (TargetSet | None): Prebuilt invocation-local target descriptors, when available.

        Returns:
            dict[str, Any]: Updated parameters dictionary with shape and transform-specific params

        """
        if targets is None:
            shape = self._extract_shared_shape_from_data(data)
            if shape is not None:
                params["shape"] = shape
        else:
            for view in targets.ordered:
                if view.descriptor.shape is not None:
                    shape = view.descriptor.shape
                    shape_indices = _SHARED_SHAPE_INDICES.get(view.descriptor.layout)
                    if shape_indices is not None:
                        shared_shape = tuple(shape[index] for index in shape_indices)
                    elif view.canonical_type in _BATCH_SHARED_SHAPE_TARGETS:
                        shared_shape = shape[1:]
                    else:
                        shared_shape = shape
                    params["shape"] = shared_shape
                    break

        bbox_processor = None if invocation is None else invocation.get_processor("bboxes")
        if isinstance(bbox_processor, BboxProcessor):
            params["bbox_type"] = bbox_processor.params.bbox_type

        # Add transform-specific params
        self._add_transform_specific_params(params)

        return params

    def _build_target_set(self, data: Mapping[str, Any]) -> TargetSet:
        functions = self._dispatch_functions()
        aliases = self._additional_targets
        invocation = get_current_invocation()
        if invocation is not None and invocation.frame_state is not None:
            aliases = {**aliases, **invocation.frame_state.routes}
        return TargetSet.from_data(data, {name: aliases.get(name, name) for name in functions})

    def _validate_spatial_targets(self, targets: TargetSet) -> None:
        if self._sampling_spatial_rank is not None:
            targets.aligned_spatial_shape(self._sampling_spatial_rank)

    @staticmethod
    def _shared_shape_from_data_key(key: str, value: Any) -> tuple[int, ...]:
        if key in {"volume", "mask3d"}:
            _, height, width = get_volume_shape(value)
            channel_count = value.shape[0] if isinstance(value, torch.Tensor) else value.shape[-1]
            return (
                (height, width, channel_count)
                if isinstance(value, torch.Tensor) or value.ndim == 4
                else (height, width)
            )
        if key in {"volumes", "masks3d"}:
            _, height, width = get_volumes_shape(value)
            channel_count = 1
            if value.ndim == 5:
                channel_count = value.shape[1] if isinstance(value, torch.Tensor) else value.shape[-1]
            return height, width, channel_count
        if not isinstance(value, torch.Tensor):
            return value.shape if key in {"image", "mask"} else value.shape[1:]
        if key in {"image", "mask"}:
            return value.shape[1], value.shape[2], value.shape[0]
        if key in {"images", "masks"}:
            return value.shape[2], value.shape[3], value.shape[1]
        return value.shape[1:]

    def _extract_shared_shape_from_data(self, data: dict[str, Any]) -> tuple[int, ...] | None:
        """Return the shared shape needed by the no-sampler execution fast path.

        Data-dependent samplers receive target descriptors and must not call this helper.
        """
        for key in ("image", "images", "volume", "volumes", "mask", "masks", "mask3d", "masks3d"):
            value = data.get(key)
            if value is not None:
                return self._shared_shape_from_data_key(key, value)
        return None

    def _add_transform_specific_params(self, params: dict[str, Any]) -> None:
        """Add transform-specific parameters to params dict (interpolation, fill, fill_mask).
        Called from update_transform_params. Mutates params in place.
        """
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill"):
            params["fill"] = self.fill
        if hasattr(self, "fill_mask"):
            params["fill_mask"] = self.fill_mask

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Generates parameters and stores realized replay policy in call-local data, never retaining per-sample values
        on transform instances.

        Override this method in every transform that samples data-dependent parameters. `params` contains execution
        parameters, `data` contains all invocation data, and `targets` describes active transform targets.
        Return parameters and target-specific values consumed by `apply_*` methods. Write constructor-valid
        realized policy values to `sampling.applied_overrides`. The default supports deterministic transforms with no
        sampled parameters.
        """
        del params, data, targets, sampling
        return SampledParams(params={})

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """Build the dispatch table from the transform's declared targets."""
        raw_targets = (self._targets,) if isinstance(self._targets, Targets | str) else self._targets
        handlers: dict[str, Callable[..., Any]] = {}
        for target in raw_targets:
            name = _TARGET_NAMES[target] if isinstance(target, Targets) else target
            if name in handlers:
                raise TypeError(f"{self.__class__.__name__} declares target {name!r} more than once")
            method_name = self._target_apply_methods.get(name, _TARGET_APPLY_METHODS.get(name, f"apply_to_{name}"))
            handler = getattr(self, method_name, None)
            if handler is None:
                raise TypeError(f"{self.__class__.__name__} declares {name!r} without {method_name}()")
            implementation = getattr(handler, "__func__", handler)
            if name == "user_data" and implementation is BasicTransform.apply_to_user_data:
                raise TypeError("user_data must have a transform-specific apply_to_user_data() implementation")
            handlers[name] = handler
        return handlers

    def apply_to_user_data(self, data: Any, **params: Any) -> Any:
        """Handle `user_data` for a custom transform that explicitly declares it in `_targets`.

        Built-in transforms leave this key unchanged. A custom transform can add the string
        `"user_data"` to its `_targets` and override this method to update annotations such as captions.

        Args:
            data (Any): Arbitrary user-defined data of any type.
            **params (Any): Transform parameters (same as passed to other apply_* methods).

        Returns:
            Any: The (optionally modified) user data. Must return the same type as the input.

        Examples:
            >>> import albumentations as A
            >>> class FlipAwareTransform(A.HorizontalFlip):
            ...     '''Flip images and update captions.
            ...
            ...     Targets:
            ...         image, images, mask, masks, bboxes, keypoints, volume, mask3d, user_data
            ...     '''
            ...     _targets = (*A.HorizontalFlip._targets, "user_data")
            ...
            ...     def apply_to_user_data(self, data: dict, **params) -> dict:
            ...         return {"caption": data["caption"].replace("left", "right")}

        """
        return data

    def _set_keys(self) -> None:
        """Build runtime dispatch exclusively from the effective `_targets` declaration."""
        self._key2func = self.targets
        self._available_keys = set(self._key2func)

    @property
    def available_keys(self) -> set[str]:
        """Returns set of available keys (target names this transform can process). Includes
        built-in targets and add_targets additions.
        """
        return self._available_keys

    def add_targets(self, additional_targets: dict[str, str]) -> None:
        """Register additional targets transformed like an existing one (e.g. {'image2': 'image'}).
        Need at least 'image' in pipeline.

        Args:
            additional_targets (dict[str, str]): keys - new target name, values
                - old target name. ex: {'image2': 'image'}

        """
        for k, v in additional_targets.items():
            if k in self._additional_targets and v != self._additional_targets[k]:
                raise ValueError(
                    f"Trying to overwrite existed additional targets. "
                    f"Key={k} Exists={self._additional_targets[k]} New value: {v}",
                )
            self._additional_targets[k] = v
            if v in self._available_keys:
                self._key2func[k] = self._key2func[v]
                self._available_keys.add(k)

    @property
    def targets_as_params(self) -> list[str]:
        """Targets used to get params dependent on targets. Used to check input has all required
        targets before apply. Override to list keys (e.g. ['image']).
        """
        return []

    @classmethod
    def get_class_fullname(cls) -> str:
        """Get the full qualified name of the class. Returns shortest fullname for serialization
        (e.g. albumentations.HorizontalFlip).

        Returns:
            str: The shortest class fullname.

        """
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the transform class is serializable. True for all registered transforms; used
        by serialization to skip non-serializable classes.

        Returns:
            bool: True if the class is serializable, False otherwise.

        """
        return True

    def get_base_init_args(self) -> dict[str, Any]:
        """Returns base init args (e.g. p) for serialization. Subclasses may override
        to add more; merged into to_dict_private output.
        """
        return {"p": self.p}

    def get_transform_init_args(self) -> dict[str, Any]:
        """Get transform initialization arguments for serialization. Returns dict of init param
        names and values, excluding empty containers and seed.

        Returns a dictionary of parameter names and their values, excluding parameters
        that are not actually set on the instance or that shouldn't be serialized.
        """
        arg_names = self.get_transform_init_args_names()

        args = {}
        for name in arg_names:
            # Only include parameters that are actually set as instance attributes
            # and have non-default values
            if hasattr(self, name):
                value = getattr(self, name)
                # Skip attributes that are basic containers with no content
                if not (isinstance(value, (list, dict, tuple, set)) and len(value) == 0):
                    args[name] = value

        # Remove seed explicitly (it's not meant to be serialized)
        args.pop("seed", None)

        return args

    def to_dict_private(self) -> dict[str, Any]:
        """Returns a dictionary representation of the transform for serialization.
        Excludes internal parameters; includes __class_fullname__ and init args.
        """
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())

        transform_args = self.get_transform_init_args()

        # Add transform args to state
        state.update(transform_args)

        # Remove strict from serialization
        state.pop("strict", None)

        return state


class DualTransform(BasicTransform):
    """Base class for spatial transforms that apply to images and masks.

    Targets:
        image, images, mask, masks, volume, volumes, mask3d, masks3d

    Concrete subclasses declare bbox and keypoint support when they implement those routes.

    Class Attributes:
        _supported_bbox_types (set[str]): Set of supported bounding box types.
            Valid values: {"hbb"} for axis-aligned boxes only, {"hbb", "obb"} for both axis-aligned
            and oriented boxes. Default: {"hbb"}. Transforms that support OBB should override this.

    Methods:
        apply(img: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to the image.

            img: Input image of shape (H, W, C).
            **params: Additional parameters specific to the transform.

            Returns Transformed image of the same shape as input.

        apply_to_images(images: ImageType, **params: Any) -> ImageType:
            Apply the transform to multiple images.

            images: Input images of shape (N, H, W, C).
            **params: Additional parameters specific to the transform.

            Returns Transformed images in the same format as input.

        apply_to_mask(mask: ImageType, **params: Any) -> ImageType:
            Apply the transform to a mask.

            mask: Input mask of shape (H, W), (H, W, C) for multi-channel masks
            **params: Additional parameters specific to the transform.

            Returns Transformed mask in the same format as input.

        apply_to_masks(masks: ImageType, **params: Any) -> ImageType:
            Apply the transform to multiple masks.

            masks: Array of shape (N, H, W) or (N, H, W, C) where N is number of masks
            **params: Additional parameters specific to the transform.
            Returns Transformed masks in the same format as input.

        apply_to_keypoints(keypoints: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to keypoints.

            keypoints: Array of shape (N, 2+) where N is the number of keypoints.
                **params: Additional parameters specific to the transform.
            Returns Transformed keypoints array of shape (N, 2+).

        apply_to_bboxes(bboxes: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to bounding boxes.

            bboxes: Array of shape (N, 4+) where N is the number of bounding boxes,
                    and each row is in the format [x_min, y_min, x_max, y_max].
            **params: Additional parameters specific to the transform.

            Returns Transformed bounding boxes array of shape (N, 4+).

        apply_to_volume(volume: VolumeType, **params: Any) -> VolumeType:
            Apply the transform to a volume.

            volume: Input volume of shape (D, H, W, C).
            **params: Additional parameters specific to the transform.

            Returns Transformed volume of the same shape as input.

        apply_to_mask3d(mask: VolumeType, **params: Any) -> VolumeType:
            Apply the transform to a 3D mask.

            mask: Input 3D mask of shape (D, H, W) or (D, H, W, C)
            **params: Additional parameters specific to the transform.

            Returns Transformed 3D mask in the same format as input.

    Note:
        - All `apply_*` methods should maintain the input shape and format of the data.
        - When applying transforms to masks, ensure that discrete values (e.g., class labels) are preserved.
        - For keypoints and bounding boxes, the transformation should maintain their relative positions
            with respect to the transformed image.
        - The difference between `apply_to_mask` and `apply_to_masks` is mainly in how they handle 3D arrays:
            `apply_to_mask` treats a 3D array as a multi-channel mask, while `apply_to_masks` treats it as
            multiple single-channel masks.

    """

    _targets: tuple[Targets | str, ...] | Targets | str = (
        Targets.IMAGE,
        Targets.IMAGES,
        Targets.MASK,
        Targets.MASKS,
        Targets.VOLUME,
        Targets.VOLUMES,
        Targets.MASK3D,
        Targets.MASKS3D,
    )
    _target_apply_methods: ClassVar[Mapping[str, str]] = {"volume": "apply_to_images"}
    _sampling_spatial_rank = 2

    _supported_bbox_types: frozenset[str] = frozenset({"hbb"})  # Default: only axis-aligned boxes
    _semantic_mask_label_mappings: dict[str, dict[int, int]]
    _semantic_mask_uint8_luts: dict[str, NDArray[np.uint8]]

    def __init__(self, p: float = 0.5, **kwargs: Any):
        super().__init__(p=p, **kwargs)
        self._semantic_mask_label_mappings = {}
        self._semantic_mask_uint8_luts = {}

    def set_semantic_mask_label_mappings(self, mappings: dict[str, dict[int, int]]) -> None:
        """Set transform-aware semantic-mask label mappings, discard no-op entries, and compile reusable
        uint8 lookup tables once per instance.
        """
        compiled_mappings = {
            transform_name: {
                source_label: target_label
                for source_label, target_label in mapping.items()
                if source_label != target_label
            }
            for transform_name, mapping in mappings.items()
        }
        compiled_uint8_luts: dict[str, NDArray[np.uint8]] = {}
        for transform_name, mapping in compiled_mappings.items():
            if mapping and all(0 <= label <= 255 for pair in mapping.items() for label in pair):
                lut = np.arange(256, dtype=np.uint8)
                for source_label, target_label in mapping.items():
                    lut[source_label] = target_label
                compiled_uint8_luts[transform_name] = lut
        self._semantic_mask_label_mappings = compiled_mappings
        self._semantic_mask_uint8_luts = compiled_uint8_luts

    def apply_to_keypoints(self, keypoints: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        msg = f"Method apply_to_keypoints is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_bboxes(self, bboxes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        raise NotImplementedError(f"BBoxes not implemented for {self.__class__.__name__}")

    def apply_to_mask(self, mask: ImageType, *args: Any, **params: Any) -> ImageType:
        return self.apply(mask, *args, **params)

    def apply_to_masks(self, masks: StackedMasks4D, *args: Any, **params: Any) -> StackedMasks4D:
        """Apply the per-row mask transform to a `StackedMasks4D` `(N, H, W, C)` and return a
        stack that upholds the row-alignment contract with bboxes and keypoints.

        Row-alignment contract (enforced by `Compose._resync_instance_ids` after every
        transform when `instance_binding` is active):

        - `len(returned_masks) == len(returned_bboxes)` must hold simultaneously with the
          same call's `apply_to_bboxes` return.
        - Row `i` of the returned stack must describe the same instance as row `i` of the
          returned bboxes (so `_bbox_instance_id == arange(N)` after Compose's resync).
        - If your transform drops bbox rows from `apply_to_bboxes` (e.g. min-area or
          out-of-frame culling), it MUST drop the corresponding mask rows from
          `apply_to_masks`. Compose's bbox-processor mirror covers the case where
          BboxProcessor is the SOLE filter; transform-internal filters need their own
          shared keep-mask plumbed via `sample_parameters` (see Mosaic /
          CopyAndPaste for the canonical pattern).
        - The default per-row implementation below preserves alignment for transforms
          whose `apply_to_mask` is total (no row drops).

        Violating this contract surfaces as a `RuntimeError` from `_resync_instance_ids`.
        """
        if masks.size == 0:
            return masks
        return cast(
            "StackedMasks4D",
            self._apply_to_batch(masks, lambda mask: self.apply_to_mask(mask, *args, **params)),
        )

    def apply_to_mask3d(self, mask3d: VolumeType, *args: Any, **params: Any) -> VolumeType:
        return self._apply_to_batch(mask3d, lambda mask: self.apply_to_mask(mask, *args, **params))

    def apply_to_masks3d(
        self,
        masks3d: Annotated[NDArray[np.generic], torch.Tensor],
        *args: Any,
        **params: Any,
    ) -> NDArray[np.generic] | torch.Tensor:
        """Apply the existing single-mask3d route to each item in a collection."""
        apply_fn = getattr(self, self._target_apply_methods.get("mask3d", _TARGET_APPLY_METHODS["mask3d"]))
        if isinstance(masks3d, np.ndarray):
            return self._apply_to_batch(masks3d, lambda mask3d: apply_fn(mask3d, *args, **params))
        return self._apply_to_tensor_batch(masks3d, lambda mask3d: apply_fn(mask3d, *args, **params))

    def _get_label_transform_name(self, **params: Any) -> str | None:
        """Get the transform name to use for label mapping. For most transforms returns class
        name; for D4/SquareSymmetry maps group_element to base name.

        For most transforms, this is just the class name. For D4/SquareSymmetry,
        we map the group element to the corresponding base transform name.

        Args:
            **params (Any): Transform parameters, may contain group_element for D4 transforms

        Returns:
            str | None: Transform name to use for label mapping, or None if no mapping should be applied

        """
        class_name = self.__class__.__name__

        # Handle D4 and SquareSymmetry transforms (including subclasses)
        if class_name in ("D4", "SquareSymmetry") or any(
            base.__name__ in ("D4", "SquareSymmetry") for base in self.__class__.__mro__
        ):
            group_element = params.get("group_element", "e")
            # Map D4 group elements to base transform names
            d4_to_base_transform = {
                "h": "HorizontalFlip",
                "v": "VerticalFlip",
                "t": "Transpose",
                "hvt": "Transpose",  # Anti-diagonal is also a transpose-like operation
                "e": None,  # Identity - no label swapping
                "r90": None,  # Rotations don't change semantic labels
                "r180": None,
                "r270": None,
            }
            return d4_to_base_transform.get(group_element)

        # Only parity-changing transforms should apply label mappings
        parity_changing_transforms = {"HorizontalFlip", "VerticalFlip", "Transpose"}
        return next(
            (base.__name__ for base in self.__class__.__mro__ if base.__name__ in parity_changing_transforms),
            None,
        )

    def _apply_label_mapping_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Apply label mapping by reordering entire keypoint rows. For keypoint regression, row
        index encodes semantics; flip/transpose swap rows via mapping.

        For keypoint regression tasks, the row index encodes semantic meaning
        (e.g., row 0 = left eye heatmap). On transforms like HorizontalFlip,
        we need to swap entire rows, not just relabel them.

        Args:
            keypoints (np.ndarray): Keypoints array with potential label columns attached
            **params (Any): Transform parameters

        Returns:
            np.ndarray: Keypoints array with rows reordered based on label mapping

        """
        processor = self.get_processor("keypoints")
        if not processor or not hasattr(processor, "encoded_label_mappings"):
            return keypoints

        if not processor.params.label_fields or keypoints.size == 0 or keypoints.shape[1] <= 5:
            return keypoints

        transform_name = self._get_label_transform_name(**params)
        if transform_name is None or transform_name not in processor.encoded_label_mappings:
            return keypoints

        # Only copy if we actually have mappings to apply
        field_mappings = processor.encoded_label_mappings[transform_name]
        if not field_mappings:
            return keypoints

        return self._swap_keypoint_rows_by_labels(keypoints, processor.params.label_fields, field_mappings)

    @staticmethod
    def _remap_semantic_mask_labels(
        mask: NDArray[np.generic] | torch.Tensor,
        mapping: dict[int, int],
        uint8_lut: NDArray[np.uint8] | None,
    ) -> NDArray[np.generic] | torch.Tensor:
        is_empty = mask.size == 0 if isinstance(mask, np.ndarray) else mask.numel() == 0
        if is_empty:
            return mask
        if isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and uint8_lut is not None:
            return sz_lut(cast("NDArray[np.uint8]", mask), uint8_lut, inplace=False)

        if isinstance(mask, torch.Tensor):
            result = mask.clone()
            for source_label, target_label in mapping.items():
                target = torch.tensor(target_label, dtype=mask.dtype, device=mask.device)
                torch.where(mask == source_label, target, result, out=result)
            return result

        result = mask.copy()
        for source_label, target_label in mapping.items():
            result[mask == source_label] = target_label
        return result

    def _apply_label_mapping_to_semantic_masks(self, data: dict[str, Any], **params: Any) -> dict[str, Any]:
        transform_name = self._get_label_transform_name(**params)
        if transform_name is None:
            return data
        mapping = self._semantic_mask_label_mappings.get(transform_name)
        if not mapping:
            return data
        uint8_lut = self._semantic_mask_uint8_luts.get(transform_name)

        for data_name, value in data.items():
            canonical_name = self._additional_targets.get(data_name, data_name)
            if (
                data_name in self._key2func
                and canonical_name in {"mask", "masks", "mask3d", "masks3d"}
                and isinstance(value, (np.ndarray, torch.Tensor))
            ):
                data[data_name] = self._remap_semantic_mask_labels(value, mapping, uint8_lut)
        return data

    def _swap_keypoint_rows_by_labels(
        self,
        keypoints: np.ndarray,
        label_fields: Sequence[str],
        field_mappings: dict[str, dict[int, int]],
    ) -> np.ndarray:
        """Swap keypoint rows based on label mappings. Used when transform changes left/right or
        similar; swaps entire rows so coords and labels stay consistent.

        Args:
            keypoints (np.ndarray): Keypoints array with label columns
            label_fields (Sequence[str]): List of label field names
            field_mappings (dict[str, dict[int, int]]): Mapping of field names to label swaps

        Returns:
            np.ndarray: Keypoints array with rows swapped

        """
        result = keypoints.copy()
        label_col_start = keypoints.shape[1] - len(label_fields)
        instance_id_col_idx = None
        if "_kp_instance_id" in label_fields:
            candidate_col_idx = label_col_start + label_fields.index("_kp_instance_id")
            if candidate_col_idx < keypoints.shape[1]:
                instance_id_col_idx = candidate_col_idx

        invocation = get_current_invocation()
        if instance_id_col_idx is None and invocation is not None and invocation.frame_state is not None:
            instance_id_col_idx = invocation.frame_state.frame_columns.get("keypoints")

        # For each label field with mapping, perform row swapping
        for i, label_field in enumerate(label_fields):
            if label_field in field_mappings:
                col_idx = label_col_start + i
                if col_idx < keypoints.shape[1]:
                    mapping = field_mappings[label_field]
                    if mapping:  # Only process if mapping is not empty
                        result = self._apply_single_field_mapping(result, col_idx, mapping, instance_id_col_idx)
                        # Only apply mapping for the first label field that has mappings
                        break

        return result

    def _apply_single_field_mapping(
        self,
        keypoints: np.ndarray,
        col_idx: int,
        mapping: dict[int, int],
        instance_id_col_idx: int | None = None,
    ) -> np.ndarray:
        """Apply label mapping to a single label column. Swaps rows for paired labels or updates
        unpaired; used internally by _swap_keypoint_rows_by_labels.

        Args:
            keypoints (np.ndarray): Keypoints array
            col_idx (int): Column index of the label field
            mapping (dict[int, int]): Label swap mapping
            instance_id_col_idx (int | None): Optional column that keeps bound instance ids. When provided,
                row swaps are constrained to each instance-id group.

        Returns:
            np.ndarray: Keypoints array with rows swapped

        """
        if instance_id_col_idx is not None:
            for instance_id in np.unique(keypoints[:, instance_id_col_idx]):
                instance_indices = np.where(keypoints[:, instance_id_col_idx] == instance_id)[0]
                keypoints[instance_indices] = self._apply_single_field_mapping(
                    keypoints[instance_indices].copy(),
                    col_idx,
                    mapping,
                )
            return keypoints

        col_data = keypoints[:, col_idx].astype(int)
        processed_labels = set()

        for from_label, to_label in mapping.items():
            if from_label in processed_labels or to_label in processed_labels:
                continue

            from_indices = np.where(col_data == from_label)[0]
            to_indices = np.where(col_data == to_label)[0]

            # If both labels exist in data, swap entire rows
            if len(from_indices) > 0 and len(to_indices) > 0:
                # Swap entire rows (coordinates + all labels)
                temp_rows = keypoints[from_indices].copy()
                keypoints[from_indices] = keypoints[to_indices]
                keypoints[to_indices] = temp_rows
                processed_labels.add(from_label)
                processed_labels.add(to_label)
            # If only from_label exists (unpaired), just update its label
            elif len(from_indices) > 0:
                keypoints[from_indices, col_idx] = to_label
                processed_labels.add(from_label)

        return keypoints

    def apply_with_params(self, sampled_params: SampledParams, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Apply a dual transform with its parameters, including configured keypoint and transform-aware
        semantic-mask label mappings.
        """
        res = super().apply_with_params(sampled_params, *args, **kwargs)

        if "keypoints" in res and res["keypoints"] is not None:
            res["keypoints"] = self._apply_label_mapping_to_keypoints(
                res["keypoints"],
                **sampled_params.params_for("keypoints"),
            )

        if self._semantic_mask_label_mappings:
            res = self._apply_label_mapping_to_semantic_masks(res, **sampled_params.params)

        return res

    def apply_with_uniform_params(self, params: Mapping[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Apply one parameter mapping to every target while preserving dual-target label mappings."""
        res = super().apply_with_uniform_params(params, *args, **kwargs)
        if "keypoints" in res and res["keypoints"] is not None:
            res["keypoints"] = self._apply_label_mapping_to_keypoints(res["keypoints"], **params)
        if self._semantic_mask_label_mappings:
            res = self._apply_label_mapping_to_semantic_masks(res, **params)
        return res


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image (and volume) only. Does not transform masks, bboxes, or
    keypoints; use DualTransform for those.

    Targets:
        image, images, volume, volumes
    """

    _targets = (Targets.IMAGE, Targets.IMAGES, Targets.VOLUME, Targets.VOLUMES)


class NoOp(DualTransform):
    """Identity transform (does nothing). Passes all targets through unchanged. Use as placeholder
    or in conditional pipelines.

    Targets:
        image, images, mask, masks, bboxes, keypoints, volume, volumes, mask3d, masks3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Create transform pipeline with NoOp
        >>> transform = A.Compose([
        ...     A.NoOp(p=1.0),  # Always applied, but does nothing
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Verify nothing has changed
        >>> np.array_equal(image, transformed['image'])  # True
        >>> np.array_equal(mask, transformed['mask'])  # True
        >>> np.array_equal(bboxes, transformed['bboxes'])  # True
        >>> np.array_equal(keypoints, transformed['keypoints'])  # True
        >>> bbox_labels == transformed['bbox_labels']  # True
        >>> keypoint_labels == transformed['keypoint_labels']  # True
        >>>
        >>> # NoOp is often used as a placeholder or for testing
        >>> # For example, in conditional transforms:
        >>> condition = False  # Some condition
        >>> transform = A.Compose([
        ...     A.HorizontalFlip(p=1.0) if condition else A.NoOp(p=1.0)
        ... ])

    """

    _targets = ALL_TARGETS
    _target_apply_methods: ClassVar[Mapping[str, str]] = {"volume": "apply_to_volume"}
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})  # NoOp passes all bbox types

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return keypoints

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return bboxes

    def apply(self, img: Annotated[ImageType, torch.Tensor], **params: Any) -> ImageType:
        return img

    def apply_to_images(self, images: Annotated[ImageType, torch.Tensor], **params: Any) -> ImageType:
        return images

    def apply_to_mask(self, mask: Annotated[ImageType, torch.Tensor], **params: Any) -> ImageType:
        return mask

    def apply_to_masks(self, masks: Annotated[StackedMasks4D, torch.Tensor], **params: Any) -> StackedMasks4D:
        return masks

    def apply_to_volume(self, volume: Annotated[VolumeType, torch.Tensor], **params: Any) -> VolumeType:
        return volume

    def apply_to_mask3d(self, mask3d: Annotated[VolumeType, torch.Tensor], **params: Any) -> VolumeType:
        return mask3d

    def apply_to_volumes(self, volumes: Annotated[NDArray[np.generic], torch.Tensor], **params: Any) -> Any:
        return volumes

    def apply_to_masks3d(self, masks3d: Annotated[NDArray[np.generic], torch.Tensor], **params: Any) -> Any:
        return masks3d


class Transform3D(DualTransform):
    """Base class for 3D transforms that apply to volume data and mask3d.

    Concrete subclasses can declare keypoint support when they implement it.

    Targets:
        volume, volumes, mask3d, masks3d

    Target layouts:
        volume: NumPy array of shape (D, H, W, C)
        volumes: NumPy array of shape (N, D, H, W, C)
        mask3d: NumPy array of shape (D, H, W) or (D, H, W, C)
        masks3d: NumPy array of shape (N, D, H, W) or (N, D, H, W, C)
    """

    _targets: tuple[Targets | str, ...] | Targets | str = (
        Targets.VOLUME,
        Targets.VOLUMES,
        Targets.MASK3D,
        Targets.MASKS3D,
    )
    _target_apply_methods: ClassVar[Mapping[str, str]] = {"volume": "apply_to_volume"}
    _sampling_spatial_rank = 3

    def apply_to_volume(self, volume: VolumeType, *args: Any, **params: Any) -> VolumeType:
        """Apply transform to single 3D volume. Override in subclasses; input shape (D, H, W, C)
        or (D, H, W). Returns same shape and dtype.
        """
        raise NotImplementedError

    def apply_to_mask3d(self, mask3d: VolumeType, *args: Any, **params: Any) -> VolumeType:
        """Apply transform to a single 3D mask. Delegates to apply_to_volume. Input shape (D, H, W) or
        (D, H, W, C). Output shape unchanged. For VolumeTransform.
        """
        return self.apply_to_volume(mask3d, *args, **params)

    def _apply_label_mapping_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Remap keypoint label fields after 3D geometry while retaining transformed coordinates and row order so each
        record matches manual annotation.
        """
        processor = self.get_processor("keypoints")
        transform_name = self._get_label_transform_name(**params)
        if (
            not isinstance(processor, KeypointsProcessor)
            or not processor.params.label_fields
            or keypoints.size == 0
            or transform_name is None
        ):
            return keypoints

        field_mappings = processor.encoded_label_mappings.get(transform_name)
        if not field_mappings:
            return keypoints

        result = keypoints.copy()
        for label_offset, label_field in enumerate(processor.params.label_fields):
            mapping = field_mappings.get(label_field)
            column_index = keypoints.shape[1] - len(processor.params.label_fields) + label_offset
            if not mapping or column_index >= keypoints.shape[1]:
                continue
            source_values = keypoints[:, column_index]
            for source_label, target_label in mapping.items():
                result[source_values == source_label, column_index] = target_label
        return result


class VolumeOnlyTransform(BasicTransform):
    """Provide a base for volume-intensity transforms that leave masks and keypoints untouched, keeping acquisition
    artifacts separate from label geometry changes.

    Unlike `Transform3D`, subclasses do not dispatch to `mask3d` or
    `keypoints`. Compose therefore preserves those targets unchanged, which
    is appropriate for acquisition and photometric artifacts that do not alter
    label geometry.

    Targets:
        volume, volumes
    """

    _targets = (Targets.VOLUME, Targets.VOLUMES)

    def apply_to_volume(self, volume: VolumeType, *args: Any, **params: Any) -> VolumeType:
        raise NotImplementedError
