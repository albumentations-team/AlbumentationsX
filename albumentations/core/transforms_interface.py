"""Module containing base interfaces for all transform implementations. of alb

This module defines the fundamental transform interfaces that form the base hierarchy for
all transformation classes in Albumentations. It provides abstract classes and mixins that
define common behavior for image, keypoint, bounding box, and volumetric transformations.
The interfaces handle parameter validation, random state management, target type checking,
and serialization capabilities that are inherited by concrete transform implementations.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar, cast
from warnings import warn

import cv2
import numpy as np
import torch
from albucore import batch_transform, sz_lut
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from albumentations.core.bbox_utils import BboxProcessor
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
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
    ImageType,
    StackedMasks4D,
    Targets,
    VolumeType,
)
from .utils import format_args

__all__ = [
    "BasicTransform",
    "CustomTransformsApplyMixin",
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

    _targets: tuple[Targets, ...] | Targets  # targets that this transform can work on
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
    _supports_cpu_tensor: ClassVar[bool] = False
    _cpu_tensor_targets: ClassVar[frozenset[str] | None] = None
    _cpu_tensor_channels: ClassVar[frozenset[int] | None] = None
    _sampling_spatial_rank: ClassVar[int | None] = None
    _runtime_generated_params: ClassVar[frozenset[str]] = frozenset()
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
        removed_hooks = sorted(cls._removed_sampling_hooks.intersection(cls.__dict__))
        if removed_hooks:
            names = ", ".join(removed_hooks)
            raise TypeError(
                f"{cls.__name__} defines removed sampling hook(s): {names}. "
                "Implement sample_parameters(params, data, targets, sampling) instead.",
            )

    @property
    def supports_cpu_tensor(self) -> bool:
        """Return whether this transform can run CPU Tensor inputs directly, allowing Compose to keep data in Tensor
        form without a NumPy bridge.
        """
        return self._supports_cpu_tensor

    @property
    def cpu_tensor_targets(self) -> frozenset[str] | None:
        """Return canonical targets this transform handles directly as CPU Tensor inputs, allowing Compose to decide
        whether to keep a Tensor route.

        Compose bridges a pipeline through NumPy when its Tensor targets fall outside this set.
        """
        return self._cpu_tensor_targets

    @property
    def cpu_tensor_channels(self) -> frozenset[int] | None:
        """Return image channel counts covered by this Tensor capability, or `None` when
        the accepted Tensor route is independent of the channel count.
        """
        return self._cpu_tensor_channels

    def supports_cpu_tensor_targets(self, targets: frozenset[str]) -> bool:
        """Return whether accepted Tensor capability routes cover every caller-provided
        canonical target before Compose samples parameters or enters transform dispatch.

        `None` means that this transform's direct Tensor route is target agnostic. Transforms with a narrower route
        declare canonical target names explicitly so Compose can choose a NumPy bridge before sampling parameters.
        """
        return self.supports_cpu_tensor and (
            self._cpu_tensor_targets is None or targets.issubset(self._cpu_tensor_targets)
        )

    def supports_cpu_tensor_inputs(self, tensor_inputs: tuple[tuple[str, Any], ...]) -> bool:
        """Check every supplied target and image channel count to decide whether this transform can run them directly
        as CPU Tensor data.

        Compose uses a False result to select its one-time NumPy bridge for the full pipeline. Transform helpers never
        receive Tensor data through an ad hoc conversion.
        """
        targets = frozenset(target for target, _ in tensor_inputs)
        if not self.supports_cpu_tensor_targets(targets):
            return False
        if self._cpu_tensor_channels is None:
            return True
        return all(
            target not in {"image", "images", "volume"} or value.shape[0] in self._cpu_tensor_channels
            for target, value in tensor_inputs
        )

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
            # Get the list of valid arguments for this transform
            valid_args = {"p", "strict"}  # Base valid args
            if hasattr(self, "InitSchema"):
                valid_args.update(self.InitSchema.model_fields.keys())

            # Check for invalid arguments
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
        if not self.replay_mode and not force_apply and self.p <= 0.0:
            return kwargs
        if self.replay_mode:
            state = invocation.transform_state(self) if invocation.collect_applied else None
            return self._apply_replay(state, **kwargs)

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

    def _apply_replay(self, state: TransformInvocationState | None, **kwargs: Any) -> Any:
        """Applies recorded replay parameters without new sampling, preserving the optional caller-local observation
        behavior of sampled leaves.
        """
        if not self.applied_in_replay:
            return kwargs
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

    def apply_with_uniform_params(self, params: Mapping[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Apply one parameter mapping to every target without target-specific values."""
        res: dict[str, Any] = {}
        for key, arg in kwargs.items():
            if key in self._key2func and arg is not None:
                res[key] = self._key2func[key](arg, **params)
            else:
                res[key] = arg
        return res

    def apply_with_params(
        self,
        sampled_params: SampledParams,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply transforms with parameters. Dispatches each target (image, mask, bboxes, etc.) to
        the corresponding apply_* method.
        """
        res: dict[str, Any] = {}
        for key, arg in kwargs.items():
            if key in self._key2func and arg is not None:
                target_function = self._key2func[key]
                res[key] = target_function(arg, **sampled_params.params_for(key))
            else:
                res[key] = arg
        return res

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

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        """Apply transform on images. Input shape (N, H, W, C); uses _apply_to_batch with per-image
        apply. Returns same format. Batch API.

        Args:
            images (ImageType): Input images as numpy array of shape:
                - (num_images, height, width, channels)
                - (num_images, height, width) for grayscale
            *args (Any): Additional positional arguments
            **params (Any): Additional parameters specific to the transform

        Returns:
            ImageType: Transformed images as numpy array in the same format as input

        """
        return self._apply_to_batch(images, lambda img: self.apply(img, **params))

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
        return self.apply_to_images(volume, *args, **params)

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
                    shared_shape: tuple[int, ...]
                    if view.descriptor.layout == "image_chw":
                        shared_shape = (shape[1], shape[2], shape[0])
                    elif view.descriptor.layout in {"images_clhw", "volume_cdhw"}:
                        shared_shape = (shape[2], shape[3], shape[0])
                    elif view.canonical_type in {"images", "volume", "masks", "mask3d"}:
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
        canonical_by_name = {name: self._additional_targets.get(name, name) for name in self._key2func}
        return TargetSet.from_data(data, canonical_by_name)

    def _validate_spatial_targets(self, targets: TargetSet) -> None:
        if self._sampling_spatial_rank is not None:
            targets.aligned_spatial_shape(self._sampling_spatial_rank)

    @staticmethod
    def _shared_shape_from_data_key(key: str, value: Any) -> tuple[int, ...]:
        if key == "image":
            if isinstance(value, torch.Tensor):
                return value.shape[1], value.shape[2], value.shape[0]
            return value.shape
        if key in {"images", "volume"}:
            if isinstance(value, torch.Tensor):
                return value.shape[2], value.shape[3], value.shape[0]
            return value.shape[1:]
        return value.shape if key == "mask" else value.shape[1:]

    def _extract_shared_shape_from_data(self, data: dict[str, Any]) -> tuple[int, ...] | None:
        """Return the shared shape needed by the no-sampler execution fast path.

        Data-dependent samplers receive target descriptors and must not call this helper.
        """
        for key in ("image", "images", "volume", "mask", "masks", "mask3d"):
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
        """Get mapping of target keys to their corresponding processing functions (e.g. image ->
        apply, mask -> apply_to_mask). Subclasses override.

        Returns:
            dict[str, Callable[..., Any]]: Dictionary mapping target keys to their processing functions.

        """
        # mapping for targets and methods for which they depend
        # for example:
        # >>  {"image": self.apply}
        # >>  {"masks": self.apply_to_masks}
        raise NotImplementedError

    def apply_to_user_data(self, data: Any, **params: Any) -> Any:
        """Apply transform to user-defined data. By default returns data unchanged (passthrough).
        Override to update custom keys (e.g. captions) from params.

        By default, returns the data unchanged (passthrough). Override in a subclass to
        update arbitrary user data in response to geometric or photometric transforms.

        Args:
            data (Any): Arbitrary user-defined data of any type.
            **params (Any): Transform parameters (same as passed to other apply_* methods).

        Returns:
            Any: The (optionally modified) user data. Must return the same type as the input.

        Examples:
            >>> import albumentations as A
            >>> class FlipAwareTransform(A.HorizontalFlip):
            ...     def apply_to_user_data(self, data: dict, **params) -> dict:
            ...         return {"caption": data["caption"].replace("left", "right")}

        """
        return data

    def _set_keys(self) -> None:
        """Set _available_keys and _key2func from _targets and targets. Adds user_data as
        passthrough. Called from __init__. Override targets in subclass.
        """
        if not hasattr(self, "_targets"):
            self._available_keys = set()
        else:
            self._available_keys = {
                target.value.lower()
                for target in (self._targets if isinstance(self._targets, tuple) else [self._targets])
            }
        self._available_keys.update(self.targets.keys())
        self._key2func = {key: self.targets[key] for key in self._available_keys if key in self.targets}
        # user_data is always available regardless of _targets - passthrough by default
        self._available_keys.add("user_data")
        self._key2func["user_data"] = self.apply_to_user_data

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
            if v in self._available_keys:
                self._additional_targets[k] = v
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
        # Get the parameter names
        arg_names = self.get_transform_init_args_names()

        # Create a dictionary of parameter values
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

        # Get transform init args (our improved method handles all types of transforms)
        transform_args = self.get_transform_init_args()

        # Add transform args to state
        state.update(transform_args)

        # Remove strict from serialization
        state.pop("strict", None)

        return state


class DualTransform(BasicTransform):
    """Base class for transforms that apply to both image and annotations (masks, bboxes, keypoints),
    keeping them spatially consistent.

    When a transform is applied to an image, all associated entities (masks, bounding boxes, keypoints) are
    such as masks, bounding boxes, and keypoints. This class ensures that when a transform is applied to an image,
    all associated entities are transformed accordingly to maintain consistency between the image and its annotations.

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

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """Get mapping of target keys to their corresponding processing functions for DualTransform
        (image, mask, bboxes, keypoints, etc.).

        Returns:
            dict[str, Callable[..., Any]]: Dictionary mapping target keys to their processing functions.

        """
        # Note: keypoint label swapping is handled within apply_to_keypoints
        # No separate targets needed for label fields
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "mask3d": self.apply_to_mask3d,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "volume": self.apply_to_images,
            "user_data": self.apply_to_user_data,
        }

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

    @batch_transform("spatial")
    def apply_to_mask3d(self, mask3d: VolumeType, *args: Any, **params: Any) -> VolumeType:
        return self.apply_to_mask(mask3d, *args, **params)

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
        # Get the keypoint processor
        processor = self.get_processor("keypoints")
        if not processor or not hasattr(processor, "encoded_label_mappings"):
            return keypoints

        # Check if there are label fields and the array has extra columns
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
            return sz_lut(mask, uint8_lut, inplace=False)

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
                and canonical_name in {"mask", "masks", "mask3d"}
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
        label_col_start = 5  # After [x, y, z, angle, scale]
        instance_id_col_idx = None
        if "_kp_instance_id" in label_fields:
            candidate_col_idx = label_col_start + label_fields.index("_kp_instance_id")
            if candidate_col_idx < keypoints.shape[1]:
                instance_id_col_idx = candidate_col_idx

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

        # Apply label mapping to keypoints if they were transformed
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
    """

    _targets = (Targets.IMAGE, Targets.VOLUME)

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """Get mapping of target keys to their corresponding processing functions for
        ImageOnlyTransform (image, images, volume, user_data).

        Returns:
            dict[str, Callable[..., Any]]: Dictionary mapping target keys to their processing functions.

        """
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "volume": self.apply_to_volume,
            "user_data": self.apply_to_user_data,
        }


class NoOp(DualTransform):
    """Identity transform (does nothing). Passes all targets through unchanged. Use as placeholder
    or in conditional pipelines.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})  # NoOp passes all bbox types
    _supports_cpu_tensor = True

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """Return identity handlers that preserve every Tensor target without the NumPy batch
        dispatch inherited by `DualTransform` for the volume route.
        """
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "mask3d": self.apply_to_mask3d,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "volume": self.apply_to_volume,
            "user_data": self.apply_to_user_data,
        }

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return keypoints

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return bboxes

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return img

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return images

    def apply_to_mask(self, mask: ImageType, **params: Any) -> ImageType:
        return mask

    def apply_to_masks(self, masks: StackedMasks4D, **params: Any) -> StackedMasks4D:
        return masks

    def apply_to_volume(self, volume: VolumeType, **params: Any) -> VolumeType:
        return volume

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> VolumeType:
        return mask3d


class Transform3D(DualTransform):
    """Base class for all 3D transforms. Inherits from DualTransform; applies to volume data,
    mask3d, keypoints. Override apply_to_volume and apply_to_mask3d.

    Transform3D inherits from DualTransform because 3D transforms can be applied to both
    volume data and masks, similar to how 2D DualTransforms work with images and masks.

    Targets:
        volume: 3D numpy array of shape (D, H, W, C)
        mask3d: 3D numpy array of shape (D, H, W) or (D, H, W, C)
        keypoints: 3D numpy array of shape (N, 3)
    """

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
            column_index = NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS + label_offset
            if not mapping or column_index >= keypoints.shape[1]:
                continue
            source_values = keypoints[:, column_index]
            for source_label, target_label in mapping.items():
                result[source_values == source_label, column_index] = target_label
        return result

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "volume": self.apply_to_volume,
            "mask3d": self.apply_to_mask3d,
            "keypoints": self.apply_to_keypoints,
            "user_data": self.apply_to_user_data,
        }


class VolumeOnlyTransform(BasicTransform):
    """Provide a base for volume-intensity transforms that leave masks and keypoints untouched, keeping acquisition
    artifacts separate from label geometry changes.

    Unlike `Transform3D`, subclasses do not dispatch to `mask3d` or
    `keypoints`. Compose therefore preserves those targets unchanged, which
    is appropriate for acquisition and photometric artifacts that do not alter
    label geometry.
    """

    _targets = (Targets.VOLUME,)

    def apply_to_volume(self, volume: VolumeType, *args: Any, **params: Any) -> VolumeType:
        raise NotImplementedError

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "volume": self.apply_to_volume,
            "user_data": self.apply_to_user_data,
        }


class CustomTransformsApplyMixin:
    """Mixin that auto-registers custom apply_to_<X> methods as handlers for data key <X>.
    Place before base in MRO so _set_keys discovers them.

    Define methods named `apply_to_<key>` in your transform subclass; they are
    discovered at init time and routed through the standard `apply_with_params`
    pipeline. Custom targets receive the same parameters from `sample_parameters`, respect
    the `p=` probability, and compose correctly with Compose and ReplayCompose.

    Placement in inheritance list
        Must come BEFORE the albumentations base class so MRO resolves
        `_set_keys` first::

            class MyTransform(CustomTransformsApplyMixin, A.DualTransform):
                def apply_to_label(self, label, **params):
                    return (label + params["factor']) % 4

    Registration rules
        Methods named `apply_to_<X>` are registered if they are:
        - Defined in the concrete subclass or any class between it and this mixin in the MRO
        - Not already covered by `self.targets` (built-ins take priority)
        - Not `apply_to_user_data` (handled separately by the base)

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
        >>>
        >>> class Rotate90WithLabel(A.CustomTransformsApplyMixin, A.DualTransform):
        ...     def sample_parameters(self, params, data, targets, sampling):
        ...         return SampledParams(params={"k": 1})
        ...     def apply(self, img, k=0, **p):
        ...         return np.rot90(img, k)
        ...     def apply_to_mask(self, mask, k=0, **p):
        ...         return np.rot90(mask, k)
        ...     def apply_to_label(self, label, k=0, **p):
        ...         return (label + k) % 4
        >>>
        >>> transform = A.Compose([Rotate90WithLabel(p=1.0)])
        >>> out = transform(image=image, mask=mask, label=0)
        >>> out["label"]
        1

    """

    _APPLY_PREFIX = "apply_to_"
    _EXCLUDED_KEYS = frozenset({"user_data"})
    _key2func: dict[str, Any]
    _available_keys: set[str]

    def _set_keys(self) -> None:
        # Build _key2func from self.targets using base class
        base_set_keys = cast("Callable[[Any], None]", BasicTransform.__dict__["_set_keys"])
        base_set_keys(self)

        # Search apply_to_<X> functions defined within the child class
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(self._APPLY_PREFIX):
                continue
            key = name[len(self._APPLY_PREFIX) :]
            if key in self._EXCLUDED_KEYS:
                continue
            if key in self._key2func:  # built-in already registered
                continue
            if not self._is_user_defined(name):
                continue
            self._available_keys.add(key)
            self._key2func[key] = method

    def _is_user_defined(self, method_name: str) -> bool:
        """True if method_name is defined on subclass or parents before mixin in MRO (not from albumentations base).
        Used to register only user-defined apply_to_<X>.
        """
        for mro_class in type(self).__mro__:
            if mro_class is CustomTransformsApplyMixin:
                break
            if method_name in mro_class.__dict__:
                return True
        return False
