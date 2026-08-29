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
import types
import warnings
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, ClassVar, Protocol, Union, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import PydanticSchemaGenerationError, PydanticUserError, TypeAdapter, ValidationError

from .analytics.collectors import collect_pipeline_info, get_environment_info

# Telemetry imports
from .analytics.settings import settings
from .analytics.telemetry import get_telemetry_client
from .bbox_utils import BboxParams, BboxProcessor
from .hub_mixin import HubMixin
from .invocation import (
    ChannelRestorationState,
    ComposeInvocationState,
    InvocationContext,
    InvocationRngOwner,
    clear_completed_observation,
    get_current_invocation,
)
from .keypoints_utils import KeypointParams, KeypointsProcessor
from .serialization import (
    SERIALIZABLE_REGISTRY,
    Serializable,
    get_shortest_class_fullname,
    instantiate_nonserializable,
    register_additional_transforms,
)
from .tensor import (
    TENSOR_ANNOTATION_TARGETS,
    TENSOR_CHANNEL_AXIS,
    TENSOR_TARGETS,
    numpy_to_tensor_annotation,
    tensor_to_numpy_annotation,
    validate_tensor_input,
    validate_tensor_metadata_input,
)
from .tracing import TraceOptions, TraceResult, _ExecutionTrace
from .transforms_interface import BasicTransform, DualTransform
from .type_definitions import StackedMasks4D, Targets
from .utils import DataProcessor, format_args, get_shape

__all__ = [
    "BaseCompose",
    "BboxParams",
    "Compose",
    "ComposeTransformNotFoundError",
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

_TRACE_NODE_KIND_COMPOSITION = "composition"
_TRACE_NODE_KIND_LEAF = "leaf"
_TRACE_STATUS_APPLIED = "applied"
_TRACE_STATUS_SKIPPED_PROBABILITY = "skipped_probability"
_TRACE_STATUS_SKIPPED_REPLAY = "skipped_replay"
_TRACE_STATUS_SKIPPED_SELECTION = "skipped_selection"

_REPLAY_PARAM_ANNOTATIONS_CACHE: dict[type, dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class _TensorBoundaryState:
    """Keep Tensor annotation restoration local to one Compose invocation."""

    has_tensor_inputs: bool = False
    annotation_targets: tuple[tuple[str, str], ...] = ()


_EMPTY_TENSOR_BOUNDARY_STATE = _TensorBoundaryState()


def _normalize_semantic_mask_label(label: Any, label_kind: str) -> int:
    if isinstance(label, str):
        try:
            normalized_label = int(label)
        except ValueError as exc:
            raise TypeError(f"semantic mask {label_kind} labels must be integers") from exc
        if label == str(normalized_label):
            return normalized_label
    elif isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_)):
        return int(label)
    raise TypeError(f"semantic mask {label_kind} labels must be integers")


def _normalize_semantic_mask_label_mappings(mappings: Any) -> dict[str, dict[int, int]] | None:
    """Normalize integer semantic-mask label keys after JSON transport and reject invalid, ambiguous,
    or duplicate mappings before use.
    """
    if mappings is None:
        return None
    if not isinstance(mappings, dict):
        raise TypeError("semantic_mask_label_mappings must be a dictionary")

    normalized: dict[str, dict[int, int]] = {}
    for transform_name, mapping in mappings.items():
        if not isinstance(transform_name, str):
            raise TypeError("semantic_mask_label_mappings transform names must be strings")
        if not isinstance(mapping, dict):
            raise TypeError(f"semantic_mask_label_mappings['{transform_name}'] must be a dictionary")

        normalized_mapping: dict[int, int] = {}
        for source_label, target_label in mapping.items():
            normalized_source = _normalize_semantic_mask_label(source_label, "source")
            normalized_target = _normalize_semantic_mask_label(target_label, "target")
            if normalized_source in normalized_mapping:
                raise ValueError(f"Duplicate semantic mask source label after normalization: {normalized_source}")
            normalized_mapping[normalized_source] = normalized_target

        normalized[transform_name] = normalized_mapping
    return normalized


def _get_replay_param_annotations(cls: type) -> dict[str, Any]:
    """Inspect the selected replay constructor once and cache its annotated public parameters for transport-shape
    normalization during reconstruction.
    """
    if cls not in _REPLAY_PARAM_ANNOTATIONS_CACHE:
        signature = inspect.signature(cls)
        _REPLAY_PARAM_ANNOTATIONS_CACHE[cls] = {
            name: parameter.annotation
            for name, parameter in signature.parameters.items()
            if name != "self" and parameter.annotation is not inspect.Parameter.empty
        }
    return _REPLAY_PARAM_ANNOTATIONS_CACHE[cls]


def _normalize_replay_value(annotation: Any, value: Any) -> Any:
    """Normalize one JSON-transported value into a constructor-valid shape without changing values that already
    satisfy the public annotation.

    Applied records intentionally store resolved scalars. JSON transport also changes tuples
    to lists. The constructor may require a two-value tuple, a mapping of axis names to
    tuples, or a pair of compound values. Try those lossless shapes in that order while
    leaving values that already satisfy the annotation untouched.
    """
    try:
        adapter = TypeAdapter(annotation)
    except (PydanticSchemaGenerationError, PydanticUserError):
        return value
    try:
        adapter.validate_python(value)
    except ValidationError:
        pass
    except (PydanticSchemaGenerationError, PydanticUserError):
        return value
    else:
        return value

    candidates: list[Any] = []
    if isinstance(value, dict):
        candidates.append({key: (item, item) for key, item in value.items()})
    if value is not None:
        candidates.append((value, value))

    for candidate in candidates:
        try:
            adapter.validate_python(candidate)
        except ValidationError:
            continue
        except (PydanticSchemaGenerationError, PydanticUserError):
            return value
        return candidate
    return value


def _normalize_config_for_replay(cls: type, config: dict[str, Any]) -> dict[str, Any]:
    """Normalize every transported configuration field against the replay constructor, restoring tuple and range
    shapes lost across JSON.
    """
    annotations = _get_replay_param_annotations(cls)
    return {
        key: _normalize_replay_value(annotations[key], value) if key in annotations else value
        for key, value in config.items()
    }


REPR_INDENT_STEP = 2

TransformType = Union[BasicTransform, "BaseCompose"]
TransformsSeqType = Sequence[TransformType]


class _UnactivatedNodeHandler(Protocol):
    """Describe one compiled ordinary-node executor that receives all call-local state explicitly, keeping selector
    dispatch type-safe without runtime inspection.

    The protocol keeps the selector dispatch table type-safe while each handler
    specializes behavior for one immutable configured container kind.
    """

    def __call__(
        self,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class _CompiledChild:
    """Store one immutable configured edge and its annotation effects, so repeated dispatch schedules required filters
    without recursively scanning the graph.
    """

    transform: TransformType
    may_change_bboxes: bool
    may_change_keypoints: bool
    parent_finalizes_bboxes: bool
    parent_finalizes_keypoints: bool


class ComposeTransformNotFoundError(ArithmeticError, ValueError):
    """Raised when compose subtraction cannot find a requested transform class, preserving ValueError
    compatibility while satisfying operator semantics.
    """


AVAILABLE_KEYS = (
    "image",
    "mask",
    "masks",
    "bboxes",
    "keypoints",
    "volume",
    "mask3d",
    "user_data",
)

MASK_KEYS = (
    "mask",  # 2D mask
    "masks",  # Multiple 2D masks
    "mask3d",  # 3D mask
)

# Keys related to image data
IMAGE_KEYS = {"image", "images"}
CHECK_BBOX_PARAM = {"bboxes"}
CHECK_KEYPOINTS_PARAM = {"keypoints"}
VOLUME_KEYS = {"volume"}
_SPATIAL_ADDITIONAL_TARGETS = frozenset((*IMAGE_KEYS, *MASK_KEYS, *VOLUME_KEYS))

_VALID_INSTANCE_BINDING_TARGETS = frozenset({"mask", "masks", "bboxes", "keypoints"})
# Distinct ferry-key constants used to shuttle per-row instance ids through `data`
# between unpack and per-processor preprocess. They MUST remain different dict-key
# strings because the bbox per-row id list and the kp per-row id list can have
# different lengths and both need to coexist in `data` simultaneously. Conceptually
# they encode the same logical instance-id namespace (last column of `bboxes` / last
# column of `keypoints`); resync logic re-establishes that pairing each step.
_BBOX_INSTANCE_ID = "_bbox_instance_id"
_KP_INSTANCE_ID = "_kp_instance_id"
_INSTANCE_ID_FERRY_KEYS = frozenset({_BBOX_INSTANCE_ID, _KP_INSTANCE_ID})


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


class BaseCompose(InvocationRngOwner, Serializable):
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
    check_each_transform: tuple[DataProcessor[Any], ...] | None = None
    main_compose: bool = True
    applied_in_replay: bool = False
    _filters_annotations_during_dispatch: bool = False

    @property
    def filters_annotations_during_dispatch(self) -> bool:
        """Report whether this container filters annotations in its selected branch so its parent
        avoids duplicate geometry work after the container returns.
        """
        return self._filters_annotations_during_dispatch

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

        self.transforms = tuple(transforms)
        self._compile_execution_plan()
        self.p = p
        self.invocation_key = object()

        self.replay_mode = False
        self._additional_targets: dict[str, str] = {}
        self._available_keys: set[str] = set()
        self._configured_processors: dict[str, BboxProcessor | KeypointsProcessor] = {}
        self._set_keys()
        self.set_mask_interpolation(mask_interpolation)
        self._initialize_invocation_rng(seed)
        self._ensure_configured_random_sources()
        self.save_applied_params = save_applied_params

    def _compile_execution_plan(self) -> None:
        """Rebuild immutable edge metadata after construction or unpickling, so repeated calls reuse target effects
        and executor selection without graph scans.

        The configured sequence is already frozen; this method derives only runtime
        dispatch metadata, never call-local state or serialized random streams.
        """
        self._compiled_children = tuple(self._compile_child(transform) for transform in self.transforms)
        basic_children: tuple[BasicTransform, ...] | None = (
            cast("tuple[BasicTransform, ...]", self.transforms)
            if all(isinstance(transform, BasicTransform) for transform in self.transforms)
            else None
        )
        self._unactivated_basic_children: tuple[BasicTransform, ...] | None = (
            basic_children
            if basic_children is not None
            and all(
                type(transform).apply_in_invocation is BasicTransform.apply_in_invocation
                for transform in self.transforms
            )
            else None
        )
        self._unactivated_compiled_graph = all(
            self._node_supports_unactivated_invocation(transform) for transform in self.transforms
        )

    @classmethod
    def _compile_child(cls, transform: TransformType) -> _CompiledChild:
        """Derive one child's static target effects once, so parent dispatch schedules required annotation work
        without recursively reclassifying the graph.
        """
        may_change_bboxes = cls._transform_may_change_target(transform, Targets.BBOXES)
        may_change_keypoints = cls._transform_may_change_target(transform, Targets.KEYPOINTS)
        child_finalizes_annotations = (
            isinstance(transform, BaseCompose) and transform.filters_annotations_during_dispatch
        )
        return _CompiledChild(
            transform=transform,
            may_change_bboxes=may_change_bboxes,
            may_change_keypoints=may_change_keypoints,
            parent_finalizes_bboxes=may_change_bboxes and not child_finalizes_annotations,
            parent_finalizes_keypoints=may_change_keypoints and not child_finalizes_annotations,
        )

    @classmethod
    def _node_supports_unactivated_invocation(cls, transform: TransformType) -> bool:
        """Identify graph nodes that use the explicit invocation contract so ordinary training can skip ContextVar
        activation without losing call-local state.

        A node is eligible when it inherits the standard explicit-invocation leaf
        dispatcher or is one of the configured container kinds with eligible children.
        """
        if isinstance(transform, BasicTransform):
            return type(transform).apply_in_invocation is BasicTransform.apply_in_invocation
        if not isinstance(transform, BaseCompose) or transform.replay_mode:
            return False
        return type(transform) in {
            Compose,
            OneOf,
            SomeOf,
            RandomOrder,
            OneOrOther,
            SelectiveChannelTransform,
            Sequential,
        } and all(cls._node_supports_unactivated_invocation(child) for child in transform.transforms)

    def _track_transform_params(self, transform: TransformType, data: dict[str, Any]) -> None:
        """Append a (class_fullname, applied_config) tuple to applied_transforms when
        save_applied_params=True. Skipped transforms (empty applied_config) are not recorded.
        """
        if "applied_transforms" in data and isinstance(transform, BasicTransform) and transform.applied_config:
            data["applied_transforms"].append(
                (transform.get_applied_replay_class().get_class_fullname(), transform.applied_config.copy()),
            )

    def _apply_child(
        self,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext | None,
        *,
        force_apply: bool = False,
    ) -> dict[str, Any]:
        """Run a configured child through active invocation so public Compose boundaries stay
        independent while internal edges share one executor and call-local state.

        A public :class:`Compose` call always creates its own root boundary, even if it is
        made from a custom transform while another pipeline is running. Configured child
        nodes therefore bypass that public entry point explicitly. This small dispatcher is
        the only place where graph execution crosses from a container to one of its nodes.
        """
        if isinstance(transform, BasicTransform):
            if invocation is None:
                if not force_apply and transform.p <= 0.0:
                    return data
                if transform.can_apply_without_invocation(force_apply=force_apply):
                    return transform.apply_without_invocation(
                        force_apply=force_apply,
                        publish_observation=False,
                        **data,
                    )
                msg = "configured transform requires an invocation"
                raise RuntimeError(msg)
            return transform.apply_in_invocation(invocation, force_apply=force_apply, **data)
        if isinstance(transform, BaseCompose):
            if invocation is None:
                msg = "configured container requires an invocation"
                raise RuntimeError(msg)
            return transform.apply_in_invocation(invocation, force_apply=force_apply, **data)
        # Extension nodes are invoked only on the generic route. The ordinary
        # executor admits BasicTransform and BaseCompose nodes with explicit
        # invocation contracts.
        return transform(force_apply=True, **data) if force_apply else transform(**data)

    @classmethod
    def _apply_unactivated_node(
        cls,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool = False,
    ) -> dict[str, Any]:
        """Run an explicit-invocation node through the compiled ordinary executor, omitting trace and annotation
        branches after construction selected this route.

        The root admits this executor only after construction proved that every node
        is a native explicit-invocation node and the call has no processors,
        observation, trace, Tensor bridge, or grayscale repair. Keeping this loop
        separate removes per-node tracing and annotation branches.
        """
        if isinstance(transform, BasicTransform):
            return transform.apply_in_invocation(invocation, force_apply=force_apply, **data)
        handlers: dict[type[BaseCompose], _UnactivatedNodeHandler] = {
            Compose: cls._apply_unactivated_compose,
            OneOf: cls._apply_unactivated_one_of,
            SomeOf: cls._apply_unactivated_some_of,
            RandomOrder: cls._apply_unactivated_some_of,
            OneOrOther: cls._apply_unactivated_one_or_other,
            SelectiveChannelTransform: cls._apply_unactivated_selective_channels,
            Sequential: cls._apply_unactivated_sequential,
        }
        handler = handlers.get(type(transform))
        if handler is None:
            msg = f"Unsupported unactivated node type: {type(transform).__name__}"
            raise RuntimeError(msg)
        return handler(transform, data, invocation, force_apply=force_apply)

    @staticmethod
    def _should_apply_unactivated_node(
        transform: TransformType,
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> bool:
        """Evaluate a composition probability from the supplied call-local stream, preserving deterministic endpoints
        without reading configured-object random state.
        """
        if force_apply or transform.p >= 1.0:
            return True
        return transform.p > 0.0 and invocation.py_random.random() < transform.p

    @classmethod
    def _apply_unactivated_compose(
        cls,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> dict[str, Any]:
        compose = cast("Compose", transform)
        compose.validate_nested_boundary(data)
        if not cls._should_apply_unactivated_node(compose, invocation, force_apply=force_apply):
            return data
        return cls._apply_unactivated_children(compose, data, invocation)

    @classmethod
    def _apply_unactivated_one_of(
        cls,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> dict[str, Any]:
        one_of = cast("OneOf", transform)
        if not one_of.transforms_ps or not cls._should_apply_unactivated_node(
            one_of, invocation, force_apply=force_apply
        ):
            return data
        index = invocation.random_generator.choice(len(one_of.transforms), p=one_of.transforms_ps)
        return cls._apply_unactivated_node(one_of.transforms[index], data, invocation, force_apply=True)

    @classmethod
    def _apply_unactivated_some_of(
        cls,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> dict[str, Any]:
        some_of = cast("SomeOf", transform)
        if not cls._should_apply_unactivated_node(some_of, invocation, force_apply=force_apply):
            return data
        for index in some_of.sample_indices(invocation):
            data = cls._apply_unactivated_node(some_of.transforms[int(index)], data, invocation)
        return data

    @classmethod
    def _apply_unactivated_one_or_other(
        cls,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> dict[str, Any]:
        del force_apply
        one_or_other = cast("OneOrOther", transform)
        index = one_or_other.sample_branch_index(invocation)
        return cls._apply_unactivated_node(one_or_other.transforms[index], data, invocation, force_apply=True)

    @classmethod
    def _apply_unactivated_selective_channels(
        cls,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> dict[str, Any]:
        selective = cast("SelectiveChannelTransform", transform)
        if not cls._should_apply_unactivated_node(selective, invocation, force_apply=force_apply):
            return data
        image = data["image"]
        sub_image = np.ascontiguousarray(image[:, :, selective.channels])
        for child in selective.transforms:
            sub_image = cls._apply_unactivated_node(child, {"image": sub_image}, invocation)["image"]
        output_image = image.copy()
        for channel_index, channel in zip(selective.channels, cv2.split(sub_image), strict=True):
            output_image[:, :, channel_index] = channel
        data["image"] = np.ascontiguousarray(output_image)
        return data

    @classmethod
    def _apply_unactivated_sequential(
        cls,
        transform: TransformType,
        data: dict[str, Any],
        invocation: InvocationContext,
        *,
        force_apply: bool,
    ) -> dict[str, Any]:
        sequential = cast("Sequential", transform)
        if not cls._should_apply_unactivated_node(sequential, invocation, force_apply=force_apply):
            return data
        return cls._apply_unactivated_children(sequential, data, invocation)

    @classmethod
    def _apply_unactivated_children(
        cls,
        container: "BaseCompose",
        data: dict[str, Any],
        invocation: InvocationContext,
    ) -> dict[str, Any]:
        """Run an immutable configured child sequence through the ordinary executor, preserving order and selectors
        while skipping generic observed-mode wrappers.
        """
        for transform in container.transforms:
            data = cls._apply_unactivated_node(transform, data, invocation)
        return data

    def _execute_child(
        self,
        index: int,
        data: dict[str, Any],
        invocation: InvocationContext | None,
        *,
        force_apply: bool = False,
        track_applied: bool = False,
        tracking_data: dict[str, Any] | None = None,
        finalize_annotations: bool = True,
        resync_instances: bool = False,
    ) -> dict[str, Any]:
        """Execute a configured edge through normal dispatch, then perform annotation, trace, and
        instance-binding work in observable order at this graph boundary.

        The trace path wraps this exact route; it never resamples a probability or
        reimplements a composition selection.  Annotation finalization occurs before
        a leaf record so snapshots observe the same public intermediate state that
        the following node receives.
        """
        compiled_child = self._compiled_children[index]
        transform = compiled_child.transform
        trace = None if invocation is None else invocation.trace_session

        if trace is None:
            data = self._apply_child(transform, data, invocation, force_apply=force_apply)
            if track_applied:
                self._track_transform_params(transform, data if tracking_data is None else tracking_data)
            if finalize_annotations:
                data = self._finalize_annotations_after(data, transform, compiled_child, invocation)
            if resync_instances and compiled_child.may_change_bboxes:
                self._resync_instance_ids(data)
            return data

        with trace.node_scope(trace.child_path(index)):
            start_ns = perf_counter_ns() if trace.context.options.include_timing else None
            data = self._apply_child(transform, data, invocation, force_apply=force_apply)
            if track_applied:
                self._track_transform_params(transform, data if tracking_data is None else tracking_data)
            if finalize_annotations:
                data = self._finalize_annotations_after(data, transform, compiled_child, invocation)
            if resync_instances and compiled_child.may_change_bboxes:
                self._resync_instance_ids(data)
            if isinstance(transform, BasicTransform):
                applied = transform.applied_in_replay if transform.replay_mode else bool(transform.params)
                trace.emit_leaf(
                    class_fullname=transform.get_applied_replay_class().get_class_fullname(),
                    status=_TRACE_STATUS_APPLIED if applied else self._leaf_skip_status(transform),
                    params=transform.applied_config if applied else None,
                    data=data if applied else None,
                    elapsed_ns=None if start_ns is None else perf_counter_ns() - start_ns,
                )
        return data

    def _should_apply_in_context(self, context: InvocationContext, *, force_apply: bool) -> bool:
        """Evaluate container probability from the active call-local stream, keeping concurrent calls isolated and
        avoiding configured generator access during dispatch.
        """
        if force_apply or self.p >= 1.0:
            return True
        return self.p > 0.0 and context.py_random.random() < self.p

    @property
    def processors(self) -> dict[str, BboxProcessor | KeypointsProcessor]:
        """Returns configured processors outside execution and call-local sessions during a call, isolating label
        encoders and temporary metadata between samples.
        """
        if not self._configured_processors:
            return self._configured_processors
        invocation = get_current_invocation()
        if invocation is None:
            return self._configured_processors
        return invocation.processors(self._configured_processors)

    @processors.setter
    def processors(self, value: dict[str, BboxProcessor | KeypointsProcessor]) -> None:
        self._configured_processors = value

    def _new_invocation_context(
        self,
        *,
        collect_applied: bool | None = None,
        invocation_seed: int | None = None,
        prime_py_random: bool = False,
    ) -> "InvocationContext":
        """Creates one call-local context whose worker stream is reserved only when needed, while explicit invocation
        seeds remain isolated from the default worker stream.
        """
        return super()._create_invocation_context(
            collect_applied=self.save_applied_params if collect_applied is None else collect_applied,
            root_key=self.invocation_key,
            invocation_seed=invocation_seed,
            prime_py_random=prime_py_random,
        )

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
                    cast("Any", transform).mask_interpolation = self.mask_interpolation
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

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Execute a configured container inside its parent invocation, keeping nested graph work in one scope without
        reopening public boundaries or random state.

        Concrete containers implement this separate internal entry point so graph
        dispatch does not re-enter their public call boundary.
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
        """Build a detached constructor representation that preserves child order and policy, so serialization and
        graph edits can recreate an equivalent composition.

        Returns:
            dict[str, Any]: Dictionary representation of the composition.

        """
        return {
            "__class_fullname__": self.get_class_fullname(),
            "transforms": [t.to_dict_private() for t in self.transforms],
            **self._get_reconstruction_kwargs(),
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
            **self._get_reconstruction_kwargs(),
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

    @staticmethod
    def _transform_may_change_target(transform: TransformType, target: Targets) -> bool:
        """Determine whether a node can change an annotation target, allowing image-only nodes to
        bypass shape construction, clipping, and annotation work.

        This is deliberately structural rather than parameter-dependent: filter
        scheduling must remain correct for custom transforms and stochastic nodes,
        while image-only nodes avoid shape construction, clipping, and boolean-mask
        allocations altogether.
        """
        if isinstance(transform, BasicTransform):
            return target.value.lower() in transform.available_keys
        if isinstance(transform, SelectiveChannelTransform):
            return False
        if isinstance(transform, BaseCompose):
            return any(BaseCompose._transform_may_change_target(child, target) for child in transform.transforms)
        return True

    def _needs_annotation_filter(self, transform: TransformType | None, processor: DataProcessor[Any]) -> bool:
        """Decide whether this container must filter after a node, avoiding redundant geometry passes
        while selectors defer branch cleanup to their parent.

        Containers that filter their own selected children (`Compose`, `SomeOf`,
        and `Sequential`) publish an already-current annotation state to their
        parent. Branch selectors do not, so their parent performs exactly one filter
        after the selected branch.
        """
        if transform is None:
            return True
        target = Targets.BBOXES if isinstance(processor, BboxProcessor) else Targets.KEYPOINTS
        if not self._transform_may_change_target(transform, target):
            return False
        return not isinstance(transform, BaseCompose) or not transform.filters_annotations_during_dispatch

    @staticmethod
    def _compiled_child_requires_filter(compiled_child: _CompiledChild, processor: DataProcessor[Any]) -> bool:
        """Read a precomputed processor effect, so image-only edges skip shape discovery and filtering while geometry
        edges retain required annotation work.
        """
        return (
            compiled_child.parent_finalizes_bboxes
            if isinstance(processor, BboxProcessor)
            else compiled_child.parent_finalizes_keypoints
        )

    def _may_change_bboxes(self, transform: TransformType) -> bool:
        """Determine whether a node can change bbox alignment so bound masks and keypoints resynchronize
        only after geometry-capable execution.
        """
        return self._transform_may_change_target(transform, Targets.BBOXES)

    def _resync_instance_ids(self, data: dict[str, Any]) -> None:
        """Provide a root instance-binding resynchronization hook while ordinary containers avoid
        binding scans after nodes that cannot alter bboxes.
        """
        del data

    @staticmethod
    def _leaf_skip_status(transform: BasicTransform) -> str:
        """Choose trace skip status from leaf policy, distinguishing replay suppression from probability
        non-selection for consumers reconstructing the graph run.
        """
        return _TRACE_STATUS_SKIPPED_REPLAY if transform.replay_mode else _TRACE_STATUS_SKIPPED_PROBABILITY

    @staticmethod
    def _emit_skipped_trace_subtree(
        trace: _ExecutionTrace,
        transform: TransformType,
        path: tuple[int, ...],
        status: str,
    ) -> None:
        """Publish a complete skipped subtree without invoking nodes, preserving configured paths for
        metadata, debugging, and replay inspection.
        """
        if isinstance(transform, BasicTransform):
            trace.context.emit(
                node_path=path,
                class_fullname=transform.get_applied_replay_class().get_class_fullname(),
                node_kind=_TRACE_NODE_KIND_LEAF,
                status=status,
            )
            return
        trace.context.emit(
            node_path=path,
            class_fullname=transform.get_class_fullname(),
            node_kind=_TRACE_NODE_KIND_COMPOSITION,
            status=status,
        )
        for index, child in enumerate(transform.transforms):
            BaseCompose._emit_skipped_trace_subtree(trace, child, (*path, index), status)

    @staticmethod
    def _emit_unselected_trace_children(
        trace: _ExecutionTrace,
        transform: "BaseCompose",
        selected_indices: set[int],
    ) -> None:
        """Publish skipped-selection records for unchosen selector branches so trace consumers see
        configured nodes as well as the work that executed.
        """
        for index, child in enumerate(transform.transforms):
            if index not in selected_indices:
                BaseCompose._emit_skipped_trace_subtree(
                    trace,
                    child,
                    trace.child_path(index),
                    _TRACE_STATUS_SKIPPED_SELECTION,
                )

    def _emit_trace_composition(
        self,
        trace: _ExecutionTrace | None,
        *,
        status: str = _TRACE_STATUS_APPLIED,
    ) -> None:
        """Publish a completed composition only after normal dispatch commits annotation and instance
        state, matching the result its parent node sees.
        """
        if trace is not None:
            trace.emit_composition(class_fullname=self.get_class_fullname(), status=status)

    def _emit_trace_skip(self, trace: _ExecutionTrace | None, status: str) -> None:
        """Publish every configured node in a skipped composition subtree, recording its reason and
        structural extent without executing it.
        """
        if trace is not None:
            self._emit_skipped_trace_subtree(trace, self, trace.node_path, status)

    def check_data_post_transform(
        self,
        data: dict[str, Any],
        transform: TransformType | None = None,
        compiled_child: _CompiledChild | None = None,
        invocation: InvocationContext | None = None,
    ) -> dict[str, Any]:
        """Check and filter data after transformation. Runs each check_each_transform
        processor (e.g. bbox filter) on matching data keys. Returns filtered data dict.

        When `instance_binding` is active and the bbox processor drops rows, this method
        mirrors that survival decision onto `masks` rows or `mask` channels (positionally)
        and `keypoints` (by surviving `_bbox_instance_id` value) so bound targets remain
        aligned at every transform boundary. Without this mirror-drop the resync would have
        to reconstruct the survival decision from snapshot state — phase 3b of the rewrite
        collapses that machinery into a single keep-mask plumbed straight from
        `BboxProcessor`.

        Args:
            data (dict[str, Any]): Dictionary containing transformed data
            transform (TransformType | None): Node that just ran. When provided, processors whose target cannot
                change are skipped.
            compiled_child (_CompiledChild | None): Precomputed target effects for a configured graph edge.
            invocation (InvocationContext | None): Active call-local processor state for a configured edge.

        Returns:
            dict[str, Any]: Filtered data dictionary

        """
        if not self.check_each_transform:
            return data

        binding = getattr(self, "_instance_binding", None)
        filter_processors = tuple(
            configured_proc
            for configured_proc in self.check_each_transform
            if (
                self._needs_annotation_filter(transform, configured_proc)
                if compiled_child is None
                else self._compiled_child_requires_filter(compiled_child, configured_proc)
            )
        )
        if not filter_processors:
            return data

        shape = get_shape(data)

        for configured_proc in filter_processors:
            proc = None if invocation is None else invocation.get_processor_session(configured_proc)
            proc = configured_proc if proc is None else proc
            if binding is not None and "bboxes" in binding and isinstance(proc, BboxProcessor):
                self._bbox_filter_with_mirror(proc, data, shape, binding)
                if invocation is not None:
                    invocation.mark_processor_filtered(configured_proc)
                continue
            for data_name, data_value in data.items():
                if data_name in proc.data_fields or (
                    data_name in self._additional_targets and self._additional_targets[data_name] in proc.data_fields
                ):
                    data[data_name] = proc.filter(data_value, shape)
            if invocation is not None:
                invocation.mark_processor_filtered(configured_proc)
        return data

    def _finalize_annotations_after(
        self,
        data: dict[str, Any],
        transform: TransformType,
        compiled_child: _CompiledChild,
        invocation: InvocationContext | None,
    ) -> dict[str, Any]:
        """Run post-node annotation policy, passing the transform only to the base scheduler so it can
        make target-aware filtering decisions for affected annotations.

        A custom Compose policy owns its full post-node boundary.  The base policy
        receives the transform so it can schedule target-aware filtering.
        """
        if type(self).check_data_post_transform is BaseCompose.check_data_post_transform:
            return self.check_data_post_transform(data, transform, compiled_child, invocation)
        return self.check_data_post_transform(data)

    def _bbox_filter_with_mirror(
        self,
        proc: "BboxProcessor",
        data: dict[str, Any],
        shape: tuple[int, ...],
        binding: frozenset[str],
    ) -> None:
        """Run the bbox filter, mirror its keep-mask onto masks (positional) and keypoints (by
        surviving id), and realign transform-filtered inputs first.

        Two stages ensure the bbox-row count drives both transform-level filtering and the
        bbox-processor visibility pass:

        1. Pre-filter realignment. Some transforms (CoarseDropout, Crop with `min_area`,
           etc.) drop bbox rows inside their own `apply_to_bboxes` without touching the
           bound masks. The surviving bboxes carry their original `_bbox_instance_id` in
           the last column. That id selects a row in `masks` or a channel in `mask`, so we
           fancy-index the bound masks down to the surviving id set and drop keypoints whose
           `_kp_instance_id` is no longer present. This establishes the positional layout
           required by the rest of this method.
        2. BboxProcessor filter + post-filter mirror. Standard `filter_with_keep_mask` on
           bboxes; mirror the resulting `keep_mask` positionally onto mask rows or channels
           and by surviving id onto keypoints.
        """
        bboxes_pre = data.get("bboxes")
        if isinstance(bboxes_pre, np.ndarray) and bboxes_pre.shape[0] > 0:
            pre_bbox_ids: np.ndarray | None = bboxes_pre[:, -1].astype(np.int64, copy=True)
            n_pre = bboxes_pre.shape[0]
        else:
            pre_bbox_ids = None
            n_pre = 0

        if pre_bbox_ids is not None:
            self._prefilter_realign(data, binding, pre_bbox_ids, n_pre)

        keep_mask = self._run_bbox_filter(proc, data, shape)

        if keep_mask is None or pre_bbox_ids is None or n_pre == 0:
            return

        self._mirror_keep_mask(data, binding, keep_mask, pre_bbox_ids, n_pre)

    def _prefilter_realign(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        pre_bbox_ids: np.ndarray,
        n_pre: int,
    ) -> None:
        """Drop bound mask rows or channels and keypoints absent from `pre_bbox_ids` so data remains positionally
        aligned before the bbox processor's filter runs.
        """
        if "masks" in binding:
            masks_pre = data.get("masks")
            if masks_pre is not None and isinstance(masks_pre, (np.ndarray, torch.Tensor)) and len(masks_pre) > n_pre:
                valid = pre_bbox_ids[(pre_bbox_ids >= 0) & (pre_bbox_ids < len(masks_pre))]
                data["masks"] = self._index_axis(masks_pre, valid, 0)
        elif "mask" in binding:
            mask_pre = data.get("mask")
            instance_axis = self._mask_instance_axis(data, mask_pre)
            if mask_pre is not None and instance_axis is not None and mask_pre.shape[instance_axis] > n_pre:
                channel_count = mask_pre.shape[instance_axis]
                valid = pre_bbox_ids[(pre_bbox_ids >= 0) & (pre_bbox_ids < channel_count)]
                data["mask"] = self._index_axis(mask_pre, valid, instance_axis)
        if "keypoints" in binding:
            kps_pre = data.get("keypoints")
            if isinstance(kps_pre, np.ndarray) and kps_pre.shape[0] > 0:
                kp_keep_pre = np.isin(kps_pre[:, -1].astype(np.int64, copy=False), pre_bbox_ids)
                if not kp_keep_pre.all():
                    data["keypoints"] = kps_pre[kp_keep_pre]

    def _run_bbox_filter(
        self,
        proc: "BboxProcessor",
        data: dict[str, Any],
        shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Run `BboxProcessor.filter_with_keep_mask` on every matching field in `data`,
        write back the filtered bboxes, return the canonical `"bboxes"` keep-mask only.

        Only the canonical `"bboxes"` field's keep-mask is returned: instance binding lives
        on the primary instances, and `additional_targets`-aliased bbox arrays describe a
        SEPARATE coordinate space whose survival decision must not drive masks/keypoints
        belonging to the primary `"bboxes"` row order.
        """
        keep_mask: np.ndarray | None = None
        for data_name, data_value in data.items():
            if data_name in proc.data_fields or (
                data_name in self._additional_targets and self._additional_targets[data_name] in proc.data_fields
            ):
                filtered, current_keep_mask = proc.filter_with_keep_mask(
                    data_value,
                    cast("tuple[int, int, int]", shape),
                )
                data[data_name] = filtered
                if data_name == "bboxes":
                    keep_mask = current_keep_mask
        return keep_mask

    def _mirror_keep_mask(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        keep_mask: np.ndarray,
        pre_bbox_ids: np.ndarray,
        n_pre: int,
    ) -> None:
        """Mirror the bbox processor's positional keep mask onto bound mask rows or channels and keypoints by
        surviving instance id so all targets stay aligned.
        """
        if "masks" in binding:
            masks = data.get("masks")
            if masks is not None and isinstance(masks, (np.ndarray, torch.Tensor)) and len(masks) == n_pre:
                data["masks"] = self._index_axis(masks, keep_mask, 0)
        elif "mask" in binding:
            mask = data.get("mask")
            instance_axis = self._mask_instance_axis(data, mask)
            if mask is not None and instance_axis is not None and mask.shape[instance_axis] == n_pre:
                data["mask"] = self._index_axis(mask, keep_mask, instance_axis)
        if "keypoints" in binding:
            kps = data.get("keypoints")
            if isinstance(kps, np.ndarray) and kps.shape[0] > 0:
                surviving_bbox_ids = pre_bbox_ids[keep_mask]
                kp_keep = np.isin(kps[:, -1].astype(np.int64, copy=False), surviving_bbox_ids)
                if not kp_keep.all():
                    data["keypoints"] = kps[kp_keep]

    def _drop_instances_with_empty_bound_masks(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
    ) -> None:
        """Remove bbox-bound instances whose transformed masks contain no non-zero pixels, including their bboxes,
        labels, and keypoints.
        """
        bboxes = data.get("bboxes")
        if not isinstance(bboxes, np.ndarray) or bboxes.shape[0] == 0:
            return

        keep_mask = self._nonempty_bound_mask_rows(data, binding, bboxes.shape[0])
        if keep_mask is None:
            return
        if keep_mask.all():
            return

        instance_axis = 0 if "masks" in binding else self._mask_instance_axis(data, data.get("mask"))
        if instance_axis is None:
            return

        bbox_instance_ids = bboxes[:, -1].astype(np.int64, copy=False)
        surviving_instance_ids = bbox_instance_ids[keep_mask]
        data["bboxes"] = bboxes[keep_mask]
        if "masks" in binding:
            data["masks"] = self._index_axis(data["masks"], keep_mask, instance_axis)
        else:
            data["mask"] = self._index_axis(data["mask"], keep_mask, instance_axis)

        if "keypoints" in binding:
            keypoints = data.get("keypoints")
            if isinstance(keypoints, np.ndarray) and keypoints.shape[0] > 0:
                keypoint_keep_mask = np.isin(
                    keypoints[:, -1].astype(np.int64, copy=False),
                    surviving_instance_ids,
                )
                if not keypoint_keep_mask.all():
                    data["keypoints"] = keypoints[keypoint_keep_mask]

    def _nonempty_bound_mask_rows(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        instance_count: int,
    ) -> np.ndarray | None:
        """Return one boolean per bound instance indicating whether its transformed mask contains at least one non-zero
        pixel after augmentation.
        """
        if "masks" in binding:
            masks = data.get("masks")
            if isinstance(masks, np.ndarray) and len(masks) == instance_count:
                return np.any(masks, axis=tuple(range(1, masks.ndim)))
        elif "mask" in binding:
            mask = data.get("mask")
            if isinstance(mask, np.ndarray) and mask.ndim >= 3 and mask.shape[-1] == instance_count:
                return np.any(mask, axis=tuple(range(mask.ndim - 1)))

        mask_with_axis = self._bound_mask_with_instance_axis(data, binding, instance_count)
        if mask_with_axis is None:
            return None

        mask, instance_axis = mask_with_axis
        mask_array = mask.numpy()
        reduction_axes = tuple(axis for axis in range(mask_array.ndim) if axis != instance_axis % mask_array.ndim)
        return np.any(mask_array, axis=reduction_axes)

    def _bound_mask_with_instance_axis(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        instance_count: int,
    ) -> tuple[Any, int] | None:
        """Locate the bound mask container and identify its instance axis across NumPy layouts and terminal CPU
        tensors produced by `ToTensorV2` for aligned filtering.
        """
        if "masks" in binding:
            masks = data.get("masks")
            if masks is not None and isinstance(masks, (np.ndarray, torch.Tensor)) and len(masks) == instance_count:
                return masks, 0
        elif "mask" in binding:
            mask = data.get("mask")
            instance_axis = self._mask_instance_axis(data, mask)
            if mask is not None and instance_axis is not None and mask.shape[instance_axis] == instance_count:
                return mask, instance_axis
        return None

    def _mask_instance_axis(self, data: dict[str, Any], mask: Any) -> int | None:
        """Identify whether a packed instance mask stores instances on the trailing NumPy axis or the leading axis
        produced by transposed tensor output.
        """
        if not isinstance(mask, (np.ndarray, torch.Tensor)) or mask.ndim < 3:
            return None
        if isinstance(mask, np.ndarray):
            return -1

        shape = get_shape(data)
        spatial_shape = tuple(shape[:2])
        trailing_layout = tuple(mask.shape[:2]) == spatial_shape
        leading_layout = tuple(mask.shape[-2:]) == spatial_shape
        if trailing_layout and not leading_layout:
            return -1
        if leading_layout and not trailing_layout:
            return 0
        if trailing_layout and leading_layout:
            configured_axis = self._configured_tensor_mask_instance_axis(self.transforms)
            return -1 if configured_axis is None else configured_axis
        return None

    @staticmethod
    def _configured_tensor_mask_instance_axis(transforms: Sequence[Any]) -> int | None:
        """Inspect nested transforms to disambiguate whether terminal ToTensorV2 places packed mask instances on the
        leading or trailing axis.
        """
        for transform in reversed(transforms):
            transform_type = type(transform)
            is_to_tensor_v2 = any(
                base_type.__name__ == "ToTensorV2" and base_type.__module__ == "albumentations.pytorch.transforms"
                for base_type in transform_type.__mro__
            )
            if is_to_tensor_v2:
                return 0 if transform.transpose_mask else -1
            nested_transforms = getattr(transform, "transforms", None)
            if isinstance(nested_transforms, Sequence):
                nested_axis = BaseCompose._configured_tensor_mask_instance_axis(nested_transforms)
                if nested_axis is not None:
                    return nested_axis
        return None

    @staticmethod
    def _index_axis(array: Any, index: Any, axis: int) -> Any:
        """Select instance rows or channels along one axis while preserving the input array type and using safe index
        tensors for PyTorch values.
        """
        if isinstance(array, torch.Tensor) and isinstance(index, np.ndarray):
            if np.issubdtype(index.dtype, np.bool_):
                index = np.flatnonzero(index)
            index_tensor = array.new_tensor(index.tolist()).long()
            return array.index_select(axis % array.ndim, index_tensor)
        selection = [slice(None)] * array.ndim
        selection[axis] = index
        return array[tuple(selection)]

    def _validate_transforms(self, transforms: Sequence[Any]) -> None:
        """Validate that composition operators receive only transform leaves, preserving graph-edit semantics and
        preventing invalid children during composition.

        Args:
            transforms (Sequence[Any]): Objects to validate in configured order.

        Raises:
            TypeError: If any element is not a BasicTransform instance.

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
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            self._validate_transforms(other)
            other_list = list(other)
        else:
            other_list = [other]
            self._validate_transforms(other_list)

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

    def __sub__(self, other: type[BasicTransform]) -> "BaseCompose | types.NotImplementedType":
        """Remove transform by class type. Removes first matching; returns new instance.
        Use - (e.g. compose - A.HorizontalFlip). Returns NotImplemented for other types.

        Removes the first transform in the compose that matches the provided transform class.

        Args:
            other (type[BasicTransform]): Transform class to remove (e.g., A.HorizontalFlip)

        Returns:
            BaseCompose | types.NotImplementedType: New compose instance with transform removed, or NotImplemented.

        Raises:
            ComposeTransformNotFoundError: If no transform of that type is found in the compose

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
        raise ComposeTransformNotFoundError(f"No transform of type {class_name} found in the compose pipeline")

    def _create_new_instance(self, new_transforms: TransformsSeqType) -> "BaseCompose":
        """Create new instance of same class with new transforms. Copies init params
        and random state from self. Called by __add__, __radd__, __sub__.

        Args:
            new_transforms (TransformsSeqType): List of transforms for the new instance

        Returns:
            BaseCompose: New instance of the same class

        """
        reconstruction_kwargs = self._get_reconstruction_kwargs()
        reconstruction_kwargs["transforms"] = new_transforms

        new_instance = self.__class__(**reconstruction_kwargs)

        # Copy random state from original instance to new instance
        if hasattr(self, "random_generator") and hasattr(self, "py_random"):
            new_instance.set_random_state(
                self.random_generator,
                self.py_random,
                runtime_context=self._rng_context,
                manual=self._manual_random_state,
            )

        return new_instance

    def _get_reconstruction_kwargs(self) -> dict[str, Any]:
        """Expose the constructor policy shared by serialization and graph edits, letting subclasses add behavior
        fields through one reconstruction contract.

        Subclasses extend this method with their class-specific policy. Both serialization
        and composition operators use this one projection.

        Returns:
            dict[str, Any]: Dictionary of initialization parameters

        """
        return {
            "p": self.p,
        }

    def __getstate__(self) -> dict[str, Any]:
        """Return portable configuration without locks or derived dispatch metadata, so workers rebuild their immutable
        plan rather than inherit runtime artifacts.

        The receiving process reconstructs immutable edge descriptors once instead
        of transporting call-time dispatch artifacts between workers.
        """
        state = self._get_invocation_pickle_state()
        state.pop("_compiled_children", None)
        state.pop("_unactivated_basic_children", None)
        state.pop("_unactivated_compiled_graph", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickled compose objects and clear runtime worker context so the first worker
        call can resynchronize against the active DataLoader seed.
        """
        self._restore_invocation_pickle_state(state)
        self.transforms = tuple(self.transforms)
        self._compile_execution_plan()


class Compose(BaseCompose, HubMixin):
    """Compose multiple transforms sequentially. Supports bbox_params, keypoint_params,
    additional_targets, strict, seed; supports +, -, __radd__.

    This class allows you to chain multiple image augmentation transforms and apply them
    in a specified order. It also handles bounding box and keypoint transformations if
    the appropriate parameters are provided.

    The configured child order is frozen after construction. Mathematical operators create
    a new Compose with the requested child sequence while preserving bbox, keypoint, and
    additional-target policy.

    Args:
        transforms (Sequence[BasicTransform | BaseCompose]): Ordered transforms to apply. Compose stores an immutable
            tuple of this configuration.
        bbox_params (dict[str, Any] | BboxParams | None): Parameters for bounding box transforms.
            Can be a dict of params or a BboxParams object. Default is None.
        keypoint_params (dict[str, Any] | KeypointParams | None): Parameters for keypoint transforms.
            Can be a dict of params or a KeypointParams object. Default is None.
        additional_targets (dict[str, str] | None): A dictionary mapping additional target names
            to their types. For example, {'image2': 'image'}. Passing a spatial alias also
            requires passing its canonical target (`image` in this example). Default is None.
        semantic_mask_label_mappings (dict[str, dict[int, int]] | None): Label replacements applied to spatial mask
            targets when a realized transform emits a label-mapping event. The outer key is the emitted event name;
            `D4` and `SquareSymmetry` emit the corresponding base reflection event (`HorizontalFlip`, `VerticalFlip`,
            or `Transpose`) rather than their class name. Other transforms may emit their own events, such as
            `Flip3D`. The inner dictionary maps source class IDs to target class IDs. `Flip3D` emits its event for a
            realized reflection across an odd number of axes and remaps `mask3d` and its aliases, not 2D `mask` or
            `masks` targets. Default: None.
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
        instance_binding (Sequence[str] | None): Targets that describe the same object in each
            `instances` item. Supported targets are `mask` or `masks`, `bboxes`, and `keypoints`.
            Compose transforms these targets together and removes all fields for an instance when
            its bbox fails bbox filtering. When masks and bboxes are bound, Compose also removes
            the instance if its transformed mask contains no non-zero pixels. Default is None.

    Examples:
        >>> # Basic usage:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.RandomCrop(width=256, height=256),
        ...     A.HorizontalFlip(p=0.5),
        ...     A.RandomBrightnessContrast(p=0.2),
        ... ], seed=137)
        >>> transformed = transform(image=image)

        >>> # Swap left/right semantic-mask class IDs after a realized horizontal flip:
        >>> transform = A.Compose(
        ...     [A.HorizontalFlip(p=1.0)],
        ...     semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        ... )
        >>> transformed = transform(image=image, mask=mask)

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
        - Semantic-mask mappings replace class IDs simultaneously, so paired swaps do not overwrite one another.
          Unmapped labels stay unchanged. For D4/SquareSymmetry, configure the realized reflection name:
          `HorizontalFlip`, `VerticalFlip`, or `Transpose`; identity and rotations do not remap labels.
          `Flip3D` emits its mapping event only when the realized reflection includes an odd number of axes.
        - Configure semantic-mask mappings only when the transformed and relabeled sample remains valid for your
          domain. Compose applies the declared mapping but cannot verify its semantic truth.
        - When strict mode is enabled, it performs additional validation to ensure data and transform
          configuration correctness.
        - Pipeline modification operators (+, -, __radd__) preserve all Compose parameters including
          bbox_params, keypoint_params, additional_targets, and other configuration settings.
        - All operators return new Compose instances without modifying the original pipeline.
        - Concurrent calls to one Compose instance, including `run_with_trace()`, keep all mutable execution
          state in separate invocations. A short root-only seed reservation establishes each caller's private
          random streams; pass `invocation_seed` when a sample must be independent of reservation order.

    """

    _filters_annotations_during_dispatch = True

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
        semantic_mask_label_mappings: dict[str, dict[int, int]] | None = None,
    ):
        super().__init__(
            transforms=transforms,
            p=p,
            mask_interpolation=mask_interpolation,
            seed=seed,
            save_applied_params=save_applied_params,
        )

        self.telemetry = telemetry
        self._resolve_processors(bbox_params, keypoint_params)

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self._instance_binding = self._setup_instance_binding(instance_binding)

        self.add_targets(additional_targets)
        normalized_mask_mappings = _normalize_semantic_mask_label_mappings(semantic_mask_label_mappings)
        self.semantic_mask_label_mappings = normalized_mask_mappings
        self._set_semantic_mask_label_mappings_for_transforms(self.transforms, normalized_mask_mappings)
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
        self._tensor_metadata_keys = self._declared_tensor_metadata_keys(self.transforms)
        self._has_tensor_terminal = self._contains_tensor_terminal(self.transforms)

        self.save_applied_params = save_applied_params

        # Telemetry runs after nested composes so main_compose=False is already set on them.
        self._maybe_send_telemetry(telemetry)

    def _set_semantic_mask_label_mappings_for_transforms(
        self,
        transforms: TransformsSeqType,
        mappings: dict[str, dict[int, int]] | None,
    ) -> None:
        effective_mappings = mappings or {}
        for transform in transforms:
            if isinstance(transform, DualTransform):
                transform.set_semantic_mask_label_mappings(effective_mappings)
            elif isinstance(transform, Compose):
                if transform.semantic_mask_label_mappings is None:
                    self._set_semantic_mask_label_mappings_for_transforms(transform.transforms, mappings)
            elif isinstance(transform, BaseCompose):
                self._set_semantic_mask_label_mappings_for_transforms(transform.transforms, mappings)

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
        user_to_internal = {user_name: internal_name for internal_name, user_name in self._kp_label_map.items()}
        kp_params.label_mapping = self._remap_label_mapping_fields(kp_params.label_mapping, user_to_internal)
        internal_fields.append(_KP_INSTANCE_ID)
        kp_params.label_fields = internal_fields
        kp_params.remove_invisible = False
        kp_params.check_each_transform = False

    @staticmethod
    def _remap_label_mapping_fields(
        label_mapping: dict[str, dict[str, dict[Any, Any]]] | None,
        field_map: dict[str, str],
    ) -> dict[str, dict[str, dict[Any, Any]]]:
        """Return a label_mapping copy with label-field keys renamed through field_map while preserving
        transform-specific mappings for each configured transform.

        Instance binding temporarily rewrites user keypoint label fields such as "name" to
        internal fields such as "_ibl_kp_name". The public label_mapping should still use the
        user field names, so Compose translates them when entering and leaving the bound path.
        """
        if not label_mapping:
            return {}
        return {
            transform_name: {field_map.get(field_name, field_name): mapping for field_name, mapping in fields.items()}
            for transform_name, fields in label_mapping.items()
        }

    def _set_check_args_for_transforms(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                self._set_check_args_for_transforms(transform.transforms)
                transform.check_each_transform = self.check_each_transform
            if isinstance(transform, Compose):
                transform.disable_check_args_private()

    def disable_check_args_private(self) -> None:
        """Disable argument checking. Sets is_check_args=False, strict=False, main_compose=False.
        Called for nested Compose so only top-level validates.
        """
        self.is_check_args = False
        self.strict = False
        self.main_compose = False

    def __call__(
        self,
        *args: Any,
        force_apply: bool = False,
        invocation_seed: int | None = None,
        **data: Any,
    ) -> dict[str, Any]:
        """Apply transformations with worker seed sync. Runs preprocess, each transform in
        order, check_data_post_transform, postprocess.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            invocation_seed (int | None): Optional sample-keyed seed for this root invocation.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        Raises:
            KeyError: If positional arguments are provided.

        """
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)
        return self._execute_root(
            data,
            force_apply=force_apply,
            invocation_seed=invocation_seed,
        )

    def _execute_root(
        self,
        data: dict[str, Any],
        *,
        force_apply: bool,
        invocation_seed: int | None,
        trace: _ExecutionTrace | None = None,
    ) -> dict[str, Any]:
        """Execute a public Compose call through one root boundary so normal and traced requests share
        validation, randomness, graph dispatch, and finalization.

        Ordinary and traced calls share validation, probability, preprocessing,
        dispatch, postprocessing, and tensor restoration.  Tracing only attaches an
        observer to this route; it does not own a second interpreter.
        """
        # Public calls never expose a completed observation from an earlier sample,
        # including when input validation fails before an invocation can open.
        clear_completed_observation()
        tensor_boundary_state = self._prepare_root_input(data)
        if not (force_apply or self.p > 0.0):
            clear_completed_observation()
            if trace is not None:
                self._emit_skipped_trace_subtree(trace, self, (), _TRACE_STATUS_SKIPPED_PROBABILITY)
            return data
        active_invocation = get_current_invocation()
        if (
            trace is None
            and type(self) is Compose
            and self._unactivated_compiled_graph
            and not self.replay_mode
            and not self.save_applied_params
            and "applied_transforms" not in data
            and not self._configured_processors
            and not self._instance_binding
            and active_invocation is None
            and self._can_execute_unactivated_ordinary_basic(data, tensor_boundary_state)
        ):
            context = self._new_invocation_context(
                collect_applied=False,
                invocation_seed=invocation_seed,
                prime_py_random=not force_apply and self.p < 1.0,
            )
            context.has_tensor_inputs = tensor_boundary_state.has_tensor_inputs
            if not self._should_apply_in_context(context, force_apply=force_apply):
                return data
            return self._apply_unactivated_compiled_transforms(data, context)

        context = self._new_invocation_context(
            collect_applied=True if trace is not None else None,
            invocation_seed=invocation_seed,
            prime_py_random=not force_apply and self.p < 1.0,
        )
        context.has_tensor_inputs = tensor_boundary_state.has_tensor_inputs
        context.trace_session = trace
        with context:
            if not self._should_apply_in_context(context, force_apply=force_apply):
                if trace is not None:
                    self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_PROBABILITY)
                return data
            result = self._apply_prepared_transforms(data, tensor_boundary_state, context)
            self._emit_trace_composition(trace)
            return result

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Execute a child Compose inside its parent root boundary, sharing invocation state without a
        second public preprocessing, validation, or postprocessing phase.

        This is intentionally separate from :meth:`__call__`: it does focused nested
        validation and graph dispatch only. It never opens processors, representation
        bridges, grayscale normalization, or a second finalization boundary.
        """
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)
        self.validate_nested_boundary(data)
        trace = invocation.trace_session
        if not self._should_apply_in_context(invocation, force_apply=force_apply):
            self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_PROBABILITY)
            return data
        result = self._apply_children(data, invocation)
        self._emit_trace_composition(trace)
        return result

    def _apply_prepared_transforms(
        self,
        data: dict[str, Any],
        tensor_boundary_state: _TensorBoundaryState,
        invocation: InvocationContext | None,
    ) -> dict[str, Any]:
        """Execute a validated root graph through its one public boundary."""
        self._open_root_boundary(data, tensor_boundary_state)
        return self._close_root_boundary(self._apply_children(data, invocation), tensor_boundary_state)

    def _apply_unactivated_compiled_transforms(
        self,
        data: dict[str, Any],
        invocation: InvocationContext,
    ) -> dict[str, Any]:
        """Run a compiled native graph without ContextVar activation when each node already receives all call-local
        sampling and execution state explicitly.

        Eligibility proves that the call has no Tensor bridge, grayscale repair,
        processors, instance binding, observation, or trace state. Built-in leaves
        and containers receive `invocation` directly, so activating it would only
        add two `ContextVar` operations and make repeated training slower.
        """
        self._validate_data(data)
        self._ensure_contiguous(data)
        if self._unactivated_basic_children is not None:
            for transform in self._unactivated_basic_children:
                data = transform.apply_in_invocation(invocation, force_apply=False, **data)
            return data
        return self._apply_unactivated_children(self, data, invocation)

    def _should_apply_in_context(self, context: InvocationContext, *, force_apply: bool) -> bool:
        """Evaluates root probability against the call-local Python stream, so concurrent calls never read a configured
        shared generator before the context becomes active.
        """
        if force_apply or self.p >= 1.0:
            return True
        return self.p > 0.0 and context.py_random.random() < self.p

    def _can_execute_unactivated_ordinary_basic(
        self,
        data: dict[str, Any],
        tensor_boundary_state: _TensorBoundaryState,
    ) -> bool:
        """Recognize a compiled native loop that needs an invocation but no ContextVar, selecting it only when
        repair, processors, tracing, and observation are absent.

        The loop passes its invocation to every built-in edge and sampler explicitly.
        Its only root-local state would be grayscale repair, Tensor validation,
        annotation processors, or shape validation, all of which select another
        executor.
        """
        if not self._unactivated_compiled_graph or tensor_boundary_state.has_tensor_inputs:
            return False

        shape_checked_inputs = 0
        for key, value in data.items():
            canonical_key = self._additional_targets.get(key, key)
            if not isinstance(value, np.ndarray):
                continue

            expected_ndim = self._GRAYSCALE_KEYS.get(canonical_key)
            if expected_ndim is not None and value.ndim == expected_ndim:
                return False
            if not self.is_check_shapes or canonical_key not in self._GRAYSCALE_KEYS or value.size == 0:
                continue

            if canonical_key in {"images", "masks", "volume", "mask3d"} and value.ndim not in {3, 4}:
                return False
            shape_checked_inputs += 1
            if shape_checked_inputs > 1:
                return False
        return True

    def validate_nested_boundary(self, data: dict[str, Any]) -> None:
        """Runs nested policy validation without a second boundary, preserving root Tensor conversion and preprocessing
        while checking aliases and shape consistency.
        """
        if self._additional_targets:
            self._validate_additional_target_sources(data)
        if self.is_check_shapes:
            shapes, volume_shapes = self._gather_shapes_from_data(data)
            self._check_shape_consistency(shapes, volume_shapes)

    def run_with_trace(
        self,
        *,
        options: TraceOptions | None = None,
        force_apply: bool = False,
        invocation_seed: int | None = None,
        **data: Any,
    ) -> TraceResult:
        """Apply this pipeline once and return normal output plus per-node records, keeping tracing local so ordinary
        execution retains its existing behavior.

        Args:
            options (TraceOptions | None): Per-call trace configuration that is never serialized with the pipeline.
            force_apply (bool): Apply this compose regardless of its probability.
            invocation_seed (int | None): Optional sample-keyed seed for this root invocation.
            **data (Any): Named targets accepted by :meth:`__call__`.

        Returns:
            TraceResult: Final targets and, unless observer-only mode was selected, trace records.

        """
        options = TraceOptions() if options is None else options
        self._validate_trace_options(options, data)
        trace = _ExecutionTrace(options)
        result = self._execute_root(
            data,
            force_apply=force_apply,
            invocation_seed=invocation_seed,
            trace=trace,
        )
        return trace.finish(result)

    def _validate_trace_options(self, options: TraceOptions, data: dict[str, Any]) -> None:
        known_targets = set(data) | self._available_keys | set(AVAILABLE_KEYS)
        unknown_targets = set(options.snapshot_targets).difference(known_targets)
        if unknown_targets:
            unknown = ", ".join(sorted(unknown_targets))
            raise ValueError(f"Unknown trace snapshot targets: {unknown}")

    def _invocation_state(self) -> ComposeInvocationState:
        """Returns mutable normalization and instance-binding state for this Compose, keeping temporary target
        restoration details out of configuration.
        """
        invocation = get_current_invocation()
        if invocation is None:
            msg = "Compose execution state requires an active invocation"
            raise RuntimeError(msg)
        return invocation.compose_state(self)

    def _prepare_root_input(self, data: dict[str, Any]) -> _TensorBoundaryState:
        """Validates one public input boundary and initializes observation storage before sampling, so invalid target
        combinations fail before random state advances.
        """
        if self._additional_targets:
            self._validate_additional_target_sources(data)
        tensor_boundary_state = self._validate_tensor_inputs(data)
        if self.save_applied_params and self.main_compose:
            data["applied_transforms"] = []
        return tensor_boundary_state

    def _open_root_boundary(self, data: dict[str, Any], tensor_boundary_state: _TensorBoundaryState) -> None:
        """Normalize public data before graph dispatch."""
        self.preprocess(data, tensor_boundary_state)

    def _close_root_boundary(self, data: dict[str, Any], tensor_boundary_state: _TensorBoundaryState) -> dict[str, Any]:
        """Finish root postprocessing and restore processor-owned Tensor annotations."""
        result = self.postprocess(data)
        if tensor_boundary_state.annotation_targets:
            self._restore_tensor_annotations(result, tensor_boundary_state)
        return result

    def _apply_children(self, data: dict[str, Any], invocation: InvocationContext | None) -> dict[str, Any]:
        """Runs child nodes and root policy in order, reusing the active invocation and resynchronizing bound instances
        only when the configured graph requires it.
        """
        track_applied = "applied_transforms" in data
        for index in range(len(self.transforms)):
            if not track_applied and "applied_transforms" in data:
                track_applied = True
            data = self._execute_child(
                index,
                data,
                invocation,
                track_applied=track_applied,
                resync_instances=bool(self.main_compose and self._instance_binding),
            )
        return data

    def _validate_additional_target_sources(self, data: dict[str, Any]) -> None:
        """Validate that every supplied spatial alias has a populated canonical source at the public Compose
        input boundary for each invocation.
        """
        for alias, target in self._additional_targets.items():
            if target in _SPATIAL_ADDITIONAL_TARGETS and data.get(alias) is not None and data.get(target) is None:
                msg = f"Additional target '{alias}' requires canonical target '{target}' to be present."
                raise ValueError(msg)

    def _validate_tensor_inputs(self, data: dict[str, Any]) -> _TensorBoundaryState:
        """Validate every supplied Tensor target before Compose samples probability or parameters."""
        has_direct_tensor = any(isinstance(value, torch.Tensor) for value in data.values())
        if not has_direct_tensor and (
            not self._tensor_metadata_keys or not any(key in data for key in self._tensor_metadata_keys)
        ):
            return _EMPTY_TENSOR_BOUNDARY_STATE

        annotation_targets: list[tuple[str, str]] = []
        has_tensor_target = False
        for data_name, value in data.items():
            canonical_name = self._additional_targets.get(data_name, data_name)
            if not isinstance(value, torch.Tensor) or canonical_name not in TENSOR_TARGETS:
                continue
            has_tensor_target = True
            validate_tensor_input(value, data_name, canonical_name)
            if canonical_name in TENSOR_ANNOTATION_TARGETS:
                annotation_targets.append((data_name, canonical_name))

        for path, metadata_target, value in self._iter_declared_tensor_metadata_inputs(self.transforms, data):
            has_tensor_target = True
            validate_tensor_metadata_input(value, path, metadata_target)

        if not has_tensor_target:
            return _EMPTY_TENSOR_BOUNDARY_STATE
        if self._has_tensor_terminal:
            raise TypeError(
                "ToTensorV2 and ToTensor3D accept NumPy input only; remove them from this Tensor Compose pipeline",
            )
        return _TensorBoundaryState(has_tensor_inputs=True, annotation_targets=tuple(annotation_targets))

    @staticmethod
    def _declared_tensor_metadata_keys(transforms: TransformsSeqType) -> frozenset[str]:
        """Cache metadata keys that may contain declared Tensor targets at the public boundary."""
        keys: set[str] = set()
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                keys.update(Compose._declared_tensor_metadata_keys(transform.transforms))
            elif isinstance(transform, BasicTransform):
                keys.update(transform.get_tensor_metadata_keys())
        return frozenset(keys)

    @staticmethod
    def _iter_declared_tensor_metadata_inputs(
        transforms: TransformsSeqType,
        data: Mapping[str, Any],
    ) -> Iterator[tuple[str, str | None, torch.Tensor]]:
        """Walk only declared target-bearing metadata before the root samples randomness."""
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                yield from Compose._iter_declared_tensor_metadata_inputs(transform.transforms, data)
            elif isinstance(transform, BasicTransform):
                yield from transform.iter_tensor_metadata_inputs(data)

    @staticmethod
    def _contains_tensor_terminal(transforms: TransformsSeqType) -> bool:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                if Compose._contains_tensor_terminal(transform.transforms):
                    return True
            elif getattr(transform, "_is_tensor_terminal", False):
                return True
        return False

    @staticmethod
    def from_applied_transforms(
        applied_transforms: list[tuple[str, dict[str, Any]]],
        **compose_kwargs: Any,
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
            **compose_kwargs (Any): Keyword arguments forwarded to the reconstructed `Compose`, such as
                `bbox_params`, `keypoint_params`, or `additional_targets`. The pipeline probability is always 1.0.

        Returns:
            Compose: A pipeline with p=1.0 for all transforms and constructor params
                fixed to the values sampled in the original run.

        """
        register_additional_transforms()
        transforms = []
        for class_name, config in applied_transforms:
            cls = SERIALIZABLE_REGISTRY[class_name]

            replay_config = _normalize_config_for_replay(cls, config)
            replay_config["p"] = 1.0

            transforms.append(cls(**replay_config))

        if "p" in compose_kwargs:
            msg = "from_applied_transforms() fixes the reconstructed Compose probability at 1.0; do not pass p"
            raise ValueError(msg)
        return Compose(transforms, p=1.0, **compose_kwargs)

    def preprocess(self, data: Any, tensor_boundary_state: _TensorBoundaryState | None = None) -> None:
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
        tensor_boundary_state = tensor_boundary_state or _EMPTY_TENSOR_BOUNDARY_STATE
        if tensor_boundary_state.annotation_targets:
            self._bridge_tensor_annotations_to_numpy(data, tensor_boundary_state)
        self._preprocess_processors(data)

    def _bridge_tensor_annotations_to_numpy(
        self,
        data: dict[str, Any],
        tensor_boundary_state: _TensorBoundaryState,
    ) -> None:
        """Convert accepted Tensor bbox and keypoint matrices once for existing NumPy-only
        processors, keeping all public annotations Tensor at Compose entry and exit.
        """
        for data_name, canonical_name in tensor_boundary_state.annotation_targets:
            value = data.get(data_name)
            if isinstance(value, torch.Tensor):
                data[data_name] = tensor_to_numpy_annotation(value, canonical_name)

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
        shape_check_targets = {"image", "mask", "images", "volume", "mask3d", "masks"}

        for data_name, data_value in data.items():
            # Resolve aliases via additional_targets so e.g. {'custom_image_key': 'image'}
            # gets the same shape-consistency check as the canonical 'image' key.
            canonical = self._additional_targets.get(data_name, data_name)
            if canonical not in shape_check_targets:
                continue

            # Skip empty data
            if data_value is None or not isinstance(data_value, (np.ndarray, torch.Tensor)):
                continue

            # Skip arrays with size 0 (empty arrays)
            if (isinstance(data_value, np.ndarray) and data_value.size == 0) or (
                isinstance(data_value, torch.Tensor) and data_value.numel() == 0
            ):
                continue

            self._process_data_shape(canonical, data_value, shapes, volume_shapes)

        return shapes, volume_shapes

    def _process_data_shape(
        self,
        data_name: str,
        data_value: np.ndarray | torch.Tensor,
        shapes: list[tuple[int, ...]],
        volume_shapes: list[tuple[int, ...]],
    ) -> None:
        """Process shape of a single data item. Appends (H,W) or (D,H,W) to shapes or
        volume_shapes depending on data_name (image, mask, images, volume, etc.).
        """
        if isinstance(data_value, torch.Tensor):
            self._process_tensor_data_shape(data_name, data_value, shapes, volume_shapes)
            return

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

    @staticmethod
    def _process_tensor_data_shape(
        data_name: str,
        data_value: torch.Tensor,
        shapes: list[tuple[int, ...]],
        volume_shapes: list[tuple[int, ...]],
    ) -> None:
        """Append shape metadata for a validated Tensor target without converting its public
        channel-first image or channel-free mask layout to a NumPy representation.
        """
        if data_name in {"image", "images", "mask", "masks", "volume", "mask3d"}:
            shapes.append(tuple(data_value.shape[-2:]))
        if data_name in {"volume", "mask3d"}:
            volume_shapes.append(tuple(data_value.shape[-3:]))

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
        if not self.main_compose or not self._configured_processors:
            return

        invocation = get_current_invocation()
        processors = (
            invocation.activate_processors(self._configured_processors)
            if invocation is not None
            else self._configured_processors
        )
        for processor in processors.values():
            processor.ensure_data_valid(data)
        for processor in processors.values():
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
        "mask3d": 3,  # (D, H, W) => (D, H, W, 1)
    }

    def _add_grayscale_channels(self, data: dict[str, Any]) -> None:
        """Normalize optional-channel public targets before transform dispatch."""
        invocation_state: ComposeInvocationState | None = None

        for key, value in data.items():
            canonical = self._additional_targets.get(key, key)
            expected_ndim = self._GRAYSCALE_KEYS.get(canonical)
            if expected_ndim is None:
                continue
            if not isinstance(value, (np.ndarray, torch.Tensor)) or value.ndim != expected_ndim:
                continue

            if isinstance(value, torch.Tensor) and canonical not in TENSOR_CHANNEL_AXIS:
                continue
            if invocation_state is None:
                invocation_state = self._invocation_state()
            original_value = value
            if isinstance(value, np.ndarray):
                data[key] = np.expand_dims(value, axis=-1)
            else:
                data[key] = value.unsqueeze(TENSOR_CHANNEL_AXIS[canonical])
            invocation_state.channel_restorations[key] = ChannelRestorationState(
                canonical_target=canonical,
                original_value=original_value,
                normalized_value=data[key],
            )

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply post-processing after all transforms. Runs processor postprocess and
        _remove_grayscale_channels when main_compose. Returns data dict.

        Args:
            data (dict[str, Any]): Data after transformation.

        Returns:
            dict[str, Any]: Post-processed data.

        """
        if self.main_compose:
            if self._instance_binding:
                self._filter_bound_bboxes_before_postprocess(data)

            if self._configured_processors:
                invocation = get_current_invocation()
                processors = self.processors
                for name, configured_processor in self._configured_processors.items():
                    processors[name].postprocess(
                        data,
                        filter_already_applied=(
                            invocation is not None and invocation.was_processor_filtered(configured_processor)
                        ),
                    )

            if self._instance_binding:
                invocation_state = self._invocation_state()
                if invocation_state.repack_after_processors:
                    self._repack_instances(data)

            # Remove channel dimensions that were added during preprocessing
            self._remove_grayscale_channels(data)

        return data

    def _restore_tensor_annotations(
        self,
        data: dict[str, Any],
        tensor_boundary_state: _TensorBoundaryState,
    ) -> None:
        """Restore Tensor bbox and keypoint matrices after NumPy processing so all spatial
        targets in the public Compose result remain Tensors.
        """
        for data_name, canonical_name in tensor_boundary_state.annotation_targets:
            value = data.get(data_name)
            if isinstance(value, np.ndarray):
                data[data_name] = numpy_to_tensor_annotation(value, canonical_name)

    def _filter_bound_bboxes_before_postprocess(self, data: dict[str, Any]) -> None:
        """Filter bound bboxes at the final boundary and mirror their keep mask before postprocessing removes internal
        instance ids required to align masks and keypoints.
        """
        binding = self._instance_binding
        if binding is None or "bboxes" not in binding:
            return

        bbox_processor = self.processors.get("bboxes")
        if not isinstance(bbox_processor, BboxProcessor):
            return

        configured_processor = self._configured_processors.get("bboxes")
        invocation = get_current_invocation()
        filter_already_applied = (
            configured_processor is not None
            and invocation is not None
            and invocation.was_processor_filtered(configured_processor)
        )
        if not filter_already_applied:
            self._bbox_filter_with_mirror(bbox_processor, data, get_shape(data), binding)
            if configured_processor is not None and invocation is not None:
                invocation.mark_processor_filtered(configured_processor)
        self._drop_instances_with_empty_bound_masks(data, binding)
        self._resync_instance_ids(data)

    def _remove_grayscale_channels(self, data: dict[str, Any]) -> None:
        """Restore optional public channel axes after transform dispatch."""
        invocation = get_current_invocation()
        if invocation is None:
            return
        invocation_state = invocation.get_compose_state(self)
        if invocation_state is None:
            return

        for key, restoration in invocation_state.channel_restorations.items():
            if key not in data:
                continue
            value = data[key]
            canonical = restoration.canonical_target

            if value is restoration.normalized_value:
                data[key] = restoration.original_value
            elif isinstance(value, np.ndarray) and value.shape[-1] == 1:
                data[key] = np.squeeze(value, axis=-1)
            elif (
                isinstance(value, torch.Tensor)
                and isinstance(restoration.original_value, np.ndarray)
                and canonical in {"mask", "masks"}
                and value.shape[-1] == 1
            ):
                data[key] = torch.squeeze(value, dim=-1)
            elif (
                isinstance(value, torch.Tensor)
                and not isinstance(restoration.original_value, np.ndarray)
                and value.shape[TENSOR_CHANNEL_AXIS[canonical]] == 1
            ):
                data[key] = torch.squeeze(value, dim=TENSOR_CHANNEL_AXIS[canonical])

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

    def _resync_instance_ids(self, data: dict[str, Any]) -> None:
        """Rebase the per-row instance id namespace to `arange(N)` and assert the structural
        invariant before the next transform sees the data.

        Phase 4 of the instance-binding rewrite: positional alignment is now upheld
        upstream — `Mosaic` shares one keep-mask across bboxes/masks/keypoints in
        `sample_parameters`, `CopyAndPaste` emits dense-id output, and
        `check_data_post_transform` mirrors any bbox-processor drop onto masks (positional)
        and keypoints (by surviving id). All this method has to do is:

        1. Assert that the bound `masks` row count or packed `mask` instance-axis count
           equals `len(bboxes)`.
        2. Translate `_kp_instance_id` from old bbox ids to the new positional ids.
        3. Stamp `_bbox_instance_id = arange(N)` so the next transform sees a dense namespace.

        The snapshot machinery (`_snapshot_pre_processor_bbox_ids`,
        `_mask_positions_for_surviving_ids`) the 2.2.2 hotfix added is intentionally
        deleted: it encoded a recovery branch for the dual-mask-layout case that no longer
        exists after phases 2/2b/3b.
        """
        binding = self._instance_binding
        if binding is None or "bboxes" not in binding:
            return
        bboxes_arr = data.get("bboxes")
        if not isinstance(bboxes_arr, np.ndarray) or bboxes_arr.shape[0] == 0:
            return

        n = bboxes_arr.shape[0]
        old_ids = bboxes_arr[:, -1].astype(np.int64, copy=True)

        self._raise_if_bound_mask_alignment_is_invalid(data, binding, n)

        if "keypoints" in binding:
            kp_arr = data.get("keypoints")
            if isinstance(kp_arr, np.ndarray) and kp_arr.shape[0] > 0:
                kp_ids_col = kp_arr[:, -1].astype(np.int64, copy=False)
                # Defense in depth: orphan kp ids (no matching bbox) should already be filtered
                # by `_bbox_filter_with_mirror`/`_prefilter_realign`. If we still see one here,
                # the upstream chain is broken and silently passing it through corrupts the new
                # positional namespace, so fail fast.
                bbox_id_set = set(old_ids.tolist())
                orphans = [int(k) for k in kp_ids_col.tolist() if int(k) not in bbox_id_set]
                if orphans:
                    self._raise_for_orphan_keypoints(orphans)
                if not np.array_equal(old_ids, np.arange(n, dtype=np.int64)):
                    old_to_new = {int(old): new for new, old in enumerate(old_ids.tolist())}
                    kp_arr[:, -1] = np.array(
                        [old_to_new[int(k)] for k in kp_ids_col],
                        dtype=kp_arr.dtype,
                    )

        bboxes_arr[:, -1] = np.arange(n, dtype=bboxes_arr.dtype)

    def _raise_if_bound_mask_alignment_is_invalid(
        self,
        data: dict[str, Any],
        binding: frozenset[str],
        bboxes_len: int,
    ) -> None:
        """Validate that stacked mask rows or packed mask channels remain aligned with bbox rows after each transform
        and before instance IDs are rebased.
        """
        if "masks" in binding:
            masks = data.get("masks")
            if masks is not None and isinstance(masks, (np.ndarray, torch.Tensor)) and len(masks) != bboxes_len:
                self._raise_for_invariant_violation(len(masks), bboxes_len)
        elif "mask" in binding:
            mask = data.get("mask")
            instance_axis = self._mask_instance_axis(data, mask)
            mask_count = None if mask is None or instance_axis is None else int(mask.shape[instance_axis])
            if mask_count != bboxes_len:
                self._raise_for_invariant_violation(mask_count, bboxes_len, target_name="mask")

    def _raise_for_invariant_violation(
        self,
        mask_count: int | None,
        bboxes_len: int,
        *,
        target_name: str = "masks",
    ) -> None:
        if target_name == "masks":
            mismatch = f"len(masks)={mask_count} != len(bboxes)={bboxes_len}"
            guidance = (
                "The last transform's `apply_to_masks` must keep masks positionally aligned with bboxes "
                "(one mask row per bbox row, in order). If this is custom transform code, share the per-row "
                "keep-mask across `apply_to_{bboxes,masks,keypoints}` instead of filtering each independently."
            )
        else:
            actual = "unresolved" if mask_count is None else str(mask_count)
            mismatch = f"packed mask instance count={actual} != len(bboxes)={bboxes_len}"
            guidance = (
                "The last transform's `apply_to_mask` must keep one packed mask channel per bbox row, in order. "
                "If this is custom transform code, share the instance keep-mask between `apply_to_bboxes` and "
                "`apply_to_mask` instead of filtering them independently."
            )
        msg = f"Instance-binding invariant violated: {mismatch}. {guidance}"
        raise RuntimeError(msg)

    def _raise_for_orphan_keypoints(self, orphans: list[int]) -> None:
        sample = orphans[:5]
        more = "" if len(orphans) <= 5 else f" (+{len(orphans) - 5} more)"
        msg = (
            f"Instance-binding invariant violated: keypoints reference "
            f"{len(orphans)} bbox id(s) that no longer exist: {sample}{more}. "
            "The last transform must drop keypoints whose parent bbox was filtered out, "
            "or `_bbox_filter_with_mirror` must mirror the bbox keep-mask onto keypoints."
        )
        raise RuntimeError(msg)

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
        invocation_state = self._invocation_state()
        invocation_state.instance_count = num_instances

        if num_instances == 0:
            self._init_empty_instance_data(data, binding)
            invocation_state.repack_after_processors = True
            return

        instance_dicts = self._validate_instances(instances)
        self._unpack_masks(data, binding, instance_dicts)
        self._unpack_bboxes(data, binding, instance_dicts, num_instances)
        self._unpack_keypoints(data, binding, instance_dicts)
        self._unpack_bbox_labels(data, instance_dicts)
        self._unpack_kp_labels(data, instance_dicts)
        invocation_state.repack_after_processors = True

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
            # Stack as (N, H, W); preprocessing expands it to canonical (N, H, W, 1) and records the added
            # trailing singleton in this invocation so the repack path restores the public mask shape.
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
        on the post-transform `_resync_instance_ids` invariant being in place.

        `_resync_instance_ids` runs every iteration of the run loop, so the row-alignment
        invariant holds here: when `bboxes` is bound, `_bbox_instance_id == range(N)` and (when
        `masks` is also bound) `len(data["masks"]) == N`. So bbox/mask/kp row indices all coincide
        and a single linear `for row_idx in range(n)` rebuilds the instance dicts. The invocation retains the original
        count for mask- or keypoint-only bindings where no bbox-driven filtering can reduce the instance set.
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
        invocation_state = self._invocation_state()
        n = len(bbox_ids) if "bboxes" in binding else invocation_state.instance_count
        if n is None:
            msg = "Instance binding must record its input count before repacking"
            raise RuntimeError(msg)
        bound_mask = self._bound_mask_with_instance_axis(data, binding, n)
        mask_instance_axis = bound_mask[1] if bound_mask is not None else None

        data["instances"] = [
            self._repack_one_instance(
                data,
                binding,
                bbox_row_idx=row_idx,
                mask_row_idx=row_idx,
                kp_group_id=row_idx,
                kp_ids=kp_ids,
                mask_instance_axis=mask_instance_axis,
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
        mask_instance_axis: int | None,
    ) -> dict[str, Any]:
        inst: dict[str, Any] = {}
        self._repack_mask_into(inst, data, binding, mask_row_idx, mask_instance_axis)
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
        mask_instance_axis: int | None,
    ) -> None:
        if "masks" in binding and "masks" in data:
            mask = data["masks"][original_instance_idx]
            added = "masks" in self._invocation_state().channel_restorations
            if added and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            inst["mask"] = mask
        elif "mask" in binding and "mask" in data and mask_instance_axis is not None:
            inst["mask"] = self._index_axis(data["mask"], original_instance_idx, mask_instance_axis)

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
            user_fields = [label_map.get(f, f) for f in label_fields if f not in _INSTANCE_ID_FERRY_KEYS]
            params_dict = {**params_dict, "label_fields": user_fields}
        if params_dict.get("label_mapping"):
            params_dict = {
                **params_dict,
                "label_mapping": self._remap_label_mapping_fields(params_dict["label_mapping"], label_map),
            }
        return params_dict

    def _get_reconstruction_kwargs(self) -> dict[str, Any]:
        """Return a complete detached Compose policy, including defaults, so construction routes recreate equivalent
        validation, random, processor, and telemetry behavior.

        """
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")

        bbox_params = self._clean_params_dict(
            bbox_processor.params.to_dict_private() if bbox_processor else None,
            self._bbox_label_map,
        )
        keypoint_params = self._clean_params_dict(
            keypoints_processor.params.to_dict_private() if keypoints_processor else None,
            self._kp_label_map,
        )

        return {
            "bbox_params": copy.deepcopy(bbox_params),
            "keypoint_params": copy.deepcopy(keypoint_params),
            "additional_targets": self.additional_targets.copy(),
            "semantic_mask_label_mappings": copy.deepcopy(self.semantic_mask_label_mappings),
            "p": self.p,
            "is_check_shapes": self.is_check_shapes,
            "strict": self.strict,
            "mask_interpolation": self.mask_interpolation,
            "seed": self._base_seed,
            "save_applied_params": self.save_applied_params,
            "telemetry": self.telemetry,
            "instance_binding": sorted(self._instance_binding) if self._instance_binding else None,
        }

    def get_dict_with_id(self) -> dict[str, Any]:
        """Get dict with object IDs for replay. Extends super with bbox_params,
        keypoint_params, additional_targets, params, is_check_shapes.

        Returns:
            dict[str, Any]: Dictionary with composition data and object IDs.

        """
        return super().get_dict_with_id()

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
        shape_check_targets = {"image", "mask", "images", "volume", "mask3d", "masks"}
        if internal_name not in shape_check_targets:
            return

        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError(f"{data_name} must be a NumPy array or torch.Tensor")

        # Skip arrays with size 0 (empty arrays)
        if (isinstance(data, np.ndarray) and data.size == 0) or (isinstance(data, torch.Tensor) and data.numel() == 0):
            return

        # Process the shape based on data type
        self._process_data_shape(internal_name, data, shapes, volume_shapes)

    def _check_shape_consistency(self, shapes: list[tuple[int, ...]], volume_shapes: list[tuple[int, ...]]) -> None:
        """Check consistency of shapes. When is_check_shapes, ensures all 2D shapes match
        and all 3D shapes match. Raises ValueError if inconsistent.
        """
        # Check H,W consistency
        self._check_shapes(shapes, self.is_check_shapes)

        # Check D,H,W consistency for volume data and 3D masks
        if self.is_check_shapes and volume_shapes and volume_shapes.count(volume_shapes[0]) != len(volume_shapes):
            raise ValueError(
                "Depth, Height and Width of volume and mask3d should be equal. "
                "You can disable shapes check by setting is_check_shapes=False.",
            )


class OneOf(BaseCompose):
    """Apply one of the child transforms at random; probabilities normalized as weights.
    Selected transform runs with force_apply=True.

    Args:
        transforms (Sequence[BasicTransform | BaseCompose]): Ordered transforms to choose from; stored immutably.
        p (float): probability of applying selected transform. Default: 0.5.

    """

    _filters_annotations_during_dispatch = True

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
        invocation = get_current_invocation()
        if invocation is None:
            invocation = self._new_invocation_context()
            with invocation:
                return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)
        return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Select and execute one child through parent state, preserving weighted branch choice without reopening a
        boundary or dynamically resolving invocation state.
        """
        trace = invocation.trace_session

        if self.replay_mode:
            if not self.applied_in_replay:
                self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_REPLAY)
                return data
            selected_indices = {
                index
                for index, transform in enumerate(self.transforms)
                if getattr(transform, "applied_in_replay", False)
            }
            if trace is not None:
                self._emit_unselected_trace_children(trace, self, selected_indices)
            for index in selected_indices:
                data = self._execute_child(index, data, invocation, track_applied="applied_transforms" in data)
            self._emit_trace_composition(trace)
            return data

        if not self.transforms_ps or not self._should_apply_in_context(invocation, force_apply=force_apply):
            self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_PROBABILITY)
            return data

        idx: int = invocation.random_generator.choice(len(self.transforms), p=self.transforms_ps)
        if trace is not None:
            self._emit_unselected_trace_children(trace, self, {idx})
        data = self._execute_child(
            idx,
            data,
            invocation,
            force_apply=True,
            track_applied="applied_transforms" in data,
        )
        self._emit_trace_composition(trace)
        return data


class SomeOf(BaseCompose):
    """Select exactly n transforms from the list and apply them. Selection uniform; each
    runs with its own p. Use replace=True for sampling with replacement.

    The selection of which `n` transforms to apply is done **uniformly at random**
    from the provided list. Each transform in the list has an equal chance of being selected.

    Once the `n` transforms are selected, each one is applied **based on its
    individual probability** `p`.

    Args:
        transforms (Sequence[BasicTransform | BaseCompose]): Ordered transforms to choose from; stored immutably.
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

    _filters_annotations_during_dispatch = True

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
        invocation = get_current_invocation()
        if invocation is None:
            invocation = self._new_invocation_context()
            with invocation:
                return self.apply_in_invocation(invocation, *arg, force_apply=force_apply, **data)
        return self.apply_in_invocation(invocation, *arg, force_apply=force_apply, **data)

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Select and execute children through parent state, preserving sampled order and replacement without reopening
        a boundary or resolving invocation per edge.
        """
        trace = invocation.trace_session

        if self.replay_mode:
            if not self.applied_in_replay:
                self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_REPLAY)
                return data
            selected_indices = {
                index
                for index, transform in enumerate(self.transforms)
                if getattr(transform, "applied_in_replay", False)
            }
            if trace is not None:
                self._emit_unselected_trace_children(trace, self, selected_indices)
            for index in selected_indices:
                data = self._execute_child(index, data, invocation, track_applied="applied_transforms" in data)
            self._emit_trace_composition(trace)
            return data

        if not self._should_apply_in_context(invocation, force_apply=force_apply):
            self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_PROBABILITY)
            return data

        indices_to_consider = self.sample_indices(invocation)
        if trace is not None:
            self._emit_unselected_trace_children(trace, self, set(indices_to_consider))
        for index in indices_to_consider:
            data = self._execute_child(int(index), data, invocation, track_applied="applied_transforms" in data)
        self._emit_trace_composition(trace)
        return data

    def sample_indices(self, invocation: InvocationContext) -> NDArray[np.int_]:
        """Sample SomeOf child indices and sort them into stable execution order, while retaining replacement
        behavior when a selection repeats the same child.

        Returns:
            NDArray[np.int_]: Selected child indices in execution order.

        """
        # Use uniform probability for selection, ignoring individual p values here.
        indices = invocation.random_generator.choice(
            len(self.transforms),
            size=self.n,
            replace=self.replace,
        )
        indices.sort()
        return indices

    def _get_reconstruction_kwargs(self) -> dict[str, Any]:
        base_params = super()._get_reconstruction_kwargs()
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
        transforms (TransformsSeqType): Ordered transformations to choose from; stored immutably.
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
        - Inherits from SomeOf, but overrides `get_indices` to ensure random order without sorting.
        - Selection is uniform; application depends on individual transform probabilities.

    """

    def __init__(self, transforms: TransformsSeqType, n: int = 1, replace: bool = False, p: float = 1):
        # Initialize using SomeOf's logic (which now does uniform selection setup)
        super().__init__(transforms=transforms, n=n, replace=replace, p=p)

    def sample_indices(self, invocation: InvocationContext) -> NDArray[np.int_]:
        """Sample RandomOrder child indices without sorting, preserving the chosen random order when selected
        transforms may not commute or may repeat.

        Returns:
            NDArray[np.int_]: Selected child indices in their sampled execution order.

        """
        # Preserve random order because selected transforms may not commute or may repeat.
        return invocation.random_generator.choice(
            len(self.transforms),
            size=self.n,
            replace=self.replace,
        )


class OneOrOther(BaseCompose):
    """Select one or the other transform. Selected runs with force_apply=True. Exactly two
    transforms; p chooses first vs second. Like OneOf n=2 but binary choice.
    """

    _filters_annotations_during_dispatch = True

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
        invocation = get_current_invocation()
        if invocation is None:
            invocation = self._new_invocation_context()
            with invocation:
                return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)
        return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Choose and execute one branch from parent random state, retaining binary selector semantics without a second
        boundary or configured-generator lookup.
        """
        trace = invocation.trace_session

        if self.replay_mode:
            if not self.applied_in_replay:
                self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_REPLAY)
                return data
            for index in range(len(self.transforms)):
                data = self._execute_child(index, data, invocation, track_applied="applied_transforms" in data)
            self._emit_trace_composition(trace)
            return data

        index = self.sample_branch_index(invocation)
        if trace is not None:
            self._emit_unselected_trace_children(trace, self, {index})
        data = self._execute_child(
            index,
            data,
            invocation,
            force_apply=True,
            track_applied="applied_transforms" in data,
        )
        self._emit_trace_composition(trace)
        return data

    def sample_branch_index(self, invocation: InvocationContext) -> int:
        """Chooses the first branch with probability `p` without deterministic draws, keeping selection reproducible and
        preserving stream positions for later sampling.
        """
        apply_first = self.p >= 1.0 or (self.p > 0.0 and invocation.py_random.random() < self.p)
        return 0 if apply_first else len(self.transforms) - 1


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
        invocation = get_current_invocation()
        if invocation is None:
            invocation = self._new_invocation_context()
            with invocation:
                return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)
        return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Apply channel-local children through parent state, extracting and reinserting channels without reopening
        preprocessing or resolving invocation per edge.
        """
        trace = invocation.trace_session

        if not self._should_apply_in_context(invocation, force_apply=force_apply):
            self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_PROBABILITY)
            return data

        image = data["image"]
        sub_image = np.ascontiguousarray(image[:, :, self.channels])

        def build_snapshot(sub_data: Mapping[str, Any]) -> Mapping[str, Any]:
            snapshot = data.copy()
            output_image = image.copy()
            for channel_index, channel in zip(self.channels, cv2.split(sub_data["image"]), strict=True):
                output_image[:, :, channel_index] = channel
            snapshot["image"] = np.ascontiguousarray(output_image)
            return snapshot

        if trace is None:
            for index in range(len(self.transforms)):
                sub_image = self._execute_child(
                    index,
                    {"image": sub_image},
                    invocation,
                    track_applied="applied_transforms" in data,
                    tracking_data=data,
                    finalize_annotations=False,
                )["image"]
        else:
            with trace.snapshot_scope(build_snapshot):
                for index in range(len(self.transforms)):
                    sub_image = self._execute_child(
                        index,
                        {"image": sub_image},
                        invocation,
                        track_applied="applied_transforms" in data,
                        tracking_data=data,
                        finalize_annotations=False,
                    )["image"]

        output_image = image.copy()
        for channel_index, channel in zip(self.channels, cv2.split(sub_image), strict=True):
            output_image[:, :, channel_index] = channel
        data["image"] = np.ascontiguousarray(output_image)
        self._emit_trace_composition(trace)
        return data

    def _get_reconstruction_kwargs(self) -> dict[str, Any]:
        """Extend the portable policy with channel selection, so serialization and graph edits retain channel-local
        augmentation without runtime image or trace data.

        """
        base_params = super()._get_reconstruction_kwargs()
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
            Ordered transformations to compose; stored immutably.
        bbox_params (dict[str, Any] | BboxParams | None):
            Parameters for bounding box transforms.
        keypoint_params (dict[str, Any] | KeypointParams | None):
            Parameters for keypoint transforms.
        additional_targets (dict[str, str] | None):
            Dictionary of additional targets.
        semantic_mask_label_mappings (dict[str, dict[int, int]] | None):
            Transform-aware semantic-mask class-ID replacements.
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
        semantic_mask_label_mappings: dict[str, dict[int, int]] | None = None,
        strict: bool = False,
        mask_interpolation: int | None = None,
        save_applied_params: bool = False,
        telemetry: bool = True,
    ):
        super().__init__(
            transforms,
            bbox_params,
            keypoint_params,
            additional_targets,
            p,
            is_check_shapes,
            strict=strict,
            mask_interpolation=mask_interpolation,
            seed=seed,
            save_applied_params=save_applied_params,
            telemetry=telemetry,
            instance_binding=instance_binding,
            semantic_mask_label_mappings=semantic_mask_label_mappings,
        )
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key
        self._available_keys.add(save_key)

    def _new_invocation_context(
        self,
        *,
        collect_applied: bool | None = None,
        invocation_seed: int | None = None,
        prime_py_random: bool = False,
    ) -> "InvocationContext":
        """Creates an observing ReplayCompose context, retaining leaf parameters for replay while keeping sampled values
        out of configured transform instances.
        """
        return super()._new_invocation_context(
            collect_applied=True,
            invocation_seed=invocation_seed,
            prime_py_random=prime_py_random,
        )

    def __call__(
        self,
        *args: Any,
        force_apply: bool = False,
        invocation_seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply transforms and record params for replay. Stores in save_key; fill_with_params
        and fill_applied complete serialized form for replay().

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            invocation_seed (int | None): Optional sample-keyed seed for this root invocation.
            **kwargs (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data and replay information.

        """
        kwargs[self.save_key] = defaultdict(dict)
        result = super().__call__(
            *args,
            force_apply=force_apply,
            invocation_seed=invocation_seed,
            **kwargs,
        )
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Execute a replay container through parent invocation, retaining serialization without creating
        another public root or duplicating preprocessing work.
        """
        data[self.save_key] = defaultdict(dict)
        result = super().apply_in_invocation(invocation, *args, force_apply=force_apply, **data)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    def run_with_trace(
        self,
        *,
        options: TraceOptions | None = None,
        force_apply: bool = False,
        invocation_seed: int | None = None,
        **data: Any,
    ) -> TraceResult:
        """Apply this replayable pipeline and return replay data plus a trace, without making observation
        configuration part of the saved augmentation payload.

        """
        data[self.save_key] = defaultdict(dict)
        trace_result = super().run_with_trace(
            options=options,
            force_apply=force_apply,
            invocation_seed=invocation_seed,
            **data,
        )
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, trace_result.data[self.save_key])
        self.fill_applied(serialized)
        trace_result.data[self.save_key] = serialized
        return trace_result

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
    def replay_with_trace(
        saved_augmentations: dict[str, Any],
        *,
        options: TraceOptions | None = None,
        **data: Any,
    ) -> TraceResult:
        """Replay saved augmentations with a structural trace, exposing recorded decisions without resampling
        probabilities or storing observation configuration.

        """
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        if not isinstance(augs, Compose):
            msg = "A replay trace requires a serialized Compose or ReplayCompose pipeline"
            raise TypeError(msg)
        return augs.run_with_trace(options=options, force_apply=True, **data)

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

    def fill_with_params(
        self,
        serialized: dict[str, Any],
        all_params: Any,
        occurrence_indices: dict[int, int] | None = None,
    ) -> None:
        """Fill serialized transform data with params for replay. Copies from all_params by
        id into serialized['params']; recurses into transforms. Mutates serialized.

        Args:
            serialized (dict[str, Any]): Serialized transform data.
            all_params (Any): Parameters to fill in.
            occurrence_indices (dict[int, int] | None): Number of consumed parameter payloads per configured object.

        """
        occurrence_indices = {} if occurrence_indices is None else occurrence_indices
        transform_id = serialized.get("id")
        if not isinstance(transform_id, int):
            msg = "serialized replay node has no integer id"
            raise TypeError(msg)
        params = all_params.get(transform_id)
        if isinstance(params, list):
            occurrence_index = occurrence_indices.get(transform_id, 0)
            params = params[occurrence_index] if occurrence_index < len(params) else None
            occurrence_indices[transform_id] = occurrence_index + 1
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params, occurrence_indices)

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

    def _get_reconstruction_kwargs(self) -> dict[str, Any]:
        base_params = super()._get_reconstruction_kwargs()
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

    _filters_annotations_during_dispatch = True

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
        invocation = get_current_invocation()
        if invocation is None:
            invocation = self._new_invocation_context()
            with invocation:
                return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)
        return self.apply_in_invocation(invocation, *args, force_apply=force_apply, **data)

    def apply_in_invocation(
        self,
        invocation: InvocationContext,
        /,
        *args: Any,
        force_apply: bool,
        **data: Any,
    ) -> dict[str, Any]:
        """Execute sequential children through parent state, preserving ordered probabilities without reopening
        preprocessing or resolving invocation per edge.
        """
        trace = invocation.trace_session

        if self.replay_mode:
            if not self.applied_in_replay:
                self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_REPLAY)
                return data
        elif not self._should_apply_in_context(invocation, force_apply=force_apply):
            self._emit_trace_skip(trace, _TRACE_STATUS_SKIPPED_PROBABILITY)
            return data

        for index in range(len(self.transforms)):
            data = self._execute_child(index, data, invocation, track_applied="applied_transforms" in data)
        self._emit_trace_composition(trace)
        return data
