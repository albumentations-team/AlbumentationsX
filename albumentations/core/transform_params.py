"""Invocation-local target metadata and structured transform parameters."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, overload

import numpy as np
import torch
from albucore import MAX_VALUES_BY_DTYPE

from .type_definitions import Targets
from .utils import get_volume_shape

_TARGET_ORDER = {
    "image": 0,
    "images": 1,
    "volume": 2,
    "mask": 3,
    "masks": 4,
    "mask3d": 5,
    "bboxes": 6,
    "keypoints": 7,
    "user_data": 8,
}
_PARAMETER_SCHEMA = 2
_PARAMETER_PAYLOAD_KEYS = frozenset({"parameter_schema", "target_schema", "params", "target_params"})
_TARGET_PARAMETER_PAYLOAD_KEYS = frozenset({"targets", "params", "requirements"})
_TARGET_REQUIREMENT_PAYLOAD_KEYS = frozenset(
    {
        "shape",
        "spatial_shape",
        "spatial_shape_suffix",
        "channels",
        "dtype",
        "value_scale",
        "layout",
        "sampling_topology",
    },
)


class SampledParamsError(ValueError):
    """Raised when sampled parameters cannot be applied to an invocation."""


@dataclass(frozen=True, slots=True)
class TargetDescriptor:
    """Representation metadata for one active target."""

    name: str
    canonical_type: str
    shape: tuple[int, ...] | None
    spatial_shape: tuple[int, ...] | None
    channels: int | None
    dtype: Any
    value_scale: float | None
    layout: str
    sampling_topology: str


@dataclass(frozen=True, slots=True)
class TargetView:
    """A borrowed target value and its invocation-local descriptor."""

    descriptor: TargetDescriptor
    value: Any

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def canonical_type(self) -> str:
        return self.descriptor.canonical_type


@dataclass(frozen=True, slots=True)
class TargetRequirement:
    """Representation constraints required by a materialized target parameter."""

    shape: tuple[int, ...] | None = None
    spatial_shape: tuple[int, ...] | None = None
    spatial_shape_suffix: tuple[int, ...] | None = None
    channels: int | None = None
    dtype: Any = None
    value_scale: float | None = None
    layout: str | None = None
    sampling_topology: str | None = None

    def check(self, descriptor: TargetDescriptor) -> bool:
        """Return whether a descriptor satisfies the declared materialization constraints."""
        return all(
            (expected is None or actual == expected or (field_name == "dtype" and str(actual) == str(expected)))
            for field_name, expected, actual in (
                ("shape", self.shape, descriptor.shape),
                ("spatial_shape", self.spatial_shape, descriptor.spatial_shape),
                (
                    "spatial_shape_suffix",
                    self.spatial_shape_suffix,
                    None
                    if descriptor.spatial_shape is None or self.spatial_shape_suffix is None
                    else descriptor.spatial_shape[-len(self.spatial_shape_suffix) :],
                ),
                ("channels", self.channels, descriptor.channels),
                ("dtype", self.dtype, descriptor.dtype),
                ("value_scale", self.value_scale, descriptor.value_scale),
                ("layout", self.layout, descriptor.layout),
                ("sampling_topology", self.sampling_topology, descriptor.sampling_topology),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "spatial_shape": self.spatial_shape,
            "spatial_shape_suffix": self.spatial_shape_suffix,
            "channels": self.channels,
            "dtype": None if self.dtype is None else str(self.dtype),
            "value_scale": self.value_scale,
            "layout": self.layout,
            "sampling_topology": self.sampling_topology,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetRequirement:
        if set(payload) != _TARGET_REQUIREMENT_PAYLOAD_KEYS:
            raise SampledParamsError("unsupported target parameter requirement schema")
        return cls(
            shape=None if payload.get("shape") is None else tuple(payload["shape"]),
            spatial_shape=None if payload.get("spatial_shape") is None else tuple(payload["spatial_shape"]),
            spatial_shape_suffix=(
                None if payload.get("spatial_shape_suffix") is None else tuple(payload["spatial_shape_suffix"])
            ),
            channels=payload.get("channels"),
            dtype=payload.get("dtype"),
            value_scale=payload.get("value_scale"),
            layout=payload.get("layout"),
            sampling_topology=payload.get("sampling_topology"),
        )


@dataclass(frozen=True, slots=True)
class TargetParams:
    """Parameters for a named set of actual targets with compatible representations."""

    targets: tuple[str, ...]
    params: Mapping[str, Any]
    requirements: Mapping[str, TargetRequirement]

    def __post_init__(self) -> None:
        if not self.targets:
            raise SampledParamsError("target parameter sets cannot be empty")
        if len(set(self.targets)) != len(self.targets):
            raise SampledParamsError("target parameter sets cannot repeat target keys")
        if set(self.targets) != set(self.requirements):
            raise SampledParamsError("target parameter sets require one requirement per target")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "requirements", dict(self.requirements))

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "params": dict(self.params),
            "requirements": {name: requirement.to_dict() for name, requirement in self.requirements.items()},
        }


class TargetSet:
    """Deterministically ordered active targets for one transform step."""

    def __init__(self, views: tuple[TargetView, ...]) -> None:
        self.ordered = views
        self._by_name = {view.name: view for view in views}
        self._aligned_spatial_shapes: dict[int, tuple[int, ...] | None] = {}

    @classmethod
    def from_data(cls, data: Mapping[str, Any], canonical_by_name: Mapping[str, str]) -> TargetSet:
        active = [(name, value) for name, value in data.items() if value is not None and name in canonical_by_name]
        if len(active) == 1:
            name, value = active[0]
            return cls((TargetView(_describe_target(name, canonical_by_name[name], value), value),))
        views = [TargetView(_describe_target(name, canonical_by_name[name], value), value) for name, value in active]
        views.sort(key=lambda view: _target_sort_key(view.descriptor))
        return cls(tuple(views))

    def by_name(self, name: str) -> TargetView:
        return self._by_name[name]

    def by_canonical_type(self, target_type: Targets | str) -> tuple[TargetView, ...]:
        canonical = target_type.value if isinstance(target_type, Targets) else target_type
        return tuple(view for view in self.ordered if view.canonical_type == canonical)

    def image_like(self) -> tuple[TargetView, ...]:
        return tuple(view for view in self.ordered if view.canonical_type in {"image", "images", "volume"})

    def primary_image_like(self) -> TargetView:
        """Return the canonical primary raster target used by image-level policies."""
        for view in self.ordered:
            if view.canonical_type == "image":
                return view
        for view in self.ordered:
            if view.canonical_type in {"images", "volume"}:
                return view
        raise SampledParamsError("transform sampling requires an image-like target")

    def group_by(self, key: Callable[[TargetView], Hashable]) -> tuple[tuple[TargetView, ...], ...]:
        grouped: dict[Hashable, list[TargetView]] = {}
        for view in self.ordered:
            grouped.setdefault(key(view), []).append(view)
        return tuple(tuple(views) for views in grouped.values())

    def group_image_like_by(self, key: Callable[[TargetView], Hashable]) -> tuple[tuple[TargetView, ...], ...]:
        """Group only image-like targets, excluding annotations and user data."""
        grouped: dict[Hashable, list[TargetView]] = {}
        for view in self.image_like():
            grouped.setdefault(key(view), []).append(view)
        return tuple(tuple(views) for views in grouped.values())

    @overload
    def aligned_spatial_shape(self, rank: Literal[2]) -> tuple[int, int] | None: ...

    @overload
    def aligned_spatial_shape(self, rank: Literal[3]) -> tuple[int, int, int] | None: ...

    @overload
    def aligned_spatial_shape(self, rank: int) -> tuple[int, ...] | None: ...

    def aligned_spatial_shape(self, rank: int) -> tuple[int, ...] | None:
        """Return the spatial shape after verifying that all relevant targets are aligned."""
        if rank not in {2, 3}:
            raise ValueError(f"spatial rank must be 2 or 3, got {rank}")
        if rank in self._aligned_spatial_shapes:
            return self._aligned_spatial_shapes[rank]

        shapes = {
            shape
            for view in self.ordered
            if _target_has_elements(view.value)
            if (shape := _spatial_shape_for_rank(view, rank)) is not None
        }
        if not shapes:
            fallback = next(
                (shape for view in self.ordered if (shape := _spatial_shape_for_rank(view, rank)) is not None),
                None,
            )
            self._aligned_spatial_shapes[rank] = fallback
            return fallback
        if len(shapes) != 1:
            raise SampledParamsError(f"transform sampling requires aligned spatial targets, got {sorted(shapes)}")

        shape = next(iter(shapes))
        self._aligned_spatial_shapes[rank] = shape
        return shape

    @overload
    def require_aligned_spatial_shape(self, rank: Literal[2]) -> tuple[int, int]: ...

    @overload
    def require_aligned_spatial_shape(self, rank: Literal[3]) -> tuple[int, int, int]: ...

    @overload
    def require_aligned_spatial_shape(self, rank: int) -> tuple[int, ...]: ...

    def require_aligned_spatial_shape(self, rank: int) -> tuple[int, ...]:
        """Return the spatial shape required by a sampler with aligned targets."""
        shape = self.aligned_spatial_shape(rank)
        if shape is None:
            raise SampledParamsError("transform sampling requires a spatial target")
        return shape

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def schema(self) -> dict[str, str]:
        return {view.name: view.canonical_type for view in self.ordered}


@dataclass(frozen=True, slots=True)
class SampledParams:
    """Parameters and target-specific values sampled for one transform event."""

    params: Mapping[str, Any]
    target_params: tuple[TargetParams, ...] = ()
    target_schema: Mapping[str, str] | None = None

    def validate(
        self,
        targets: TargetSet,
        required_params: Mapping[str, frozenset[str]],
        transform_name: str,
    ) -> None:
        active_names = targets.names
        self._validate_target_schema(targets, active_names, transform_name)
        seen = self._validate_target_params(targets, active_names, transform_name)
        self._validate_required(required_params, seen, transform_name)

    def _validate_target_schema(
        self,
        targets: TargetSet,
        active_names: frozenset[str],
        transform_name: str,
    ) -> None:
        if self.target_schema is None:
            return
        if set(self.target_schema) != active_names:
            raise SampledParamsError(
                f"{transform_name} sampled parameter targets do not match the active invocation",
            )
        for name, canonical in self.target_schema.items():
            if targets.by_name(name).canonical_type != canonical:
                raise SampledParamsError(
                    f"{transform_name} sampled parameter canonical target mismatch for {name!r}",
                )

    def _validate_target_params(
        self,
        targets: TargetSet,
        active_names: frozenset[str],
        transform_name: str,
    ) -> dict[str, set[str]]:
        seen: dict[str, set[str]] = {}
        for target_params in self.target_params:
            if any(name not in active_names for name in target_params.targets):
                raise SampledParamsError(f"{transform_name} sampled parameters name an inactive target")
            for name in target_params.targets:
                descriptor = targets.by_name(name).descriptor
                requirement = target_params.requirements[name]
                if not requirement.check(descriptor):
                    raise SampledParamsError(
                        f"{transform_name} sampled parameter requirements do not match target {name!r}",
                    )
                duplicate = seen.setdefault(name, set()).intersection(target_params.params)
                if duplicate or set(target_params.params).intersection(self.params):
                    raise SampledParamsError(
                        f"{transform_name} sampled parameters have duplicate keys for target {name!r}",
                    )
                seen[name].update(target_params.params)
        return seen

    def _validate_required(
        self,
        required_params: Mapping[str, frozenset[str]],
        seen: Mapping[str, set[str]],
        transform_name: str,
    ) -> None:
        for name, required in required_params.items():
            available = set(self.params).union(seen.get(name, set()))
            missing = required - available
            if missing:
                raise SampledParamsError(
                    f"{transform_name} sampled parameters are missing {sorted(missing)} for target {name!r}",
                )

    def params_for(self, target_name: str) -> Mapping[str, Any]:
        if not self.target_params:
            return self.params
        resolved = dict(self.params)
        for target_params in self.target_params:
            if target_name in target_params.targets:
                resolved.update(target_params.params)
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_schema": _PARAMETER_SCHEMA,
            "target_schema": None if self.target_schema is None else dict(self.target_schema),
            "params": dict(self.params),
            "target_params": [target_params.to_dict() for target_params in self.target_params],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SampledParams:
        if payload.get("parameter_schema") != _PARAMETER_SCHEMA or set(payload) != _PARAMETER_PAYLOAD_KEYS:
            raise SampledParamsError("unsupported or legacy transform parameter schema")
        raw_target_params = payload.get("target_params", ())
        if not isinstance(raw_target_params, list):
            raise SampledParamsError("target parameters must be a list")
        target_params: list[TargetParams] = []
        for item in raw_target_params:
            if not isinstance(item, Mapping) or set(item) != _TARGET_PARAMETER_PAYLOAD_KEYS:
                raise SampledParamsError("unsupported target parameter schema")
            raw_requirements = item["requirements"]
            if not isinstance(raw_requirements, Mapping):
                raise SampledParamsError("target parameter requirements must be a mapping")
            requirements: dict[str, TargetRequirement] = {}
            for name, requirement in raw_requirements.items():
                if not isinstance(name, str) or not isinstance(requirement, Mapping):
                    raise SampledParamsError(
                        "target parameter requirements must map target names to requirement objects"
                    )
                requirements[name] = TargetRequirement.from_dict(requirement)
            target_params.append(
                TargetParams(
                    targets=tuple(item["targets"]),
                    params=dict(item["params"]),
                    requirements=requirements,
                ),
            )
        target_schema = payload.get("target_schema")
        return cls(
            params=dict(payload.get("params", {})),
            target_params=tuple(target_params),
            target_schema=None if target_schema is None else dict(target_schema),
        )


def _target_has_elements(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    size = getattr(value, "size", None)
    return not isinstance(size, int) or size > 0


def _spatial_shape_for_rank(view: TargetView, rank: int) -> tuple[int, ...] | None:
    spatial_shape = view.descriptor.spatial_shape
    if spatial_shape is None:
        return None
    if rank == 3:
        return tuple(spatial_shape) if view.canonical_type in {"volume", "mask3d"} and len(spatial_shape) == 3 else None
    return tuple(spatial_shape[-2:]) if len(spatial_shape) >= 2 else None


def _target_sort_key(descriptor: TargetDescriptor) -> tuple[int, int, str]:
    canonical = descriptor.canonical_type
    priority = _TARGET_ORDER.get(canonical, len(_TARGET_ORDER))
    return priority, 0 if descriptor.name == canonical else 1, descriptor.name


def _describe_target(name: str, canonical_type: str, value: Any) -> TargetDescriptor:
    shape = tuple(int(dim) for dim in value.shape) if hasattr(value, "shape") else None
    dtype = getattr(value, "dtype", None)
    volume_shape = (
        get_volume_shape(value)
        if canonical_type in {"volume", "mask3d"} and isinstance(value, np.ndarray | torch.Tensor)
        else None
    )
    return _describe_target_cached(name, canonical_type, shape, dtype, isinstance(value, torch.Tensor), volume_shape)


def _value_scale(dtype: Any) -> float | None:
    if dtype is None:
        return None
    try:
        return MAX_VALUES_BY_DTYPE.get(np.dtype(str(dtype).removeprefix("torch.")))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4096)
def _describe_target_cached(
    name: str,
    canonical_type: str,
    shape: tuple[int, ...] | None,
    dtype: Any,
    tensor: bool,
    volume_shape: tuple[int, int, int] | None,
) -> TargetDescriptor:
    layout = canonical_type
    topology = canonical_type
    channels: int | None = None
    spatial_shape: tuple[int, ...] | None = None

    if shape is not None:
        if canonical_type == "image":
            spatial_shape = shape[1:3] if tensor else shape[:2]
            channels = shape[0] if tensor else (shape[-1] if len(shape) > 2 else 1)
            layout, topology = ("image_chw", "image_2d") if tensor else ("image_hwc", "image_2d")
        elif canonical_type == "images":
            spatial_shape = shape[2:4] if tensor else shape[1:3]
            channels = shape[1] if tensor else (shape[-1] if len(shape) > 3 else 1)
            layout, topology = ("images_nchw", "batch_2d") if tensor else ("images_nhwc", "batch_2d")
        elif canonical_type == "volume":
            spatial_shape = volume_shape
            channels = shape[0] if tensor else (shape[-1] if len(shape) > 3 else 1)
            layout, topology = ("volume_cdhw", "volume_3d") if tensor else ("volume_dhwc", "volume_3d")
        elif canonical_type == "mask":
            spatial_shape = shape[1:3] if tensor else shape[:2]
            channels = shape[0] if tensor else (shape[-1] if len(shape) > 2 else 1)
            layout, topology = ("mask_chw", "mask_2d") if tensor else ("mask_hwc", "mask_2d")
        elif canonical_type == "masks":
            spatial_shape = shape[2:4] if tensor else shape[1:3]
            channels = shape[1] if tensor else (shape[-1] if len(shape) > 3 else 1)
            layout, topology = ("masks_nchw", "batch_mask_2d") if tensor else ("masks_nhwc", "batch_mask_2d")
        elif canonical_type == "mask3d":
            spatial_shape = volume_shape
            channels = shape[0] if tensor else (shape[-1] if len(shape) > 3 else 1)
            layout, topology = ("mask3d_cdhw", "mask_3d") if tensor else ("mask3d_dhwc", "mask_3d")

    return TargetDescriptor(
        name,
        canonical_type,
        shape,
        spatial_shape,
        channels,
        dtype,
        _value_scale(dtype),
        layout,
        topology,
    )


@lru_cache(maxsize=1024)
def required_parameter_names(function: Callable[..., Any]) -> frozenset[str]:
    """Return required keyword parameters after the target value argument."""
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())[1:]
    return frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        and parameter.default is inspect.Parameter.empty
    )


def requirements_for_views(
    views: tuple[TargetView, ...],
    *,
    shape: bool = False,
    spatial_shape: bool = False,
    spatial_shape_suffix: bool = False,
    channels: bool = False,
    dtype: bool = False,
    value_scale: bool = False,
    layout: bool = False,
    sampling_topology: bool = False,
) -> dict[str, TargetRequirement]:
    """Build replay constraints from a deterministic compatible target group."""
    return {
        view.name: TargetRequirement(
            shape=view.descriptor.shape if shape else None,
            spatial_shape=view.descriptor.spatial_shape if spatial_shape else None,
            spatial_shape_suffix=(
                None
                if not spatial_shape_suffix or view.descriptor.spatial_shape is None
                else tuple(view.descriptor.spatial_shape[-2:])
            ),
            channels=view.descriptor.channels if channels else None,
            dtype=view.descriptor.dtype if dtype else None,
            value_scale=view.descriptor.value_scale if value_scale else None,
            layout=view.descriptor.layout if layout else None,
            sampling_topology=view.descriptor.sampling_topology if sampling_topology else None,
        )
        for view in views
    }


__all__ = [
    "SampledParams",
    "SampledParamsError",
    "TargetDescriptor",
    "TargetParams",
    "TargetRequirement",
    "TargetSet",
    "TargetView",
    "required_parameter_names",
    "requirements_for_views",
]
