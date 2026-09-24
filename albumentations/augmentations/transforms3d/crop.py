"""Explicit integer-origin crops on aligned volume targets."""

import math
from collections.abc import Mapping
from numbers import Integral
from typing import Annotated, Any, Literal, cast

import numpy as np
import torch
from pydantic import AfterValidator, BeforeValidator, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import check_range_bounds
from albumentations.core.transform_params import SampledParams, TargetParams, TargetRequirement, TargetSet
from albumentations.core.transforms_interface import BaseTransformInitSchema, Transform3D
from albumentations.core.type_definitions import Targets, VolumeType

__all__ = ["Crop3D"]


def _integer_triplet(value: Any) -> tuple[int, int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in value)
    ):
        raise ValueError("Expected three integer voxel coordinates, excluding bool")
    return int(value[0]), int(value[1]), int(value[2])


def _finite_fill(value: float) -> int | float:
    if not math.isfinite(value):
        raise ValueError("fill must be finite")
    return value


def _preserve_integer_fill(value: Any) -> Any:
    return int(value) if isinstance(value, np.integer) else value


_FillValue = Annotated[int | float, BeforeValidator(_preserve_integer_fill), AfterValidator(_finite_fill)]


def _origin_key(value: str) -> str:
    if not value:
        raise ValueError("origin_key must be nonempty")
    return value


def _overrides(value: dict[str, dict[str, int | float]]) -> dict[str, dict[str, int | float]]:
    if any(not name or set(policy) != {"fill"} for name, policy in value.items()):
        raise ValueError("Each target override must specify only fill for a named mask")
    return value


def _validate_floating_fill(value: float, maximum: float) -> None:
    if abs(value) > maximum:
        raise ValueError("fill must remain finite in the target dtype")


def _validate_fill(value: float, dtype: Any) -> None:
    if isinstance(dtype, torch.dtype):
        if dtype.is_floating_point:
            _validate_floating_fill(value, torch.finfo(dtype).max)
            return
        tensor_bounds = torch.iinfo(dtype)
        minimum, maximum = tensor_bounds.min, tensor_bounds.max
    else:
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            _validate_floating_fill(value, float(np.finfo(dtype).max))
            return
        if dtype.kind not in "biu":
            return
        if dtype.kind == "b":
            if value not in (0, 1):
                raise ValueError("Boolean mask fill must be 0 or 1")
            return
        array_bounds = np.iinfo(dtype)
        minimum, maximum = array_bounds.min, array_bounds.max
    if int(value) != value or not minimum <= value <= maximum:
        raise ValueError(f"fill must be an integer representable in {dtype}")


class Crop3D(Transform3D):
    """Extract a voxel window at a supplied integer origin, with optional constant padding, to select aligned
    regions of volumes and masks.

    The caller supplies the start in depth, height, width order. This transform preserves that start,
    including negative coordinates, and copies values exactly without interpolation or anatomical localization.

    Args:
        size (tuple[int, int, int]): Positive output depth, height and width in voxels.
        origin (tuple[int, int, int] | None): Signed integer start in DHW order. Exactly one of origin and
            origin_key must be set. Default: None.
        origin_key (str | None): Key in per-call user_data containing the signed integer DHW start. Default: None.
        origin_space (Literal["voxel"]): Coordinate units. Only voxel coordinates are supported. Default: "voxel".
        pad_if_needed (bool): Allow windows extending outside the source, including wholly outside windows.
            When False, any out-of-bounds window raises ValueError. Default: False.
        fill (int | float): Constant volume padding value, representable in the input dtype. Default: 0.
        fill_mask (int | float): Default mask padding value. Integer masks require an exactly representable
            integer fill; Boolean masks require 0 or 1. Default: 0.
        target_overrides (dict[str, dict[str, int | float]] | None): Named registered mask overrides containing
            only fill, for example {"coverage": {"fill": 0}}. These targets must be present when applied.
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - NumPy volumes use DHW or DHWC through Compose; CPU Tensor volumes use CDHW.
        - All aligned targets share one window. XYZ keypoints move by minus the reversed DHW origin.
        - Inputs are not mutated. In-bounds crops may share input storage; padded results own their allocation.
        - This voxel-only transform does not update physical metadata. Volume batches are unsupported.
        - Applied configurations freeze the integer origin. Replay also requires the captured input spatial shape
          and rejects a supplied per-call origin that differs from the captured one.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.ones((3, 5, 7), dtype=np.float32)
        >>> transform = A.Compose([A.Crop3D((2, 3, 4), origin=(-1, 1, 2), pad_if_needed=True)])
        >>> result = transform(volume=volume)
        >>> result["volume"].shape
        (2, 3, 4)

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[
            tuple[int, int, int],
            BeforeValidator(_integer_triplet),
            AfterValidator(check_range_bounds(1, None)),
        ]
        origin: Annotated[tuple[int, int, int], BeforeValidator(_integer_triplet)] | None
        origin_key: Annotated[str, AfterValidator(_origin_key)] | None
        origin_space: Literal["voxel"]
        pad_if_needed: bool
        fill: _FillValue
        fill_mask: _FillValue
        target_overrides: Annotated[dict[str, dict[str, _FillValue]], AfterValidator(_overrides)] | None

        @model_validator(mode="after")
        def _check_origin_source(self) -> Self:
            if (self.origin is None) == (self.origin_key is None):
                raise ValueError("Set exactly one of origin or origin_key")
            return self

    def __init__(
        self,
        size: tuple[int, int, int],
        origin: tuple[int, int, int] | None = None,
        origin_key: str | None = None,
        origin_space: Literal["voxel"] = "voxel",
        pad_if_needed: bool = False,
        fill: float = 0,
        fill_mask: float = 0,
        target_overrides: dict[str, dict[str, int | float]] | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.size = size
        self.origin = origin
        self.origin_key = origin_key
        self.origin_space = origin_space
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.fill_mask = fill_mask
        self.target_overrides = target_overrides

    def _resolve_origin(
        self,
        data: Mapping[str, Any],
        fallback: tuple[int, int, int] | None = None,
    ) -> tuple[int, int, int]:
        if self.origin is not None:
            return self.origin
        user = data.get("user_data", {})
        if not isinstance(user, Mapping):
            raise TypeError("user_data must be a mapping")
        if self.origin_key not in user:
            if fallback is not None:
                return fallback
            raise ValueError(f"Missing origin in user_data[{self.origin_key!r}]")
        return _integer_triplet(user[self.origin_key])

    def _validate_inputs(self, data: Mapping[str, Any]) -> None:
        if data.get("volume") is None:
            raise ValueError("Crop3D requires volume")
        if any(
            self._additional_targets.get(name, name) in {"volumes", "masks3d", "bboxes"}
            for name, value in data.items()
            if value is not None
        ):
            raise ValueError("Crop3D does not support volume batches or bboxes")

    def _build_target_set(self, data: Mapping[str, Any]) -> TargetSet:
        # Origin is captured as a parameter; replay must not require the original control dictionary.
        targets = super()._build_target_set(data)
        return TargetSet(tuple(view for view in targets.ordered if view.canonical_type != "user_data"))

    def _target_parameters(self, targets: TargetSet) -> tuple[TargetParams, ...]:
        overrides = self.target_overrides or {}
        shape = targets.by_name("volume").descriptor.spatial_shape
        masks = {view.name for view in targets.by_canonical_type("mask3d")}
        if not set(overrides) <= masks:
            raise ValueError("Every target override must name a present registered mask3d target")
        groups = []
        for view in targets.ordered:
            if view.canonical_type not in {"volume", "mask3d"}:
                continue
            if view.descriptor.spatial_shape != shape:
                raise ValueError("Crop3D requires aligned spatial shapes, including empty targets")
            default = cast("float", self.fill_mask if view.canonical_type == "mask3d" else self.fill)
            fill = overrides.get(view.name, {}).get("fill", default)
            _validate_fill(fill, view.descriptor.dtype)
            groups.append(
                TargetParams(
                    targets=(view.name,),
                    params={"target_fill": fill},
                    requirements={view.name: TargetRequirement(spatial_shape=view.descriptor.spatial_shape)},
                ),
            )
        return tuple(groups)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        self._validate_inputs(data)
        shape = targets.require_aligned_spatial_shape(3)
        origin = self._resolve_origin(data)
        crop_coords, padding = f3d.crop3d_window(shape, origin, self.size, self.pad_if_needed)
        sampling.applied_overrides.update(origin=origin, origin_key=None)
        return SampledParams(
            params={"crop_coords": crop_coords, "padding": padding, "origin": origin, "input_shape": shape},
            target_params=self._target_parameters(targets),
        )

    def apply_with_params(self, sampled_params: SampledParams, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Apply the captured window, validating fresh replay inputs and any supplied origin."""
        if self.replay_mode:
            self._validate_inputs(kwargs)
            targets = self._build_target_set(kwargs)
            if targets.require_aligned_spatial_shape(3) != tuple(sampled_params.params["input_shape"]):
                raise ValueError("Replay input shape differs from the captured shape")
            origin = tuple(sampled_params.params["origin"])
            if self._resolve_origin(kwargs, origin) != origin:
                raise ValueError("Replay origin differs from the captured origin")
            self._target_parameters(targets)
        return super().apply_with_params(sampled_params, *args, **kwargs)

    def apply_to_volume(
        self,
        volume: VolumeType | torch.Tensor,
        crop_coords: tuple[int, int, int, int, int, int],
        padding: tuple[int, int, int, int, int, int],
        target_fill: float,
        **params: Any,
    ) -> VolumeType:
        return cast("VolumeType", f3d.crop_and_pad_volume(volume, crop_coords, padding, target_fill))

    def apply_to_mask3d(
        self,
        mask3d: VolumeType | torch.Tensor,
        crop_coords: tuple[int, int, int, int, int, int],
        padding: tuple[int, int, int, int, int, int],
        target_fill: float,
        **params: Any,
    ) -> VolumeType:
        return cast("VolumeType", f3d.crop_and_pad_volume(mask3d, crop_coords, padding, target_fill))

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        origin: tuple[int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.shift_keypoints(keypoints, np.array([-origin[2], -origin[1], -origin[0]]))
