"""Model-bound input normalization after accelerator transfer."""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from functools import cache
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

ComputePrecision = Literal["fp32", "fp16", "bf16"]
InputNormalizationMode = Literal["none", "fixed", "per_sample", "per_sample_nonzero"]

_UNBATCHED_IMAGE_DIMENSIONS: Final = 3
_MIN_BATCHED_IMAGE_DIMENSIONS: Final = 4
_FP32_MAX: Final = torch.finfo(torch.float32).max
# With at most 2**24 FP16 elements per plane, this keeps the FP32 M2 sum below overflow.
_ZSCORE_MIN_INPUT_SCALE: Final = 2.0**-32
_FUSED_LOW_PRECISION_INPUTS: Final = (torch.uint8, torch.float16)
_OUTPUT_DTYPES: Final[dict[ComputePrecision, torch.dtype]] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class InputNormalizer(nn.Module):
    """Normalize image or volume batches on their device.

    Normalization runs outside autograd.

    Args:
        mode (InputNormalizationMode): "none" only scales and casts. "fixed" uses supplied statistics.
            "per_sample" standardizes each channel of each sample independently.
            "per_sample_nonzero" uses only nonzero positions and keeps zeros as padding.
        output_precision (ComputePrecision): Output dtype, selected after FP32 normalization.
        mean (float | Sequence[float] | Tensor | None): Scalar or per-channel means for fixed normalization.
        standard_deviation (float | Sequence[float] | Tensor | None): Positive scalar or per-channel standard
            deviations for fixed normalization.
        input_scale (float): Positive divisor applied before normalization.

    """

    def __init__(
        self,
        mode: InputNormalizationMode,
        *,
        output_precision: ComputePrecision,
        mean: float | Sequence[float] | Tensor | None = None,
        standard_deviation: float | Sequence[float] | Tensor | None = None,
        input_scale: float = 1.0,
    ) -> None:
        """Copy fixed statistics into buffers that follow the module to its device."""
        super().__init__()
        if mode not in {"none", "fixed", "per_sample", "per_sample_nonzero"}:
            message = f"Unsupported input normalization mode {mode!r}."
            raise ValueError(message)
        try:
            output_dtype = _OUTPUT_DTYPES[output_precision]
        except KeyError as error:
            message = f"Unsupported input normalization output precision {output_precision!r}."
            raise ValueError(message) from error
        if not bool(torch.isfinite(torch.tensor(input_scale))) or input_scale <= 0:
            message = "Input normalization scale must be finite and positive."
            raise ValueError(message)
        mean_tensor: Tensor | None
        standard_deviation_tensor: Tensor | None
        if mode == "fixed":
            if mean is None or standard_deviation is None:
                message = "Fixed input normalization requires mean and standard_deviation."
                raise ValueError(message)
            mean_tensor = torch.as_tensor(mean, dtype=torch.float32).detach().reshape(-1).clone()
            standard_deviation_tensor = (
                torch.as_tensor(standard_deviation, dtype=torch.float32).detach().reshape(-1).clone()
            )
            if mean_tensor.numel() == 0 or mean_tensor.numel() != standard_deviation_tensor.numel():
                message = "Fixed input normalization statistics must have the same nonzero length."
                raise ValueError(message)
            if not bool(torch.isfinite(mean_tensor).all()):
                message = "Fixed input normalization mean must be finite."
                raise ValueError(message)
            if not bool(torch.isfinite(standard_deviation_tensor).all()) or bool(
                (standard_deviation_tensor <= 0).any()
            ):
                message = "Fixed input normalization standard_deviation must be finite and positive."
                raise ValueError(message)
        else:
            if mean is not None or standard_deviation is not None:
                message = f"Input normalization mode {mode!r} does not accept fixed statistics."
                raise ValueError(message)
            mean_tensor = standard_deviation_tensor = None
        self.mode = mode
        self.output_precision = output_precision
        self.output_dtype = output_dtype
        self.input_scale = float(input_scale)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("standard_deviation", standard_deviation_tensor)

    @torch.no_grad()
    def forward(self, data: Tensor) -> Tensor:
        """Return normalized values on the input device.

        Args:
            data (Tensor): BCHW or BCDHW tensor. The "none" and "fixed" modes also accept CHW.

        Returns:
            Tensor: Normalized data with the input shape and configured output dtype.

        """
        if data.ndim < _UNBATCHED_IMAGE_DIMENSIONS:
            message = f"Input must have channel and spatial dimensions, got {tuple(data.shape)!r}."
            raise ValueError(message)
        data = data.detach()
        if (
            self.mode == "none"
            and self.input_scale == 1.0
            and data.dtype in {torch.uint8, torch.float16, torch.bfloat16, torch.float32}
        ):
            return data.to(dtype=self.output_dtype, copy=True)
        context = (
            torch.autocast(data.device.type, enabled=False)
            if torch.is_autocast_enabled(data.device.type)
            else nullcontext()
        )
        with context:
            normalized = self._normalize(data)
        return normalized.to(dtype=self.output_dtype)

    def _normalize(self, data: Tensor) -> Tensor:
        """Compute normalization with FP32 arithmetic."""
        if self.mode == "fixed":
            return self._normalize_fixed(data)
        if self.mode != "none":
            if data.ndim < _MIN_BATCHED_IMAGE_DIMENSIONS:
                message = "Per-sample normalization requires a BCHW or BCDHW tensor."
                raise ValueError(message)
            if (
                self.output_dtype in {torch.float16, torch.bfloat16}
                and data.dtype in _FUSED_LOW_PRECISION_INPUTS
                and (data.dtype == torch.uint8 or self.output_dtype == torch.bfloat16)
                and self.input_scale >= _ZSCORE_MIN_INPUT_SCALE
                and (route := self._cuda_reduced_precision_route(data)) is not None
                and (plane_size := math.prod(data.shape[2:])) <= 2**24
            ):
                cuda, inverse_scale = route
                return cuda.normalize_zscore(
                    data,
                    plane_size,
                    self.mode == "per_sample_nonzero",
                    inverse_scale,
                    self.output_dtype,
                )
        result = data.to(dtype=torch.float32, copy=True)
        if self.mode == "per_sample_nonzero":
            spatial_dimensions = tuple(range(2, data.ndim))
            count_data = data if data.dtype == torch.float64 or data.is_complex() else result
            # Zero-order norm counts support without materializing an integer mask cast.
            count = (
                torch.linalg.vector_norm(count_data, ord=0, dim=spatial_dimensions, keepdim=True).float().clamp_min(1)
            )
            if self.input_scale != 1.0:
                result.div_(self.input_scale)
            mean = result.sum(dim=spatial_dimensions, keepdim=True) / count
            centered = result.sub_(mean).masked_fill_(data == 0, 0.0)
            spatial_variance, residual_mean = torch.var_mean(
                centered, dim=spatial_dimensions, correction=0, keepdim=True
            )
            # E[x^2] = Var(x) + E[x]^2; rescale from all spatial positions to the original support.
            variance = (spatial_variance + residual_mean.square()) * (math.prod(data.shape[2:]) / count)
            return _standardize(centered, variance)
        if self.input_scale != 1.0:
            result.div_(self.input_scale)
        if self.mode == "none":
            return result
        spatial_dimensions = tuple(range(2, data.ndim))
        variance, mean = torch.var_mean(result, dim=spatial_dimensions, correction=0, keepdim=True)
        return _standardize(result.sub_(mean), variance)

    def _normalize_fixed(self, data: Tensor) -> Tensor:
        mean = cast("Tensor", self.mean)
        standard_deviation = cast("Tensor", self.standard_deviation)
        channel_dimension = 0 if data.ndim == _UNBATCHED_IMAGE_DIMENSIONS else 1
        channels = data.shape[channel_dimension]
        if mean.numel() not in {1, channels}:
            message = f"Fixed input statistics have {mean.numel()} channels, but input has {channels}."
            raise ValueError(message)
        if (
            self.output_dtype in {torch.float16, torch.bfloat16}
            and (
                data.dtype in _FUSED_LOW_PRECISION_INPUTS
                or (data.dtype == torch.float32 and self.output_dtype == torch.bfloat16)
            )
            and (route := self._cuda_reduced_precision_route(data)) is not None
            and _can_use_fixed_statistics(data, mean, standard_deviation)
        ):
            cuda, inverse_scale = route
            return cuda.normalize_fixed(
                data,
                mean,
                standard_deviation,
                channel_dimension,
                inverse_scale,
                self.output_dtype,
            )
        result = data.to(dtype=torch.float32, copy=True)
        if self.input_scale != 1.0:
            result.div_(self.input_scale)
        shape = [1] * data.ndim
        shape[channel_dimension] = -1
        return result.sub_(mean.float().view(shape)).div_(standard_deviation.float().view(shape))

    def _cuda_reduced_precision_route(self, data: Tensor) -> tuple[ModuleType, float] | None:
        if self.input_scale <= 0.0 or not _can_fuse_input(data):
            return None
        # PyTorch CUDA implements division by a host scalar via a double-precision reciprocal rounded to FP32.
        inverse_scale = 1.0 / self.input_scale
        if inverse_scale <= _FP32_MAX and (cuda := _cuda_normalization(data.device)) is not None:
            return cuda, inverse_scale
        return None


def _can_fuse_input(data: Tensor) -> bool:
    return (
        data.device.type == "cuda"
        and not torch.compiler.is_compiling()
        and type(data) is Tensor
        and not torch._C._are_functorch_transforms_active()  # noqa: SLF001 - vmap tensors may lack addressable storage.
        and not data.is_neg()
        and not any(getattr(data, "names", ()))
        and 0 < data.numel() <= 2**31 - 1024
        and (
            data.is_contiguous()
            or (data.ndim == 4 and data.is_contiguous(memory_format=torch.channels_last))
            or (data.ndim == 5 and data.is_contiguous(memory_format=torch.channels_last_3d))
        )
    )


def _can_use_fixed_statistics(data: Tensor, mean: Tensor, std: Tensor) -> bool:
    return mean.shape == std.shape and all(
        type(stat) is Tensor
        and stat.device == data.device
        and stat.dtype == torch.float32
        and stat.ndim == 1
        and stat.is_contiguous()
        and not stat.is_neg()
        and not any(getattr(stat, "names", ()))
        for stat in (mean, std)
    )


@cache
def _cuda_normalization(device: torch.device) -> ModuleType | None:
    if sys.platform != "linux" or torch.version.cuda is None or torch.cuda.get_device_capability(device) < (8, 0):
        return None
    try:
        from . import _normalization_cuda
    except ImportError:
        return None
    else:
        return _normalization_cuda


def _standardize(centered: Tensor, variance: Tensor) -> Tensor:
    """Scale centered data with a variance floor for constant channels."""
    inverse = torch.rsqrt(variance.clamp_min(torch.finfo(centered.dtype).eps))
    return centered.mul_(inverse)
