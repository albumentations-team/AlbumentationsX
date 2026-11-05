"""Model-bound input normalization after accelerator transfer."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence

ComputePrecision = Literal["fp32", "fp16", "bf16"]
InputNormalizationMode = Literal["none", "fixed", "per_sample", "per_sample_nonzero"]

_UNBATCHED_IMAGE_DIMENSIONS: Final = 3
_MIN_BATCHED_IMAGE_DIMENSIONS: Final = 4
_OUTPUT_DTYPES: Final[dict[ComputePrecision, torch.dtype]] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class InputNormalizer(nn.Module):
    """Normalize image or volume batches on their device.

    Args:
        mode (InputNormalizationMode): "none" only scales and casts. "fixed" uses supplied statistics.
            "per_sample" standardizes each channel of each sample independently.
            "per_sample_nonzero" uses only nonzero positions and keeps zeros as padding with zero gradient.
        output_precision (ComputePrecision): Output dtype, selected after FP32 normalization.
        mean (float | Sequence[float] | None): Scalar or per-channel means for fixed normalization.
        standard_deviation (float | Sequence[float] | None): Positive scalar or per-channel standard deviations
            for fixed normalization.
        input_scale (float): Positive divisor applied before normalization.

    """

    def __init__(
        self,
        mode: InputNormalizationMode,
        *,
        output_precision: ComputePrecision,
        mean: float | Sequence[float] | None = None,
        standard_deviation: float | Sequence[float] | None = None,
        input_scale: float = 1.0,
    ) -> None:
        """Register fixed statistics as buffers that follow the module to its device."""
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
            mean_tensor = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
            standard_deviation_tensor = torch.as_tensor(standard_deviation, dtype=torch.float32).reshape(-1)
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
        """Compute normalization in owned FP32 storage."""
        if self.mode in {"per_sample", "per_sample_nonzero"} and data.ndim < _MIN_BATCHED_IMAGE_DIMENSIONS:
            message = "Per-sample normalization requires a BCHW or BCDHW tensor."
            raise ValueError(message)
        result = data.to(dtype=torch.float32, copy=True)
        if self.mode == "per_sample_nonzero":
            spatial_dimensions = tuple(range(2, data.ndim))
            count_data = data if data.dtype == torch.float64 or data.is_complex() else result
            # Zero-order norm counts support without materializing an integer mask cast.
            count = (
                torch.linalg.vector_norm(count_data.detach(), ord=0, dim=spatial_dimensions, keepdim=True)
                .float()
                .clamp_min(1)
            )
            if self.input_scale != 1.0:
                result.div_(self.input_scale)
            padding = data == 0
            if result.requires_grad or torch.autograd.forward_ad.unpack_dual(result).tangent is not None:
                # Exclude padding from derivatives of the mean in both AD modes.
                result.masked_fill_(padding, 0.0)
            mean = result.sum(dim=spatial_dimensions, keepdim=True) / count
            centered = result.sub_(mean).masked_fill_(padding, 0.0)
            del padding
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
        if self.mode == "per_sample":
            spatial_dimensions = tuple(range(2, data.ndim))
            variance, mean = torch.var_mean(result, dim=spatial_dimensions, correction=0, keepdim=True)
            centered = result - mean if result.requires_grad else result.sub_(mean)
            return _standardize(centered, variance)
        mean = cast("Tensor", self.mean)
        standard_deviation = cast("Tensor", self.standard_deviation)
        channel_dimension = 0 if data.ndim == _UNBATCHED_IMAGE_DIMENSIONS else 1
        channels = data.shape[channel_dimension]
        if mean.numel() not in {1, channels}:
            message = f"Fixed input statistics have {mean.numel()} channels, but input has {channels}."
            raise ValueError(message)
        shape = [1] * data.ndim
        shape[channel_dimension] = -1
        return result.sub_(mean.float().view(shape)).div_(standard_deviation.float().view(shape))


def _standardize(centered: Tensor, variance: Tensor) -> Tensor:
    """Scale centered data with a variance floor for constant channels."""
    inverse = torch.rsqrt(variance.clamp_min(torch.finfo(centered.dtype).eps))
    return centered * inverse if centered.requires_grad else centered.mul_(inverse)
