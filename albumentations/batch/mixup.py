"""Sample and apply MixUp plans for tensors with a leading batch axis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import torch

MixingMode = Literal["pair", "element", "batch"]
_MIXUP_WORKSPACE_BYTES: Final = 64 * 1024**2


@dataclass(frozen=True)
class MixUpPlan:
    """Donor indices and original-sample weights for a batch."""

    permutation: torch.Tensor
    lambdas: torch.Tensor


def sample_mixup(
    *, batch_size: int, alpha: float, device: torch.device, generator: torch.Generator, mode: MixingMode = "pair"
) -> MixUpPlan:
    """Draw a donor index and a Beta(alpha, alpha) weight for each sample.

    Args:
        batch_size (int): Number of samples in the batch.
        alpha (float): Positive concentration for the symmetric Beta distribution.
        device (torch.device): Device of the input batch and generator.
        generator (torch.Generator): Random generator used for both pairing and weights.
        mode (MixingMode): "pair" swaps randomly paired samples with independent weights,
            "element" permutes samples with independent weights, and "batch" swaps
            randomly paired samples with one shared weight. An unpaired sample
            keeps itself with weight 1.

    Returns:
        MixUpPlan: Donor indices and weights on the specified device.

    Raises:
        ValueError: If the batch size or alpha is nonpositive or the mode is unsupported.

    """
    if batch_size < 1 or alpha <= 0 or mode not in {"pair", "element", "batch"}:
        raise ValueError("Mixing requires a positive batch size and alpha with a supported mode.")
    permutation = _permutation(batch_size, device, generator, mode)
    if batch_size == 1:
        return MixUpPlan(permutation, torch.ones(1, device=device))
    count = 1 if mode == "batch" else batch_size
    lambdas = _sample_beta(alpha, count, generator, device)
    if mode == "batch":
        lambdas = lambdas.expand(batch_size)
    lambdas = torch.where(permutation == torch.arange(batch_size, device=device), torch.ones_like(lambdas), lambdas)
    return MixUpPlan(permutation, lambdas)


def apply_mixup(values: torch.Tensor, plan: MixUpPlan) -> torch.Tensor:
    """Interpolate each sample with its donor along the batch axis.

    Args:
        values (torch.Tensor): Tensor with the batch axis first, such as BCHW or BCDHW.
        plan (MixUpPlan): Donor indices and weights for every sample on the input device.

    Returns:
        torch.Tensor: A tensor with the input shape and dtype. Interpolation uses FP32. Uint8
        results are rounded to the nearest integer (ties to even) and clipped to [0, 255].

    Raises:
        ValueError: If the input has no batch axis or the plan lengths differ from the batch size.

    """
    if values.ndim < 1 or len(values) != len(plan.permutation) or len(values) != len(plan.lambdas):
        raise ValueError("Mixing plan batch size must match values.")
    weight = (1 - plan.lambdas).view((len(values),) + (1,) * (values.ndim - 1))
    if not len(values):
        source = values.float()
        return torch.lerp(source, source, weight).to(values.dtype)
    # Limit the FP32 block to 64 MiB, or one sample if it exceeds that size.
    sample_bytes = math.prod(values.shape[1:]) * torch.float32.itemsize
    block_size = max(1, _MIXUP_WORKSPACE_BYTES // max(sample_bytes, 1))
    # Cast once when backward must accumulate all source contributions in FP32.
    source = values.float() if torch.is_grad_enabled() and values.requires_grad else values
    result = torch.empty_like(values)
    for start in range(0, len(values), block_size):
        stop = start + block_size
        mixed = source[start:stop].to(dtype=torch.float32, copy=True)
        paired = source[plan.permutation[start:stop]].float()
        mixed.lerp_(paired, weight[start:stop])
        if values.dtype is torch.uint8:
            mixed.round_().clamp_(0, 255)
        result[start:stop].copy_(mixed)
        del mixed, paired
    return result


def _permutation(batch_size: int, device: torch.device, generator: torch.Generator, mode: MixingMode) -> torch.Tensor:
    order = torch.randperm(batch_size, generator=generator, device=device)
    if mode == "element":
        return torch.roll(order, 1)
    permutation = torch.arange(batch_size, device=device)
    pairs = order[: batch_size - batch_size % 2].view(-1, 2)
    permutation[pairs[:, 0]] = pairs[:, 1]
    permutation[pairs[:, 1]] = pairs[:, 0]
    return permutation


def _sample_beta(alpha: float, count: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    concentration = torch.full((count,), alpha, dtype=torch.float32, device=device)
    standard_gamma = torch.ops.aten._standard_gamma.default  # noqa: SLF001 - generator-aware vectorized Beta sampling.
    first = standard_gamma(concentration, generator=generator)
    second = standard_gamma(concentration, generator=generator)
    return first / (first + second)
