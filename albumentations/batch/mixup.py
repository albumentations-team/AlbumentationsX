"""Sample and apply MixUp plans for tensors with a leading batch axis."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal

import torch

MixingMode = Literal["pair", "element", "batch"]
_MIXUP_WORKSPACE_BYTES: Final = 64 * 1024**2
_CPU_DONOR_VIEW_MIN_ELEMENTS: Final = 20 * 1024
_CPU_DONOR_BLOCK_ELEMENTS: Final = 128 * 1024
_BETA_LOG_DOMAIN_THRESHOLD: Final = 0.1
_FP32: Final = torch.finfo(torch.float32)


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
        alpha (float): Finite positive concentration for the symmetric Beta distribution.
        device (torch.device): Device of the input batch and generator.
        generator (torch.Generator): Random generator used for both pairing and weights.
        mode (MixingMode): "pair" swaps randomly paired samples with independent weights,
            "element" permutes samples with independent weights, and "batch" swaps
            randomly paired samples with one shared weight. An unpaired sample
            keeps itself with weight 1.

    Returns:
        MixUpPlan: Donor indices and weights on the specified device.

    Raises:
        ValueError: If the batch size is nonpositive, alpha is invalid, or the mode is unsupported.

    """
    if batch_size < 1 or not math.isfinite(alpha) or alpha <= 0 or mode not in {"pair", "element", "batch"}:
        raise ValueError("Mixing requires a positive batch size, finite positive alpha, and a supported mode.")
    permutation = _permutation(batch_size, device, generator, mode)
    if batch_size == 1:
        return MixUpPlan(permutation, torch.ones(1, dtype=torch.float32, device=device))
    count = 1 if mode == "batch" else batch_size
    lambdas = _sample_beta(alpha, count, generator, device)
    if mode == "batch":
        lambdas = lambdas.expand(batch_size)
    lambdas = torch.where(permutation == torch.arange(batch_size, device=device), torch.ones_like(lambdas), lambdas)
    return MixUpPlan(permutation, lambdas)


@torch.no_grad()
def apply_mixup(values: torch.Tensor, plan: MixUpPlan, *, flip_donor: bool = False) -> torch.Tensor:
    """Interpolate each sample with its donor along the batch axis.

    MixUp runs outside autograd.

    Args:
        values (torch.Tensor): Tensor with the batch axis first, such as BCHW or BCDHW.
        plan (MixUpPlan): Donor indices and weights for every sample on the input device.
        flip_donor (bool): Flip the donor along the last (width) axis before blending.
            Use for spatial tensors such as BHW, BCHW or BCDHW. For BHWC, first move
            channels with `values.movedim(-1, 1)`. Defaults to False.

    Returns:
        torch.Tensor: A tensor with the input shape and dtype. Interpolation uses FP32. Uint8
        results are rounded to the nearest integer (ties to even) and clipped to [0, 255].

    Raises:
        ValueError: If the input has no batch axis, the plan lengths differ from the batch size,
            or a donor flip is requested for a tensor with fewer than three dimensions.

    Examples:
        >>> import torch
        >>> from albumentations.batch import MixUpPlan, apply_mixup
        >>> images = torch.tensor([[[0., 4., 8.]], [[12., 20., 28.]]])
        >>> plan = MixUpPlan(torch.tensor([1, 0]), torch.tensor([0.75, 0.5]))
        >>> mixed = apply_mixup(images, plan, flip_donor=True)
        >>> assert mixed.tolist() == [[[7, 8, 9]], [[10, 12, 14]]]
        >>> targets = apply_mixup(torch.eye(2), plan)
        >>> assert targets.tolist() == [[0.75, 0.25], [0.5, 0.5]]

    """
    if values.ndim < 1 or len(values) != len(plan.permutation) or len(values) != len(plan.lambdas):
        raise ValueError("Mixing plan batch size must match values.")
    if flip_donor and values.ndim < 3:
        raise ValueError("Donor flipping requires spatial tensors with at least three dimensions (BHW).")
    values = values.detach()
    result = torch.empty_like(values)
    if not len(values):
        return result
    for rows, donors, recipient_weight, donor_weight in _mixup_blocks(values, plan, flip_donor=flip_donor):
        paired = donors.flip((-1,)) if flip_donor else donors
        del donors
        paired = paired.float()
        destination = result[rows]
        if flip_donor and values.dtype in (torch.bfloat16, torch.float32, torch.float64):
            # Subtracting finite opposite-sign endpoints in lerp can overflow FP32.
            paired.mul_(donor_weight)
            if values.dtype is torch.bfloat16:
                # The FP32 donor owns the accumulator. addcmul promotes the recipient before the final BF16 cast.
                destination.copy_(paired.addcmul_(values[rows], recipient_weight))
            else:
                torch.addcmul(paired, values[rows].float(), recipient_weight, out=destination)
        elif values.dtype is torch.float32:
            torch.lerp(values[rows], paired, donor_weight, out=destination)
        else:
            mixed = values[rows].float()
            if flip_donor:
                # Interpolate toward the recipient with its original weight, even near zero or a rounding tie.
                torch.lerp(paired, mixed, recipient_weight, out=mixed)
            else:
                mixed.lerp_(paired, donor_weight)
            if values.dtype is torch.uint8:
                mixed.round_().clamp_(0, 255)
            destination.copy_(mixed)
            del mixed
        del paired, recipient_weight, donor_weight
    return result


def _mixup_blocks(
    values: torch.Tensor, plan: MixUpPlan, *, flip_donor: bool
) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield recipients' row slices, donors and both FP32 weights within the workspace budget."""
    sample_elements = math.prod(values.shape[1:])
    bytes_per_sample = sample_elements * torch.float32.itemsize
    if flip_donor:
        # Gather/flip and FP32 blending each need two image buffers. The weights can also require FP32 promotion.
        working_itemsize = max(torch.float32.itemsize, values.element_size())
        bytes_per_sample = 2 * (sample_elements * working_itemsize + torch.float32.itemsize)
    block_size = _MIXUP_WORKSPACE_BYTES // max(bytes_per_sample, 1)

    permutation = plan.permutation
    can_select_views = (
        flip_donor
        and values.device.type == "cpu"
        and permutation.device.type == "cpu"
        and permutation.ndim == 1
        and permutation.dtype in (torch.int32, torch.int64)
    )
    use_views = can_select_views and (len(values) == 1 or sample_elements >= _CPU_DONOR_VIEW_MIN_ELEMENTS)
    if can_select_views and not use_views:
        # Group small images to amortize dispatch without repeatedly copying a whole CPU batch.
        block_size = min(block_size, _CPU_DONOR_BLOCK_ELEMENTS // max(sample_elements, 1))
    block_size = max(1, block_size)
    weights = plan.lambdas.detach().reshape((len(values),) + (1,) * (values.ndim - 1))
    for start in range(0, len(values), block_size):
        stop = min(start + block_size, len(values))
        rows = slice(start, stop)
        recipient_weight = weights[rows].float()
        donor_weight = 1 - recipient_weight
        if use_views:
            for row, donor_index, row_recipient_weight, row_donor_weight in zip(
                range(start, stop),
                permutation[rows].tolist(),
                recipient_weight.unbind(),
                donor_weight.unbind(),
                strict=True,
            ):
                yield slice(row, row + 1), values[donor_index].unsqueeze(0), row_recipient_weight, row_donor_weight
        else:
            yield rows, values[permutation[rows]], recipient_weight, donor_weight
        del recipient_weight, donor_weight


def _permutation(batch_size: int, device: torch.device, generator: torch.Generator, mode: MixingMode) -> torch.Tensor:
    """Assign donors by shuffling or swapping disjoint pairs."""
    order = torch.randperm(batch_size, generator=generator, device=device)
    if mode == "element":
        return torch.roll(order, 1)
    permutation = torch.arange(batch_size, device=device)
    pairs = order[: batch_size - batch_size % 2].view(-1, 2)
    permutation[pairs[:, 0]] = pairs[:, 1]
    permutation[pairs[:, 1]] = pairs[:, 0]
    return permutation


def _sample_beta(alpha: float, count: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Draw symmetric Beta weights while preserving both FP32 tails."""
    if alpha > _FP32.max / 4:
        # At these concentrations even many standard deviations round to 0.5 in FP32.
        return torch.full((count,), 0.5, dtype=torch.float32, device=device)

    # Below this threshold the probability of clipping both Gamma draws rises rapidly.
    use_log_domain = alpha < _BETA_LOG_DOMAIN_THRESHOLD
    concentration = torch.full((count,), alpha + 1 if use_log_domain else alpha, dtype=torch.float32, device=device)
    # The supported Torch 2.13 API has no generator argument on Gamma.sample.
    standard_gamma = torch.ops.aten._standard_gamma.default  # noqa: SLF001
    first = standard_gamma(concentration, generator=generator)
    second = standard_gamma(concentration, generator=generator)
    if use_log_domain:
        # log(Gamma(a)) has the distribution log(Gamma(a + 1)) - Exp(1) / a.
        # Form the log ratio before scaling to avoid subtracting two overflowed values.
        first_exponential = torch.empty_like(first).exponential_(generator=generator)
        second_exponential = torch.empty_like(second).exponential_(generator=generator)
        log_uniform_ratio = second_exponential.sub_(first_exponential).div_(max(alpha, _FP32.tiny))
        return first.log_().sub_(second.log_()).add_(log_uniform_ratio).sigmoid_()
    return first / (first + second)
