"""Horizontal and vertical flips of individual samples in a collated raster batch."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Final, Literal

import torch
from torch import randint

FlipDirection = Literal["horizontal", "vertical"]
_FLIP_DTYPES: Final = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_WORD_BITS: Final = 30


def _validate_direction(direction: FlipDirection) -> None:
    if not isinstance(direction, str) or direction not in ("horizontal", "vertical"):
        raise ValueError("Flip direction must be 'horizontal' or 'vertical'.")


def _validate_plain_tensor(tensor: torch.Tensor) -> None:
    if (
        type(tensor) is not torch.Tensor
        or tensor.device.type not in {"cpu", "cuda"}
        or tensor.layout is not torch.strided
        or tensor.is_nested
        or tensor.is_quantized
        or tensor._is_zerotensor()  # noqa: SLF001 -- immutable virtual storage also reports a strided layout
    ):
        raise ValueError("Flipping requires a plain strided CPU or CUDA tensor.")
    if tensor.is_neg() or any(getattr(tensor, "names", ())):
        raise ValueError("Flipping requires unnamed axes and materialized values.")


class FlipPlan:
    """A direction and one reusable decision per sample, shared across raster targets.

    Args:
        direction (FlipDirection): `"horizontal"` reflects width and `"vertical"` reflects height.
        selected (torch.Tensor): One-dimensional strided boolean CPU or CUDA tensor. The plan
            takes a snapshot on this device, even when the input shares storage with a target.

    """

    __slots__ = ("_direction", "_selected")

    def __init__(self, direction: FlipDirection, selected: torch.Tensor) -> None:
        _validate_direction(direction)
        _validate_plain_tensor(selected)
        if selected.dtype is not torch.bool or selected.ndim != 1:
            raise ValueError("Flip selection must be a one-dimensional boolean tensor.")
        self._direction = direction
        self._selected = selected.clone(memory_format=torch.contiguous_format)

    @property
    def direction(self) -> FlipDirection:
        """Spatial direction of the flip."""
        return self._direction

    @property
    def selected(self) -> torch.Tensor:
        """Get an independent copy of the per-sample decisions."""
        return self._selected.clone()


def _generator_device(device: torch.device | str, generator: torch.Generator) -> torch.device:
    if not isinstance(generator, torch.Generator) or not isinstance(device, (str, torch.device)):
        raise ValueError("Expected a CPU or CUDA device and a torch.Generator.")  # noqa: TRY004
    try:
        requested = torch.device(device)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid flip device {device!r}.") from exc
    actual = generator.device
    if requested.type == "cpu" and actual.type == "cpu":
        return actual
    if requested.type == "cuda" and actual.type == "cuda":
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if index == actual.index:
            return actual
    raise ValueError(f"Flip device {requested} must match the CPU or CUDA generator device {actual}.")


def _sample_selection(count: int, probability: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    if not count or probability in (0, 1):
        return torch.full((count,), probability == 1, dtype=torch.bool, device=device)
    numerator, denominator = probability.as_integer_ratio()
    bits = denominator.bit_length() - 1
    # Compare a uniform integer to the binary64 numerator, least significant word first.
    limit = 1 << min(bits, _WORD_BITS)
    selected = randint(limit, (count,), dtype=torch.int32, device=device, generator=generator) < (numerator % limit)
    for shift in range(_WORD_BITS, bits, _WORD_BITS):
        limit = 1 << min(bits - shift, _WORD_BITS)
        digit = (numerator >> shift) % limit
        word = randint(limit, (count,), dtype=torch.int32, device=device, generator=generator)
        selected = (word < digit) | ((word == digit) & selected)
    return selected


def sample_flip(
    *, batch_size: int, direction: FlipDirection, device: torch.device | str, generator: torch.Generator, p: float = 0.5
) -> FlipPlan:
    """Select each sample independently with probability `p` for a flip in one direction.

    Args:
        batch_size (int): Number of samples, including zero.
        direction (FlipDirection): `"horizontal"` or `"vertical"`.
        device (torch.device | str): CPU or CUDA device of the batch.
        generator (torch.Generator): Generator on that device. An empty batch or a probability
            of zero or one leaves its state unchanged.
        p (float): Exact Python float probability in [0, 1]. Defaults to 0.5.

    Returns:
        FlipPlan: Sampled decisions for reuse with any compatible raster target.

    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
        raise ValueError("Flip batch size must be a nonnegative integer.")
    _validate_direction(direction)
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError("Flip probability must be a finite number between zero and one.")
    actual_device = _generator_device(device, generator)
    return FlipPlan(direction, _sample_selection(batch_size, float(p), actual_device, generator))


def _validate_values(values: torch.Tensor, selected: torch.Tensor) -> None:
    _validate_plain_tensor(values)
    if values.dtype not in _FLIP_DTYPES or values.ndim < 3:
        raise ValueError("Flip values must have a supported raster dtype and shape [B, ..., H, W].")
    if values.shape[0] != selected.numel() or values.device != selected.device:
        raise ValueError("Flip selection and values must have matching batch sizes and devices.")
    in_inference_mode = torch.is_inference_mode_enabled()
    if values.is_inference() and not in_inference_mode:
        raise ValueError("An inference tensor can only be flipped inside inference mode.")
    # Inference mode hides the tangent of a dual created by the caller before entering that mode.
    with torch.inference_mode(False) if in_inference_mode else nullcontext():
        if values.requires_grad or torch.autograd.forward_ad.unpack_dual(values).tangent is not None:
            raise ValueError("In-place flips require values outside autograd.")
    if values.numel() and not values.is_contiguous():
        span = 1
        for stride, size in sorted(zip(values.stride(), values.shape, strict=True)):
            if size > 1:
                if stride < span:
                    raise ValueError("Flip values require non-overlapping strides with disjoint axis spans.")
                span += (size - 1) * stride


def _flip_planes(values: torch.Tensor, selected: torch.Tensor, dim: int, middle: list[tuple[int, int]]) -> None:
    # Only the chosen sample numbers cross devices. Each plane view keeps the caller's storage offset.
    plane_count = math.prod(size for size, _ in middle)
    for sample in selected.nonzero().flatten().tolist():
        start = values.storage_offset() + sample * values.stride(0)
        for ordinal in range(plane_count):
            offset = start
            remaining = ordinal
            for size, stride in reversed(middle):
                remaining, position = divmod(remaining, size)
                offset += position * stride
            plane = values.as_strided(values.shape[-2:], values.stride()[-2:], storage_offset=offset)
            plane.copy_(plane.flip((dim,)))


def _flip_selected(values: torch.Tensor, selected: torch.Tensor, dim: int) -> None:
    # Deterministic CUDA scatter makes a contiguous copy of a strided destination.
    scatter_copies_batch = (
        values.is_cuda and torch.are_deterministic_algorithms_enabled() and not values.is_contiguous()
    )
    if values.ndim > 5 or scatter_copies_batch:
        # Only spatial planes and at most two intervening axes reach the ordinary ATen kernels.
        middle = [
            (size, stride) for size, stride in zip(values.shape[1:-2], values.stride()[1:-2], strict=True) if size > 1
        ]
        if scatter_copies_batch or len(middle) > 2:
            _flip_planes(values, selected, dim, middle)
            return
        sizes = (values.shape[0], *(size for size, _ in middle), *values.shape[-2:])
        strides = (values.stride(0), *(stride for _, stride in middle), *values.stride()[-2:])
        values = values.as_strided(sizes, strides, storage_offset=values.storage_offset())
    values[selected] = torch.flip(values[selected], dims=(dim,))


def apply_flip_(values: torch.Tensor, plan: FlipPlan) -> torch.Tensor:
    """Flip selected samples in place across the last axis (horizontal) or penultimate axis (vertical).

    Args:
        values (torch.Tensor): Writable, unnamed strided CPU or CUDA raster of shape
            `[B, ..., H, W]`, outside autograd. Supports bool, uint8, signed integers
            and float16, bfloat16, float32, float64. Strides of non-singleton axes in ascending
            order must each span all preceding axes. An inference tensor requires inference mode.
        plan (FlipPlan): Reusable selection on the same device with one decision per sample.

    Returns:
        torch.Tensor: The original object. All channels and slices of a selected sample move together.

    Examples:
        >>> import torch
        >>> from albumentations.batch import apply_flip_, sample_flip
        >>> images = torch.arange(120).reshape(4, 2, 3, 5)
        >>> masks = images[:, 0].clone()
        >>> generator = torch.Generator(device=images.device).manual_seed(137)
        >>> for direction, p in (("horizontal", 0.5), ("vertical", 0.2)):
        ...     plan = sample_flip(batch_size=4, direction=direction, device=images.device, generator=generator, p=p)
        ...     _ = apply_flip_(images, plan)
        ...     _ = apply_flip_(masks, plan)
        >>> torch.equal(images[:, 0], masks)
        True

    """
    if not isinstance(plan, FlipPlan):
        raise ValueError("In-place flips require a FlipPlan.")  # noqa: TRY004
    selected = plan._selected  # noqa: SLF001 -- applying a plan reads its owned snapshot directly
    _validate_values(values, selected)
    dim = -1 if plan.direction == "horizontal" else -2
    if values.numel() and values.size(dim) > 1:
        _flip_selected(values, selected, dim)
    return values
