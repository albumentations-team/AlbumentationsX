"""Optional Triton kernel for in-place flips of contiguous BCHW CUDA rasters."""

from importlib import import_module
from typing import Any

import torch

triton: Any = import_module("triton")
tl: Any = import_module("triton.language")
_BLOCK_SIZE = 512
_LARGE_GRID_PAIRS = 2**24
_MAX_GRID_X = 2**31 - 1


@triton.jit
def _swap_pairs(
    data: tl.tensor,
    selected: tl.tensor,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    horizontal: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    if horizontal:
        half = width // 2
        pairs_per_sample = channels * height * half
    else:
        half = height // 2
        pairs_per_sample = channels * half * width
    tiles = tl.cdiv(pairs_per_sample, block_size)
    program = tl.program_id(0).to(tl.int64)
    sample = program // tiles
    if tl.load(selected + sample):
        pair = program % tiles * block_size + tl.arange(0, block_size)
        valid = pair < pairs_per_sample
        if horizontal:
            row_start = pair // half * width
            column = pair % half
            first = row_start + column
            second = row_start + (width - 1 - column)
        else:
            row = pair // width % half
            plane = pair // (half * width)
            column = pair % width
            first = (plane * height + row) * width + column
            second = (plane * height + (height - 1 - row)) * width + column
        image = data + sample * (channels * height * width)
        # Each pair has exactly one owner, including at tile boundaries and on odd axes.
        left = tl.load(image + first, mask=valid, other=0)
        right = tl.load(image + second, mask=valid, other=0)
        tl.store(image + first, right, mask=valid)
        tl.store(image + second, left, mask=valid)


def try_flip(values: torch.Tensor, selected: torch.Tensor, dim: int) -> bool:
    """Prepare and launch the kernel, returning False if preparation cannot complete."""
    batch, channels, height, width = values.shape
    horizontal = dim == -1
    pairs = channels * (height * (width // 2) if horizontal else (height // 2) * width)
    # Wide tiles reduce selection checks on large grids. Keep smaller tiles for rows with unaligned starts.
    block_size = 2048 if batch * pairs >= _LARGE_GRID_PAIRS and width % 32 == 0 else _BLOCK_SIZE
    programs = batch * triton.cdiv(pairs, block_size)
    if programs > _MAX_GRID_X:
        return False
    grid = (programs, 1, 1)
    args = (values, selected, channels, height, width, horizontal, block_size)
    with torch.cuda.device(values.device):
        # Triton's specialization cache also accounts for the alignment of offset views.
        try:
            kernel = _swap_pairs.warmup(*args, grid=grid, enable_fp_fusion=False)
            if kernel is None:
                return False
            launch = kernel[grid]
        except Exception:  # noqa: BLE001 -- Triton and its driver use no common preparation exception.
            return False
        # The launcher can enqueue writes and then raise. Autograd must see that mutation even on error.
        torch.autograd.graph.increment_version(values)
        launch(*args)
    return True
