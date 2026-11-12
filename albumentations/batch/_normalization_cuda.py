"""CUDA kernels for normalization of dense tensors with reduced-precision output."""

from importlib import import_module
from typing import Any

import torch
from torch import Tensor

triton: Any = import_module("triton")
tl: Any = import_module("triton.language")

_BLOCK_SIZE = 1024
_STAT_TILE_SIZE = 2048


@triton.jit
def _fixed_normalize_kernel(
    data: tl.tensor,
    mean: tl.tensor,
    std: tl.tensor,
    output: tl.tensor,
    size: tl.tensor,
    stats_count: tl.constexpr,
    channel_stride: tl.constexpr,
    inverse_scale: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    index = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = index < size
    if stats_count == 1:
        channel_mean = tl.load(mean)
        channel_std = tl.load(std)
    else:
        channel = index // channel_stride % stats_count
        channel_mean = tl.load(mean + channel)
        channel_std = tl.load(std + channel)
    value = tl.load(data + index, valid, other=0).to(tl.float32)
    if inverse_scale != 1.0:
        value = value * tl.full((), inverse_scale, tl.float32)
    normalized = tl.div_rn(value - channel_mean, channel_std)
    tl.store(output + index, normalized.to(output.dtype.element_ty), valid)


def normalize_fixed(
    data: Tensor,
    mean: Tensor,
    std: Tensor,
    channel_dimension: int,
    inverse_scale: float,
    output_dtype: torch.dtype,
) -> Tensor:
    """Write the reduced-precision result directly using FP32 arithmetic."""
    output = torch.empty_like(data, dtype=output_dtype)
    size = data.numel()
    stats_count = mean.numel()
    with torch.cuda.device(data.device):
        _fixed_normalize_kernel[(triton.cdiv(size, _BLOCK_SIZE),)](
            data,
            mean,
            std,
            output,
            size,
            stats_count,
            data.stride(channel_dimension) if stats_count > 1 else 1,
            inverse_scale,
            _BLOCK_SIZE,
            enable_fp_fusion=False,
        )
    return output


@triton.jit
def _zscore_tile_stats(
    data: tl.tensor,
    partials: tl.tensor,
    partial_count: tl.tensor,
    plane_size: tl.constexpr,
    channels: tl.constexpr,
    tiles_per_plane: tl.constexpr,
    channels_last: tl.constexpr,
    nonzero: tl.constexpr,
    inverse_scale: tl.constexpr,
    tile_size: tl.constexpr,
) -> None:
    tile = tl.program_id(0)
    plane = tile // tiles_per_plane
    position = tile % tiles_per_plane * tile_size + tl.arange(0, tile_size)
    valid = position < plane_size
    position = tl.where(valid, position, 0)
    if channels_last:
        index = (plane // channels * plane_size + position) * channels + plane % channels
    else:
        index = plane * plane_size + position
    value = tl.load(data + index, valid, other=0).to(tl.float32)
    if nonzero:
        valid = valid & (value != 0.0)
    if inverse_scale != 1.0:
        value = value * tl.full((), inverse_scale, tl.float32)
    count = tl.sum(valid.to(tl.float32), 0)
    mean = tl.div_rn(tl.sum(tl.where(valid, value, 0.0), 0), tl.maximum(count, 1.0))
    centered = tl.where(valid, value - mean, 0.0)
    m2 = tl.sum(centered * centered, 0)
    tl.store(partials + tile, count)
    tl.store(partials + partial_count + tile, mean)
    tl.store(partials + 2 * partial_count + tile, m2)


@triton.jit
def _zscore_merge_stats(
    partials: tl.tensor,
    stats: tl.tensor,
    partial_count: tl.tensor,
    planes: tl.tensor,
    tiles_per_plane: tl.constexpr,
    block_size: tl.constexpr,
    epsilon: tl.constexpr,
) -> None:
    plane = tl.program_id(0)
    tile = tl.arange(0, block_size)
    base = plane * tiles_per_plane
    count = tl.load(partials + base + tile, tile < tiles_per_plane, other=0)
    mean = tl.load(partials + partial_count + base + tile, tile < tiles_per_plane, other=0)
    m2 = tl.load(partials + 2 * partial_count + base + tile, tile < tiles_per_plane, other=0)
    # Anchor the weighted mean at a populated tile to preserve constant crops exactly.
    anchor_tile = tl.min(tl.where(count > 0.0, tile, tiles_per_plane), 0)
    anchor = tl.load(partials + partial_count + base + anchor_tile, anchor_tile < tiles_per_plane, other=0)
    total = tl.maximum(tl.sum(count, 0), 1.0)
    delta = tl.where(count > 0.0, mean - anchor, 0.0)
    combined_mean = anchor + tl.div_rn(tl.sum(delta * count, 0), total)
    deviation = tl.where(count > 0.0, mean - combined_mean, 0.0)
    variance = tl.div_rn(tl.sum(m2 + count * deviation * deviation, 0), total)
    tl.store(stats + plane, combined_mean)
    tl.store(stats + planes + plane, tl.rsqrt(tl.maximum(variance, epsilon)))


@triton.jit
def _zscore_store_output(
    data: tl.tensor,
    stats: tl.tensor,
    output: tl.tensor,
    size: tl.tensor,
    planes: tl.tensor,
    plane_size: tl.constexpr,
    channels: tl.constexpr,
    channels_last: tl.constexpr,
    nonzero: tl.constexpr,
    inverse_scale: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    index = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = index < size
    plane = index // (plane_size * channels) * channels + index % channels if channels_last else index // plane_size
    mean = tl.load(stats + plane, valid, other=0)
    inverse_std = tl.load(stats + planes + plane, valid, other=0)
    value = tl.load(data + index, valid, other=0).to(tl.float32)
    keep = value != 0.0
    if inverse_scale != 1.0:
        value = value * tl.full((), inverse_scale, tl.float32)
    centered = value - mean
    if nonzero:
        centered = tl.where(keep, centered, 0.0)
    tl.store(output + index, (centered * inverse_std).to(output.dtype.element_ty), valid)


def normalize_zscore(
    data: Tensor,
    plane_size: int,
    nonzero: bool,
    inverse_scale: float,
    output_dtype: torch.dtype,
) -> Tensor:
    """Reduce FP32 tile statistics and write the output without a volume-sized workspace."""
    size = data.numel()
    channels = data.shape[1]
    planes = size // plane_size
    channels_last = not data.is_contiguous()
    tiles = triton.cdiv(plane_size, _STAT_TILE_SIZE)
    partial_count = planes * tiles
    partials = torch.empty((3, partial_count), dtype=torch.float32, device=data.device)
    stats = torch.empty((2, planes), dtype=torch.float32, device=data.device)
    output = torch.empty_like(data, dtype=output_dtype)
    with torch.cuda.device(data.device):
        _zscore_tile_stats[(partial_count,)](
            data,
            partials,
            partial_count,
            plane_size,
            channels,
            tiles,
            channels_last,
            nonzero,
            inverse_scale,
            _STAT_TILE_SIZE,
            enable_fp_fusion=False,
        )
        _zscore_merge_stats[(planes,)](
            partials,
            stats,
            partial_count,
            planes,
            tiles,
            triton.next_power_of_2(tiles),
            torch.finfo(torch.float32).eps,
            enable_fp_fusion=False,
        )
        _zscore_store_output[(triton.cdiv(size, _BLOCK_SIZE),)](
            data,
            stats,
            output,
            size,
            planes,
            plane_size,
            channels,
            channels_last,
            nonzero,
            inverse_scale,
            _BLOCK_SIZE,
            enable_fp_fusion=False,
        )
    return output
