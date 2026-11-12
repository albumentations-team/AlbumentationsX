"""Tests for normalization on the model side of accelerator transfer."""

from __future__ import annotations

import math
import sys
from array import array
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
import torch

from albumentations.batch import InputNormalizer
from tests.helpers.tensor_memory import measure_peak_tensor_bytes

if TYPE_CHECKING:
    from pathlib import Path

    from albumentations.batch.normalization import ComputePrecision, InputNormalizationMode


@pytest.fixture(scope="module")
def triton_cuda() -> None:
    if (
        sys.platform != "linux"
        or not torch.cuda.is_available()
        or torch.version.cuda is None
        or torch.cuda.get_device_capability() < (8, 0)
    ):
        pytest.skip("Requires Linux and a CUDA GPU with compute capability 8.0 or higher.")
    pytest.importorskip("triton")


def _normalize_with_cuda_peak(normalizer: InputNormalizer, values: torch.Tensor) -> tuple[torch.Tensor, int]:
    normalizer(values)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = normalizer(values)
    torch.cuda.synchronize()
    return result, torch.cuda.max_memory_allocated() - before


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_none_mode_only_restores_fp32(dtype: torch.dtype) -> None:
    values = torch.arange(8).reshape(1, 1, 2, 2, 2).to(dtype=dtype)

    result = InputNormalizer("none", output_precision="fp32")(values)

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, values.float())


@pytest.mark.parametrize(("precision", "dtype"), [("fp16", torch.float16), ("bf16", torch.bfloat16)])
def test_none_mode_retains_fp32_rounding_before_output_conversion(
    precision: ComputePrecision, dtype: torch.dtype
) -> None:
    lower = torch.tensor(1.0, dtype=dtype).double()
    upper = torch.nextafter(torch.tensor(1.0, dtype=dtype), torch.tensor(2.0, dtype=dtype)).double()
    values = ((lower + upper) / 2 + 1e-9).reshape(1, 1, 1)
    result = InputNormalizer("none", output_precision=precision)(values)
    torch.testing.assert_close(result, values.float().to(dtype), rtol=0, atol=0)


def test_fixed_mode_broadcasts_channel_statistics_over_a_batch() -> None:
    values = torch.tensor(
        [
            [
                [[[1.0, 3.0]]],
                [[[12.0, 16.0]]],
            ],
            [
                [[[5.0, 7.0]]],
                [[[20.0, 24.0]]],
            ],
        ],
        dtype=torch.float16,
    )

    result = InputNormalizer(
        "fixed",
        output_precision="bf16",
        mean=(1.0, 10.0),
        standard_deviation=(2.0, 4.0),
    )(values)

    expected = torch.tensor(
        [
            [[[[0.0, 1.0]]], [[[0.5, 1.5]]]],
            [[[[2.0, 3.0]]], [[[2.5, 3.5]]]],
        ],
        dtype=torch.bfloat16,
    )
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, expected)


@pytest.mark.usefixtures("triton_cuda")
@pytest.mark.parametrize(
    ("input_dtype", "precision", "output_dtype"),
    [
        (torch.float16, "fp16", torch.float16),
        (torch.float16, "bf16", torch.bfloat16),
        (torch.uint8, "fp16", torch.float16),
        (torch.uint8, "bf16", torch.bfloat16),
        (torch.float32, "bf16", torch.bfloat16),
    ],
)
@pytest.mark.parametrize(
    ("shape", "memory_format", "input_scale", "scalar"),
    [
        ((3, 67, 65), torch.contiguous_format, 1.0, False),
        ((2, 5, 67, 65), torch.channels_last, 3.14159265, False),
        ((2, 7, 3, 67, 65), torch.contiguous_format, 255.0, False),
        ((2, 7, 3, 67, 65), torch.channels_last_3d, 1.0, False),
        ((2, 3, 67, 65), torch.contiguous_format, 1.0, True),
    ],
)
def test_fixed_cuda_matches_fp32_without_a_workspace(
    shape: tuple[int, ...],
    memory_format: torch.memory_format,
    input_scale: float,
    scalar: bool,
    precision: ComputePrecision,
    output_dtype: torch.dtype,
    input_dtype: torch.dtype,
) -> None:
    channel_dim = 0 if len(shape) == 3 else 1
    channels = 1 if scalar else shape[channel_dim]
    mean = torch.linspace(-0.481, 0.352, channels)
    std = torch.linspace(0.227, 1.137, channels)
    normalizer = InputNormalizer(
        "fixed",
        output_precision=precision,
        mean=mean,
        standard_deviation=std,
        input_scale=input_scale,
    ).cuda()
    if input_dtype == torch.uint8:
        values = torch.arange(math.prod(shape), device="cuda", dtype=torch.int32).remainder_(256).reshape(shape)
    else:
        values = torch.linspace(-137, 137, math.prod(shape), device="cuda", dtype=torch.float32).reshape(shape)
    values = values.to(dtype=input_dtype, memory_format=memory_format)
    original = values.clone()
    broadcast = [1] * len(shape)
    broadcast[channel_dim] = channels
    expected = original.to(dtype=torch.float32, copy=True)
    if input_scale != 1.0:
        expected.div_(input_scale)
    expected.sub_(mean.cuda().view(broadcast)).div_(std.cuda().view(broadcast))
    expected = expected.to(output_dtype)
    actual, peak = _normalize_with_cuda_peak(normalizer, values)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_stride=True)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert actual.data_ptr() != values.data_ptr()
    assert peak <= actual.nbytes + 4096


@pytest.mark.usefixtures("triton_cuda")
@pytest.mark.parametrize("mode", ["per_sample", "per_sample_nonzero"])
@pytest.mark.parametrize(
    ("input_dtype", "precision", "output_dtype"),
    [
        (torch.float16, "bf16", torch.bfloat16),
        (torch.uint8, "fp16", torch.float16),
        (torch.uint8, "bf16", torch.bfloat16),
    ],
)
@pytest.mark.parametrize(
    ("shape", "memory_format", "input_scale"),
    [
        ((2, 2, 67, 65), torch.contiguous_format, 1.0),
        ((2, 2, 3, 67, 65), torch.contiguous_format, 255.0),
        ((2, 2, 3, 67, 65), torch.channels_last_3d, 1.0),
    ],
)
def test_zscore_cuda_writes_the_requested_dtype_without_a_workspace(
    mode: InputNormalizationMode,
    shape: tuple[int, ...],
    memory_format: torch.memory_format,
    input_scale: float,
    input_dtype: torch.dtype,
    precision: ComputePrecision,
    output_dtype: torch.dtype,
) -> None:
    values = torch.zeros(shape, device="cuda", dtype=input_dtype)
    values[0, 0].flatten()[::137] = 17.0
    values[0, 0].flatten()[::274] = 18.0 if input_dtype == torch.uint8 else 17.015625
    values[0, 1].fill_(137.0)
    values[1, 1].flatten()[137] = 237.0 if input_dtype == torch.uint8 else -137.0
    values = values.contiguous(memory_format=memory_format)
    original = values.clone()
    normalizer = InputNormalizer(mode, output_precision=precision, input_scale=input_scale).cuda()
    rows = values.reshape(shape[0] * shape[1], -1)
    expected_rows = []
    for row in rows:
        keep = row != 0 if mode == "per_sample_nonzero" else torch.ones_like(row, dtype=torch.bool)
        output = torch.zeros_like(row, dtype=torch.float64)
        selected = row[keep].double() / input_scale
        if selected.numel():
            variance, mean = torch.var_mean(selected, correction=0)
            output[keep] = (selected - mean) * variance.clamp_min(torch.finfo(torch.float32).eps).rsqrt()
        expected_rows.append(output)
    expected = torch.stack(expected_rows).reshape(shape).to(output_dtype)
    actual, peak = _normalize_with_cuda_peak(normalizer, values)

    torch.testing.assert_close(actual, expected, rtol=torch.finfo(output_dtype).eps, atol=5e-5)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert actual.stride() == values.stride()
    assert actual.data_ptr() != values.data_ptr()
    assert peak <= actual.nbytes + 4096
    if mode == "per_sample_nonzero":
        assert torch.count_nonzero(actual[values == 0]) == 0


@pytest.mark.usefixtures("triton_cuda")
def test_zscore_cuda_handles_extreme_input_scale() -> None:
    normalizer = InputNormalizer("per_sample", output_precision="bf16", input_scale=1e-36).cuda()
    values = torch.ones((1, 1, 32, 64), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(normalizer(values), torch.zeros_like(values, dtype=torch.bfloat16), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.uint8])
def test_fixed_cuda_accepts_vmap_and_strided_inputs(input_dtype: torch.dtype) -> None:
    normalizer = InputNormalizer("fixed", output_precision="fp16", mean=0.137, standard_deviation=0.371).cuda()
    values = torch.arange(2 * 3 * 5 * 7, device="cuda").reshape(2, 3, 5, 7).to(input_dtype)
    expected = ((values.float() - 0.137) / cast("torch.Tensor", normalizer.standard_deviation)).half()
    torch.testing.assert_close(normalizer(values.transpose(-1, -2)), expected.transpose(-1, -2), rtol=0, atol=0)
    torch.testing.assert_close(torch.vmap(normalizer)(values), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "make_statistics",
    [
        pytest.param(lambda values: torch.tensor(values, requires_grad=True), id="tensor"),
        pytest.param(lambda values: array("d", values), id="buffer"),
    ],
)
def test_fixed_mode_owns_its_statistics(make_statistics) -> None:
    mean = make_statistics([1.0, 10.0])
    standard_deviation = make_statistics([2.0, 4.0])
    normalizer = InputNormalizer("fixed", output_precision="fp32", mean=mean, standard_deviation=standard_deviation)
    with torch.no_grad():
        mean[0] = 0.0
        standard_deviation[1] = 1.0

    values = torch.tensor([3.0, 18.0]).reshape(1, 2, 1, 1)
    torch.testing.assert_close(normalizer(values).flatten(), torch.tensor([1.0, 2.0]))


def test_per_sample_mode_vectorizes_statistics_over_batch_and_channels() -> None:
    values = torch.tensor(
        [
            [[[[0.0, 1.0], [2.0, 3.0]]]],
            [[[[20.0, 22.0], [24.0, 26.0]]]],
        ],
        dtype=torch.float16,
    )

    result = InputNormalizer("per_sample", output_precision="fp32")(values)

    assert result.dtype == torch.float32
    torch.testing.assert_close(result.mean(dim=(-3, -2, -1)), torch.zeros((2, 1)))
    torch.testing.assert_close(result.var(dim=(-3, -2, -1), correction=0), torch.ones((2, 1)))


@pytest.mark.parametrize("spatial_shape", [(2, 2), (2, 1, 2)])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("output_precision", "output_dtype"),
    [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_nonzero_statistics_are_independent_per_sample_and_channel(
    spatial_shape: tuple[int, ...],
    input_dtype: torch.dtype,
    output_precision: ComputePrecision,
    output_dtype: torch.dtype,
) -> None:
    values = torch.tensor(
        [[[0.0, 0.0, 10.0, 20.0], [0.0, -9.0, -3.0, 0.0]], [[1.0, 3.0, 5.0, 0.0], [-2.0, 0.0, 2.0, 0.0]]],
        dtype=input_dtype,
    ).reshape(2, 2, *spatial_shape)
    expected = torch.tensor(
        [
            [[0.0, 0.0, -1.0, 1.0], [0.0, -1.0, 1.0, 0.0]],
            [[-(1.5**0.5), 0.0, 1.5**0.5, 0.0], [-1.0, 0.0, 1.0, 0.0]],
        ],
        dtype=output_dtype,
    ).reshape_as(values)
    original = values.clone()

    result = InputNormalizer("per_sample_nonzero", output_precision=output_precision)(values)

    assert result.dtype == output_dtype
    assert result.device == values.device
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(values, original)


@pytest.mark.parametrize("spatial_shape", [(2, 2), (2, 1, 2)])
def test_nonzero_statistics_handle_empty_constant_and_single_voxel_masks(spatial_shape: tuple[int, ...]) -> None:
    values = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.0, 7.0, 7.0, 0.0]], [[0.0, -3.0, -3.0, 0.0], [0.0, 0.0, 0.0, 8.0]]]
    ).reshape(2, 2, *spatial_shape)

    result = InputNormalizer("per_sample_nonzero", output_precision="fp32")(values)

    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.zeros_like(values))


@pytest.mark.parametrize("mode", ["per_sample", "per_sample_nonzero"])
def test_per_sample_statistics_run_in_fp32_with_autocast_disabled(
    mode: InputNormalizationMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = torch.tensor([0.0, 0.0, 60000.0, 60032.0], dtype=torch.float16).reshape(1, 1, 2, 2)
    normalizer = InputNormalizer(mode, output_precision="fp32")
    rsqrt = torch.rsqrt

    def checked_rsqrt(variance: torch.Tensor) -> torch.Tensor:
        assert not torch.is_autocast_enabled("cpu")
        assert variance.dtype == torch.float32
        return rsqrt(variance)

    monkeypatch.setattr(torch, "rsqrt", checked_rsqrt)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = normalizer(values)

    expected = torch.tensor([0.0, 0.0, -1.0, 1.0]).reshape_as(values)
    if mode == "per_sample":
        expected = (values.float() - values.float().mean()) / values.float().std(correction=0)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)


def test_nonzero_statistics_support_scaled_uint8_inputs() -> None:
    values = torch.tensor([0, 0, 100, 200], dtype=torch.uint8).reshape(1, 1, 2, 2)

    result = InputNormalizer("per_sample_nonzero", output_precision="fp32", input_scale=255.0)(values)

    expected = torch.tensor([0.0, 0.0, -1.0, 1.0]).reshape_as(values)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("spatial_shape", [(65, 67), (19, 65, 67)])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nonzero_statistics_match_selected_voxels_on_strided_inputs(
    spatial_shape: tuple[int, ...], input_dtype: torch.dtype
) -> None:
    """Use explicit voxel selection as the reference for non-contiguous inputs."""
    values = (torch.arange(4 * math.prod(spatial_shape)) % 9 - 4).reshape(2, 2, *spatial_shape)
    values = values.to(dtype=input_dtype).transpose(-1, -2)
    original = values.clone()
    expected = torch.zeros_like(values, dtype=torch.float32)
    for sample in range(values.shape[0]):
        for channel in range(values.shape[1]):
            mask = values[sample, channel] != 0
            selected = values[sample, channel][mask].float() / 255.0
            expected[sample, channel][mask] = (selected - selected.mean()) / selected.std(correction=0)

    result = InputNormalizer("per_sample_nonzero", output_precision="fp32", input_scale=255.0)(values)

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(values, original)


def test_nonzero_support_is_counted_before_fp32_conversion() -> None:
    values = torch.tensor([0.0, 1e-50, 10.0, 20.0], dtype=torch.float64).reshape(1, 1, 2, 2)

    result = InputNormalizer("per_sample_nonzero", output_precision="fp32")(values)

    expected = torch.tensor([0.0, -(1.5**0.5), 0.0, 1.5**0.5]).reshape_as(values)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("mode", ["none", "fixed", "per_sample", "per_sample_nonzero"])
@pytest.mark.parametrize("input_scale", [1.0, 255.0])
@pytest.mark.parametrize(("output_precision", "output_dtype"), [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_normalization_keeps_one_fp32_workspace(
    tmp_path: Path,
    mode: InputNormalizationMode,
    input_scale: float,
    output_precision: ComputePrecision,
    output_dtype: torch.dtype,
) -> None:
    values = (torch.arange(8 * 2 * 8 * 64 * 64) % 17).reshape(8, 2, 8, 64, 64).half()
    normalizer = InputNormalizer(
        mode,
        output_precision=output_precision,
        input_scale=input_scale,
        **({"mean": 0.2, "standard_deviation": 1.1} if mode == "fixed" else {}),
    )
    result, peak_bytes = measure_peak_tensor_bytes(lambda: normalizer(values), tmp_path / "memory.json")
    fp32_bytes = values.numel() * torch.float32.itemsize
    output_bytes = values.numel() * output_dtype.itemsize
    mask_bytes = values.numel() if mode == "per_sample_nonzero" else 0
    budget = fp32_bytes + max(mask_bytes, output_bytes if output_dtype != torch.float32 else 0) + 4096
    assert peak_bytes <= budget
    assert result.dtype == output_dtype
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("mode", ["none", "fixed", "per_sample", "per_sample_nonzero"])
def test_normalization_returns_owned_storage_without_gradients(mode: InputNormalizationMode) -> None:
    values = torch.arange(60, dtype=torch.float32).reshape(2, 2, 3, 5).transpose(-1, -2).requires_grad_()
    original = values.detach().clone()
    normalizer = InputNormalizer(
        mode,
        output_precision="fp32",
        **({"mean": 0.2, "standard_deviation": 1.1} if mode == "fixed" else {}),
    )

    result = normalizer(values)

    assert not result.requires_grad
    result.zero_()
    torch.testing.assert_close(values, original)


def test_unknown_normalization_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported input normalization mode"):
        InputNormalizer(cast("InputNormalizationMode", "positive"), output_precision="fp32")


def test_fixed_mode_scales_an_unbatched_uint8_image_before_standardization() -> None:
    values = torch.full((3, 2, 2), 255, dtype=torch.uint8)

    result = InputNormalizer(
        "fixed",
        output_precision="fp16",
        mean=(0.5, 0.25, 0.75),
        standard_deviation=(0.5, 0.25, 0.25),
        input_scale=255.0,
    )(values)

    expected = torch.tensor([1.0, 3.0, 1.0]).view(3, 1, 1).expand_as(result).half()
    assert result.dtype == torch.float16
    torch.testing.assert_close(result, expected)


def test_fixed_mode_rejects_channel_count_mismatch() -> None:
    normalizer = InputNormalizer("fixed", output_precision="fp32", mean=(0.0, 1.0), standard_deviation=(1.0, 1.0))

    with pytest.raises(ValueError, match="statistics have 2 channels"):
        normalizer(torch.zeros((1, 1, 2, 2, 2)))


def test_normalization_leaves_an_already_disabled_autocast_context_alone() -> None:
    normalizer = InputNormalizer("fixed", output_precision="fp32", mean=(0.25,), standard_deviation=(0.5,))
    values = torch.tensor([[[[0.0, 0.25, 0.5, 1.0]]]], dtype=torch.float16)
    assert not torch.is_autocast_enabled("cpu")
    with patch.object(torch, "autocast", wraps=torch.autocast) as autocast:
        actual = normalizer(values)
    torch.testing.assert_close(actual, (values.float() - 0.25) / 0.5)
    autocast.assert_not_called()
