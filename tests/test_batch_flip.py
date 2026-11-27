"""Geometry, sampling, and storage contracts for flips of collated batches."""

from __future__ import annotations

import math
import multiprocessing
import py_compile
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch

import albumentations as A
import albumentations.batch.flip as flipping
from albumentations.batch import FlipPlan, apply_flip_, sample_flip
from tests.helpers.tensor_memory import measure_peak_tensor_bytes

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from albumentations.batch.flip import FlipDirection


@pytest.mark.parametrize(("direction", "transform"), [("horizontal", A.HorizontalFlip), ("vertical", A.VerticalFlip)])
def test_flip_matches_individual_compose_calls(direction: FlipDirection, transform: type) -> None:
    images = torch.arange(4 * 3 * 3 * 5, dtype=torch.uint8).reshape(4, 3, 3, 5)
    masks = images[:, 0].clone()
    original_images, original_masks = images.clone(), masks.clone()
    plan = FlipPlan(direction, torch.tensor([True, False, False, True]))
    reference = A.Compose([transform(p=1)])
    assert apply_flip_(images, plan) is images
    assert apply_flip_(masks, plan) is masks
    for index, chosen in enumerate([True, False, False, True]):
        image = original_images[index].permute(1, 2, 0).numpy()
        mask = original_masks[index].numpy()
        expected = reference(image=image, mask=mask) if chosen else {"image": image, "mask": mask}
        np.testing.assert_array_equal(images[index].permute(1, 2, 0).numpy(), expected["image"])
        np.testing.assert_array_equal(masks[index].numpy(), expected["mask"])
    apply_flip_(images, plan)
    apply_flip_(masks, plan)
    assert torch.equal(images, original_images)
    assert torch.equal(masks, original_masks)


@pytest.fixture(
    params=[
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda", 0), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
    ids=["cpu", "cuda"],
)
def device(request: pytest.FixtureRequest) -> torch.device:
    return request.param


@pytest.fixture(params=["horizontal", "vertical"])
def direction(request: pytest.FixtureRequest) -> FlipDirection:
    return request.param


def _reverse(values: torch.Tensor, direction: FlipDirection) -> torch.Tensor:
    axis = -1 if direction == "horizontal" else -2
    positions = torch.arange(values.size(axis) - 1, -1, -1, device=values.device)
    return values.index_select(axis, positions)


def test_two_plans_align_images_volumes_and_lower_resolution_masks(device: torch.device) -> None:
    horizontal = FlipPlan("horizontal", torch.tensor([False, True, False, True], device=device))
    vertical = FlipPlan("vertical", torch.tensor([False, False, True, True], device=device))
    for shape in [(4, 3, 5), (4, 2, 6, 10), (4, 2, 3, 4, 7)]:
        values = torch.arange(np.prod(shape), device=device).reshape(shape)
        original = values.clone()
        apply_flip_(values, horizontal)
        apply_flip_(values, vertical)
        for index in range(4):
            expected = original[index]
            if index in (1, 3):
                expected = _reverse(expected, "horizontal")
            if index in (2, 3):
                expected = _reverse(expected, "vertical")
            assert torch.equal(values[index], expected)
        apply_flip_(values, horizontal)
        apply_flip_(values, vertical)
        assert torch.equal(values, original)


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0), (2, 0, 3, 4), (1, 1, 3), (1, 3, 1)])
def test_empty_and_singleton_axes(device: torch.device, direction: FlipDirection, shape: tuple[int, ...]) -> None:
    values = torch.arange(np.prod(shape), device=device).reshape(shape)
    expected = _reverse(values, direction)
    assert apply_flip_(values, FlipPlan(direction, torch.ones(shape[0], device=device, dtype=torch.bool))) is values
    assert torch.equal(values, expected)


@pytest.mark.parametrize("rank", [64, 65, 66])
def test_high_rank_via_original_storage(device: torch.device, direction: FlipDirection, rank: int) -> None:
    parent = torch.arange(48, device=device).reshape(2, 4, 6)
    expected = parent.clone()
    crop = parent[:, 1:3, 1:4]
    values = crop.view((2,) + (1,) * (rank - 3) + (2, 3))
    strides, pointer = values.stride(), values.data_ptr()
    expected[0, 1:3, 1:4] = _reverse(crop[0], direction)
    assert apply_flip_(values, FlipPlan(direction, torch.tensor([True, False], device=device))) is values
    assert values.stride() == strides
    assert values.data_ptr() == pointer
    assert torch.equal(parent, expected)


def test_high_rank_with_three_middle_axes(device: torch.device, direction: FlipDirection) -> None:
    parent = torch.arange(2 * 2 * 2 * 2 * 4 * 5, device=device).reshape(2, 2, 2, 2, 4, 5)
    expected = parent.clone()
    crop = parent[..., 1:3, 1:4]
    values = crop.view((2, 2, 2, 2) + (1,) * 60 + (2, 3))
    strides, pointer = values.stride(), values.data_ptr()
    expected[1, ..., 1:3, 1:4] = _reverse(crop[1], direction)
    assert apply_flip_(values, FlipPlan(direction, torch.tensor([False, True], device=device))) is values
    assert values.stride() == strides
    assert values.data_ptr() == pointer
    assert torch.equal(parent, expected)


def test_manual_plan_snapshots_a_view_of_the_target(device: torch.device) -> None:
    mask = torch.tensor([[[True, False]]], device=device)
    plan = FlipPlan("horizontal", mask[:, 0, 0])
    for expected in ([[[False, True]]], [[[True, False]]]):
        apply_flip_(mask, plan)
        assert mask.tolist() == expected
        assert plan.selected.tolist() == [True]

    source = torch.tensor([True, True, False, True], device=device)
    plan = FlipPlan("vertical", source[::2])
    source.fill_(False)
    plan.selected.fill_(True)
    assert plan.direction == "vertical"
    assert plan.selected.tolist() == [True, False]
    values = torch.arange(8, device=device).reshape(2, 2, 2)
    apply_flip_(values, plan)
    assert values.tolist() == [[[2, 3], [0, 1]], [[4, 5], [6, 7]]]
    with pytest.raises(AttributeError):
        plan.direction = "horizontal"
    with pytest.raises(AttributeError):
        plan.selected = source


def _layout(values: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "channels_last":
        return values.contiguous(memory_format=torch.channels_last)
    if layout == "transpose":
        return values.transpose(-1, -2)
    if layout == "step":
        return values[..., 1::2]
    return values


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transpose", "step"])
def test_in_place_values_and_layout(
    device: torch.device, direction: FlipDirection, dtype: torch.dtype, layout: str
) -> None:
    numbers = torch.arange(2 * 3 * 4 * 7, device=device).reshape(2, 3, 4, 7)
    parent = (numbers % 3 == 0) if dtype is torch.bool else numbers.to(dtype)
    values = _layout(parent, layout)
    alias = values.view_as(values)
    original = values.clone()
    expected_parent = parent.clone()
    if layout == "step":
        expected_parent[1, ..., 1::2] = _reverse(original[1], direction)
    strides, pointer = values.stride(), values.data_ptr()
    result = apply_flip_(values, FlipPlan(direction, torch.tensor([False, True], device=device)))
    assert result is values
    assert (result.shape, result.dtype, result.device) == (original.shape, dtype, device)
    assert (result.stride(), result.data_ptr()) == (strides, pointer)
    assert torch.equal(alias[0], original[0])
    assert torch.equal(alias[1], _reverse(original[1], direction))
    if layout == "step":
        assert torch.equal(parent, expected_parent)


def _binary64_from_bits(bits: int) -> float:
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def _assert_exact_threshold(
    device: torch.device, probability: float, numerator: int, width: int, outcomes: list[int]
) -> None:
    generator = torch.Generator(device=device).manual_seed(137)
    shift = 0

    def known_words(high, size, **kwargs):
        nonlocal shift
        bits = min(30, width - shift)
        assert high == 1 << bits
        assert size == (len(outcomes),)
        assert kwargs == {"dtype": torch.int32, "device": device, "generator": generator}
        words = [(outcome >> shift) % high for outcome in outcomes]
        shift += bits
        return torch.tensor(words, dtype=torch.int32, device=device)

    with patch.object(flipping, "randint", known_words):
        plan = sample_flip(
            batch_size=len(outcomes), direction="horizontal", p=probability, device=device, generator=generator
        )
    assert shift == width
    assert plan.selected.tolist() == [outcome < numerator for outcome in outcomes]


@pytest.mark.parametrize(
    ("probability", "numerator", "width", "outcomes"),
    [
        (3 / 8, 3, 3, list(range(8))),
        (1 - 2**-30, 2**30 - 1, 30, [2**30 - 2, 2**30 - 1]),
        ((2**30 + 1) / 2**31, 2**30 + 1, 31, [2**30, 2**30 + 1, 2**30 + 2]),
        (_binary64_from_bits(1), 1, 1074, [0, *(1 << bit for bit in range(1074))]),
        (_binary64_from_bits(2**52 - 1), 2**52 - 1, 1074, [2**52 - 2, 2**52 - 1]),
        (2**-1022, 1, 1022, [0, 1]),
    ],
    ids=["three-eighths", "below-one", "two-words", "smallest-subnormal", "largest-subnormal", "smallest-normal"],
)
def test_exact_binary64_threshold(
    device: torch.device, probability: float, numerator: int, width: int, outcomes: list[int]
) -> None:
    _assert_exact_threshold(device, probability, numerator, width, outcomes)


def _check_subnormal_sampling_modes() -> bool:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(137)
    state = generator.get_state()
    for flush in (False, True):
        if not torch.set_flush_denormal(flush):
            return False
        _assert_exact_threshold(device, _binary64_from_bits(1), 1, 1074, [0, 1])
        _assert_exact_threshold(device, _binary64_from_bits(6), 3, 1073, [2, 3])
        with pytest.raises(ValueError, match="probability"):
            sample_flip(
                batch_size=2, direction="vertical", device=device, generator=generator, p=_binary64_from_bits(2**63 + 1)
            )
        plan = sample_flip(batch_size=2, direction="vertical", device=device, generator=generator, p=-0.0)
        assert plan.selected.tolist() == [False, False]
        assert torch.equal(generator.get_state(), state)
    return True


def test_subnormal_sampling_with_flush_denormal() -> None:
    # A fresh process isolates both floating-point modes from the rest of the suite.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        if not executor.submit(_check_subnormal_sampling_modes).result():
            pytest.skip("This CPU does not support changing the denormal mode.")


def _default_state(device: torch.device) -> torch.Tensor:
    return torch.get_rng_state() if device.type == "cpu" else torch.cuda.get_rng_state(device)


@pytest.mark.parametrize("probability", [0.25, 0.75, 0.2])
def test_explicit_rng_reproduces_decisions_and_next_draw(device: torch.device, probability: float) -> None:
    first, second = (torch.Generator(device=device).manual_seed(137) for _ in range(2))
    before = _default_state(device)
    args = {"batch_size": 137, "direction": "vertical", "device": device, "p": probability}
    plan = sample_flip(**args, generator=first)
    assert torch.equal(plan.selected, sample_flip(**args, generator=second).selected)
    if probability in (0.25, 0.75):
        reference = torch.Generator(device=device).manual_seed(137)
        uniform = torch.randint(4, (137,), dtype=torch.int32, device=device, generator=reference)
        assert torch.equal(plan.selected, uniform < int(4 * probability))
    next_draw = torch.rand(8, device=device, generator=first)
    assert torch.equal(next_draw, torch.rand(8, device=device, generator=second))
    state = first.get_state()
    mask = plan.selected
    values = torch.arange(137 * 6, device=device).reshape(137, 2, 3)
    original = values.clone()
    apply_flip_(values, plan)
    assert torch.equal(values[mask], _reverse(original[mask], "vertical"))
    assert torch.equal(values[~mask], original[~mask])
    assert torch.equal(plan.selected, mask)
    assert torch.equal(first.get_state(), state)
    assert torch.equal(_default_state(device), before)


@pytest.mark.parametrize(("count", "probability"), [(0, 0.2), (4, 0), (4, 1)])
def test_deterministic_selection_preserves_rng(device: torch.device, count: int, probability: float) -> None:
    generator = torch.Generator(device=device).manual_seed(137)
    before, default_before = generator.get_state(), _default_state(device)
    plan = sample_flip(batch_size=count, direction="vertical", device=device, generator=generator, p=probability)
    assert plan.selected.device == device
    assert plan.selected.tolist() == [probability == 1] * count
    assert torch.equal(before, generator.get_state())
    assert torch.equal(default_before, _default_state(device))


@pytest.mark.parametrize(
    "invalid",
    [
        {"p": float("nan")},
        {"p": float("inf")},
        {"p": -float("inf")},
        {"p": -0.1},
        {"p": 1.01},
        {"p": 2**1024},
        {"p": "0.5"},
        {"p": True},
        {"batch_size": -1},
        {"batch_size": 1.5},
        {"batch_size": True},
        {"direction": "diagonal"},
        {"direction": []},
        {"device": "meta"},
        {"device": "broken"},
        {"device": "cpu:1"},
        {"device": "cpu:256"},
        {"device": "cpu:1", "p": 1},
        {"device": "cpu:256", "batch_size": 0},
        {"device": None},
        {"generator": None},
        {"device": "cuda", "p": 0},
        {"device": "cuda:1", "p": 1},
        {"device": "cuda", "batch_size": 0},
        {"p": float("nan"), "batch_size": 0},
    ],
)
def test_invalid_sampling_precedes_rng_and_retry(invalid: dict) -> None:
    generator = torch.Generator().manual_seed(137)
    args = {"batch_size": 17, "direction": "horizontal", "device": "cpu", "generator": generator}
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        sample_flip(**(args | invalid))
    reference = torch.Generator().manual_seed(137)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert torch.equal(sample_flip(**args).selected, sample_flip(**(args | {"generator": reference})).selected)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("device", ["cpu:0", torch.device("cpu:0")])
def test_cpu_zero_index_matches_generator(device: torch.device | str) -> None:
    generator = torch.Generator().manual_seed(137)
    reference = torch.Generator().manual_seed(137)
    plan = sample_flip(batch_size=17, direction="horizontal", device=device, generator=generator)
    expected = sample_flip(batch_size=17, direction="horizontal", device="cpu", generator=reference)
    assert plan.selected.device == torch.device("cpu")
    assert torch.equal(plan.selected, expected.selected)
    assert torch.equal(generator.get_state(), reference.get_state())


@pytest.mark.parametrize("kind", ["dtype", "rank", "sparse", "subclass", "meta", "list"])
def test_invalid_manual_selection(kind: str) -> None:
    selected = torch.tensor([True, False])
    cases = {
        "dtype": selected.long(),
        "rank": selected.view(2, 1),
        "sparse": selected.to_sparse(),
        "subclass": torch.nn.Parameter(selected, requires_grad=False),
        "meta": selected.to("meta"),
        "list": [True, False],
    }
    with pytest.raises(ValueError):
        FlipPlan("horizontal", cases[kind])
    with pytest.raises(ValueError):
        FlipPlan("other", selected)
    assert selected.tolist() == [True, False]


@pytest.mark.parametrize("selection", [[False, True], [False, False]])
@pytest.mark.parametrize(
    "kind",
    [
        "dtype",
        "rank",
        "batch_size",
        "subclass",
        "grad",
        "negative",
        "expand",
        "overlap",
        "interleave",
    ],
)
def test_invalid_values_preserve_storage(device: torch.device, selection: list[bool], kind: str) -> None:
    parent = torch.arange(12, dtype=torch.float32, device=device)
    values = parent.view(2, 2, 3)
    if kind == "dtype":
        parent = parent.to(torch.complex64)
        values = parent.view(2, 2, 3)
    elif kind == "rank":
        values = parent.view(2, 6)
    elif kind == "batch_size":
        values = parent.view(3, 2, 2)
    elif kind == "subclass":
        values = torch.nn.Parameter(values, requires_grad=False)
    elif kind == "grad":
        values.requires_grad_()
    elif kind == "negative":
        values = torch._neg_view(values)
    elif kind == "expand":
        values = values[:1].expand_as(values)
    elif kind == "overlap":
        values = parent.as_strided((2, 1, 2), (1, 2, 1))
    elif kind == "interleave":
        values = parent.as_strided((2, 2, 2), (2, 3, 4))
    original = parent.clone()
    plan = FlipPlan("horizontal", torch.tensor(selection, device=device))
    with pytest.raises(ValueError):
        apply_flip_(values, plan)
    assert torch.equal(parent, original)
    assert plan.selected.tolist() == selection


@pytest.mark.parametrize("selection", [[True, False], [False, False]])
def test_sparse_and_nested_targets_preserve_storage(device: torch.device, selection: list[bool]) -> None:
    plan = FlipPlan("vertical", torch.tensor(selection, device=device))
    dense = torch.arange(12, device=device).reshape(2, 2, 3)
    sparse = dense.to_sparse().coalesce()
    indices, entries = sparse.indices().clone(), sparse.values().clone()
    with pytest.raises(ValueError):
        apply_flip_(sparse, plan)
    assert torch.equal(sparse.indices(), indices)
    assert torch.equal(sparse.values(), entries)
    nested = torch.nested.nested_tensor([dense[0], dense[1, :, :1]], device=device)
    original = nested.values().clone()
    with pytest.raises(ValueError):
        apply_flip_(nested, plan)
    assert torch.equal(nested.values(), original)
    assert plan.selected.tolist() == selection


@pytest.mark.parametrize("selection", [[True, False], [False, False]])
def test_width_quantization_is_rejected_before_writing(selection: list[bool]) -> None:
    values = torch.quantize_per_channel(
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        torch.tensor([0.1, 0.5]),
        torch.tensor([3, 7]),
        axis=2,
        dtype=torch.quint8,
    )
    original = values.int_repr()
    scales, zeros, axis = (
        values.q_per_channel_scales().clone(),
        values.q_per_channel_zero_points().clone(),
        values.q_per_channel_axis(),
    )
    plan = FlipPlan("horizontal", torch.tensor(selection))
    with pytest.raises(ValueError):
        apply_flip_(values, plan)
    assert torch.equal(values.int_repr(), original)
    assert torch.equal(values.q_per_channel_scales(), scales)
    assert torch.equal(values.q_per_channel_zero_points(), zeros)
    assert values.q_per_channel_axis() == axis
    assert plan.selected.tolist() == selection


@pytest.mark.skipif(not hasattr(torch.Tensor, "refine_names"), reason="PyTorch build has no named tensors")
def test_named_values_and_selection_are_rejected() -> None:
    plain = torch.arange(12).reshape(2, 2, 3)
    original = plain.clone()
    mask = torch.tensor([False, False])
    with pytest.raises(ValueError):
        FlipPlan("horizontal", mask.refine_names("B"))
    plan = FlipPlan("vertical", mask)
    with pytest.raises(ValueError):
        apply_flip_(plain.refine_names("B", "H", "W"), plan)
    assert torch.equal(plain, original)
    assert torch.equal(plan.selected, mask)


@pytest.mark.parametrize("inference_mode", [False, True])
def test_forward_ad_and_invalid_arguments(inference_mode: bool) -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    original = values.clone()
    tangent = torch.ones_like(values)
    plan = FlipPlan("horizontal", torch.tensor([True, False]))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(values, tangent)
        with torch.inference_mode(inference_mode), pytest.raises(ValueError):
            apply_flip_(dual, plan)
    for invalid in [values.to("meta"), [], None]:
        with pytest.raises(ValueError):
            apply_flip_(invalid, plan)
    with pytest.raises(ValueError):
        apply_flip_(values, None)
    assert torch.equal(tangent, torch.ones_like(values))
    assert torch.equal(values, original)
    assert plan.selected.tolist() == [True, False]


def test_immutable_zero_storage_is_rejected() -> None:
    mask = torch._efficientzerotensor((2,), dtype=torch.bool)
    with pytest.raises(ValueError):
        FlipPlan("horizontal", mask)
    values = torch._efficientzerotensor((2, 2, 3))
    plan = FlipPlan("vertical", torch.tensor([False, False]))
    with pytest.raises(ValueError):
        apply_flip_(values, plan)
    assert torch.equal(values.clone(), torch.zeros(2, 2, 3))
    assert plan.selected.tolist() == [False, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_generator_and_plan_device_boundaries() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(137)
    state = generator.get_state()
    mismatches = (torch.device("cpu"), torch.device("cuda", device.index + 1), "cuda:255", f"cuda:{device.index + 256}")
    for size, probability in [(0, 0.5), (4, 0), (4, 1), (4, 0.5)]:
        for mismatch in mismatches:
            with pytest.raises(ValueError):
                sample_flip(batch_size=size, direction="vertical", p=probability, device=mismatch, generator=generator)
    assert torch.equal(state, generator.get_state())
    requested = sample_flip(batch_size=17, direction="vertical", device=str(device), generator=generator)
    reference = torch.Generator(device=device).manual_seed(137)
    expected = sample_flip(batch_size=17, direction="vertical", device=device, generator=reference)
    assert torch.equal(requested.selected, expected.selected)
    assert torch.equal(generator.get_state(), reference.get_state())
    plan = sample_flip(batch_size=2, direction="horizontal", p=0, device="cuda", generator=generator)
    assert plan.selected.device == device
    values = torch.arange(12).reshape(2, 2, 3)
    original = values.clone()
    with pytest.raises(ValueError):
        apply_flip_(values, plan)
    assert torch.equal(values, original)
    assert plan.selected.tolist() == [False, False]


def test_inference_and_model_backward(device: torch.device) -> None:
    plan = FlipPlan("horizontal", torch.tensor([True, False], device=device))
    original = torch.arange(12, device=device, dtype=torch.float32).reshape(2, 2, 3)
    expected = original.clone()
    expected[0] = _reverse(original[0], "horizontal")
    with torch.inference_mode():
        inference = original.clone()
        pointer = inference.data_ptr()
        assert apply_flip_(inference, plan) is inference
        assert inference.data_ptr() == pointer
        assert torch.equal(inference, expected)
    with pytest.raises(ValueError):
        apply_flip_(inference, plan)
    assert torch.equal(inference, expected)
    apply_flip_(original, plan)
    model = torch.nn.Linear(6, 1, bias=False, device=device)
    model(original.flatten(1)).sum().backward()
    assert torch.equal(model.weight.grad, expected.flatten(1).sum(dim=0, keepdim=True))


@pytest.fixture
def triton_flip_backend() -> ModuleType:
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("Requires a Triton-capable NVIDIA GPU.")
    pytest.importorskip("triton")
    return import_module("albumentations.batch._flip_cuda")


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_cuda_triton_flip_bits_and_aliases(
    triton_flip_backend: ModuleType, dtype: torch.dtype, direction: FlipDirection, count: int
) -> None:
    shape = (4, 3, 17, 73)  # Odd centre and a partial final tile on both axes.
    parent = torch.arange(math.prod(shape) + 2, device="cuda").to(dtype)
    bits = parent.view(torch.int32) if dtype == torch.float32 else parent
    if dtype == torch.float32:
        bits[1:5] = torch.tensor([0x7FC01234, -2147483648, 0x7F800000, 1], device="cuda", dtype=torch.int32)
    values = parent[1:-1].view(shape)
    expected = bits.clone()
    chosen = [0, 3, 1, 2][:count]
    selected = torch.zeros(len(values), device=values.device, dtype=torch.bool)
    selected[chosen] = True
    plan = FlipPlan(direction, selected)
    for index in chosen:
        expected[1:-1].view(shape)[index].copy_(_reverse(bits[1:-1].view(shape)[index], direction))
    version = parent._version
    with (
        patch.object(triton_flip_backend, "try_flip", wraps=triton_flip_backend.try_flip) as fused,
        patch.object(flipping, "_flip_cuda_indexed", side_effect=AssertionError("Unexpected fallback")),
    ):
        assert apply_flip_(values, plan) is values
    fused.assert_called_once()
    assert torch.equal(bits, expected) and torch.equal(plan.selected, selected)
    assert parent._version == version + 1


@pytest.mark.usefixtures("triton_flip_backend")
def test_cuda_triton_updates_saved_tensor_and_accepts_inference() -> None:
    values = torch.arange(12, dtype=torch.float32, device="cuda").reshape(2, 1, 2, 3)
    plan = FlipPlan("horizontal", torch.tensor([True, False], device=values.device))
    weight = torch.ones_like(values, requires_grad=True)
    loss = (weight * values).sum()
    expected = values.clone()
    expected[0].copy_(_reverse(values[0], "horizontal"))
    with patch.object(flipping, "_flip_cuda_indexed", side_effect=AssertionError("Unexpected fallback")):
        with torch.inference_mode():
            inference = values.clone()
            assert apply_flip_(inference, plan) is inference
            assert torch.equal(inference, expected)
        apply_flip_(values, plan)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.usefixtures("triton_flip_backend")
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((65536, 1, 1, 3), "horizontal"),
        ((8, 1, 2047, 2051), "horizontal"),
        ((16, 1, 2049, 2080), "horizontal"),
        ((16, 1, 2049, 2080), "vertical"),
    ],
)
def test_cuda_triton_large_grid(shape: tuple[int, ...], axis: FlipDirection) -> None:
    values = torch.arange(math.prod(shape), dtype=torch.uint8, device="cuda").view(shape)
    selected = torch.arange(len(values), device=values.device) == len(values) - 1
    expected = values.clone()
    expected[-1].copy_(_reverse(values[-1], axis))
    with patch.object(flipping, "_flip_cuda_indexed", side_effect=AssertionError("Unexpected fallback")):
        assert apply_flip_(values, FlipPlan(axis, selected)) is values
    assert torch.equal(values, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
@pytest.mark.parametrize("kind", ["float16", "step", "channels_last", "BHW", "BCDHW", "rank6", "planes"])
@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_cuda_hybrid_uses_selected_indices(kind: str, count: int) -> None:
    parent = torch.arange(4 * 3 * 5 * 8, device="cuda", dtype=torch.float32).reshape(4, 3, 5, 8)
    values = {
        "float16": parent.half,
        "step": lambda: parent[..., 1::2],
        "channels_last": lambda: parent.to(memory_format=torch.channels_last),
        "BHW": lambda: parent[:, 0],
        "BCDHW": lambda: parent.unsqueeze(2),
        "rank6": lambda: parent[:, :, None, None],
        "planes": lambda: parent[:, :, None, None].repeat(1, 1, 2, 2, 1, 1),
    }[kind]()
    selected = torch.arange(4, device=values.device) < count
    plan = FlipPlan("vertical", selected)
    expected = values.clone()
    for index in range(count):
        expected[index].copy_(_reverse(values[index], "vertical"))
    with (
        patch.object(flipping, "_cuda_flip_backend", side_effect=AssertionError("Ineligible Triton input")),
        patch.object(flipping, "_flip_cuda_indexed", wraps=flipping._flip_cuda_indexed) as hybrid,
    ):
        assert apply_flip_(values, plan) is values
    hybrid.assert_called_once()
    assert torch.equal(hybrid.call_args.args[1], torch.arange(count, device=values.device))
    assert torch.equal(values, expected)
    assert torch.equal(plan.selected, selected)


@pytest.mark.skipif(torch.version.cuda is None or not torch.cuda.is_available(), reason="Requires NVIDIA CUDA.")
@pytest.mark.parametrize("missing", ["capability", "triton", "kernel_source"])
def test_cuda_flip_with_unavailable_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str) -> None:
    values = torch.arange(60, device="cuda", dtype=torch.float32).reshape(4, 1, 3, 5)
    selected = torch.tensor([True, False, False, True], device=values.device)
    expected = values.clone()
    expected[[0, 3]] = _reverse(values[[0, 3]], "horizontal")
    monkeypatch.delitem(sys.modules, "albumentations.batch._flip_cuda", raising=False)
    if missing == "triton":
        monkeypatch.setitem(sys.modules, "triton", None)
    elif missing == "kernel_source":
        pytest.importorskip("triton")
        source = Path(flipping.__file__).with_name("_flip_cuda.py")
        py_compile.compile(str(source), str(tmp_path / "_flip_cuda.pyc"), str(tmp_path / source.name), doraise=True)
        monkeypatch.setattr(import_module("albumentations.batch"), "__path__", [str(tmp_path)])
    flipping._cuda_flip_backend.cache_clear()
    try:
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)) as capability,
            patch.object(flipping, "import_module", wraps=import_module) as backend_import,
            patch.object(flipping, "_flip_cuda_indexed", wraps=flipping._flip_cuda_indexed) as hybrid,
        ):
            if missing == "capability":
                capability.side_effect = RuntimeError("CUDA capability query failed")
            assert apply_flip_(values, FlipPlan("horizontal", selected)) is values
        capability.assert_called_once_with(values.device)
        if missing == "capability":
            backend_import.assert_not_called()
        else:
            backend_import.assert_called_once_with("albumentations.batch._flip_cuda")
        hybrid.assert_called_once()
        assert torch.equal(values, expected)
    finally:
        flipping._cuda_flip_backend.cache_clear()


@pytest.mark.usefixtures("triton_flip_backend")
def test_cuda_flip_invalid_triton_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRITON_OVERRIDE_ARCH", "invalid")
    values = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 1, 3, 4)
    plan = FlipPlan("horizontal", torch.tensor([True, False], device=values.device))
    expected = values.clone()
    expected[0].copy_(_reverse(values[0], "horizontal"))
    assert apply_flip_(values, plan) is values
    assert torch.equal(values, expected)


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("warmup", RuntimeError("Compiler unavailable")),
        ("warmup", ValueError("Invalid architecture")),
        ("launcher", AssertionError("CUDA driver library unavailable")),
    ],
    ids=["compiler", "architecture", "driver"],
)
def test_cuda_flip_falls_back_before_launch(triton_flip_backend: ModuleType, stage: str, error: Exception) -> None:
    values = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 1, 3, 4)
    plan = FlipPlan("vertical", torch.tensor([False, True], device=values.device))
    expected = values.clone()
    expected[1].copy_(_reverse(values[1], "vertical"))
    with (
        patch.object(triton_flip_backend._swap_pairs, "warmup") as prepare,
        patch.object(flipping, "_flip_cuda_indexed", wraps=flipping._flip_cuda_indexed) as hybrid,
    ):
        if stage == "warmup":
            prepare.side_effect = error
        else:
            prepare.return_value.__getitem__.side_effect = error
        assert apply_flip_(values, plan) is values
    hybrid.assert_called_once()
    assert torch.equal(values, expected)


def test_cuda_flip_propagates_launch_error(triton_flip_backend: ModuleType) -> None:
    values = torch.ones((2, 1, 2, 3), device="cuda", dtype=torch.float32)
    plan = FlipPlan("vertical", torch.ones(2, device=values.device, dtype=torch.bool))
    with (
        patch.object(triton_flip_backend._swap_pairs, "warmup") as prepare,
        patch.object(flipping, "_flip_cuda_indexed", side_effect=AssertionError("Retried an in-place launch")),
    ):
        prepare.return_value.__getitem__.return_value.side_effect = RuntimeError("Launch failed")
        with pytest.raises(RuntimeError, match="Launch failed"):
            apply_flip_(values, plan)


@pytest.mark.usefixtures("triton_flip_backend")
def test_cuda_flip_error_after_write_invalidates_saved_tensor() -> None:
    knobs = pytest.importorskip("triton.knobs")
    values = torch.arange(3, dtype=torch.float32, device="cuda").reshape(1, 1, 1, 3)
    expected = _reverse(values, "horizontal")
    plan = FlipPlan("horizontal", torch.ones(1, dtype=torch.bool, device=values.device))
    weight = torch.ones_like(values, requires_grad=True)
    loss = (weight * values).sum()

    def fail_after_launch(_metadata: object) -> None:
        raise RuntimeError("Failed after flip launch")

    with (
        patch.object(knobs.runtime, "launch_exit_hook", fail_after_launch),
        pytest.raises(RuntimeError, match="Failed after flip launch"),
    ):
        apply_flip_(values, plan)
    assert torch.equal(values, expected)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_cuda_hybrid_deterministic_sparse_selection(tmp_path: Path) -> None:
    parent = torch.arange(16 * 4 * 128 * 256, device="cuda", dtype=torch.float32).reshape(16, 4, 128, 256)
    values = parent[..., 1::2]
    selected = torch.arange(len(values), device=values.device) % 8 == 3
    plan = FlipPlan("horizontal", selected)
    expected = parent.clone()
    for index in (3, 11):
        expected[index, ..., 1::2].copy_(_reverse(values[index], "horizontal"))
    previous = torch.get_deterministic_debug_mode()
    try:
        torch.use_deterministic_algorithms(True)
        result, peak = _measure_peak(lambda: apply_flip_(values, plan), values.device, tmp_path / "unused.json")
    finally:
        torch.set_deterministic_debug_mode(previous)
    assert result is values and torch.equal(parent, expected)
    assert peak <= 5 * values[0].numel() * values.element_size()


def test_cpu_flip_buffers_one_sample_at_a_time(tmp_path: Path) -> None:
    values = torch.zeros((16, 3, 64, 64), dtype=torch.float32)
    plan = FlipPlan("horizontal", torch.arange(len(values)) % 2 == 0)
    _, peak = measure_peak_tensor_bytes(lambda: apply_flip_(values, plan), tmp_path / "flip.json")
    sample_bytes = values[0].numel() * values.element_size()
    assert peak <= 2 * sample_bytes


def _measure_peak(operation: Callable[[], torch.Tensor], device: torch.device, path: Path) -> tuple[torch.Tensor, int]:
    if device.type == "cpu":
        return measure_peak_tensor_bytes(operation, path)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = operation()
    torch.cuda.synchronize(device)
    return result, torch.cuda.max_memory_allocated(device) - before


@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transpose"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_temporary_storage_tracks_selected_pixels(
    device: torch.device, direction: FlipDirection, layout: str, deterministic: bool, tmp_path: Path
) -> None:
    previous = torch.get_deterministic_debug_mode()
    try:
        torch.use_deterministic_algorithms(deterministic)
        values = _layout(
            torch.arange(16 * 4 * 128 * 512, dtype=torch.float32, device=device).reshape(16, 4, 128, 512), layout
        )
        chosen = torch.arange(16, device=device) == 7
        plan = FlipPlan(direction, chosen)
        expected = values.clone()
        expected[7].copy_(_reverse(values[7], direction))
        strides, pointer = values.stride(), values.data_ptr()
        apply_flip_(values, plan)
        apply_flip_(values, plan)
        result, peak = _measure_peak(lambda: apply_flip_(values, plan), device, tmp_path / "flip.json")
        control, control_peak = _measure_peak(values.clone, device, tmp_path / "clone.json")
        assert control_peak >= 16 * 1024**2
        assert peak <= 4 * 1024**2
        assert result is values
        assert (result.stride(), result.data_ptr()) == (strides, pointer)
        assert torch.equal(result, expected)
        assert torch.equal(control, expected)
    finally:
        torch.set_deterministic_debug_mode(previous)
