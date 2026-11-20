"""Geometry, sampling, and storage contracts for flips of collated batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import albumentations as A
import albumentations.batch.flip as flipping
from albumentations.batch import FlipPlan, apply_flip_, sample_flip
from tests.helpers.tensor_memory import measure_peak_tensor_bytes

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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


@pytest.mark.parametrize(
    ("probability", "outcomes"),
    [
        (3 / 8, list(range(8))),
        (1 - 2**-30, [2**30 - 2, 2**30 - 1]),
        ((2**30 + 1) / 2**31, [2**30, 2**30 + 1, 2**30 + 2]),
        (2**-1074, [0, *(1 << bit for bit in range(1074))]),
    ],
    ids=["three-eighths", "below-one", "two-words", "smallest-subnormal"],
)
def test_exact_binary64_threshold(
    device: torch.device, probability: float, outcomes: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    numerator, denominator = probability.as_integer_ratio()
    width = denominator.bit_length() - 1
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

    monkeypatch.setattr(flipping, "randint", known_words)
    plan = sample_flip(
        batch_size=len(outcomes), direction="horizontal", p=probability, device=device, generator=generator
    )
    assert shift == width
    assert plan.selected.tolist() == [outcome < numerator for outcome in outcomes]


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
        {"p": "0.5"},
        {"p": True},
        {"batch_size": -1},
        {"batch_size": 1.5},
        {"batch_size": True},
        {"direction": "diagonal"},
        {"direction": []},
        {"device": "meta"},
        {"device": "broken"},
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
    for size, probability in [(0, 0.5), (4, 0), (4, 1)]:
        for mismatch in [torch.device("cpu"), torch.device("cuda", device.index + 1)]:
            with pytest.raises(ValueError):
                sample_flip(batch_size=size, direction="vertical", p=probability, device=mismatch, generator=generator)
    assert torch.equal(state, generator.get_state())
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
