"""Behavior and memory contracts for batch MixUp."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

import albumentations.batch.mixup as mixing
from albumentations.batch import InputNormalizer, MixUpPlan, apply_mixup, sample_mixup
from tests.helpers.tensor_memory import measure_peak_tensor_bytes

if TYPE_CHECKING:
    from pathlib import Path


def test_mixup_pairs_leave_an_odd_sample_untouched() -> None:
    plan = sample_mixup(
        batch_size=3, alpha=1.0, device=torch.device("cpu"), generator=torch.Generator().manual_seed(137)
    )
    assert torch.count_nonzero(plan.permutation == torch.arange(3)) == 1
    untouched = int(torch.nonzero(plan.permutation == torch.arange(3), as_tuple=False).item())
    assert plan.lambdas[untouched] == 1


@pytest.mark.parametrize("mode", ["pair", "element", "batch"])
def test_mixup_draws_are_reproducible_from_the_supplied_generator(mode: mixing.MixingMode) -> None:
    first = sample_mixup(
        batch_size=4,
        alpha=0.7,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(137),
        mode=mode,
    )
    second = sample_mixup(
        batch_size=4,
        alpha=0.7,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(137),
        mode=mode,
    )
    torch.testing.assert_close(first.permutation, second.permutation)
    torch.testing.assert_close(first.lambdas, second.lambdas)


def test_mixup_batch_mode_shares_one_weight_except_for_the_unpaired_sample() -> None:
    plan = sample_mixup(
        batch_size=5,
        alpha=1.0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(137),
        mode="batch",
    )
    paired = plan.permutation != torch.arange(5)
    assert paired.sum() == 4
    torch.testing.assert_close(plan.lambdas[~paired], torch.ones(1))
    torch.testing.assert_close(plan.lambdas[paired], plan.lambdas[paired][0].expand(4))


def test_mixup_preserves_uint8_transport_contract_with_explicit_rounding() -> None:
    plan = MixUpPlan(permutation=torch.tensor([1, 0]), lambdas=torch.tensor([0.5, 0.5]))
    mixed = apply_mixup(torch.tensor([[[[0]]], [[[255]]]], dtype=torch.uint8), plan)
    assert mixed.dtype is torch.uint8
    assert mixed.flatten().tolist() == [128, 128]


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
def test_mixup_flips_only_the_donor_width(dtype: torch.dtype) -> None:
    values = torch.tensor([[[[0, 4], [8, 12]]], [[[20, 28], [32, 40]]]], dtype=dtype)
    original = values.clone()
    plan = MixUpPlan(torch.tensor([1, 0]), torch.tensor([0.75, 0.5]))
    expected = torch.tensor([[[[7, 8], [16, 17]]], [[[12, 14], [22, 24]]]], dtype=dtype)
    torch.testing.assert_close(apply_mixup(values, plan, flip_donor=True), expected)
    torch.testing.assert_close(apply_mixup(values, plan, flip_donor=False), apply_mixup(values, plan))
    torch.testing.assert_close(values, original)


def test_mixup_rejects_flipping_class_targets() -> None:
    plan = MixUpPlan(torch.tensor([1, 0]), torch.tensor([0.75, 0.5]))
    with pytest.raises(ValueError, match="spatial tensors"):
        apply_mixup(torch.ones(2, 10), plan, flip_donor=True)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("flip_donor", [False, True])
def test_mixup_bounds_workspace_without_changing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, flip_donor: bool
) -> None:
    """Force multiple blocks and compare against one full-tensor interpolation."""
    values = (torch.arange(9 * 2 * 3 * 64 * 65) % 251).reshape(9, 2, 3, 64, 65).to(dtype).transpose(-1, -2)
    original = values.clone()
    plan = MixUpPlan(torch.tensor([8, 0, 7, 1, 6, 2, 5, 3, 4]), torch.linspace(0, 1, len(values)))
    workspace = 2 * values[0].numel() * torch.float32.itemsize
    monkeypatch.setattr(mixing, "_MIXUP_WORKSPACE_BYTES", workspace)
    reference_float = values.float()
    donors = reference_float[plan.permutation]
    if flip_donor:
        donors = donors.flip((-1,))
    reference = torch.lerp(reference_float, donors, (1 - plan.lambdas).view(-1, 1, 1, 1, 1))
    expected = reference.round().clamp(0, 255).to(dtype) if dtype == torch.uint8 else reference.to(dtype)
    result, peak_bytes = measure_peak_tensor_bytes(
        lambda: apply_mixup(values, plan, flip_donor=flip_donor), tmp_path / f"{dtype}-{flip_donor}.json"
    )
    workspace_blocks = 1 if dtype is torch.float32 else 3
    assert peak_bytes <= result.numel() * result.element_size() + workspace_blocks * workspace + 4096
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(values, original)


def test_batch_operations_return_training_inputs_outside_autograd() -> None:
    """Check graph isolation and the model's ability to save the outputs for backward."""
    values = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2).requires_grad_()
    lambdas = torch.tensor([0.75, 0.25], requires_grad=True)
    permutation = torch.tensor([1, 0])
    normalizer = InputNormalizer("per_sample_nonzero", output_precision="fp32")

    with torch.autograd.forward_ad.dual_level():
        dual_values = torch.autograd.forward_ad.make_dual(values, torch.ones_like(values))
        dual_lambdas = torch.autograd.forward_ad.make_dual(lambdas, torch.ones_like(lambdas))
        plan = MixUpPlan(permutation, dual_lambdas)
        mixed = apply_mixup(dual_values, plan)
        flipped = apply_mixup(dual_values, plan, flip_donor=True)
        normalized = normalizer(dual_values)
        for output in (mixed, flipped, normalized):
            assert torch.autograd.forward_ad.unpack_dual(output).tangent is None
            assert not output.requires_grad
            assert not output.is_inference()

    model = torch.nn.Linear(4, 1, bias=False)
    batch = normalized.flatten(1) + mixed.flatten(1) + flipped.flatten(1)
    model(batch).sum().backward()
    torch.testing.assert_close(model.weight.grad, batch.sum(dim=0, keepdim=True))
    assert values.grad is None
    assert lambdas.grad is None


@pytest.mark.parametrize("flip_donor", [False, True])
def test_mixup_donor_flip_supports_an_empty_batch(flip_donor: bool) -> None:
    values = torch.empty((0, 2, 3, 7), dtype=torch.float16)
    plan = MixUpPlan(torch.empty(0, dtype=torch.int64), torch.empty(0))

    result = apply_mixup(values, plan, flip_donor=flip_donor)

    torch.testing.assert_close(result, values)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_mixup_interpolates_with_weights_in_other_float_dtypes(dtype: torch.dtype) -> None:
    values = torch.tensor([0.0, 4.0], dtype=torch.float32)
    lambdas = torch.tensor([0.75, 0.25], dtype=dtype)
    result = apply_mixup(values, MixUpPlan(torch.tensor([1, 0]), lambdas))

    torch.testing.assert_close(result, torch.tensor([1.0, 1.0], dtype=torch.float32))


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ([[2.0], [10.0]], [[4.0], [8.0]]),
        ([[0.9, 0.1], [0.2, 0.8]], [[0.725, 0.275], [0.375, 0.625]]),
    ],
)
def test_mixup_reuses_the_input_plan_for_float_targets(
    targets: list[list[float]],
    expected: list[list[float]],
) -> None:
    plan = MixUpPlan(permutation=torch.tensor([1, 0]), lambdas=torch.tensor([0.75, 0.75]))
    inputs = torch.tensor([[[[0.0]]], [[[4.0]]]], dtype=torch.float16)
    torch.testing.assert_close(apply_mixup(inputs, plan).flatten(), torch.tensor([1.0, 3.0], dtype=torch.float16))
    torch.testing.assert_close(apply_mixup(torch.tensor(targets), plan), torch.tensor(expected))


def test_mixup_can_use_the_input_plan_to_soften_class_labels() -> None:
    plan = MixUpPlan(permutation=torch.tensor([1, 0]), lambdas=torch.tensor([0.75, 0.75]))
    labels = torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=2).float()

    mixed_labels = apply_mixup(labels, plan)

    torch.testing.assert_close(mixed_labels, torch.tensor([[0.75, 0.25], [0.25, 0.75]]))


def test_mixup_beta_sampling_returns_finite_batch_lambdas() -> None:
    plan = sample_mixup(
        batch_size=8,
        alpha=1.0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(137),
    )

    assert torch.isfinite(plan.lambdas).all()
    assert torch.all((plan.lambdas >= 0) & (plan.lambdas <= 1))


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -float("inf")])
def test_mixup_rejects_nonfinite_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="finite positive alpha"):
        sample_mixup(
            batch_size=2,
            alpha=alpha,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(137),
        )


@pytest.mark.parametrize("alpha", [0.01, 1e-40])
def test_mixup_small_alpha_concentrates_weights_at_both_endpoints(alpha: float) -> None:
    plan = sample_mixup(
        batch_size=4096,
        alpha=alpha,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(137),
    )
    low_fraction = (plan.lambdas <= 0.01).float().mean()
    high_fraction = (plan.lambdas >= 0.99).float().mean()

    assert 0.45 < low_fraction < 0.55
    assert 0.45 < high_fraction < 0.55


def test_mixup_large_finite_alpha_rounds_to_the_midpoint() -> None:
    plan = sample_mixup(
        batch_size=2,
        alpha=1e300,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(137),
    )

    torch.testing.assert_close(plan.lambdas, torch.full((2,), 0.5))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_mixup_donor_flip_example(device: str) -> None:
    images = torch.tensor([[[0.0, 4.0, 8.0]], [[12.0, 20.0, 28.0]]], device=device)
    plan = MixUpPlan(torch.tensor([1, 0], device=device), torch.tensor([0.75, 0.5], device=device))
    assert apply_mixup(images, plan, flip_donor=True).tolist() == [[[7, 8, 9]], [[10, 12, 14]]]
    standard = [[[3, 8, 13]], [[6, 12, 18]]]
    assert apply_mixup(images, plan).tolist() == standard
    assert apply_mixup(images, plan, flip_donor=False).tolist() == standard
    assert apply_mixup(torch.eye(2, device=device), plan).tolist() == [[0.75, 0.25], [0.5, 0.5]]


@pytest.mark.parametrize("shape", [(2, 3, 5), (2, 4, 3, 5), (2, 4, 2, 3, 5), (2, 3, 1)])
def test_mixup_donor_flip_width_coordinates(shape: tuple[int, ...]) -> None:
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    plan = MixUpPlan(torch.tensor([1, 0]), torch.zeros(2))
    expected = values[[1, 0]][..., list(reversed(range(shape[-1])))]
    assert torch.equal(apply_mixup(values, plan, flip_donor=True), expected)


@pytest.mark.parametrize("shape", [(), (2,), (0, 2)])
def test_mixup_donor_flip_rejects_low_rank(shape: tuple[int, ...]) -> None:
    batch = shape[0] if shape else 0
    plan = MixUpPlan(torch.arange(batch), torch.ones(batch))
    with pytest.raises(ValueError):
        apply_mixup(torch.empty(shape), plan, flip_donor=True)


def test_mixup_donor_flip_manual_self_match() -> None:
    values = torch.tensor([[[0.0, 4.0, 8.0]]])
    plan = MixUpPlan(torch.tensor([0]), torch.tensor([0.75]))
    assert apply_mixup(values, plan, flip_donor=True).tolist() == [[[2, 4, 6]]]
    with pytest.raises(TypeError):
        apply_mixup(values, plan, True)


@pytest.mark.parametrize("mode", ["pair", "element", "batch"])
def test_mixup_donor_flip_sampled_plan(mode: mixing.MixingMode) -> None:
    generator = torch.Generator().manual_seed(137)
    values = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    plan = sample_mixup(batch_size=5, alpha=1.0, device=values.device, generator=generator, mode=mode)
    original = values.clone()
    donors = plan.permutation.clone()
    weights = plan.lambdas.clone()
    cpu_state = torch.get_rng_state()
    sampler_state = generator.get_state()
    flipped = apply_mixup(values, plan, flip_donor=True)
    standard = apply_mixup(values, plan)
    explicit = apply_mixup(values, plan, flip_donor=False)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(generator.get_state(), sampler_state)
    assert torch.equal(standard, explicit)
    for index, (donor, weight) in enumerate(zip(donors, weights, strict=True)):
        recipient = weight * original[index]
        donor_values = original[donor]
        torch.testing.assert_close(standard[index], recipient + (1 - weight) * donor_values)
        torch.testing.assert_close(flipped[index], recipient + (1 - weight) * donor_values[:, [2, 1, 0]])
    if mode == "pair":
        self_match = donors == torch.arange(5)
        assert self_match.sum() == 1
        assert torch.equal(flipped[self_match], original[self_match])
    flipped.zero_()
    assert torch.equal(values, original)
    assert torch.equal(plan.permutation, donors)
    assert torch.equal(plan.lambdas, weights)


def test_mixup_bounds_workspace_for_oversized_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    values = torch.arange(8 * 32 * 64, dtype=torch.float32).reshape(8, 32, 64)
    plan = MixUpPlan(torch.arange(7, -1, -1), torch.full((8,), 0.75))
    monkeypatch.setattr(mixing, "_MIXUP_WORKSPACE_BYTES", 4096)
    batch, height, width = torch.meshgrid(torch.arange(8), torch.arange(32), torch.arange(64), indexing="ij")
    expected = 0.75 * values[batch, height, width] + 0.25 * values[7 - batch, height, 63 - width]
    result, peak_bytes = measure_peak_tensor_bytes(
        lambda: apply_mixup(values, plan, flip_donor=True), tmp_path / "memory.json"
    )
    assert torch.equal(result, expected)
    assert peak_bytes <= result.nbytes + 2 * values[0].nbytes + 4096


def test_mixup_numeric_transport_uint8() -> None:
    # Half-integers round to even. Extrapolation must saturate both ends.
    values = torch.tensor([[[0, 2]], [[1, 1]], [[0, 255]], [[0, 255]]], dtype=torch.uint8)
    plan = MixUpPlan(torch.tensor([1, 0, 3, 2]), torch.tensor([0.5, 0.5, 2.0, 2.0], dtype=torch.float16))
    assert apply_mixup(values, plan, flip_donor=True).tolist() == [[[0, 2]], [[2, 0]], [[0, 255]], [[0, 255]]]


def test_mixup_numeric_transport_float64() -> None:
    values = torch.tensor([[[0.0, 16_777_217.0]]], dtype=torch.float64)
    plan = MixUpPlan(torch.tensor([0]), torch.tensor([0.5], dtype=torch.float64))
    result = apply_mixup(values, plan, flip_donor=True)
    assert result.dtype == torch.float64
    assert result.tolist() == [[[8_388_608.0, 8_388_608.0]]]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("recipient_weight", [1.0, 0.0, 0.5])
def test_mixup_donor_flip_large_opposite_endpoints(dtype: torch.dtype, recipient_weight: float) -> None:
    values = torch.tensor([[[2e38, -2e38]]], dtype=torch.float32).to(dtype)
    plan = sample_mixup(batch_size=1, alpha=1.0, device=values.device, generator=torch.Generator().manual_seed(137))
    if recipient_weight == 1:
        assert plan.lambdas.item() == 1
        expected = values
    else:
        plan = MixUpPlan(plan.permutation, torch.tensor([recipient_weight]))
        expected = values[..., [1, 0]] if recipient_weight == 0 else torch.zeros_like(values)
    assert torch.equal(apply_mixup(values, plan, flip_donor=True), expected)


@pytest.mark.parametrize("width", [16383, 16384])
def test_mixup_donor_flip_cpu_view_boundary(width: int) -> None:
    values = torch.arange(6 * width, dtype=torch.float32).reshape(3, 1, 2 * width)[..., ::2]
    original = values.clone()
    # Two recipients share the last donor. Both index spellings must resolve to the same row.
    plan = MixUpPlan(torch.tensor([-1, 2, 0], dtype=torch.int32), torch.tensor([0.0, 0.5, 1.0]))
    result = apply_mixup(values, plan, flip_donor=True)
    reversed_last = original[2, :, torch.arange(width - 1, -1, -1)]
    expected = torch.stack((reversed_last, 0.5 * (original[1] + reversed_last), original[2]))
    assert torch.equal(result, expected)
    assert torch.equal(values, original)
