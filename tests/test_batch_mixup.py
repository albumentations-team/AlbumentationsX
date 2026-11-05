"""Behavior and memory contracts for batch MixUp."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import albumentations.batch.mixup as mixing
from albumentations.batch import MixUpPlan, apply_mixup, sample_mixup
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


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32])
def test_mixup_bounds_workspace_without_changing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    """Force multiple blocks and compare against one full-tensor interpolation."""
    values = (torch.arange(9 * 2 * 3 * 64 * 65) % 251).reshape(9, 2, 3, 64, 65).to(dtype).transpose(-1, -2)
    original = values.clone()
    plan = MixUpPlan(torch.tensor([8, 0, 7, 1, 6, 2, 5, 3, 4]), torch.linspace(0, 1, len(values)))
    workspace = 2 * values[0].numel() * torch.float32.itemsize
    monkeypatch.setattr(mixing, "_MIXUP_WORKSPACE_BYTES", workspace)
    reference_float = values.float()
    reference = torch.lerp(reference_float, reference_float[plan.permutation], (1 - plan.lambdas).view(-1, 1, 1, 1, 1))
    expected = reference.round().clamp(0, 255).to(dtype) if dtype == torch.uint8 else reference.to(dtype)
    result, peak_bytes = measure_peak_tensor_bytes(lambda: apply_mixup(values, plan), tmp_path / "memory.json")
    assert peak_bytes <= result.numel() * result.element_size() + 3 * workspace + 4096
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(values, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mixup_preserves_input_and_lambda_gradients_across_blocks(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    values = (torch.arange(5 * 2 * 3 * 7).reshape(5, 2, 3, 7) / 13).to(dtype).requires_grad_()
    reference = values.detach().clone().requires_grad_()
    lambdas = torch.linspace(0, 1, 5).requires_grad_()
    reference_lambdas = lambdas.detach().clone().requires_grad_()
    permutation = torch.tensor([3, 3, 1, 0, 4])
    monkeypatch.setattr(mixing, "_MIXUP_WORKSPACE_BYTES", values[0].numel() * 4)
    reference_float = reference.float()
    expected = torch.lerp(reference_float, reference_float[permutation], (1 - reference_lambdas).view(-1, 1, 1, 1))
    expected = expected.to(dtype)
    result = apply_mixup(values, MixUpPlan(permutation, lambdas))
    weights = torch.linspace(-1, 2, values.numel()).reshape_as(values)
    (expected * weights).sum().backward()
    (result * weights).sum().backward()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(values.grad, reference.grad)
    torch.testing.assert_close(lambdas.grad, reference_lambdas.grad)


def test_mixup_preserves_empty_input_gradients() -> None:
    values = torch.empty((0, 2, 3, 7), requires_grad=True)
    plan = MixUpPlan(torch.empty(0, dtype=torch.int64), torch.empty(0))
    result = apply_mixup(values, plan)
    result.sum().backward()
    assert result.shape == values.shape
    torch.testing.assert_close(values.grad, torch.empty_like(values))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_mixup_converts_weights_to_fp32_and_preserves_their_gradients(dtype: torch.dtype) -> None:
    values = torch.tensor([0.0, 4.0], dtype=torch.float32)
    lambdas = torch.tensor([0.75, 0.25], dtype=dtype, requires_grad=True)
    result = apply_mixup(values, MixUpPlan(torch.tensor([1, 0]), lambdas))
    result.sum().backward()

    torch.testing.assert_close(result, torch.tensor([1.0, 1.0], dtype=torch.float32))
    torch.testing.assert_close(lambdas.grad, torch.tensor([-4.0, 4.0], dtype=dtype))


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
