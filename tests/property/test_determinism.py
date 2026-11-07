"""Determinism invariants for Compose, ReplayCompose, and serialization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from hypothesis import given

import albumentations as A
from tests.helpers.contract_assertions import assert_contract_values_equal
from tests.property.strategies import compose_seeds, image_arrays

pytestmark = pytest.mark.property

COMPOSITION_NAMES = (
    "Compose",
    "ReplayCompose",
    "OneOf",
    "SomeOf",
    "RandomOrder",
    "OneOrOther",
    "Sequential",
    "SelectiveChannelTransform",
)


def _nested_pipeline(composition_name: str, seed: int) -> A.Compose:
    children = [
        A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.7),
        A.RandomBrightnessContrast(p=0.6),
    ]
    composition_type = getattr(A, composition_name)
    if composition_name in {"SomeOf", "RandomOrder"}:
        node = composition_type(children, n=2, replace=True, p=0.8)
    elif composition_name == "OneOrOther":
        node = composition_type(*children, p=0.8)
    elif composition_name == "SelectiveChannelTransform":
        node = composition_type(children, channels=(0,), p=0.8)
    else:
        node = composition_type(children, p=0.8)
    return A.Compose([A.Sequential([node], p=0.9)], seed=seed, strict=True, save_applied_params=True)


@pytest.mark.parametrize("composition_name", COMPOSITION_NAMES)
@given(image=image_arrays(), seed=compose_seeds())
def test_nested_compositions_preserve_fixed_seed_sequences_and_trace(
    composition_name: str, image: np.ndarray, seed: int
) -> None:
    first = _nested_pipeline(composition_name, seed)
    second = _nested_pipeline(composition_name, seed)
    for _ in range(3):
        expected = first(image=image)
        actual = second.run_with_trace(image=image).data
        assert_contract_values_equal(actual, expected)


@pytest.mark.parametrize("composition_name", COMPOSITION_NAMES)
@given(image=image_arrays(), seed=compose_seeds())
def test_nested_explicit_seeds_survive_reordering_and_concurrent_calls(
    composition_name: str, image: np.ndarray, seed: int
) -> None:
    pipeline = _nested_pipeline(composition_name, seed)
    seeds = (137, 138, 139)
    expected = {sample_seed: pipeline(image=image, invocation_seed=sample_seed) for sample_seed in seeds}
    pipeline(image=image)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(
            executor.map(lambda sample_seed: pipeline(image=image, invocation_seed=sample_seed), reversed(seeds))
        )
    for sample_seed, result in zip(reversed(seeds), actual, strict=True):
        assert_contract_values_equal(result, expected[sample_seed])


@given(image=image_arrays(channels=(1, 3)), seed=compose_seeds())
def test_replay_compose_reproduces_output(image: np.ndarray, seed: int) -> None:
    transform = A.ReplayCompose([A.HorizontalFlip(p=1.0), A.RandomRotate90(p=1.0)], seed=seed)

    first = transform(image=image)
    replayed = A.ReplayCompose.replay(first["replay"], image=image)

    np.testing.assert_array_equal(first["image"], replayed["image"])


@given(image=image_arrays(channels=(1, 3)), seed=compose_seeds())
def test_serialized_compose_preserves_seeded_behavior(image: np.ndarray, seed: int) -> None:
    transform = A.Compose([A.HorizontalFlip(p=1.0), A.RandomRotate90(p=1.0)], seed=seed, strict=True)
    restored = A.from_dict(A.to_dict(transform))

    np.testing.assert_array_equal(transform(image=image)["image"], restored(image=image)["image"])
