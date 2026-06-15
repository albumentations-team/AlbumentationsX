"""Determinism invariants for Compose, ReplayCompose, and serialization."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import albumentations as A
from tests.property.strategies import compose_seeds, image_arrays

pytestmark = pytest.mark.property

TRANSFORM_NAMES = ("HorizontalFlip", "VerticalFlip", "RandomRotate90")


def _transform_from_name(name: str) -> A.BasicTransform:
    transform_cls = getattr(A, name)
    return transform_cls(p=1.0)


@given(image=image_arrays(), seed=compose_seeds(), transform_name=st.sampled_from(TRANSFORM_NAMES))
def test_same_seed_and_input_produce_same_output(image: np.ndarray, seed: int, transform_name: str) -> None:
    first = A.Compose([_transform_from_name(transform_name)], seed=seed, strict=True)
    second = A.Compose([_transform_from_name(transform_name)], seed=seed, strict=True)

    np.testing.assert_array_equal(first(image=image)["image"], second(image=image)["image"])


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
