"""Discrete-choice contracts for CropAndPad."""

import json
import random
from typing import Any

import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"px": 1, "percent": 0.1},
        {"px": 1, "px_choices": (1, 2)},
        {"px": 1, "percent_choices": (0.1, 0.2)},
        {"percent": 0.1, "px_choices": (1, 2)},
        {"percent": 0.1, "percent_choices": (0.1, 0.2)},
        {"px_choices": (1, 2), "percent_choices": (0.1, 0.2)},
    ],
)
def test_crop_and_pad_requires_exactly_one_amount_source(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        A.CropAndPad(**kwargs)


@pytest.mark.parametrize("kwargs", [{"px_choices": ()}, {"percent_choices": ()}])
def test_crop_and_pad_rejects_empty_choices(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        A.CropAndPad(**kwargs)


def test_percent_choices_accept_the_same_numeric_domain_as_percent() -> None:
    fixed = A.CropAndPad(percent=2.0)
    discrete = A.CropAndPad(percent_choices=(2.0,))

    assert fixed.percent == 2.0
    assert discrete.percent_choices == (2.0,)


@pytest.mark.parametrize(
    ("choice_parameter", "fixed_parameter", "choices"),
    [
        ("px_choices", "px", (-13, 4, 17)),
        ("percent_choices", "percent", (-0.25, 0.1, 0.35)),
    ],
)
@pytest.mark.parametrize("sample_independently", [False, True])
def test_crop_and_pad_choice_sampling_is_seeded_and_discrete(
    choice_parameter: str,
    fixed_parameter: str,
    choices: tuple[int, ...] | tuple[float, ...],
    sample_independently: bool,
) -> None:
    seed = 137
    choice_kwargs: dict[str, Any] = {choice_parameter: choices}
    transform = A.CropAndPad(
        **choice_kwargs,
        sample_independently=sample_independently,
        keep_size=False,
        p=1,
    )
    transform.set_random_seed(seed)

    transform(image=np.zeros((41, 67, 3), dtype=np.uint8))

    rng = random.Random(seed)
    expected = tuple(rng.choice(choices) for _ in range(4)) if sample_independently else (rng.choice(choices),) * 4
    applied_config = transform.get_applied_config()
    assert applied_config[fixed_parameter] == expected
    assert applied_config[choice_parameter] is None


def test_crop_and_pad_percent_choices_use_height_for_vertical_sides_and_width_for_horizontal_sides() -> None:
    transform = A.CropAndPad(percent_choices=(0.1,), keep_size=False, p=1)

    result = transform(image=np.zeros((40, 90, 3), dtype=np.uint8))

    assert result["image"].shape == (48, 108, 3)


@pytest.mark.parametrize(
    "amount_kwargs",
    [
        pytest.param({"px": -1_000}, id="fixed"),
        pytest.param({"px_choices": (-1_000,)}, id="choice"),
    ],
)
def test_crop_and_pad_cropping_preserves_at_least_one_pixel_per_axis(amount_kwargs: dict[str, Any]) -> None:
    transform = A.CropAndPad(**amount_kwargs, keep_size=False, p=1)

    result = transform(image=np.zeros((7, 11, 3), dtype=np.uint8))

    assert result["image"].shape == (1, 1, 3)


def test_crop_and_pad_choice_respects_keep_size() -> None:
    image = np.zeros((37, 61, 3), dtype=np.uint8)

    result = A.CropAndPad(px_choices=(9,), keep_size=True, p=1)(image=image)

    assert result["image"].shape == image.shape


@pytest.mark.parametrize(
    ("choice_parameter", "fixed_parameter", "choices"),
    [
        ("px_choices", "px", (-13, 4, 17)),
        ("percent_choices", "percent", (-0.25, 0.1, 0.35)),
    ],
)
def test_crop_and_pad_choice_applied_config_reconstructs_the_sampled_transform(
    choice_parameter: str,
    fixed_parameter: str,
    choices: tuple[int, ...] | tuple[float, ...],
) -> None:
    image = np.arange(41 * 67 * 3, dtype=np.uint8).reshape(41, 67, 3)
    choice_kwargs: dict[str, Any] = {choice_parameter: choices}
    pipeline = A.Compose(
        [A.CropAndPad(**choice_kwargs, sample_independently=True, keep_size=False, p=1)],
        save_applied_params=True,
        seed=137,
    )

    result = pipeline(image=image)
    transported_record = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    recorded_config = transported_record[0][1]
    replayed = A.Compose.from_applied_transforms(transported_record)(image=image)

    assert recorded_config[choice_parameter] is None
    assert len(recorded_config[fixed_parameter]) == 4
    np.testing.assert_array_equal(replayed["image"], result["image"])
