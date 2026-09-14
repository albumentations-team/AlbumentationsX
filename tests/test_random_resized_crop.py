"""Focused regression tests for rejection-free 2D RandomResizedCrop sampling."""

import math
from unittest.mock import Mock

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.crops._sampling import (
    _integral_linear_exp,
    _sample_standard_2d_area,
    _standard_2d_segments,
    sample_2d_crop_shape,
)


@pytest.mark.parametrize("sampling_method", ("standard", "uniform_scale"))
def test_random_resized_crop_samples_a_feasible_shape_without_retries(sampling_method: str) -> None:
    random = Mock()
    random.random.side_effect = (0.2, 0.8)

    crop_shape = sample_2d_crop_shape(
        height=120,
        width=240,
        scale=(0.08, 1.0),
        ratio=(0.75, 4.0 / 3.0),
        sampling_method=sampling_method,
        py_random=random,
    )

    assert crop_shape is not None
    crop_height, crop_width = crop_shape
    assert 1 <= crop_height <= 120
    assert 1 <= crop_width <= 240
    assert random.random.call_count == 2


def test_random_resized_crop_keeps_a_degenerate_feasible_crop() -> None:
    assert sample_2d_crop_shape(
        height=80,
        width=120,
        scale=(1.0, 1.0),
        ratio=(1.5, 1.5),
        sampling_method="standard",
        py_random=Mock(random=Mock(return_value=0.5)),
    ) == (80, 120)


def test_standard_area_cdf_inversion_is_numerically_precise() -> None:
    segments = _standard_2d_segments(120, 240, 0.08, 1.0, 0.75, 4.0 / 3.0)
    total_mass = segments[-1][4] + segments[-1][5]

    for quantile in np.linspace(1e-9, 1.0 - 1e-9, 101):
        area = _sample_standard_2d_area(segments, quantile)
        log_area = math.log(area)
        cumulative = 0.0
        for left, right, slope, intercept, prior_mass, mass in segments:
            if log_area <= right:
                cumulative = (
                    prior_mass
                    + _integral_linear_exp(slope, intercept, log_area)
                    - _integral_linear_exp(
                        slope,
                        intercept,
                        left,
                    )
                )
                break
            cumulative = prior_mass + mass
        assert cumulative / total_mass == pytest.approx(quantile, abs=1e-10)


def test_random_resized_crop_reports_no_shape_only_for_an_empty_region() -> None:
    assert (
        sample_2d_crop_shape(
            height=80,
            width=120,
            scale=(0.0, 0.0),
            ratio=(0.75, 4.0 / 3.0),
            sampling_method="standard",
            py_random=Mock(),
        )
        is None
    )


@pytest.mark.parametrize(
    "ratio",
    ((0.0, 1.0), (float("inf"), 1.0), (float("nan"), 1.0)),
)
def test_random_resized_crop_rejects_nonpositive_or_nonfinite_ratios(ratio: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        A.RandomResizedCrop(size=(32, 32), ratio=ratio)


def test_random_resized_crop_uniform_scale_replays_exactly() -> None:
    image = np.arange(120 * 240, dtype=np.uint8).reshape(120, 240, 1)
    transform = A.ReplayCompose(
        [A.RandomResizedCrop(size=(32, 48), sampling_method="uniform_scale", p=1.0)],
        seed=137,
    )

    result = transform(image=image)
    replayed = A.ReplayCompose.replay(result["replay"], image=image)

    np.testing.assert_array_equal(replayed["image"], result["image"])


def test_random_resized_crop_applied_config_replaces_policy_with_realized_shape() -> None:
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    pipeline = A.Compose(
        [A.RandomResizedCrop(size=(32, 48), sampling_method="standard", p=1.0)],
        save_applied_params=True,
        seed=137,
        strict=True,
    )

    result = pipeline(image=image)
    _, applied_config = result["applied_transforms"][0]

    assert applied_config["sampling_method"] == "standard"
    assert applied_config["scale"][0] == applied_config["scale"][1]
    assert applied_config["ratio"][0] == applied_config["ratio"][1]
    assert math.isfinite(applied_config["scale"][0])
