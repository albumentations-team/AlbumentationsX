import json
from typing import Any

import numpy as np
import pytest

import albumentations as A
from albumentations.core.transform_params import SampledParams


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    ("num_channels", "channels_to_drop", "dropout_ratio", "expected_count"),
    [
        (3, None, 0, 0),
        (3, None, 0.1, 0),
        (3, None, 0.5, 2),
        (5, None, 0.5, 3),
        (3, None, 1, 2),
        (1, None, 0, 0),
        (1, None, 1, 0),
        (1, (0,), 0.5, 0),
        (10, (6, 0, 4, 2), 0.5, 2),
        (10, (0, 2, 4, 6), 1, 4),
        (10, (0, 2, 4, 6), 0.124, 0),
        (10, (0, 2, 4, 6), 0.125, 1),
        (10, (0, 2, 4, 6), 0.625, 3),
        (4, (0, 1, 2, 3), 1, 3),
        (3, (2,), 1, 1),
    ],
)
def test_channel_dropout_fraction_and_protected_channels(
    dtype: type[np.generic],
    num_channels: int,
    channels_to_drop: tuple[int, ...] | None,
    dropout_ratio: float,
    expected_count: int,
) -> None:
    source_value, fill = (137, 7) if dtype == np.uint8 else (0.75, 0.125)
    image = np.full((4, 7, num_channels), source_value, dtype=dtype)
    image.setflags(write=False)
    original = image.copy()
    transform = A.Compose(
        [A.ChannelDropout(channels_to_drop=channels_to_drop, dropout_ratio=dropout_ratio, fill=fill, p=1)],
        seed=137,
        strict=True,
    )

    result = transform(image=image)["image"]

    dropped = np.flatnonzero(np.all(result == fill, axis=(0, 1)))
    eligible = set(range(num_channels)) if channels_to_drop is None else set(channels_to_drop)
    assert len(dropped) == expected_count
    assert set(dropped) <= eligible
    assert len(dropped) < num_channels
    expected = original.copy()
    expected[..., dropped] = fill
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
    assert result.dtype == image.dtype


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels_to_drop": ()}, "at least 1"),
        ({"channels_to_drop": (0, 0)}, "duplicate"),
        ({"channels_to_drop": (-1,)}, "greater than or equal to 0"),
        ({"channels_to_drop": (0.5,)}, "valid integer"),
        ({"channels_to_drop": (True,)}, "valid integer"),
        ({"dropout_ratio": -0.1}, "greater than or equal to 0"),
        ({"dropout_ratio": 1.1}, "less than or equal to 1"),
        ({"dropout_ratio": float("nan")}, "less than or equal to 1"),
        ({"dropout_ratio": float("inf")}, "less than or equal to 1"),
    ],
)
def test_channel_dropout_rejects_invalid_configuration(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        A.ChannelDropout(**kwargs, strict=True)


@pytest.mark.parametrize("dropout_ratio", [0, 0.5, 1])
def test_channel_dropout_validates_indices_for_every_target(dropout_ratio: float) -> None:
    transform = A.Compose(
        [A.ChannelDropout(channels_to_drop=(4, 0), dropout_ratio=dropout_ratio, p=1)],
        strict=True,
    )

    with pytest.raises(ValueError, match="input's 3 channels"):
        transform(
            image=np.ones((4, 7, 5), dtype=np.float32),
            volume=np.ones((2, 4, 7, 3), dtype=np.float32),
        )


@pytest.mark.parametrize("channels_to_drop", [None, (4, 0, 2)])
def test_channel_dropout_shares_selection_across_equal_channel_targets(
    channels_to_drop: tuple[int, ...] | None,
) -> None:
    image = np.ones((4, 7, 5), dtype=np.float32)
    data = {
        "image": image,
        "image2": image.copy(),
        "images": np.stack([image, image]),
        "volume": np.stack([image, image, image]),
        "volumes": np.ones((2, 3, 4, 7, 5), dtype=np.float32),
    }
    transform = A.Compose(
        [A.ChannelDropout(channels_to_drop=channels_to_drop, dropout_ratio=0.5, p=1)],
        additional_targets={"image2": "image"},
        seed=137,
        strict=True,
    )

    result = transform(**data)

    dropped = np.flatnonzero(np.all(result["image"] == 0, axis=(0, 1)))
    assert len(dropped) == (3 if channels_to_drop is None else 2)
    for name, source in data.items():
        expected = source.copy()
        expected[..., dropped] = 0
        np.testing.assert_array_equal(result[name], expected)
        np.testing.assert_array_equal(source, 1)


def test_channel_dropout_rescales_count_for_different_channel_targets_and_replays() -> None:
    data = {
        "image": np.ones((4, 7, 3), dtype=np.float32),
        "volume": np.ones((2, 4, 7, 10), dtype=np.float32),
    }
    dropout = A.ChannelDropout(dropout_ratio=0.5, p=1)
    transform = A.ReplayCompose([dropout])
    transform.set_random_seed(137)

    result = transform(**data)
    sampled = SampledParams.from_dict(dropout.get_applied_params())
    replayed = A.ReplayCompose.replay(json.loads(json.dumps(result["replay"])), **data)

    assert len(sampled.params_for("image")["selected_channels"]) == 2
    assert len(sampled.params_for("volume")["selected_channels"]) == 5
    for name, source in data.items():
        expected = source.copy()
        expected[..., sampled.params_for(name)["selected_channels"]] = 0
        np.testing.assert_array_equal(result[name], expected)
        np.testing.assert_array_equal(replayed[name], expected)


@pytest.mark.parametrize("dropout_ratio", [0, 0.5, 1])
@pytest.mark.parametrize(
    ("target", "shape"),
    [("image", (4, 7)), ("images", (2, 4, 7)), ("volume", (3, 4, 7)), ("volumes", (2, 3, 4, 7))],
)
def test_channel_dropout_preserves_channel_free_grayscale(
    dropout_ratio: float,
    target: str,
    shape: tuple[int, ...],
) -> None:
    image = np.full(shape, 137, dtype=np.uint8)
    transform = A.Compose([A.ChannelDropout(dropout_ratio=dropout_ratio, p=1)], strict=True)

    result = transform(**{target: image})[target]

    assert result.shape == image.shape
    np.testing.assert_array_equal(result, image)


def test_channel_dropout_probability_skips_the_entire_operation() -> None:
    image = np.full((4, 7, 5), 137, dtype=np.uint8)
    transform = A.Compose([A.ChannelDropout(dropout_ratio=1, p=0)], strict=True)

    np.testing.assert_array_equal(transform(image=image)["image"], image)


def test_channel_dropout_seed_reproduces_selection_from_eligible_channels() -> None:
    image = np.ones((2, 3, 8), dtype=np.float32)
    transform = A.Compose(
        [A.ChannelDropout(channels_to_drop=(0, 2, 4, 6), dropout_ratio=0.5, p=1)],
        seed=137,
        strict=True,
    )

    first = [transform(image=image)["image"] for _ in range(16)]
    transform.set_random_seed(137)
    repeated = [transform(image=image)["image"] for _ in range(16)]

    selections = set()
    for original, replayed in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(original, replayed)
        selections.add(tuple(np.flatnonzero(np.all(original == 0, axis=(0, 1)))))
    assert len(selections) > 1
    assert set().union(*map(set, selections)) == {0, 2, 4, 6}
