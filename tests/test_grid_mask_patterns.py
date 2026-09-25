"""GridMask cell geometry and compatibility contracts."""

import hashlib
import json

import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (
            0,
            [
                "d99deda5f9cd18f80ab97b09b4e1943f7cd51b53d77de8d46b7e64f786beea3c",
                "36e1a564474527933e63a07a62bbee8f09a7e0535ff68b290c3d3b21d28db773",
                "c1aecec673e388fdec29006cd6a8221662f963239c76475df7d5ab0035c9c0c2",
            ],
        ),
        (
            0.25,
            [
                "f9df875e32d51515ac7981a785564f6503ebebf150aa7e36fe3ba7e2723fb02c",
                "538b845f69e9ab5f6434200628cdba2b2864b7fe5d55661304945c5bd49f984a",
                "4a6f6e0f42250df4b5e7858e272b966be966290e8183b6d4c4dabb3c405c659d",
            ],
        ),
    ],
)
def test_default_seeded_outputs_unchanged(rotation, expected):
    # Captured on main a58cedf before introducing cell patterns; three calls
    # also protect the RNG position carried forward to subsequent invocations.
    image = np.arange(24 * 30 * 3, dtype=np.uint8).reshape(24, 30, 3)
    transform = A.Compose([A.GridMask(rotation_range=(rotation, rotation), p=1)], seed=137)
    actual = [hashlib.sha256(transform(image=image)["image"].tobytes()).hexdigest() for _ in range(3)]
    assert actual == expected


@pytest.mark.parametrize("pattern", ["top_left", "top_left_inverse", "diagonal"])
@pytest.mark.parametrize("shape", [(8, 8), (8, 12)])
def test_cell_patterns_match_reference_quadrants(pattern, shape):
    # Both immutable competition references in issue #321 use these quadrants
    # when the top-left side ratio is one half and the grid offset is zero.
    cell = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
    if pattern == "top_left_inverse":
        cell = 1 - cell
    elif pattern == "diagonal":
        cell[2:, 2:] = 0
    if shape[1] == 12:
        cell = np.repeat(cell, [2, 1, 2, 1], axis=1)
    expected = np.tile(cell, (2, 2))
    image = np.ones((*shape, 1), dtype=np.uint8)
    transform = A.Compose(
        [A.GridMask(num_grid_range=(2, 2), pattern=pattern, shift_xy=(0, 0), p=1)],
        seed=137,
        strict=True,
    )
    result = transform(image=image)["image"]
    np.testing.assert_array_equal(result[..., 0], expected)
    np.testing.assert_array_equal(image, 1)


@pytest.mark.parametrize("pattern", ["top_left", "top_left_inverse", "diagonal"])
@pytest.mark.parametrize("rotation", [0, 0.25])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_cell_mask_alignment_and_replay(pattern, rotation, dtype, channels):
    image = np.ones((24, 30, channels), dtype=dtype)
    mask = np.full((24, 30), 9, dtype=np.uint8)
    transform = A.ReplayCompose(
        [A.GridMask(pattern=pattern, rotation_range=(rotation, rotation), fill_mask=0, p=1)],
    )
    result = transform(image=image, mask=mask)
    np.testing.assert_array_equal(result["image"][..., 0] == 0, result["mask"] == 0)
    replay = A.ReplayCompose.replay(result["replay"], image=image, mask=mask)
    np.testing.assert_array_equal(replay["image"], result["image"])
    np.testing.assert_array_equal(replay["mask"], result["mask"])
    np.testing.assert_array_equal(image, 1)
    np.testing.assert_array_equal(mask, 9)


@pytest.mark.parametrize("pattern", ["top_left", "top_left_inverse", "diagonal"])
def test_cell_applied_configuration_records_phase(pattern):
    image = np.ones((24, 30, 3), dtype=np.uint8)
    transform = A.Compose(
        [A.GridMask(pattern=pattern, cell_mask_ratio_range=(0.3, 0.7), rotation_range=(-0.3, 0.3), p=1)],
        seed=137,
        save_applied_params=True,
    )
    result = transform(image=image)
    records = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    replay = A.Compose.from_applied_transforms(records)(image=image)
    np.testing.assert_array_equal(replay["image"], result["image"])


@pytest.mark.parametrize("ratio", [(0, 0.5), (0.5, 1), (0.8, 0.2), (-0.1, 0.5)])
def test_invalid_cell_ratio(ratio):
    with pytest.raises(ValueError):
        A.GridMask(pattern="top_left", cell_mask_ratio_range=ratio)


def test_invalid_pattern():
    with pytest.raises(ValueError):
        A.GridMask(pattern="unknown")


@pytest.mark.parametrize("pattern", ["top_left", "top_left_inverse", "diagonal"])
def test_dfdc_continuous_keep_ratio(pattern):
    # DFDC's keep_ratio=0.64 gives sqrt(1-0.64)=0.6: a 3x3 region
    # in a 5x5 cell. Mode 2 retains its independent bottom-right quarter.
    cell = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    if pattern == "top_left_inverse":
        cell = 1 - cell
    elif pattern == "diagonal":
        cell[2:, 2:] = 0
    transform = A.Compose(
        [
            A.GridMask(num_grid_range=(2, 2), pattern=pattern, cell_mask_ratio_range=(0.6, 0.6), shift_xy=(0, 0), p=1),
        ],
        seed=137,
        strict=True,
    )
    result = transform(image=np.ones((10, 10, 1), dtype=np.uint8))["image"]
    np.testing.assert_array_equal(result[..., 0], np.tile(cell, (2, 2)))


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (7, 11), (19, 4)])
@pytest.mark.parametrize("num_grid", [2, 3, 7])
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("pattern", ["top_left", "top_left_inverse", "diagonal"])
def test_fractional_cell_boundaries(shape, num_grid, ratio, pattern):
    height, width = shape
    cell_height, cell_width = height / num_grid, width / num_grid
    shift_x, shift_y = 1 % max(1, width // num_grid), 1 % max(1, height // num_grid)

    # Independent scalar oracle: test membership in half-open mathematical
    # cell intervals. This includes truncated and subpixel cells.
    def contains(coordinate, cell_size, start, end):
        return any(
            int(index * cell_size + start * cell_size) <= coordinate < int(index * cell_size + end * cell_size)
            for index in range(num_grid + 1)
        )

    expected = np.ones(shape, dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            top_left = contains(row + shift_y, cell_height, 0, ratio) and contains(
                col + shift_x,
                cell_width,
                0,
                ratio,
            )
            bottom_right = contains(row + shift_y, cell_height, 0.5, 1) and contains(
                col + shift_x,
                cell_width,
                0.5,
                1,
            )
            drop = not top_left if pattern == "top_left_inverse" else top_left
            if pattern == "diagonal":
                drop |= bottom_right
            expected[row, col] = not drop
    image = np.ones((*shape, 1), dtype=np.uint8)
    transform = A.Compose(
        [
            A.GridMask(
                num_grid_range=(num_grid, num_grid),
                pattern=pattern,
                cell_mask_ratio_range=(ratio, ratio),
                shift_xy=(1, 1),
                p=1,
            ),
        ],
        seed=137,
        strict=True,
    )
    np.testing.assert_array_equal(transform(image=image)["image"][..., 0], expected)
