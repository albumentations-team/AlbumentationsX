"""Issues #68/#533: paired image and semantic mask aliases retain their own pixels."""

from copy import deepcopy
from typing import Any, Literal

import cv2
import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

import albumentations as A


def _paired_item(
    donor_id: int,
    dtype: type[np.uint8] | type[np.float32],
    thermal_layout: Literal["hw", "hwc1", "hwc3"],
) -> dict[str, np.ndarray]:
    coordinates = np.arange(35, dtype=np.uint8).reshape(5, 7)
    image = np.stack(
        (
            np.full_like(coordinates, donor_id + 1),
            coordinates + 40 * donor_id,
            coordinates + 40 * donor_id + 20,
        ),
        axis=-1,
    )
    thermal = coordinates + 180 - 30 * donor_id
    if thermal_layout == "hwc1":
        thermal = thermal[..., np.newaxis]
    elif thermal_layout == "hwc3":
        thermal = np.stack((thermal, thermal - 20, thermal - 40), axis=-1)
    item = {"image": image.astype(dtype), "thermal": thermal.astype(dtype)}
    if dtype is np.float32:
        item = {name: value / np.float32(256) for name, value in item.items()}
    return item


def _pipeline(
    grid_yx: tuple[int, int],
    fit_mode: Literal["cover", "contain"],
    additional_targets: dict[str, str] | None = None,
) -> A.Compose:
    return A.Compose(
        [
            A.Mosaic(
                grid_yx=grid_yx,
                cell_shape=(5, 7),
                target_size=(5 * grid_yx[0], 7 * grid_yx[1]),
                center_range=(0.5, 0.5),
                fit_mode=fit_mode,
                interpolation=cv2.INTER_NEAREST,
                p=1,
            ),
        ],
        additional_targets=additional_targets,
        seed=137,
        strict=True,
    )


def _assert_unchanged(items: list[dict[str, np.ndarray]], originals: list[dict[str, np.ndarray]]) -> None:
    for item, original in zip(items, originals, strict=True):
        assert item.keys() == original.keys()
        for name in original:
            np.testing.assert_array_equal(item[name], original[name])


def _rgb_placements(image: np.ndarray, items: list[dict[str, np.ndarray]]) -> dict[tuple[int, int], int]:
    """Match complete expected tiles, including their spatial pattern, without production helpers."""
    assert image.shape == (10, 14, 3)
    assert image.dtype == items[0]["image"].dtype
    placements = {}
    for row in range(2):
        for column in range(2):
            tile = image[5 * row : 5 * (row + 1), 7 * column : 7 * (column + 1)]
            matches = [index for index, item in enumerate(items) if np.array_equal(tile, item["image"])]
            assert len(matches) == 1, "Each output tile must preserve one complete donor image"
            placements[row, column] = matches[0]
    assert sorted(placements.values()) == [0, 1, 2, 3]
    return placements


def _semantic_item(donor_id: int) -> dict[str, np.ndarray]:
    pattern = np.arange(35).reshape(5, 7)
    return {
        "image": _paired_item(donor_id, np.uint8, "hw")["image"],
        "mask": np.full((5, 7), donor_id + 1, dtype=np.uint8),
        "auxmask": (pattern + 1000 + 40 * donor_id).astype(np.uint16),
        "planes": np.stack((pattern - 300 - donor_id, pattern + 400 + donor_id), axis=-1).astype(np.int16),
    }


@pytest.mark.parametrize("direct", [False, True])
def test_mosaic_semantic_alias_identity(direct: bool) -> None:
    item = _semantic_item(0)
    pipeline = _pipeline((1, 1), "contain", {"auxmask": "mask", "planes": "mask"})
    transform = pipeline.transforms[0] if direct else pipeline
    result = transform(**item, mosaic_metadata=[])

    for name, value in item.items():
        np.testing.assert_array_equal(result[name], value, strict=True)
        assert not np.shares_memory(result[name], value)


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
@pytest.mark.parametrize("donor_count", [0, 1, 3, 5])
def test_mosaic_semantic_alias_paired_donors(fit_mode: Literal["cover", "contain"], donor_count: int) -> None:
    items = [_semantic_item(index) for index in range(donor_count + 1)]
    originals = deepcopy(items)
    result = _pipeline((2, 2), fit_mode, {"auxmask": "mask", "planes": "mask"})(**items[0], mosaic_metadata=items[1:])

    selected = []
    for row in range(2):
        for column in range(2):
            region = np.s_[row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7]
            donor_id = int(result["image"][region][0, 0, 0]) - 1
            selected.append(donor_id)
            for name in items[0]:
                np.testing.assert_array_equal(result[name][region], originals[donor_id][name], strict=True)
    assert selected.count(0) == max(1, 4 - donor_count)
    assert len(set(selected) - {0}) == min(3, donor_count)
    _assert_unchanged(items, originals)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("thermal_layout", ["hw", "hwc1", "hwc3"])
def test_mosaic_identity_preserves_additional_image(
    dtype: type[np.uint8] | type[np.float32],
    thermal_layout: Literal["hw", "hwc1", "hwc3"],
) -> None:
    item = _paired_item(0, dtype, thermal_layout)
    original = {name: value.copy() for name, value in item.items()}
    result = _pipeline((1, 1), "contain", {"thermal": "image"})(**item, mosaic_metadata=[])

    _assert_unchanged([item], [original])
    np.testing.assert_array_equal(result["image"], original["image"])
    assert result["thermal"].dtype == original["thermal"].dtype
    copied_rgb = np.array_equal(result["thermal"], result["image"])
    assert result["thermal"].shape == original["thermal"].shape, f"thermal copied RGB output: {copied_rgb}"
    np.testing.assert_array_equal(
        result["thermal"],
        original["thermal"],
        err_msg=f"The alias must preserve its own pixels; thermal copied RGB output: {copied_rgb}",
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("thermal_layout", ["hw", "hwc1", "hwc3"])
@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_additional_image_uses_paired_donors(
    dtype: type[np.uint8] | type[np.float32],
    thermal_layout: Literal["hw", "hwc1", "hwc3"],
    fit_mode: Literal["cover", "contain"],
) -> None:
    items = [_paired_item(index, dtype, thermal_layout) for index in range(4)]
    originals = [{name: value.copy() for name, value in item.items()} for item in items]
    result = _pipeline((2, 2), fit_mode, {"thermal": "image"})(**items[0], mosaic_metadata=items[1:])

    _assert_unchanged(items, originals)
    placements = _rgb_placements(result["image"], originals)
    assert result["thermal"].dtype == originals[0]["thermal"].dtype
    copied_rgb = np.array_equal(result["thermal"], result["image"])
    expected_shape = (10, 14, *originals[0]["thermal"].shape[2:])
    assert result["thermal"].shape == expected_shape, f"thermal copied RGB output: {copied_rgb}"
    for (row, column), donor_id in placements.items():
        thermal_tile = result["thermal"][5 * row : 5 * (row + 1), 7 * column : 7 * (column + 1)]
        np.testing.assert_array_equal(
            thermal_tile,
            originals[donor_id]["thermal"],
            err_msg=f"Thermal must use paired donor pixels; thermal copied RGB output: {copied_rgb}",
        )


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_canonical_rgb_preserves_donor_tiles(
    dtype: type[np.uint8] | type[np.float32],
    fit_mode: Literal["cover", "contain"],
) -> None:
    items = [{"image": _paired_item(index, dtype, "hw")["image"]} for index in range(4)]
    originals = [{"image": item["image"].copy()} for item in items]
    result = _pipeline((2, 2), fit_mode)(**items[0], mosaic_metadata=items[1:])

    _assert_unchanged(items, originals)
    _rgb_placements(result["image"], originals)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_packed_rgb_thermal_preserves_paired_tiles(
    dtype: type[np.uint8] | type[np.float32],
    fit_mode: Literal["cover", "contain"],
) -> None:
    items = [_paired_item(index, dtype, "hw") for index in range(4)]
    packed = [{"image": np.dstack((item["image"], item["thermal"]))} for item in items]
    originals = [{"image": item["image"].copy()} for item in packed]
    result = _pipeline((2, 2), fit_mode)(**packed[0], mosaic_metadata=packed[1:])

    _assert_unchanged(packed, originals)
    assert result["image"].shape == (10, 14, 4)
    placements = _rgb_placements(result["image"][..., :3], items)
    for (row, column), donor_id in placements.items():
        thermal_tile = result["image"][5 * row : 5 * (row + 1), 7 * column : 7 * (column + 1), 3]
        np.testing.assert_array_equal(thermal_tile, items[donor_id]["thermal"])


@pytest.mark.parametrize("donor_count", [0, 1, 3, 5])
def test_mosaic_multiple_aliases_share_donor_selection(donor_count: int) -> None:
    items = [_paired_item(index, np.uint8, "hw") for index in range(donor_count + 1)]
    for item in items:
        item["infrared"] = item["thermal"].astype(np.float32)[..., None] / 256
    result = _pipeline((2, 2), "cover", {"thermal": "image", "infrared": "image"})(
        **items[0], mosaic_metadata=items[1:]
    )
    selected = []
    for row in range(2):
        for column in range(2):
            region = np.s_[row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7]
            donor_id = int(result["image"][region][0, 0, 0]) - 1
            selected.append(donor_id)
            for name in ("image", "thermal", "infrared"):
                assert result[name].dtype == items[0][name].dtype
                np.testing.assert_array_equal(result[name][region], items[donor_id][name])
    assert selected.count(0) == max(1, 4 - donor_count)
    assert len(set(selected) - {0}) == min(3, donor_count)


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_alias_uses_resize_crop_and_padding(fit_mode: Literal["cover", "contain"]) -> None:
    pattern = np.array([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=np.uint8)
    thermal = (255 - pattern).astype(np.float32) / 256
    transform = A.Compose(
        [
            A.Mosaic(
                grid_yx=(1, 1),
                cell_shape=(4, 4),
                target_size=(4, 4),
                fit_mode=fit_mode,
                interpolation=cv2.INTER_NEAREST,
                p=1,
            )
        ],
        additional_targets={"thermal": "image"},
        seed=137,
        strict=True,
    )
    result = transform(image=np.repeat(pattern[..., None], 3, axis=-1), thermal=thermal, mosaic_metadata=[])
    if fit_mode == "cover":
        expected = np.repeat(np.repeat(thermal, 2, axis=0), 2, axis=1)[:, :4]
    else:
        expected = np.zeros((4, 4), dtype=np.float32)
        expected[1:3] = thermal
    np.testing.assert_array_equal(result["thermal"], expected)
    assert result["image"].dtype == np.uint8


def test_mosaic_alias_linear_resize_matches_known_bilinear_values() -> None:
    thermal = np.array([[0, 1], [1, 0]], dtype=np.float32)
    expected = np.array(
        [[0, 0.25, 0.75, 1], [0.25, 0.375, 0.625, 0.75], [0.75, 0.625, 0.375, 0.25], [1, 0.75, 0.25, 0]],
        dtype=np.float32,
    )
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(4, 4), target_size=(4, 4), p=1)],
        additional_targets={"thermal": "image"},
        seed=137,
    )
    result = transform(image=np.zeros((2, 2, 3), dtype=np.uint8), thermal=thermal, mosaic_metadata=[])
    np.testing.assert_allclose(result["thermal"], expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
@pytest.mark.parametrize("channels", [4, 5])
def test_mosaic_alias_variable_donor_sizes_and_multichannel_padding(
    fit_mode: Literal["cover", "contain"],
    channels: int,
) -> None:
    items = []
    for index, shape in enumerate(((5, 7), (10, 14), (3, 7), (5, 14))):
        pattern = (np.arange(np.prod(shape)).reshape(shape) % 20 + 10 + index * 20).astype(np.uint8)
        items.append(
            {
                "image": np.repeat(pattern[..., None], 3, axis=-1),
                "thermal": np.repeat((pattern + 128)[..., None], channels, axis=-1),
            }
        )
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(7, 9), fit_mode=fit_mode, interpolation=cv2.INTER_NEAREST, p=1)],
        additional_targets={"thermal": "image"},
        seed=137,
        strict=True,
    )
    result = transform(**items[0], mosaic_metadata=items[1:])
    rgb = result["image"][..., 0]
    expected = np.where(rgb > 0, rgb + 128, 0).astype(np.uint8)
    np.testing.assert_array_equal(result["thermal"], np.repeat(expected[..., None], channels, axis=-1))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_alias_resize_supports_128_channels(
    dtype: type[np.uint8] | type[np.float32],
    fit_mode: Literal["cover", "contain"],
) -> None:
    channels = np.arange(128, dtype=dtype)
    if dtype is np.float32:
        channels /= np.float32(128)
    thermal = np.broadcast_to(channels, (2, 2, 128)).copy()
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(4, 4), target_size=(4, 4), fit_mode=fit_mode, p=1)],
        additional_targets={"thermal": "image"},
        seed=137,
    )

    result = transform(image=np.full((2, 2, 3), 7, dtype=np.uint8), thermal=thermal, mosaic_metadata=[])
    assert result["thermal"].shape == (4, 4, 128)
    assert result["thermal"].dtype == dtype
    np.testing.assert_array_equal(result["thermal"], np.broadcast_to(channels, (4, 4, 128)))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_alias_rejects_129_channels_before_geometry(
    dtype: type[np.uint8] | type[np.float32],
    fit_mode: Literal["cover", "contain"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels = np.arange(129, dtype=dtype)
    if dtype is np.float32:
        channels /= np.float32(129)
    mosaic = A.Mosaic(grid_yx=(1, 1), cell_shape=(4, 4), target_size=(4, 4), fit_mode=fit_mode, p=1)
    transform = A.Compose([mosaic], additional_targets={"thermal": "image"}, seed=137)

    def unexpected_geometry(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Mosaic sampled geometry before validating alias channels")

    with monkeypatch.context() as patcher:
        patcher.setattr(A.Mosaic, "_calculate_geometry", unexpected_geometry)
        with pytest.raises(ValueError, match=r"target 'thermal'.*at most 128 channels"):
            transform(
                image=np.full((2, 2, 3), 7, dtype=np.uint8),
                thermal=np.broadcast_to(channels, (2, 2, 129)).copy(),
                mosaic_metadata=[],
            )


def test_mosaic_alias_rejects_129_channels_without_resize() -> None:
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(2, 2), target_size=(2, 2), p=1)],
        additional_targets={"thermal": "image"},
        seed=137,
    )
    with pytest.raises(ValueError, match=r"target 'thermal'.*at most 128 channels"):
        transform(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            thermal=np.zeros((2, 2, 129), dtype=np.uint8),
            mosaic_metadata=[],
        )


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_alias_rejects_129_channel_surplus_donor(fit_mode: Literal["cover", "contain"]) -> None:
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(4, 4), target_size=(4, 4), fit_mode=fit_mode, p=1)],
        additional_targets={"thermal": "image"},
        seed=137,
    )
    with pytest.raises(ValueError, match=r"donor 0 target 'thermal'.*at most 128 channels"):
        transform(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            thermal=np.zeros((2, 2, 128), dtype=np.uint8),
            mosaic_metadata=[
                {
                    "image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "thermal": np.zeros((2, 2, 129), dtype=np.uint8),
                }
            ],
        )


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "bad_alias",
    [
        None,
        np.zeros((5, 6), dtype=np.uint8),
        np.zeros((5, 7), dtype=np.float32),
        np.zeros((5, 7), dtype=np.uint16),
        np.zeros((5, 7, 2), dtype=np.uint8),
        np.zeros((1, 5, 7, 1), dtype=np.uint8),
        np.zeros((5, 7, 0), dtype=np.uint8),
    ],
)
def test_mosaic_rejects_incomplete_or_incompatible_surplus_alias(bad_alias: Any, strict: bool) -> None:
    primary = _paired_item(0, np.uint8, "hw")
    donors = [_paired_item(index, np.uint8, "hw") for index in range(1, 6)]
    donors[-1]["thermal"] = bad_alias
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), p=1)],
        additional_targets={"thermal": "image"},
        seed=137,
        strict=strict,
    )
    original = deepcopy(donors)
    with pytest.raises(ValueError, match="Mosaic donor 4 target 'thermal'"):
        transform(**primary, mosaic_metadata=donors)
    np.testing.assert_equal(donors, original)


def test_mosaic_rejects_fill_incompatible_with_alias_channels() -> None:
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), fill=(1, 2, 3), p=1)],
        additional_targets={"thermal": "image"},
        seed=137,
    )
    with pytest.raises(ValueError, match="fill must match channels of target 'thermal'"):
        transform(**_paired_item(0, np.uint8, "hw"), mosaic_metadata=[])


@pytest.mark.parametrize("alias", ["unused", "additional_images"])
def test_mosaic_registered_but_absent_alias_does_not_require_donor_data(alias: str) -> None:
    items = [{"image": _paired_item(index, np.uint8, "hw")["image"]} for index in range(4)]
    for index, item in enumerate(items[1:], start=1):
        item[alias] = _paired_item(index, np.uint8, "hw")["thermal"]
    original = deepcopy(items)
    result = _pipeline((2, 2), "contain", {alias: "image"})(**items[0], mosaic_metadata=items[1:])
    assert alias not in result
    _rgb_placements(result["image"], items)
    _assert_unchanged(items, original)


def test_mosaic_canonical_invalid_donor_keeps_warning_and_skip_policy() -> None:
    primary = _paired_item(0, np.uint8, "hw")
    with pytest.warns(UserWarning, match="skipped due to incompatibility"):
        result = _pipeline((2, 2), "contain", {"thermal": "image"})(
            **primary, mosaic_metadata=[{"image": np.zeros((5, 7, 2), dtype=np.uint8)}]
        )
    np.testing.assert_array_equal(result["thermal"], np.tile(primary["thermal"], (2, 2)))


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_alias_preserves_annotation_and_mask_alignment(fit_mode: Literal["cover", "contain"]) -> None:
    items: list[dict[str, Any]] = [_paired_item(index, np.uint8, "hw") for index in range(4)]
    for index, item in enumerate(items):
        item.update(mask=np.full((5, 7), index + 1, dtype=np.uint8), bboxes=[[1, 1, 5, 4]], keypoints=[[2, 2]])
        if index:
            item.update(bbox_labels={"classes": [f"class{index}"]}, keypoint_labels={"landmarks": [f"point{index}"]})
        else:
            item.update(classes=["class0"], landmarks=["point0"])
    original = deepcopy(items)
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), center_range=(0.5, 0.5), fit_mode=fit_mode, p=1)],
        additional_targets={"thermal": "image"},
        bbox_params=A.BboxParams(coord_format="pascal_voc", bbox_type="hbb", label_fields=["classes"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["landmarks"]),
        seed=137,
        strict=True,
    )
    result = transform(**items[0], mosaic_metadata=items[1:])
    placements = _rgb_placements(result["image"], items)
    assert len(result["bboxes"]) == len(result["keypoints"]) == 4
    for (row, column), donor_id in placements.items():
        region = np.s_[row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7]
        np.testing.assert_array_equal(result["thermal"][region], items[donor_id]["thermal"])
        np.testing.assert_array_equal(result["mask"][region], items[donor_id]["mask"])
        bbox = result["bboxes"][list(result["classes"]).index(f"class{donor_id}")]
        point = result["keypoints"][list(result["landmarks"]).index(f"point{donor_id}")]
        np.testing.assert_allclose(bbox, np.array([1, 1, 5, 4]) + [7 * column, 5 * row, 7 * column, 5 * row], atol=1e-6)
        np.testing.assert_allclose(point, [2 + 7 * column, 2 + 5 * row], atol=1e-6)
    np.testing.assert_equal(items, original)


def test_mosaic_alias_survives_masks_only_preprocessing() -> None:
    items = [_paired_item(index, np.uint8, "hw") for index in range(4)]
    for item in items:
        item["masks"] = np.ones((1, 5, 7), dtype=np.uint8)
    original = deepcopy(items)
    result = _pipeline((2, 2), "contain", {"thermal": "image"})(**items[0], mosaic_metadata=items[1:])
    for (row, column), donor_id in _rgb_placements(result["image"], items).items():
        np.testing.assert_array_equal(
            result["thermal"][row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7], items[donor_id]["thermal"]
        )
    assert result["masks"].shape == (4, 10, 14)
    np.testing.assert_equal(items, original)


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_alias_survives_instance_id_remapping(fit_mode: Literal["cover", "contain"]) -> None:
    item = _paired_item(0, np.uint8, "hw")
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 1:6] = 1
    transform = A.Compose(
        [
            A.Mosaic(
                grid_yx=(1, 1),
                cell_shape=(5, 7),
                target_size=(5, 7),
                center_range=(0.5, 0.5),
                fit_mode=fit_mode,
                interpolation=cv2.INTER_NEAREST,
                p=1,
            )
        ],
        additional_targets={"thermal": "image"},
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
        instance_binding=["masks", "bboxes"],
        seed=137,
    )
    result = transform(
        **item,
        instances=[
            {
                "mask": mask,
                "bbox": np.array([1, 1, 6, 4], dtype=np.float32),
                "bbox_labels": {"class_id": 137},
            }
        ],
        mosaic_metadata=[],
    )

    np.testing.assert_array_equal(result["thermal"], item["thermal"])
    assert len(result["instances"]) == 1
    np.testing.assert_array_equal(result["instances"][0]["mask"], mask)
    np.testing.assert_array_equal(result["instances"][0]["bbox"], [1, 1, 6, 4])
    assert result["instances"][0]["bbox_labels"]["class_id"] == 137


def test_mosaic_alias_preserves_read_only_strided_inputs_and_output_ownership() -> None:
    items = [_paired_item(index, np.uint8, "hw") for index in range(4)]
    for item in items:
        for name, value in item.items():
            item[name] = value[:, ::-1]
            item[name].setflags(write=False)
    original = deepcopy(items)
    transform = _pipeline((2, 2), "cover", {"thermal": "image"})
    first = transform(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    second = transform(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    np.testing.assert_array_equal(first["thermal"], second["thermal"])
    first["thermal"][:] = 0
    assert np.any(second["thermal"])
    _assert_unchanged(items, original)


@pytest.mark.parametrize("p", [0, 1])
def test_mosaic_direct_transform_with_add_targets(p: int) -> None:
    item = _paired_item(0, np.uint8, "hw")
    transform = A.Mosaic(grid_yx=(1, 1), cell_shape=(5, 7), target_size=(5, 7), p=p)
    transform.add_targets({"thermal": "image"})
    transform.set_random_seed(137)
    result = transform(**item, mosaic_metadata=[])
    np.testing.assert_array_equal(result["thermal"], item["thermal"])


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_direct_transform_accepts_hwc1_donor_for_hw_alias(fit_mode: Literal["cover", "contain"]) -> None:
    primary = _paired_item(0, np.uint8, "hw")
    donor = _paired_item(1, np.uint8, "hwc1")
    transform = A.Mosaic(
        grid_yx=(1, 2),
        cell_shape=(5, 7),
        target_size=(5, 14),
        center_range=(0.5, 0.5),
        fit_mode=fit_mode,
        interpolation=cv2.INTER_NEAREST,
        p=1,
    )
    transform.add_targets({"thermal": "image"})
    transform.set_random_seed(137)
    result = transform(**primary, mosaic_metadata=[donor])

    assert result["thermal"].shape == (5, 14)
    for column in range(2):
        region = np.s_[:, 7 * column : 7 * (column + 1)]
        image_tile = result["image"][region]
        source = primary if np.array_equal(image_tile, primary["image"]) else donor
        expected_thermal = source["thermal"] if source["thermal"].ndim == 2 else source["thermal"][..., 0]
        np.testing.assert_array_equal(result["thermal"][region], expected_thermal)


def test_mosaic_alias_replay_and_constructor_serialization() -> None:
    items = [_paired_item(index, np.uint8, "hw") for index in range(4)]
    compose = _pipeline((2, 2), "contain", {"thermal": "image"})
    restored = A.from_dict(A.to_dict(compose))
    expected = compose(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    actual = restored(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    np.testing.assert_array_equal(actual["thermal"], expected["thermal"])
    replay = A.ReplayCompose(compose.transforms, additional_targets={"thermal": "image"})
    replay.set_random_seed(137)
    recorded = replay(**items[0], mosaic_metadata=items[1:])
    replayed = A.ReplayCompose.replay(recorded["replay"], **items[0], mosaic_metadata=items[1:])
    for name in ("image", "thermal"):
        np.testing.assert_array_equal(replayed[name], recorded[name])


def test_mosaic_alias_tensor_bridge_matches_numpy_pairing() -> None:
    items = [_paired_item(index, np.uint8, "hwc1") for index in range(4)]
    transform = _pipeline((2, 2), "contain", {"thermal": "image"})
    expected = transform(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    tensor_primary = {name: torch.from_numpy(value).permute(2, 0, 1) for name, value in items[0].items()}
    actual = transform(**tensor_primary, mosaic_metadata=items[1:], invocation_seed=137)
    assert isinstance(actual["thermal"], torch.Tensor)
    for name in ("image", "thermal"):
        np.testing.assert_array_equal(actual[name].permute(1, 2, 0).numpy(), expected[name])


@pytest.mark.parametrize("explicit_seed", [False, True])
def test_mosaic_alias_validation_before_sampling_preserves_retry(explicit_seed: bool) -> None:
    items = [_paired_item(index, np.uint8, "hw") for index in range(6)]
    transform = _pipeline((2, 2), "cover", {"thermal": "image"})
    fresh = _pipeline((2, 2), "cover", {"thermal": "image"})
    seed_kwargs = {"invocation_seed": 137} if explicit_seed else {}
    with pytest.raises(ValueError, match="donor 0 target 'thermal'"):
        transform(**items[0], mosaic_metadata=[{"image": items[1]["image"]}], **seed_kwargs)
    # With p=1 throughout, no generator is consumed before the alias validation failure.
    expected = fresh(**items[0], mosaic_metadata=items[1:], **seed_kwargs)
    actual = transform(**items[0], mosaic_metadata=items[1:], **seed_kwargs)
    for name in ("image", "thermal"):
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("primary_hwc1", [False, True])
def test_mosaic_semantic_alias_hw_hwc1_interchange(direct: bool, primary_hwc1: bool) -> None:
    items = [_semantic_item(index) for index in range(4)]
    for index, item in enumerate(items):
        if (index == 0) == primary_hwc1:
            item["auxmask"] = item["auxmask"][..., None]
    pipeline = _pipeline((2, 2), "cover", {"auxmask": "mask", "planes": "mask"})
    transform = pipeline.transforms[0] if direct else pipeline
    result = transform(**items[0], mosaic_metadata=items[1:])
    assert result["auxmask"].shape == ((10, 14, 1) if primary_hwc1 else (10, 14))
    for (row, column), donor_id in _rgb_placements(result["image"], items).items():
        tile = result["auxmask"][row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7]
        np.testing.assert_array_equal(tile.reshape(5, 7), items[donor_id]["auxmask"].reshape(5, 7))


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
@pytest.mark.parametrize("downscale", [False, True])
def test_mosaic_semantic_alias_real_geometry(fit_mode: Literal["cover", "contain"], downscale: bool) -> None:
    pattern = np.array([[-310, -309, -308, -307], [-210, -209, -208, -207]], dtype=np.int16)
    source = np.repeat(np.repeat(pattern, 2, axis=0), 2, axis=1) if downscale else pattern
    target_size = (2, 4) if downscale else (4, 4)
    transform = A.Compose(
        [
            A.Mosaic(
                grid_yx=(1, 1),
                cell_shape=target_size,
                target_size=target_size,
                fit_mode=fit_mode,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=77,
                fill_mask=13,
                p=1,
            )
        ],
        additional_targets={"auxmask": "mask"},
        seed=137,
    )
    result = transform(
        image=np.zeros((*source.shape, 3), dtype=np.uint8),
        mask=np.ones(source.shape, dtype=np.uint8),
        auxmask=source,
        mosaic_metadata=[],
    )
    if downscale:
        expected = pattern
    elif fit_mode == "cover":
        expected = np.repeat(np.repeat(pattern, 2, axis=0), 2, axis=1)[:, :4]
    else:
        expected = np.full((4, 4), 13, dtype=np.int16)
        expected[1:3] = pattern
        np.testing.assert_array_equal(result["image"][[0, 3]], np.full((2, 4, 3), 77, dtype=np.uint8))
    np.testing.assert_array_equal(result["auxmask"], expected, strict=True)


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_semantic_alias_donors_with_different_sizes(fit_mode: Literal["cover", "contain"]) -> None:
    items = []
    expected_tiles = []
    for index, scale in enumerate((2, 1, 4, 1)):
        pattern = np.arange(6, dtype=np.int16).reshape(2, 3) + 1000 + 100 * index
        source = pattern.repeat(scale, axis=0).repeat(scale, axis=1)
        expected = pattern.repeat(2, axis=0).repeat(2, axis=1)
        if index == 3:
            source = np.full((2, 6), 1370, np.int16)
            expected[:] = 1370
            if fit_mode == "contain":
                expected[[0, 3]] = 13
        items.append(
            {
                "image": np.full((*source.shape, 3), index + 1, np.uint8),
                "mask": np.full(source.shape, index + 1, np.uint8),
                "auxmask": source,
            }
        )
        expected_tiles.append(expected)
    original = deepcopy(items)
    transform = A.Compose(
        [
            A.Mosaic(
                cell_shape=(4, 6),
                target_size=(8, 12),
                center_range=(0.5, 0.5),
                fit_mode=fit_mode,
                interpolation=cv2.INTER_NEAREST,
                fill=77,
                fill_mask=13,
                p=1,
            )
        ],
        additional_targets={"auxmask": "mask"},
        seed=137,
    )
    result = transform(**items[0], mosaic_metadata=items[1:])
    selected = []
    for row in range(2):
        for column in range(2):
            region = np.s_[row * 4 : (row + 1) * 4, column * 6 : (column + 1) * 6]
            donor_id = int(result["image"][region][2, 3, 0]) - 1
            selected.append(donor_id)
            np.testing.assert_array_equal(result["auxmask"][region], expected_tiles[donor_id], strict=True)
    assert sorted(selected) == [0, 1, 2, 3]
    _assert_unchanged(items, original)


@pytest.mark.parametrize(
    ("dtype", "channels", "offset"),
    [(np.uint8, 1, 10), (np.float32, 2, 0.5), (np.uint16, 5, 1000), (np.int16, 1, -300), (np.int32, 2, -70000)],
)
def test_mosaic_semantic_alias_nearest_dtype_and_channels(dtype: Any, channels: int, offset: float) -> None:
    source = (np.arange(6 * channels).reshape(2, 3, channels) + offset).astype(dtype)
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(4, 6), target_size=(4, 6), p=1)],
        additional_targets={"auxmask": "mask"},
        seed=137,
    )
    result = transform(
        image=np.zeros((2, 3, 3), dtype=np.uint8),
        mask=np.zeros((2, 3), dtype=np.uint8),
        auxmask=source,
        mosaic_metadata=[],
    )
    np.testing.assert_array_equal(result["auxmask"], source.repeat(2, axis=0).repeat(2, axis=1), strict=True)


def test_mosaic_semantic_alias_uses_mask_linear_interpolation() -> None:
    source = np.array([[0, 1], [1, 0]], dtype=np.float32)
    expected = np.array(
        [[0, 0.25, 0.75, 1], [0.25, 0.375, 0.625, 0.75], [0.75, 0.625, 0.375, 0.25], [1, 0.75, 0.25, 0]],
        dtype=np.float32,
    )
    transform = A.Compose(
        [
            A.Mosaic(
                grid_yx=(1, 1),
                cell_shape=(4, 4),
                target_size=(4, 4),
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_LINEAR,
                p=1,
            )
        ],
        additional_targets={"auxmask": "mask"},
        seed=137,
    )
    result = transform(image=np.zeros((2, 2, 3), np.uint8), mask=1 - source, auxmask=source, mosaic_metadata=[])
    np.testing.assert_allclose(result["auxmask"], expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(result["mask"], 1 - expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("alias", [False, True])
def test_mosaic_semantic_alias_int32_linear_matches_canonical_rejection(alias: bool) -> None:
    source = np.array([[-70000, 80000, 0], [0, 80000, -70000]], dtype=np.int32)
    data = {"mask": np.zeros((2, 3), np.uint8), "auxmask": source} if alias else {"mask": source}
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(4, 6), target_size=(4, 6), mask_interpolation=cv2.INTER_LINEAR, p=1)],
        additional_targets={"auxmask": "mask"} if alias else None,
        seed=137,
    )
    with pytest.raises(cv2.error):
        transform(image=np.zeros((2, 3, 3), np.uint8), **data, mosaic_metadata=[])


@pytest.mark.parametrize("fit_mode", ["cover", "contain"])
def test_mosaic_semantic_alias_128_channels(fit_mode: Literal["cover", "contain"]) -> None:
    source = np.broadcast_to(np.arange(128, dtype=np.uint8), (2, 4, 128)).copy()
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(4, 4), target_size=(4, 4), fit_mode=fit_mode, fill_mask=137, p=1)],
        additional_targets={"auxmask": "mask"},
        seed=137,
    )
    result = transform(
        image=np.zeros((2, 4, 3), np.uint8), mask=np.ones((2, 4), np.uint8), auxmask=source, mosaic_metadata=[]
    )
    expected = np.broadcast_to(source[0, 0], (4, 4, 128)).copy()
    if fit_mode == "contain":
        expected[[0, 3]] = 137
    np.testing.assert_array_equal(result["auxmask"], expected, strict=True)


@pytest.mark.parametrize("donor", [False, True])
def test_mosaic_semantic_alias_rejects_129_channels_before_sampling(donor: bool, mocker: MockerFixture) -> None:
    items = [_semantic_item(index) for index in range(2)]
    items[int(donor)]["auxmask"] = np.zeros((5, 7, 129), dtype=np.uint16)
    transform = _pipeline((1, 1), "cover", {"auxmask": "mask", "planes": "mask"})
    geometry = mocker.spy(transform.transforms[0], "_calculate_geometry")
    selection = mocker.spy(transform.transforms[0], "_select_additional_items")
    with pytest.raises(ValueError, match=r"target 'auxmask'.*at most 128 channels"):
        transform(**items[0], mosaic_metadata=items[1:])
    geometry.assert_not_called()
    selection.assert_not_called()


@pytest.mark.parametrize("fill_mask", [(3, 4), (3,), (3, 4, 5)])
def test_mosaic_semantic_alias_tuple_fill(fill_mask: tuple[int, ...]) -> None:
    source = np.full((2, 4, 2), -300, dtype=np.int16)
    transform = A.Compose(
        [
            A.Mosaic(
                grid_yx=(1, 1),
                cell_shape=(4, 4),
                target_size=(4, 4),
                fit_mode="contain",
                fill_mask=fill_mask,
                p=1,
            )
        ],
        additional_targets={"auxmask": "mask"},
        seed=137,
    )
    data = dict(image=np.zeros((2, 4, 3), np.uint8), mask=source + 1, auxmask=source, mosaic_metadata=[])
    if len(fill_mask) != 2:
        with pytest.raises(ValueError, match="fill_mask must match channels"):
            transform(**data)
        return
    result = transform(**data)
    expected = np.empty((4, 4, 2), dtype=np.int16)
    expected[:] = fill_mask
    expected[1:3] = source
    np.testing.assert_array_equal(result["auxmask"], expected, strict=True)


def test_mosaic_semantic_alias_tuple_fill_checks_each_target() -> None:
    transform = A.Compose(
        [A.Mosaic(grid_yx=(1, 1), cell_shape=(5, 7), target_size=(5, 7), fill_mask=(3,), p=1)],
        additional_targets={"auxmask": "mask", "planes": "mask"},
        seed=137,
    )
    with pytest.raises(ValueError, match="fill_mask must match channels of target 'planes'"):
        transform(**_semantic_item(0), mosaic_metadata=[])


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "bad_alias",
    [
        "missing",
        None,
        [1, 2],
        np.zeros((0, 7), np.uint16),
        np.zeros((5, 7, 0), np.uint16),
        np.zeros((1, 5, 7, 1), np.uint16),
        np.zeros((5, 6), np.uint16),
        np.zeros((5, 7), np.int16),
        np.zeros((5, 7, 2), np.uint16),
    ],
)
def test_mosaic_semantic_alias_invalid_surplus_pair(bad_alias: Any, strict: bool, mocker: MockerFixture) -> None:
    items: list[dict[str, Any]] = [_semantic_item(index) for index in range(6)]
    if isinstance(bad_alias, str):
        del items[-1]["auxmask"]
    else:
        items[-1]["auxmask"] = bad_alias
    original = deepcopy(items)
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), p=1)],
        additional_targets={"auxmask": "mask", "planes": "mask"},
        seed=137,
        strict=strict,
    )
    geometry = mocker.spy(transform.transforms[0], "_calculate_geometry")
    selection = mocker.spy(transform.transforms[0], "_select_additional_items")
    with pytest.raises(ValueError, match="Mosaic donor 4 target 'auxmask'"):
        transform(**items[0], mosaic_metadata=items[1:])
    geometry.assert_not_called()
    selection.assert_not_called()
    np.testing.assert_equal(items, original)


@pytest.mark.parametrize(
    "bad_alias", [[1, 2], np.zeros((0, 7), np.uint16), np.zeros((1, 5, 7, 1), np.uint16), np.zeros((5, 6), np.uint16)]
)
def test_mosaic_semantic_alias_invalid_primary(bad_alias: Any) -> None:
    item = _semantic_item(0)
    item["auxmask"] = bad_alias
    transform = _pipeline((1, 1), "cover", {"auxmask": "mask", "planes": "mask"})
    with pytest.raises((TypeError, ValueError)):
        transform(**item, mosaic_metadata=[])


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("fill_mask", [0, (0,)])
def test_mosaic_semantic_alias_requires_canonical_primary_before_sampling(
    direct: bool,
    missing: bool,
    fill_mask: int | tuple[int, ...],
    mocker: MockerFixture,
) -> None:
    item: dict[str, Any] = _semantic_item(0)
    del item["planes"]
    if missing:
        del item["mask"]
    else:
        item["mask"] = None
    pipeline = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), fill_mask=fill_mask, p=1)],
        additional_targets={"auxmask": "mask"},
        seed=137,
    )
    mosaic = pipeline.transforms[0]
    transform = mosaic if direct else pipeline
    geometry = mocker.spy(mosaic, "_calculate_geometry")
    selection = mocker.spy(mosaic, "_select_additional_items")

    with pytest.raises(ValueError, match=r"canonical.*mask"):
        transform(**item, mosaic_metadata=[])

    geometry.assert_not_called()
    selection.assert_not_called()


@pytest.mark.parametrize(("grid", "donor_count"), [((2, 2), 3), ((2, 2), 5), ((1, 1), 2)])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("explicit_seed", [False, True])
def test_mosaic_semantic_alias_requires_canonical_donor_before_sampling(
    grid: tuple[int, int],
    donor_count: int,
    missing: bool,
    explicit_seed: bool,
    mocker: MockerFixture,
) -> None:
    items: list[dict[str, Any]] = [_semantic_item(index) for index in range(donor_count + 1)]
    invalid = deepcopy(items)
    if missing:
        del invalid[-1]["mask"]
    else:
        invalid[-1]["mask"] = None
    original = deepcopy(invalid)
    transform = _pipeline(grid, "cover", {"auxmask": "mask", "planes": "mask"})
    fresh = _pipeline(grid, "cover", {"auxmask": "mask", "planes": "mask"})
    geometry = mocker.spy(transform.transforms[0], "_calculate_geometry")
    selection = mocker.spy(transform.transforms[0], "_select_additional_items")
    seed_kwargs = {"invocation_seed": 137} if explicit_seed else {}
    with pytest.raises(ValueError, match=rf"donor {donor_count - 1} requires a non-None canonical mask.*auxmask"):
        transform(**invalid[0], mosaic_metadata=invalid[1:], **seed_kwargs)
    geometry.assert_not_called()
    selection.assert_not_called()
    np.testing.assert_equal(invalid, original)
    expected = fresh(**items[0], mosaic_metadata=items[1:], **seed_kwargs)
    actual = transform(**items[0], mosaic_metadata=items[1:], **seed_kwargs)
    for name in items[0]:
        np.testing.assert_array_equal(actual[name], expected[name], strict=True)


@pytest.mark.parametrize("explicit_seed", [False, True])
@pytest.mark.parametrize("grid", [(1, 1), (2, 2)])
def test_mosaic_semantic_alias_unused_invalid_pair_preserves_retry(
    explicit_seed: bool,
    grid: tuple[int, int],
) -> None:
    items = [_semantic_item(index) for index in range(6)]
    invalid = deepcopy(items)
    del invalid[-1]["auxmask"]
    transform = _pipeline(grid, "cover", {"auxmask": "mask", "planes": "mask"})
    fresh = _pipeline(grid, "cover", {"auxmask": "mask", "planes": "mask"})
    seed_kwargs = {"invocation_seed": 137} if explicit_seed else {}
    with pytest.raises(ValueError, match="donor 4 target 'auxmask'"):
        transform(**invalid[0], mosaic_metadata=invalid[1:], **seed_kwargs)
    actual = transform(**items[0], mosaic_metadata=items[1:], **seed_kwargs)
    expected = fresh(**items[0], mosaic_metadata=items[1:], **seed_kwargs)
    for name in items[0]:
        np.testing.assert_array_equal(actual[name], expected[name], strict=True)


@pytest.mark.parametrize("mode", ["canonical", "absent", "none", "image"])
def test_mosaic_inactive_mask_alias_keeps_missing_canonical_donor_fill(mode: str) -> None:
    items: list[dict[str, Any]] = [{"image": _paired_item(index, np.uint8, "hw")["image"]} for index in range(4)]
    items[0]["mask"] = np.ones((5, 7), dtype=np.uint8)
    items[2]["mask"] = None
    aliases = {} if mode == "canonical" else {"auxmask": "mask"}
    if mode == "none":
        items[0]["auxmask"] = None
    if mode == "image":
        aliases["thermal"] = "image"
        for index, item in enumerate(items):
            item["thermal"] = _paired_item(index, np.uint8, "hw")["thermal"]
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), center_range=(0.5, 0.5), fill_mask=13, p=1)],
        additional_targets=aliases,
        seed=137,
    )
    result = transform(**items[0], mosaic_metadata=items[1:])
    for (row, column), donor_id in _rgb_placements(result["image"], items).items():
        expected = np.full((5, 7), 1 if donor_id == 0 else 13, dtype=np.uint8)
        np.testing.assert_array_equal(result["mask"][row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7], expected)
        if mode == "image":
            np.testing.assert_array_equal(
                result["thermal"][row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7],
                items[donor_id]["thermal"],
            )
    assert result.get("auxmask") is None


def test_mosaic_semantic_alias_canonical_invalid_donor_is_skipped() -> None:
    primary = _semantic_item(0)
    with pytest.warns(UserWarning, match="skipped due to incompatibility"):
        result = _pipeline((2, 2), "contain", {"auxmask": "mask", "planes": "mask"})(
            **primary,
            mosaic_metadata=[{"image": np.zeros((5, 7, 2), dtype=np.uint8)}],
        )
    np.testing.assert_array_equal(result["auxmask"], np.tile(primary["auxmask"], (2, 2)))
    np.testing.assert_array_equal(result["planes"], np.tile(primary["planes"], (2, 2, 1)))


@pytest.mark.parametrize("empty", [False, True])
def test_mosaic_semantic_alias_with_images_and_annotations(empty: bool) -> None:
    items: list[dict[str, Any]] = [_semantic_item(index) for index in range(4)]
    for index, item in enumerate(items):
        item.update(
            thermal=_paired_item(index, np.uint8, "hw")["thermal"],
            bboxes=np.empty((0, 4), np.float32) if empty else [[1, 1, 5, 4]],
            keypoints=np.empty((0, 2), np.float32) if empty else [[2, 2]],
        )
        labels = [] if empty else [index]
        if index:
            item.update(bbox_labels={"classes": labels}, keypoint_labels={"landmarks": labels})
        else:
            item.update(classes=labels, landmarks=labels)
    original = deepcopy(items)
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), center_range=(0.5, 0.5), p=1)],
        additional_targets={"thermal": "image", "auxmask": "mask", "planes": "mask"},
        bbox_params=A.BboxParams(coord_format="pascal_voc", bbox_type="hbb", label_fields=["classes"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["landmarks"]),
        seed=137,
        strict=True,
    )
    result = transform(**items[0], mosaic_metadata=items[1:])
    assert len(result["bboxes"]) == len(result["keypoints"]) == (0 if empty else 4)
    for (row, column), donor_id in _rgb_placements(result["image"], items).items():
        region = np.s_[row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7]
        for name in ("thermal", "mask", "auxmask", "planes"):
            np.testing.assert_array_equal(result[name][region], items[donor_id][name], strict=True)
        if not empty:
            bbox = result["bboxes"][list(result["classes"]).index(donor_id)]
            point = result["keypoints"][list(result["landmarks"]).index(donor_id)]
            np.testing.assert_allclose(
                bbox, np.array([1, 1, 5, 4]) + [7 * column, 5 * row, 7 * column, 5 * row], atol=1e-6
            )
            np.testing.assert_allclose(point, [2 + 7 * column, 2 + 5 * row], atol=1e-6)
    np.testing.assert_equal(items, original)


def test_mosaic_semantic_alias_masks_only_preprocessing() -> None:
    items = [_semantic_item(index) for index in range(4)]
    for item in items:
        item["masks"] = np.ones((1, 5, 7), dtype=np.uint8)
    original = deepcopy(items)
    result = _pipeline((2, 2), "contain", {"auxmask": "mask", "planes": "mask"})(
        **items[0],
        mosaic_metadata=items[1:],
    )
    assert result["masks"].shape == (4, 10, 14)
    np.testing.assert_array_equal(result["masks"].sum(axis=0), np.ones((10, 14)))
    for (row, column), donor_id in _rgb_placements(result["image"], items).items():
        region = np.s_[row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7]
        for name in ("auxmask", "planes"):
            np.testing.assert_array_equal(result[name][region], items[donor_id][name], strict=True)
    np.testing.assert_equal(items, original)


@pytest.mark.parametrize("filtered", [False, True])
def test_mosaic_semantic_alias_multicell_instance_remapping(filtered: bool) -> None:
    items: list[dict[str, Any]] = [_semantic_item(index) for index in range(4)]
    instance_mask = np.zeros((5, 7), dtype=np.uint8)
    instance_mask[1:4, 1:5] = 1
    items[0]["instances"] = [
        {"mask": instance_mask, "bbox": np.array([1, 1, 5, 4], np.float32), "bbox_labels": {"classes": 0}},
    ]
    for index, item in enumerate(items[1:], start=1):
        small = filtered and index == 1
        mask = instance_mask.copy()
        if small:
            mask[:] = 0
            mask[1, 1] = 1
        item.update(
            masks=mask[None],
            bboxes=np.array([[1, 1, 2, 2] if small else [1, 1, 5, 4]], np.float32),
            bbox_labels={"classes": [index]},
        )
    original = deepcopy(items)
    transform = A.Compose(
        [A.Mosaic(cell_shape=(5, 7), target_size=(10, 14), center_range=(0.5, 0.5), p=1)],
        additional_targets={"auxmask": "mask", "planes": "mask"},
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            bbox_type="hbb",
            label_fields=["classes"],
            min_area=5 if filtered else 0,
        ),
        instance_binding=["masks", "bboxes"],
        seed=137,
    )
    result = transform(**items[0], mosaic_metadata=items[1:])
    by_class = {instance["bbox_labels"]["classes"]: instance for instance in result["instances"]}
    assert set(by_class) == ({0, 2, 3} if filtered else {0, 1, 2, 3})
    for (row, column), donor_id in _rgb_placements(result["image"], items).items():
        region = np.s_[row * 5 : (row + 1) * 5, column * 7 : (column + 1) * 7]
        for name in ("auxmask", "planes"):
            np.testing.assert_array_equal(result[name][region], items[donor_id][name], strict=True)
        if donor_id in by_class:
            expected_mask = np.zeros((10, 14), dtype=np.uint8)
            expected_mask[region] = instance_mask
            np.testing.assert_array_equal(by_class[donor_id]["mask"], expected_mask)
            np.testing.assert_allclose(
                by_class[donor_id]["bbox"],
                np.array([1, 1, 5, 4]) + [7 * column, 5 * row, 7 * column, 5 * row],
                atol=1e-6,
            )
    np.testing.assert_equal(items, original)


def test_mosaic_semantic_alias_readonly_views_and_output_ownership() -> None:
    items = [_semantic_item(index) for index in range(4)]
    for item in items:
        for name, value in item.items():
            item[name] = value[:, ::-1]
            item[name].setflags(write=False)
    original = deepcopy(items)
    transform = _pipeline((2, 2), "contain", {"auxmask": "mask", "planes": "mask"})
    with pytest.raises(ValueError, match="canonical mask"):
        transform(**items[0], mosaic_metadata=[{name: value for name, value in items[1].items() if name != "mask"}])
    first = transform(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    second = transform(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    for name in items[0]:
        np.testing.assert_array_equal(first[name], second[name], strict=True)
        assert not np.shares_memory(first[name], second[name])
        for item in items:
            assert not np.shares_memory(first[name], item[name])
        first[name][:] = 0
        assert np.any(second[name])
    _assert_unchanged(items, original)


@pytest.mark.parametrize("direct", [False, True])
def test_mosaic_semantic_alias_skip_keeps_identity(direct: bool) -> None:
    item = _semantic_item(0)
    mosaic = A.Mosaic(p=0)
    mosaic.add_targets({"auxmask": "mask", "planes": "mask"})
    transform = mosaic if direct else A.Compose([mosaic], seed=137)
    result = transform(**item, mosaic_metadata=[{"image": item["image"]}])
    for name in item:
        assert result[name] is item[name]


@pytest.mark.parametrize("drift", ["dtype", "shape"])
def test_mosaic_semantic_alias_replay_and_serialization(drift: str) -> None:
    items = [_semantic_item(index) for index in range(4)]
    aliases = {"auxmask": "mask", "planes": "mask"}
    compose = _pipeline((2, 2), "contain", aliases)
    restored = A.from_dict(A.to_dict(compose))
    expected = compose(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    actual = restored(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    replay = A.ReplayCompose(compose.transforms, additional_targets=aliases)
    replay.set_random_seed(137)
    recorded = replay(**items[0], mosaic_metadata=items[1:])
    replayed = A.ReplayCompose.replay(recorded["replay"], **items[0], mosaic_metadata=items[1:])
    for name in items[0]:
        np.testing.assert_array_equal(actual[name], expected[name], strict=True)
        np.testing.assert_array_equal(replayed[name], recorded[name], strict=True)
    changed = dict(items[0])
    changed["auxmask"] = (
        items[0]["auxmask"].astype(np.float32)
        if drift == "dtype"
        else np.repeat(items[0]["auxmask"][..., None], 2, axis=-1)
    )
    with pytest.raises(ValueError, match="requirements do not match target 'auxmask'"):
        A.ReplayCompose.replay(recorded["replay"], **changed, mosaic_metadata=items[1:])


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
@pytest.mark.parametrize("hw", [False, True])
def test_mosaic_semantic_alias_tensor_fallback(dtype: Any, hw: bool) -> None:
    items = [_semantic_item(index) for index in range(4)]
    for item in items:
        item["auxmask"] = (item["auxmask"] % 251).astype(dtype)
        if not hw:
            item["auxmask"] = item["auxmask"][..., None]
    transform = _pipeline((2, 2), "cover", {"auxmask": "mask", "planes": "mask"})
    expected = transform(**items[0], mosaic_metadata=items[1:], invocation_seed=137)
    tensor_primary = {
        name: torch.from_numpy(value) if value.ndim == 2 else torch.from_numpy(value).permute(2, 0, 1)
        for name, value in items[0].items()
    }
    actual = transform(**tensor_primary, mosaic_metadata=items[1:], invocation_seed=137)
    for name in items[0]:
        assert isinstance(actual[name], torch.Tensor)
        value = actual[name].numpy() if actual[name].ndim == 2 else actual[name].permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(value, expected[name], strict=True)
