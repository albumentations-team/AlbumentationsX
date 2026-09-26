"""Issue #68: paired image targets must retain their own pixels through Mosaic."""

from copy import deepcopy
from typing import Any, Literal

import cv2
import numpy as np
import pytest
import torch

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
