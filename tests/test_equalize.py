import json
import warnings
from typing import Literal

import cv2
import numpy as np
import pytest
import torch
from albucore import from_float, to_float
from PIL import Image, ImageOps

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel


def pil_equalize(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    channels = []
    for channel in range(image.shape[-1]):
        selected = None if mask is None else mask[..., 0 if mask.shape[-1] == 1 else channel]
        pil_mask = None if selected is None else Image.fromarray((selected != 0).astype(np.uint8) * 255)
        channels.append(np.asarray(ImageOps.equalize(Image.fromarray(image[..., channel]), mask=pil_mask)))
    return np.stack(channels, axis=-1)


def cv_equalize(image: np.ndarray) -> np.ndarray:
    return np.stack([cv2.equalizeHist(image[..., channel]) for channel in range(image.shape[-1])], axis=-1)


def issue_image(channels: int) -> np.ndarray:
    image = np.full((16, 16, channels), 32, dtype=np.uint8)
    image[-1, -1] = 224
    return image


def partial_mask_fixture(channels: int = 3) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((32, 32, channels), 32, dtype=np.uint8)
    image[15, -1] = 224
    image[16:24] = 64
    image[24:] = 128
    mask = np.zeros((32, 32, 1), dtype=np.uint8)
    mask[:16] = 137
    return image, mask


def run_equalize(
    image: np.ndarray,
    route: str,
    mode: Literal["pil", "cv"] = "pil",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if route == "functional":
        return fpixel.equalize(image, mode=mode, mask=mask)
    transform = A.Equalize(mode=mode, mask=mask, p=1)
    if route == "direct":
        return transform(image=image)["image"]
    return A.Compose([transform])(image=image)["image"]


@pytest.mark.parametrize("route", ["functional", "direct", "compose"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_equalize_pil_issue_values(route: str, channels: int, masked: bool) -> None:
    image = issue_image(channels)
    mask = np.full((16, 16, 1), 255, dtype=np.uint8) if masked else None
    expected = pil_equalize(image.copy(), mask)
    result = run_equalize(image, route, mask=mask)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert result.flags.writeable


@pytest.mark.parametrize("route", ["functional", "direct", "compose"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("mode,masked", [("pil", False), ("pil", True), ("cv", True)])
def test_equalize_preserves_inputs(route: str, channels: int, mode: Literal["pil", "cv"], masked: bool) -> None:
    image = issue_image(channels)
    before = image.copy()
    mask = np.full((16, 16, 1), 137, dtype=np.uint8) if masked else None
    mask_before = None if mask is None else mask.copy()
    result = run_equalize(image, route, mode=mode, mask=mask)
    np.testing.assert_array_equal(image, before)
    if mask is not None:
        np.testing.assert_array_equal(mask, mask_before)
    assert not np.shares_memory(result, image)
    assert result.flags.writeable


def test_equalize_compose_channel_free_grayscale() -> None:
    image = issue_image(1)[..., 0]
    before = image.copy()
    expected = pil_equalize(before[..., None])[..., 0]
    result = run_equalize(image, "compose")
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
    assert result.shape == image.shape


@pytest.mark.parametrize(
    "image",
    [
        np.repeat(np.array([0, 32, 100, 255], dtype=np.uint8), [256, 300, 200, 268]).reshape(32, 32, 1),
        np.random.default_rng(137).integers(0, 256, (37, 53, 3), dtype=np.uint8),
        np.full((32, 32, 1), 137, dtype=np.uint8),
        np.full((1, 1, 1), 137, dtype=np.uint8),
        np.array([32, 224], dtype=np.uint8).reshape(1, 2, 1),
        issue_image(1),
    ],
    ids=["nonuniform", "random-nonsquare", "constant", "singleton", "step-zero", "step-one"],
)
def test_equalize_pil_histogram_boundaries(image: np.ndarray) -> None:
    before = image.copy()
    expected = pil_equalize(before)
    result = run_equalize(image, "functional")
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("mask_kind", ["partial", "bool", "zero"])
def test_equalize_pil_histogram_mask_applies_to_whole_image(mask_kind: str) -> None:
    image, mask = partial_mask_fixture()
    if mask_kind == "bool":
        mask = mask.astype(bool)
    elif mask_kind == "zero":
        mask.fill(0)
    before, mask_before = image.copy(), mask.copy()
    expected = pil_equalize(before, mask_before)
    result = run_equalize(image, "compose", mask=mask)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(mask, mask_before)
    if mask_kind == "zero":
        np.testing.assert_array_equal(result, before)
    else:
        assert np.any(result[16:] != before[16:])


def test_equalize_pil_per_channel_mask() -> None:
    image = np.random.default_rng(137).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[:20, :, 0] = 1
    mask[15:, :, 1] = 137
    mask[:, 10:40, 2] = 255
    before, mask_before = image.copy(), mask.copy()
    expected = pil_equalize(before, mask_before)
    result = run_equalize(image, "direct", mask=mask)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("channels,by_channels", [(1, True), (3, False)])
def test_equalize_rejects_incompatible_masks(channels: int, by_channels: bool) -> None:
    with pytest.raises(ValueError, match="mask"):
        A.Compose([A.Equalize(mask=np.ones((16, 16, 3), dtype=np.uint8), by_channels=by_channels, p=1)])(
            image=issue_image(channels),
        )


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_equalize_cv_full_histogram(channels: int, masked: bool) -> None:
    image = issue_image(channels)
    before = image.copy()
    expected = cv_equalize(before)
    mask = np.full((16, 16, 1), 255, dtype=np.uint8) if masked else None
    result = run_equalize(image, "compose", mode="cv", mask=mask)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)


def test_equalize_cv_partial_mask() -> None:
    image, mask = partial_mask_fixture(1)
    image[-1, -2:, 0] = [255, 224]
    before, mask_before = image.copy(), mask.copy()
    selection = mask[..., 0] != 0
    selected_expected = cv2.equalizeHist(before[..., 0][selection]).ravel()
    expected = np.zeros_like(before)
    expected[15, -1] = 255
    # No selected histogram mass lies between 32 and 224; both intermediate levels map to zero.
    expected[-1, -2:] = 255
    result = run_equalize(image, "functional", mode="cv", mask=mask)
    np.testing.assert_array_equal(result[..., 0][selection], selected_expected)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("constant", [False, True])
def test_equalize_cv_empty_or_constant_histogram(constant: bool) -> None:
    image = np.full((16, 16, 1), 137, dtype=np.uint8) if constant else issue_image(1)
    before = image.copy()
    mask = np.full((16, 16, 1), int(constant), dtype=np.uint8)
    result = run_equalize(image, "functional", mode="cv", mask=mask)
    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("mode", ["pil", "cv"])
@pytest.mark.parametrize("layout", ["channel-view", "spatial-strided", "read-only"])
def test_equalize_owns_active_lut_output(mode: Literal["pil", "cv"], layout: str) -> None:
    backing = np.random.default_rng(137).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image = backing[..., 1:2]
    if layout == "spatial-strided":
        image = image[::2, ::2]
    if layout == "read-only":
        image.setflags(write=False)
    before = backing.copy()
    mask = np.ones((*image.shape[:2], 1), dtype=np.uint8)
    mask.setflags(write=False)
    mask_before = mask.copy()
    expected = pil_equalize(image.copy(), mask) if mode == "pil" else cv_equalize(image.copy())
    result = run_equalize(image, "functional", mode=mode, mask=mask)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(backing, before)
    np.testing.assert_array_equal(mask, mask_before)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert result.flags.writeable
    assert not np.shares_memory(result, backing)


@pytest.mark.parametrize("masked", [False, True])
def test_equalize_pil_float_conversion(masked: bool) -> None:
    image = np.random.default_rng(137).random((37, 53, 3), dtype=np.float32)
    before = image.copy()
    mask = np.zeros((37, 53, 1), dtype=np.uint8) if masked else None
    if mask is not None:
        mask[:20] = 1
    expected = to_float(pil_equalize(from_float(before, target_dtype=np.uint8), mask))
    result = run_equalize(image, "compose", mask=mask)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
    assert result.dtype == np.float32
    assert result.min() >= 0 and result.max() <= 1


@pytest.mark.parametrize("masked", [False, True])
def test_equalize_pil_luminance(masked: bool) -> None:
    image = np.random.default_rng(137).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    before = image.copy()
    mask = np.zeros((37, 53, 1), dtype=np.uint8) if masked else None
    if mask is not None:
        mask[:20] = 1
    expected = cv2.cvtColor(before, cv2.COLOR_RGB2YCrCb)
    expected[..., :1] = pil_equalize(expected[..., :1].copy(), mask)
    expected = cv2.cvtColor(expected, cv2.COLOR_YCrCb2RGB)
    result = A.Compose([A.Equalize(mode="pil", by_channels=False, mask=mask, p=1)])(image=image)["image"]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("mode", ["pil", "cv"])
@pytest.mark.parametrize(
    "target,leading_shape",
    [
        ("images", (2,)),
        ("images", (1,)),
        ("images", (0,)),
        ("volume", (2,)),
        ("volume", (1,)),
        ("volume", (0,)),
        ("volumes", (2, 2)),
        ("volumes", (1, 1)),
        ("volumes", (0, 2)),
        ("volumes", (2, 0)),
    ],
)
def test_equalize_collections_use_independent_histograms(
    mode: Literal["pil", "cv"],
    target: str,
    leading_shape: tuple[int, ...],
) -> None:
    rng = np.random.default_rng(137)
    image = rng.integers(0, 64, (*leading_shape, 32, 32, 1), dtype=np.uint8)
    for index, item in enumerate(image.reshape(-1, 32, 32, 1)):
        item[:] += index * 40
    before = image.copy()
    mask = np.ones((32, 32, 1), dtype=np.uint8)
    expected = np.empty_like(before)
    for index in np.ndindex(leading_shape):
        expected[index] = pil_equalize(before[index], mask) if mode == "pil" else cv_equalize(before[index])
    result = A.Compose([A.Equalize(mode=mode, mask=mask, p=1)])(**{target: image})[target]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_equalize_static_mask_constructor_and_replay() -> None:
    image, mask = partial_mask_fixture()
    before, mask_before = image.copy(), mask.copy()
    expected = pil_equalize(before, mask_before)
    transform = A.Compose([A.Equalize(mode="pil", mask=mask, p=1)])
    restored = A.from_dict(json.loads(json.dumps(A.to_dict(transform))))
    np.testing.assert_array_equal(restored(image=image)["image"], expected)
    np.testing.assert_array_equal(image, before)
    recorded = A.ReplayCompose([A.Equalize(mode="pil", mask=mask, p=1)])(image=image)
    np.testing.assert_array_equal(recorded["image"], expected)
    fresh = before.copy()
    replayed = A.ReplayCompose.replay(recorded["replay"], image=fresh)
    np.testing.assert_array_equal(replayed["image"], expected)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(fresh, before)
    np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("callable_mask", [False, True])
def test_equalize_applied_configuration_is_unmasked(callable_mask: bool) -> None:
    image, mask = partial_mask_fixture()
    before, mask_before = image.copy(), mask.copy()
    masked_expected = pil_equalize(before, mask_before)
    unmasked_expected = pil_equalize(before)
    assert np.any(masked_expected != unmasked_expected)

    def select_mask(image: np.ndarray, selection: np.ndarray) -> np.ndarray:
        np.testing.assert_array_equal(image, before)
        return selection

    transform = A.Equalize(
        mode="pil",
        mask=select_mask if callable_mask else mask,
        mask_params=("selection",) if callable_mask else (),
        p=1,
    )
    result = A.Compose([transform], save_applied_params=True)(image=image, selection=mask)
    np.testing.assert_array_equal(result["image"], masked_expected)
    np.testing.assert_array_equal(image, before)
    records = json.loads(json.dumps(result["applied_transforms"]))
    replay = A.Compose.from_applied_transforms(records)
    fresh = before.copy()
    np.testing.assert_array_equal(replay(image=fresh)["image"], unmasked_expected)
    np.testing.assert_array_equal(fresh, before)
    np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("target", ["image", "images", "volume"])
def test_equalize_cpu_tensor_fallback(target: str) -> None:
    image = issue_image(1)
    expected = pil_equalize(image)
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
    oracle = torch.from_numpy(expected.transpose(2, 0, 1).copy())
    if target != "image":
        axis = 0 if target == "images" else 1
        tensor = torch.stack([tensor, tensor], dim=axis)
        oracle = torch.stack([oracle, oracle], dim=axis)
    before = tensor.clone()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = A.Compose([A.Equalize(mode="pil", p=1)])(**{target: tensor})[target]
    assert not any("not writable" in str(warning.message) for warning in caught)
    assert isinstance(result, torch.Tensor)
    assert result.shape == tensor.shape
    assert result.dtype == tensor.dtype
    torch.testing.assert_close(result, oracle, rtol=0, atol=0)
    torch.testing.assert_close(tensor, before, rtol=0, atol=0)


@pytest.mark.parametrize("channels", [2, 4])
def test_equalize_transform_rejects_unsupported_channels(channels: int) -> None:
    with pytest.raises(ValueError, match="only supported for RGB and grayscale"):
        run_equalize(issue_image(channels), "compose")


def test_equalize_functional_five_channels() -> None:
    image = np.random.default_rng(137).integers(0, 256, (37, 53, 5), dtype=np.uint8)
    before = image.copy()
    expected = pil_equalize(before)
    result = run_equalize(image, "functional")
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
