"""Tests for apply_to_mask(s) methods in crop and flip transforms."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import albumentations as A
from albumentations.core.transform_params import SampledParams
from albumentations.core.type_definitions import Targets
from tests.helpers.transform_cases import TRANSFORM_CONTRACT_CASES, TransformContractCase


def _inherits_dual_transform_mask3d(case: TransformContractCase) -> bool:
    if issubclass(case.transform_cls, A.Transform3D):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    raw_targets = transform._targets
    targets = raw_targets if isinstance(raw_targets, tuple) else (raw_targets,)
    return Targets.MASK3D in targets and case.transform_cls.apply_to_mask3d is A.DualTransform.apply_to_mask3d


INHERITED_MASK3D_CASES = tuple(case for case in TRANSFORM_CONTRACT_CASES if _inherits_dual_transform_mask3d(case))


def _make_mask3d(depth: int, height: int, width: int, channels: int, dtype: np.dtype[Any]) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[None, :, None, None]
    columns = np.arange(width, dtype=np.float32)[None, None, :, None]
    depths = np.arange(depth, dtype=np.float32)[:, None, None, None]
    channel_offsets = np.arange(channels, dtype=np.float32)[None, None, None, :]
    values = (17 * depths + 3 * rows + 5 * columns + 29 * channel_offsets) % 251
    if np.issubdtype(dtype, np.floating):
        return (values / 250).astype(dtype)
    return values.astype(dtype)


def _make_registered_mask3d_data(case: TransformContractCase, seed: int = 137) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mask3d = _make_mask3d(2, 48, 64, 3, np.dtype(np.uint8))
    volume = np.flip(mask3d, axis=2).copy()

    def data_factory(unused_rng: np.random.Generator) -> dict[str, Any]:
        data: dict[str, Any] = {"volume": volume, "mask3d": mask3d}
        if "mask" in case.required_targets:
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[5:21, 7:29] = 1
            mask[27:43, 35:59] = 2
            data["mask"] = mask
        return data

    return case.make_data(rng, data_factory)


def _resolved_mask3d_params(transform: A.BasicTransform) -> Mapping[str, Any]:
    return SampledParams.from_dict(transform.get_applied_params()).params_for("mask3d")


def _assert_public_mask3d_matches_per_depth(
    transform: A.DualTransform,
    mask3d: np.ndarray,
    *,
    data: dict[str, Any] | None = None,
    compose_kwargs: Mapping[str, Any] | None = None,
    seed: int = 137,
) -> np.ndarray:
    source = {"mask3d": mask3d} if data is None else data
    pipeline = A.Compose(
        [transform],
        save_applied_params=True,
        strict=True,
        seed=seed,
        **copy.deepcopy(dict(compose_kwargs or {})),
    )
    result = pipeline(**source)["mask3d"]
    params = _resolved_mask3d_params(transform)
    normalized_mask3d = mask3d[..., None] if mask3d.ndim == 3 else mask3d
    expected = np.stack([transform.apply_to_mask(mask, **params) for mask in normalized_mask3d])
    if mask3d.ndim == 3:
        expected = expected[..., 0]
    np.testing.assert_array_equal(result, expected)
    return result


@pytest.mark.parametrize("case", INHERITED_MASK3D_CASES, ids=lambda case: case.case_id)
def test_inherited_mask3d_registered_modes_match_per_depth(case: TransformContractCase):
    """Every registered inherited mode uses one resolved parameter set independently per depth slice."""
    data = _make_registered_mask3d_data(case)
    source_snapshot = copy.deepcopy(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    result = _assert_public_mask3d_matches_per_depth(
        transform,
        data["mask3d"],
        data=data,
        compose_kwargs=case.primary_compose_kwargs,
    )
    np.testing.assert_array_equal(data["mask3d"], source_snapshot["mask3d"])
    assert result.dtype == data["mask3d"].dtype
    assert result.flags.writeable


def _border_transform(transform_cls: type[A.DualTransform], fill_mask: float | tuple[float, ...]) -> A.DualTransform:
    common = {"fill_mask": fill_mask, "p": 1}
    if transform_cls is A.Pad:
        transform = A.Pad(padding=4, **common)
    elif transform_cls is A.PadIfNeeded:
        transform = A.PadIfNeeded(min_height=56, min_width=72, **common)
    elif transform_cls is A.LetterBox:
        transform = A.LetterBox(size=(56, 72), **common)
    elif transform_cls is A.Rotate:
        transform = A.Rotate(angle_range=(17, 17), **common)
    elif transform_cls is A.SafeRotate:
        transform = A.SafeRotate(angle_range=(17, 17), **common)
    elif transform_cls is A.ShiftScaleRotate:
        transform = A.ShiftScaleRotate(shift_range=(0.1, 0.1), scale_range=(0, 0), rotate_range=(17, 17), **common)
    elif transform_cls is A.Affine:
        transform = A.Affine(
            scale=(1, 1),
            translate_px={"x": (3, 3), "y": (2, 2)},
            rotate=(17, 17),
            **common,
        )
    elif transform_cls is A.Perspective:
        transform = A.Perspective(scale=(0.15, 0.15), **common)
    else:
        raise AssertionError(transform_cls)
    return transform


@pytest.mark.parametrize(
    "transform_cls",
    [A.Pad, A.PadIfNeeded, A.LetterBox, A.Rotate, A.SafeRotate, A.ShiftScaleRotate, A.Affine, A.Perspective],
)
@pytest.mark.parametrize("fill_mask", [11, (11, 13, 17)], ids=["scalar", "per-channel"])
def test_inherited_mask3d_border_values_match_per_depth(transform_cls, fill_mask):
    mask3d = _make_mask3d(2, 48, 64, 3, np.dtype(np.uint8))
    _assert_public_mask3d_matches_per_depth(_border_transform(transform_cls, fill_mask), mask3d)


@pytest.mark.parametrize("per_channel", [False, True], ids=["shared", "per-channel"])
@pytest.mark.parametrize("channels", [3, 5])
def test_pixel_dropout_mask3d_matches_per_depth(per_channel: bool, channels: int):
    mask3d = _make_mask3d(2, 48, 64, channels, np.dtype(np.uint8))
    transform = A.PixelDropout(
        dropout_prob=0.25,
        per_channel=per_channel,
        drop_value=2,
        mask_drop_value=tuple(range(11, 11 + channels)),
        p=1,
    )
    _assert_public_mask3d_matches_per_depth(transform, mask3d)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_inherited_mask3d_representation_matrix(depth, channels, dtype):
    mask3d = _make_mask3d(depth, 31, 47, channels, np.dtype(dtype))
    result = _assert_public_mask3d_matches_per_depth(A.Morphological(scale=(3, 3), p=1), mask3d)
    assert result.shape == mask3d.shape
    assert result.dtype == mask3d.dtype
    assert result.min() >= mask3d.min()
    assert result.max() <= mask3d.max()


def test_pixel_dropout_grayscale_mask3d_matches_normalized_per_depth():
    mask3d = _make_mask3d(2, 31, 47, 1, np.dtype(np.uint8))[..., 0]
    transform = A.PixelDropout(
        dropout_prob=0.25,
        per_channel=True,
        drop_value=2,
        mask_drop_value=11,
        p=1,
    )
    result = _assert_public_mask3d_matches_per_depth(transform, mask3d)
    assert result.shape == mask3d.shape


@pytest.mark.parametrize(
    "transform,expected_shape",
    [
        (A.Resize(24, 32, p=1), (0, 24, 32, 3)),
        (A.Pad(4, p=1), (0, 56, 72, 3)),
        (A.LetterBox((24, 32), p=1), (0, 24, 32, 3)),
        (A.Morphological(scale=(3, 3), p=1), (0, 48, 64, 3)),
    ],
    ids=["resize", "pad", "letter-box", "morphological"],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_inherited_empty_volume_and_mask3d_shapes_stay_aligned(transform, expected_shape, dtype):
    volume = np.empty((0, 48, 64, 3), dtype=dtype)
    mask3d = np.empty((0, 48, 64, 3), dtype=dtype)
    result = A.Compose([transform], seed=137, strict=True)(volume=volume, mask3d=mask3d)
    assert result["volume"].shape == expected_shape
    assert result["mask3d"].shape == expected_shape
    assert result["volume"].dtype == volume.dtype
    assert result["mask3d"].dtype == mask3d.dtype
    assert result["volume"].flags.writeable
    assert result["mask3d"].flags.writeable


@pytest.mark.parametrize("case", INHERITED_MASK3D_CASES, ids=lambda case: case.case_id)
def test_inherited_empty_mask3d_matches_nonempty_item_geometry(case: TransformContractCase):
    results = []
    for empty in (False, True):
        data = _make_registered_mask3d_data(case)
        if empty:
            data["volume"] = np.empty((0, 48, 64, 3), dtype=np.uint8)
            data["mask3d"] = np.empty((0, 48, 64, 3), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
        pipeline = A.Compose(
            [transform],
            seed=137,
            strict=True,
            **copy.deepcopy(dict(case.primary_compose_kwargs)),
        )
        results.append(pipeline(**data))

    nonempty_result, empty_result = results
    assert empty_result["volume"].shape == (0, *nonempty_result["volume"].shape[1:])
    assert empty_result["mask3d"].shape == (0, *nonempty_result["mask3d"].shape[1:])


def test_inherited_mask3d_route_preserves_custom_empty_metadata():
    class CropAndCastMask(A.DualTransform):
        def apply(self, img, **params):
            return img

        def apply_to_mask(self, mask, **params):
            return mask[:5, :7].astype(np.float32)

    def apply(mask3d):
        transform = A.Compose([CropAndCastMask(p=1)], seed=137, strict=True)
        return transform(mask3d=mask3d)["mask3d"]

    one_item_result = apply(np.zeros((1, 10, 12, 3), dtype=np.uint8))
    empty_result = apply(np.empty((0, 10, 12, 3), dtype=np.uint8))

    assert empty_result.shape == (0, *one_item_result.shape[1:])
    assert empty_result.dtype == one_item_result.dtype


def test_custom_empty_mask_metadata_hook_avoids_item_materialization():
    class CropAndCastMask(A.DualTransform):
        def apply(self, img, **params):
            raise AssertionError("declared empty-batch metadata must bypass the image transform")

        def apply_to_mask(self, mask, **params):
            raise AssertionError("declared empty-batch metadata must bypass the mask transform")

        def _get_empty_batch_item_metadata(self, item_shape, item_dtype, target_name, params):
            assert target_name == "mask"
            return (5, 7, *item_shape[2:]), np.dtype(np.float32)

    mask3d = np.empty((0, 10, 12, 3), dtype=np.uint8)
    transform = A.Compose([CropAndCastMask(p=1)], seed=137, strict=True)
    result = transform(mask3d=mask3d)["mask3d"]

    assert result.shape == (0, 5, 7, 3)
    assert result.dtype == np.float32


def test_inherited_empty_mask3d_output_is_writeable_for_readonly_input():
    mask3d = np.empty((0, 10, 12, 3), dtype=np.uint8)
    mask3d.setflags(write=False)
    transform = A.Compose([A.Morphological(scale=(3, 3), p=1)], seed=137, strict=True)
    result = transform(mask3d=mask3d)["mask3d"]

    assert result.flags.writeable
    assert not np.shares_memory(result, mask3d)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("read_only", [False, True], ids=["writeable-input", "read-only-input"])
def test_inherited_mask3d_output_is_owned_and_writeable(depth: int, read_only: bool):
    mask3d = _make_mask3d(depth, 31, 47, 3, np.dtype(np.uint8))
    if read_only:
        mask3d.setflags(write=False)
    snapshot = mask3d.copy()
    transform = A.PixelDropout(dropout_prob=0.25, mask_drop_value=None, p=1)
    result = _assert_public_mask3d_matches_per_depth(transform, mask3d)
    np.testing.assert_array_equal(mask3d, snapshot)
    assert result.flags.writeable
    assert not np.shares_memory(result, mask3d)


def test_singleton_mask3d_reuses_allocated_slice_result(monkeypatch):
    mask3d = _make_mask3d(1, 31, 47, 3, np.dtype(np.uint8))
    transform = A.Morphological(scale=(3, 3), p=1)
    apply_to_mask = transform.apply_to_mask
    slice_results = []

    def tracked_apply_to_mask(mask, *args, **params):
        result = apply_to_mask(mask, *args, **params)
        slice_results.append(result)
        return result

    monkeypatch.setattr(transform, "apply_to_mask", tracked_apply_to_mask)
    result = A.Compose([transform], seed=137, strict=True)(mask3d=mask3d)["mask3d"]

    assert len(slice_results) == 1
    assert np.shares_memory(result, slice_results[0])
    assert not np.shares_memory(result, mask3d)
    assert result.flags.writeable


def test_inherited_mask3d_noncontiguous_input_is_not_mutated():
    base = _make_mask3d(2, 62, 94, 3, np.dtype(np.uint8))
    mask3d = base[:, ::2, ::2]
    assert not mask3d.flags.c_contiguous
    snapshot = mask3d.copy()
    result = _assert_public_mask3d_matches_per_depth(A.Morphological(scale=(3, 3), p=1), mask3d)
    np.testing.assert_array_equal(mask3d, snapshot)
    assert result.flags.writeable
    assert not np.shares_memory(result, mask3d)


# Test crops with single mask (empty and non-empty)
@pytest.mark.parametrize(
    "transform_class,init_params,expected_shape",
    [
        (A.RandomCrop, {"height": 50, "width": 60}, (50, 60)),
        (A.CenterCrop, {"height": 40, "width": 70}, (40, 70)),
        (A.Crop, {"x_min": 10, "y_min": 15, "x_max": 70, "y_max": 55}, (40, 60)),
    ],
)
@pytest.mark.parametrize("mask_shape", [(100, 120), (100, 120, 3)])
def test_crop_apply_to_mask_single(transform_class, init_params, expected_shape, mask_shape):
    """Test that apply_to_mask works correctly for crops with non-square dimensions."""
    transform = transform_class(**init_params, p=1.0)
    mask = np.random.randint(0, 2, mask_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 120, 3), dtype=np.uint8), mask=mask)

    # Check that mask was cropped
    assert result["mask"].shape[:2] == expected_shape
    if len(mask_shape) == 3:
        assert result["mask"].shape[2] == mask_shape[2]


def test_crop_apply_to_mask_empty_channels():
    """Test that apply_to_mask handles empty channel dimension correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create mask with 0 channels (empty)
    mask = np.empty((100, 100, 0), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask=mask)

    # Check that mask was cropped with correct shape
    assert result["mask"].shape == (50, 50, 0)
    assert result["mask"].dtype == np.uint8


# Test crops with masks batch (empty and non-empty) - using non-square dimensions
@pytest.mark.parametrize(
    "transform_class,init_params,expected_shape",
    [
        (A.RandomCrop, {"height": 50, "width": 60}, (50, 60)),
        (A.CenterCrop, {"height": 40, "width": 70}, (40, 70)),
        (A.Crop, {"x_min": 10, "y_min": 15, "x_max": 70, "y_max": 55}, (40, 60)),
    ],
)
@pytest.mark.parametrize("num_masks", [1, 3, 5])
@pytest.mark.parametrize("channels", [None, 1, 3])
def test_crop_apply_to_masks_batch(transform_class, init_params, expected_shape, num_masks, channels):
    """Test that apply_to_masks works correctly for batch processing with non-square dimensions."""
    transform = transform_class(**init_params, p=1.0)

    if channels is None:
        masks_shape = (num_masks, 100, 120)
    else:
        masks_shape = (num_masks, 100, 120, channels)

    masks = np.random.randint(0, 2, masks_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 120, 3), dtype=np.uint8), masks=masks)

    # Check that all masks were cropped
    assert result["masks"].shape[0] == num_masks
    assert result["masks"].shape[1:3] == expected_shape
    if channels is not None:
        assert result["masks"].shape[3] == channels


def test_crop_apply_to_masks_empty_batch():
    """Test that apply_to_masks handles empty batch correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create empty batch of masks
    masks = np.empty((0, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks=masks)

    # Check that empty batch returns correct cropped dimensions
    assert result["masks"].shape == (0, 50, 50)
    assert result["masks"].dtype == np.uint8


def test_crop_apply_to_masks_empty_batch_with_channels():
    """Test that apply_to_masks handles empty batch with channels correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create empty batch of masks with channels
    masks = np.empty((0, 100, 100, 3), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks=masks)

    # Check that empty batch returns correct cropped dimensions
    assert result["masks"].shape == (0, 50, 50, 3)
    assert result["masks"].dtype == np.uint8


# Test flips with single mask (empty and non-empty) - using non-square images
@pytest.mark.parametrize("transform_class", [A.HorizontalFlip, A.VerticalFlip, A.Transpose])
@pytest.mark.parametrize("mask_shape", [(80, 120), (80, 120, 3)])
def test_flip_apply_to_mask_single(transform_class, mask_shape):
    """Test that apply_to_mask works correctly for flips with non-square images."""
    transform = transform_class(p=1.0)
    mask = np.random.randint(0, 2, mask_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask)

    # Check that mask shape is handled correctly
    if transform_class == A.Transpose:
        # Transpose swaps H and W: (80, 120) -> (120, 80)
        assert result["mask"].shape[0] == mask_shape[1]
        assert result["mask"].shape[1] == mask_shape[0]
    else:
        # Other flips preserve dimensions
        assert result["mask"].shape[:2] == mask_shape[:2]

    if len(mask_shape) == 3:
        assert result["mask"].shape[2] == mask_shape[2]


def test_flip_apply_to_mask_empty_channels():
    """Test that HorizontalFlip and VerticalFlip handle empty channel dimension correctly."""
    for transform_class in [A.HorizontalFlip, A.VerticalFlip]:
        transform = transform_class(p=1.0)
        # Create mask with 0 channels (empty)
        mask = np.empty((80, 120, 0), dtype=np.uint8)

        # Apply the transform through Compose
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask)

        # Check correct shape (flips preserve spatial dimensions)
        assert result["mask"].shape == (80, 120, 0)
        assert result["mask"].dtype == np.uint8


def test_transpose_empty_mask_swaps_dimensions():
    """Test that Transpose swaps dimensions correctly for empty single mask."""
    transform = A.Transpose(p=1.0)
    aug = A.Compose([transform])

    # Test with 3D mask (H, W, 0) - 0 channels
    mask_empty_channels = np.empty((80, 120, 0), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask=mask_empty_channels)
    # Transpose should swap H and W: (80, 120, 0) -> (120, 80, 0)
    assert result["mask"].shape == (120, 80, 0)
    assert result["mask"].dtype == np.uint8


# Test flips with masks batch (empty and non-empty) - using non-square images
@pytest.mark.parametrize("transform_class", [A.HorizontalFlip, A.VerticalFlip])
@pytest.mark.parametrize("num_masks", [1, 3, 5])
@pytest.mark.parametrize("channels", [None, 1, 3])
def test_flip_apply_to_masks_batch(transform_class, num_masks, channels):
    """Test that apply_to_masks works correctly for batch processing with non-square images."""
    transform = transform_class(p=1.0)

    if channels is None:
        masks_shape = (num_masks, 80, 120)
    else:
        masks_shape = (num_masks, 80, 120, channels)

    masks = np.random.randint(0, 2, masks_shape, dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks)

    # Check that all masks were processed
    assert result["masks"].shape[0] == num_masks

    # Check spatial dimensions (non-transpose flips preserve dimensions)
    assert result["masks"].shape[1:3] == (80, 120)

    if channels is not None:
        assert result["masks"].shape[3] == channels


def test_flip_apply_to_masks_empty_batch():
    """Test that HorizontalFlip and VerticalFlip handle empty batch correctly."""
    for transform_class in [A.HorizontalFlip, A.VerticalFlip]:
        transform = transform_class(p=1.0)
        # Create empty batch of masks
        masks = np.empty((0, 80, 120), dtype=np.uint8)

        # Apply the transform through Compose
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks)

        # Check that empty batch preserves dimensions
        assert result["masks"].shape == (0, 80, 120)
        assert result["masks"].dtype == np.uint8


def test_transpose_empty_masks_swaps_dimensions():
    """Test that Transpose swaps dimensions correctly for empty masks."""
    transform = A.Transpose(p=1.0)

    # Test with 2D masks (N, H, W)
    masks_2d = np.empty((0, 80, 120), dtype=np.uint8)
    aug = A.Compose([transform])
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks_2d)
    # Transpose should swap H and W: (0, 80, 120) -> (0, 120, 80)
    assert result["masks"].shape == (0, 120, 80)
    assert result["masks"].dtype == np.uint8

    # Test with 3D masks (N, H, W, C)
    masks_3d = np.empty((0, 80, 120, 3), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), masks=masks_3d)
    # Transpose should swap H and W: (0, 80, 120, 3) -> (0, 120, 80, 3)
    assert result["masks"].shape == (0, 120, 80, 3)
    assert result["masks"].dtype == np.uint8


@pytest.mark.parametrize(
    "group_element,expected_swap",
    [
        ("e", False),  # identity - no swap
        ("r90", True),  # rotation 90 - swaps dimensions for non-square images
        ("r180", False),  # rotation 180 - no swap
        ("r270", True),  # rotation 270 - swaps dimensions for non-square images
        ("v", False),  # vertical flip - no swap
        ("h", False),  # horizontal flip - no swap
        ("t", True),  # transpose - swaps dimensions
        ("hvt", True),  # anti-diagonal transpose - swaps dimensions
    ],
)
def test_d4_empty_mask_dimension_handling(group_element, expected_swap):
    """Test that D4 correctly handles dimension swapping for empty masks based on group element."""
    # Create a D4 transform but we'll call apply_to_mask directly with specific group element
    transform = A.D4(p=1.0)

    # Test with single empty mask (H, W, C) - using non-square image
    mask = np.empty((80, 120, 3), dtype=np.uint8)
    result_mask = transform.apply_to_mask(mask, group_element=group_element)

    if expected_swap:
        # Group elements that swap dimensions: r90, r270, t, hvt
        # Should swap H and W: (80, 120, 3) -> (120, 80, 3)
        assert result_mask.shape == (120, 80, 3), f"Failed for group_element={group_element}"
    else:
        # Other group elements preserve dimensions: e, r180, v, h
        assert result_mask.shape == (80, 120, 3), f"Failed for group_element={group_element}"
    assert result_mask.dtype == np.uint8


@pytest.mark.parametrize(
    "group_element,expected_swap",
    [
        ("e", False),
        ("r90", True),
        ("r180", False),
        ("r270", True),
        ("v", False),
        ("h", False),
        ("t", True),
        ("hvt", True),
    ],
)
def test_d4_empty_masks_batch_dimension_handling(group_element, expected_swap):
    """Test that D4 correctly handles dimension swapping for empty mask batches."""
    transform = A.D4(p=1.0)

    # Test with batch of empty masks (N, H, W, C)
    masks = np.empty((0, 80, 120, 3), dtype=np.uint8)
    result_masks = transform.apply_to_masks(masks, group_element=group_element)

    if expected_swap:
        # Group elements that swap dimensions: r90, r270, t, hvt
        # Should swap H and W: (0, 80, 120, 3) -> (0, 120, 80, 3)
        assert result_masks.shape == (0, 120, 80, 3), f"Failed for group_element={group_element}"
    else:
        # Other group elements preserve dimensions: e, r180, v, h
        assert result_masks.shape == (0, 80, 120, 3), f"Failed for group_element={group_element}"
    assert result_masks.dtype == np.uint8


@pytest.mark.parametrize(
    "group_element,expected_swap",
    [
        ("e", False),
        ("r90", True),
        ("r270", True),
        ("t", True),
        ("hvt", True),
    ],
)
def test_d4_empty_mask3d_dimension_handling(group_element, expected_swap):
    """Test that D4 correctly handles dimension swapping for empty mask3d."""
    transform = A.D4(p=1.0)

    # Test with empty mask3d (D, H, W, C)
    mask3d = np.empty((10, 80, 120, 3), dtype=np.uint8)
    result_mask3d = transform.apply_to_mask3d(mask3d, group_element=group_element)

    if expected_swap:
        # Group elements that swap dimensions: r90, r270, t, hvt
        # Should swap H and W: (10, 80, 120, 3) -> (10, 120, 80, 3)
        assert result_mask3d.shape == (10, 120, 80, 3), f"Failed for group_element={group_element}"
    else:
        # Other group elements preserve dimensions: e, r180, v, h
        assert result_mask3d.shape == (10, 80, 120, 3), f"Failed for group_element={group_element}"
    assert result_mask3d.dtype == np.uint8


def test_crop_apply_to_mask3d_empty():
    """Test that apply_to_mask3d handles empty mask correctly."""
    transform = A.Crop(x_min=10, y_min=10, x_max=60, y_max=60, p=1.0)
    # Create empty mask3d (0 depth)
    mask3d = np.empty((0, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask3d=mask3d)

    # Check correct shape
    assert result["mask3d"].shape == (0, 50, 50)
    assert result["mask3d"].dtype == np.uint8


# Test flips with mask3d
@pytest.mark.parametrize("transform_class", [A.HorizontalFlip, A.VerticalFlip, A.Transpose, A.D4])
def test_flip_apply_to_mask3d(transform_class):
    """Test that apply_to_mask3d works correctly for flips."""
    transform = transform_class(p=1.0)
    # mask3d has shape (D, H, W)
    mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)

    # Apply the transform through Compose
    aug = A.Compose([transform])
    result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask3d=mask3d)

    # Check shape (depth preserved)
    if transform_class == A.Transpose:
        assert result["mask3d"].shape == (10, 100, 100)  # Square, so same after transpose
    else:
        assert result["mask3d"].shape == (10, 100, 100)


def test_flip_apply_to_mask3d_empty():
    """Test that HorizontalFlip and VerticalFlip handle empty mask3d correctly."""
    for transform_class in [A.HorizontalFlip, A.VerticalFlip]:
        transform = transform_class(p=1.0)
        # Create empty mask3d (0 depth)
        mask3d = np.empty((0, 80, 120), dtype=np.uint8)

        # Apply the transform through Compose
        aug = A.Compose([transform])
        result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask3d=mask3d)

        # Check correct shape
        assert result["mask3d"].shape == (0, 80, 120)
        assert result["mask3d"].dtype == np.uint8


def test_transpose_empty_mask3d_swaps_dimensions():
    """Test that Transpose swaps dimensions correctly for empty mask3d."""
    transform = A.Transpose(p=1.0)
    aug = A.Compose([transform])

    # Test with 3D mask3d (D, H, W) where D=0
    mask3d_empty_depth = np.empty((0, 80, 120), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask3d=mask3d_empty_depth)
    # Transpose should swap H and W: (0, 80, 120) -> (0, 120, 80)
    assert result["mask3d"].shape == (0, 120, 80)
    assert result["mask3d"].dtype == np.uint8

    # Test with 4D mask3d (D, H, W, C) where D=0
    mask3d_empty_depth_4d = np.empty((0, 80, 120, 3), dtype=np.uint8)
    result = aug(image=np.zeros((80, 120, 3), dtype=np.uint8), mask3d=mask3d_empty_depth_4d)
    # Transpose should swap H and W: (0, 80, 120, 3) -> (0, 120, 80, 3)
    assert result["mask3d"].shape == (0, 120, 80, 3)
    assert result["mask3d"].dtype == np.uint8


# Test that batch processing is actually faster than loop (integration test)
@pytest.mark.parametrize("transform_class,init_params", [(A.CenterCrop, {"height": 50, "width": 50})])
def test_crop_masks_batch_vs_loop(transform_class, init_params):
    """Test that batch processing gives same results as loop processing."""
    transform = transform_class(**init_params, p=1.0)
    num_masks = 5
    masks = np.random.randint(0, 2, (num_masks, 100, 100), dtype=np.uint8)

    # Apply using batch method
    aug = A.Compose([transform])
    result_batch = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), masks=masks)

    # Apply using loop (simulating old behavior)
    result_loop_masks = []
    for i in range(num_masks):
        mask_result = aug(image=np.zeros((100, 100, 3), dtype=np.uint8), mask=masks[i])
        result_loop_masks.append(mask_result["mask"])
    result_loop = np.stack(result_loop_masks)

    # Check that results are identical
    np.testing.assert_array_equal(result_batch["masks"], result_loop)
