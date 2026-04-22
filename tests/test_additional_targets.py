"""Tests for additional_targets alias resolution across shape/metadata helpers and
end-to-end transform pipelines.

Covers the family of helpers that historically read data by hardcoded keys
(`image`, `images`, `mask`, ...) and ignored `_additional_targets`:

- `albumentations.core.utils.get_image_data`
- `albumentations.core.utils.get_shape`
- `albumentations.core.utils.get_volume_shape`
- `BasicTransform._extract_shape_from_data`
- `BasicTransform.get_image_data`
- `Compose._gather_shapes_from_data`
- `Compose._add_grayscale_channels` / `_remove_grayscale_channels`
- `check_data_post_transform` (via `get_shape`)

Plus integration tests reproducing the user-reported bug (KeyError 'shape',
"No valid image/volume data found in data dict") for every transform that needs
shape metadata at param-sampling time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import albumentations as A
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.utils import get_image_data, get_shape, get_volume_shape

# ---------------------------------------------------------------------------
# get_image_data
# ---------------------------------------------------------------------------


class TestGetImageData:
    def test_canonical_image(self) -> None:
        data = {"image": np.zeros((10, 20, 3), dtype=np.uint8)}
        meta = get_image_data(data)
        assert meta == {"dtype": np.uint8, "height": 10, "width": 20, "num_channels": 3}

    def test_canonical_images(self) -> None:
        data = {"images": np.zeros((4, 10, 20, 3), dtype=np.float32)}
        meta = get_image_data(data)
        assert meta["height"] == 10
        assert meta["width"] == 20
        assert meta["num_channels"] == 3
        assert meta["dtype"] == np.float32

    def test_canonical_volume(self) -> None:
        data = {"volume": np.zeros((5, 10, 20, 3), dtype=np.uint16)}
        meta = get_image_data(data)
        assert meta["height"] == 10
        assert meta["width"] == 20
        assert meta["num_channels"] == 3
        assert meta["dtype"] == np.uint16

    def test_canonical_volumes(self) -> None:
        data = {"volumes": np.zeros((2, 5, 10, 20, 3), dtype=np.uint8)}
        meta = get_image_data(data)
        assert meta["height"] == 10
        assert meta["width"] == 20
        assert meta["num_channels"] == 3

    def test_aliased_image(self) -> None:
        data = {"custom_image_key": np.zeros((10, 20, 3), dtype=np.uint8)}
        meta = get_image_data(data, {"custom_image_key": "image"})
        assert meta == {"dtype": np.uint8, "height": 10, "width": 20, "num_channels": 3}

    def test_aliased_images(self) -> None:
        data = {"my_imgs": np.zeros((4, 10, 20, 3), dtype=np.uint8)}
        meta = get_image_data(data, {"my_imgs": "images"})
        assert meta["height"] == 10 and meta["width"] == 20

    def test_aliased_volume(self) -> None:
        data = {"my_vol": np.zeros((5, 10, 20, 3), dtype=np.uint8)}
        meta = get_image_data(data, {"my_vol": "volume"})
        assert meta["height"] == 10 and meta["width"] == 20

    def test_aliased_volumes(self) -> None:
        data = {"my_vols": np.zeros((2, 5, 10, 20, 3), dtype=np.uint8)}
        meta = get_image_data(data, {"my_vols": "volumes"})
        assert meta["height"] == 10 and meta["width"] == 20

    def test_priority_canonical_wins_over_alias(self) -> None:
        """When both canonical and alias resolve to 'image', the canonical key is used."""
        canonical = np.zeros((10, 20, 3), dtype=np.uint8)
        aliased = np.zeros((50, 60, 1), dtype=np.uint8)
        data = {"image": canonical, "custom_image_key": aliased}
        meta = get_image_data(data, {"custom_image_key": "image"})
        assert meta["height"] == 10 and meta["width"] == 20

    def test_priority_image_over_images(self) -> None:
        data = {
            "images": np.zeros((4, 50, 60, 3), dtype=np.uint8),
            "image": np.zeros((10, 20, 3), dtype=np.uint8),
        }
        meta = get_image_data(data)
        assert meta["height"] == 10 and meta["width"] == 20

    def test_aliased_image_when_canonical_missing(self) -> None:
        data = {
            "custom_img": np.zeros((10, 20, 3), dtype=np.uint8),
            "bboxes": np.array([[0, 0, 1, 1]]),
            "labels": [1],
        }
        meta = get_image_data(data, {"custom_img": "image"})
        assert meta["height"] == 10

    def test_ignores_non_image_keys(self) -> None:
        data = {
            "bboxes": np.array([[0, 0, 1, 1]]),
            "keypoints": np.array([[0, 0]]),
            "labels": [1],
        }
        with pytest.raises(ValueError, match="No valid image/volume data"):
            get_image_data(data)

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid image/volume data"):
            get_image_data({})

    def test_none_value_skipped(self) -> None:
        data = {"image": None, "custom_img": np.zeros((10, 20, 3), dtype=np.uint8)}
        meta = get_image_data(data, {"custom_img": "image"})
        assert meta["height"] == 10

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_dtype_preserved(self, dtype: type) -> None:
        data = {"image": np.zeros((4, 5, 3), dtype=dtype)}
        assert get_image_data(data)["dtype"] == np.dtype(dtype)

    def test_grayscale_after_expand(self) -> None:
        data = {"image": np.zeros((10, 20, 1), dtype=np.uint8)}
        meta = get_image_data(data)
        assert meta["num_channels"] == 1


# ---------------------------------------------------------------------------
# get_shape
# ---------------------------------------------------------------------------


class TestGetShape:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"image": np.zeros((10, 20, 3), dtype=np.uint8)}, (10, 20)),
            ({"images": np.zeros((4, 10, 20, 3), dtype=np.uint8)}, (10, 20)),
            ({"volume": np.zeros((5, 10, 20, 3), dtype=np.uint8)}, (10, 20)),
            ({"volumes": np.zeros((2, 5, 10, 20, 3), dtype=np.uint8)}, (10, 20)),
        ],
    )
    def test_canonical_keys(self, data: dict[str, Any], expected: tuple[int, int]) -> None:
        assert get_shape(data) == expected

    @pytest.mark.parametrize(
        ("data", "aliases", "expected"),
        [
            ({"x_img": np.zeros((10, 20, 3), dtype=np.uint8)}, {"x_img": "image"}, (10, 20)),
            ({"x_imgs": np.zeros((4, 10, 20, 3), dtype=np.uint8)}, {"x_imgs": "images"}, (10, 20)),
            ({"x_vol": np.zeros((5, 10, 20, 3), dtype=np.uint8)}, {"x_vol": "volume"}, (10, 20)),
            ({"x_vols": np.zeros((2, 5, 10, 20, 3), dtype=np.uint8)}, {"x_vols": "volumes"}, (10, 20)),
        ],
    )
    def test_aliased_keys(
        self,
        data: dict[str, Any],
        aliases: dict[str, str],
        expected: tuple[int, int],
    ) -> None:
        assert get_shape(data, aliases) == expected

    def test_canonical_wins_over_alias(self) -> None:
        data = {
            "image": np.zeros((10, 20, 3), dtype=np.uint8),
            "x_img": np.zeros((50, 60, 3), dtype=np.uint8),
        }
        assert get_shape(data, {"x_img": "image"}) == (10, 20)

    def test_no_image_raises(self) -> None:
        with pytest.raises(ValueError, match="No image or volume found"):
            get_shape({"bboxes": np.array([[0, 0, 1, 1]])})

    def test_torch_chw_under_alias(self) -> None:
        torch = pytest.importorskip("torch")
        data = {"x_img": torch.zeros(3, 10, 20)}
        assert get_shape(data, {"x_img": "image"}) == (10, 20)


# ---------------------------------------------------------------------------
# get_volume_shape
# ---------------------------------------------------------------------------


class TestGetVolumeShape:
    def test_canonical_volume(self) -> None:
        data = {"volume": np.zeros((5, 10, 20, 3), dtype=np.uint8)}
        assert get_volume_shape(data) == (5, 10, 20)

    def test_canonical_volumes(self) -> None:
        data = {"volumes": np.zeros((2, 5, 10, 20, 3), dtype=np.uint8)}
        assert get_volume_shape(data) == (5, 10, 20)

    def test_aliased_volume(self) -> None:
        data = {"my_vol": np.zeros((5, 10, 20, 3), dtype=np.uint8)}
        assert get_volume_shape(data, {"my_vol": "volume"}) == (5, 10, 20)

    def test_returns_none_when_no_volume(self) -> None:
        data = {"image": np.zeros((10, 20, 3), dtype=np.uint8)}
        assert get_volume_shape(data) is None

    def test_canonical_volume_none_uses_aliased_key(self) -> None:
        """`volume: None` must not shadow a non-`None` value under an alias (regression: PR review)."""
        data = {
            "volume": None,
            "my_vol": np.zeros((5, 10, 20, 3), dtype=np.uint8),
        }
        assert get_volume_shape(data, {"my_vol": "volume"}) == (5, 10, 20)

    def test_canonical_volume_wins_over_alias_when_both_set(self) -> None:
        vol = np.zeros((5, 10, 20, 3), dtype=np.uint8)
        other = np.zeros((2, 3, 4, 3), dtype=np.uint8)
        data = {"my_vol": other, "volume": vol}
        assert get_volume_shape(data, {"my_vol": "volume"}) == (5, 10, 20)


# ---------------------------------------------------------------------------
# BasicTransform._extract_shape_from_data + .get_image_data
# ---------------------------------------------------------------------------


class _DummyDual(A.DualTransform):
    """Minimal dual transform exposing _extract_shape_from_data for tests."""

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        return mask


@pytest.mark.parametrize(
    ("data_key", "shape_in", "expected"),
    [
        ("image", (10, 20, 3), (10, 20, 3)),
        ("mask", (10, 20), (10, 20)),
        ("images", (4, 10, 20, 3), (10, 20, 3)),
        ("masks", (4, 10, 20), (10, 20)),
        ("volume", (5, 10, 20, 3), (10, 20, 3)),
        ("mask3d", (5, 10, 20), (10, 20)),
        ("volumes", (2, 5, 10, 20, 3), (10, 20, 3)),
        ("masks3d", (2, 5, 10, 20), (10, 20)),
    ],
)
def test_extract_shape_aliased(
    data_key: str,
    shape_in: tuple[int, ...],
    expected: tuple[int, ...],
) -> None:
    """Each canonical target keyed under an alias still resolves to the right .shape."""
    transform = _DummyDual(p=1.0)
    alias = f"my_{data_key}"
    transform.add_targets({alias: data_key})
    data = {alias: np.zeros(shape_in, dtype=np.uint8)}
    assert transform._extract_shape_from_data(data) == expected


def test_extract_shape_no_data_returns_none() -> None:
    transform = _DummyDual(p=1.0)
    assert transform._extract_shape_from_data({"bboxes": np.zeros((0, 4))}) is None


def test_extract_shape_canonical_wins_over_alias() -> None:
    transform = _DummyDual(p=1.0)
    transform.add_targets({"img2": "image"})
    data = {
        "image": np.zeros((10, 20, 3), dtype=np.uint8),
        "img2": np.zeros((50, 60, 3), dtype=np.uint8),
    }
    assert transform._extract_shape_from_data(data) == (10, 20, 3)


def test_basictransform_get_image_data_uses_aliases() -> None:
    transform = _DummyDual(p=1.0)
    transform.add_targets({"my_img": "image"})
    data = {"my_img": np.zeros((10, 20, 3), dtype=np.uint8)}
    meta = transform.get_image_data(data)
    assert meta["height"] == 10
    assert meta["width"] == 20
    assert meta["num_channels"] == 3


def test_basictransform_get_image_data_raises_without_image() -> None:
    transform = _DummyDual(p=1.0)
    with pytest.raises(ValueError, match="No valid image/volume data"):
        transform.get_image_data({"bboxes": np.zeros((0, 4))})


# ---------------------------------------------------------------------------
# Compose._gather_shapes_from_data
# ---------------------------------------------------------------------------


def test_gather_shapes_aliased_consistent() -> None:
    compose = A.Compose([A.NoOp()])
    compose.add_targets({"img2": "image", "msk2": "mask"})
    data = {
        "img2": np.zeros((10, 20, 3), dtype=np.uint8),
        "msk2": np.zeros((10, 20), dtype=np.uint8),
    }
    shapes, volume_shapes = compose._gather_shapes_from_data(data)
    assert shapes == [(10, 20), (10, 20)]
    assert volume_shapes == []


def test_gather_shapes_aliased_inconsistent_raises() -> None:
    compose = A.Compose([A.NoOp()], is_check_shapes=True)
    compose.add_targets({"img2": "image", "msk2": "mask"})
    with pytest.raises(ValueError):
        compose(
            img2=np.zeros((10, 20, 3), dtype=np.uint8),
            msk2=np.zeros((50, 60), dtype=np.uint8),
        )


# ---------------------------------------------------------------------------
# Compose grayscale channel handling
# ---------------------------------------------------------------------------


def test_grayscale_aliased_image_expanded_and_stripped() -> None:
    compose = A.Compose([A.HorizontalFlip(p=1.0)])
    compose.add_targets({"img2": "image"})
    out = compose(img2=np.zeros((10, 20), dtype=np.uint8))
    assert out["img2"].shape == (10, 20)


def test_grayscale_aliased_mask_expanded_and_stripped() -> None:
    compose = A.Compose([A.HorizontalFlip(p=1.0)])
    compose.add_targets({"msk2": "mask"})
    out = compose(
        image=np.zeros((10, 20, 3), dtype=np.uint8),
        msk2=np.zeros((10, 20), dtype=np.uint8),
    )
    assert out["msk2"].shape == (10, 20)


def test_grayscale_aliased_image_with_channel_no_strip() -> None:
    """If user already passes (H,W,C), preprocessor must not strip on output."""
    compose = A.Compose([A.HorizontalFlip(p=1.0)])
    compose.add_targets({"img2": "image"})
    out = compose(img2=np.zeros((10, 20, 3), dtype=np.uint8))
    assert out["img2"].shape == (10, 20, 3)


def test_grayscale_aliased_added_channel_dim_tracking() -> None:
    """Internal bookkeeping keys the alias under the user key with canonical map."""
    compose = A.Compose([A.HorizontalFlip(p=1.0)])
    compose.add_targets({"img2": "image"})
    # Run preprocessing only, then inspect bookkeeping
    data = {"img2": np.zeros((10, 20), dtype=np.uint8)}
    compose._preprocess_arrays(data)
    assert compose._added_channel_dim == {"img2": True}
    assert compose._added_channel_canonical == {"img2": "image"}
    assert data["img2"].shape == (10, 20, 1)


# ---------------------------------------------------------------------------
# Integration: user-reported bug pipeline
# ---------------------------------------------------------------------------


def _make_user_pipeline() -> A.Compose:
    """Replicates the pipeline from the GitHub issue."""
    return A.Compose(
        [
            A.Resize(p=1.0, height=64, width=64, interpolation=1, mask_interpolation=0),
            A.HorizontalFlip(p=1.0),
            A.Affine(
                p=1.0,
                border_mode=1,
                fill=0.0,
                fit_output=False,
                interpolation=1,
                keep_ratio=True,
                mask_interpolation=0,
                rotate=(-15.0, 15.0),
                rotate_method="largest_box",
            ),
            A.ColorJitter(p=1.0),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.ISONoise(p=1.0),
                    A.MultiplicativeNoise(p=1.0),
                ],
                p=1.0,
            ),
            A.Normalize(p=1.0),
        ],
        p=1.0,
        seed=43,
    )


def test_user_issue_pipeline_runs_with_aliased_keys() -> None:
    pipeline = _make_user_pipeline()
    pipeline.add_targets({"custom_image_key": "image", "custom_mask_key": "mask"})
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    out = pipeline(custom_image_key=img, custom_mask_key=mask)
    assert set(out.keys()) == {"custom_image_key", "custom_mask_key"}
    assert out["custom_image_key"].shape == (64, 64, 3)
    assert out["custom_mask_key"].shape == (64, 64)


def test_user_issue_pipeline_many_seeds() -> None:
    """OneOf with all branches at p=1.0 hits each noise transform across seeds; with the bug, this
    used to KeyError 'shape' from Affine and ValueError from GaussNoise/ISONoise/MultiplicativeNoise.
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    for seed in range(20):
        pipeline = A.Compose(
            [
                A.Affine(p=1.0),
                A.ColorJitter(p=1.0),
                A.OneOf(
                    [A.GaussNoise(p=1.0), A.ISONoise(p=1.0), A.MultiplicativeNoise(p=1.0)],
                    p=1.0,
                ),
            ],
            seed=seed,
        )
        pipeline.add_targets({"custom_image_key": "image", "custom_mask_key": "mask"})
        pipeline(custom_image_key=img, custom_mask_key=mask)


# ---------------------------------------------------------------------------
# Integration: every shape-dependent transform under aliased keys
# ---------------------------------------------------------------------------


# Each entry is a transform constructor that forces p=1.0 and exercises a
# get_params_dependent_on_data path that needs `params["shape"]` or get_image_data.
_SHAPE_DEPENDENT_TRANSFORMS: list[tuple[str, A.BasicTransform]] = [
    ("Affine", A.Affine(p=1.0)),
    ("ColorJitter", A.ColorJitter(p=1.0)),
    ("ChannelShuffle", A.ChannelShuffle(p=1.0)),
    ("FancyPCA", A.FancyPCA(p=1.0)),
    ("GaussNoise", A.GaussNoise(p=1.0)),
    ("MultiplicativeNoise", A.MultiplicativeNoise(p=1.0)),
    ("ISONoise", A.ISONoise(p=1.0)),
    ("SaltAndPepper", A.SaltAndPepper(p=1.0)),
    ("RandomFog", A.RandomFog(p=1.0)),
    ("RandomRain", A.RandomRain(p=1.0)),
    ("RandomGravel", A.RandomGravel(p=1.0)),
    ("RandomShadow", A.RandomShadow(p=1.0)),
    ("RandomSunFlare", A.RandomSunFlare(p=1.0)),
    ("ChannelDropout", A.ChannelDropout(p=1.0)),
    ("PlasmaShadow", A.PlasmaShadow(p=1.0)),
]


@pytest.mark.parametrize(
    ("name", "transform"),
    _SHAPE_DEPENDENT_TRANSFORMS,
    ids=[name for name, _ in _SHAPE_DEPENDENT_TRANSFORMS],
)
def test_shape_dependent_transforms_under_aliased_keys(
    name: str,
    transform: BasicTransform,
) -> None:
    """Every transform that consults shape/image metadata at param time must work
    when the user only passes aliased keys (the original GitHub issue).
    """
    img = np.full((40, 50, 3), 128, dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    pipeline = A.Compose([transform], seed=0)
    pipeline.add_targets({"custom_image_key": "image", "custom_mask_key": "mask"})
    out = pipeline(custom_image_key=img, custom_mask_key=mask)
    assert "custom_image_key" in out
    assert out["custom_image_key"].shape == img.shape
    assert "custom_mask_key" in out
    assert out["custom_mask_key"].shape == mask.shape


# ---------------------------------------------------------------------------
# Integration: bbox postprocess path (get_shape used by check_data_post_transform)
# ---------------------------------------------------------------------------


def test_bbox_postprocess_with_aliased_image() -> None:
    pipeline = A.Compose(
        [A.HorizontalFlip(p=1.0), A.Affine(p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["lab"]),
        seed=0,
    )
    pipeline.add_targets({"custom_image_key": "image", "custom_mask_key": "mask"})
    out = pipeline(
        custom_image_key=np.zeros((100, 100, 3), dtype=np.uint8),
        custom_mask_key=np.zeros((100, 100), dtype=np.uint8),
        bboxes=[[0, 0, 10, 10]],
        lab=[1],
    )
    assert "bboxes" in out
    assert len(out["bboxes"]) == 1


def test_keypoint_postprocess_with_aliased_image() -> None:
    pipeline = A.Compose(
        [A.HorizontalFlip(p=1.0), A.Affine(p=1.0)],
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["lab"]),
        seed=0,
    )
    pipeline.add_targets({"custom_image_key": "image"})
    out = pipeline(
        custom_image_key=np.zeros((100, 100, 3), dtype=np.uint8),
        keypoints=[[10, 20]],
        lab=[1],
    )
    assert "keypoints" in out
    assert len(out["keypoints"]) == 1


# ---------------------------------------------------------------------------
# Integration: mixed default + alias
# ---------------------------------------------------------------------------


def test_mixed_default_and_alias_image() -> None:
    """Both `image` and an aliased `image2` must be transformed identically."""
    img1 = np.full((40, 50, 3), 10, dtype=np.uint8)
    img2 = np.full((40, 50, 3), 200, dtype=np.uint8)
    pipeline = A.Compose([A.HorizontalFlip(p=1.0)], seed=0)
    pipeline.add_targets({"image2": "image"})
    out = pipeline(image=img1, image2=img2)
    np.testing.assert_array_equal(out["image"], img1[:, ::-1])
    np.testing.assert_array_equal(out["image2"], img2[:, ::-1])


def test_mixed_default_and_alias_mask() -> None:
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    mask1 = np.zeros((40, 50), dtype=np.uint8)
    mask1[:, :10] = 1
    mask2 = np.zeros((40, 50), dtype=np.uint8)
    mask2[:, -10:] = 5
    pipeline = A.Compose([A.HorizontalFlip(p=1.0)], seed=0)
    pipeline.add_targets({"mask2": "mask"})
    out = pipeline(image=img, mask=mask1, mask2=mask2)
    np.testing.assert_array_equal(out["mask"], mask1[:, ::-1])
    np.testing.assert_array_equal(out["mask2"], mask2[:, ::-1])


# ---------------------------------------------------------------------------
# Regression: the exact errors reported in the GitHub issue
# ---------------------------------------------------------------------------


def test_regression_keyerror_shape_no_longer_raised() -> None:
    """Affine used to raise `KeyError: 'shape'` when only aliased keys were present."""
    pipeline = A.Compose([A.Affine(p=1.0)], seed=0)
    pipeline.add_targets({"custom_image_key": "image", "custom_mask_key": "mask"})
    pipeline(
        custom_image_key=np.zeros((40, 50, 3), dtype=np.uint8),
        custom_mask_key=np.zeros((40, 50), dtype=np.uint8),
    )


def test_regression_no_valid_image_data_no_longer_raised() -> None:
    """GaussNoise/ISONoise/MultiplicativeNoise used to raise
    'No valid image/volume data found in data dict'.
    """
    pipeline = A.Compose([A.GaussNoise(p=1.0)], seed=0)
    pipeline.add_targets({"custom_image_key": "image"})
    pipeline(custom_image_key=np.zeros((40, 50, 3), dtype=np.uint8))
