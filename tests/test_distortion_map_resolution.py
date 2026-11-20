"""Tests for reduced-resolution distortion map generation."""

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import _functional_distortion as distortion
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams
from tests.utils import make_sampling_args

DISTORTION_TRANSFORMS = [
    pytest.param(A.GridDistortion, {"num_steps": 5, "distort_range": (-0.2, 0.2)}, id="GridDistortion"),
    pytest.param(A.OpticalDistortion, {"distort_range": (-0.05, 0.05)}, id="OpticalDistortion"),
    pytest.param(A.PiecewiseAffine, {"scale_range": (0.03, 0.05)}, id="PiecewiseAffine"),
    pytest.param(A.ThinPlateSpline, {"scale_range": (0.2, 0.4)}, id="ThinPlateSpline"),
    pytest.param(
        A.WaterRefraction,
        {"amplitude_range": (0.02, 0.04), "wavelength_range": (0.1, 0.2)},
        id="WaterRefraction",
    ),
    pytest.param(A.PixelSpread, {"radius": 3}, id="PixelSpread"),
]

NOOP_DISTORTION_TRANSFORMS = [
    pytest.param(A.GridDistortion, {"num_steps": 5, "distort_range": (0, 0)}, id="GridDistortion"),
    pytest.param(A.OpticalDistortion, {"distort_range": (0, 0)}, id="OpticalDistortion"),
    pytest.param(A.WaterRefraction, {"amplitude_range": (0, 0)}, id="WaterRefraction"),
]


def _make_image(shape: tuple[int, int, int] = (64, 48, 3)) -> np.ndarray:
    return np.random.default_rng(137).integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize(
    ("shape", "map_resolution", "radius"),
    [((67, 53, 3), 1.0, 2), ((67, 53, 3), 0.5, 2), ((1, 1, 3), 0.5, 2), ((67, 53, 3), 1.0, 2**25)],
)
def test_pixel_spread_preserves_seeded_coordinate_maps(
    shape: tuple[int, int, int],
    map_resolution: float,
    radius: int,
) -> None:
    image = _make_image(shape)
    transform = A.PixelSpread(radius=radius, map_resolution_range=(map_resolution, map_resolution), p=1)
    transform.set_random_seed(137)
    result = transform.sample_parameters(
        *make_sampling_args(transform, {"image": image}),
        SamplingContext.from_owner(transform, {}),
    ).params

    height, width = (max(2, int(size * map_resolution)) for size in shape[:2])
    scaled_radius = max(1, round(radius * map_resolution))
    offsets = np.random.default_rng(137).integers(
        -scaled_radius,
        scaled_radius + 1,
        size=(height, width, 2),
        dtype=np.int32,
    )
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    expected_x, expected_y = fgeometric.upscale_distortion_maps(
        (cols + offsets[..., 1]).astype(np.float32),
        (rows + offsets[..., 0]).astype(np.float32),
        shape[:2],
        transform.interpolation,
    )
    np.testing.assert_array_equal(result["map_x"].view(np.uint32), expected_x.view(np.uint32))
    np.testing.assert_array_equal(result["map_y"].view(np.uint32), expected_y.view(np.uint32))


@pytest.mark.parametrize(("num_control_points", "map_resolution"), [(2, 1.0), (4, 0.5)])
def test_thin_plate_spline_cached_maps_and_outputs_match_uncached(
    num_control_points: int,
    map_resolution: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = A.ReplayCompose(
        [
            A.ThinPlateSpline(
                num_control_points=num_control_points,
                map_resolution_range=(map_resolution, map_resolution),
                p=1,
            ),
        ],
    )
    image = _make_image((67, 53, 3))
    mask = image[..., 0].copy()
    cache = distortion._cached_tps_grid_basis
    cache.cache_clear()
    try:
        pipeline.set_random_seed(137)
        cold = pipeline(image=image, mask=mask)
        pipeline.set_random_seed(138)
        warm = pipeline(image=image, mask=mask)
        assert cache.cache_info().hits == 1
        monkeypatch.setattr(distortion, "_TPS_GRID_BASIS_MAX_BYTES", 0)

        for seed, cached in ((137, cold), (138, warm)):
            pipeline.set_random_seed(seed)
            uncached = pipeline(image=image, mask=mask)
            for target in ("image", "mask"):
                np.testing.assert_array_equal(cached[target], uncached[target])
            cached_params = cached["replay"]["transforms"][0]["params"]
            uncached_params = uncached["replay"]["transforms"][0]["params"]
            cached_maps = SampledParams.from_dict(cached_params).params_for("image")
            uncached_maps = SampledParams.from_dict(uncached_params).params_for("image")
            for key in ("map_x", "map_y"):
                np.testing.assert_array_equal(cached_maps[key].view(np.uint32), uncached_maps[key].view(np.uint32))
    finally:
        cache.cache_clear()


def test_thin_plate_spline_cache_evicts_and_bypasses_large_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = distortion._cached_tps_grid_basis
    cache.cache_clear()
    # At five control points, a 16-by-16 grid uses exactly 8 KiB.
    monkeypatch.setattr(distortion, "_TPS_GRID_BASIS_MAX_BYTES", 8 * 1024)
    pipeline = A.Compose([A.ThinPlateSpline(num_control_points=2, p=1)], seed=137, telemetry=False)
    try:
        for height, width in ((16, 16), (16, 15), (16, 16), (16, 17), (16, 14), (16, 15)):
            pipeline(image=_make_image((height, width, 3)))
        info = cache.cache_info()
        assert (info.currsize, info.hits, info.misses) == (2, 1, 4)
    finally:
        cache.cache_clear()


@pytest.mark.parametrize(("transform_cls", "params"), DISTORTION_TRANSFORMS)
def test_map_resolution_range_accepts_valid_tuple(transform_cls, params):
    transform = transform_cls(**params, map_resolution_range=(0.25, 1.0), p=1.0)

    assert transform.map_resolution_range == (0.25, 1.0)


@pytest.mark.parametrize(
    "map_resolution_range",
    [
        (0.0, 1.0),
        (-0.1, 1.0),
        (0.25, 1.1),
        (0.75, 0.25),
    ],
)
def test_map_resolution_range_rejects_invalid_values(map_resolution_range):
    with pytest.raises(ValueError):
        A.GridDistortion(map_resolution_range=map_resolution_range)


@pytest.mark.parametrize(("transform_cls", "params"), DISTORTION_TRANSFORMS)
def test_low_map_resolution_returns_full_size_maps(transform_cls, params):
    image = _make_image()
    transform = transform_cls(**params, map_resolution_range=(0.25, 0.25), p=1.0)
    transform.set_random_seed(137)

    result = transform.sample_parameters(
        *make_sampling_args(transform, {"image": image}),
        SamplingContext.from_owner(transform, {}),
    ).params

    assert result["map_x"].shape == image.shape[:2]
    assert result["map_y"].shape == image.shape[:2]
    assert result["map_x"].dtype == np.float32
    assert result["map_y"].dtype == np.float32


@pytest.mark.parametrize(("transform_cls", "params"), DISTORTION_TRANSFORMS)
@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 1, 3)])
def test_low_map_resolution_returns_full_size_maps_for_tiny_images(transform_cls, params, shape):
    image = _make_image(shape)
    transform = transform_cls(**params, map_resolution_range=(0.25, 0.25), p=1.0)
    transform.set_random_seed(137)

    result = transform.sample_parameters(
        *make_sampling_args(transform, {"image": image}),
        SamplingContext.from_owner(transform, {}),
    ).params

    assert result["map_x"].shape == image.shape[:2]
    assert result["map_y"].shape == image.shape[:2]
    assert result["map_x"].dtype == np.float32
    assert result["map_y"].dtype == np.float32


def test_upscale_distortion_maps_preserves_identity_map():
    height, width = 13, 17
    map_height, map_width = 4, 5
    y_coords, x_coords = np.meshgrid(
        np.arange(map_height, dtype=np.float32),
        np.arange(map_width, dtype=np.float32),
        indexing="ij",
    )

    map_x, map_y = fgeometric.upscale_distortion_maps(x_coords, y_coords, (height, width))

    expected_y, expected_x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    np.testing.assert_array_equal(map_x, expected_x)
    np.testing.assert_array_equal(map_y, expected_y)


@pytest.mark.parametrize(("transform_cls", "params"), NOOP_DISTORTION_TRANSFORMS)
def test_noop_distortion_stays_noop_with_low_map_resolution(transform_cls, params):
    image = _make_image()
    transform = A.Compose(
        [transform_cls(**params, map_resolution_range=(0.25, 0.25), p=1.0)],
        seed=137,
        strict=True,
    )

    result = transform(image=image)

    np.testing.assert_array_equal(result["image"], image)


@pytest.mark.parametrize(("transform_cls", "params"), DISTORTION_TRANSFORMS)
def test_map_resolution_range_records_sampled_scalar(transform_cls, params):
    image = _make_image()
    transform = transform_cls(**params, map_resolution_range=(0.25, 0.75), p=1.0)
    transform(image=image)

    sampled = transform.applied_config["map_resolution_range"]
    assert isinstance(sampled, float)
    assert 0.25 <= sampled <= 0.75


@pytest.mark.parametrize(("transform_cls", "params"), DISTORTION_TRANSFORMS)
def test_default_map_resolution_matches_explicit_full_resolution(transform_cls, params):
    image = _make_image()
    default_transform = transform_cls(**params, p=1.0)
    explicit_transform = transform_cls(**params, map_resolution_range=(1.0, 1.0), p=1.0)
    default_transform.set_random_seed(137)
    explicit_transform.set_random_seed(137)

    default_result = default_transform(image=image)
    explicit_result = explicit_transform(image=image)

    np.testing.assert_array_equal(default_result["image"], explicit_result["image"])


@pytest.mark.parametrize(("transform_cls", "params"), DISTORTION_TRANSFORMS)
def test_low_map_resolution_applies_consistently_to_targets(transform_cls, params):
    image = _make_image()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[16:48, 12:36] = 1
    bboxes = np.array([[12, 16, 36, 48]], dtype=np.float32)
    keypoints = np.array([[24, 32]], dtype=np.float32)

    transform = A.Compose(
        [transform_cls(**params, map_resolution_range=(0.25, 0.25), p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
        seed=137,
        strict=True,
    )

    result = transform(
        image=image,
        mask=mask,
        bboxes=bboxes,
        bbox_labels=[1],
        keypoints=keypoints,
        keypoint_labels=[2],
    )

    assert result["image"].shape == image.shape
    assert result["mask"].shape == mask.shape
    assert len(result["bboxes"]) == len(result["bbox_labels"])
    assert len(result["keypoints"]) == len(result["keypoint_labels"])
