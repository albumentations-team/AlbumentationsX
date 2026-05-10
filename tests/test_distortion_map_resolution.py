"""Tests for reduced-resolution distortion map generation."""

import numpy as np
import pytest

import albumentations as A

DISTORTION_TRANSFORMS = [
    pytest.param(A.ElasticTransform, {"alpha": 2, "sigma": 20}, id="ElasticTransform"),
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


def _make_image(shape: tuple[int, int, int] = (64, 48, 3)) -> np.ndarray:
    return np.random.default_rng(137).integers(0, 256, shape, dtype=np.uint8)


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
        A.ElasticTransform(map_resolution_range=map_resolution_range)


@pytest.mark.parametrize(("transform_cls", "params"), DISTORTION_TRANSFORMS)
def test_low_map_resolution_returns_full_size_maps(transform_cls, params):
    image = _make_image()
    transform = transform_cls(**params, map_resolution_range=(0.25, 0.25), p=1.0)
    transform.set_random_seed(137)

    result = transform.get_params_dependent_on_data({"shape": image.shape}, {"image": image})

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

    result = transform.get_params_dependent_on_data({"shape": image.shape}, {"image": image})

    assert result["map_x"].shape == image.shape[:2]
    assert result["map_y"].shape == image.shape[:2]
    assert result["map_x"].dtype == np.float32
    assert result["map_y"].dtype == np.float32


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
