import gc
import weakref
from collections.abc import Callable

import albucore._float_cast as cast_router
import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.blur import functional as fblur
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.mixing import functional as fmixing
from albumentations.augmentations.pixel import _functional_illumination as fillumination


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_glass_blur_readonly_fortran_matches_contiguous(dtype: type[np.generic], channels: int, mode: str) -> None:
    values = np.random.default_rng(137).random((24, 32, channels), dtype=np.float32)
    image = (values * 255).astype(np.uint8) if dtype == np.uint8 else values.astype(dtype)
    source = np.asfortranarray(image)
    source.flags.writeable = False
    original = source.tobytes()

    def pipeline() -> A.Compose:
        return A.Compose([A.GlassBlur(max_delta=1, iterations=1, mode=mode, p=1)], seed=137)

    expected = pipeline()(image=image)["image"]
    result = pipeline()(image=source)["image"]

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == source.dtype
    assert result.shape == source.shape
    assert source.tobytes() == original


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (lambda image: fblur.box_blur(image, 3), lambda image: cv2.blur(image, (3, 3))),
        (
            lambda image: fgeometric.erode(image, np.ones((3, 3), dtype=np.uint8)),
            lambda image: cv2.erode(image, np.ones((3, 3), dtype=np.uint8)),
        ),
        (
            lambda image: fgeometric.dilate(image, np.ones((3, 3), dtype=np.uint8)),
            lambda image: cv2.dilate(image, np.ones((3, 3), dtype=np.uint8)),
        ),
    ],
    ids=["blur", "erode", "dilate"],
)
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "negative"])
def test_float_cast_keeps_owned_contiguous_opencv_workspace(
    operation: Callable[[np.ndarray], np.ndarray],
    reference: Callable[[np.ndarray], np.ndarray],
    channels: int,
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cast_router, "_NUMKONG_X86_CAST", True)
    image = np.random.default_rng(137).random((128, 160, channels), dtype=np.float32).astype(np.float16)
    if layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "negative":
        image = image[:, ::-1]
    image.setflags(write=False)
    original = image.tobytes()
    expected = reference(image.astype(np.float32, order="C")).reshape(image.shape).astype(np.float16)

    result = operation(image)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))
    assert result.flags.c_contiguous
    assert result.flags.writeable
    assert not np.shares_memory(result, image)
    assert image.tobytes() == original


@pytest.mark.parametrize("artifact", [A.GibbsRinging, A.KSpaceSpikeNoise])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_seeded_mri_pipeline_and_replay_match_numpy_cast_reference(
    artifact: type[A.BasicTransform],
    dtype: type[np.generic],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cast_router, "_NUMKONG_X86_CAST", True)
    volume = np.random.default_rng(137).random((16, 32, 32, 1), dtype=np.float32).astype(dtype)
    volume.setflags(write=False)
    original = volume.tobytes()
    mask = (volume[..., 0] > 0.5).astype(np.uint8)
    keypoints = np.array([[0, 0, 0], [31, 0, 0], [0, 31, 0], [0, 0, 15]], dtype=np.float32)
    pipeline = A.ReplayCompose(
        [
            A.Affine(scale=(0.95, 1.05), p=1),
            A.BiasField(p=1),
            A.Anisotropy3D(axes=(1, 2), p=1),
            artifact(p=1),
            A.RandomGamma(p=1),
        ],
        seed=137,
        strict=True,
        keypoint_params=A.KeypointParams(coord_format="xyz", remove_invisible=False),
    )

    actual = pipeline(volume=volume, mask3d=mask, keypoints=keypoints)

    monkeypatch.setattr(cast_router, "_NUMKONG_X86_CAST", False)
    expected = A.ReplayCompose.replay(actual["replay"], volume=volume, mask3d=mask, keypoints=keypoints)

    unsigned = np.uint16 if dtype == np.float16 else np.uint32
    np.testing.assert_array_equal(actual["volume"].view(unsigned), expected["volume"].view(unsigned))
    np.testing.assert_array_equal(actual["mask3d"], expected["mask3d"])
    np.testing.assert_array_equal(actual["keypoints"], expected["keypoints"])
    assert volume.tobytes() == original


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_lens_flare_float32_addition_preserves_source_and_cast_order(
    dtype: type[np.generic],
    strided: bool,
) -> None:
    image = np.full((128, 160, 3), 64 if dtype == np.uint8 else 0.25, dtype=dtype)
    if strided:
        image = image[:, ::-1]
    image.setflags(write=False)
    original = image.tobytes()
    expected = image.astype(np.float32)
    expected[0] += np.float32(51 if dtype == np.uint8 else 0.2)
    expected = expected.astype(dtype)

    result = fillumination.apply_lens_flare(
        image,
        flare_center=(0, 0),
        ghosts=[],
        starburst_angles=np.array([0], dtype=np.float32),
        starburst_intensity=0.2,
        bloom_radius=0,
    )

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == dtype
    assert not np.shares_memory(result, image)
    assert image.tobytes() == original


def test_soft_blend_releases_cast_workspaces_before_output_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    base = np.full((137, 137, 3), 0.25, dtype=np.float16)
    donor = np.full_like(base, 0.75)
    alpha = np.full(base.shape[:2], 0.25, dtype=np.float32)
    workspaces: list[weakref.ReferenceType[np.ndarray]] = []
    original_cast = fmixing.convert_dtype

    def tracked_cast(
        image: np.ndarray, target_dtype: np.dtype | type[np.generic], *, max_value: float | None = None
    ) -> np.ndarray:
        if np.dtype(target_dtype) == np.float16:
            gc.collect()
            assert len(workspaces) == 2
            assert all(reference() is None for reference in workspaces)
        result = original_cast(image, target_dtype, max_value=max_value)
        if np.dtype(target_dtype) == np.float32:
            workspaces.append(weakref.ref(result))
        return result

    monkeypatch.setattr(fmixing, "convert_dtype", tracked_cast)

    result = fmixing.blend_images_using_alpha(base, donor, alpha)

    np.testing.assert_array_equal(result, np.full_like(base, 0.375))
    np.testing.assert_array_equal(base, np.full_like(base, 0.25))
    np.testing.assert_array_equal(donor, np.full_like(donor, 0.75))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_morphology_public_grayscale_boundary_passes_explicit_channels_to_core(
    dtype: type[np.generic],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import albucore

    original_erode = albucore.erode
    seen_shapes: list[tuple[int, ...]] = []

    def erode(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        seen_shapes.append(image.shape)
        assert image.ndim == 3 and image.shape[-1] == 1
        return original_erode(image, kernel)

    monkeypatch.setattr(albucore, "erode", erode)
    values = np.random.default_rng(137).random((13, 17), dtype=np.float32)
    image = (values * 255).astype(dtype) if dtype == np.uint8 else values.astype(dtype)
    mask = (values > 0.5).astype(np.uint8)
    image.flags.writeable = False
    mask.flags.writeable = False
    pipeline = A.Compose([A.Morphological(scale=(3, 3), operation="erosion", p=1)], seed=137)

    result = pipeline(image=image, mask=mask)

    assert seen_shapes == [(13, 17, 1), (13, 17, 1)]
    assert result["image"].shape == image.shape
    assert result["mask"].shape == mask.shape
    assert result["image"].dtype == image.dtype
