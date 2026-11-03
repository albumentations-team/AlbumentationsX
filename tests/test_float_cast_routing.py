from unittest.mock import Mock

import albucore._float_cast as cast_router
import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.blur import functional as fblur
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.medical import functional as fmedical
from albumentations.augmentations.pixel import _functional_color as fcolor
from albumentations.augmentations.pixel import _functional_illumination as fillumination
from albumentations.augmentations.pixel import _functional_weather as fweather
from albumentations.augmentations.pixel import dithering_functional as fdithering


@pytest.mark.parametrize(
    ("module", "prepare"),
    [
        (fmedical, fmedical._float32_working_image),
        (fdithering, fdithering._float32_working_image),
        (fcolor, fcolor._iso_noise_working_image),
        (fweather, fweather._spatter_working_image),
    ],
    ids=["medical", "dithering", "iso-noise", "spatter"],
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "readonly"])
def test_float32_workspace_uses_shared_cast_without_changing_ownership(module, prepare, dtype, layout, monkeypatch):
    image = np.linspace(-2, 2, 128 * 160 * 3, dtype=dtype).reshape(128, 160, 3)
    image[0, 0] = [-0.0, np.inf, -np.inf]
    if layout == "transpose":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.setflags(write=False)
    original = image.tobytes()
    shared_cast = Mock(wraps=module.from_float)
    monkeypatch.setattr(module, "from_float", shared_cast)

    working, source_dtype = prepare(image)

    assert source_dtype == image.dtype
    assert working.dtype == np.float32
    np.testing.assert_array_equal(working.view(np.uint32), image.astype(np.float32).view(np.uint32))
    assert image.tobytes() == original
    assert shared_cast.call_count == int(dtype == np.float16)
    if dtype == np.float16:
        assert not np.shares_memory(working, image)
        assert working.flags.writeable
    else:
        assert working is image


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
def test_float_cast_keeps_owned_contiguous_opencv_workspace(operation, reference, channels, layout, monkeypatch):
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
def test_seeded_mri_pipeline_and_replay_match_numpy_cast_reference(artifact, dtype, monkeypatch):
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

    def numpy_cast(image, target, *, out=None):
        if out is not None:
            np.copyto(out, image, casting="unsafe")
            return out
        return image.astype(target, copy=False)

    monkeypatch.setattr(cast_router, "_cast_float_numpy", numpy_cast)
    monkeypatch.setattr("albucore.convert._cast_float_numpy", numpy_cast)
    expected = A.ReplayCompose.replay(actual["replay"], volume=volume, mask3d=mask, keypoints=keypoints)

    unsigned = np.uint16 if dtype == np.float16 else np.uint32
    np.testing.assert_array_equal(actual["volume"].view(unsigned), expected["volume"].view(unsigned))
    np.testing.assert_array_equal(actual["mask3d"], expected["mask3d"])
    np.testing.assert_array_equal(actual["keypoints"], expected["keypoints"])
    assert volume.tobytes() == original


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_lens_flare_float32_addition_preserves_source_and_cast_order(dtype, strided):
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
