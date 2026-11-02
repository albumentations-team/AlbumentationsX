import cv2
import numpy as np
import pytest
import torch

import albumentations as A


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float32)], ids=["float16", "float32"])
@pytest.mark.parametrize("boundary", ["direct", "compose"])
def test_resize_preserves_foreign_float_byte_order(dtype, boundary):
    native = np.random.default_rng(137).random((9, 13, 3), dtype=np.float32).astype(dtype)
    foreign = native.astype(dtype.newbyteorder("S"))
    foreign.setflags(write=False)
    original = foreign.copy()

    def run(image):
        transform = A.Resize(height=11, width=17, interpolation=cv2.INTER_LINEAR, p=1)
        if boundary == "direct":
            return transform(image=image)["image"]
        return A.Compose([transform], strict=True, telemetry=False, seed=137)(image=image)["image"]

    expected = run(native)
    actual = run(foreign)

    assert actual.dtype == foreign.dtype
    assert not np.shares_memory(actual, foreign)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(foreign, original)


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float32)], ids=["float16", "float32"])
@pytest.mark.parametrize("target,shape", [("volume", (3, 9, 13, 2)), ("volumes", (2, 3, 9, 13, 2))])
def test_resize3d_preserves_foreign_float_byte_order(dtype, target, shape):
    native = np.random.default_rng(137).random(shape, dtype=np.float32).astype(dtype)
    foreign = native.astype(dtype.newbyteorder("S"))
    foreign.setflags(write=False)
    original = foreign.copy()

    def run(volume):
        transform = A.Resize3D(size=(4, 7, 10), interpolation=cv2.INTER_LINEAR, p=1)
        return A.Compose([transform], strict=True, telemetry=False, seed=137)(**{target: volume})[target]

    expected = run(native)
    actual = run(foreign)

    assert actual.dtype == foreign.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(foreign, original)


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float32)], ids=["float16", "float32"])
def test_foreign_float_byte_order_survives_compose_to_tensor(dtype):
    native = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32).astype(dtype)
    foreign = native.astype(dtype.newbyteorder("S"))
    foreign.setflags(write=False)
    original = foreign.copy()

    def run(image):
        return A.Compose(
            [
                A.Resize(height=15, width=17, interpolation=cv2.INTER_LINEAR, p=1),
                A.GaussianBlur(sigma_range=(1.0, 1.0), blur_range=(0, 0), p=1),
                A.ToTensorV2(p=1),
            ],
            strict=True,
            telemetry=False,
            seed=137,
        )(image=image)["image"]

    expected = run(native)
    actual = run(foreign)

    expected_dtype = torch.float16 if dtype.type is np.float16 else torch.float32
    assert actual.dtype == expected_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(foreign, original)


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float32)], ids=["float16", "float32"])
def test_replay_accepts_foreign_float_byte_order(dtype):
    native = np.random.default_rng(137).random((9, 13, 3), dtype=np.float32).astype(dtype)
    foreign = native.astype(dtype.newbyteorder("S"))
    foreign.setflags(write=False)
    original = foreign.copy()
    pipeline = A.ReplayCompose(
        [A.Resize(height=11, width=17, interpolation=cv2.INTER_LINEAR, p=1)],
        strict=True,
        telemetry=False,
        seed=137,
    )

    expected = pipeline(image=native)
    actual = A.ReplayCompose.replay(expected["replay"], image=foreign)["image"]

    assert actual.dtype == foreign.dtype
    np.testing.assert_array_equal(actual, expected["image"])
    np.testing.assert_array_equal(foreign, original)


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float32)], ids=["float16", "float32"])
def test_noop_and_skipped_transform_preserve_foreign_array_identity(dtype):
    foreign = np.random.default_rng(137).random((9, 13, 1), dtype=np.float32).astype(dtype.newbyteorder("S"))
    foreign.setflags(write=False)

    assert A.NoOp(p=1)(image=foreign)["image"] is foreign
    assert A.Compose([A.NoOp(p=1)], strict=True, telemetry=False)(image=foreign)["image"] is foreign
    assert A.Resize(height=11, width=17, p=0)(image=foreign)["image"] is foreign
