from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.transforms_interface import BasicTransform

_DROPOUT_CASES = [
    pytest.param(
        A.CoarseDropout,
        {"num_holes_range": (1, 1), "hole_height_range": (8, 8), "hole_width_range": (8, 8)},
        False,
        id="coarse-dropout",
    ),
    pytest.param(
        A.Erasing,
        {"scale": (0.05, 0.05), "ratio": (1.0, 1.0)},
        False,
        id="erasing",
    ),
    pytest.param(
        A.ConstrainedCoarseDropout,
        {
            "num_holes_range": (1, 1),
            "hole_height_range": (0.2, 0.2),
            "hole_width_range": (0.2, 0.2),
            "mask_indices": [1],
        },
        True,
        id="constrained-coarse-dropout",
    ),
]

_DROPOUT_FILLS = [
    pytest.param(0.5, id="scalar"),
    pytest.param((0.1, 0.5, 0.9), id="per-channel"),
    pytest.param("random", id="random"),
    pytest.param("random_uniform", id="random-uniform"),
    pytest.param("grayscale", id="grayscale"),
    pytest.param("inpaint_telea", id="inpaint-telea"),
    pytest.param("inpaint_ns", id="inpaint-ns"),
]


@pytest.mark.parametrize(("transform_cls", "kwargs", "requires_mask"), _DROPOUT_CASES)
@pytest.mark.parametrize("fill", _DROPOUT_FILLS)
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_dropout_fill_modes_support_float16_compose(
    transform_cls: type[BasicTransform],
    kwargs: dict[str, Any],
    requires_mask: bool,
    fill: float | tuple[float, ...] | str,
    container: str,
) -> None:
    rng = np.random.default_rng(137)
    image = rng.uniform(0.1, 0.9, (32, 40, 3)).astype(np.float16)
    if container == "tensor":
        source: np.ndarray | torch.Tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
        original = source.clone()
    else:
        source = image.copy()
        original = source.copy()

    transform = transform_cls(**kwargs, fill=fill, fill_mask=None, p=1.0)
    pipeline = A.Compose([transform], seed=137, strict=True, telemetry=False)
    targets: dict[str, Any] = {"image": source}
    if requires_mask:
        targets["mask"] = np.ones(image.shape[:2], dtype=np.uint8)

    result = pipeline(**targets)
    output = result["image"]
    if container == "tensor":
        assert isinstance(output, torch.Tensor)
        assert output.dtype is torch.float16
        assert tuple(output.shape) == (3, 32, 40)
        output_array = output.permute(1, 2, 0).numpy()
        torch.testing.assert_close(source, original, rtol=0, atol=0)
    else:
        assert isinstance(output, np.ndarray)
        assert output.shape == image.shape
        assert output.dtype == np.float16
        output_array = output
        np.testing.assert_array_equal(source, original)

    assert np.isfinite(output_array).all()
    assert output_array.min() >= 0
    assert output_array.max() <= 1
    assert not np.array_equal(output_array, image)
    if requires_mask:
        np.testing.assert_array_equal(result["mask"], targets["mask"])


@pytest.mark.parametrize("target", ["volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_coarse_dropout3d_supports_float16_compose(target: str, container: str) -> None:
    rng = np.random.default_rng(137)
    volume = rng.uniform(0.1, 0.9, (4, 32, 40, 3)).astype(np.float16)
    source_array = volume if target == "volume" else np.stack((volume, volume.copy()))
    if container == "tensor":
        if target == "volume":
            source: np.ndarray | torch.Tensor = torch.from_numpy(source_array.transpose(3, 0, 1, 2).copy())
        else:
            source = torch.from_numpy(source_array.transpose(0, 4, 1, 2, 3).copy())
        original = source.clone()
    else:
        source = source_array.copy()
        original = source.copy()

    transform = A.CoarseDropout3D(
        num_holes_range=(1, 1),
        hole_depth_range=(0.25, 0.25),
        hole_height_range=(0.2, 0.2),
        hole_width_range=(0.2, 0.2),
        fill=0.5,
        p=1.0,
    )
    result = A.Compose([transform], seed=137, strict=True, telemetry=False)(**{target: source})[target]

    if container == "tensor":
        assert isinstance(result, torch.Tensor)
        assert result.dtype is torch.float16
        torch.testing.assert_close(source, original, rtol=0, atol=0)
        result_array = result.numpy()
    else:
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float16
        np.testing.assert_array_equal(source, original)
        result_array = result

    expected_shape = tuple(source.shape)
    assert result_array.shape == expected_shape
    assert np.isfinite(result_array).all()
    assert not np.array_equal(result_array, original.numpy() if container == "tensor" else original)
