import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core import transforms_interface
from albumentations.core.type_definitions import c4_group_elements, d4_group_elements


@pytest.mark.parametrize(
    ("transform_cls", "group_element"),
    [
        (A.HorizontalFlip, "h"),
        (A.VerticalFlip, "v"),
        *((A.RandomRotate90, element) for element in c4_group_elements),
        *((A.D4, element) for element in d4_group_elements),
    ],
)
@pytest.mark.parametrize("channels", [None, 1, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("batch_size", [0, 4])
def test_volume_symmetries_use_one_batch_operation(
    monkeypatch, transform_cls, group_element, channels, dtype, tensor, strided, batch_size
):
    shape = (batch_size, 3, 5, 7) + (() if channels is None else (channels,))
    volumes = (np.arange(np.prod(shape)).reshape(shape) % 251).astype(dtype)
    if dtype == np.float32:
        volumes /= 251
    if tensor:
        volumes = torch.from_numpy(volumes)
        volumes = volumes.unsqueeze(1) if channels is None else volumes.permute(0, 4, 1, 2, 3)
        if not strided:
            volumes = volumes.contiguous()
    if strided:
        volumes = volumes[..., ::2] if tensor else volumes[:, :, :, ::2]
    source = volumes.clone() if tensor else volumes.copy()
    numpy_source = volumes.numpy().transpose(0, 2, 3, 4, 1) if tensor else volumes
    operations = {
        "e": lambda value: value,
        "r90": lambda value: np.rot90(value, 1, (2, 3)),
        "r180": lambda value: np.rot90(value, 2, (2, 3)),
        "r270": lambda value: np.rot90(value, 3, (2, 3)),
        "v": lambda value: value[:, :, ::-1],
        "h": lambda value: value[:, :, :, ::-1],
        "t": lambda value: value.swapaxes(2, 3),
        "hvt": lambda value: value[:, :, ::-1, ::-1].swapaxes(2, 3),
    }
    expected = operations[group_element](numpy_source)

    def fail(*args, **kwargs):
        raise AssertionError("volumes entered a per-volume handler or Tensor bridge")

    monkeypatch.setattr(transform_cls, "apply_to_volume", fail)
    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", fail)
    kwargs = {"group_element": group_element} if transform_cls in {A.RandomRotate90, A.D4} else {}
    result = A.Compose([transform_cls(p=1, **kwargs)], strict=True, telemetry=False)(volumes=volumes)["volumes"]

    assert result.dtype == volumes.dtype
    if tensor:
        np.testing.assert_array_equal(result.numpy().transpose(0, 2, 3, 4, 1), expected)
        torch.testing.assert_close(volumes, source, rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(volumes, source)
        if volumes.size and not strided:
            assert np.shares_memory(result, volumes)
