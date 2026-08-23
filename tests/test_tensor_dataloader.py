"""Integration coverage for Tensor-native Compose and PyTorch collation."""

from __future__ import annotations

import torch

import albumentations as A


class _TensorComposeDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Small picklable dataset that applies the supported Tensor Compose path."""

    def __init__(self) -> None:
        self.images = torch.arange(12 * 3 * 11 * 13, dtype=torch.float32).reshape(12, 3, 11, 13)
        self.transform = A.Compose([A.NoOp(p=1.0)], strict=True)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.transform(image=self.images[index])


class _TensorFallbackVolumeDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Small picklable volume dataset that requires Compose's NumPy bridge."""

    def __init__(self) -> None:
        self.volumes = torch.arange(6 * 1 * 3 * 5 * 7, dtype=torch.float32).reshape(6, 1, 3, 5, 7)
        self.masks = torch.arange(6 * 3 * 5 * 7, dtype=torch.uint8).reshape(6, 3, 5, 7) % 4
        self.transform = A.Compose(
            [A.Flip3D(flip_axes=(2,), p=1.0), A.CenterCrop3D(size=(2, 3, 5), p=1.0)],
            strict=True,
        )

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.transform(volume=self.volumes[index], mask3d=self.masks[index])


def test_tensor_compose_output_collates_into_model_ready_nchw() -> None:
    loader = torch.utils.data.DataLoader(
        _TensorComposeDataset(),
        batch_size=4,
        num_workers=0,
    )

    batches = list(loader)

    assert [batch["image"].shape for batch in batches] == [(4, 3, 11, 13)] * 3
    assert all(batch["image"].dtype == torch.float32 for batch in batches)


def test_tensor_compose_bridge_output_collates_volume_and_mask3d() -> None:
    loader = torch.utils.data.DataLoader(
        _TensorFallbackVolumeDataset(),
        batch_size=2,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["volume"].shape == (2, 1, 2, 3, 5)
    assert batch["mask3d"].shape == (2, 2, 3, 5)
    assert batch["volume"].dtype == torch.float32
    assert batch["mask3d"].dtype == torch.uint8
