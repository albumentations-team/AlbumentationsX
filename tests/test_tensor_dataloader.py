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


def test_tensor_compose_output_collates_into_model_ready_nchw() -> None:
    loader = torch.utils.data.DataLoader(
        _TensorComposeDataset(),
        batch_size=4,
        num_workers=0,
    )

    batches = list(loader)

    assert [batch["image"].shape for batch in batches] == [(4, 3, 11, 13)] * 3
    assert all(batch["image"].dtype == torch.float32 for batch in batches)
