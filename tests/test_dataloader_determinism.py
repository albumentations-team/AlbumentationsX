"""Reproduce worker-count and persistent-worker behavior with real spawned workers."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A

pytestmark = pytest.mark.pytorch


class _DropoutDataset(Dataset):
    def __init__(self, *, sample_keyed: bool):
        self.sample_keyed = sample_keyed
        self.transform = A.Compose(
            [
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(0.25, 0.25),
                    hole_width_range=(0.25, 0.25),
                    fill=0,
                    p=1.0,
                ),
            ],
            seed=137,
        )

    def __len__(self):
        return 8

    def __getitem__(self, index):
        image = np.full((24, 32, 3), 137, dtype=np.uint8)
        seed = 137 + index if self.sample_keyed else None
        return self.transform(image=image, invocation_seed=seed)["image"]


def _epochs(*, num_workers, persistent_workers=False, sample_keyed=False, generator_seed=137):
    dataset = _DropoutDataset(sample_keyed=sample_keyed)
    indices = list(range(len(dataset)))
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=indices,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        generator=torch.Generator().manual_seed(generator_seed),
        multiprocessing_context="spawn" if num_workers else None,
        timeout=30 if num_workers else 0,
    )
    epochs = []
    for epoch in range(2):
        indices[:] = range(epoch * len(dataset), (epoch + 1) * len(dataset))
        epochs.append(torch.cat(list(loader)).numpy())
    return np.stack(epochs)


@pytest.mark.parametrize(("num_workers", "persistent_workers"), [(1, False), (2, False), (2, True)])
def test_sample_keyed_seeds_are_independent_of_dataloader_workers(num_workers, persistent_workers):
    expected = _epochs(num_workers=0, sample_keyed=True)
    actual = _epochs(
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        sample_keyed=True,
        generator_seed=138,
    )

    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual[0], actual[1])


@pytest.mark.parametrize("persistent_workers", [False, True])
def test_worker_streams_reproduce_fixed_configuration_and_advance_between_epochs(persistent_workers):
    first = _epochs(num_workers=2, persistent_workers=persistent_workers)
    second = _epochs(num_workers=2, persistent_workers=persistent_workers)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert not np.array_equal(first[0, :4], first[0, 4:])
    assert not np.array_equal(first, _epochs(num_workers=0))
