"""Tests for worker-aware seed functionality in Compose."""

import multiprocessing
import pickle
import random
import sys

import numpy as np
import pytest

import albumentations as A
from albumentations.core.random_utils import _derive_effective_seed, _RuntimeRngContext

try:
    import torch
    import torch.utils.data

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


class MockWorkerInfo:
    """Mock torch.utils.data.get_worker_info() response."""

    def __init__(self, id: int):
        self.id = id


if _TORCH_AVAILABLE:

    class TestDataset(torch.utils.data.Dataset):
        """Dataset for worker seed testing - must be at module level for pickling."""

        def __init__(self, transform):
            self.transform = transform
            self.worker_results = {}

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else -1

            img = np.zeros((10, 10, 3), dtype=np.uint8)
            img[:, :5] = 255

            result = self.transform(image=img)
            was_flipped = result["image"][0, 0, 0] == 0

            if worker_id not in self.worker_results:
                self.worker_results[worker_id] = []
            self.worker_results[worker_id].append((idx, was_flipped))

            return float(was_flipped)

    class SimpleDataset(torch.utils.data.Dataset):
        """Dataset for epoch diversity testing - must be at module level for pickling."""

        def __init__(self, transform):
            self.transform = transform
            self.data = [np.ones((10, 10, 3), dtype=np.uint8) * 255] * 4

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx].copy()
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return float(np.sum(image))

    class DropoutPositionDataset(torch.utils.data.Dataset):
        """Dataset that exposes sampled dropout geometry for DataLoader RNG regression tests."""

        def __init__(self, *, use_compose: bool, seed: int | None = None):
            dropout = A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(0.25, 0.25),
                hole_width_range=(0.25, 0.25),
                fill=0,
                p=1.0,
            )
            if use_compose:
                self.transform = A.Compose([dropout], seed=seed)
            else:
                dropout.set_random_seed(seed)
                self.transform = dropout
            self.data = np.ones((64, 32, 32, 3), dtype=np.uint8) * 255

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            result = self.transform(image=self.data[idx])
            coords = np.argwhere(result["image"][:, :, 0] == 0)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            return torch.tensor([y_min, x_min, y_max, x_max], dtype=torch.int64)


def _fork_worker_context() -> str:
    if not _TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    if sys.platform == "win32":
        pytest.skip("fork multiprocessing context is not available on Windows")
    if sys.platform == "darwin":
        pytest.skip("fork multiprocessing context is not supported for these tests on macOS")
    if "fork" not in torch.multiprocessing.get_all_start_methods():
        pytest.skip("fork multiprocessing context is not available")
    return "fork"


def _first_batches_across_epochs(
    dataset: "torch.utils.data.Dataset",
    *,
    generator_seed: int = 137,
    num_workers: int = 4,
    epochs: int = 3,
    persistent_workers: bool = False,
) -> list[list[list[int]]]:
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        generator=generator,
        multiprocessing_context=_fork_worker_context(),
    )

    first_batches = []
    if persistent_workers:
        for _epoch in range(epochs):
            for batch_index, batch in enumerate(loader):
                if batch_index == 0:
                    first_batches.append(batch.tolist())
        return first_batches

    for _epoch in range(epochs):
        first_batches.append(next(iter(loader)).tolist())
    return first_batches


def _assert_any_epoch_differs(first_batches: list[list[list[int]]]) -> None:
    unique_batches = {tuple(tuple(position) for position in batch) for batch in first_batches}
    assert len(unique_batches) > 1, f"All epochs produced identical dropout positions: {first_batches}"


def test_worker_seed_without_torch():
    """Test that worker seed functionality works when PyTorch is not available."""
    # Create compose (worker-aware seed is now always enabled)
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
        ],
        seed=137,
    )

    # Should work fine without PyTorch
    img = np.ones((100, 100, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result["image"].shape == (100, 100, 3)


@pytest.mark.skipif(
    "torch" not in sys.modules and not any("torch" in str(p) for p in sys.path),
    reason="PyTorch not available",
)
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="Multiprocessing test incompatible with spawn method used on macOS/Windows",
)
def test_worker_seed_with_torch():
    """Test worker seed functionality with PyTorch available."""
    # Test with worker-aware seed (now always enabled)
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
        ],
        seed=137,
    )

    dataset = TestDataset(transform)

    # Test 1: Verify different workers produce different results with same indices
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        persistent_workers=True,  # Use persistent to ensure consistent worker assignment
        shuffle=False,
    )

    # Collect one epoch of data
    dataset.worker_results.clear()
    results = []
    for batch in loader:
        results.append(batch.item())

    # Check that different workers produced different patterns
    if len(dataset.worker_results) >= 2:
        # Get results from two different workers for the same indices
        worker_ids = list(dataset.worker_results.keys())
        if len(worker_ids) >= 2:
            worker0_results = dict(dataset.worker_results[worker_ids[0]])
            worker1_results = dict(dataset.worker_results[worker_ids[1]])

            # Find common indices processed by both workers
            common_indices = set(worker0_results.keys()) & set(worker1_results.keys())

            if len(common_indices) >= 2:
                # Workers should produce different results for at least some indices
                differences = sum(1 for idx in common_indices if worker0_results[idx] != worker1_results[idx])

                # With p=0.5, we expect roughly half to be different
                # But we'll accept any difference as proof of different seeds
                assert differences > 0, "Different workers produced identical results"


@pytest.mark.skipif(
    "torch" not in sys.modules and not any("torch" in str(p) for p in sys.path),
    reason="PyTorch not available",
)
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="Multiprocessing test incompatible with spawn method used on macOS/Windows",
)
def test_dataloader_epoch_diversity():
    """Test that DataLoader produces different augmentations across epochs with worker-aware seed."""
    # Create transform with fixed seed (worker-aware seed is always enabled)
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=1.0, brightness_range=(-0.3, 0.3)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(angle_range=(-30, 30), p=0.5),
        ],
        seed=42,
    )

    dataset = SimpleDataset(transform=transform)

    # Test with persistent_workers=False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        persistent_workers=False,
    )

    # Collect data from multiple epochs
    epoch_data = []
    for _epoch in range(3):
        epoch_batch_sums = []
        for batch in dataloader:
            # Convert batch to list and sum all values
            batch_sum = float(torch.sum(batch))
            epoch_batch_sums.append(batch_sum)
        epoch_data.append(epoch_batch_sums)

    # Check that epochs produce different results
    # At least one epoch should differ from the others
    assert not (epoch_data[0] == epoch_data[1] == epoch_data[2]), (
        f"All epochs produced identical augmentations: {epoch_data}"
    )


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_unseeded_compose_dataloader_respawns_use_new_worker_seed():
    """Unseeded Compose should not replay identical dropout geometry when workers respawn."""
    first_batches = _first_batches_across_epochs(DropoutPositionDataset(use_compose=True))

    _assert_any_epoch_differs(first_batches)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_unseeded_direct_transform_dataloader_respawns_use_new_worker_seed():
    """Direct BasicTransform usage should get the same worker RNG protection as Compose."""
    first_batches = _first_batches_across_epochs(DropoutPositionDataset(use_compose=False))

    _assert_any_epoch_differs(first_batches)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_seeded_compose_dataloader_reproducible_with_same_torch_generator_seed():
    """Compose seed plus DataLoader generator seed should reproduce the worker-derived sequence."""
    first_run = _first_batches_across_epochs(
        DropoutPositionDataset(use_compose=True, seed=137),
        generator_seed=138,
    )
    second_run = _first_batches_across_epochs(
        DropoutPositionDataset(use_compose=True, seed=137),
        generator_seed=138,
    )

    assert first_run == second_run


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_seeded_compose_dataloader_seed_changes_child_random_params():
    """Changing Compose seed should change child transform params under the same worker seeds."""
    seed_137_batch = _first_batches_across_epochs(
        DropoutPositionDataset(use_compose=True, seed=137),
        generator_seed=140,
        epochs=1,
    )[0]
    seed_138_batch = _first_batches_across_epochs(
        DropoutPositionDataset(use_compose=True, seed=138),
        generator_seed=140,
        epochs=1,
    )[0]

    assert seed_137_batch != seed_138_batch


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_persistent_workers_advance_rng_across_epochs():
    """Persistent workers should keep advancing RNG instead of resetting to worker seed every call."""
    first_batches = _first_batches_across_epochs(
        DropoutPositionDataset(use_compose=True, seed=137),
        generator_seed=139,
        num_workers=2,
        persistent_workers=True,
    )

    _assert_any_epoch_differs(first_batches)


def test_compose_serialization():
    """Test that Compose serialization works properly."""
    # Test with worker-aware seed (always enabled)
    transform1 = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
        ],
        seed=137,
    )

    # Serialize and deserialize
    serialized = transform1.to_dict()

    # Test deserialization
    transform2 = A.from_dict(serialized)
    assert hasattr(transform2, "seed")
    assert transform2.seed == 137


def test_compose_serialization_preserves_base_seed_after_manual_random_state():
    """Manual RNG state should not rewrite the user seed that serialization exposes."""
    transform = A.Compose([A.HorizontalFlip(p=0.5)], seed=137)
    transform.set_random_state(np.random.default_rng(138), random.Random(139))

    serialized = transform.to_dict()

    assert serialized["transform"]["seed"] == 137


def test_derive_effective_seed():
    """Effective seeds should cleanly separate user seed from worker seed."""
    assert _derive_effective_seed(None, None) is None
    assert _derive_effective_seed(137, None) == 137
    assert _derive_effective_seed(None, 138) == 138
    assert _derive_effective_seed(137, 138) == 275
    assert _derive_effective_seed(2**32 - 1, 2) == 1


def test_unpickled_compose_resets_runtime_context():
    """Pickled Compose objects should re-sync against the worker context after unpickling."""
    transform = A.Compose([A.HorizontalFlip(p=0.5)], seed=137)
    transform._rng_context = _RuntimeRngContext(worker_seed=138, effective_seed=275)

    restored = pickle.loads(pickle.dumps(transform))  # noqa: S301 - controlled round-trip in a test

    assert restored.seed == 137
    assert restored._base_seed == 137
    assert restored._rng_context is None


def test_unpickled_basic_transform_resets_runtime_context():
    """Pickled direct transforms should re-sync against the worker context after unpickling."""
    transform = A.CoarseDropout(p=1.0)
    transform._rng_context = _RuntimeRngContext(worker_seed=138, effective_seed=138)

    restored = pickle.loads(pickle.dumps(transform))  # noqa: S301 - controlled round-trip in a test

    assert restored._base_seed is None
    assert restored._rng_context is None


def test_manual_compose_random_state_disables_worker_sync(monkeypatch):
    """Explicit Compose RNG objects should stay under user control in worker contexts."""
    transform = A.Compose([A.HorizontalFlip(p=0.5)], seed=137)
    transform.set_random_state(np.random.default_rng(138), random.Random(139))
    original_random_generator = transform.random_generator
    monkeypatch.setattr(
        "albumentations.core.composition._get_runtime_rng_context",
        lambda _base_seed: _RuntimeRngContext(worker_seed=140, effective_seed=277),
    )

    transform._sync_runtime_random_state()

    assert transform.random_generator is original_random_generator


def test_child_transform_preserves_parent_seeded_runtime_context(monkeypatch):
    """Child transforms should keep the parent Compose effective seed in the same worker."""

    def runtime_context(base_seed: int | None) -> _RuntimeRngContext:
        effective_seed = _derive_effective_seed(base_seed, 140)
        assert effective_seed is not None
        return _RuntimeRngContext(
            worker_seed=140,
            effective_seed=effective_seed,
        )

    monkeypatch.setattr("albumentations.core.composition._get_runtime_rng_context", runtime_context)
    monkeypatch.setattr("albumentations.core.transforms_interface._get_runtime_rng_context", runtime_context)
    transform = A.Compose([A.HorizontalFlip(p=1.0)], seed=137)
    expected_context = _RuntimeRngContext(worker_seed=140, effective_seed=277)

    transform._sync_runtime_random_state()
    child_transform = transform.transforms[0]

    assert child_transform._rng_context == expected_context

    child_transform._sync_runtime_random_state()

    assert child_transform._rng_context == expected_context


def test_manual_basic_transform_random_state_disables_worker_sync(monkeypatch):
    """Explicit direct-transform RNG objects should stay under user control in worker contexts."""
    transform = A.HorizontalFlip(p=0.5)
    transform.set_random_state(np.random.default_rng(138), random.Random(139))
    original_random_generator = transform.random_generator
    monkeypatch.setattr(
        "albumentations.core.transforms_interface._get_runtime_rng_context",
        lambda _base_seed: _RuntimeRngContext(worker_seed=140, effective_seed=140),
    )

    transform._sync_runtime_random_state()

    assert transform.random_generator is original_random_generator


def test_effective_seed_calculation():
    """Test the _get_effective_seed method directly."""
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
        ],
        seed=137,
    )

    # Test with None seed
    assert transform._get_effective_seed(None) is None

    # Test without worker context
    assert transform._get_effective_seed(137) == 137

    # Test seed overflow
    large_seed = 2**32 - 1
    result = transform._get_effective_seed(large_seed)
    assert 0 <= result < 2**32


def test_deterministic_behavior_single_process():
    """Test that transforms are deterministic in a single process."""
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        seed=137,
    )

    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Reset seed and get results
    results = []
    for _ in range(3):
        transform.set_random_seed(137)
        result = transform(image=img.copy())
        results.append(result["image"])

    # All results should be identical
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_deterministic_behavior_property():
    """Property test: same seed always produces same result for any image."""
    import hypothesis.strategies as st
    from hypothesis import given, settings
    from hypothesis.extra import numpy as npst

    @given(
        image=npst.arrays(
            dtype=np.uint8,
            shape=st.tuples(
                st.integers(20, 100),  # height
                st.integers(20, 100),  # width
                st.just(3),  # RGB channels
            ),
            elements=st.integers(0, 255),
        ),
        seed=st.integers(0, 10000),
    )
    @settings(max_examples=30, deadline=3000)
    def property_test(image, seed):
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ],
            seed=seed,
        )

        # Apply twice with same seed - must produce identical results
        transform.set_random_seed(seed)
        result1 = transform(image=image.copy())["image"]

        transform.set_random_seed(seed)
        result2 = transform(image=image.copy())["image"]

        np.testing.assert_array_equal(result1, result2)

    property_test()


def test_multiple_compose_instances():
    """Test that multiple Compose instances with same seed produce same results."""
    # Create two instances with same configuration
    transform1 = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(angle_range=(-45, 45), p=0.5),
        ],
        seed=137,
    )

    transform2 = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(angle_range=(-45, 45), p=0.5),
        ],
        seed=137,
    )

    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Both should produce the same result
    result1 = transform1(image=img.copy())
    result2 = transform2(image=img.copy())

    np.testing.assert_array_equal(result1["image"], result2["image"])


def worker_process_simulation(worker_id: int, base_seed: int, num_iterations: int) -> list[bool]:
    """Simulate a worker process with given ID and seed.

    Returns list of booleans indicating whether HorizontalFlip was applied.
    """
    # Each worker uses a different seed to simulate worker diversity
    # This simulates what would happen with torch.initial_seed()
    # Use a hash to get more diverse seeds
    import hashlib

    worker_seed = int(hashlib.md5(f"{base_seed}_{worker_id}".encode()).hexdigest()[:8], 16)

    # Create transform with a unique seed per worker
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
        ],
        seed=worker_seed,
    )  # Simulating worker-aware behavior

    # Run iterations
    results = []
    # Create an asymmetric image so we can detect flips
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :5] = 255  # Left half white

    for _ in range(num_iterations):
        result = transform(image=img.copy())
        # Check if image was flipped by checking left corner
        was_flipped = result["image"][0, 0, 0] == 0  # If flipped, left corner will be black
        results.append(was_flipped)

    return results


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Multiprocessing test skipped on Windows",
)
def test_worker_seed_diversity():
    """Test that different workers produce different augmentation sequences."""
    base_seed = 137
    num_workers = 4
    num_iterations = 20

    # Run simulation for each worker
    with multiprocessing.Pool(processes=num_workers) as pool:
        worker_results = []
        for worker_id in range(num_workers):
            result = pool.apply_async(
                worker_process_simulation,
                args=(worker_id, base_seed, num_iterations),
            )
            worker_results.append(result)

        # Collect results
        sequences = [result.get() for result in worker_results]

    # Check that workers produced different sequences
    unique_sequences = {tuple(seq) for seq in sequences}
    assert len(unique_sequences) > 1, "All workers produced identical augmentation sequences"

    # Each worker should have some flips and some non-flips (with high probability)
    for worker_id, sequence in enumerate(sequences):
        num_flips = sum(sequence)
        assert 0 < num_flips < num_iterations, (
            f"Worker {worker_id} produced extreme results: {num_flips}/{num_iterations} flips"
        )
