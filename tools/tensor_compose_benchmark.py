"""Measure the shared Tensor-native Compose lifecycle once per infrastructure change.

Per-transform capability work must not run this DataLoader benchmark. It belongs
to Compose, bridge, batching, collation, and milestone changes; transform routes
use their direct and model-ready Compose benchmarks instead.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

import albumentations as A  # noqa: N812

InputRepresentation = Literal["numpy", "tensor"]
NumpyImages = NDArray[np.generic]


class _ComposeDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Picklable, pre-created inputs for steady-state DataLoader measurements."""

    def __init__(self, images: NumpyImages | torch.Tensor, representation: InputRepresentation) -> None:
        self.images = images
        self.representation = representation
        if representation == "numpy":
            self.transform = A.Compose([A.NoOp(p=1.0), A.ToTensorV2(p=1.0)], strict=True)
        else:
            self.transform = A.Compose([A.NoOp(p=1.0)], strict=True)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.transform(image=self.images[index])


def _raw_samples(callable_: Callable[[], None], repeats: int) -> list[float]:
    """Return millisecond samples after the caller has performed warm-up."""
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        callable_()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def _consume(loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]]) -> None:
    """Consume every collated batch so timing includes output construction."""
    for batch in loader:
        _ = batch["image"].sum().item()


def _compose_samples(
    image: NumpyImages,
    tensor: torch.Tensor,
    *,
    repeats: int,
    iterations: int,
) -> dict[str, list[float]]:
    numpy_transform = A.Compose([A.NoOp(p=1.0), A.ToTensorV2(p=1.0)], strict=True)
    tensor_transform = A.Compose([A.NoOp(p=1.0)], strict=True)

    numpy_result = numpy_transform(image=image)["image"]
    tensor_result = tensor_transform(image=tensor)["image"]
    torch.testing.assert_close(numpy_result, tensor_result)

    def time_transform(transform: A.Compose, value: NumpyImages | torch.Tensor) -> list[float]:
        for _ in range(16):
            transform(image=value)

        def run() -> None:
            for _ in range(iterations):
                transform(image=value)

        return [sample / iterations for sample in _raw_samples(run, repeats)]

    return {
        "numpy_with_to_tensor": time_transform(numpy_transform, image),
        "tensor_direct": time_transform(tensor_transform, tensor),
    }


def _dataloader_samples(
    images: NumpyImages | torch.Tensor,
    representation: InputRepresentation,
    *,
    batch_size: int,
    num_workers: int,
    repeats: int,
) -> list[float]:
    loader = torch.utils.data.DataLoader(
        _ComposeDataset(images, representation),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    try:
        _consume(loader)
        samples = _raw_samples(lambda: _consume(loader), repeats)
        return [sample / len(images) for sample in samples]
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, choices=(256, 512, 1024))
    parser.add_argument("--channels", type=int, default=3, choices=(1, 3, 5))
    parser.add_argument("--dtype", default="uint8", choices=("uint8", "float32"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", default="0,2,8")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--skip-dataloader", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and write an inspectable route-result JSON artifact."""
    args = _parse_args()
    if args.samples <= 0 or args.batch_size <= 0 or args.repeats <= 0 or args.iterations <= 0:
        raise ValueError("samples, batch-size, repeats, and iterations must be positive")
    worker_counts = tuple(int(value) for value in args.workers.split(","))
    if any(value < 0 for value in worker_counts):
        raise ValueError("worker counts must be non-negative")

    np_dtype = np.dtype(args.dtype)
    generator = np.random.default_rng(137)
    images: NumpyImages = generator.integers(
        0,
        256,
        size=(args.samples, args.size, args.size, args.channels),
        dtype=np.uint8,
    )
    if np_dtype == np.dtype("float32"):
        images = images.astype(np.float32) / np.float32(255)
    tensors = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))

    result: dict[str, Any] = {
        "schema_version": 1,
        "environment": {
            "albumentations": A.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "input": {
            "channels": args.channels,
            "dtype": args.dtype,
            "shape_numpy": list(images.shape[1:]),
            "shape_tensor": list(tensors.shape[1:]),
        },
        "bridges": [],
        "compose_model_ready_ms": _compose_samples(
            images[0], tensors[0], repeats=args.repeats, iterations=args.iterations
        ),
    }
    if not args.skip_dataloader:
        result["dataloader_model_ready_ms_per_sample"] = {
            str(workers): {
                "numpy_with_to_tensor": _dataloader_samples(
                    images,
                    "numpy",
                    batch_size=args.batch_size,
                    num_workers=workers,
                    repeats=args.repeats,
                ),
                "tensor_direct": _dataloader_samples(
                    tensors,
                    "tensor",
                    batch_size=args.batch_size,
                    num_workers=workers,
                    repeats=args.repeats,
                ),
            }
            for workers in worker_counts
        }
    result["correctness"] = "passed"
    result["decision"] = "record evidence; accept no route from one machine alone"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Tensor Compose benchmark evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
