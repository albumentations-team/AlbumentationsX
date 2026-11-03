"""Run one Linux memory workload; defer reporting imports until measurement ends."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import albumentations

cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

mode, case = sys.argv[1:3]
page_size = os.sysconf("SC_PAGE_SIZE")


def rss() -> int:
    with Path("/proc/self/statm").open() as handle:
        return int(handle.read().split()[1]) * page_size


if mode == "rss":
    pipe = (
        albumentations.Blur(p=1) if case == "direct" else albumentations.Compose([albumentations.Blur(p=1)], seed=137)
    )
    if case == "direct":
        pipe.set_random_seed(137)
    rng = np.random.default_rng(0)

    def call() -> None:
        height, width = (
            (1280, 1200) if case == "fixed" else (1344 - int(rng.integers(0, 160)), 1280 - int(rng.integers(0, 160)))
        )
        shape = (height, width, 1) if case == "hwc1" else (height, width)
        image = rng.random(shape, dtype=np.float32)
        if case == "image_mask":
            pipe(image=image, mask=rng.random(shape, dtype=np.float32))
        elif case == "explicit":
            pipe(image=image, invocation_seed=137)
        else:
            pipe(image=image)

    for _ in range(100):
        call()
    gc.collect()
    before = rss()
    for _ in range(800):
        call()
    gc.collect()
    after = rss()
    measurement = {"start_bytes": before, "end_bytes": after, "growth_bytes": after - before, "calls": 800}
elif mode == "peak":
    if case in {"blur", "blur_float16"}:
        dtype = np.float16 if case == "blur_float16" else np.float32
        pipe = albumentations.Compose([albumentations.Blur(p=1)], seed=137)
        small = {"image": np.ones((32, 32), dtype=dtype)}
        data = {"image": np.ones((1024, 1024), dtype=dtype)}
        target = "image"
    elif case == "volume_pad":
        albumentations.Compose([albumentations.PadIfNeeded3D(min_zyx=(2, 8, 8), p=1)], seed=137)(
            volume=np.ones((1, 4, 4, 1), dtype=np.float32),
        )
        pipe = albumentations.Compose([albumentations.PadIfNeeded3D(min_zyx=(24, 288, 288), p=1)], seed=137)
        small = None
        data = {"volume": np.ones((16, 256, 256, 1), dtype=np.float32)}
        target = "volume"
    else:
        pipe = albumentations.Compose([albumentations.Normalize(p=1)], seed=137)
        target = "images" if case == "batch_normalize" else "image"
        shape = (4, 512, 512, 3) if target == "images" else (1024, 1024, 3)
        small = {target: np.full((4, 16, 16, 3) if target == "images" else (32, 32, 3), 137, dtype=np.uint8)}
        data = {target: np.full(shape, 137, dtype=np.uint8)}
        if case == "tensor_normalize":
            small[target] = torch.from_numpy(small[target].transpose(2, 0, 1).copy())
            data[target] = torch.from_numpy(data[target].transpose(2, 0, 1).copy())
    if small is not None:
        pipe(**small)
    del small
    gc.collect()
    with Path("/proc/self/clear_refs").open("w") as handle:
        handle.write("5")
    before = rss()
    result = pipe(**data)
    with Path("/proc/self/status").open() as handle:
        peak = next(int(line.split()[1]) * 1024 for line in handle if line.startswith("VmHWM:"))
    output = result[target]
    measurement = {
        "start_bytes": before,
        "peak_bytes": peak,
        "working_peak_bytes": max(0, peak - before),
        "output_bytes": output.nbytes,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
    }
else:
    raise ValueError(f"Unknown memory mode: {mode}")


def report_measurement() -> None:
    """Import reporting code after allocator-sensitive measurement."""
    import hashlib
    import importlib.metadata
    import json
    import platform

    import albucore
    from threadpoolctl import threadpool_info

    source = Path(albumentations.__file__).resolve()
    expected_source = Path(__file__).resolve().parents[1] / "albumentations" / "__init__.py"
    if source != expected_source or not gc.isenabled():
        raise RuntimeError("Memory probe must import this checkout and leave cyclic GC enabled")
    if cv2.getNumThreads() != 1 or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Memory probe requires one compute thread")
    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise RuntimeError("Memory probe found a multithreaded numerical backend")
    print(
        json.dumps(
            {
                "mode": mode,
                "case": case,
                **measurement,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "libc": platform.libc_ver(),
                    "page_size": page_size,
                    "source": str(source),
                    "albucore_source": albucore.__file__,
                    "invocation_sha256": hashlib.sha256(
                        source.with_name("core").joinpath("invocation.py").read_bytes()
                    ).hexdigest(),
                    "versions": {
                        name: importlib.metadata.version(name)
                        for name in ("albucore", "numpy", "torch", "opencv-python-headless", "numkong", "stringzilla")
                    },
                },
            },
        ),
    )


report_measurement()
