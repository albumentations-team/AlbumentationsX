---
name: benchmark
description: Run performance benchmarks for transform changes. Use when the user asks to benchmark, measure performance, compare speed, or when changes affect apply methods, functional layer, get_params, or core pipeline code.
---

# Benchmark

Any change touching `apply_*`, `functional.py`, `get_params`, `get_params_dependent_on_data`, `composition.py`, or `transforms_interface.py` **must** include benchmark results.

## Standard Matrix

Always benchmark all 9 combinations:

| Size       | Channels | Use case                              |
|------------|----------|---------------------------------------|
| 256×256    | 1        | Grayscale classification              |
| 256×256    | 3        | RGB classification                    |
| 256×256    | 5        | Multispectral                         |
| 512×512    | 1        | Depth maps                            |
| 512×512    | 3        | Detection/segmentation (YOLO, U-Net)  |
| 512×512    | 5        | Multispectral segmentation            |
| 1024×1024  | 1        | Medical imaging                       |
| 1024×1024  | 3        | High-res segmentation                 |
| 1024×1024  | 5        | Satellite imagery                     |

Skip channel counts the transform explicitly doesn't support.

## Template: Isolated Function

```python
import timeit
import numpy as np

SIZES = {"small": (256, 256), "medium": (512, 512), "large": (1024, 1024)}
CHANNELS = [1, 3, 5]
N = 100

for size_name, (h, w) in SIZES.items():
    for ch in CHANNELS:
        shape = (h, w) if ch == 1 else (h, w, ch)
        img = np.random.randint(0, 256, shape, dtype=np.uint8)

        old_t = timeit.timeit(lambda img=img: old_func(img, **params), number=N)
        new_t = timeit.timeit(lambda img=img: new_func(img, **params), number=N)
        print(f"{size_name} {h}x{w}x{ch}: old={old_t:.4f}s new={new_t:.4f}s speedup={old_t/new_t:.2f}x")
```

## Template: Full Pipeline (Compose)

```python
import timeit
import numpy as np
import albumentations as A

SIZES = {"small": (256, 256), "medium": (512, 512), "large": (1024, 1024)}
CHANNELS = [1, 3, 5]

transform = A.Compose([A.YourTransform(p=1.0)])

for size_name, (h, w) in SIZES.items():
    for ch in CHANNELS:
        shape = (h, w) if ch == 1 else (h, w, ch)
        img = np.random.randint(0, 256, shape, dtype=np.uint8)

        t = timeit.timeit(lambda img=img: transform(image=img), number=100)
        print(f"{size_name} {h}x{w}x{ch}: {t:.4f}s (100 calls)")
```

## Workflow

1. **Before**: run benchmark on the current `main` / original code, save output to a JSON file
2. **After**: run benchmark on the modified code, save output to a JSON file
3. **Compare**: load both JSON files, compute speedup = old_time / new_time for each transform/size combo
4. **Report** results in the PR/commit message body

### JSON Output Format

Save benchmark results as JSON for automated comparison:

```python
import json

results = {}
for transform_name, (h, w), ch, elapsed in all_results:
    key = f"{transform_name}_{h}x{w}x{ch}"
    results[key] = {"time": elapsed, "iterations": N}

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Comparison Script Pattern

```python
import json

with open("bench_old.json") as f:
    old = json.load(f)
with open("bench_new.json") as f:
    new = json.load(f)

for key in sorted(old):
    if key in new:
        speedup = old[key]["time"] / new[key]["time"]
        indicator = "FASTER" if speedup > 1.05 else "SLOWER" if speedup < 0.95 else "SAME"
        print(f"{key}: {old[key]['time']:.4f}s -> {new[key]['time']:.4f}s  {speedup:.2f}x  {indicator}")
```

## Reporting Format

```
Benchmark (uint8, 100 iterations):

Function direct:
  256x256x1   — Before: 0.0200s After: 0.0100s Speedup: 2.00x
  256x256x3   — Before: 0.0500s After: 0.0300s Speedup: 1.67x
  ...

Compose single:
  256x256x1   — 0.0120s
  256x256x3   — 0.0340s
  ...
```

## Template: Batch (apply_to_images)

When benchmarking batch optimizations (kernel pre-computation, 4D indexing, pre-allocated loops):

```python
import timeit
import numpy as np
import albumentations as A

BATCH_SIZES = [4, 8, 16]
SIZES = {"small": (256, 256), "medium": (512, 512)}

transform = A.Compose([A.YourTransform(p=1.0)])

for batch_size in BATCH_SIZES:
    for size_name, (h, w) in SIZES.items():
        # Grayscale batch — benefits from reshape trick
        images = [np.random.randint(0, 256, (h, w, 1), dtype=np.uint8) for _ in range(batch_size)]
        t = timeit.timeit(lambda: transform(images=images), number=50)
        print(f"batch={batch_size} {size_name} {h}x{w}x1: {t:.4f}s")

        # RGB batch — baseline
        images_rgb = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(batch_size)]
        t = timeit.timeit(lambda: transform(images=images_rgb), number=50)
        print(f"batch={batch_size} {size_name} {h}x{w}x3: {t:.4f}s")
```

## Rules

- Run on the **same machine**, back-to-back, same conditions
- Use at least **100 iterations** for fast functions; fewer for slow ones (aim for >1s total)
- Test **both uint8 and float32** if the change affects dtype handling
- A **>5% regression** on any combination requires justification or rework
- If adding a new transform, benchmark against the equivalent naive numpy implementation
- For batch optimizations, compare 1-channel vs 3-channel to verify speedup holds across channel counts
