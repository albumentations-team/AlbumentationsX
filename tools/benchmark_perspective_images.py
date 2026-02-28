"""Benchmark: perspective_images approaches.

Old:  @batch_transform("spatial") — reshape (N,H,W,C) -> (H,W,N*C), one warpPerspective call
New:  per-frame loop with warp_perspective + dst= pre-allocation
Opt:  hybrid — 1ch batches use old stacking trick (one call on H,W,N), multi-ch use per-frame loop
"""

import timeit

import cv2
import numpy as np
from albucore import warp_perspective


# ─── old: batch_transform("spatial") path ────────────────────────────────────

def _old_perspective_images(
    images: np.ndarray, matrix: np.ndarray, max_width: int, max_height: int,
    border_val: float, border_mode: int, keep_size: bool, interpolation: int,
) -> np.ndarray:
    n, height, width, channels = images.shape
    flat = np.moveaxis(images, 0, -2).reshape(height, width, -1)
    if keep_size:
        adj = np.array([[width / max_width, 0, 0], [0, height / max_height, 0], [0, 0, 1]]) @ matrix
        dsize = (width, height)
    else:
        adj = matrix
        dsize = (max_width, max_height)
    warped = warp_perspective(flat, adj, dsize, flags=interpolation, border_mode=border_mode, border_value=border_val)
    out_h, out_w = warped.shape[:2]
    return np.moveaxis(warped.reshape(out_h, out_w, n, channels), -2, 0)


# ─── new: per-frame warp_perspective + dst= ──────────────────────────────────

def _new_perspective_images(
    images: np.ndarray, matrix: np.ndarray, max_width: int, max_height: int,
    border_val: float, border_mode: int, keep_size: bool, interpolation: int,
) -> np.ndarray:
    height, width = images.shape[1], images.shape[2]
    if keep_size:
        adjusted_matrix = np.array([[width / max_width, 0, 0], [0, height / max_height, 0], [0, 0, 1]]) @ matrix
        dsize = (width, height)
        result = np.empty_like(images)
    else:
        adjusted_matrix = matrix
        dsize = (max_width, max_height)
        result = np.empty((images.shape[0], max_height, max_width, *images.shape[3:]), dtype=images.dtype)
    for i in range(images.shape[0]):
        warp_perspective(
            images[i], adjusted_matrix, dsize,
            flags=interpolation, border_mode=border_mode, border_value=border_val, dst=result[i],
        )
    return result


# ─── optimized: stack 1ch frames, per-frame loop for multi-ch ────────────────

def _opt_perspective_images(
    images: np.ndarray, matrix: np.ndarray, max_width: int, max_height: int,
    border_val: float, border_mode: int, keep_size: bool, interpolation: int,
) -> np.ndarray:
    height, width = images.shape[1], images.shape[2]
    if keep_size:
        adjusted_matrix = np.array([[width / max_width, 0, 0], [0, height / max_height, 0], [0, 0, 1]]) @ matrix
        dsize = (width, height)
    else:
        adjusted_matrix = matrix
        dsize = (max_width, max_height)

    n = images.shape[0]
    num_channels = 1 if images.ndim == 3 else images.shape[3]

    border_val_cv2 = int(border_val) if isinstance(border_val, float) and border_val == int(border_val) else border_val

    if num_channels == 1:
        STACK_THRESHOLD = 256 * 256  # pixels
        flat = images if images.ndim == 3 else images[:, :, :, 0]       # (N,H,W)
        if height * width <= STACK_THRESHOLD:
            stacked = np.ascontiguousarray(flat.transpose(1, 2, 0))     # (H,W,N) — 1 copy
            border_scalar = border_val[0] if isinstance(border_val, (list, np.ndarray)) else border_val
            warped = warp_perspective(
                stacked, adjusted_matrix, dsize,
                flags=interpolation, border_mode=border_mode, border_value=border_scalar,
            )
            out = np.moveaxis(warped, -1, 0)                            # view (N,H',W') — free
        else:
            out = np.empty((n, dsize[1], dsize[0]), dtype=images.dtype)
            for i in range(n):
                cv2.warpPerspective(
                    flat[i], adjusted_matrix, dsize, dst=out[i],
                    flags=interpolation, borderMode=border_mode, borderValue=border_val_cv2,
                )
        return out[:, :, :, np.newaxis] if images.ndim == 4 else out

    result = np.empty((n, dsize[1], dsize[0]) + images.shape[3:], dtype=images.dtype)
    if num_channels <= 4:
        for i in range(n):
            cv2.warpPerspective(
                images[i], adjusted_matrix, dsize, dst=result[i],
                flags=interpolation, borderMode=border_mode, borderValue=border_val_cv2,
            )
    else:
        for i in range(n):
            warp_perspective(
                images[i], adjusted_matrix, dsize,
                flags=interpolation, border_mode=border_mode, border_value=border_val, dst=result[i],
            )
    return result


# ─── helpers ─────────────────────────────────────────────────────────────────

def make_matrix(h: int, w: int) -> tuple[np.ndarray, int, int]:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    rng = np.random.default_rng(137)
    jitter = (rng.uniform(-0.1, 0.1, src.shape) * np.array([w, h])).astype(np.float32)
    dst = np.clip(src + jitter, 0, [w, h]).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return matrix, int(w * 0.9), int(h * 0.9)


BATCH_SIZE = 16
SIZES = [("256x256", 256, 256), ("512x512", 512, 512), ("1024x1024", 1024, 1024)]
CHANNELS = [1, 3, 5]
DTYPES = [("uint8", np.uint8), ("float32", np.float32)]
N_ITER = 30
WARMUP = 3


def run_bench() -> None:
    rng = np.random.default_rng(137)

    for keep_size in [True, False]:
        print(f"\n{'='*80}")
        print(f"keep_size={keep_size}")
        print(f"{'='*80}")
        print(f"{'Config':<26} {'Old':>8} {'New':>8} {'Opt':>8} {'Opt/Old':>8} {'Opt/New':>8}")
        print(f"{'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for dtype_name, dtype in DTYPES:
            for size_name, h, w in SIZES:
                matrix, max_w, max_h = make_matrix(h, w)

                for ch in CHANNELS:
                    shape = (BATCH_SIZE, h, w, ch)
                    images = rng.integers(0, 256, shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)

                    kw = {
                        "matrix": matrix, "max_width": max_w, "max_height": max_h,
                        "border_val": 0.0, "border_mode": cv2.BORDER_CONSTANT,
                        "keep_size": keep_size, "interpolation": cv2.INTER_LINEAR,
                    }

                    for _ in range(WARMUP):
                        _old_perspective_images(images, **kw)
                        _new_perspective_images(images, **kw)
                        _opt_perspective_images(images, **kw)

                    old_t = timeit.timeit(lambda imgs=images, k=kw: _old_perspective_images(imgs, **k), number=N_ITER)
                    new_t = timeit.timeit(lambda imgs=images, k=kw: _new_perspective_images(imgs, **k), number=N_ITER)
                    opt_t = timeit.timeit(lambda imgs=images, k=kw: _opt_perspective_images(imgs, **k), number=N_ITER)

                    label = f"{dtype_name} {size_name}x{ch}ch"
                    print(f"{label:<26} {old_t:>8.3f} {new_t:>8.3f} {opt_t:>8.3f} {old_t/opt_t:>7.2f}x {new_t/opt_t:>7.2f}x")

    print()


if __name__ == "__main__":
    from albumentations._version import __version__
    print(f"albumentations version: {__version__}")
    print(f"Batch size: {BATCH_SIZE}, Iterations: {N_ITER}, Warmup: {WARMUP}")
    run_bench()
