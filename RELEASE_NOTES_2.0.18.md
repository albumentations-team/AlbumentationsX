# AlbumentationsX 2.0.18 Release Notes

## New Transform: [PhotoMetricDistort](https://explore.albumentations.ai/transform/PhotoMetricDistort)

`PhotoMetricDistort` implements the photometric distortion pipeline from the [SSD paper](https://arxiv.org/abs/1512.02325), matching the API and default parameters of torchvision's `RandomPhotometricDistort`.

Each of the five distortions — brightness, contrast, saturation, hue, and channel shuffle — is applied independently with probability `distort_p`. Contrast placement is randomized: it can appear either before or after the HSV-space adjustments (saturation + hue), mirroring the SSD paper's stochastic ordering.

```python
import albumentations as A

transform = A.PhotoMetricDistort(
    brightness_range=(0.875, 1.125),  # multiplicative factor
    contrast_range=(0.5, 1.5),        # multiplicative factor
    saturation_range=(0.5, 1.5),      # multiplicative factor
    hue_range=(-0.05, 0.05),          # additive factor, range [-0.5, 0.5]
    distort_p=0.5,                    # probability per individual distortion
    p=0.5,                            # probability the whole transform runs
)
```

**Key differences from torchvision:**

- torchvision uses a single `p` that applies to each distortion; AlbumentationsX separates `distort_p` (per-distortion) from `p` (the overall transform gate), giving independent control.
- All default parameter values are identical to torchvision's `RandomPhotometricDistort`.

**Supported targets:** image, volume
**Supported dtypes:** uint8, float32
**Channels:** 1 (grayscale) and 3 (RGB) only — same constraint as torchvision

---

### [ColorJitter](https://explore.albumentations.ai/transform/ColorJitter): Fused Brightness + Contrast

As part of the same PR, `ColorJitter` received a performance improvement. The four color operations are applied in a random order each call. When brightness and contrast happen to be adjacent in the shuffle (approximately 50% of calls, since 12 of the 24 possible orderings place them next to each other), they are now fused into a single operation:

- **uint8**: the two transforms are composed analytically into a single 256-entry LUT applied in one `cv2.LUT` call instead of two sequential passes
- **float32**: a single pre-allocated output buffer with in-place numpy ops avoids 4+ intermediate array allocations

| dtype   | v2.0.17 (100 calls) | v2.0.18 (100 calls) | speedup |
|---------|---------------------|---------------------|---------|
| uint8   | 0.173s              | 0.162s              | 1.07x   |
| float32 | 0.749s              | 0.689s              | 1.09x   |

The speedup is modest in the aggregate because the fusion only applies to the ~50% of calls where brightness and contrast are adjacent; the other ~50% are unchanged.

---

## Performance: Multi-Channel Speedups (5+ Channels)

AlbumentationsX now requires **OpenCV ≥ 4.13** and **albucore ≥ 0.0.39**. OpenCV 4.13 extended native multi-channel support for several operations. The previous code always split images with more than 4 channels into ≤4-channel chunks and processed them sequentially. The new code calls OpenCV directly when supported, falling back to chunking only when genuinely required.

All benchmarks: 512×512 images, 100 iterations, Apple M-series, macOS.

---

### [Blur](https://explore.albumentations.ai/transform/Blur) (`cv2.blur`)

`cv2.blur` has always supported arbitrary channel counts natively — the chunking was unnecessary. The new code calls it directly with `dst=img` (in-place).

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.013s  | 0.013s  | 1.0x    |
| 3        | 0.026s  | 0.027s  | 1.0x    |
| 5        | 0.230s  | 0.058s  | **4.0x** |
| 8        | 0.411s  | 0.108s  | **3.8x** |
| 16       | 0.834s  | 0.217s  | **3.8x** |
| 32       | 1.819s  | 0.447s  | **4.1x** |
| 64       | 3.246s  | 0.785s  | **4.1x** |
| 128      | 10.145s | 3.730s  | **2.7x** |

---

### [MedianBlur](https://explore.albumentations.ai/transform/MedianBlur)

OpenCV 4.13 added native multi-channel support for `cv2.medianBlur` when `ksize` is 3 or 5. For `ksize ≥ 7`, OpenCV's internal SIMD path still asserts `channels ≤ 4`, so chunking remains necessary there.

**ksize = 5 — native multi-channel path (fast):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.026s  | 0.023s  | 1.1x    |
| 3        | 0.046s  | 0.045s  | 1.0x    |
| 5        | 0.241s  | 0.152s  | **1.6x** |
| 8        | 0.388s  | 0.257s  | **1.5x** |
| 16       | 0.780s  | 0.534s  | **1.5x** |
| 32       | 1.673s  | 1.112s  | **1.5x** |
| 64       | 3.043s  | 1.888s  | **1.6x** |
| 128      | 9.478s  | 3.901s  | **2.4x** |

**ksize = 7 — still chunked for >4 channels (no change expected):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 5        | 0.264s  | 0.251s  | 1.0x    |
| 8        | 0.435s  | 0.454s  | 1.0x    |
| 16       | 0.871s  | 0.930s  | 1.0x    |
| 32       | 1.900s  | 1.705s  | 1.0x    |
| 64       | 3.306s  | 3.461s  | 1.0x    |
| 128      | 10.544s | 11.500s | 1.0x    |

---

### [Affine](https://explore.albumentations.ai/transform/Affine), [Rotate](https://explore.albumentations.ai/transform/Rotate), [ShiftScaleRotate](https://explore.albumentations.ai/transform/ShiftScaleRotate), [SafeRotate](https://explore.albumentations.ai/transform/SafeRotate) (warp_affine)

These transforms now route through `albucore.warp_affine`, which calls `cv2.warpAffine` directly for multi-channel images when the interpolation mode is `INTER_NEAREST`, `INTER_LINEAR`, or `INTER_AREA`. For `INTER_CUBIC`, `INTER_LANCZOS4`, and `INTER_LINEAR_EXACT`, chunking is still required.

**The default interpolation is `INTER_LINEAR`, so most users get the speedup automatically.**

**[Affine](https://explore.albumentations.ai/transform/Affine) (scale + rotate, INTER_LINEAR — default):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.099s  | 0.024s  | 4.1x    |
| 3        | 0.053s  | 0.025s  | 2.1x    |
| 5        | 0.254s  | 0.055s  | **4.6x** |
| 8        | 0.389s  | 0.046s  | **8.5x** |
| 16       | 0.745s  | 0.059s  | **12.6x** |
| 32       | 1.622s  | 0.100s  | **16.2x** |
| 64       | 3.191s  | 0.125s  | **25.5x** |
| 128      | 10.273s | 0.272s  | **37.8x** |

**[Affine](https://explore.albumentations.ai/transform/Affine) (INTER_CUBIC — still chunked for >4 channels):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 5        | 0.339s  | 0.325s  | 1.0x    |
| 8        | 0.647s  | 0.533s  | 1.2x    |
| 16       | 1.133s  | 1.071s  | 1.1x    |
| 32       | 2.330s  | 2.266s  | 1.0x    |
| 64       | 4.329s  | 4.306s  | 1.0x    |
| 128      | 12.197s | 13.501s | 1.0x    |

**[Rotate](https://explore.albumentations.ai/transform/Rotate) (INTER_LINEAR):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.021s  | 0.035s  | 0.6x    |
| 3        | 0.027s  | 0.037s  | 0.7x    |
| 5        | 0.240s  | 0.142s  | **1.7x** |
| 8        | 0.496s  | 0.063s  | **7.9x** |
| 16       | 0.834s  | 0.061s  | **13.7x** |
| 32       | 1.554s  | 0.103s  | **15.1x** |
| 64       | 2.957s  | 0.166s  | **17.8x** |
| 128      | 9.115s  | 0.294s  | **31.0x** |

**[ShiftScaleRotate](https://explore.albumentations.ai/transform/ShiftScaleRotate) (INTER_LINEAR):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 5        | 0.239s  | 0.150s  | **1.6x** |
| 8        | 0.407s  | 0.092s  | **4.4x** |
| 16       | 0.764s  | 0.069s  | **11.1x** |
| 32       | 1.592s  | 0.124s  | **12.8x** |
| 64       | 2.993s  | 0.201s  | **14.9x** |
| 128      | 9.504s  | 0.333s  | **28.5x** |

**[SafeRotate](https://explore.albumentations.ai/transform/SafeRotate) (INTER_LINEAR):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 5        | 0.239s  | 0.058s  | **4.1x** |
| 8        | 0.392s  | 0.049s  | **8.0x** |
| 16       | 0.764s  | 0.066s  | **11.6x** |
| 32       | 1.660s  | 0.109s  | **15.2x** |
| 64       | 3.040s  | 0.144s  | **21.1x** |
| 128      | 9.559s  | 0.330s  | **29.0x** |

---

### [Perspective](https://explore.albumentations.ai/transform/Perspective) (warp_perspective)

Same interpolation-mode rules as `warp_affine`.

**[Perspective](https://explore.albumentations.ai/transform/Perspective) (INTER_LINEAR — default):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.045s  | 0.049s  | 1.0x    |
| 3        | 0.030s  | 0.033s  | 0.9x    |
| 5        | 0.248s  | 0.064s  | **3.9x** |
| 8        | 0.399s  | 0.057s  | **7.0x** |
| 16       | 0.779s  | 0.068s  | **11.5x** |
| 32       | 1.823s  | 0.108s  | **16.9x** |
| 64       | 3.142s  | 0.205s  | **15.3x** |
| 128      | 9.644s  | 0.460s  | **21.0x** |

**[Perspective](https://explore.albumentations.ai/transform/Perspective) (INTER_CUBIC — still chunked for >4 channels):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 5        | 0.423s  | 0.444s  | 1.0x    |
| 8        | 0.741s  | 0.652s  | 1.1x    |
| 16       | 1.099s  | 1.128s  | 1.0x    |
| 32       | 2.412s  | 2.647s  | 0.9x    |
| 64       | 4.407s  | 4.909s  | 0.9x    |
| 128      | 12.777s | 14.327s | 0.9x    |

---

### [ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform), [GridDistortion](https://explore.albumentations.ai/transform/GridDistortion), [OpticalDistortion](https://explore.albumentations.ai/transform/OpticalDistortion), [ThinPlateSpline](https://explore.albumentations.ai/transform/ThinPlateSpline), [PiecewiseAffine](https://explore.albumentations.ai/transform/PiecewiseAffine) (remap)

All transforms that inherit from `BaseDistortion` route through `albucore.remap`. The native multi-channel path is available for `INTER_NEAREST`, `INTER_LINEAR`, `INTER_AREA`, and `INTER_LINEAR_EXACT`. For `INTER_CUBIC` and `INTER_LANCZOS4`, chunking remains necessary.

Note: [ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform) dominates its time on `cv2.remap` but also generates random displacement fields — the speedup for high channel counts reflects both the faster remap and that field generation is channel-independent. [GridDistortion](https://explore.albumentations.ai/transform/GridDistortion) and [OpticalDistortion](https://explore.albumentations.ai/transform/OpticalDistortion) show even larger speedups because their maps are smaller and the remap dominates.

**[ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform) (INTER_LINEAR):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 1.488s  | 1.477s  | 1.0x    |
| 3        | 1.498s  | 1.452s  | 1.0x    |
| 5        | 1.599s  | 1.557s  | 1.0x    |
| 8        | 1.688s  | 1.471s  | 1.1x    |
| 16       | 1.891s  | 1.618s  | 1.2x    |
| 32       | 2.280s  | 1.484s  | **1.5x** |
| 64       | 2.993s  | 1.669s  | **1.8x** |
| 128      | 6.025s  | 1.535s  | **3.9x** |

**[ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform) (INTER_CUBIC — still chunked for >4 channels):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 5        | 1.642s  | 1.618s  | 1.0x    |
| 8        | 1.748s  | 1.838s  | 1.0x    |
| 16       | 2.033s  | 2.148s  | 1.0x    |
| 32       | 2.574s  | 2.635s  | 1.0x    |
| 64       | 3.606s  | 3.824s  | 0.9x    |
| 128      | 7.409s  | 8.672s  | 0.9x    |

**[GridDistortion](https://explore.albumentations.ai/transform/GridDistortion) (INTER_LINEAR):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.121s  | 0.057s  | 2.1x    |
| 3        | 0.131s  | 0.071s  | 1.8x    |
| 5        | 0.353s  | 0.104s  | **3.4x** |
| 8        | 0.518s  | 0.106s  | **4.9x** |
| 16       | 0.937s  | 0.119s  | **7.9x** |
| 32       | 1.725s  | 0.149s  | **11.6x** |
| 64       | 3.186s  | 0.152s  | **20.9x** |
| 128      | 9.466s  | 0.238s  | **39.8x** |

**[OpticalDistortion](https://explore.albumentations.ai/transform/OpticalDistortion) (INTER_LINEAR):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.129s  | 0.079s  | 1.6x    |
| 3        | 0.135s  | 0.084s  | 1.6x    |
| 5        | 0.352s  | 0.123s  | **2.9x** |
| 8        | 0.520s  | 0.111s  | **4.7x** |
| 16       | 0.930s  | 0.131s  | **7.1x** |
| 32       | 1.725s  | 0.169s  | **10.2x** |
| 64       | 3.205s  | 0.177s  | **18.1x** |
| 128      | 10.661s | 0.268s  | **39.8x** |

---

### [Pad](https://explore.albumentations.ai/transform/Pad), [PadIfNeeded](https://explore.albumentations.ai/transform/PadIfNeeded), [CenterCrop](https://explore.albumentations.ai/transform/CenterCrop), [RandomCrop](https://explore.albumentations.ai/transform/RandomCrop), [Crop](https://explore.albumentations.ai/transform/Crop), [CropAndPad](https://explore.albumentations.ai/transform/CropAndPad) (copy_make_border)

All padding operations now route through `albucore.copy_make_border`. For constant-value padding with a scalar fill (the most common case), `cv2.copyMakeBorder` supports arbitrary channel counts natively — no chunking needed. Per-channel fill with more than 4 distinct values still uses chunking.

**[PadIfNeeded](https://explore.albumentations.ai/transform/PadIfNeeded) (scalar fill, `fill=0`):**

| channels | v2.0.17 | v2.0.18  | speedup   |
|----------|---------|----------|-----------|
| 1        | 0.002s  | 0.002s   | 1.0x      |
| 3        | 0.004s  | 0.003s   | 1.3x      |
| 5        | 0.253s  | 0.015s   | **16.9x** |
| 8        | 0.448s  | 0.027s   | **16.9x** |
| 16       | 0.885s  | 0.050s   | **17.7x** |
| 32       | 1.623s  | 0.017s   | **98.1x** |
| 64       | 3.260s  | 0.041s   | **79.3x** |
| 128      | 11.453s | 0.085s   | **134.5x** |

**[CenterCrop](https://explore.albumentations.ai/transform/CenterCrop) (pad when crop > image size):**

| channels | v2.0.17 | v2.0.18  | speedup   |
|----------|---------|----------|-----------|
| 5        | 0.257s  | 0.017s   | **15.3x** |
| 8        | 0.447s  | 0.026s   | **16.9x** |
| 16       | 0.873s  | 0.052s   | **16.7x** |
| 32       | 1.717s  | 0.017s   | **99.8x** |
| 64       | 3.263s  | 0.045s   | **73.1x** |
| 128      | 11.246s | 0.069s   | **162.4x** |

**[CropAndPad](https://explore.albumentations.ai/transform/CropAndPad):**

| channels | v2.0.17 | v2.0.18 | speedup |
|----------|---------|---------|---------|
| 1        | 0.013s  | 0.014s  | 1.0x    |
| 3        | 0.036s  | 0.018s  | 2.0x    |
| 5        | 0.531s  | 0.286s  | **1.9x** |
| 8        | 0.928s  | 0.580s  | **1.6x** |
| 16       | 1.936s  | 1.007s  | **1.9x** |
| 32       | 3.732s  | 1.819s  | **2.1x** |
| 64       | 7.015s  | 3.610s  | **1.9x** |
| 128      | 24.289s | 14.005s | **1.7x** |

> Note: `CropAndPad` includes a resize step that still uses separate cv2 calls; the speedup is lower than pure pad transforms.

---

## Summary of Affected Transforms

| Group | Transforms | Speedup condition | When unchanged |
|---|---|---|---|
| warp_affine | [Affine](https://explore.albumentations.ai/transform/Affine), [Rotate](https://explore.albumentations.ai/transform/Rotate), [ShiftScaleRotate](https://explore.albumentations.ai/transform/ShiftScaleRotate), [SafeRotate](https://explore.albumentations.ai/transform/SafeRotate) | `INTER_LINEAR/NEAREST/AREA` (default) | `INTER_CUBIC/LANCZOS4/LINEAR_EXACT` |
| warp_perspective | [Perspective](https://explore.albumentations.ai/transform/Perspective) | same | same |
| remap | [ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform), [GridDistortion](https://explore.albumentations.ai/transform/GridDistortion), [OpticalDistortion](https://explore.albumentations.ai/transform/OpticalDistortion), [ThinPlateSpline](https://explore.albumentations.ai/transform/ThinPlateSpline), [PiecewiseAffine](https://explore.albumentations.ai/transform/PiecewiseAffine) | `INTER_LINEAR/NEAREST/AREA/LINEAR_EXACT` (default) | `INTER_CUBIC/LANCZOS4` |
| copy_make_border | [Pad](https://explore.albumentations.ai/transform/Pad), [PadIfNeeded](https://explore.albumentations.ai/transform/PadIfNeeded), [CenterCrop](https://explore.albumentations.ai/transform/CenterCrop), [RandomCrop](https://explore.albumentations.ai/transform/RandomCrop), [Crop](https://explore.albumentations.ai/transform/Crop), [CropAndPad](https://explore.albumentations.ai/transform/CropAndPad) | scalar or ≤4-element fill (default) | per-channel fill with >4 values |
| box_blur | [Blur](https://explore.albumentations.ai/transform/Blur) | always | — |
| median_blur | [MedianBlur](https://explore.albumentations.ai/transform/MedianBlur) | ksize 3 or 5 | ksize ≥ 7 for >4 channels |

## Requirements

- OpenCV ≥ 4.13.0.92 (previously ≥ 4.9.0.80)
- albucore == 0.0.39 (previously == 0.0.36)
