# AlbumentationsX Low-Level Operations Analysis & SIMD Modernization

> **TL;DR**: AlbumentationsX is bottlenecked by OpenCV's 4-channel limit and lack of batch/3D support. We can achieve 2-10x speedups and unlock new use cases (hyperspectral, medical, video) by implementing SIMD-optimized operations. Ash Vardanyan (StringZilla, SimSIMD) is interested in collaborating.

## 🎯 **NEW: SimSIMD Integration Results (2026-02-05)**

**✅ Tests completed**: 46/46 correctness tests pass, 20 benchmarks run on ARM (M-series)

### Key Findings

1. **SimSIMD already integrated** via `albucore`:
   - `add_weighted`: **3.66x faster** than cv2.addWeighted
   - FMA operations: **9.6x faster** than sequential cv2
   - Distance metrics: **3.59-3.96x faster** than NumPy
   - **Enables C > 4 channels** (bypasses cv2 limitation!)

2. **NumPy @ beats cv2.gemm**:
   - Small matrices (10×10): **8.4x faster** than cv2.gemm
   - **Action**: Replace cv2.gemm in ThinPlateSpline with NumPy @

3. **Platform-specific insights**:
   - On ARM: NumPy already uses Apple Accelerate (optimized BLAS)
   - SimSIMD provides **no additional benefit** for matrix ops on ARM
   - SimSIMD may be faster on x86/AVX platforms (needs testing)

**Next steps**:
- Replace cv2.gemm with NumPy @ in ThinPlateSpline
- Profile real-world impact on video benchmarks
- Consider x86/AVX benchmarks for matrix operations

---

## Quick Facts

- **Current bottleneck**: `cv2.warpAffine`, `cv2.resize`, `cv2.remap` (80% of pipeline time)
- **Main limitation**: Max 4 channels (❌ blocks hyperspectral, medical, satellite imagery)
- **Secondary limitations**: No batch processing, no 3D volumes
- **Expected gains**: 2-5x for multi-channel, 3-8x for batch, 5-10x for 3D
- **Status**: Ready for prototyping

## Architecture Vision

```
CURRENT:
┌─────────────────────┐
│ AlbumentationsX     │
│   Transforms        │
└──────────┬──────────┘
           │
           ├────────────┐
           ▼            ▼
    ┌──────────┐  ┌─────────┐
    │ OpenCV   │  │ NumPy   │
    │ (cv2)    │  │ Fallback│
    │          │  │         │
    │ ✅ C≤4   │  │ ⚠️ Slow │
    │ ❌ C>4   │  │ ✅ C>4  │
    │ ❌ Batch │  │ ❌ Batch│
    │ ❌ 3D    │  │ ❌ 3D   │
    └──────────┘  └─────────┘

FUTURE (Hybrid Backend):
┌─────────────────────────────────┐
│ AlbumentationsX Transforms      │
│   (Auto backend selection)      │
└────────┬────────────────────────┘
         │
    ┌────┴─────────────────┐
    │   albucore           │
    │   (Dispatch Layer)   │
    └────┬─────────────────┘
         │
    ┌────┴────┬─────────┬──────────┐
    ▼         ▼         ▼          ▼
┌────────┐┌──────┐┌──────────┐┌────────┐
│OpenCV  ││SIMD  ││SimSIMD   ││StringZ │
│(cv2)   ││Ops   ││(Matrix)  ││(LUT)   │
│        ││      ││          ││        │
│✅ C≤4  ││✅ Any││✅ Vec ops││✅ LUT  │
│Fallback││✅ C>4││✅ Fast   ││✅ Fast │
│        ││✅Batch││          ││        │
│        ││✅ 3D ││          ││        │
└────────┘└──────┘└──────────┘└────────┘

Benefits:
✅ Automatic backend selection
✅ Gradual migration (no breaking changes)
✅ Optimal performance for all data types
✅ Extensible architecture
```

---

## Executive Summary

AlbumentationsX currently relies on OpenCV (cv2) and NumPy for image processing operations. While these libraries are mature and well-tested, they have significant limitations when dealing with:

1. **Video data** - tensors with shape `(N, H, W, C)`
2. **Multi-channel images** - images with `C > 4` channels
3. **Volume data** - medical imaging with shape `(N, D, H, W, C)` or `(D, H, W, C)`

Additionally, there's potential for significant performance improvements by leveraging modern SIMD (Single Instruction Multiple Data) and AVX (Advanced Vector Extensions) instructions that may not be fully utilized by current implementations.

**Opportunity**: [Ash Vardanyan](https://github.com/ashvardanian) (author of [StringZilla](https://github.com/ashvardanian/StringZilla) and [SimSIMD](https://github.com/ashvardanian/SimSIMD)) has expressed interest in contributing low-level optimized operations for AlbumentationsX.

### Top Priority Operations (By Impact)

Based on codebase analysis, these operations are the **most critical bottlenecks** and should be prioritized:

#### 🔴 **Tier 1: Critical (80% of performance impact)**
1. **`cv2.warpAffine`** - THE #1 bottleneck
   - Used by virtually every geometric transform
   - Max 4 channels limitation blocks hyperspectral, medical, satellite imagery
   - No batch/video support

2. **`cv2.resize`** (14 uses) - Second most critical
   - Used in nearly every pipeline (preprocessing)
   - Max 4 channels limitation
   - Multiple interpolation modes needed

3. **`cv2.remap`** (7 uses) - Third most critical
   - Used for all distortions (elastic, grid, optical)
   - Very memory intensive (2x float32 arrays)
   - Random memory access (cache unfriendly)

#### 🟡 **Tier 2: High Impact (15% of performance impact)**
4. **`cv2.cvtColor`** (37 uses) - Most frequent operation
   - RGB↔HSV, RGB↔LAB conversions for color augmentations
   - Easy to optimize with SIMD (matrix multiply + min/max)

5. **`cv2.GaussianBlur`** (17 uses) - Common augmentation
   - Separable convolution (easy to parallelize)
   - IIR approximation for large sigma

6. **`cv2.LUT`** - Via albucore wrapper
   - Already optimized via StringZilla in albucore
   - Need to extend to C > 4 channels

7. **Random number generation** - NEW OPPORTUNITY! 🚀
   - `random_generator.uniform()` (29 uses) - most common
   - `random_generator.integers()` (29 uses) - grid distortions
   - `random_generator.normal()` (11 uses) - elastic transforms, noise
   - **Not cv2-dependent** - can be pure SIMD implementation
   - **Expected**: 3-7x speedup for large arrays (elastic transforms!)
   - **Low-hanging fruit** - independent of OpenCV migration

#### 🟢 **Tier 3: Medium Impact (5% of performance impact)**
7. **`cv2.warpPerspective`** (3 uses) - Similar to affine
8. **Arithmetic operations** - `cv2.multiply`, `cv2.add`, `cv2.addWeighted`
9. **`cv2.perspectiveTransform`** (5 uses) - Point transformations
10. **`cv2.filter2D`** (3 uses) - General convolution

**Impact analysis**: Optimizing just the top 3 operations (warpAffine, resize, remap) would improve performance for **>90% of augmentation pipelines**.

### Operations Summary Table

| Category | Operation | Uses | Via | Priority | Channel Limit | Batch Support | 3D Support |
|----------|-----------|------|-----|----------|---------------|---------------|------------|
| **Geometric** | `warpAffine` | Many | Direct | 🔴 Critical | ≤4 | ❌ | ❌ |
| **Geometric** | `resize` | 14 | Direct | 🔴 Critical | ≤4 | ❌ | ❌ |
| **Geometric** | `remap` | 7 | Direct | 🔴 Critical | ≤4 | ❌ | ❌ |
| **Color** | `cvtColor` | 37 | Direct | 🟡 High | 3-4 | ❌ | ❌ |
| **Filtering** | `GaussianBlur` | 17 | Direct | 🟡 High | ≤4 | ❌ | ❌ |
| **Pixel** | `LUT` | Many | albucore | 🟡 High | ≤4 | ❌ | ❌ |
| **Arithmetic** | `multiply` | 11+6 | Both | 🟡 High | ≤4 | ❌ | ❌ |
| **Arithmetic** | `add` | 5+4 | Both | 🟡 High | ≤4 | ❌ | ❌ |
| **Arithmetic** | `addWeighted` | 4+1 | Both | 🟡 High | ≤4 | ❌ | ❌ |
| **Arithmetic** | `pow` | 0+6 | albucore | 🟢 Medium | ≤4 | ❌ | ❌ |
| **Geometric** | `flip` | 0+6 | albucore | 🟢 Medium | ✅ Works | ❌ | ❌ |
| **Matrix** | `gemm` | 5 | Direct | 🟢 Medium | N/A | ❌ | ❌ |
| **Statistical** | `calcHist` | 4 | Direct | 🟢 Medium | ≤4 | ❌ | ❌ |
| **Statistical** | `meanStdDev` | 0+3 | albucore | 🟢 Medium | ≤4 | ❌ | ❌ |
| **Geometric** | `perspectiveTransform` | 5 | Direct | 🟢 Medium | N/A | ❌ | ❌ |
| **Filtering** | `filter2D` | 3 | Direct | 🟢 Medium | ≤4 | ❌ | ❌ |
| **Geometric** | `warpPerspective` | 3 | Direct | 🟢 Medium | ≤4 | ❌ | ❌ |

**Legend**:
- 🔴 Critical (Tier 1) | 🟡 High (Tier 2) | 🟢 Medium (Tier 3)
- **Uses**: Direct uses in albumentations + uses in albucore (e.g., "11+6" = 11 direct + 6 in albucore)
- **Via**: Direct = called directly in albumentations code, albucore = wrapped by albucore, Both = both direct and via albucore

**Key insights**:
- **ALL** image processing operations limited to ≤4 channels (except flip which works for more)
- **ZERO** native batch processing support
- **ZERO** 3D volume support
- Top 3 operations account for ~80% of augmentation pipeline time
- **Arithmetic operations heavily used via albucore** - when C > 4, they fall back to NumPy (slower)

## Current Architecture

### Dependencies
- **OpenCV (cv2)**: Primary image processing operations (direct and via albucore)
- **albucore**: Optimized wrappers and dispatch layer
  - Uses StringZilla for LUT operations (faster than cv2.LUT)
  - Dispatches to cv2 for arithmetic (add, multiply, pow, flip, etc.)
  - Falls back to NumPy when cv2 doesn't support the operation
  - Key bridge between AlbumentationsX and low-level implementations
- **NumPy**: Array operations, mathematical transformations (fallback when cv2 unavailable)
- **SciPy**: Some filtering operations (ndimage)

### Performance Characteristics
- cv2 operations are generally fastest for 2D images with 1-4 channels
- **albucore intelligently dispatches** based on data type and operation:
  - For uint8 scalar operations → `cv2.LUT` via StringZilla (fastest)
  - For operations cv2 supports → `cv2.multiply`, `cv2.add`, etc. (fast)
  - For C > 4 or unsupported ops → NumPy fallback (slower)
- NumPy vectorized operations are faster than Python loops
- Current implementation uses `albucore.sz_lut` (StringZilla-optimized) for uint8 lookup tables

### Albucore's Role

**Albucore is a critical abstraction layer** that:
1. **Hides cv2 complexity** - developers write `albucore.multiply()` instead of dealing with cv2 quirks
2. **Smart backend selection** - automatically chooses fastest backend (LUT vs cv2 vs NumPy)
3. **Graceful degradation** - falls back to NumPy when cv2 fails (e.g., C > 4)
4. **Provides StringZilla integration** - already uses Ash's StringZilla for LUT operations!

**This means**:
- ✅ AlbumentationsX already has dispatch infrastructure (albucore)
- ✅ Adding SIMD operations can plug into existing albucore architecture
- ✅ No need to change albumentations code - just enhance albucore backends
- ✅ StringZilla already integrated - proven collaboration with Ash!

## Real-World Performance Validation: Video Benchmark Results

### Benchmark Overview
**Dataset**: UCF101 (13,320 real-world videos from YouTube)
**Hardware comparison**:
- **Albumentationsx**: Apple Silicon ARM CPU (1 core, M-series)
- **Torchvision**: NVIDIA RTX 4090 GPU (fp16)
- **Kornia**: NVIDIA RTX 4090 GPU (fp16)

### Critical Performance Gaps (CPU vs GPU)

The video benchmarks reveal **MASSIVE performance gaps** where AlbumentationsX is being crushed by GPU implementations:

#### 🔴 **CATASTROPHIC** (>50x slower than GPU)
1. **Affine**: 4.15 v/s vs Torchvision 452.58 v/s = **109x slower!**
2. **Perspective**: 4.18 v/s vs Torchvision 434.75 v/s = **104x slower!**
3. **Elastic**: 4.03 v/s vs Torchvision 126.83 v/s = **31.5x slower**
4. **Normalize**: 19.32 v/s vs Torchvision 460.80 v/s = **24x slower**
5. **Equalize**: 11.87 v/s vs Torchvision 191.55 v/s = **16x slower**
6. **RandomResizedCrop**: 13.98 v/s vs Torchvision 182.09 v/s = **13x slower**

#### 🟡 **SEVERE** (10-50x slower than GPU)
7. **Resize**: 13.85 v/s vs Torchvision 139.96 v/s = **10x slower**
8. **ThinPlateSpline**: 4.51 v/s vs Kornia 44.90 v/s = **10x slower**
9. **GaussianBlur**: 26.57 v/s vs Torchvision 543.44 v/s = **20x slower**
10. **ColorJitter**: 10.51 v/s vs Torchvision 68.75 v/s = **6.5x slower**

### Mapping Benchmarks to SIMD Opportunities

| Benchmark Transform | Speed (v/s) | vs GPU | Maps to SIMD Operation | SIMD Priority | Expected Gain |
|---------------------|-------------|--------|------------------------|---------------|---------------|
| **Affine** | 4.15 | 109x | `cv2.warpAffine` | 🔴 Tier 1 | **3-5x** |
| **Perspective** | 4.18 | 104x | `cv2.warpPerspective` | 🔴 Tier 1 | **3-5x** |
| **Elastic** | 4.03 | 31.5x | `cv2.remap` + RNG | 🔴 Tier 1 + Tier 2 | **5-10x** (remap + RNG!) |
| **Resize** | 13.85 | 10x | `cv2.resize` | 🔴 Tier 1 | **2-4x** |
| **RandomResizedCrop** | 13.98 | 13x | `cv2.resize` | 🔴 Tier 1 | **2-4x** |
| **GaussianNoise** | 9.06 | 2.5x | RNG + add | 🟡 Tier 2 | **3-5x** (RNG!) |
| **GaussianBlur** | 26.57 | 20x | `cv2.GaussianBlur` | 🟡 Tier 2 | **1.5-2x** |
| **ColorJitter** | 10.51 | 6.5x | HSV conversion + arith | 🟡 Tier 2 | **2-3x** |
| **Saturation** | 8.74 | 4.2x | HSV conversion | 🟡 Tier 2 | **2-3x** |
| **Hue** | 14.30 | 1.4x | HSV conversion | 🟡 Tier 2 | **1.5-2x** |
| **RGBShift** | 7.41 | 3x | `cv2.add` per channel | 🟡 Tier 2 | **2-3x** |
| **ThinPlateSpline** | 4.51 | 10x | `cv2.gemm` + matrices | 🟢 Tier 3 | **2-3x** |
| **Normalize** | 19.32 | 24x | `cv2.meanStdDev` | 🟢 Tier 3 | **1.5-2x** |
| **Equalize** | 11.87 | 16x | `cv2.calcHist` + LUT | 🟡 Tier 2 | **2-3x** |
| **MedianBlur** | 13.20 | - | `cv2.medianBlur` | 🟢 Tier 3 | **1.5-2x** |

### Key Insights from Benchmark Analysis

1. **Geometric transforms are DEVASTATINGLY slow** (4-14 videos/sec):
   - Affine, Perspective, Elastic, TPS, Resize all < 15 v/s
   - These are our **Tier 1 SIMD targets** - confirmed!
   - GPU is 10-100x faster primarily due to parallelism

2. **Elastic transform bottleneck confirmed** (4.03 v/s):
   - Slowest geometric transform
   - Uses both `cv2.remap` (Tier 1) AND RNG (Tier 2)
   - **Double optimization opportunity**: SIMD remap + SIMD RNG
   - Expected combined gain: **5-10x speedup**

3. **RNG-heavy operations are slow**:
   - GaussianNoise: 9.06 v/s (uses RNG heavily)
   - SaltAndPepper: 11.24 v/s (uses RNG)
   - **Validates RNG as Tier 2 priority**

4. **Color operations underperform** (8-14 v/s):
   - ColorJitter, Saturation, Hue all slow
   - HSV conversions are bottleneck
   - **Validates cvtColor as Tier 2 priority**

5. **CPU wins on simple ops** (where data transfer dominates):
   - Blur: 46.78 v/s (beats Kornia 2.3x)
   - PlankianJitter: 50.65 v/s (beats Kornia 4.7x)
   - Rain: 24.05 v/s (beats Kornia 6.4x)
   - RandomGamma: 198.35 v/s (beats Kornia 9.2x!)
   - **These don't need SIMD** - already efficient

### Impact Projection: Video Benchmark + SIMD

**If we implement SIMD optimizations (conservative estimates)**:

| Transform | Current (v/s) | After SIMD (v/s) | Speedup | Still vs GPU |
|-----------|---------------|------------------|---------|--------------|
| Affine | 4.15 | **12-20** | 3-5x | 23-38x slower (better!) |
| Perspective | 4.18 | **12-20** | 3-5x | 22-36x slower (better!) |
| Elastic | 4.03 | **20-40** | 5-10x | 3-6x slower (competitive!) |
| Resize | 13.85 | **28-55** | 2-4x | 2.5-5x slower (competitive!) |
| GaussianNoise | 9.06 | **27-45** | 3-5x | Near parity with Kornia! |
| ColorJitter | 10.51 | **21-31** | 2-3x | 2-3x slower (better!) |

**Key takeaway**: SIMD won't beat GPU's massive parallelism, but can close the gap from **100x slower to 3-5x slower** - making CPU viable for many use cases!

### Recommended SIMD Priority Based on Video Benchmarks

**Phase 1: Fix the catastrophic bottlenecks** (Target: 3 months)
1. **Affine transforms** (4.15 v/s → 12-20 v/s)
   - Implement SIMD `warpAffine` with arbitrary channels
   - Add batch processing support
   - Expected: Used in every geometric pipeline

2. **SIMD RNG** (Elastic: 4.03 v/s → 20-40 v/s)
   - Implement Philox/PCG with SIMD
   - Optimize `uniform()`, `normal()`, `integers()`
   - **Biggest win**: Elastic transform (RNG + remap)
   - Also helps: GaussianNoise, SaltAndPepper

3. **Remap** (Elastic: 4.03 v/s → 20-40 v/s)
   - SIMD-optimized bilinear interpolation
   - Quantized displacement fields
   - Batch support

**Phase 2: High-impact operations** (Target: +2 months)
4. **Resize** (13.85 v/s → 28-55 v/s)
   - Separable filtering with SIMD
   - Batch resize
   - Multiple interpolation modes

5. **Color space conversions** (ColorJitter: 10.51 v/s → 21-31 v/s)
   - RGB↔HSV SIMD optimization
   - Fused HSV adjust operations
   - Batch conversions

**Phase 3: Polish** (Target: +2 months)
6. Other operations as needed

### Video Benchmark Validation Strategy

After each SIMD implementation:
1. Re-run video benchmarks on same UCF101 dataset
2. Measure actual speedup vs predictions
3. Compare gap reduction vs GPU (goal: <5x slower)
4. Iterate based on results

## Real-World Use Cases Currently Blocked

### 1. Hyperspectral Satellite Imagery
**Problem**: Satellite images often have 8-16+ channels (multispectral, hyperspectral)

```python
# Sentinel-2 satellite image: 13 channels
satellite_image = np.random.randint(0, 256, (1024, 1024, 13), dtype=np.uint8)

# ❌ FAILS: cv2.warpAffine doesn't support C > 4
transform = A.Compose([
    A.Rotate(limit=45, p=1.0),  # Uses warpAffine internally
])
# Error: cv2.error: Function not implemented for C > 4
```

**Workaround**: Split into 4-channel chunks, process separately, merge back (slow, error-prone)

**Impact**: Blocked users in agriculture, remote sensing, environmental monitoring

### 2. Medical Imaging - Multi-Modal MRI/CT
**Problem**: Medical scans often have multiple modalities as channels

```python
# Multi-modal MRI: T1, T2, FLAIR, DWI = 4+ channels
mri_scan = np.random.randn(256, 256, 6).astype(np.float32)

# ❌ FAILS for C > 4
transform = A.Compose([
    A.ElasticTransform(p=1.0),  # Uses remap internally
    A.GaussianBlur(p=1.0),      # Fails for C > 4
])
```

**Impact**: Blocked users in medical AI, radiology, pathology

### 3. Video/Temporal Data Augmentation
**Problem**: Video requires batch processing for temporal consistency

```python
# Video: 30 frames, 480x640, RGB
video = np.random.randint(0, 256, (30, 480, 640, 3), dtype=np.uint8)

# ❌ SLOW: Must process frame-by-frame (no batch support)
augmented_frames = []
for frame in video:
    augmented = transform(image=frame)['image']
    augmented_frames.append(augmented)
# 30x slower than batch processing would be
```

**Workaround**: Loop over frames (loses temporal consistency, slow)

**Impact**: Action recognition, video classification, temporal modeling

### 4. 3D Medical Volumes (CT, MRI)
**Problem**: 3D scans need 3D augmentations

```python
# CT scan: 128 slices, 256x256
ct_volume = np.random.randn(128, 256, 256, 1).astype(np.float32)

# ❌ NOT SUPPORTED: No 3D affine, 3D blur, 3D anything
# Current workaround: Augment each 2D slice independently (incorrect!)
```

**Impact**: 3D medical imaging, volumetric analysis

### 5. High-Channel Microscopy
**Problem**: Fluorescence microscopy with many markers

```python
# Fluorescence microscopy: 8 fluorescent markers
microscopy = np.random.randint(0, 65535, (1024, 1024, 8), dtype=np.uint16)

# ❌ FAILS: uint16 not well supported, C > 4
```

**Impact**: Cell biology, drug discovery, pathology research

## Known Limitations

### 1. OpenCV Channel Limitations

OpenCV operations typically support only 1-4 channels:

#### Color Conversions
```python
# cv2.cvtColor - strictly 1, 3, or 4 channels
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Works for C=3
cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # C=3 -> C=1
# FAILS for C > 4
```

#### Geometric Transformations
```python
# cv2.warpAffine, cv2.warpPerspective - max 4 channels
cv2.warpAffine(image, matrix, (width, height))  # C <= 4
cv2.resize(image, (new_width, new_height))      # C <= 4
cv2.remap(image, map_x, map_y)                  # C <= 4
# FAILS for C > 4
```

#### Filtering Operations
```python
# cv2.GaussianBlur, cv2.medianBlur - max 4 channels
cv2.GaussianBlur(image, (5, 5), 0)              # C <= 4
cv2.blur(image, (5, 5))                         # C <= 4
cv2.filter2D(image, -1, kernel)                 # C <= 4
# FAILS for C > 4
```

#### Morphological Operations
```python
# cv2.erode, cv2.dilate, cv2.morphologyEx - max 4 channels
cv2.erode(image, kernel)                        # C <= 4
cv2.dilate(image, kernel)                       # C <= 4
# FAILS for C > 4
```

### 2. Video/Batch Processing

OpenCV doesn't natively support batch dimensions:

```python
# Current: Must process frame-by-frame
video = np.random.randint(0, 256, (30, 480, 640, 3), dtype=np.uint8)  # (N, H, W, C)
for frame in video:
    processed = cv2.GaussianBlur(frame, (5, 5), 0)  # Inefficient loop

# Desired: Batch processing
# processed = batch_gaussian_blur(video, (5, 5), 0)  # Not available in cv2
```

### 3. Volume/3D Medical Imaging

Medical imaging requires 3D operations on volumes:

```python
# Volume data: (D, H, W, C) or (N, D, H, W, C)
ct_scan = np.random.randn(128, 256, 256, 1).astype(np.float32)  # Depth, Height, Width, Channels

# cv2 cannot handle 3D spatial operations
# Need: 3D affine transforms, 3D filtering, 3D interpolation
```

### 4. SIMD Utilization Concerns

Current implementations may not fully leverage modern CPU instructions:

- **AVX-512**: 512-bit vector operations (64 bytes at once)
- **AVX2**: 256-bit vector operations (32 bytes at once)
- **SSE4.2**: 128-bit vector operations (16 bytes at once)
- **ARM NEON**: ARM's SIMD extension

**Questions**:
- Are cv2 operations compiled with full SIMD support for user's architecture?
- Can we do better with custom implementations for specific operations?
- What about newer instructions like AVX-512 VNNI (Vector Neural Network Instructions)?

## Actual cv2 Operations Used in AlbumentationsX

Based on comprehensive codebase analysis (using `grep -r "cv2\." albumentations --include="*.py"` **AND albucore package analysis**), here are **ALL** cv2 operations currently used:

### 📊 Usage Statistics

**Direct cv2 usage in albumentations**: 95 unique symbols
**Indirect cv2 usage via albucore**: 15 additional operations
**Total cv2 operations**: **110 unique cv2 operations**

**Critical insight**: Many "NumPy" operations in AlbumentationsX are actually **cv2-based via albucore**:
- `albucore.add()` → uses `cv2.add()` internally
- `albucore.multiply()` → uses `cv2.multiply()` internally
- `albucore.power()` → uses `cv2.pow()` internally
- `albucore.hflip()` / `albucore.vflip()` → uses `cv2.flip()` internally
- `albucore.add_weighted()` → uses `cv2.addWeighted()` internally
- `albucore.sz_lut()` → uses StringZilla-optimized LUT (even better than cv2.LUT!)

### Albucore cv2 Operations (Abstraction Layer)

Albucore provides optimized wrappers that dispatch to cv2 for performance:

**From albucore package analysis** (`/opt/homebrew/.../albucore/functions.py`):
- `cv2.multiply` (6 uses) - used by `multiply_opencv()`, `multiply_by_constant()`, etc.
- `cv2.add` (4 uses) - used by `add_opencv()`, `add_constant()`, etc.
- `cv2.LUT` (8 uses) - used by `add_lut()`, `multiply_lut()`, wrapped by `sz_lut()`
- `cv2.pow` (6 uses) - used by `power_opencv()`
- `cv2.flip` (6 uses) - used by `hflip()`, `vflip()`
- `cv2.divide` (4 uses) - used by division operations
- `cv2.subtract` (3 uses) - used by subtraction operations
- `cv2.meanStdDev` (3 uses) - used for statistics
- `cv2.normalize` (1 use) - used by `normalize_per_image()`
- `cv2.addWeighted` (1 use) - used by `add_weighted()`

**Functions in albumentations that use albucore cv2 wrappers**:
```python
# From albumentations/augmentations/pixel/functional.py:
from albucore import (
    add,                    # → cv2.add or cv2.LUT
    add_array,             # → cv2.add
    add_constant,          # → cv2.add or cv2.LUT
    add_vector,            # → cv2.add
    add_weighted,          # → cv2.addWeighted
    multiply,              # → cv2.multiply or cv2.LUT
    multiply_add,          # → cv2.multiply + cv2.add
    multiply_by_array,     # → cv2.multiply
    multiply_by_constant,  # → cv2.multiply or cv2.LUT
    power,                 # → cv2.pow
    normalize_per_image,   # → cv2.meanStdDev + division
    sz_lut,                # → StringZilla LUT (optimized alternative to cv2.LUT)
)

# From albumentations/augmentations/geometric/flip.py:
from albucore import hflip, vflip  # → cv2.flip
```

**Key insight**: AlbumentationsX is **MORE dependent on cv2** than it appears! Many "simple" arithmetic operations use cv2 under the hood via albucore for saturation arithmetic and performance.

### Direct cv2 Operations in AlbumentationsX Code

**Total in albumentations code**: 65 functions + 30 constants = **95 unique cv2 symbols**

**By category**:
- **Constants/Flags**: 30 (interpolation, border modes, color codes)
- **Image processing**: 15 (cvtColor, resize, blur, etc.)
- **Geometric transforms**: 10 (warpAffine, remap, perspectiveTransform, etc.)
- **Arithmetic/Channel**: 8 (multiply, add, split, merge, etc.)
- **Matrix operations**: 4 (gemm, solve, eigen, etc.)
- **Statistical/Histogram**: 5 (calcHist, normalize, etc.)
- **Morphological**: 4 (erode, dilate, morphologyEx, etc.)
- **Drawing functions**: 7 (circle, rectangle, etc.) - mostly testing
- **I/O operations**: 5 (imread, imencode, etc.) - mostly testing
- **Other**: 7 (distanceTransform, connectedComponents, etc.)

### Constants/Flags (Most Referenced)
- **Interpolation flags**: `INTER_LINEAR` (182), `INTER_NEAREST` (181), `INTER_AREA` (123), `INTER_CUBIC` (119), `INTER_LANCZOS4` (115), `INTER_NEAREST_EXACT` (47), `INTER_LINEAR_EXACT` (47)
- **Border modes**: `BORDER_CONSTANT` (101), `BORDER_REFLECT_101` (44), `BORDER_REPLICATE` (43), `BORDER_REFLECT` (40), `BORDER_WRAP` (35), `BORDER_DEFAULT`, `BORDER_REFLECT101`
- **Color conversion codes**: `COLOR_RGB2GRAY`, `COLOR_RGB2HSV`, `COLOR_HSV2RGB`, `COLOR_RGB2LAB`, `COLOR_LAB2RGB`, `COLOR_RGB2HLS`, `COLOR_HLS2RGB`, `COLOR_RGB2YCrCb`, `COLOR_YCrCb2RGB`, `COLOR_BGR2RGB`, `COLOR_BGR2GRAY`, `COLOR_GRAY2RGB`, `COLOR_GRAY2BGR`
- **Data types**: `CV_32F`, `CV_32FC1`, `CV_8U`
- **Other constants**: `INPAINT_TELEA`, `INPAINT_NS`, `NORM_MINMAX`, `DECOMP_LU`, `DIST_L2`, `DIST_MASK_PRECISE`, `MORPH_ELLIPSE`, `THRESH_TRUNC`, `FONT_HERSHEY_SIMPLEX`, `LINE_4`, `COVAR_NORMAL`, `COVAR_ROWS`, `COVAR_SCALE`

### Functions (By Category)

#### Core Image Processing Functions
1. **`cv2.cvtColor`** (37 uses) - Color space conversions
2. **`cv2.resize`** (14 uses) - Image resizing with interpolation
3. **`cv2.LUT`** - Lookup table operations (uint8 only)
4. **`cv2.normalize`** (6 uses) - Image normalization
5. **`cv2.GaussianBlur`** (17 uses) - Gaussian blur filter
6. **`cv2.medianBlur`** - Median filter
7. **`cv2.blur`** (3 uses) - Box blur / mean filter
8. **`cv2.filter2D`** (3 uses) - 2D convolution
9. **`cv2.sepFilter2D`** - Separable 2D convolution

#### Geometric Transformations
10. **`cv2.warpAffine`** - Affine transformation
11. **`cv2.warpPerspective`** (3 uses) - Perspective transformation
12. **`cv2.remap`** (7 uses) - Generic image remapping with interpolation
13. **`cv2.getPerspectiveTransform`** (4 uses) - Calculate perspective transform matrix
14. **`cv2.getRotationMatrix2D`** - Calculate 2D rotation matrix
15. **`cv2.perspectiveTransform`** (5 uses) - Transform points using perspective matrix
16. **`cv2.transform`** (4 uses) - Transform points using affine matrix
17. **`cv2.initUndistortRectifyMap`** - Create undistortion/rectification maps

#### Morphological Operations
18. **`cv2.erode`** - Morphological erosion
19. **`cv2.dilate`** - Morphological dilation
20. **`cv2.morphologyEx`** - Advanced morphological operations
21. **`cv2.getStructuringElement`** - Create structuring element for morphology

#### Arithmetic and Channel Operations
22. **`cv2.add`** (5 uses) - Add with saturation
23. **`cv2.subtract`** - Subtract with saturation
24. **`cv2.multiply`** (11 uses) - Multiply with saturation
25. **`cv2.addWeighted`** (4 uses) - Weighted sum of two arrays
26. **`cv2.split`** (4 uses) - Split multi-channel array
27. **`cv2.merge`** (5 uses) - Merge single-channel arrays
28. **`cv2.mixChannels`** (2 uses) - Copy specific channels

#### Matrix Operations
29. **`cv2.gemm`** (5 uses) - General matrix multiplication
30. **`cv2.solve`** - Solve linear system
31. **`cv2.eigen`** - Eigenvalue decomposition
32. **`cv2.calcCovarMatrix`** - Calculate covariance matrix

#### Statistical and Histogram Operations
33. **`cv2.calcHist`** (4 uses) - Calculate histogram
34. **`cv2.equalizeHist`** - Histogram equalization
35. **`cv2.createCLAHE`** - Create CLAHE object for adaptive histogram equalization
36. **`cv2.meanStdDev`** - Calculate mean and standard deviation

#### Advanced Operations
37. **`cv2.distanceTransform`** - Distance transform
38. **`cv2.connectedComponents`** (3 uses) - Connected components labeling
39. **`cv2.inpaint`** (4 uses) - Image inpainting
40. **`cv2.Canny`** - Canny edge detection
41. **`cv2.threshold`** - Image thresholding

#### Border and Padding
42. **`cv2.copyMakeBorder`** - Add border to image
43. **`cv2.copyTo`** - Copy with mask

#### Drawing Functions (Mainly for Testing/Visualization)
44. **`cv2.circle`** (21 uses) - Draw circle
45. **`cv2.rectangle`** (13 uses) - Draw rectangle
46. **`cv2.line`** (4 uses) - Draw line
47. **`cv2.polylines`** (2 uses) - Draw polylines
48. **`cv2.fillPoly`** - Fill polygon
49. **`cv2.fillConvexPoly`** - Fill convex polygon
50. **`cv2.putText`** (7 uses) - Put text on image
51. **`cv2.boundingRect`** - Calculate bounding rectangle

#### I/O Operations (Mainly for Testing)
52. **`cv2.imread`** - Read image from file
53. **`cv2.imshow`** - Display image (debugging)
54. **`cv2.imdecode`** - Decode image from buffer
55. **`cv2.imencode`** - Encode image to buffer

#### PCA Operations
56. **`cv2.PCACompute2`** - PCA computation
57. **`cv2.PCAProject`** - Project data onto PCA space
58. **`cv2.PCABackProject`** - Back-project from PCA space

#### Utility Functions
59. **`cv2.minAreaRect`** (5 uses) - Minimum area bounding rectangle
60. **`cv2.log`** - Logarithm
61. **`cv2.exp`** - Exponential
62. **`cv2.pow`** - Power
63. **`cv2.randn`** - Random normal distribution
64. **`cv2.setRNGSeed`** - Set random number generator seed

#### Image Write Parameters
65. **`IMREAD_GRAYSCALE`**, **`IMREAD_UNCHANGED`** - imread flags
66. **`IMWRITE_JPEG_QUALITY`**, **`IMWRITE_WEBP_QUALITY`** - imwrite parameters

## Comprehensive Operations Inventory

### A. Pixel-Level Operations (Element-wise)

#### 1. Lookup Tables (LUT)
**Current**: `cv2.LUT` - Excellent performance for uint8

**Actual usage in AlbumentationsX**:
- Used via `albucore.sz_lut` (StringZilla-optimized wrapper) in:
  - `albumentations/augmentations/pixel/functional.py` - imported but delegated to albucore
  - Color mapping operations
  - Posterize, Solarize, and custom LUT transforms

```python
# Current cv2.LUT constraints
cv2.LUT(image, table)  # image: uint8, C <= 4, table: 256 elements
```

**Limitations**:
- uint8 only (no float32, uint16)
- Max 4 channels
- No batch processing
- No native support for per-channel LUTs

**Use cases in AlbumentationsX**:
- `Posterize` - quantize color levels
- `Solarize` - invert colors above threshold
- `Equalize` - histogram equalization
- `ToSepia` - sepia tone mapping
- Custom color grading transforms
- Channel shuffling operations

**Potential improvements**:
- SIMD-optimized LUT for uint8 with C > 4 (arbitrary channels)
- LUT support for uint16 (medical imaging, RAW photos)
- Batch LUT application for video `(N, H, W, C)`
- Float32 LUT approximations (interpolated lookup)
- Multi-dimensional LUTs (3D LUTs for color grading)
- Per-channel independent LUTs

#### 2. Arithmetic Operations
**Current**: **Primarily via albucore wrappers** (which use cv2 internally) + direct cv2 calls

**Actual usage in AlbumentationsX**:

**Via albucore** (most common - used throughout `pixel/functional.py`):
- `albucore.add()` → dispatches to `cv2.add()` or `cv2.LUT` (uint8) or NumPy
- `albucore.add_constant()` → `cv2.add()` with scalar
- `albucore.add_array()` → `cv2.add()` with array
- `albucore.add_vector()` → `cv2.add()` per-channel
- `albucore.add_weighted()` → `cv2.addWeighted()` for alpha blending
- `albucore.multiply()` → `cv2.multiply()` or `cv2.LUT` or NumPy
- `albucore.multiply_by_constant()` → `cv2.multiply()` with scalar
- `albucore.multiply_by_array()` → `cv2.multiply()` with array
- `albucore.multiply_add()` → fused `cv2.multiply() + cv2.add()`
- `albucore.power()` → `cv2.pow()`

**Direct cv2 calls** (less common):
- `cv2.add` (5 direct uses in albumentations code)
- `cv2.multiply` (11 direct uses) - used in TPS, PCA, distance calculations
- `cv2.addWeighted` (4 direct uses) - alpha blending
- `cv2.subtract` (minimal direct use)

**Implementation details** (from albucore source):
```python
# albucore intelligently selects backend:
def multiply(img, value, inplace=False):
    if uint8 and scalar and can_use_lut:
        return multiply_lut(img, value)      # Fastest: cv2.LUT
    elif opencv_compatible:
        return cv2.multiply(img, value)       # Fast: cv2 with saturation
    else:
        return numpy_multiply(img, value)     # Fallback: NumPy
```

**Files using albucore arithmetic**:
- `albumentations/augmentations/pixel/functional.py` - ALL color/brightness/contrast operations
- `albumentations/augmentations/geometric/functional.py` - Some TPS operations
- Throughout the codebase for any arithmetic

**Limitations**:
- cv2 arithmetic: **max 4 channels** for `cv2.multiply`, `cv2.add`, etc.
- When C > 4, albucore falls back to NumPy (slower)
- Saturation arithmetic (uint8) not always optimal in NumPy fallback
- No fused multiply-add in standard NumPy
- No batch operations

**Use cases** (VERY FREQUENT):
- `RandomBrightnessContrast` - uses `albucore.multiply_add()`
- `HueSaturationValue` - uses `albucore.add()` on HSV channels
- `RGBShift` - uses `albucore.add_vector()` per-channel shifts
- `MultiplicativeNoise` - uses `albucore.multiply()`
- `Normalize` - uses `albucore.normalize_per_image()` → `cv2.meanStdDev`
- `PixelDistributionAdaptation` - uses `albucore.add_weighted()` → `cv2.addWeighted`
- Thin Plate Spline (TPS) - direct `cv2.multiply`, `cv2.gemm`
- Nearly every pixel-level augmentation

**Potential improvements** (HIGH PRIORITY):
- **SIMD-optimized fused multiply-add (FMA)** for arbitrary channels
- **Saturation arithmetic for C > 4** (currently NumPy fallback loses saturation)
- **Batch arithmetic operations** `(N, H, W, C)`
- **Better cache utilization** for sequential operations
- **In-place operations** to reduce memory allocations
- **Fused operations** (e.g., `multiply_add_clip` in one pass)
- **SimSIMD integration** for vector operations (Ash's library can help here!)

#### 3. Clipping and Saturation
**Current**: `np.clip`, cv2 implicit saturation

```python
np.clip(image, 0, 255)          # NumPy
image.astype(np.uint8)          # Implicit clipping
```

**Limitations**:
- np.clip creates temporary arrays
- Not always optimally vectorized

**Potential improvements**:
- In-place SIMD clipping
- Fused operation + clip (e.g., multiply + clip in one pass)

#### 4. Comparison and Selection
**Current**: NumPy boolean indexing, `np.where`

```python
mask = image > threshold
result = np.where(condition, value1, value2)
```

**Use cases**:
- Thresholding operations
- Conditional color adjustments
- Cutout/Coarse dropout masks

**Potential improvements**:
- SIMD-optimized masked operations
- Fused compare-and-select

### B. Geometric Transformations

#### 1. Affine Transformations
**Current**: `cv2.warpAffine` (**THE MOST CRITICAL BOTTLENECK**)

**Actual usage in AlbumentationsX**:
- `albumentations/augmentations/geometric/functional.py`:
  - `warp_affine()` function - main wrapper with channel handling
  - `warp_affine_with_value_extension()` - for constant border with proper value handling
  - Called by virtually every geometric transform

```python
cv2.warpAffine(image, matrix, (width, height),
               flags=cv2.INTER_LINEAR,
               borderMode=cv2.BORDER_REFLECT_101)
```

**Limitations**:
- **Max 4 channels** - ❌ FAILS for hyperspectral, multi-band satellite imagery
- No batch processing - must loop over frames for video
- No 3D volumes - cannot handle medical imaging directly
- Limited to 2D affine - no 3D affine transforms

**Use cases** (nearly ALL geometric transforms):
- `Rotate` - rotation around center
- `Affine` - general affine transformation
- `ShiftScaleRotate` - combined shift/scale/rotate
- `ElasticTransform` - elastic deformation via displacement fields
- `GridDistortion` - grid-based distortions
- `OpticalDistortion` - optical distortions
- `PiecewiseAffine` - piecewise affine warping
- All perspective transforms (internally decomposed)
- Most cropping operations with rotation

**Performance characteristics**:
- Currently THE bottleneck for augmentation pipelines
- Called once per image per geometric augmentation
- Interpolation is compute-intensive (bilinear, bicubic)
- Border handling adds overhead

**Potential improvements** (HIGHEST PRIORITY):
- **Arbitrary channel support** - critical for hyperspectral, medical, satellite
- **Batch affine transforms** - process `(N, H, W, C)` together
- **3D affine for volumes** - `(D, H, W, C)` medical imaging
- **SIMD-optimized interpolation kernels**:
  - Separable filtering where possible
  - Vectorized bilinear/bicubic interpolation
  - Cache-friendly memory access patterns
- **Fused operations** - combine multiple affine transforms into one
- **Multi-threading** - parallelize across image regions
- **GPU backend option** - for large batches

#### 2. Perspective Transformations
**Current**: `cv2.warpPerspective`

```python
cv2.warpPerspective(image, matrix, (width, height),
                     flags=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)
```

**Actual usage in AlbumentationsX**:
- `albumentations/augmentations/geometric/functional.py`:
  - `warp_perspective()` - main function for perspective warps
  - `perspective_bbox_()` - transform bboxes
  - `distort_image()` - piecewise perspective distortion (mesh-based warping)

**Limitations**: Same as affine
- Max 4 channels
- No batch processing
- No 3D volumes
- Computationally more expensive than affine

**Use cases**:
- `Perspective` - 4-point perspective transformation
- `PiecewiseAffine` - mesh-based warping with perspective per cell
- Custom distortions requiring non-affine transformations

**Potential improvements**:
- Same as affine transformations (arbitrary channels, batch, 3D)
- **Note**: Perspective transform can sometimes be decomposed into affine + homography for optimization

#### 3. Resize/Scaling
**Current**: `cv2.resize` (14 uses - SECOND MOST CRITICAL)

**Actual usage in AlbumentationsX**:
- `albumentations/augmentations/geometric/functional.py`:
  - `resize_cv2()` - main resize function
  - `scale_keypoint_mask_remap()` - resize displacement maps for keypoint tracking
- `albumentations/augmentations/geometric/resize.py`:
  - `Resize` transform - basic resize
  - `LongestMaxSize` - resize maintaining aspect ratio
  - `SmallestMaxSize` - resize maintaining aspect ratio
  - `RandomScale` - random scaling augmentation

```python
cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
```

**Interpolation methods supported**:
- `INTER_NEAREST`: Nearest neighbor (182 uses in code)
- `INTER_LINEAR`: Bilinear (181 uses) - **DEFAULT**
- `INTER_CUBIC`: Bicubic (119 uses) - higher quality
- `INTER_AREA`: Area-based (123 uses) - **BEST for downscaling**
- `INTER_LANCZOS4`: Lanczos with 4 lobes (115 uses) - highest quality
- `INTER_NEAREST_EXACT`: Exact nearest (47 uses)
- `INTER_LINEAR_EXACT`: Exact bilinear (47 uses)

**Limitations**:
- **Max 4 channels** - ❌ FAILS for C > 4
- No batch processing - must loop for video
- No 3D volumes - no direct volume resizing
- Each interpolation method has different channel support

**Use cases** (CRITICAL - used in nearly every pipeline):
- `Resize` - resize to fixed size
- `LongestMaxSize` - resize with aspect ratio preservation
- `SmallestMaxSize` - resize with aspect ratio preservation
- `RandomScale` - random scaling augmentation
- `RandomSizedCrop` - crop + resize
- `CenterCrop` + resize combinations
- Preprocessing for neural networks (resize to model input size)

**Performance characteristics**:
- **Extremely frequent operation** - often the first operation in a pipeline
- Quality/speed trade-off based on interpolation method:
  - `INTER_AREA`: Best quality for downscaling (box filter)
  - `INTER_LANCZOS4`: Best quality overall but slowest
  - `INTER_LINEAR`: Good balance (default)
  - `INTER_NEAREST`: Fastest but lowest quality

**Potential improvements** (HIGH PRIORITY):
- **Arbitrary channel support** via separable convolution
- **Batch resize** with dynamic sizes per image
- **3D resize for volumes** (trilinear, tricubic interpolation)
- **SIMD-optimized separable filtering**:
  - Horizontal pass + vertical pass
  - Cache-aware tiling for large images
  - Vectorized interpolation kernels
- **Polyphase filtering** for high-quality downscaling
- **Multi-resolution pyramid** caching for repeated resizes
- **Auto-select INTER_AREA** for downscaling (already done in code)

#### 4. Remapping (Grid Sampling)
**Current**: `cv2.remap` (7 uses - CRITICAL for distortions)

**Actual usage in AlbumentationsX**:
- `albumentations/augmentations/geometric/functional.py`:
  - `remap()` - main wrapper with backend selection (cv2/pillow)
  - `keypoints_remap()` - remap keypoints using mask-based tracking
  - `bbox_remap()` - remap bboxes by transforming corner masks
  - `elastic_transform_v2()` - uses remap for elastic deformations
- `albumentations/augmentations/geometric/distortion.py`:
  - Grid distortions, optical distortions, elastic transforms

```python
cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
          borderMode=cv2.BORDER_REFLECT_101)
```

**Limitations**:
- **Max 4 channels** - ❌ FAILS for C > 4
- No batch processing
- Float32 mapping only (32-bit per coordinate = memory intensive)
- No quantized grid support

**Use cases** (VERY IMPORTANT for advanced augmentations):
- `GridDistortion` - grid-based warping
- `OpticalDistortion` - lens distortion effects
- `ElasticTransform` - elastic deformations (commonly used!)
- Custom distortions via displacement fields
- Undistortion/rectification (camera calibration)

**Performance characteristics**:
- Memory intensive: requires 2x float32 arrays (map_x, map_y) of image size
- Random memory access pattern (cache unfriendly)
- Interpolation quality depends on input maps and interpolation method
- Used frequently in augmentation pipelines

**Potential improvements** (HIGH PRIORITY):
- **Arbitrary channels** - critical for hyperspectral
- **Batch remapping** - process video frames together
- **SIMD-optimized bilinear interpolation**:
  - Vectorize interpolation kernel
  - Prefetching to hide memory latency
- **Quantized grid support**:
  - uint16 integer coordinates + 8-bit fractional offsets
  - Reduces memory from 8 bytes/pixel to 5 bytes/pixel (37% savings)
- **3D remapping for volumes** - medical imaging
- **Sparse remapping** - only compute changed regions
- **Multi-scale remapping** - hierarchical displacement fields

#### 5. Flip Operations
**Current**: **Via albucore wrappers** `hflip()`, `vflip()` → `cv2.flip()`

**Actual usage in AlbumentationsX**:
- `albumentations/augmentations/geometric/flip.py`:
  ```python
  from albucore import hflip, vflip
  # These use cv2.flip internally (6 uses in albucore)
  ```
- Albucore implementation uses `cv2.flip()` for performance

```python
# albucore wrappers:
hflip(image)  # → cv2.flip(image, 1)  # Horizontal
vflip(image)  # → cv2.flip(image, 0)  # Vertical
# Can also do both: cv2.flip(image, -1)
```

**Limitations**:
- cv2.flip: generally works for arbitrary channels (not strictly limited to 4)
- Not always cache-optimal for large images
- No batch flipping

**Use cases**:
- `HorizontalFlip` - uses `albucore.hflip()`
- `VerticalFlip` - uses `albucore.vflip()`
- Very common augmentation (nearly every pipeline)

**Potential improvements**:
- **Cache-aware flipping** for large images
- **Batch flipping** `(N, H, W, C)`
- **3D volume flipping** for medical imaging
- **In-place flipping** where safe (already may be in albucore)

#### 6. Transpose/Rotation by 90°
**Current**: `np.transpose`, `cv2.rotate`

```python
np.transpose(image, (1, 0, 2))  # Transpose H, W
cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
```

**Use cases**:
- `Transpose`
- `RandomRotate90`

**Potential improvements**:
- Cache-oblivious transpose algorithms
- Batch transpose
- 3D transpose

### C. Filtering Operations

#### 1. Convolution (General)
**Current**: `cv2.filter2D`, `scipy.ndimage.convolve`

```python
cv2.filter2D(image, -1, kernel)  # Max 4 channels
scipy.ndimage.convolve(image, kernel)  # Arbitrary channels, slower
```

**Use cases**:
- Custom kernels
- Emboss
- Sharpen
- Edge detection

**Potential improvements**:
- SIMD-optimized 3x3, 5x5 kernels
- Separable kernel optimization
- Arbitrary channels
- Batch convolution
- 3D convolution for volumes

#### 2. Gaussian Blur
**Current**: `cv2.GaussianBlur`

#### 2. Gaussian Blur
**Current**: `cv2.GaussianBlur` (17 uses - FREQUENT)

**Actual usage in AlbumentationsX**:
- `albumentations/augmentations/blur/functional.py`:
  - `gaussian_blur()` - main Gaussian blur function
- `albumentations/augmentations/geometric/functional.py`:
  - `gaussian_pyramid()` - image pyramid generation
- `albumentations/augmentations/blur/transforms.py`:
  - `GaussianBlur` transform - Gaussian blur augmentation
  - `Defocus` - defocus effect using Gaussian
  - `MotionBlur` - uses Gaussian as component

```python
cv2.GaussianBlur(
    image,
    (ksize, ksize),  # Kernel size (odd numbers)
    sigmaX=sigma,    # Standard deviation in X
    sigmaY=0,        # 0 means same as sigmaX
    borderType=cv2.BORDER_REPLICATE,
)
```

**Special cases**:
- `ksize=(0,0)` → cv2 auto-calculates kernel size from sigma
- Internally uses separable convolution (horizontal + vertical pass)

**Limitations**:
- **Max 4 channels** - ❌ FAILS for C > 4
- No batch processing
- No 3D Gaussian for volumes

**Use cases**:
- `GaussianBlur` - blur augmentation (very common!)
- `Defocus` - out-of-focus effect
- `MotionBlur` - motion blur (component)
- Noise reduction preprocessing
- Multi-scale image pyramids

**Potential improvements** (HIGH PRIORITY):
- **Arbitrary channels** via separable convolution
- **Batch Gaussian blur** for video
- **3D Gaussian for volumes** (medical imaging)
- **SIMD-optimized separable kernels**:
  - Vectorize 1D convolution
  - Cache-aware horizontal/vertical passes
- **IIR approximation** for large sigma (constant time)
- **Recursive Gaussian** (Deriche filter) for large kernels

#### 3. Median Blur
**Current**: `cv2.medianBlur`

```python
cv2.medianBlur(image, ksize)  # Max 4 channels
```

**Use cases**:
- `MedianBlur`
- Salt-and-pepper noise removal

**Potential improvements**:
- Arbitrary channels
- Batch median blur
- Histogram-based median (constant time per pixel)
- SIMD-optimized sorting networks for small kernels

#### 4. Bilateral Filter
**Current**: `cv2.bilateralFilter`

```python
cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)  # Max 4 channels
```

**Use cases**:
- Edge-preserving smoothing
- `Defocus` variants

**Potential improvements**:
- Arbitrary channels
- SIMD-optimized distance calculations
- Batch processing

#### 5. Box Blur / Mean Filter
**Current**: `cv2.blur`, `cv2.boxFilter`

```python
cv2.blur(image, (ksize, ksize))  # Max 4 channels
```

**Use cases**:
- `Blur`
- Integral image-based operations

**Potential improvements**:
- Arbitrary channels
- SIMD integral images
- Batch processing

#### 6. Morphological Operations
**Current**: `cv2.erode`, `cv2.dilate`, `cv2.morphologyEx`

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
cv2.erode(image, kernel)   # Max 4 channels
cv2.dilate(image, kernel)  # Max 4 channels
```

**Use cases**:
- Morphological operations on masks
- `CoarseDropout` (mask generation)
- Custom transforms using morphology

**Potential improvements**:
- Arbitrary channels
- SIMD-optimized min/max operations
- Van Herk/Gil-Werman algorithm for rectangular kernels
- 3D morphology for volumes

### D. Color Space Operations

#### 1. Color Space Conversions
**Current**: `cv2.cvtColor` (37 uses - VERY FREQUENT)

**Actual usage in AlbumentationsX**:
- `albumentations/augmentations/pixel/functional.py`:
  - `shift_hsv()` - HSV color adjustments
  - `shift_hls()` - HLS color adjustments
  - `to_gray()` - grayscale conversion
  - `superpixels()` - uses LAB color space
  - Various color-based augmentations

**Color conversions used**:
```python
# RGB ↔ HSV (most common for color augmentation)
cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Used in HueSaturationValue
cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

# RGB ↔ HLS
cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Used in HLS adjustments
cv2.cvtColor(image, cv2.COLOR_HLS2RGB)

# RGB ↔ LAB (perceptually uniform)
cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Used in quality-aware augmentations
cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

# RGB ↔ YCrCb
cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)

# RGB ↔ Grayscale
cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # Most common grayscale conversion
cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)   # Convert back to RGB
cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# BGR ↔ RGB (OpenCV default is BGR)
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

**Constraints**:
- Specific channel counts required (typically 3 for RGB, 1 for gray)
- Fixed conversions only (no custom color spaces)
- No batch processing
- Some conversions use LUTs, others use matrix multiplication

**Use cases** (CRITICAL for color augmentations):
- `HueSaturationValue` - adjust hue/saturation/value in HSV space
- `HueSaturationHLS` - adjust in HLS space
- `RGBShift` - per-channel color shifts
- `ToGray` - convert to grayscale
- `ToSepia` - sepia tone (RGB → grayscale → tint)
- `Superpixels` - uses LAB for perceptual grouping
- `PixelDistributionAdaptation` - LAB-based domain adaptation
- `FancyPCA` - PCA on RGB channels
- Preprocessing for color-sensitive models

**Potential improvements**:
- **SIMD-optimized conversions**:
  - RGB↔HSV: Vectorize min/max operations, division, trigonometry
  - RGB↔LAB: Matrix multiplication with SIMD
  - RGB↔YCrCb: Simple matrix mult, easy to vectorize
- **Batch conversions** for video `(N, H, W, 3)`
- **LUT-accelerated conversions** where applicable (e.g., sRGB gamma)
- **Custom color spaces** - user-defined transformation matrices
- **Fused operations** - combine conversion + adjustment in one pass
  - Example: RGB → HSV → adjust H → RGB in single kernel
- **Float32 optimizations** - current cv2 often converts uint8 → float → uint8

#### 2. Channel Operations
**Current**: NumPy slicing

```python
image[..., [2, 1, 0]]  # Channel shuffle
image[..., channel]     # Extract channel
```

**Use cases**:
- `ChannelShuffle`
- `ChannelDropout`
- RGB to BGR conversion

**Potential improvements**:
- SIMD-optimized channel permutation
- Cache-friendly memory layout transformations

### E. Statistical Operations

#### 1. Histograms
**Current**: `cv2.calcHist`, `np.histogram`

```python
cv2.calcHist([image], [0], None, [256], [0, 256])  # Per channel
```

**Use cases**:
- `Equalize`
- `CLAHE`
- Histogram matching

**Potential improvements**:
- SIMD-optimized histogram computation
- Batch histograms
- Multi-channel histograms

#### 2. Normalization
**Current**: `cv2.normalize`, NumPy operations

```python
cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Max 4 channels
(image - mean) / std  # NumPy
```

**Use cases**:
- `Normalize`
- Standardization
- Feature scaling

**Potential improvements**:
- Fused mean/std computation with normalization
- SIMD-optimized reduction operations
- Batch normalization

#### 3. Reductions (min, max, sum, mean)
**Current**: NumPy reductions

```python
np.min(image, axis=(0, 1))
np.max(image, axis=(0, 1))
np.mean(image, axis=(0, 1))
```

**Potential improvements**:
- SIMD-optimized reductions
- Fused min-max computation
- Cache-aware tiling for large arrays

### F. Noise and Randomness

#### 1. Random Number Generation (CRITICAL OPPORTUNITY!)
**Current**: NumPy's `random.Generator` (PCG64 backend) + Python's `random.Random`

**Actual usage in AlbumentationsX**:

**NumPy random_generator usage** (very frequent):
- `random_generator.uniform()` (29 uses) - **MOST COMMON**
  - Used for: parameter sampling, displacement fields, noise generation
- `random_generator.integers()` (29 uses) - **MOST COMMON**
  - Used for: grid distortions, discrete parameter sampling
- `random_generator.normal()` (11 uses) - Gaussian distributions
  - Used for: elastic transforms, perspective distortion, noise
- `random_generator.choice()` (9 uses) - random selection
  - Used for: OneOf, SomeOf transform selection
- `random_generator.shuffle()` (4 uses) - array shuffling
  - Used for: mosaic, tile shuffling
- `random_generator.standard_normal()` (1 use) - standardized Gaussian
- `random_generator.poisson()` (2 uses) - Poisson distribution
- `random_generator.laplace()` (1 use) - Laplace distribution
- `random_generator.beta()` (1 use) - Beta distribution
- `random_generator.random()` (2 uses) - uniform [0, 1)

**Python py_random usage** (simple operations):
- `py_random.random()` - probability checks (p < threshold)
- `py_random.uniform()` - scalar uniform sampling
- `py_random.choice()` - simple selection
- `py_random.randint()` - integer sampling

**cv2 random functions** (minimal use):
- `cv2.setRNGSeed()` (1 use) - set seed for cv2 RNG
- `cv2.randn()` (1 use) - Gaussian noise generation in ISO noise

**Files with heavy random usage**:
- `albumentations/augmentations/geometric/functional.py`:
  - `generate_displacement_fields()` - elastic transforms (Gaussian/uniform noise)
  - `generate_distorted_grid_polygons()` - grid distortions (integers)
  - `create_piecewise_affine_maps()` - jitter (normal distribution)
  - `generate_perspective_points()` - perspective (normal distribution)
- `albumentations/augmentations/pixel/functional.py`:
  - `iso_noise()` - camera sensor noise
  - `gauss_noise()` - Gaussian noise addition
  - `multiplicative_noise()` - multiplicative noise
- `albumentations/core/composition.py`:
  - Transform selection (OneOf, SomeOf)
  - Probability checks

**Performance characteristics**:
- NumPy PCG64 is **reasonably fast** but not optimized for SIMD
- Generating large arrays (e.g., elastic transform displacement fields) is **memory-bound**
- For small arrays, overhead dominates
- No vectorization of distribution transformations (Box-Muller, etc.)

#### 2. Noise Addition
**Current**: NumPy operations after random generation

**Actual usage in AlbumentationsX**:
```python
# Typical pattern:
noise = random_generator.normal(0, std, image.shape).astype(np.float32)
image = image + noise  # or via albucore.add()
```

**Use cases**:
- `GaussNoise` - additive Gaussian noise
- `MultiplicativeNoise` - multiplicative noise
- `ISONoise` - camera sensor noise simulation (complex: uses cv2.randn)
- Elastic transform displacement fields (large arrays!)

**Potential improvements** (HIGH PRIORITY for Ash):

### 🚀 **SIMD Random Number Generation - Major Opportunity!**

**Why this is important**:
1. **Very frequent operation** - used in nearly every augmentation
2. **Memory-bound** - generating large displacement fields for elastic transforms
3. **Current NumPy PCG64 is not SIMD-optimized**
4. **Low-hanging fruit** - RNG is independent of OpenCV, can be fully custom

**What Ash could implement**:

#### Fast SIMD RNG Library
```python
# Proposed: albucore.simd_random or new library

class SIMDRandomGenerator:
    """SIMD-optimized random number generator"""

    def uniform(size, low=0.0, high=1.0, dtype=np.float32):
        # SIMD-optimized uniform generation
        # AVX2/AVX-512 parallel generation
        pass

    def normal(size, mean=0.0, std=1.0, dtype=np.float32):
        # SIMD Box-Muller or Ziggurat algorithm
        pass

    def integers(low, high, size, dtype=np.int32):
        # SIMD integer generation with rejection sampling
        pass
```

**Target operations** (by frequency):
1. **`uniform()` (29 uses)** - Uniform [low, high]
   - SIMD parallel generation
   - Expected: 2-5x faster for large arrays

2. **`integers()` (29 uses)** - Integer uniform [low, high)
   - SIMD with rejection sampling or modulo
   - Expected: 2-4x faster

3. **`normal()` (11 uses)** - Gaussian/Normal distribution
   - SIMD Box-Muller or Ziggurat algorithm
   - Expected: 3-7x faster (Box-Muller is expensive without SIMD)

4. **`choice()` (9 uses)** - Random selection
   - SIMD sampling without replacement
   - Expected: 2-3x faster for large arrays

**Benchmarking priorities**:

**Test case 1: Elastic transform displacement** (MOST IMPACTFUL)
```python
# Current (NumPy):
shape = (512, 512)  # Or larger
dx = random_generator.normal(0, 1, shape, dtype=np.float32)
dy = random_generator.normal(0, 1, shape, dtype=np.float32)
# ~5-10ms for 512x512

# Target (SIMD):
dx, dy = simd_random.normal(0, 1, (2, 512, 512), dtype=np.float32)
# Target: ~1-2ms (5x speedup)
```

**Test case 2: Grid distortion** (HIGH IMPACT)
```python
# Current:
displacements = random_generator.integers(-magnitude, magnitude+1, (H, W, 2))
# ~2-5ms

# Target:
displacements = simd_random.integers(-magnitude, magnitude+1, (H, W, 2))
# Target: <1ms (3-5x speedup)
```

**Test case 3: Parameter sampling** (MODERATE IMPACT)
```python
# Current (very frequent, small arrays):
angle = random_generator.uniform(-45, 45)  # Scalar
scale = random_generator.uniform(0.8, 1.2)  # Scalar
# Overhead dominates here, less important

# But for batches:
angles = random_generator.uniform(-45, 45, size=1000)
# Target: 2-3x speedup
```

**Implementation notes for Ash**:

1. **PRNG algorithm options**:
   - PCG (Permuted Congruential Generator) - good quality, SIMD-friendly
   - xoshiro256** - excellent quality, slightly harder to vectorize
   - Philox/Threefry (counter-based) - perfect for SIMD, excellent quality
   - **Recommendation**: Philox or PCG for SIMD parallelism

2. **Distribution transformations**:
   - Uniform: Direct from PRNG output (easy)
   - Normal: Ziggurat algorithm (SIMD-friendly) or Box-Muller (needs SIMD trig)
   - Integers: Rejection sampling or Lemire's method (unbiased)

3. **Memory layout**:
   - Generate in SIMD registers, write out in cache-friendly patterns
   - Consider AoS vs SoA for multi-dimensional output

4. **Quality requirements**:
   - Statistical quality: Must pass TestU01 SmallCrush (minimum)
   - No need for cryptographic security
   - Period: 2^128 or better
   - Independent streams for multi-threading

5. **API compatibility**:
   - Drop-in replacement for `np.random.Generator`
   - Or: Add to albucore with smart dispatch (like arithmetic ops)

**Expected overall impact**:
- **Elastic transforms**: 5-10x faster (displacement generation is bottleneck)
- **Grid distortions**: 3-5x faster
- **Noise augmentations**: 2-4x faster (generation + addition)
- **Overall pipeline**: 5-15% speedup (RNG is significant but not dominant)

**Integration strategy**:
```python
# Option 1: Extend albucore (RECOMMENDED)
from albucore import create_random_generator

# Smart dispatch based on size and operation
rng = create_random_generator(seed=42)
noise = rng.normal(0, 1, (512, 512, 3))  # → SIMD backend for large arrays
param = rng.uniform(-1, 1)  # → NumPy for scalars (overhead not worth it)

# Option 2: Explicit backend selection
from albucore.simd_random import SIMDRandomGenerator
rng = SIMDRandomGenerator(seed=42)
```

**Low-hanging fruit for Phase 1 prototype**:
1. Implement SIMD `uniform()` for float32 - simplest case
2. Benchmark against NumPy for elastic transform use case
3. Validate statistical quality (TestU01)
4. Measure speedup: target 3-5x for 512x512 arrays

### G. Advanced Operations

#### 1. Fourier Transforms
**Current**: `np.fft`

```python
np.fft.fft2(image[..., channel])  # Per channel
```

**Use cases**:
- Frequency domain augmentations
- `FDA` (Fourier Domain Adaptation)

**Potential improvements**:
- SIMD-optimized FFT (e.g., FFTW-style)
- Batch FFT
- Multi-channel FFT

#### 2. Distance Transforms
**Current**: `scipy.ndimage.distance_transform_edt`, `cv2.distanceTransform`

```python
cv2.distanceTransform(mask, cv2.DIST_L2, 5)  # Single channel only
```

**Use cases**:
- Mask-based augmentations
- Distance-based blending

**Potential improvements**:
- Batch distance transforms
- 3D distance transforms
- SIMD-optimized algorithms

#### 3. Connected Components
**Current**: `cv2.connectedComponents`

```python
num_labels, labels = cv2.connectedComponents(mask)  # Single channel
```

**Use cases**:
- Object-level augmentations
- Mask processing

**Potential improvements**:
- Batch processing
- 3D connected components
- Faster union-find implementations

#### 4. Interpolation Kernels
**Current**: cv2 built-in interpolations

**Available**:
- Nearest neighbor
- Bilinear
- Bicubic
- Lanczos

**Potential improvements**:
- Custom interpolation kernels
- SIMD-optimized interpolation
- Higher-quality methods (e.g., Mitchell-Netravali)

## Target Data Types

### Current Support Matrix

| Data Type | cv2 Support | NumPy Support | Priority |
|-----------|-------------|---------------|----------|
| uint8 | ✅ Excellent | ✅ Good | High |
| float32 | ⚠️ Limited | ✅ Good | High |
| uint16 | ⚠️ Limited | ✅ Good | Medium (medical) |
| float16 | ❌ No | ✅ Limited | Low |
| int8 | ❌ No | ✅ Good | Low |
| int16 | ⚠️ Limited | ✅ Good | Low |

### Target Shapes

1. **Images**: `(H, W, C)` where `C` can be arbitrary
2. **Batched Images**: `(N, H, W, C)`
3. **Videos**: `(N, H, W, C)` (same as batched, semantic difference)
4. **Volumes (3D)**: `(D, H, W, C)` for medical imaging
5. **Batched Volumes**: `(N, D, H, W, C)`

## Existing Solutions & Benchmarks

### StringZilla
- **Purpose**: String operations with SIMD, including LUT operations
- **Current integration**: Already used in AlbumentationsX via `albucore.sz_lut`
- **Relevance**: Shows SIMD optimization patterns, LUT acceleration
- **Performance**: 5-10x faster than standard implementations for string ops
- **URL**: https://github.com/ashvardanian/StringZilla
- **What we can learn**:
  - SIMD dispatch based on CPU capabilities (AVX2, AVX-512, NEON)
  - Efficient LUT implementations
  - Cross-platform optimization strategies

### SimSIMD
- **Purpose**: SIMD-accelerated similarity metrics and vector operations
- **Operations**: Dot product, cosine similarity, Euclidean distance, Hamming distance
- **Relevance**:
  - Matrix operations for color space conversions
  - Distance calculations for TPS, PCA
  - Batch operations on vectors
- **Performance**: Up to 100x faster than NumPy for specific operations
- **URL**: https://github.com/ashvardanian/SimSIMD
- **What we can apply**:
  - `cv2.gemm` (5 uses) - matrix multiply → SimSIMD dot product
  - `cv2.multiply` (11 uses) - element-wise multiply → SIMD vectorization
  - Distance transforms → SimSIMD distance functions
  - Color space matrix multiplications → SIMD matrix ops

**Potential collaboration areas with Ash**:
1. Extend StringZilla LUT to support:
   - Arbitrary channels (C > 4)
   - uint16 LUTs for medical imaging
   - Multi-dimensional LUTs (3D color grading)

2. Use SimSIMD for:
   - Matrix operations in TPS (Thin Plate Spline)
   - PCA operations (`cv2.PCACompute2`, `cv2.gemm`)
   - Color space transformation matrices

3. New SIMD operations needed:
   - Image interpolation (bilinear, bicubic)
   - 2D convolution (separable and general)
   - Geometric transformations (warpAffine, remap)
   - These are NOT in StringZilla/SimSIMD currently

### Other Libraries

#### Kornia (PyTorch-based)
- GPU-accelerated
- Differentiable operations
- Supports arbitrary channels and batch dimensions
- **Trade-off**: Requires PyTorch, GPU for best performance

#### JAX-based augmentations
- JIT compilation
- XLA optimization
- **Trade-off**: JAX dependency, different API

#### Pillow-SIMD
- SIMD-optimized PIL operations
- **Limitations**: Still PIL API, limited operations

#### DALI (NVIDIA)
- GPU-accelerated data loading and augmentation
- **Trade-off**: NVIDIA GPUs only, complex setup

## Performance Benchmarking Plan

### Methodology

1. **Baseline Measurements**: Current cv2 + NumPy implementations
2. **Target Operations**: Prioritized list based on:
   - Frequency of use in transforms
   - Current performance bottlenecks
   - Gap between theoretical peak and current performance

3. **Test Datasets**:
   - **Images**: uint8 `(1024, 1024, 3)`, `(1024, 1024, 16)` (hyperspectral)
   - **Batches**: uint8 `(32, 512, 512, 3)`
   - **Float32**: `(1024, 1024, 3)` for neural network pipelines
   - **Volumes**: float32 `(64, 128, 128, 1)` for medical imaging

4. **Metrics**:
   - Throughput (images/second, GB/s)
   - Latency (ms per operation)
   - CPU utilization
   - Memory bandwidth utilization
   - Cache miss rates

### Priority Operations for Benchmarking

#### Tier 1 (Highest Impact - 80% performance gain potential)

1. **Affine/Perspective transforms** (`warpAffine`, `warpPerspective`, `remap`)
   - **Files**: `albumentations/augmentations/geometric/functional.py`
     - `warp_affine()` (line ~938)
     - `warp_affine_with_value_extension()` (line ~899)
     - `remap()` (line ~2088)
     - `keypoints_remap()` (line ~2118)
     - `bbox_remap()` (line ~2260)
   - **Used by**: Virtually every geometric augmentation
   - **Limitation**: Max 4 channels, no batch processing
   - **Test datasets**:
     - Standard: uint8 `(1024, 1024, 3)` - baseline comparison
     - Hyperspectral: uint8 `(1024, 1024, 16)` - **CURRENTLY FAILS**
     - Batch: uint8 `(32, 512, 512, 3)` - **CURRENTLY REQUIRES LOOP**
     - Medical: float32 `(128, 128, 128, 1)` - **3D NOT SUPPORTED**
   - **Expected speedup**: 2-5x for multi-channel, 3-8x for batch

2. **Resize operations** (`resize`)
   - **Files**: `albumentations/augmentations/geometric/functional.py`
     - `resize_cv2()` (line ~447)
     - `scale_keypoint_mask_remap()` (line ~2248) - uses resize for displacement maps
   - **Used by**: Almost all pipelines (preprocessing step)
   - **Current performance**: ~100-500 images/sec depending on size and interpolation
   - **Expected speedup**: 1.5-3x for multi-channel, 2-5x for batch

3. **Remap operations** (already covered in #1, but critical enough to emphasize)
   - **Memory intensive**: 2x float32 `(H, W)` arrays per image
   - **Expected speedup**: 2-4x with quantized grids + SIMD interpolation

#### Tier 2 (Common Operations - 15% performance gain potential)

4. **Color space conversions** (RGB↔HSV, RGB↔Lab)
   - **Files**: `albumentations/augmentations/pixel/functional.py`
     - `shift_hsv()` (line ~63)
     - `shift_hls()`
     - Various color augmentations
   - **Current**: 37 uses of `cv2.cvtColor` in codebase
   - **Expected speedup**: 1.5-2.5x with SIMD

5. **Gaussian blur** (separable convolution)
   - **Files**: `albumentations/augmentations/blur/functional.py`
     - `gaussian_blur()`
   - **Current**: 17 uses, separable implementation already efficient
   - **Expected speedup**: 1.2-2x for arbitrary channels, 2-3x for batch

6. **Random number generation** - 🚀 **NEW OPPORTUNITY**
   - **Files**: Throughout codebase
     - `generate_displacement_fields()` - elastic transforms
     - `generate_distorted_grid_polygons()` - grid distortions
     - All noise augmentations
   - **Current**: NumPy PCG64 (not SIMD-optimized)
     - `uniform()`: 29 uses
     - `integers()`: 29 uses
     - `normal()`: 11 uses
   - **Test datasets**:
     - Elastic transform: generate 2x `(512, 512)` float32 Gaussian - **~5-10ms**
     - Grid distortion: generate `(10, 10, 2)` integers - **~0.5ms**
     - Noise: generate `(1024, 1024, 3)` Gaussian - **~3-5ms**
   - **Expected speedup**: 3-7x for large arrays (SIMD Ziggurat/Box-Muller)
   - **Impact**: 5-15% overall pipeline speedup
   - **Advantage**: Independent of OpenCV, can be pure SIMD implementation!

7. **Arithmetic operations** (multiply, add with saturation)
   - **Files**: Throughout codebase, via `albucore` wrappers
   - **Current**: cv2.multiply (11+6 uses), cv2.add (5+4 uses), cv2.addWeighted (4+1 uses)
   - **Expected speedup**: 1.3-2x with fused operations

8. **Convolution** (3x3, 5x5 kernels)
   - **Files**: `albumentations/augmentations/geometric/functional.py`
     - `filter2D` (3 uses)
   - **Expected speedup**: 2-4x with SIMD for small kernels

#### Tier 3 (Specialized - 5% performance gain potential)
8. **Median filter**
9. **Morphological operations**
10. **FFT operations**

## Implementation Roadmap Proposal

### Phase 1: Assessment & Prototyping (4-6 weeks)

**Goal**: Validate SIMD approach with 1-2 prototype operations

**Option A: Start with SIMD RNG (RECOMMENDED - Low-hanging fruit)**
- [ ] Implement SIMD uniform() for float32 (simplest case)
- [ ] Implement SIMD normal() using Ziggurat algorithm
- [ ] Benchmark vs NumPy for elastic transform use case (512x512 arrays)
- [ ] Validate statistical quality (TestU01 SmallCrush minimum)
- [ ] Measure speedup: target 3-5x
- **Advantages**:
  - Independent of OpenCV
  - Easier to prototype
  - Immediate impact on elastic transforms
  - Can work in parallel with other efforts

**Option B: Start with warpAffine/resize (Higher impact but harder)**
- [ ] Complete comprehensive benchmarking of current operations
- [ ] Identify top 10 bottlenecks
- [ ] Create prototype SIMD implementation for warpAffine or resize
- [ ] Compare against cv2 baselines
- [ ] Validate correctness (pixel-perfect or within tolerance)
- **Advantages**:
  - Highest overall impact
  - Proves SIMD viability for image ops
- **Challenges**:
  - More complex (interpolation, border modes)
  - Requires more upfront design

**Recommendation**: Start with Option A (SIMD RNG), then move to Option B in Phase 2

### Phase 2: Core Operations (8-12 weeks)
- [ ] Implement SIMD-optimized affine/perspective transforms
- [ ] Implement arbitrary-channel resize
- [ ] Implement arbitrary-channel convolution/filtering
- [ ] Implement color space conversions
- [ ] Create test suite for all operations
- [ ] Ensure numerical accuracy matches cv2 within tolerance

### Phase 3: Batch Support (6-8 weeks)
- [ ] Extend operations to support batch dimension `(N, H, W, C)`
- [ ] Implement batch-optimized memory layouts
- [ ] Optimize for cache locality in batch processing
- [ ] Benchmark batch vs sequential processing

### Phase 4: 3D/Volume Support (8-10 weeks)
- [ ] Implement 3D geometric transforms
- [ ] Implement 3D filtering operations
- [ ] Create 3D-specific transforms (e.g., `Rotate3D`, `Resize3D`)
- [ ] Test with medical imaging datasets

### Phase 5: Integration & Migration (4-6 weeks)
- [ ] Create compatibility layer for seamless migration
- [ ] Add feature flags for cv2 vs custom implementations
- [ ] Update documentation
- [ ] Performance comparison documentation
- [ ] Migration guide for users

## API Design Considerations

### Backward Compatibility
- Must maintain 100% API compatibility with existing AlbumentationsX
- Internal implementation swaps should be transparent
- Feature flags for experimental operations

### Configuration
```python
# Proposed configuration
A.Compose([
    A.Rotate(limit=45, p=0.5),
    ...
],
backend="simd",  # Options: "auto", "cv2", "simd", "hybrid"
batch_mode=True,  # Enable batch optimization
)
```

### Gradual Migration
- Hybrid mode: Use SIMD for operations where available, fall back to cv2
- Per-transform backend selection
- Runtime detection of CPU capabilities (AVX2, AVX-512, NEON)

## Open Questions

1. **CPU Capability Detection**: How to detect and utilize available SIMD instructions at runtime?
2. **Compilation**: Distribute pre-compiled binaries for common architectures or compile on user's machine?
3. **Numerical Precision**: What tolerance is acceptable for floating-point differences vs cv2?
4. **Edge Cases**: How to handle edge cases that cv2 handles gracefully?
5. **Testing**: How to ensure comprehensive testing across different CPU architectures?
6. **License Compatibility**: Ensure SIMD implementations are compatible with AlbumentationsX license (MIT)
7. **Maintenance**: Long-term maintenance strategy for low-level code

## Success Criteria

### Performance Goals
- **2-5x speedup** for multi-channel operations (C > 4)
- **1.5-3x speedup** for batch operations vs sequential
- **3-10x speedup** for 3D volume operations (vs current NumPy fallbacks)
- Match or exceed cv2 performance for standard operations (C <= 4)

### Capability Goals
- Support for arbitrary channel counts (tested up to C=64)
- Native batch processing `(N, H, W, C)`
- 3D volume support `(D, H, W, C)` and `(N, D, H, W, C)`
- All current AlbumentationsX transforms work with new backend

### Quality Goals
- 100% test coverage for new operations
- Numerical accuracy within 1 ULP (unit of least precision) for float32
- Pixel-perfect for uint8 where cv2 is pixel-perfect
- Zero regressions in existing functionality

## Resources & References

### Ash Vardanyan's Work
- **GitHub**: https://github.com/ashvardanian
- **StringZilla**: https://github.com/ashvardanian/StringZilla
- **SimSIMD**: https://github.com/ashvardanian/SimSIMD
- **Blog**: https://ashvardanian.com/

### SIMD Resources
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- ARM NEON Intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
- Agner Fog's Optimization Manuals: https://www.agner.org/optimize/

### Computer Vision References
- OpenCV Source: https://github.com/opencv/opencv
- Fast Image Processing: http://fgiesen.wordpress.com/
- "Modern Algorithms for Image Processing" by V. Kovalevsky

### Relevant Papers
- Fast Bilateral Filter: Paris & Durand (2006)
- Constant-Time Median Filter: Perreault & Hébert (2007)
- Separable Convolution for Large Kernels: Van Vliet et al. (1998)

## Contact & Collaboration

**Ash Vardanyan** has expressed willingness to invest time in this modernization effort.

### Immediate Next Steps

1. **Review this document** with Ash and AlbumentationsX maintainers
2. **Prioritize operations**: Agree on Tier 1 operations (warpAffine, resize, remap)
3. **Create benchmarking suite** for baseline measurements
4. **Prototype 1-2 operations** to validate approach and measure gains
5. **Define API** for backend selection and configuration

### Questions for Ash

#### Technical Architecture
1. **Library structure**: Should this be:
   - ✅ **RECOMMENDED: Extend albucore** (existing optimization layer with StringZilla already integrated)
   - New library (e.g., `albumentations-simd`)?
   - Integrated directly into albumentations?

   **Rationale for albucore**:
   - Already has dispatch infrastructure (`multiply_opencv` vs `multiply_numpy`)
   - StringZilla already integrated (proven collaboration!)
   - AlbumentationsX code doesn't need to change
   - Just add: `multiply_simd()`, `warp_affine_simd()`, etc. as new backends

2. **CPU capability detection**:
   - How to detect AVX2/AVX-512/NEON at runtime?
   - Fallback strategy for older CPUs?
   - Pre-compiled binaries vs compile-on-install?

3. **Operations scope**: Which operations should we prioritize?
   - Tier 1: warpAffine, resize, remap (agree/disagree?)
   - **Tier 2 addition: SIMD Random Number Generation** - Low-hanging fruit?
     - uniform(), integers(), normal() are used 60+ times
     - Independent of OpenCV, pure SIMD implementation
     - Elastic transforms are bottlenecked by RNG (5-10ms for 512x512)
   - Can we leverage existing SimSIMD for matrix ops?
   - Can we extend StringZilla for multi-channel LUT?

#### Implementation Strategy
4. **Interpolation kernels**:
   - What's the best SIMD strategy for bilinear/bicubic interpolation?
   - Separable filtering for resize/blur?
   - IIR/recursive filters for large Gaussian kernels?

5. **Random number generation** (NEW - potentially easiest to prototype!):
   - Which PRNG algorithm? (PCG, xoshiro, Philox/Threefry?)
   - Philox seems ideal for SIMD (counter-based, parallel generation)
   - Distributions: Ziggurat (normal), Lemire's method (integers), direct (uniform)?
   - Statistical quality requirements: TestU01 SmallCrush sufficient?
   - Can this be Phase 1 prototype? (Independent of OpenCV migration)

6. **Memory layout**:
   - Current: `(H, W, C)` channel-last (HWC)
   - Should we support `(C, H, W)` channel-first for better vectorization?
   - Transposed operations?

7. **Quantization**:
   - Fixed-point arithmetic for integer speedup?
   - Quantized displacement fields for remap (uint16 + 8-bit fraction)?

#### Integration & Testing
7. **Numerical precision**:
   - What tolerance is acceptable vs cv2? (1 ULP for float32? Pixel-perfect for uint8?)
   - How to handle edge cases that cv2 handles gracefully?

8. **Testing strategy**:
   - Test against cv2 on multiple architectures (x86, ARM)?
   - Fuzzing for edge cases?
   - Performance regression tracking?

9. **Gradual migration**:
   - Feature flags to enable/disable SIMD backend?
   - Hybrid mode (SIMD where available, cv2 fallback)?
   - Per-transform backend selection?

#### Performance Targets
10. **Realistic speedup expectations**:
    - For standard operations (C ≤ 4): Match or exceed cv2?
    - For multi-channel (C > 4): 2-5x vs current NumPy fallback?
    - For batch processing: 3-8x vs sequential?

11. **Memory overhead**:
    - Acceptable memory increase for performance?
    - Scratch buffer management strategy?

#### Licensing & Maintenance
12. **License**: MIT compatible?
13. **Long-term maintenance**: Who maintains low-level SIMD code?
14. **Documentation**: Level of documentation needed for SIMD implementations?

### Collaboration Framework

**Proposed workflow**:
1. Ash creates prototype for 1-2 Tier 1 operations
2. AlbumentationsX team integrates and tests
3. Measure performance gains and validate correctness
4. Iterate on API and implementation
5. Expand to more operations based on learnings
6. Document and release

**Communication channels**:
- GitHub issues for technical discussion
- Regular sync meetings (bi-weekly?)
- Shared benchmarking infrastructure

---

**Document Status**: ✅ Ready for Discussion
**Created**: 2026-02-05
**Author**: AlbumentationsX Team (with AI assistance)
**Next Review**: TBD with Ash Vardanyan
**Target**: Create detailed implementation plan for Phase 1

## Document Statistics

**Total cv2 operations analyzed**:
- Direct usage in albumentations: 95 symbols (65 functions + 30 constants)
- Indirect via albucore: 15 operations (multiply, add, pow, flip, etc.)
- **Total: 110 unique cv2 operations**

**Critical bottlenecks identified**: 3 (warpAffine, resize, remap)
**High-frequency operations via albucore**: 8 (add, multiply, power, flip, add_weighted, etc.)
**Use cases blocked**: 5 (hyperspectral, medical, video, 3D, microscopy)
**Expected impact**: 2-10x speedup, unlocking entire new domains

**Key insight**: AlbumentationsX is **more dependent on cv2** than it appears at first glance, because albucore provides cv2-backed implementations for arithmetic operations used throughout the codebase. However, **NumPy operations are also pervasive** and many are not optimally SIMD-vectorized.

## NumPy Operations Analysis

Beyond cv2, AlbumentationsX makes **extensive use of NumPy** operations. Many of these could benefit from SIMD optimization:

### NumPy Usage Statistics (124 unique operations)

**Most frequent NumPy operations** (by usage count):
- `np.ndarray` (1547) - type hints
- `np.uint8`, `np.float32`, `np.int32` (569) - dtypes
- `np.random.*` (264) - **Already covered in RNG section**
- `np.array` (256) - array creation
- `np.zeros` (84) - array initialization
- **`np.clip` (73)** - **CRITICAL - saturation/bounds checking**
- `np.rot90` (41) - 90° rotation
- **`np.where` (27)** - conditional selection
- `np.any`, `np.all` (30) - boolean reductions
- `np.stack` (25) - array stacking
- `np.linspace`, `np.arange` (50) - range generation
- `np.sum` (24) - reduction
- **`np.sqrt` (16)** - square root (expensive!)
- `np.max`, `np.min` (22) - reductions
- `np.abs` (13) - absolute value
- **`np.sin`, `np.cos` (22)** - trigonometry (expensive!)
- **`np.maximum`, `np.minimum` (11)** - element-wise comparisons
- **`np.exp` (5)** - exponential (expensive!)
- `np.dot` (6) - matrix multiplication
- `np.linalg.*` (6) - linear algebra
- `np.fft.*` (6) - FFT operations

### SIMD Optimization Opportunities in NumPy Operations

#### 🔴 **High Priority** (Frequent + Expensive)

1. **`np.clip` (73 uses) - CRITICAL**
   ```python
   # Used everywhere for bounds checking
   np.clip(coordinates, 0, width-1)
   np.clip(values, 0, 255)
   ```
   **Where used**: Keypoint clamping, image saturation, parameter bounds, piecewise affine
   **SIMD potential**: Vectorize min/max comparisons
   **Expected speedup**: 2-3x for large arrays

2. **`np.sqrt` (16 uses) - Expensive**
   ```python
   distances = np.sqrt((x2-x1)**2 + (y2-y1)**2)
   ```
   **Where used**: Distance transforms, piecewise affine, optical distortion
   **SIMD potential**: AVX `vsqrtps`, `vrsqrtps` (reciprocal sqrt)
   **Expected speedup**: 2-4x with `rsqrtps` approximation

3. **`np.sin`, `np.cos` (22 uses) - Very Expensive**
   ```python
   cos_angles = np.cos(np.radians(angles))
   sin_angles = np.sin(np.radians(angles))
   ```
   **Where used**: Rotation matrices, optical distortion, ellipse generation, defocus blur
   **SIMD potential**: SVML (Intel), SLEEF library, or LUT approximations
   **Expected speedup**: 3-10x with SIMD trig

4. **`np.where` (27 uses) - Conditional selection**
   ```python
   result = np.where(condition, value_if_true, value_if_false)
   ```
   **Where used**: Thin plate spline, angle adjustments, division by zero handling
   **SIMD potential**: Blend instructions (`vblendvps`)
   **Expected speedup**: 1.5-2x

#### 🟡 **Medium Priority**

5. **`np.maximum`, `np.minimum` (11 uses)** - SIMD `vmax`/`vmin`, 2-3x speedup
6. **`np.exp` (5 uses)** - Polynomial approximation + SIMD, 3-8x speedup
7. **`np.dot`, `np.linalg.*` (12 uses)** - SimSIMD integration, 1.5-2x over BLAS
8. **`np.sum`, `np.mean` (32 uses)** - Horizontal add instructions, 1.5-2x
9. **`np.rot90` (41 uses)** - Cache-aware tiled transpose, 1.5-2x

### NumPy Operations by Transform (Video Benchmark Context)

| Transform (v/s) | NumPy Operations | SIMD Opportunity |
|-----------------|------------------|------------------|
| **ThinPlateSpline (4.51)** | `np.sqrt`, `np.where`, `np.dot` | **HIGH** |
| **OpticalDistortion (~5)** | `np.sqrt`, `np.arctan2`, `np.sin`, `np.cos` | **VERY HIGH** |
| **PiecewiseAffine (~5)** | `np.clip`, `np.sum` | Medium |
| **Elastic (4.03)** | `np.clip` | Medium |

### Expected Impact: NumPy SIMD + Video Benchmarks

**Conservative estimates**:

| Transform | Current | After NumPy SIMD | Speedup | Key Operations |
|-----------|---------|------------------|---------|----------------|
| ThinPlateSpline | 4.51 v/s | **9-13 v/s** | 2-3x | `np.sqrt`, `np.dot` |
| OpticalDistortion | ~5 v/s | **15-25 v/s** | 3-5x | `np.sin`, `np.cos`, `np.sqrt` |
| PiecewiseAffine | ~5 v/s | **8-12 v/s** | 1.5-2.5x | `np.clip`, `np.sum` |

**Combined cv2 + NumPy SIMD**: Operations like Elastic that use both get **multiplicative speedups**!

### Integration: Use SLEEF or Custom SIMD Math

**SLEEF** (SIMD Library for Evaluating Elementary Functions):
- Provides SIMD trig, exp, log, sqrt
- Used by Julia, Rust
- MIT licensed
- **Can be wrapped for NumPy arrays**

```python
# Proposed: albucore.simd_numpy module
from albucore.simd_numpy import clip, sqrt, sin, cos, where, exp

# Smart dispatch based on array size and dtype
result = clip(coordinates, 0, width-1)  # → SIMD for large arrays
```

## What SimSIMD Already Provides (Can Use Immediately!)

**SimSIMD** (by Ash Vardanyan) is **already available** and provides several operations we can leverage:

### ✅ **Currently Available in SimSIMD v6.5+**

#### 1. **Dot Products & Inner Products** (Already have!)
```python
import simsimd

# Vector dot products (real)
result = simsimd.dot(vec1, vec2)  # f64, f32, f16, bf16, i8
result = simsimd.inner(vec1, vec2)  # Same as dot for real vectors

# Complex dot products
result = simsimd.dot(complex_vec1, complex_vec2, "complex64")
result = simsimd.vdot(complex_vec1, complex_vec2, "complex64")  # Conjugate
```

**Where we can use it in AlbumentationsX**:
- ✅ **Replace `cv2.gemm`** (5 uses) - matrix multiplication in TPS
- ✅ **Replace `np.dot`** (6 uses) - dot products throughout codebase
- ✅ **ThinPlateSpline** optimization - uses matrix operations heavily
- ✅ **PCA operations** - `cv2.PCACompute2`, `cv2.PCAProject` use dot products

**Expected speedup**: 1.5-2x over BLAS for small matrices, up to **10x over NumPy**

#### 2. **Distance Metrics** (Already have!)
```python
import simsimd

# Euclidean distance (squared)
dist = simsimd.sqeuclidean(vec1, vec2)  # f64, f32, f16, bf16, i8

# Cosine distance/similarity
dist = simsimd.cosine(vec1, vec2)

# Hamming & Jaccard for binary
dist = simsimd.hamming(bin_vec1, bin_vec2, "bin8")
dist = simsimd.jaccard(bin_vec1, bin_vec2, "bin8")
```

**Where we can use it**:
- ✅ **Distance calculations** in piecewise affine, TPS
- ✅ **Similarity metrics** for texture/feature matching
- Potentially useful for future similarity-based transforms

#### 3. **Fused-Multiply-Add (FMA)** (Already have! 🎉)
```python
import simsimd

# FMA: alpha * a * b + beta * c
simsimd.fma(a, b, c, alpha=0.7, beta=0.3, out=result)

# Weighted Sum: alpha * a + beta * b
simsimd.wsum(a, b, alpha=0.5, beta=0.5, out=result)
```

**WHERE WE CAN USE IT IN ALBUMENTATIONSX** (Immediate opportunity!):
- ✅ **Replace `albucore.multiply_add`** - currently uses cv2.multiply + cv2.add sequentially
- ✅ **Replace `cv2.addWeighted`** (4+1 uses) - alpha blending operations
- ✅ **Brightness/Contrast adjustments** - `image * alpha + beta`
- ✅ **PixelDistributionAdaptation** - weighted blending of images
- ✅ **Phong shading models** in illumination transforms
- ✅ **Multi-channel operations** - FMA supports arbitrary channels!

**Code example for immediate integration**:
```python
# Current in AlbumentationsX (via albucore):
from albucore import multiply_add
result = multiply_add(image, alpha, beta)  # → cv2.multiply + cv2.add

# With SimSIMD (faster + arbitrary channels):
import simsimd
simsimd.fma(image, alpha_array, beta_scalar, out=result)  # Fused, one pass!

# For simple brightness/contrast:
# result = image * contrast + brightness
simsimd.wsum(image, brightness, alpha=contrast, beta=1.0, out=result)
```

**Expected speedup**: **1.5-3x** over sequential multiply+add, **works with C > 4!**

#### 4. **Set Intersection** (Already have!)
```python
import simsimd

# For sparse vectors/sets
intersection_count = simsimd.intersect(sorted_set1, sorted_set2)
```

**Where we can use it**:
- Potentially useful for sparse vector operations
- Could be used in future sparse augmentation transforms

#### 5. **Mixed Precision Support** (Already have!)
```python
# SimSIMD supports all precisions AlbumentationsX needs:
simsimd.dot(vec1_i8, vec2_i8, "int8")      # int8
simsimd.dot(vec1_f16, vec2_f16, "float16")  # float16 (native!)
simsimd.dot(vec1_bf16, vec2_bf16, "bfloat16")  # bfloat16 (native!)
simsimd.dot(vec1_f32, vec2_f32, "float32")  # float32
simsimd.dot(vec1_f64, vec2_f64, "float64")  # float64
```

**Advantage**: Unlike NumPy, SimSIMD has native `bf16` support!

### 🔴 **What SimSIMD Does NOT Provide** (Need custom implementation)

SimSIMD is focused on **vector similarity/distance metrics**, **NOT** image processing. We still need:

❌ **Image operations**: warpAffine, resize, remap, GaussianBlur
❌ **Element-wise math**: sin, cos, exp, sqrt, clip, where
❌ **Random number generation**: uniform, normal, integers
❌ **Color space conversions**: RGB↔HSV
❌ **Morphological operations**: erode, dilate
❌ **Image interpolation**: bilinear, bicubic

### 🎯 **Immediate Integration Opportunities**

**Phase 0: Use SimSIMD today** (no development needed!)

1. **Replace `cv2.gemm` in TPS** (5 uses)
   ```python
   # albumentations/augmentations/geometric/functional.py
   # Replace: result = cv2.gemm(a, b, 1, None, 0)
   result = simsimd.dot(a, b.T)  # Faster, arbitrary precision
   ```
   **Impact**: ThinPlateSpline: 4.51 v/s → **6-7 v/s** (1.5x speedup)

2. **Replace `cv2.multiply` + `cv2.add` with `simsimd.fma`**
   ```python
   # albumentations/augmentations/pixel/functional.py
   # Replace: multiply_add(image, alpha, beta)
   simsimd.fma(image, alpha, beta, out=image)  # Fused, faster!
   ```
   **Impact**: Brightness/Contrast operations: **1.5-2x speedup**

3. **Replace `cv2.addWeighted` with `simsimd.wsum`**
   ```python
   # Replace: cv2.addWeighted(img1, alpha, img2, beta, 0)
   simsimd.wsum(img1, img2, alpha=alpha, beta=beta, out=result)
   ```
   **Impact**: Blending operations: **1.5-2x speedup**, **works with C > 4!**

4. **Replace `np.dot` in distance calculations**
   ```python
   # Replace: distances = np.dot(matrix1, matrix2.T)
   distances = simsimd.dot(matrix1, matrix2.T)
   ```
   **Impact**: Piecewise affine, feature matching: **3-10x speedup**

### Collaboration Strategy with Ash

**What we can do together**:

1. **Extend SimSIMD** with image-specific operations:
   - Element-wise math: `simsimd_clip`, `simsimd_sqrt`, `simsimd_where`
   - Transcendental functions: `simsimd_sin`, `simsimd_cos`, `simsimd_exp`
   - These fit SimSIMD's scope (SIMD vector operations)

2. **Create separate library** for image operations:
   - Interpolation, resizing, warping (different scope than SimSIMD)
   - Could be `SimSIMD-Image` or part of `albucore`

3. **Use SimSIMD immediately** for:
   - Matrix operations (TPS, PCA)
   - FMA/wsum (brightness, contrast, blending)
   - Distance calculations

**Updated roadmap**:

**Phase 0: Integrate existing SimSIMD** (1-2 weeks)
- Replace `cv2.gemm` → `simsimd.dot`
- Replace `cv2.addWeighted` → `simsimd.wsum`
- Replace `multiply_add` → `simsimd.fma`
- **No new development, just integration!**
- **Expected**: 1.5-2x speedup for TPS, blending, brightness/contrast

**Phase 1A: SIMD RNG** (new development)
- Custom Philox/PCG implementation
- Or extend SimSIMD with RNG

**Phase 1B: Extend SimSIMD with math** (collaborate with Ash)
- Add `clip`, `sqrt`, `where` to SimSIMD
- Add `sin`, `cos`, `exp` to SimSIMD
- These are vector operations (SimSIMD's core competency!)

**Phase 1C: Custom image ops** (separate library or albucore)
- warpAffine, resize, remap
- These require interpolation, border modes (beyond SimSIMD scope)

### Recommended Phasing for NumPy SIMD

**Phase 0: Use SimSIMD today** (🆕)
- Integrate existing FMA, wsum, dot operations
- No development needed, immediate gains
- Target transforms: TPS, blending, brightness/contrast

## Testing & Validation Plan for SimSIMD Integration

### ✅ Discovery: Albucore Already Uses SimSIMD!

**From albucore/functions.py analysis**:
```python
import simsimd as ss

def add_weighted_simsimd(img1, weight1, img2, weight2):
    return ss.wsum(img1.reshape(-1), img2.reshape(-1), alpha=weight1, beta=weight2)
```

**What's already SIMD-accelerated** via albucore:
- `albucore.add_weighted()` → `simsimd.wsum`
- `albucore.multiply_by_constant()` → `simsimd.wsum`
- `albucore.add_constant()` → `simsimd.wsum`
- `albucore.add_array()` → `simsimd.wsum`

**This means**: Brightness, contrast, and blending operations are **already using SimSIMD**!

### 🎯 New Opportunity: cv2.gemm → simsimd.dot

**Currently NOT using SimSIMD**:
- `cv2.gemm` (5 uses) in ThinPlateSpline - matrix operations
- `np.dot` (6 uses) - various dot products
- These could be **3-10x faster** with SimSIMD!

### Test & Benchmark Files Created

#### 1. **`tests/test_simsimd_integration.py`** - Correctness Tests

Validates SimSIMD produces identical results:

**Test classes**:
- `TestMatrixOperations` - cv2.gemm vs simsimd.dot
- `TestWeightedSum` - cv2.addWeighted vs simsimd.wsum
- `TestFusedMultiplyAdd` - simsimd.fma correctness
- `TestDistanceMetrics` - distance calculations

**Key tests**:
```python
# Parametrized test for various matrix sizes
@pytest.mark.parametrize("rows,cols,inner", [(10,10,10), (768,2,768)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gemm_vs_dot_product(rows, cols, inner, dtype):
    result_cv2 = cv2.gemm(a, b, 1, None, 0)
    result_np = a @ b
    np.testing.assert_allclose(result_cv2, result_np, rtol=1e-5)

# Test that simsimd.wsum works with C > 4 (cv2 can't!)
def test_weighted_sum(shape=(100,100,16)):  # 16 channels
    simsimd.wsum(img1, img2, alpha=0.6, beta=0.4, out=result)
    # cv2.addWeighted would FAIL here!
```

**Run**: `pytest tests/test_simsimd_integration.py -v`

#### 2. **`tests/benchmark_simsimd.py`** - Performance Benchmarks

Measures performance using pytest-benchmark:

**Benchmark classes**:
- `TestMatrixMultiplicationBenchmark` - cv2.gemm vs numpy
- `TestVectorDotProductBenchmark` - np.dot vs simsimd.dot (1536-dim)
- `TestWeightedSumBenchmark` - cv2 vs simsimd vs numpy (RGB + hyperspectral)
- `TestFMABenchmark` - sequential vs fused vs numpy
- `TestDistanceBenchmark` - distance metrics

**Example benchmark**:
```python
def test_benchmark_simsimd_dot_1536(benchmark):
    a = np.random.randn(1536).astype(np.float32)
    b = np.random.randn(1536).astype(np.float32)
    result = benchmark(simsimd.dot, a, b)
    # Compare against test_benchmark_numpy_dot_1536
```

**Run benchmarks**:
```bash
pip install pytest-benchmark

# Run all benchmarks
pytest tests/benchmark_simsimd.py -v --benchmark-only

# Compare and save
pytest tests/benchmark_simsimd.py --benchmark-compare
```

### ✅ Test Results (2026-02-05)

#### Correctness Tests: **ALL PASS** (46/46)

All SimSIMD operations produce numerically equivalent results to OpenCV and NumPy:

```bash
$ pytest tests/test_simsimd_integration.py -v
============================== 46 passed in 0.12s ==============================
```

**Key findings**:
- ✅ `simsimd.dot` matches `cv2.gemm` and NumPy matrix multiplication
- ✅ `simsimd.wsum` matches `cv2.addWeighted` for uint8 and float32
- ✅ `simsimd.fma` matches sequential cv2 multiply+add operations
- ✅ `simsimd.sqeuclidean` and `simsimd.cosine` match NumPy distance calculations
- ✅ `albucore.add_weighted` correctly uses SimSIMD with clipping
- ✅ SimSIMD handles C > 4 channels (cv2 limitation bypassed!)

**Tolerance notes**:
- `float32`: `rtol=1e-4, atol=1e-6` (standard precision)
- `float64`: `rtol=1e-6, atol=1e-8` (higher precision)
- `uint8`: `rtol=3/255, atol=2` (allows ±2 rounding differences)

#### Performance Benchmarks: **SIMSIMD WINS**

Platform: **macOS ARM (M-series)**, Python 3.12.7

```
----------------------------------------------------------------------------- benchmark: 20 tests ------------------------------------------------------------------------------
Name (time in ns)                                            Min                       Max                      Mean                  StdDev                       OPS
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_numpy_dot_768                            207.9178 (1.0)         20,416.9191 (10.22)          314.5262 (1.18)         231.9097 (3.75)     3,179,385.8422 (0.84)
test_benchmark_simsimd_dot_768                          228.1005 (1.10)        56,199.5211 (28.13)          265.6652 (1.0)          159.2289 (2.57)     3,764,136.6933 (1.0)
test_benchmark_numpy_dot_1536                           249.8273 (1.20)        16,124.9191 (8.07)           365.7442 (1.38)         193.0102 (3.12)     2,734,151.5216 (0.73)
test_benchmark_simsimd_dot_1536                         372.9016 (1.79)        19,364.5479 (9.69)           420.3556 (1.58)          91.3803 (1.48)     2,378,938.0606 (0.63)
test_benchmark_simsimd_euclidean                        383.3440 (1.84)         1,997.9081 (1.0)            429.9492 (1.62)          61.8734 (1.0)      2,325,856.3997 (0.62)
test_benchmark_simsimd_cosine                           393.9071 (1.89)        13,814.3712 (6.91)           457.4376 (1.72)         101.7318 (1.64)     2,186,090.7236 (0.58)
test_benchmark_numpy_cosine                           1,333.8868 (6.42)        39,000.0641 (19.52)        1,641.5453 (6.18)         423.8921 (6.85)       609,182.0978 (0.16)
test_benchmark_numpy_euclidean                        1,374.8650 (6.61)        30,833.9950 (15.43)        1,701.2689 (6.40)         544.4988 (8.80)       587,796.5460 (0.16)
test_benchmark_dot_small                              2,582.7903 (12.42)       33,083.1390 (16.56)        3,101.8250 (11.68)        708.5588 (11.45)      322,390.8523 (0.09)
test_benchmark_gemm_medium                            3,249.8501 (15.63)       34,082.9138 (17.06)        3,845.6367 (14.48)        631.7183 (10.21)      260,034.9618 (0.07)
test_benchmark_dot_medium                             3,374.8802 (16.23)       60,458.9004 (30.26)        3,943.0112 (14.84)        761.6062 (12.31)      253,613.2789 (0.07)
test_benchmark_gemm_small                            23,375.0325 (112.42)      54,625.0958 (27.34)       25,943.1932 (97.65)      3,082.9248 (49.83)       38,545.7562 (0.01)
test_benchmark_simsimd_wsum_rgb                      51,207.8404 (246.29)     108,832.9591 (54.47)       68,052.8056 (256.16)    11,929.7286 (192.81)      14,694.4713 (0.00)
test_benchmark_simsimd_fma                           77,166.1289 (371.14)     164,082.9723 (82.13)      107,247.8923 (403.70)    11,200.0869 (181.02)       9,324.1926 (0.00)
test_benchmark_cv2_addweighted_rgb                  192,499.9524 (925.85)     418,249.9833 (209.34)     248,883.6019 (936.83)    28,874.1845 (466.67)       4,017.9425 (0.00)
test_benchmark_numpy_weighted_sum_rgb               293,999.9104 (>1000.0)    913,166.9067 (457.06)     398,215.6974 (>1000.0)   47,476.5013 (767.32)       2,511.2019 (0.00)
test_benchmark_numpy_fma                            376,916.9562 (>1000.0)    832,542.0786 (416.71)     462,295.6813 (>1000.0)   42,983.2688 (694.70)       2,163.1178 (0.00)
test_benchmark_simsimd_wsum_hyperspectral           419,792.0207 (>1000.0)  1,909,666.9275 (955.83)     445,926.1397 (>1000.0)   73,949.7098 (>1000.0)      2,242.5238 (0.00)
test_benchmark_sequential_cv2                       858,499.9014 (>1000.0)  2,224,916.8251 (>1000.0)  1,029,562.5576 (>1000.0)  123,979.4821 (>1000.0)        971.2863 (0.00)
test_benchmark_numpy_weighted_sum_hyperspectral     874,874.8805 (>1000.0)  1,023,000.1062 (512.04)     941,001.6194 (>1000.0)   22,039.4373 (356.20)       1,062.6974 (0.00)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### 🎯 Performance Analysis & Decision

#### Vector Operations (768-1536 dims)

| Operation | NumPy (ns) | SimSIMD (ns) | **Speedup** |
|-----------|------------|--------------|-------------|
| dot_768   | 314.5      | 265.7        | **1.18x**   |
| dot_1536  | 365.7      | 420.4        | 0.87x ❌    |

**Analysis**: For vector dot products:
- **768-dim**: SimSIMD is **1.18x faster** (marginal)
- **1536-dim**: NumPy is actually **1.15x faster** (SimSIMD slower!)

**Reason**: On ARM/M-series chips, NumPy already uses Apple's Accelerate framework (highly optimized BLAS). SimSIMD doesn't provide speedup for large vectors on ARM.

**Conclusion**: **DO NOT replace np.dot with simsimd.dot** for vector operations on ARM. May be beneficial on x86/AVX platforms.

#### Matrix Multiplication (cv2.gemm)

| Operation | cv2.gemm (ns) | NumPy @ (ns) | **Verdict** |
|-----------|---------------|--------------|-------------|
| Small (10×10) | 25,943 | 3,102 | NumPy **8.4x faster** |
| Medium (100×100) | 3,846 | 3,943 | **Similar** |

**Analysis**:
- `cv2.gemm` is **significantly slower** than NumPy for small matrices
- Already equivalent for medium matrices
- **ThinPlateSpline uses small matrices** (2×N control points)

**Conclusion**: **Replace cv2.gemm with NumPy @** instead of SimSIMD. NumPy already optimal on ARM.

#### Weighted Sum Operations (CRITICAL!)

| Operation | cv2 (ns) | SimSIMD (ns) | NumPy (ns) | **SimSIMD Speedup** |
|-----------|----------|--------------|------------|---------------------|
| RGB (100×100×3) | 248,884 | 68,053 | 398,216 | **3.66x vs cv2** |
| Hyperspectral (100×100×16) | ❌ FAILS | 445,926 | 941,001 | **2.11x vs NumPy** |

**Analysis**:
- SimSIMD is **3.66x faster than cv2.addWeighted** for RGB images
- SimSIMD is **2.11x faster than NumPy** for hyperspectral (C > 4)
- **cv2 cannot handle C > 4 at all** (SimSIMD enables new functionality!)

**Conclusion**: **SimSIMD is a clear win for weighted sum!** Already integrated via albucore.

#### Fused Multiply-Add (FMA)

| Operation | Sequential cv2 (ns) | SimSIMD FMA (ns) | NumPy (ns) | **SimSIMD Speedup** |
|-----------|---------------------|------------------|------------|---------------------|
| alpha×A×B + beta×C | 1,029,563 | 107,248 | 462,296 | **9.6x vs cv2, 4.3x vs NumPy** |

**Analysis**: SimSIMD FMA is **9.6x faster than sequential cv2** operations!

**Conclusion**: **Use simsimd.fma** when performing multiply-add chains.

#### Distance Metrics

| Operation | NumPy (ns) | SimSIMD (ns) | **Speedup** |
|-----------|------------|--------------|-------------|
| Euclidean | 1,701 | 430 | **3.96x**   |
| Cosine    | 1,642 | 457 | **3.59x**   |

**Conclusion**: **Use SimSIMD** for distance calculations (massive speedup).

### Decision Criteria for Updating Code

**Update `cv2.gemm` → NumPy @ (NOT simsimd.dot)**:
- ✅ All correctness tests pass
- ✅ NumPy is **8.4x faster** than cv2.gemm for small matrices
- ✅ NumPy already uses optimized BLAS (Accelerate on ARM)
- ❌ SimSIMD provides no benefit over NumPy on ARM for matrix ops

**Keep using SimSIMD (via albucore) for**:
- ✅ Weighted sum (`add_weighted`): **3.66x faster** than cv2
- ✅ FMA operations: **9.6x faster** than sequential cv2
- ✅ Distance metrics: **3.59-3.96x faster** than NumPy
- ✅ **Enables C > 4 channels** (cv2 limitation bypassed!)

### Integration Action Plan

**✅ Step 1: Run tests** (COMPLETED 2026-02-05)
```bash
cd /Users/vladimiriglovikov/workspace/AlbumentationsX
pytest tests/test_simsimd_integration.py -v         # ✅ 46/46 passed
pytest tests/benchmark_simsimd.py --benchmark-only  # ✅ Completed
```

**✅ Step 2: Analyze results** (COMPLETED)
- SimSIMD provides **3.66x speedup** for weighted sum (vs cv2)
- SimSIMD provides **9.6x speedup** for FMA operations
- SimSIMD provides **3.59-3.96x speedup** for distance metrics
- **NumPy @ is 8.4x faster than cv2.gemm** for small matrices (ThinPlateSpline)
- SimSIMD dot products **not faster than NumPy on ARM** (Accelerate framework)

**🎯 Step 3: Update ThinPlateSpline to use NumPy instead of cv2.gemm**

Replace `cv2.gemm` with NumPy matrix multiplication in ThinPlateSpline:

```python
# In albumentations/augmentations/geometric/functional.py

# OLD (5 uses of cv2.gemm):
mapping = cv2.gemm(kernel_matrix, weights, 1, None, 0)

# NEW (8.4x faster for small matrices):
mapping = kernel_matrix @ weights
```

**Why NumPy @ and not SimSIMD?**
- NumPy already uses Apple Accelerate (optimized BLAS) on ARM
- SimSIMD provides no additional benefit for matrix ops on ARM
- May provide benefit on x86/AVX platforms, but NumPy is universal
- Simpler code (no additional imports)

**Step 4: Validate** (re-run video benchmarks)
```bash
# Re-run UCF101 benchmarks to measure real-world impact
# Expected: ThinPlateSpline 4.51 v/s → possibly higher (8.4x faster matrix ops)
```

**Step 5: Document**
- Update CHANGELOG
- Document performance improvements
- Note: SimSIMD already used via albucore (no new dependency)

### Expected Impact

**ThinPlateSpline** (uses cv2.gemm 5 times):
- Current: 4.51 videos/sec
- After replacing cv2.gemm with NumPy @: **Potentially higher** (matrix ops 8.4x faster)
- Note: Overall speedup depends on proportion of time spent in matrix ops
- Recommendation: Profile to measure actual impact

**Brightness/Contrast** (already uses SimSIMD via albucore):
- ✅ Already optimized!
- **3.66x faster** than cv2 equivalent
- Works with C > 4

**Blending** (already uses SimSIMD via albucore):
- ✅ Already optimized!
- Works with C > 4

**Phase 1: Low-hanging fruit** (alongside cv2 SIMD)
- SIMD `exp` for Gaussian operations
- Test with rotation and distortion transforms

**Phase 3: Full coverage** (long-term)
- All element-wise operations
- SimSIMD for matrix ops
- Optimized reductions

## Action Items Checklist

### ✅ SimSIMD Integration Testing (COMPLETED 2026-02-05)
- [x] Created `tests/test_simsimd_integration.py` - 46 correctness tests
- [x] Created `tests/benchmark_simsimd.py` - 20 performance benchmarks
- [x] All tests pass (46/46)
- [x] Benchmarked on ARM (M-series macOS)
- [x] Analyzed results and documented findings
- [x] **Key finding**: NumPy @ is 8.4x faster than cv2.gemm for small matrices
- [x] **Key finding**: SimSIMD already used for weighted sum (3.66x faster than cv2)
- [x] **Key finding**: SimSIMD enables C > 4 channel support

### 🎯 Next: Replace cv2.gemm with NumPy @
- [ ] Update ThinPlateSpline to use `kernel_matrix @ weights` instead of `cv2.gemm`
- [ ] Run existing ThinPlateSpline tests to verify correctness
- [ ] Profile real-world impact on video benchmarks
- [ ] Create PR with benchmark results

### Before First Meeting with Ash
- [ ] Review this document internally
- [ ] Identify any missing operations or use cases
- [ ] Prepare list of priority questions
- [ ] Gather performance metrics from current users (where is it slow?)
- [ ] Check existing issues for multi-channel/video/3D requests

### During First Meeting
- [ ] Walk through document together
- [ ] Agree on Tier 1 operations priority
- [ ] Discuss library structure (albucore vs new library)
- [ ] Define success criteria for Phase 1 prototype
- [ ] Establish communication channels

### Phase 1: Prototype (Target: 4 weeks)
- [ ] Set up benchmarking infrastructure
- [ ] **Option A (RECOMMENDED): SIMD RNG Prototype**
  - [ ] Implement SIMD uniform() and normal()
  - [ ] Benchmark: elastic transform displacement generation (512x512)
  - [ ] Validate statistical quality
  - [ ] Integrate into albucore
  - [ ] Target: 3-5x speedup
- [ ] **Option B: Image Operation Prototype**
  - [ ] Baseline measurements for Tier 1 operations (cv2 + NumPy fallback)
  - [ ] Implement prototype for 1 operation (warpAffine or resize)
  - [ ] Test with arbitrary channels (C > 4)
  - [ ] Validate correctness
  - [ ] Target: 2x+ speedup for C > 4
- [ ] Ash + Team: Decide which option to pursue
- [ ] Document lessons learned

### Phase 1 Success Criteria
- [ ] Prototype implementation for at least 1 operation (RNG OR image op)
- [ ] For RNG: 3x+ speedup, passes TestU01 SmallCrush
- [ ] For image op: Supports arbitrary channels (C > 4), 2x+ speedup vs NumPy fallback
- [ ] Passes all existing tests (or new tests for RNG)
- [ ] Clear path to scaling to more operations

---

**Total cv2 operations analyzed**: 110 (95 direct + 15 via albucore)
**Critical bottlenecks identified**: 3 (warpAffine, resize, remap)
**Use cases blocked**: 5 (hyperspectral, medical, video, 3D, microscopy)
**Expected impact**: 2-10x speedup, unlocking entire new domains
