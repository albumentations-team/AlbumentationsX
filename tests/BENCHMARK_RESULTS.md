# Array Creation Benchmark Results

## Executive Summary

Benchmark comparing `cv2.randu()` vs numpy random methods for test fixture creation.

**Key Findings:**
- **UINT8 arrays**: `np.random.default_rng().integers()` is **2-3x FASTER** than `cv2.randu()`
- **FLOAT32 arrays**: `cv2.randu()` is **~2x FASTER** than numpy methods
- **cv2.randu() limitation**: Only supports ≤4 channels (fails for multi-channel arrays)

## Detailed Results

### Shape: (100, 100, 3) - Small Arrays

**UINT8:**
- cv2.randu: 0.04ms per call
- np.random.rng: 0.02ms per call ✓ **2.85x faster**

**FLOAT32:**
- cv2.randu: 0.04ms per call ✓ **2.1x faster**
- np.random.uniform: 0.08ms per call

### Shape: (512, 512, 3) - Medium Arrays

**UINT8:**
- cv2.randu: 1.15ms per call
- np.random.rng: 0.36ms per call ✓ **3.18x faster**

**FLOAT32:**
- cv2.randu: 1.17ms per call ✓ **2.08x faster**
- np.random.uniform: 2.43ms per call

### Shape: (1000, 1000, 3) - Large Arrays

**UINT8:**
- cv2.randu: 4.53ms per call
- np.random.rng: 1.50ms per call ✓ **3.03x faster**

**FLOAT32:**
- cv2.randu: 4.04ms per call ✓ **1.95x faster**
- np.random.uniform: 7.88ms per call

### Shape: (512, 512, 16) - Multi-Channel Arrays

**UINT8:**
- cv2.randu: N/A (doesn't support >4 channels)
- np.random.rng: 2.11ms per call ✓ **Only option**

**FLOAT32:**
- cv2.randu: N/A (doesn't support >4 channels)
- np.random.uniform: 11.14ms per call ✓ **Only option**

## Summary Statistics

| Metric | cv2.randu | numpy |
|--------|-----------|-------|
| UINT8 arrays (≤4 ch) | 0/3 wins | 3/3 wins |
| FLOAT32 arrays (≤4 ch) | 3/3 wins | 0/3 wins |
| Multi-channel (>4 ch) | Not supported | Only option |

## Recommendations for Test Fixtures

### For conftest.py

1. **UINT8 fixtures**: Switch to `np.random.default_rng().integers()`
   ```python
   rng = np.random.default_rng(137)
   SQUARE_UINT8_IMAGE = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
   ```
   **Benefit**: 2-3x faster array creation

2. **FLOAT32 fixtures**: Keep `cv2.randu()`
   ```python
   SQUARE_FLOAT_IMAGE = cv2.randu(np.empty((100, 100, 3), dtype=np.float32), 0, 1)
   ```
   **Benefit**: 2x faster than numpy

3. **Multi-channel arrays**: Use numpy (only option)
   ```python
   SQUARE_MULTI_UINT8_IMAGE = rng.integers(0, 256, (100, 100, 5), dtype=np.uint8)
   ```

### General Strategy

**MOST IMPORTANT**: Use **module-scoped** or **session-scoped** fixtures to avoid recreation:

```python
@pytest.fixture(scope="module")
def large_image():
    rng = np.random.default_rng(137)
    return rng.integers(0, 256, (1000, 1000, 3), dtype=np.uint8)
```

This eliminates creation cost entirely—much more important than the 2-3x difference between methods!

### Consistency Consideration

For **consistency across all dtypes and channels**, consider using numpy exclusively:
- Simpler mental model
- Works for all channel counts
- Only 2x slower for float32 (negligible with proper fixture scoping)
- Already 2-3x faster for uint8 (majority of test cases)

## Implementation Impact

When implementing test improvements:

1. **Phase 1 priority**: Add module/session-scoped fixtures (biggest impact)
2. **Phase 2 optimization**: Switch uint8 creation to numpy default_rng
3. **Phase 3 cleanup**: Keep cv2.randu for float32 or standardize on numpy

The choice between cv2 and numpy is secondary to proper fixture scoping!
