# Property-Based Testing Analysis & Test Randomization

## Executive Summary

After analyzing the existing test suite, here's what would actually add value without duplicating your excellent test infrastructure.

## 1. Test Randomization (Like Jest)

**Solution: `pytest-randomly`**

```bash
pip install pytest-randomly
```

This plugin:
- Randomizes test order on each run
- Seeds are printed so you can reproduce failures
- Catches hidden test dependencies
- Works with your existing `pytest-xdist` for parallel execution

**Add to `requirements-dev.txt`:**
```
pytest-randomly>=3.15.0
```

**Usage:**
```bash
# Random order
pytest

# Reproduce specific run
pytest --randomly-seed=1234

# Disable randomization temporarily
pytest -p no:randomly
```

---

## 2. Tests That Would Benefit from Hypothesis

After reviewing your codebase, these existing test patterns could be **enhanced** (not replaced) with property-based testing:

### A. Idempotence Tests (Apply Twice = Original)

**Current:** `tests/test_core.py` has manual tests for HorizontalFlip twice

**Hypothesis Enhancement:**
```python
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst

@given(
    image=npst.arrays(
        dtype=np.uint8,
        shape=npst.array_shapes(min_dims=3, max_dims=3,
                               min_side=1, max_side=100)
    )
)
def test_horizontal_flip_idempotence(image):
    """Property: flip twice returns original."""
    transform = A.HorizontalFlip(p=1.0)
    result1 = transform(image=image)["image"]
    result2 = transform(image=result1)["image"]
    np.testing.assert_array_equal(result2, image)
```

**Transforms to test:** HorizontalFlip, VerticalFlip, Transpose

---

### B. Rotation by 360° = Identity

**Current:** No systematic test

**Hypothesis Enhancement:**
```python
@given(
    image=npst.arrays(dtype=np.uint8, shape=(50, 50, 3)),
    angle=st.integers(0, 359)
)
def test_rotate_360_is_identity(image, angle):
    """Property: rotate(α) then rotate(360-α) ≈ identity."""
    transform1 = A.Rotate(limit=(angle, angle), p=1.0, border_mode=0)
    transform2 = A.Rotate(limit=(360-angle, 360-angle), p=1.0, border_mode=0)

    result = transform2(image=transform1(image=image)["image"])["image"]
    # Allow small numerical errors from interpolation
    assert np.allclose(result, image, atol=5)
```

---

### C. Determinism Tests (Same Seed = Same Output)

**Current:** Manual tests exist but not systematic

**Hypothesis Enhancement:**
```python
@given(
    image=npst.arrays(dtype=np.uint8, shape=(50, 50, 3)),
    seed=st.integers(0, 2**31-1)
)
@pytest.mark.parametrize("transform_cls", [
    A.RandomBrightnessContrast,
    A.RandomCrop,
    A.GaussNoise,
])
def test_determinism_with_seed(image, seed, transform_cls):
    """Property: same seed produces same output."""
    transform = transform_cls(p=1.0)

    transform.set_random_seed(seed)
    result1 = transform(image=image.copy())

    transform.set_random_seed(seed)
    result2 = transform(image=image.copy())

    np.testing.assert_array_equal(result1["image"], result2["image"])
```

---

### D. Value Range Preservation

**Current:** Some ad-hoc tests exist

**Hypothesis Enhancement:**
```python
@given(
    image=npst.arrays(
        dtype=st.sampled_from([np.uint8, np.float32]),
        shape=(50, 50, 3),
        elements=st.just(0) | st.just(255)  # Edge values
    )
)
@pytest.mark.parametrize("transform_cls", [
    A.HorizontalFlip,
    A.VerticalFlip,
    A.Transpose,
])
def test_value_range_preserved(image, transform_cls):
    """Property: transforms preserve value ranges."""
    if image.dtype == np.uint8:
        image = image.astype(np.uint8)  # Ensure type
    else:
        image = (image / 255.0).astype(np.float32)

    transform = transform_cls(p=1.0)
    result = transform(image=image)["image"]

    if image.dtype == np.uint8:
        assert result.min() >= 0
        assert result.max() <= 255
    else:
        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6
```

---

### E. Bbox Coordinate Bounds

**Current:** Tests exist but not exhaustive

**Hypothesis Enhancement:**
```python
@given(
    bboxes=st.lists(
        st.tuples(
            st.floats(0, 0.7),  # x_min
            st.floats(0, 0.7),  # y_min
            st.floats(0.3, 1.0),  # x_max
            st.floats(0.3, 1.0),  # y_max
        ).filter(lambda x: x[2] > x[0] and x[3] > x[1]),
        min_size=1,
        max_size=5
    )
)
def test_flip_preserves_bbox_bounds(bboxes):
    """Property: flips keep bboxes in [0, 1] range."""
    bboxes_arr = np.array(bboxes, dtype=np.float32)
    labels = list(range(len(bboxes)))

    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(
        coord_format="albumentations",
        label_fields=["labels"]
    ))

    result = transform(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        bboxes=bboxes_arr,
        labels=labels
    )

    result_bboxes = result["bboxes"]
    assert np.all(result_bboxes >= 0)
    assert np.all(result_bboxes <= 1)
```

---

## 3. Recommended Approach

**DON'T:** Replace existing tests with Hypothesis
**DO:** Add Hypothesis tests to `tests/test_property_based.py` that test **mathematical properties**

### What to Keep in `test_property_based.py`:

1. **Idempotence tests** - Flips applied twice
2. **Determinism tests** - Same seed = same output
3. **Value range tests** - uint8 [0,255], float32 [0,1]
4. **Coordinate bounds** - Bboxes/keypoints stay valid

### What to DELETE from `test_property_based.py`:

The redundant stuff:
- `TestShapeConstraints` - Your existing tests already check this via parametrization
- `TestEdgeCases` - Already covered by your `get_transforms()` infrastructure

---

## 4. Updated Implementation Plan

```python
# tests/test_property_based.py - STREAMLINED VERSION

"""Property-based tests for mathematical properties.

These tests check properties that should hold for ALL valid inputs,
complementing the existing parametrized tests.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst

import albumentations as A


# Strategy for valid images
@st.composite
def images(draw, min_size=2, max_size=50):
    dtype = draw(st.sampled_from([np.uint8, np.float32]))
    shape = draw(npst.array_shapes(min_dims=3, max_dims=3,
                                   min_side=min_size, max_side=max_size))

    if dtype == np.uint8:
        return draw(npst.arrays(dtype=dtype, shape=shape,
                               elements=st.integers(0, 255)))
    else:
        return draw(npst.arrays(dtype=dtype, shape=shape,
                               elements=st.floats(0.0, 1.0,
                                                 allow_nan=False,
                                                 allow_infinity=False)))


class TestIdempotence:
    """Test that certain operations applied twice return to original."""

    @given(images())
    @settings(max_examples=20, deadline=2000)
    def test_horizontal_flip_twice(self, image):
        transform = A.HorizontalFlip(p=1.0)
        result = transform(image=transform(image=image)["image"])["image"]
        np.testing.assert_array_equal(result, image)

    @given(images())
    @settings(max_examples=20, deadline=2000)
    def test_vertical_flip_twice(self, image):
        transform = A.VerticalFlip(p=1.0)
        result = transform(image=transform(image=image)["image"])["image"]
        np.testing.assert_array_equal(result, image)


class TestDeterminism:
    """Test that same seed produces same output."""

    @given(images(), st.integers(0, 1000))
    @settings(max_examples=15, deadline=2000)
    @pytest.mark.parametrize("transform_cls", [
        A.RandomBrightnessContrast,
        A.GaussNoise,
        A.RandomGamma,
    ])
    def test_random_transforms_deterministic(self, image, seed, transform_cls):
        transform = transform_cls(p=1.0)

        transform.set_random_seed(seed)
        result1 = transform(image=image.copy())["image"]

        transform.set_random_seed(seed)
        result2 = transform(image=image.copy())["image"]

        np.testing.assert_array_equal(result1, result2)


class TestValueRanges:
    """Test that transforms maintain valid value ranges."""

    @given(images())
    @settings(max_examples=20, deadline=2000)
    @pytest.mark.parametrize("transform_cls", [
        A.HorizontalFlip,
        A.VerticalFlip,
        A.Transpose,
        A.RandomRotate90,
    ])
    def test_geometric_preserve_ranges(self, image, transform_cls):
        transform = transform_cls(p=1.0)
        result = transform(image=image)["image"]

        if image.dtype == np.uint8:
            assert result.min() >= 0
            assert result.max() <= 255
        else:
            assert result.min() >= -1e-6
            assert result.max() <= 1.0 + 1e-6
```

---

## 5. Summary

**Delete redundant files:** ✅ Done
- `test_dtype_multichannel.py`
- `test_minimal_transform_coverage.py`
- `test_helpers.py`

**Keep and streamline:**
- `test_property_based.py` - Focus on mathematical properties only

**Add test randomization:**
```bash
pip install pytest-randomly
```

**The Real Value:**
1. ✅ Hypothesis finds edge cases automatically
2. ✅ `pytest-randomly` catches test dependencies
3. ✅ Your existing `get_transforms()` infrastructure already handles systematic testing
4. ✅ No duplication - property tests check different things than parametrized tests

The key insight: **Property-based tests verify mathematical properties (idempotence, commutativity, etc.), while your existing tests verify correctness for specific parameter combinations.** They're complementary, not redundant.
