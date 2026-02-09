# Hypothesis Property-Based Testing Integration

## Summary

Added property-based testing using Hypothesis to existing tests. Unlike the redundant standalone file, these tests **enhance** existing tests by verifying mathematical properties across a much wider input space.

## Key Principle

**Property-based tests complement, not duplicate, existing tests:**
- Existing tests: Verify specific expected outputs for known inputs
- Property tests: Verify mathematical properties hold for ANY valid input

## Tests Enhanced

### 1. ReplayCompose Determinism (`tests/test_core.py`)

**Old approach:** Loop 10 times with `np.random.random` generated images
```python
for _ in range(10):
    image = (np.random.random((8, 8)) * 255).astype(np.uint8)
    # test determinism
```

**New approach:** Hypothesis generates diverse images automatically
```python
@given(npst.arrays(dtype=np.uint8, shape=(8, 8, 3), elements=st.integers(0, 255)))
@settings(max_examples=20, deadline=2000)
def property_test(image):
    # test determinism
```

**Affected tests:**
- `test_deterministic_oneof()` - Tests OneOf with ReplayCompose
- `test_deterministic_one_or_other()` - Tests OneOrOther with ReplayCompose
- `test_deterministic_sequential()` - Tests Sequential with ReplayCompose

**Benefit:** Tests 20 diverse images instead of 10 similar ones, with better edge case coverage.

---

### 2. Idempotence Tests (`tests/test_core.py`)

**Old approach:** Test fixed, handcrafted bboxes/keypoints
```python
@pytest.mark.parametrize("bboxes", [[[0.2, 0.3, 0.4, 0.5], ...]])
def test_bbox_hflip_hflip_no_labels(bbox_format, bboxes):
    # test HorizontalFlip(HorizontalFlip(x)) = x
```

**New approach:** Hypothesis generates random valid coordinates
```python
def test_bbox_hflip_idempotence_property():
    @given(st.lists(st.tuples(
        st.floats(0.0, 0.7),  # x_min
        st.floats(0.0, 0.7),  # y_min
        st.floats(0.3, 1.0),  # x_max
        st.floats(0.3, 1.0),  # y_max
    ).filter(lambda x: x[2] > x[0] + 0.01 and x[3] > x[1] + 0.01), ...))
```

**New tests added:**
- `test_bbox_hflip_idempotence_property()` - Tests with random valid bboxes
- `test_keypoint_hflip_idempotence_property()` - Tests with random valid keypoints

**Benefit:** Discovers edge cases like near-boundary coordinates, finds potential filtering issues.

---

### 3. Value Range Preservation (`tests/functional/test_functional.py`)

**Old approach:** Test 4 hardcoded threshold values
```python
@pytest.mark.parametrize("threshold", [0.0, 1/3, 2/3, 1.0])
def test_solarize(image, threshold):
    # verify value range preserved
```

**New approach:** Hypothesis tests ANY threshold, shape, dtype
```python
def test_solarize_value_range_property():
    @given(
        dtype=st.sampled_from([np.uint8, np.float32]),
        shape=st.tuples(st.integers(10, 100), ...),
        threshold=st.floats(0.0, 1.0)
    )
    def property_test(dtype, shape, threshold):
        # verify value range preserved for ANY valid input
```

**New test added:**
- `test_solarize_value_range_property()` - Tests 50 random combinations

**Benefit:** Tests the mathematical property (value range preservation) across entire input space, not just 4 points.

---

### 4. Deterministic Seeding (`tests/test_per_worker_seed.py`)

**Old approach:** Test with fixed image and seed
```python
def test_deterministic_behavior_single_process():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # test with seed=137 only
```

**New approach:** Test with random images and seeds
```python
def test_deterministic_behavior_property():
    @given(
        image=npst.arrays(...),
        seed=st.integers(0, 10000)
    )
    def property_test(image, seed):
        # test same seed → same result for ANY image and seed
```

**New test added:**
- `test_deterministic_behavior_property()` - Tests 30 random (image, seed) pairs

**Benefit:** Verifies determinism is a property of the seeding mechanism, not just for one specific case.

---

## Why This Approach Works

### What Makes Good Property Tests

✅ **Do use Hypothesis for:**
1. **Mathematical properties:** idempotence, determinism, value ranges, shape preservation
2. **Edge case discovery:** Hypothesis automatically finds boundary conditions
3. **Expanding manual tests:** Test the same property across wider input space

❌ **Don't use Hypothesis for:**
1. **Duplicating combinatorial testing:** Your `tests/utils.py` already does this systematically
2. **Testing exact outputs:** Use parametrize for known input/output pairs
3. **Coverage for coverage's sake:** Only test properties that should universally hold

### Example: Why Hypothesis Found Real Issues

The keypoint idempotence test revealed:
```python
# Hypothesis discovered that coordinates at exactly 100.0 are invalid
ValueError: Expected y for keypoint [  0. 100.   0.   0.   0.] to be in range [0, 100), got 100.0

# And that duplicate keypoints may be filtered:
# Input:  [(0.0, 0.0), (99.5, 0.0)]
# Output: [(0.0, 0.0)]  # One filtered!
```

This is **valuable** because it tests real library behavior, not made-up edge cases.

## Running Property Tests

```bash
# Run all property tests
pytest -xvs -k "property"

# Run with random order (pytest-randomly)
pytest --randomly-seed=137 -k "property"

# Run specific property test
pytest tests/test_core.py::test_bbox_hflip_idempotence_property -xvs
```

## Guidelines for Future Property Tests

When adding Hypothesis tests to **existing** tests:

1. **Identify the mathematical property being tested**
   - Example: "HorizontalFlip applied twice returns original image"

2. **Check if existing test uses limited examples**
   - Example: `for _ in range(10)` or `@parametrize` with 3-4 values

3. **Convert to property test that generates diverse inputs**
   - Replace loops with `@given`
   - Replace fixed values with strategies

4. **Keep original parametrized test for known cases**
   - Property tests supplement, don't replace, specific test cases

## Statistics

- **7 new property tests** added to existing test files
- **0 standalone files** created
- **~140 diverse test cases** generated per full run (20-50 examples each)
- **100% integration** with existing test infrastructure
