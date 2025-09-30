# Claude Code Review Guidelines for AlbumentationsX

> **Note**: This is a quick reference guide. For comprehensive details, see:
>
> - `docs/contributing/coding_guidelines.md` - Complete coding standards and best practices
> - `.github/workflows/claude-pr-review.yml` - Automated PR review criteria and checks

## Project Overview

AlbumentationsX is a high-performance computer vision augmentation library. We prioritize performance, type safety, consistency, and clean code.

## Critical Quick Checks

### Type Hints

- **MUST** use Python 3.10+ typing: `list` not `List`, `tuple` not `Tuple`, `| None` not `Optional`
- All functions must have complete type hints
- Use `np.ndarray` with proper shape annotations where possible

### Transform Standards

- **NO** "Random" prefix in new transform names
- Parameter ranges use `_range` suffix (e.g., `brightness_range` not `brightness_limit`)
- Use `fill` not `fill_value`, `fill_mask` not `fill_mask_value`
- Use `border_mode` not `mode` or `pad_mode`
- InitSchema classes must NOT have default values (except discriminator fields for Pydantic unions)
- Default test values should be 137, not 42
- Prefer relative parameters (fractions of image size) over fixed pixel values

### Dead Code Detection

- **Flag as critical**: Unused methods, classes that are never called
- **Flag as important**: Unused imports, variables
- Dead code wastes maintenance effort and confuses developers

### Code Patterns

```python
# CORRECT - Always use ranges
def __init__(self, brightness_range: tuple[float, float] = (-0.2, 0.2)):
    self.brightness_range = brightness_range

# INCORRECT - Don't use Union types for parameters
def __init__(self, brightness: float | tuple[float, float] = 0.2):
    self.brightness = brightness
```

### Performance Requirements (Priority Order)

1. **cv2.LUT for lookup operations** - fastest for pixel-wise transformations
2. **cv2 operations over numpy** - generally faster for image processing
3. **Vectorized numpy over loops** - eliminate Python loops where possible
4. **In-place operations** - reduce memory allocations and unnecessary copies
5. **Cache computations** in `get_params` or `get_params_dependent_on_data`
6. **Remove dead code** - unused code impacts performance and maintainability
7. Apply decorators `@uint8_io` or `@float32_io` for type consistency

### Random Number Generation

- Use `self.py_random` for simple random operations (faster)
- Use `self.random_generator` only when numpy arrays are needed
- **NEVER** use `np.random` or `random` module directly
- All random operations in `get_params` or `get_params_dependent_on_data`, NOT in `apply_xxx` methods

### Testing

- All new transforms need comprehensive tests
- Use `pytest.mark.parametrize` for parameterized tests
- Test edge cases and different data types (uint8, float32)
- Test with various number of channels
- Use `np.testing` functions instead of plain `assert`
- **NEVER** create temporary test files - add permanent tests to test suite

### Documentation Requirements

Every transform MUST have a comprehensive Examples section in docstring:

```python
"""
Examples:
    >>> import numpy as np
    >>> import albumentations as A
    >>> # Prepare sample data
    >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    >>> bboxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    >>> bbox_labels = [1]
    >>> keypoints = np.array([[20, 30]], dtype=np.float32)
    >>> keypoint_labels = [0]
    >>>
    >>> # Define transform (use tuples for ranges)
    >>> transform = A.Compose([
    ...     A.YourTransform(param_range=(0.1, 0.3), p=1.0)
    ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
    ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
    >>>
    >>> # Apply and get all results
    >>> transformed = transform(
    ...     image=image,
    ...     mask=mask,
    ...     bboxes=bboxes,
    ...     bbox_labels=bbox_labels,
    ...     keypoints=keypoints,
    ...     keypoint_labels=keypoint_labels
    ... )
    >>> transformed_image = transformed['image']
    >>> transformed_mask = transformed['mask']
"""
```

## Common Issues to Flag

### Performance Anti-patterns

- Not using cv2.LUT for lookup-based transformations
- Using numpy when cv2 equivalent exists and is faster
- Using Python loops instead of vectorized numpy operations
- Creating unnecessary array copies instead of in-place operations
- Repeated array allocations in tight loops
- Dead code and unused imports

### Memory Issues

- Large temporary arrays that could be avoided
- Not using in-place operations where safe
- Unnecessary array copies (check for `.copy()` that can be eliminated)

### Type Safety

- Missing type hints
- Using old typing syntax (`List`, `Tuple`, `Optional`, `Union`)
- Incorrect numpy dtype handling
- Unsafe type conversions

### API Consistency

- Parameters not following naming conventions
- Missing InitSchema validation
- Inconsistent with similar transforms
- Not supporting arbitrary channels when possible
- Fixed pixel values instead of relative parameters

### Code Quality

- Dead code (unused methods, classes, imports)
- Default values in InitSchema classes
- Default arguments in `apply_xxx` methods
- Using wrong center calculation (`center()` vs `center_bbox()`)

## Review Priority

1. **Correctness**: Mathematical/logical errors, bugs
2. **Security**: Potential security vulnerabilities
3. **Dead Code**: Unused methods, classes (critical to remove)
4. **Performance**: Bottlenecks and inefficiencies
5. **Type Safety**: Proper typing and validation
6. **API Design**: Consistency with library patterns
7. **Documentation**: Clear examples and explanations
8. **Code Quality**: Unused imports, style improvements

## Severity Classification

When reporting issues, use these severity levels:

- 🔴 **Critical**: Must fix (bugs, security, dead code, memory leaks)
- 🟡 **Important**: Should fix (performance, code quality, unused imports)
- 🟢 **Suggestion**: Nice to have (style, minor optimizations)

## Benchmarking Suggestions

For performance-critical changes, suggest benchmarking:

```python
# Simple timing comparison
import timeit
import numpy as np

img = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)

old_time = timeit.timeit(lambda: old_implementation(img), number=100)
new_time = timeit.timeit(lambda: new_implementation(img), number=100)

print(f"Old: {old_time:.4f}s, New: {new_time:.4f}s")
print(f"Speedup: {old_time/new_time:.2f}x")
```

## What NOT to Suggest

- Creating temporary test files (add permanent tests instead)
- Renaming existing transforms (breaks backward compatibility)
- Changing existing parameter names (breaks backward compatibility)
- Using capital-case types (`List`, `Tuple`, `Optional`, `Union`)
