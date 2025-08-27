# Claude Code Review Guidelines for AlbumentationsX

## Project Overview

AlbumentationsX is a high-performance computer vision augmentation library. We prioritize performance, type safety, and consistency.

## Critical Review Points

### Type Hints

- **MUST** use Python 3.10+ typing: `list` not `List`, `tuple` not `Tuple`, `| None` not `Optional`
- All functions must have complete type hints
- Use `np.ndarray` with proper shape annotations where possible

### Transform Standards

- **NO** "Random" prefix in new transform names
- Parameter ranges use `_range` suffix (e.g., `brightness_range` not `brightness_limit`)
- Use `fill` not `fill_value`, `fill_mask` not `fill_mask_value`
- InitSchema classes must NOT have default values (except discriminator fields)
- Default test values should be 137, not 42

### Code Patterns

```python
# CORRECT
def __init__(self, brightness_range: tuple[float, float] = (-0.2, 0.2)):
    self.brightness_range = brightness_range

# INCORRECT
def __init__(self, brightness: float | tuple[float, float] = 0.2):
    self.brightness = brightness
```

### Performance Requirements (Priority Order)

1. **cv2.LUT for lookup operations** - fastest for pixel-wise transformations
2. **cv2 operations over numpy** - generally faster for image processing
3. **Vectorized numpy over loops** - eliminate Python loops where possible
4. **In-place operations** - reduce memory allocations and copies
5. **Cache computations** in `get_params_dependent_on_data`
6. Apply decorators `@uint8_io` or `@float32_io` for type consistency

### Random Number Generation

- Use `self.py_random` for simple random operations
- Use `self.random_generator` only when numpy arrays are needed
- NEVER use `np.random` or `random` directly

### Testing

- All new transforms need comprehensive tests
- Use `pytest.mark.parametrize` for parameterized tests
- Test edge cases and different data types (uint8, float32)
- Test with various number of channels

### Documentation

Every transform MUST have an Examples section in docstring:

```python
"""
Examples:
    >>> import numpy as np
    >>> import albumentations as A
    >>> # Prepare sample data
    >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> # Define transform
    >>> transform = A.Compose([
    ...     A.YourTransform(param_range=(0.1, 0.3), p=1.0)
    ... ])
    >>> # Apply and get results
    >>> transformed = transform(image=image)
    >>> transformed_image = transformed['image']
"""
```

## Common Issues to Flag

### Performance Anti-patterns

- Not using cv2.LUT for lookup-based transformations
- Using numpy when cv2 equivalent exists and is faster
- Using Python loops instead of vectorized numpy operations
- Creating unnecessary array copies instead of in-place operations
- Repeated array allocations in tight loops

### Memory Issues

- Large temporary arrays that could be avoided
- Not using in-place operations where safe
- Memory leaks in cv2 operations

### Type Safety

- Missing type hints
- Using old typing syntax
- Incorrect numpy dtype handling
- Unsafe type conversions

### API Consistency

- Parameters not following naming conventions
- Missing InitSchema validation
- Inconsistent with similar transforms
- Not supporting arbitrary channels when possible

## Review Priority

1. **Correctness**: Mathematical/logical errors
2. **Performance**: Bottlenecks and inefficiencies
3. **Type Safety**: Proper typing and validation
4. **API Design**: Consistency with library patterns
5. **Documentation**: Clear examples and explanations

## Benchmarking

For performance-critical changes, suggest benchmarking:

```python
# Simple timing comparison
import timeit
old_time = timeit.timeit(lambda: old_implementation(data), number=1000)
new_time = timeit.timeit(lambda: new_implementation(data), number=1000)
```

## Do NOT Suggest

- Creating temporary test files
- Adding documentation unless explicitly missing
- Style changes that don't affect performance
- Renaming existing transforms (backward compatibility)
