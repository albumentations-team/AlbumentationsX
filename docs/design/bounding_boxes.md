# Bounding Box Processing in Albumentations

This document describes how Albumentations handles bounding boxes, including both Horizontal Bounding Boxes (HBB) and Oriented Bounding Boxes (OBB), throughout the augmentation pipeline.

## Table of Contents

1. [Bounding Box Types](#bounding-box-types)
2. [Coordinate Formats](#coordinate-formats)
3. [The Albumentations Internal Format](#the-albumentations-internal-format)
4. [BboxParams Configuration](#bboxparams-configuration)
5. [Processing Pipeline](#processing-pipeline)
6. [Input Clipping vs Transform Clipping](#input-clipping-vs-transform-clipping)
7. [Filtering Parameters](#filtering-parameters)
8. [Transform-Level Behavior](#transform-level-behavior)
9. [Best Practices](#best-practices)

---

## Bounding Box Types

Albumentations supports two types of bounding boxes:

### Horizontal Bounding Boxes (HBB)
- **Format**: `[x_min, y_min, x_max, y_max, ...]`
- Axis-aligned rectangles
- Defined by top-left and bottom-right corners
- Cannot represent rotated objects accurately
- Faster to process

### Oriented Bounding Boxes (OBB)
- **Format**: `[x_min, y_min, x_max, y_max, angle, ...]`
- Rotated rectangles
- First 4 values define the axis-aligned bounding rectangle
- `angle` (5th value) defines rotation in degrees
- Can represent rotated objects accurately
- Angle is normalized to [-180, 180) range

**Important**: Both formats can have additional columns for labels, class IDs, track IDs, etc.

---

## Coordinate Formats

Albumentations supports multiple input formats:

### 1. Pascal VOC (pixel coordinates)
```python
[x_min, y_min, x_max, y_max]  # absolute pixel values
# Example: [10, 20, 100, 150] on 200x300 image
```

### 2. COCO (pixel coordinates)
```python
[x_min, y_min, width, height]  # absolute pixel values
# Example: [10, 20, 90, 130] on 200x300 image
```

### 3. YOLO (normalized coordinates)
```python
[x_center, y_center, width, height]  # normalized to [0, 1]
# Example: [0.3, 0.283, 0.45, 0.433] (relative to image size)
```

### 4. Albumentations (normalized coordinates)
```python
[x_min, y_min, x_max, y_max]  # normalized to [0, 1]
# Example: [0.05, 0.1, 0.5, 0.75] (relative to image size)
```

**For OBB**: Add angle as 5th element to any format:
```python
[x_min, y_min, x_max, y_max, angle]  # angle in degrees
```

---

## The Albumentations Internal Format

All bounding boxes are converted to **normalized Albumentations format** internally:
- `[x_min, y_min, x_max, y_max, ...]` for HBB
- `[x_min, y_min, x_max, y_max, angle, ...]` for OBB
- Coordinates normalized to [0, 1] range
- Additional columns (labels, etc.) preserved unchanged

**Key Point**: The internal format uses the axis-aligned bounding rectangle even for OBB. The rotation angle is stored separately as the 5th column.

---

## BboxParams Configuration

`BboxParams` controls how bounding boxes are processed throughout the pipeline.

### Basic Configuration

```python
from albumentations import BboxParams

bbox_params = BboxParams(
    bbox_format='pascal_voc',      # Input format
    bbox_type='hbb',               # 'hbb' or 'obb'
    label_fields=['class_labels'], # Names of label arrays
)
```

### Complete Parameter Reference

#### `bbox_format: Literal['pascal_voc', 'coco', 'yolo', 'albumentations']`
Input coordinate format: `'pascal_voc'`, `'coco'`, `'yolo'`, or `'albumentations'`

**Deprecated**: The `format` parameter is deprecated. Use `bbox_format` instead to avoid shadowing the built-in `format()` function.

#### `bbox_type: Literal['hbb', 'obb']`
Type of bounding box:
- `'hbb'`: Horizontal/axis-aligned bounding boxes (default)
- `'obb'`: Oriented bounding boxes (with rotation angle)

**Critical**: Never auto-detect bbox_type from the number of columns! Users may attach additional label fields.

#### `label_fields: list[str] | None = None`
Names of additional arrays that correspond to bboxes (e.g., class labels, track IDs):
```python
BboxParams(bbox_format='yolo', label_fields=['class_ids', 'track_ids'])

# Usage:
transform(
    image=image,
    bboxes=bboxes,
    class_ids=[1, 2, 3],    # Must match length of bboxes
    track_ids=[10, 11, 12]  # Must match length of bboxes
)
```

#### `min_area: float = 0.0`
Minimum area (in pixels²) for a bbox to be kept after transformation.
```python
BboxParams(min_area=100.0)  # Remove boxes smaller than 100px²
```

#### `min_visibility: float = 0.0`
Minimum visible area ratio [0.0, 1.0] after transformation:
```python
BboxParams(min_visibility=0.3)  # Remove boxes with <30% visible area
```

#### `min_width: float = 0.0` and `min_height: float = 0.0`
Minimum dimensions (in pixels) after transformation:
```python
BboxParams(min_width=10.0, min_height=10.0)  # Remove tiny boxes
```

#### `max_accept_ratio: float | None = None`
Maximum aspect ratio (width/height or height/width) to accept:
```python
BboxParams(max_accept_ratio=3.0)  # Remove boxes with aspect ratio > 3:1
```

#### `check_each_transform: bool = True`
If `True`, validates bbox compatibility with each transform at pipeline creation.

#### `clip_bboxes_on_input: bool = False`
If `True`, clips bboxes to image boundaries **once at pipeline start** (before any transforms):
```python
BboxParams(clip_bboxes_on_input=True)  # Fix invalid input coordinates (e.g., YOLO -1e-6)
```
- Runs during `preprocess()` only
- Fixes malformed input bboxes
- Independent of `clip_after_transform`

**Deprecated**: The `clip` parameter is deprecated. Use `clip_bboxes_on_input` instead for clarity.

#### `clip_after_transform: Literal[None, 'geometry'] = 'geometry'`
Controls how bboxes are clipped **after each transform** in the pipeline:

- `None`: No clipping after transforms. Boxes may temporarily go outside [0, 1] bounds.
  - Use for lenient pipelines (e.g., crop then pad)
  - Boxes can have negative coords or coords > 1

- `'geometry'`: Clip based on actual geometry (default)
  - **For HBB**: Clips `(x_min, y_min, x_max, y_max)` to [0, 1]. Fast, current behavior.
  - **For OBB**: Clips all 4 rotated corners to [0, 1] and returns axis-aligned wrapping box.
    - Ensures all corners are inside bounds
    - Sets angle to 0 (result is axis-aligned after clipping)
    - Does NOT use `cv2.minAreaRect` - that's only for rotations

**Example configurations:**

```python
# Strict: clip input errors AND after each transform
BboxParams(
    bbox_format='yolo',
    clip_bboxes_on_input=True,     # Fix input errors once
    clip_after_transform='geometry' # Clip after each transform
)

# Lenient: allow temporary excursions
BboxParams(
    bbox_format='albumentations',
    clip_bboxes_on_input=True,    # Fix input errors once
    clip_after_transform=None      # Allow boxes outside [0,1] during pipeline
)
```

#### `filter_invalid_bboxes: bool = False`
If `True`, removes bboxes with invalid coordinates (e.g., x_min >= x_max) during preprocessing.

---

## Processing Pipeline

### Pipeline Overview

```
1. Input (user format)
   ↓
2. Preprocess (BboxProcessor.preprocess)
   - Convert format → Albumentations normalized format
   - Clip to boundaries (if clip_bboxes_on_input=True)
   - Filter invalid boxes (if filter_invalid_bboxes=True)
   ↓
3. Apply Transforms (Compose)
   For each transform:
     - Execute transform.apply_to_bboxes()
     - Apply clipping (if clip_after_transform is set)
     - Filter by min_area, min_visibility, min_width, min_height
   ↓
4. Postprocess (BboxProcessor.postprocess)
   - Convert format → Original user format
   - Return results
```

### Detailed Steps

#### Step 1: Preprocessing (`BboxProcessor.preprocess()`)

```python
# Input: User format
bboxes = [[10, 20, 100, 150], [50, 60, 200, 250]]  # pascal_voc

# Convert to Albumentations format (normalized)
bboxes_normalized = [
    [0.05, 0.1, 0.5, 0.75],   # x/width, y/height
    [0.25, 0.3, 1.0, 1.25]    # Note: x_max > 1.0 (outside image!)
]

# If clip_bboxes_on_input=True: clip to [0, 1]
bboxes_clipped = [
    [0.05, 0.1, 0.5, 0.75],
    [0.25, 0.3, 1.0, 1.0]     # x_max clipped to 1.0
]

# If filter_invalid_bboxes=True: remove invalid boxes
# (boxes where x_min >= x_max or y_min >= y_max)
```

#### Step 2: Transform Application

Each transform receives normalized bboxes and returns transformed normalized bboxes.

**Example: Crop transform**

```python
# Input image: 200x200, crop to (50, 50, 150, 150) → 100x100
# Input bbox: [0.2, 0.2, 0.8, 0.8] (normalized to 200x200)

# Transform only shifts coordinates:
# 1. Denormalize: [0.2*200, 0.2*200, 0.8*200, 0.8*200] = [40, 40, 160, 160]
# 2. Shift: [40-50, 40-50, 160-50, 160-50] = [-10, -10, 110, 110]
# 3. Normalize to crop: [-10/100, -10/100, 110/100, 110/100] = [-0.1, -0.1, 1.1, 1.1]

# Result: bbox temporarily outside [0, 1]!
```

**Critical**: Transform functions like `crop_bboxes_by_coords()` and `crop_and_pad_bboxes()`:
- Only shift the first 4 coordinate columns
- Do NOT clip or validate
- Preserve all other columns (angle, labels, etc.) unchanged
- Work identically for HBB and OBB

#### Step 3: Post-Transform Filtering

After **each** transform, `Compose` applies filtering (in `filter_bboxes()`):

```python
# 1. Apply clip_after_transform
if clip_after_transform == 'geometry':
    if bbox_type == 'hbb':
        # Clip coordinates to [0, 1]
        bbox[0] = max(0.0, min(1.0, bbox[0]))  # x_min
        bbox[1] = max(0.0, min(1.0, bbox[1]))  # y_min
        bbox[2] = max(0.0, min(1.0, bbox[2]))  # x_max
        bbox[3] = max(0.0, min(1.0, bbox[3]))  # y_max
    elif bbox_type == 'obb':
        # Convert to polygon, clip corners, return axis-aligned wrapping box with angle=0
        # Angle is reset to 0; dimensions may change

# 2. Check dimensions
if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
    # Remove box (zero or negative dimensions)

# 3. Apply filters
if min_area and area < min_area:
    # Remove box
if min_visibility and visibility < min_visibility:
    # Remove box
if min_width and width < min_width:
    # Remove box
if min_height and height < min_height:
    # Remove box
if max_accept_ratio and aspect_ratio > max_accept_ratio:
    # Remove box
```

#### Step 4: Postprocessing (`BboxProcessor.postprocess()`)

```python
# Convert back from Albumentations format to user format
# Example: albumentations → pascal_voc
[0.1, 0.2, 0.5, 0.6] → [20, 40, 100, 120]  # multiply by image dimensions
```

---

## Input Clipping vs Transform Clipping

Understanding the difference between `clip_bboxes_on_input` and `clip_after_transform` is critical:

### `clip_bboxes_on_input` Parameter
- **When**: Runs ONCE at pipeline start (during `preprocess()`)
- **Purpose**: Fix invalid input data
- **Use case**: Handle malformed bboxes from external sources (e.g., YOLO coords like -1e-6)
- **Independent**: Does not affect transform behavior

```python
# Example: Fix bad YOLO coordinates
BboxParams(bbox_format='yolo', clip_bboxes_on_input=True)
# Input:  [0.5, 0.5, 0.4, 0.4, -0.000001]  # Tiny negative value
# After:  [0.5, 0.5, 0.4, 0.4, 0.0]        # Clipped to [0, 1]
```

### `clip_after_transform` Parameter
- **When**: Runs AFTER EACH transform (during `filter_bboxes()`)
- **Purpose**: Handle augmentation-induced coordinate excursions
- **Use case**: Control how boxes that go outside bounds are handled
- **Affects**: Transform pipeline behavior

```python
# Example: Crop that moves bbox outside bounds
BboxParams(clip_after_transform='geometry')
# After crop: [-0.1, -0.1, 1.1, 1.1]  # Outside bounds
# After clip: [0.0, 0.0, 1.0, 1.0]    # Clipped to [0, 1]

BboxParams(clip_after_transform=None)
# After crop: [-0.1, -0.1, 1.1, 1.1]  # Left as-is
# Later transform (e.g., pad) may bring it back inside
```

### Decision Matrix

| Scenario | `clip_bboxes_on_input` | `clip_after_transform` |
|----------|------------------------|------------------------|
| Fix malformed input | ✅ `True` | N/A |
| Strict bounds enforcement | Optional | `'geometry'` |
| Lenient pipeline (crop→pad) | Optional | `None` |
| OBB with accurate geometry | Optional | `'geometry'` |
| Performance critical (HBB only) | Optional | `'geometry'` (fast) |

---

## Filtering Parameters

Filters are applied after each transform (after clamping):

### Area-based Filtering

```python
# Remove boxes smaller than 100px²
BboxParams(min_area=100.0)

# Example:
# Box: [0.1, 0.1, 0.2, 0.15] on 1000x1000 image
# Area: (0.2-0.1)*1000 * (0.15-0.1)*1000 = 100*50 = 5000px²
# Result: KEPT (5000 >= 100)
```

### Visibility-based Filtering

```python
# Remove boxes with less than 30% visible area
BboxParams(min_visibility=0.3)

# Example after crop:
# Original box area: 1000px²
# Visible area after crop: 250px²
# Visibility: 250/1000 = 0.25
# Result: REMOVED (0.25 < 0.3)
```

### Dimension-based Filtering

```python
# Remove boxes smaller than 10px in either dimension
BboxParams(min_width=10.0, min_height=10.0)

# Example:
# Box: [0.1, 0.1, 0.105, 0.2] on 1000x1000 image
# Width: (0.105-0.1)*1000 = 5px
# Height: (0.2-0.1)*1000 = 100px
# Result: REMOVED (5 < 10)
```

### Aspect Ratio Filtering

```python
# Remove boxes with aspect ratio > 5:1
BboxParams(max_accept_ratio=5.0)

# Example:
# Box: [0.1, 0.1, 0.7, 0.15] on 1000x1000 image
# Width: 600px, Height: 50px
# Aspect ratio: max(600/50, 50/600) = 12.0
# Result: REMOVED (12.0 > 5.0)
```

---

## Transform-Level Behavior

### Coordinate Transformation

Transform functions in the functional layer (e.g., `crop_bboxes_by_coords()`) must follow these rules:

1. **Only shift first 4 columns** (`x_min`, `y_min`, `x_max`, `y_max`)
2. **Preserve all other columns** unchanged (angle for OBB, labels, etc.)
3. **No bbox type detection** - work identically for HBB and OBB
4. **No clipping/validation** - that happens in `Compose`

```python
def crop_bboxes_by_coords(
    bboxes: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Shift bbox coordinates - works for both HBB and OBB."""
    result = bboxes.copy()

    # Only transform first 4 columns
    result[:, [0, 2]] = (result[:, [0, 2]] * image_shape[1] - crop_x_min) / crop_width
    result[:, [1, 3]] = (result[:, [1, 3]] * image_shape[0] - crop_y_min) / crop_height

    # Columns 4+ (angle, labels, etc.) preserved unchanged
    return result
```

### OBB-Specific Considerations

For OBB, transforms only affect the bounding rectangle coordinates:
- The **angle** (column 4) is preserved unchanged during coordinate transformations
- The bounding rectangle shifts/scales with the image
- The angle represents the rotation of the object within that rectangle

**Example: Crop with OBB**
```python
# Input OBB: [0.2, 0.2, 0.8, 0.8, 45.0]
# Crop shifts coordinates, preserves angle
# Output OBB: [0.1, 0.1, 0.9, 0.9, 45.0]  # Same angle!
```

**Exception**: Some transforms DO change the angle:
- `Rotate`: Updates angle by adding rotation amount
- `Affine` with rotation: Updates angle accordingly
- Horizontal/Vertical flip: Negates angle in certain cases

---

## Best Practices

### 1. Always Validate Input Format

```python
# ❌ BAD: Assuming format without validation
bboxes = load_bboxes_from_file()  # Unknown format!
transform = A.Compose([...], bbox_params=A.BboxParams(bbox_format='pascal_voc'))

# ✅ GOOD: Explicit format handling
bboxes = load_bboxes_from_file()
if bbox_format == 'yolo':
    bbox_params = A.BboxParams(bbox_format='yolo', bbox_type='obb')
```

### 2. Use `clip_bboxes_on_input=True` for External Data

```python
# ✅ GOOD: Clip malformed external data
bbox_params = A.BboxParams(
    bbox_format='yolo',
    clip_bboxes_on_input=True,  # Fix rounding errors from external sources
)
```

### 3. Choose Appropriate `clip_after_transform`

```python
# Strict pipeline - always keep boxes in bounds
bbox_params = A.BboxParams(clip_after_transform='geometry')

# Lenient pipeline - allow temporary excursions
bbox_params = A.BboxParams(clip_after_transform=None)
```

### 4. Set Meaningful Filters

```python
# ❌ BAD: No filtering (tiny/invalid boxes may remain)
bbox_params = A.BboxParams(bbox_format='pascal_voc')

# ✅ GOOD: Filter out problematic boxes
bbox_params = A.BboxParams(
    bbox_format='pascal_voc',
    min_area=100.0,        # Remove tiny boxes
    min_visibility=0.3,    # Remove heavily cropped boxes
    min_width=5.0,         # Remove thin boxes
    min_height=5.0,
    max_accept_ratio=10.0, # Remove extreme aspect ratios
)
```

### 5. Never Auto-Detect bbox_type

```python
# ❌ BAD: Detecting from number of columns
if bboxes.shape[1] >= 5:
    bbox_type = 'obb'  # WRONG! Could be HBB with extra labels

# ✅ GOOD: Explicit configuration
bbox_params = A.BboxParams(
    bbox_format='albumentations',
    bbox_type='obb',  # User knows their data format
    label_fields=['class_ids', 'track_ids'],  # May add columns
)
```

### 6. Separate Label Fields

```python
# ✅ GOOD: Keep labels in separate arrays
transform = A.Compose([
    A.RandomCrop(width=512, height=512),
], bbox_params=A.BboxParams(
    bbox_format='pascal_voc',
    label_fields=['class_labels', 'track_ids']
))

result = transform(
    image=image,
    bboxes=bboxes,              # Shape: (N, 4) or (N, 5) for OBB
    class_labels=class_labels,  # Shape: (N,)
    track_ids=track_ids,        # Shape: (N,)
)
```

### 7. Handle Empty Results

```python
result = transform(image=image, bboxes=bboxes)

if len(result['bboxes']) == 0:
    # All boxes were filtered out!
    # Handle gracefully (skip image, use backup, etc.)
    pass
```

### 8. OBB Angle Conventions

```python
# ✅ GOOD: Use consistent angle convention
# Albumentations uses [-180, 180) degrees
# Angle is automatically normalized

bbox_obb = [0.2, 0.2, 0.8, 0.8, 450.0]  # Input
# After normalization: [0.2, 0.2, 0.8, 0.8, 90.0]  # 450 % 360 = 90
```

### 9. Testing Transforms

```python
# ✅ GOOD: Test edge cases
test_cases = [
    # Box at image boundary
    [0.0, 0.0, 0.1, 0.1],
    # Box crossing boundary (if clip_bboxes_on_input=False)
    [-0.1, -0.1, 0.1, 0.1],
    # Tiny box
    [0.5, 0.5, 0.501, 0.501],
    # Large box
    [0.1, 0.1, 0.9, 0.9],
]

for bbox in test_cases:
    result = transform(image=image, bboxes=[bbox])
    assert validate_bbox(result['bboxes'])
```

### 10. Performance Considerations

```python
# For HBB with many boxes and strict bounds:
bbox_params = A.BboxParams(
    bbox_format='yolo',
    bbox_type='hbb',
    clip_after_transform='geometry',  # Fast for HBB
)

# For OBB with accurate geometry (slower):
bbox_params = A.BboxParams(
    bbox_format='albumentations',
    bbox_type='obb',
    clip_after_transform='geometry',  # Refits boxes, may change angle
)

# For maximum performance (careful!):
bbox_params = A.BboxParams(
    bbox_format='albumentations',
    clip_after_transform=None,  # No clipping after transforms
)
```

---

## Summary

Key takeaways:

1. **Two bbox types**: HBB (4 values) and OBB (5 values: 4 coords + angle)
2. **Never auto-detect**: Always explicitly specify `bbox_type` in `BboxParams`
3. **Two-stage clipping**:
   - `clip_bboxes_on_input=True`: Fixes input once at start
   - `clip_after_transform`: Controls behavior after each transform
4. **Transform functions**: Only shift coordinates, preserve other columns
5. **Filtering**: Applied after each transform based on area, visibility, dimensions
6. **OBB angle**: Preserved during coordinate shifts, updated by rotation transforms
7. **Label fields**: Use `label_fields` for additional per-bbox data

**Deprecated parameters**:
- `format` → use `bbox_format` (avoids shadowing built-in)
- `clip` → use `clip_bboxes_on_input` (clearer timing)

For implementation details, see:
- `albumentations/core/bbox_utils.py`: BboxParams and BboxProcessor
- `albumentations/augmentations/crops/functional.py`: Transform functions
- `albumentations/core/composition.py`: Pipeline and filtering logic
