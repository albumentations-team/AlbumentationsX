# Keypoint Label Swapping Design Document

## Overview

This document describes the design for implementing semantic keypoint label swapping in AlbumentationsX transforms. The goal is to handle cases where keypoints have semantic meaning that changes during certain transforms (e.g., left/right eye labels should swap during horizontal flip).

## Problem Statement

Currently, AlbumentationsX transforms keypoint coordinates but ignores the semantic meaning of keypoint labels. For facial keypoints, pose estimation, and other structured keypoint datasets, this creates incorrect label assignments after transforms like horizontal flip.

### Example Problem

```python
# Before horizontal flip
keypoints = np.array([[100, 50], [200, 50]])  # [left_eye, right_eye]
labels = np.array(['left_eye', 'right_eye'])

# After horizontal flip - INCORRECT
keypoints = np.array([[200, 50], [100, 50]])  # Coordinates flipped
labels = np.array(['left_eye', 'right_eye'])   # Labels NOT flipped - WRONG!

# Should be
labels = np.array(['right_eye', 'left_eye'])   # Labels should swap too
```

## Design Principles

1. **No Hardcoded Labels**: The library should never contain hardcoded string labels
2. **User-Defined Mappings**: Users define their own label mappings in KeypointParams
3. **Integer-Based Processing**: All label transformations work on encoded integers in keypoints array
4. **Automatic Transform Integration**: DualTransform base class handles label swapping automatically
5. **Selective Field Mapping**: Users specify which label fields should be transformed

## Label Processing Flow

1. User provides labels (any hashable type) + mapping structure in KeypointParams
2. During preprocessing: Labels get encoded to integers and attached as extra columns to keypoints array
3. During transformation: DualTransform._apply_label_mapping_to_keypoints handles label column swapping
4. During postprocessing: Labels get decoded back to original format

## Implementation

### 1. Label Mapping Structure

```python
label_mapping = {
    'transform_name': {
        'label_field_name': {
            from_label: to_label,
            # ... more mappings
        },
        # ... more label fields
    },
    # ... more transforms
}
```

### 2. KeypointParams with Label Mapping

```python
# Single label field mapping
keypoint_params = A.KeypointParams(
    format='xy',
    label_fields=['keypoint_labels'],
    label_mapping={
        'HorizontalFlip': {
            'keypoint_labels': {
                'left_eye': 'right_eye',
                'right_eye': 'left_eye',
                'nose': 'nose',  # unchanged
            }
        },
        'VerticalFlip': {
            'keypoint_labels': {
                'top_lip': 'bottom_lip',
                'bottom_lip': 'top_lip',
            }
        }
    }
)

# Multiple label fields can be mapped
keypoint_params = A.KeypointParams(
    format='xy',
    label_fields=['keypoint_labels', 'visibility', 'keypoint_types'],
    label_mapping={
        'HorizontalFlip': {
            'keypoint_labels': {0: 1, 1: 0},      # Swap labels 0<->1
            'keypoint_types': {2: 3, 3: 2},       # Swap types 2<->3
            # 'visibility' not included - won't be transformed
        }
    }
)

# Integer labels work directly
keypoint_params = A.KeypointParams(
    format='xy',
    label_fields=['keypoint_labels'],
    label_mapping={
        'HorizontalFlip': {
            'keypoint_labels': {
                0: 1,  # left_eye -> right_eye
                1: 0,  # right_eye -> left_eye
                2: 2,  # nose -> nose (unchanged)
            }
        }
    }
)
```

### 3. Automatic Transform Integration

**No individual transform classes need modification** - the DualTransform base class automatically handles label swapping:

```python
class DualTransform(BasicTransform):
    def _apply_label_mapping_to_keypoints(self, keypoints: np.ndarray, **params) -> np.ndarray:
        """Apply label mapping to the label columns in the keypoints array."""
        # Gets processor, converts mappings to encoded integers, applies to label columns

    # Parity-changing transforms override apply_to_keypoints to call this method:
    def apply_to_keypoints(self, keypoints: np.ndarray, **params) -> np.ndarray:
        # Apply geometric transformation
        transformed = geometric_transform(keypoints, **params)
        # Apply label mapping to the label columns
        return self._apply_label_mapping_to_keypoints(transformed, **params)
```

### 4. Parity-Changing Transforms

Only transforms that change parity (mirror/reflect) need label swapping:

- **HorizontalFlip** - swaps left/right labels
- **VerticalFlip** - swaps top/bottom labels
- **Transpose** - swaps based on diagonal reflection
- **D4/SquareSymmetry** - maps group elements to base transforms:
  - `"h"` → `"HorizontalFlip"`
  - `"v"` → `"VerticalFlip"`
  - `"t"`, `"hvt"` → `"Transpose"`
  - `"e"`, `"r90"`, `"r180"`, `"r270"` → no label swapping

### 5. Encoding/Decoding Flow

1. **Preprocessing**: KeypointsProcessor.add_label_fields_to_data()
   - Encodes labels using LabelManager
   - Converts user mappings to encoded integer mappings
   - Attaches encoded labels as extra columns to keypoints array

2. **Transformation**: Transform.apply_to_keypoints()
   - Applies geometric transformation to coordinates
   - Calls _apply_label_mapping_to_keypoints() for label column swapping
   - Works entirely with encoded integers

3. **Postprocessing**: KeypointsProcessor.remove_label_fields_from_data()
   - Extracts label columns from keypoints array
   - Decodes labels back to original format using LabelManager

## What We Do NOT Want to Implement

### 1. Hardcoded Label Mappings

- **NO preset mappings** like 'coco_pose_17' or 'facial_68'
- **NO hardcoded string labels** in the library code
- **NO automatic label detection**

**Rationale**: Users have different label formats and conventions. Library should be agnostic.

### 2. Complex Geometric Label Transformations

- **NO automatic label updates** for arbitrary geometric transforms (affine, perspective, etc.)
- **NO angle-dependent** label swapping
- **NO coordinate-dependent** label logic

**Rationale**: Only simple, well-defined transforms (flips, 90° rotations) have clear semantic label mappings.

### 3. Separate Label Target Processing

- **NO separate apply_to_keypoint_labels methods** on individual transforms
- **NO label fields as separate targets** in the transform pipeline
- **NO custom label transform functions**

**Rationale**: Labels are part of the keypoints data structure. Keep it simple and integrated.

### 4. Multi-Parameter Keypoint Configuration

- **NO separate label_mapping_field parameter**
- **NO label_transform_functions parameter**
- **NO preset_mapping parameter**

**Rationale**: Single label_mapping parameter with nested structure handles all cases cleanly.

## Example Usage Patterns

### Basic Facial Keypoints

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),  # Rotation doesn't affect labels
], keypoint_params=A.KeypointParams(
    format='xy',
    label_fields=['keypoint_labels'],
    label_mapping={
        'HorizontalFlip': {
            'keypoint_labels': {
                'left_eye': 'right_eye',
                'right_eye': 'left_eye',
            }
        }
    }
))
```

### Multiple Label Fields

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
], keypoint_params=A.KeypointParams(
    format='xy',
    label_fields=['keypoint_labels', 'keypoint_types', 'visibility'],
    label_mapping={
        'HorizontalFlip': {
            'keypoint_labels': {'L_eye': 'R_eye', 'R_eye': 'L_eye'},
            'keypoint_types': {1: 2, 2: 1},  # Swap types 1<->2
            # 'visibility' not mapped - remains unchanged
        }
    }
))
```

### User-Defined Labels

```python
# User has their own keypoint format
transform = A.Compose([
    A.HorizontalFlip(p=1.0)
], keypoint_params=A.KeypointParams(
    format='xy',
    label_fields=['kp_labels'],
    label_mapping={
        'HorizontalFlip': {
            'kp_labels': {
                'L_eye': 'R_eye',
                'R_eye': 'L_eye',
                'L_ear': 'R_ear',
                'R_ear': 'L_ear',
                'nose_tip': 'nose_tip',  # unchanged
            }
        }
    }
))
```

## Technical Implementation

### 1. Mapping Conversion During Preprocessing

- KeypointsProcessor.convert_label_mappings_to_encoded() converts string mappings to integer mappings
- Called after labels are encoded by LabelManager
- Handles multiple label fields per transform

### 2. Label Swapping During Transformation

- DualTransform._apply_label_mapping_to_keypoints() swaps label columns
- Works on keypoints array columns after geometric transformation
- Fast integer-based operations only

### 3. Parity-Changing Transform Integration

- HorizontalFlip, VerticalFlip, Transpose override apply_to_keypoints
- D4/SquareSymmetry map group elements to base transform names
- No code changes needed for other transforms

## Testing Strategy

### 1. Unit Tests

- Test basic label mapping with string and integer labels
- Test multiple label fields
- Test partial mappings (some labels unmapped)
- Test D4/SquareSymmetry group element mapping

### 2. Integration Tests

- Test full pipeline with multiple transforms
- Test interaction with existing keypoint processing
- Test backward compatibility (no label mappings)

### 3. Performance Tests

- Benchmark label transformation overhead
- Test with large keypoint datasets
- Memory usage profiling

This design provides a clean, efficient foundation for semantic keypoint label handling while maintaining simplicity and performance.
