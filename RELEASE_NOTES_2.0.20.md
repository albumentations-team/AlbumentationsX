# Albumentations 2.0.20 Release Notes

This release brings faster video/batch processing for two widely-used transforms and a new `user_data` target that lets you pass arbitrary custom data through any augmentation pipeline.

## Performance Improvements

### [Perspective](https://explore.albumentations.ai/transform/Perspective) — up to 2.7× faster on video batches

[Perspective](https://explore.albumentations.ai/transform/Perspective) is now significantly faster when applied to video or image batches (the `images=` key / `apply_to_images` path).

Benchmark results (uint8, 16 frames per batch, comparing 2.0.19 → 2.0.20):

| Config | 2.0.19 | 2.0.20 | Speedup |
|---|---|---|---|
| 16×256×256 grayscale | 0.0027s | 0.0010s | **2.7×** |
| 32×256×256 grayscale | 0.0056s | 0.0023s | **2.4×** |
| 16×1024×1024 RGB | 0.0139s | 0.0095s | **1.5×** |
| 32×1024×1024 RGB | 0.0274s | 0.0223s | **1.2×** |
| 16×1024×1024 5ch | 0.0259s | 0.0186s | **1.4×** |
| 32×1024×1024 5ch | 0.0555s | 0.0381s | **1.5×** |

The gain is largest for grayscale (depth maps, medical slices) where the batch can be warped in a single C++ call instead of one per frame.

### [HueSaturationValue](https://explore.albumentations.ai/transform/HueSaturationValue) — up to 1.2× faster on video batches

[HueSaturationValue](https://explore.albumentations.ai/transform/HueSaturationValue) now avoids redundant memory allocations when processing batches. Gains are most visible at large resolutions.

| Config | 2.0.19 | 2.0.20 | Speedup |
|---|---|---|---|
| 8×1024×1024 RGB | 0.0325s | 0.0291s | **1.1×** |
| 16×1024×1024 RGB | 0.0940s | 0.0767s | **1.2×** |
| 32×1024×1024 RGB | 0.1456s | 0.1216s | **1.2×** |

## New Feature: `user_data` Target

You can now pass arbitrary Python objects through an Albumentations pipeline alongside images, masks, bboxes, and keypoints. By default `user_data` passes through unchanged. To update it in response to a transform, subclass the transform and override `apply_to_user_data`.

### Basic usage — passthrough

```python
import numpy as np
import albumentations as A

image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

transform = A.Compose([A.HorizontalFlip(p=1.0)])
result = transform(image=image, user_data={"caption": "a cat on the left"})

print(result["user_data"])  # {"caption": "a cat on the left"} — unchanged
```

### Custom transform that updates `user_data`

```python
class FlipAwareHorizontalFlip(A.HorizontalFlip):
    def apply_to_user_data(self, data: dict, **params) -> dict:
        caption = data["caption"].replace("left", "right").replace("right side", "left side")
        return {**data, "caption": caption}

transform = A.Compose([FlipAwareHorizontalFlip(p=1.0)])
result = transform(image=image, user_data={"caption": "a cat on the left"})
print(result["user_data"])  # {"caption": "a cat on the right"}
```

### Use case: camera intrinsics for robot/autonomous driving

When you crop or zoom an image you need to update the camera intrinsics matrix K so downstream 3D reasoning stays correct.

```python
import numpy as np
import albumentations as A

class CropAwareCenterCrop(A.CenterCrop):
    def apply_to_user_data(self, data: dict, crop_coords: tuple, **params) -> dict:
        x1, y1, x2, y2 = crop_coords
        K = data["K"].copy()
        K[0, 2] -= x1  # shift principal point cx
        K[1, 2] -= y1  # shift principal point cy
        return {**data, "K": K}

K = np.array([[500, 0, 320],
              [0, 500, 240],
              [0,   0,   1]], dtype=np.float64)

image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

transform = A.Compose([CropAwareCenterCrop(height=256, width=256, p=1.0)])
result = transform(image=image, user_data={"K": K})

print(result["user_data"]["K"])
# [[500.   0. 192.]   ← cx updated
#  [  0. 500. 112.]   ← cy updated
#  [  0.   0.   1.]]
```

### Use case: vision-language models (caption-aware augmentation)

Pass an image + text caption through the same pipeline and keep the caption semantically consistent with the augmented image.

```python
class CaptionAwareFlip(A.HorizontalFlip):
    def apply_to_user_data(self, data: dict, **params) -> dict:
        caption = (
            data["caption"]
            .replace("left", "__L__")
            .replace("right", "left")
            .replace("__L__", "right")
        )
        return {**data, "caption": caption}

transform = A.Compose([CaptionAwareFlip(p=1.0)])
result = transform(
    image=image,
    user_data={"caption": "a dog running to the left of the frame"},
)
print(result["user_data"]["caption"])
# "a dog running to the right of the frame"
```

### Other strong use cases

- **LiDAR / BEV**: pass a point cloud dict alongside the image; update 3D coordinates when the image is flipped or rotated
- **Temporal metadata**: pass frame timestamps when cropping video clips; update the time range to match the spatial crop
- **Soft labels / confidence maps**: pass pixel-wise annotation confidence; update spatial positions to match geometric transforms
- **Multi-sensor fusion**: pass IMU or GPS data that needs coordinate-frame corrections when the image is spatially transformed
