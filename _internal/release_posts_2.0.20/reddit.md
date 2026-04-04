# Albumentations 2.0.20 — faster video/batch transforms + `user_data` target

**Albumentations 2.0.20** is out with performance improvements for video/batch processing and a new way to pass custom data through pipelines.

## Performance

| Transform | Config | Speedup |
|-----------|--------|---------|
| **Perspective** | 16×256×256 grayscale | **2.7×** |
| **Perspective** | 16×1024×1024 RGB | **1.5×** |
| **HueSaturationValue** | 16×1024×1024 RGB | **1.2×** |

Largest gains for grayscale video (depth maps, medical slices) — the batch is now warped in a single C++ call instead of per-frame.

## New: `user_data` target

Pass arbitrary Python objects through your pipeline alongside images, masks, bboxes, keypoints. Override `apply_to_user_data` to update them when transforms change the image.

**Examples:**
- **Camera intrinsics** — update matrix K when cropping (robot/autonomous driving)
- **Vision-language** — keep captions in sync with flips ("left" → "right")
- **LiDAR/BEV** — pass point clouds, update 3D coords on geometric transforms
- **Multi-sensor** — IMU/GPS with coordinate-frame corrections

```python
result = transform(image=image, user_data={"caption": "a cat on the left"})
# result["user_data"] — unchanged by default, or updated via apply_to_user_data
```

`pip install albumentationsx==2.0.20`
