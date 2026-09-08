# Migrating custom augmentation wrappers

AlbumentationsX 2.4.6 and later can replace many wrappers that historical training code used to synchronize inputs,
reuse random parameters, or isolate selected channels. This guide maps each pattern to a current public API and states
where custom code is still required.

Every recipe uses one of these classifications:

- **Exact replacement:** the current API preserves the historical behavior.
- **Same intent:** the current API owns the synchronization or state, with a documented representation difference.
- **Partial replacement:** AlbumentationsX owns the augmentation step, while workload-specific code remains outside it.

## Sequence and video synchronization

**Classification: exact replacement for framewise spatial and pixel transforms.**

Historical code often sampled one transform and then reached into private parameters or reset random seeds before each
frame. Pass the complete sequence through `images` instead. NumPy sequences use `(N, H, W, C)`. AlbumentationsX samples
one realization and applies it to every frame.

### Before: rebuild a seeded pipeline for every frame

```python
import albumentations as A
import numpy as np


def augment_frames_one_by_one(frames: np.ndarray) -> np.ndarray:
    outputs = []
    for frame in frames:
        transform = A.Compose(
            [A.RandomCrop(height=20, width=28, p=1.0), A.HorizontalFlip(p=0.5)],
            seed=137,
            strict=True,
        )
        outputs.append(transform(image=frame)["image"])
    return np.stack(outputs)
```

This wrapper is runnable, but rebuilding the pipeline couples synchronization to an implementation detail and repeats
setup work.

### After: pass the sequence as one public target

```python
frames = np.arange(4 * 24 * 32 * 3, dtype=np.uint8).reshape(4, 24, 32, 3)

video_transform = A.Compose(
    [
        A.RandomCrop(height=20, width=28, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ],
    seed=137,
    strict=True,
)

augmented_frames = video_transform(images=frames)["images"]
assert augmented_frames.shape == (4, 20, 28, 3)
```

This contract synchronizes augmentation parameters, not video decoding or temporal sampling. Decode clips and choose
timestamps before calling `Compose`.

## Ordered slices and volumes

**Classification: same intent with explicit volume semantics.**

A Python loop over ordered slices can accidentally sample a different crop or flip for each slice. Pass a NumPy volume
as `(D, H, W, C)` through `volume`. A 2D transform uses one realization across depth. Native 3D transforms can also
change the depth axis when their API documents that behavior.

```python
volume = np.arange(8 * 32 * 40, dtype=np.float32).reshape(8, 32, 40, 1)

slice_transform = A.Compose(
    [A.HorizontalFlip(p=1.0)],
    strict=True,
)

augmented_volume = slice_transform(volume=volume)["volume"]
assert augmented_volume.shape == (8, 32, 40, 1)
```

The volume target does not replace acquisition-aware resampling, voxel-spacing correction, registration, or framework
container adapters. Perform those workload-specific operations at the dataset boundary.

## Parallel image-like inputs

**Classification: exact replacement when inputs share geometry.**

Use plural targets for a homogeneous stack. Use `additional_targets` when named arrays have separate meanings but must
receive the same sampled geometry.

```python
image = np.zeros((48, 64, 3), dtype=np.uint8)
depth = np.arange(48 * 64, dtype=np.float32).reshape(48, 64, 1)

paired_transform = A.Compose(
    [A.HorizontalFlip(p=1.0)],
    additional_targets={"depth": "image"},
    strict=True,
)

paired = paired_transform(image=image, depth=depth)
assert paired["image"].shape == image.shape
assert paired["depth"].shape == depth.shape
```

Mapping depth to `image` is appropriate for geometry in this example. Pixel transforms would also run on the alias.
Use a mask alias or a separate pipeline when interpolation and pixel behavior must differ.

## Structured target alignment

**Classification: exact replacement for supported boxes, masks, keypoints, and label fields.**

`BboxParams` and `KeypointParams` convert coordinates and filter invalid annotations. `instance_binding` groups each
instance mask, box, keypoint collection, and label record so crop-induced filtering removes one complete instance.

```python
image = np.zeros((80, 80, 3), dtype=np.uint8)
instances = [
    {
        "mask": np.pad(np.ones((20, 20), dtype=np.uint8), ((5, 55), (5, 55))),
        "bbox": np.array([5, 5, 25, 25], dtype=np.float32),
        "keypoints": np.array([[15.0, 15.0]], dtype=np.float32),
        "bbox_labels": {"class_id": "cat"},
    },
]

structured_transform = A.Compose(
    [A.Crop(x_min=0, y_min=0, x_max=40, y_max=40, p=1.0)],
    bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
    keypoint_params=A.KeypointParams(coord_format="xy"),
    instance_binding=["masks", "bboxes", "keypoints"],
    strict=True,
)

result = structured_transform(image=image, instances=instances)
assert result["instances"][0]["bbox_labels"]["class_id"] == "cat"
```

Use ordinary `bboxes`, `masks`, and `keypoints` when those collections do not share a one-instance-per-row contract.
Do not infer alignment from matching lengths.

## Metadata propagation

**Classification: exact replacement for metadata that augmentation should pass through unchanged.**

Use `user_data` for captions, identifiers, timestamps, or other values that do not participate in spatial processors.
The default handler preserves the object.

```python
metadata = {"clip_id": "train-137", "start_seconds": 4.5}
result = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)(
    image=image,
    user_data=metadata,
)
assert result["user_data"] == metadata
```

A custom transform may implement `apply_to_user_data` when the metadata must describe a realized augmentation. Keep
domain-specific parsing and storage outside the transform.

## Reusing sampled parameters

**Classification: exact replacement for replay, same intent for logging.**

Use `ReplayCompose` when another input must receive the exact realized transform parameters. Use
`save_applied_params=True` when the goal is inspection or logging rather than executable replay.

```python
tta = A.ReplayCompose(
    [A.RandomRotate90(p=1.0), A.HorizontalFlip(p=0.5)],
    seed=137,
    strict=True,
)

first = tta(image=image)
same_realization = A.ReplayCompose.replay(first["replay"], image=image)
np.testing.assert_array_equal(first["image"], same_realization["image"])
```

`ReplayCompose` records sampled execution. `A.to_dict` and `A.from_dict` serialize constructor configuration. Test both
boundaries when a stored pipeline must later reproduce policy and a saved invocation must reproduce one realization.

## RGB-only transforms inside multi-channel arrays

**Classification: exact replacement when the selected channel indices form the intended RGB view.**

Use `SelectiveChannelTransform` instead of splitting, augmenting, and concatenating channels manually.

### Before: split and concatenate channels

```python
multichannel = np.zeros((32, 32, 5), dtype=np.uint8)
multichannel[..., 3:] = 137

rgb = multichannel[..., :3]
extra = multichannel[..., 3:]
augmented_rgb = A.InvertImg(p=1.0)(image=rgb)["image"]
selected = np.concatenate([augmented_rgb, extra], axis=-1)
```

### After: select channels inside the pipeline

```python
multichannel = np.zeros((32, 32, 5), dtype=np.uint8)
multichannel[..., 3:] = 137

rgb_only = A.Compose(
    [
        A.SelectiveChannelTransform(
            [A.InvertImg(p=1.0)],
            channels=(0, 1, 2),
            p=1.0,
        ),
    ],
    strict=True,
)

selected = rgb_only(image=multichannel)["image"]
np.testing.assert_array_equal(selected[..., 3:], multichannel[..., 3:])
```

The channel selection defines representation, not semantics. Confirm that the selected channels are actually RGB
before using color transforms.

## Historical custom transforms with built-in replacements

| Historical pattern | Current API | Classification | Important difference |
| --- | --- | --- | --- |
| Manual time-axis flip | `TimeReverse` or the matching geometric flip | Exact replacement | The image axis must match time. |
| Time or frequency strip dropout | `XYMasking` | Exact replacement | Float ranges scale with the selected axis; integer ranges use pixels. |
| One realization over video frames | `images` | Exact replacement | Temporal decoding and frame selection remain external. |
| One realization over ordered slices | `volume` | Same intent | Native 3D transforms have separate depth semantics. |
| Paired image and depth geometry | `additional_targets` | Exact replacement for geometry | Pixel transforms follow the alias type too. |
| Manual random-state capture | `ReplayCompose` | Exact replacement | Replay data and constructor serialization are separate contracts. |
| Split RGB from extra bands | `SelectiveChannelTransform` | Exact replacement | Channel order remains the caller's responsibility. |
| Custom caption passthrough | `user_data` | Exact replacement | Metadata transformation needs an explicit custom handler. |
| Acquisition-aware resampling | External preprocessing plus `Compose` | Partial replacement | Image interpolation does not preserve domain sampling physics. |
| Framework-native sample object | Dataset adapter plus public targets | Partial replacement | `Compose` consumes named arrays and metadata, not arbitrary containers. |

## Serialization checklist

Before removing a historical wrapper:

1. Construct the replacement with `strict=True`.
2. Run representative targets through the public `Compose` call.
3. Round-trip constructor policy through `A.from_dict(A.to_dict(pipeline))`.
4. Capture and execute one `ReplayCompose` record when exact invocation replay is required.
5. Verify shape, dtype, target alignment, caller-input immutability, and deterministic seed behavior.
6. Keep decoding, resampling, collation, and model tensor conversion at their framework boundaries.

These checks distinguish a real replacement from a wrapper that only appears equivalent on one image.
