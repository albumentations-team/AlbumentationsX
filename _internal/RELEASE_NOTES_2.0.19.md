# Albumentations 2.0.19 Release Notes

We are excited to announce version 2.0.19 of Albumentations! This release brings significant improvements for Test-Time Augmentation (TTA) and massive performance boosts across almost all transforms.

## Performance Improvements
We removed the strict requirement for transforms to be contiguous before and after each transform application. This drastically decreases the copying of data for almost all transforms, which leads to massive speedups, especially for spatial and array-manipulation transforms.

You can view the latest performance numbers on our official benchmark pages:
- [Image Benchmarks](https://albumentations.ai/docs/benchmarks/image-benchmarks/)
- [Multichannel Benchmarks](https://albumentations.ai/docs/benchmarks/multichannel-benchmarks/)
- [Video Benchmarks](https://albumentations.ai/docs/benchmarks/video-benchmarks/)

## Test-Time Augmentation (TTA) Improvements
We have greatly simplified and improved the workflow for Test-Time Augmentation (TTA).

- **Group Element Selection**: Added the ability to choose a specific group element deterministically for transforms like `D4` or `SquareSymmetry`, providing the fine-grained control needed for robust TTA pipelines.
- **Inverse Transforms**: We've added an `.inverse()` method to spatial transforms like `D4`, `SquareSymmetry`, and flips (`HorizontalFlip`, `VerticalFlip`). This simplifies the process of inverting spatial transforms on predicted masks/keypoints back to the original coordinate space during TTA.

### Example of the new TTA pipeline:
Applying each `D4` symmetry, running inference, and restoring the prediction back to the original image orientation is now as simple as:

```python
import numpy as np
import albumentations as A
from albumentations.core.type_definitions import d4_group_elements

image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
predictions = []

for element in d4_group_elements:
    # 1. Deterministic transform for TTA
    aug = A.D4(p=1.0, group_element=element)

    # 2. Augment the image
    aug_image = aug(image=image)["image"]

    # 3. Model inference (placeholder)
    pred_mask = np.zeros((100, 100, 1), dtype=np.uint8)

    # 4. Magically undo the transform on the output
    restored_mask = aug.inverse()(image=pred_mask)["image"]

    predictions.append(restored_mask)
```
