# Target-Specific Sampling Inventory

**Status**: maintained with the greenfield sampling contract

This inventory is the review surface for every built-in `sample_parameters` override. It prevents a new transform from
silently reintroducing first-target sampling or target names embedded in execution parameters. The inventory is grouped by
the contract the sampler must keep, not by the transform's public family.

The audit can be regenerated with:

```bash
uv run python - <<'PY'
import ast
from pathlib import Path

for path in sorted(Path("albumentations/augmentations").rglob("*.py")):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
            isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name == "sample_parameters"
            for member in node.body
        ):
            print(path, node.name)
PY
```

The source hook enforces the boundaries mechanically: `AXG021` rejects flat sampler returns, `AXG022` rejects first-target
shape helpers, and `AXG023` rejects target-routing names in application parameters. A transform may return no target-specific parameters
when the sampled value is independent of target representation and content.

## Grouped by representation or actual target

These samplers materialize one or more `TargetParams` values. The grouping key is part of each transform's contract;
it must include every property used to construct the execution value.

- `albumentations/augmentations/blur/transforms.py`: `GaussianBlur`.
- `albumentations/augmentations/dropout/channel_dropout.py`: `ChannelDropout`.
- `albumentations/augmentations/dropout/coarse_dropout.py`: `CoarseDropout`, `Erasing`.
- `albumentations/augmentations/dropout/grid_dropout.py`: `GridDropout`.
- `albumentations/augmentations/dropout/grid_mask.py`: `GridMask`.
- `albumentations/augmentations/dropout/transforms.py`: `PixelDropout`.
- `albumentations/augmentations/dropout/xy_masking.py`: `XYMasking`.
- `albumentations/augmentations/pixel/channel.py`: `ChannelShuffle`.
- `albumentations/augmentations/pixel/color_advanced.py`: `RGBShift`, `PhotoMetricDistort`.
- `albumentations/augmentations/pixel/color_basic.py`: `RandomToneCurve`, `ExposureMatching`.
- `albumentations/augmentations/pixel/color_gray.py`: `FancyPCA`.
- `albumentations/augmentations/pixel/color_lighting.py`: `PlasmaBrightnessContrast`, `PlasmaShadow`.
- `albumentations/augmentations/pixel/noise.py`: `GaussNoise`, `MultiplicativeNoise`, `AdditiveNoise`,
  `SaltAndPepper`, `FilmGrain`, `RicianNoise`.
- `albumentations/augmentations/pixel/transforms.py`: `Dithering`, `LensFlare`.
- `albumentations/augmentations/pixel/weather.py`: `RandomSnow`, `RandomGravel`, `RandomRain`, `RandomFog`,
  `RandomSunFlare`, `RandomShadow`, `Spatter`, `AtmosphericFog`.

## Shared spatial frame

These samplers use `inputs.spatial_frame`, which is validated against every active spatial target before sampling. They
share geometry intentionally; representation-dependent values must not be added to this category without a group key.

- `albumentations/augmentations/blur/transforms.py`: `GlassBlur`.
- `albumentations/augmentations/crops/basic.py`: `RandomCrop`, `CenterCrop`, `Crop`, `CropAndPad`.
- `albumentations/augmentations/crops/bbox_safe.py`: `BBoxSafeRandomCrop`, `BBoxSubsetSafeRandomCrop`,
  `AtLeastOneBBoxRandomCrop`.
- `albumentations/augmentations/crops/sized.py`: `RandomSizedCrop`, `RandomResizedCrop`.
- `albumentations/augmentations/crops/special.py`: `RandomCropNearBBox`, `RandomCropFromBorders`.
- `albumentations/augmentations/geometric/distortion.py`: `ElasticTransform`, `PiecewiseAffine`, `OpticalDistortion`,
  `GridDistortion`, `ThinPlateSpline`, `WaterRefraction`, `PixelSpread`.
- `albumentations/augmentations/geometric/pad.py`: `PadIfNeeded`.
- `albumentations/augmentations/geometric/resize.py`: `LongestMaxSize`, `SmallestMaxSize`, `LetterBox`.
- `albumentations/augmentations/geometric/rotate.py`: `Rotate`, `SafeRotate`.
- `albumentations/augmentations/geometric/transforms.py`: `Perspective`, `Affine`, `GridElasticDeform`,
  `RandomGridShuffle`.
- `albumentations/augmentations/mixing/copy_paste.py`: `CopyAndPaste`.
- `albumentations/augmentations/mixing/domain_adaptation.py`: `FDA`.
- `albumentations/augmentations/transforms3d/transforms.py`: `Affine3D`, `Anisotropy3D`, `Resize3D`, `PadIfNeeded3D`,
  `CenterCrop3D`, `RandomCrop3D`, `CoarseDropout3D`, `Flip3D`, `CubicSymmetry`, `RandomRotate90_3D`, `GridShuffle3D`.

## Content-derived or mixed policy

These samplers read caller content or annotation metadata. They must document which target is the reference and, when the
realized value is consumed by multiple representations, move the materialization into `TargetParams` records.

- `albumentations/augmentations/crops/special.py`: `CropNonEmptyMaskIfExists`.
- `albumentations/augmentations/dropout/coarse_dropout.py`: `ConstrainedCoarseDropout`.
- `albumentations/augmentations/dropout/guided_coarse_dropout.py`: `GuidedCoarseDropout`.
- `albumentations/augmentations/dropout/mask_dropout.py`: `MaskDropout`.
- `albumentations/augmentations/mixing/domain_adaptation.py`: `HistogramMatching`, `PixelDistributionAdaptation`.
- `albumentations/augmentations/mixing/mosaic.py`: `Mosaic`.
- `albumentations/augmentations/mixing/overlay.py`: `OverlayElements`.
- `albumentations/augmentations/other/annotation_artifacts.py`: `AnnotationArtifacts`.
- `albumentations/augmentations/pixel/color_advanced.py`: `HEStain`.
- `albumentations/augmentations/pixel/color_basic.py`: `Equalize`.
- `albumentations/augmentations/text/transforms.py`: `TextImage`.

## Parameters without target-specific values

These samplers produce values independent of active target representation. If a future change makes one of these values
depend on shape, channels, dtype, topology, or target content, move it to one of the two sections above and add a mixed
representation test before changing the implementation.

- `albumentations/augmentations/blur/transforms.py`: `Blur`, `MotionBlur`, `ModeFilter`, `AdvancedBlur`, `Defocus`,
  `ZoomBlur`.
- `albumentations/augmentations/dropout/transforms.py`: `BaseDropout` (abstract contract only).
- `albumentations/augmentations/geometric/flip.py`: `D4`.
- `albumentations/augmentations/geometric/pad.py`: `Pad`.
- `albumentations/augmentations/geometric/resize.py`: `RandomScale`.
- `albumentations/augmentations/geometric/rotate.py`: `RandomRotate90`.
- `albumentations/augmentations/geometric/transforms.py`: `Morphological`.
- `albumentations/augmentations/pixel/color_advanced.py`: `ColorJitter`, `ChromaticAberration`, `PlanckianJitter`.
- `albumentations/augmentations/pixel/color_basic.py`: `HueSaturationValue`, `Solarize`, `Posterize`,
  `RandomBrightnessContrast`, `CLAHE`, `RandomGamma`.
- `albumentations/augmentations/pixel/color_gray.py`: `Colorize`.
- `albumentations/augmentations/pixel/color_lighting.py`: `Illumination`, `Vignetting`.
- `albumentations/augmentations/pixel/compression.py`: `ImageCompression`, `Downscale`.
- `albumentations/augmentations/pixel/noise.py`: `ISONoise`, `ShotNoise`.
- `albumentations/augmentations/pixel/transforms.py`: `Sharpen`, `Emboss`, `Enhance`, `Superpixels`, `RingingOvershoot`,
  `UnsharpMask`, `Halftone`.
- `albumentations/augmentations/transforms3d/transforms.py`: `Pad3D`.

## Review requirements for additions

1. Add the sampler to exactly one category and state the sharing key in its docstring or the implementation comment next
   to the `TargetParams` construction.
2. Test at least two actual target keys, including an `additional_targets` alias when the transform supports one.
3. Vary the representation property that drives materialization: shape, channels, dtype scale, layout, topology, or
   content. A test that only checks output shape does not prove the plan is target-correct.
4. Assert the structured replay payload and verify replay does not sample again.
5. Run `check-ax-coding-guidance`, the focused transform tests, and the full quality gate.
