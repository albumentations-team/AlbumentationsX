# Time-frequency augmentation

AlbumentationsX can augment a spectrogram or another time-frequency array when the array is represented as an image.
The mapping from domain axes to image axes is part of the data contract. Define it before choosing transforms.

This guide requires AlbumentationsX 2.4.6 or later. Every example below passes one sample as an explicit-channel
NumPy array with shape `(H, W, C)`.

## Choose an axis convention

The most common convention places frequency on rows and time on columns:

```text
shape = (frequency_bins, time_steps, channels)

                 image width (W)
                 time ->
              +-----------+
 image        |           |
 height (H)   | spectrum  |
 frequency    |           |
      |       +-----------+
      v
```

With that convention, X means time and Y means frequency:

```python
spectrogram[y_frequency, x_time, channel]
```

Some feature extractors return `(time_steps, frequency_bins)`. After adding a channel axis, that becomes
`(time_steps, frequency_bins, 1)`, so Y means time and X means frequency. Both conventions work, but the masking
arguments must follow the chosen image axes.

| Domain operation | Frequency on H, time on W | Time on H, frequency on W |
| --- | --- | --- |
| Time mask | `mask_x_length_range` | `mask_y_length_range` |
| Frequency mask | `mask_y_length_range` | `mask_x_length_range` |
| Time reversal | `A.TimeReverse` or `A.HorizontalFlip` | `A.VerticalFlip` |

`TimeMasking` and `FrequencyMasking` use the first convention. They provide fixed pixel-length compatibility aliases.
Use `XYMasking` for relative lengths, multiple masks, nonzero fills, or the second convention.

## Audio spectrogram pipeline

This log-mel example uses shape `(128, 400, 1)`: 128 mel bins on H, 400 time steps on W, one channel, `float32`, and
standardized values with a typical range near `[-3, 3]`. A fill of `0.0` represents the training-set mean after
standardization.

```python
import albumentations as A
import numpy as np

log_mel = np.linspace(-3.0, 3.0, 128 * 400, dtype=np.float32).reshape(128, 400, 1)

audio_transform = A.Compose(
    [
        A.XYMasking(
            num_masks_x_range=(1, 2),
            num_masks_y_range=(1, 2),
            mask_x_length_range=(0.05, 0.15),  # 5% to 15% of time
            mask_y_length_range=(0.05, 0.10),  # 5% to 10% of mel bins
            fill=0.0,
            p=1.0,
        ),
        A.TimeReverse(p=0.25),
    ],
    seed=137,
    strict=True,
)

augmented_log_mel = audio_transform(image=log_mel)["image"]
assert augmented_log_mel.shape == (128, 400, 1)
assert augmented_log_mel.dtype == np.float32
```

The float endpoints in `mask_x_length_range` and `mask_y_length_range` are fractions of their corresponding image
axes. Integer endpoints express pixel counts.

## EEG time-frequency pipeline

This example stacks 19 electrode spectrograms as channels. Its shape is `(64, 256, 19)`: 64 frequency bins on H,
256 time steps on W, 19 channels, `float32`, and per-electrode standardized values. One sampled mask is shared across
all 19 channels, preserving electrode alignment.

```python
import albumentations as A
import numpy as np

rng = np.random.default_rng(137)
eeg = rng.normal(size=(64, 256, 19)).astype(np.float32)

eeg_transform = A.Compose(
    [
        A.XYMasking(
            num_masks_x_range=(2, 3),
            num_masks_y_range=(1, 1),
            mask_x_length_range=(0.04, 0.12),
            mask_y_length_range=(0.08, 0.15),
            fill=0.0,
            p=1.0,
        ),
    ],
    seed=137,
    strict=True,
)

augmented_eeg = eeg_transform(image=eeg)["image"]
assert augmented_eeg.shape == (64, 256, 19)
assert augmented_eeg.dtype == np.float32
```

If electrodes have separate masks or missing-channel policies, apply those before or after `Compose`. Do not map an
electrode axis to H or W unless spatial transforms are intended to mix or resample electrodes.

## Dtype, range, fill, and interpolation

Choose the fill value in the same representation as the input.

| Representation | Input contract | Typical fill | Notes |
| --- | --- | --- | --- |
| Display image | `uint8`, `[0, 255]` | `0` or the encoded background | Quantization has already discarded physical scale. |
| Log magnitude or log mel | `float32`, documented finite range | Dataset mean after normalization | Zero means silence only when the preprocessing defines it that way. |
| Linear magnitude or power | `float32`, nonnegative | A physical floor or calibrated noise floor | Negative photometric outputs are usually invalid. |
| Raw complex STFT | Complex-valued array | None | Convert or augment in a signal-processing library first. AlbumentationsX image inputs are not complex. |

Remove or replace NaN and infinite values before `Compose`. Image transforms and OpenCV kernels do not define a
domain-correct policy for missing spectral measurements.

```python
finite_log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=3.0, neginf=-3.0)
```

`Resize` performs geometric interpolation on the displayed grid. It does not preserve waveform duration, sample rate,
FFT-bin centers, mel-filter definitions, or phase. Use a signal-processing resampler when those quantities matter.
If a grid resize is semantically acceptable, use linear or area interpolation for continuous magnitudes and nearest
neighbor interpolation for discrete label maps.

Keep `ToTensorV2` at the framework boundary. NumPy pipelines remain easy to inspect, serialize, and test, while the
data loader can convert the final `(H, W, C)` output to the model's channel-first representation.

## Transform safety matrix

| Category | Classification | Reason |
| --- | --- | --- |
| Axis-aware `XYMasking` | Generally safe | It removes bounded time or frequency regions without inventing new coordinates. |
| `TimeMasking` and `FrequencyMasking` | Generally safe for H=frequency, W=time | Their names assume the standard axis convention and use pixel lengths. |
| `TimeReverse` | Representation-dependent | It changes temporal order and may invalidate causal or directional labels. |
| Crop and pad | Representation-dependent | They change duration or frequency coverage and require label-aware bounds. |
| Resize | Representation-dependent | Image interpolation is not domain-correct temporal or frequency resampling. |
| Normalize | Generally safe with dataset statistics | Mean and scale must match the same representation and channel layout. |
| Brightness, contrast, or gamma | Representation-dependent | Their image semantics may not match log, power, or normalized spectral values. |
| Hue, saturation, RGB color transforms | Usually inappropriate | Color channels normally do not encode visual color in scientific arrays. |
| Rotation, perspective, elastic warp | Usually inappropriate | They mix time and frequency coordinates without a signal-domain interpretation. |

## Custom masking templates

Use a custom transform only when `XYMasking` cannot express the sampling policy. The following functional kernels keep
the axis contract visible and operate on every channel together:

```python
import numpy as np


def center_time_dropout(image: np.ndarray, fraction: float, fill: float) -> np.ndarray:
    width = image.shape[1]
    length = int(width * fraction)
    start = (width - length) // 2
    result = image.copy()
    result[:, start : start + length, :] = fill
    return result


def scattered_frequency_dropout(
    image: np.ndarray,
    rows: np.ndarray,
    fill: float,
) -> np.ndarray:
    result = image.copy()
    result[rows, :, :] = fill
    return result
```

Wrap these kernels in an `ImageOnlyTransform` when they need constructor validation, seeded parameter sampling,
serialization, and replay. Sample random row indices in `sample_parameters` through its call-local `sampling` object,
then pass the realized indices to `apply`. Keep NumPy random calls out of `apply`.

For non-center contiguous dropout, prefer `XYMasking` with one X mask. For scattered frequency dropout, record the
sampled row indices so `ReplayCompose` can apply the same rows. A reusable core transform should be proposed separately
if multiple users need that policy.

## Reproducibility

`Compose(seed=...)` provides a private random stream. Two separate pipelines with the same configuration and seed
produce the same result for the same input. Reusing one pipeline advances its stream, which intentionally produces a
new augmentation on each call. Use `ReplayCompose` when another sample must receive the exact realized masks.

Always state four facts with a published recipe: input shape, axis convention, dtype, and numeric range. Those facts
decide whether an image transform preserves the meaning of the underlying signal.
