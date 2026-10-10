"""Deterministic medical acquisition and intensity operations."""

from typing import Any, cast

import cv2
import numpy as np
import torch
from albucore import clip, clipped, float32_io, multiply, preserve_channel_dim, resize3d, warp_affine3d
from albucore import exp as albucore_exp
from scipy import fft

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.type_definitions import NUM_VOLUME_DIMENSIONS, ImageType, VolumeType
from albumentations.core.utils import get_volume_shape

from ._functional_histology import (
    STAIN_MATRICES,
    MacenkoNormalizer,
    SimpleNMF,
    StainNormalizer,
    VahadaneNormalizer,
    apply_he_stain_augmentation,
    get_normalizer,
    get_tissue_mask,
    normalize_vectors,
    order_stains_combined,
    rgb_to_optical_density,
)

_BIAS_FIELD_MIN_CONTIGUOUS_ELEMENTS = 262_144
_K_SPACE_MIN_ANALYTIC_DEPTH = 16
_K_SPACE_MIN_ANALYTIC_PLANE_SIZE = 128
_MOTION_MIN_TORCH_ELEMENTS = 1_048_576
_RICIAN_MIN_TORCH_ELEMENTS = 1_048_576

__all__ = [
    "STAIN_MATRICES",
    "MacenkoNormalizer",
    "SimpleNMF",
    "StainNormalizer",
    "VahadaneNormalizer",
    "anisotropy_3d",
    "apply_he_stain_augmentation",
    "bias_field",
    "generate_bias_field",
    "get_anisotropy_downsample_shape",
    "get_normalizer",
    "get_tissue_mask",
    "ghosting_artifact",
    "gibbs_ringing",
    "k_space_spike",
    "motion_artifact",
    "normalize_vectors",
    "order_stains_combined",
    "rgb_to_optical_density",
    "rician_noise",
]


@float32_io
@clipped
def gibbs_ringing(img: ImageType, retained_fraction: float) -> ImageType:
    """Truncate a channel-last image or volume spectrum to a conjugate-symmetric central box.

    Spatial axes use signed frequency bins with abs(k) <= floor(retained_fraction * length / 2).
    Zeroing slabs of the real half-spectrum avoids shifted spectra and a dense spatial mask.
    Strong truncation transforms only retained last-axis frequencies over the remaining axes.
    The full owned spectrum is reused before the final real inverse, limiting temporary storage.
    """
    spatial_shape = img.shape[:-1]
    cutoffs = tuple(int(retained_fraction * size / 2) for size in spatial_shape)
    if all(cutoff >= size // 2 for cutoff, size in zip(cutoffs, spatial_shape, strict=True)):
        return img

    axes = tuple(range(len(spatial_shape)))
    if retained_fraction <= 0.5:
        spectrum = fft.rfft(img, axis=axes[-1], workers=1)
        retained = spectrum[..., : cutoffs[-1] + 1, :]
        retained = fft.fftn(retained, axes=axes[:-1], workers=1, overwrite_x=True)
        spectrum[..., cutoffs[-1] + 1 :, :] = 0
    else:
        spectrum = fft.rfftn(img, axes=axes, workers=1)
        retained = spectrum
    for axis, (size, cutoff) in enumerate(zip(spatial_shape, cutoffs, strict=True)):
        index = [slice(None)] * img.ndim
        index[axis] = slice(cutoff + 1, None if axis == axes[-1] else size - cutoff)
        retained[tuple(index)] = 0
    reconstructed = fft.ifftn(retained, axes=axes[:-1], workers=1, overwrite_x=True)
    if retained_fraction <= 0.5:
        spectrum[..., : cutoffs[-1] + 1, :] = reconstructed
    else:
        spectrum = reconstructed
    return fft.irfft(spectrum, n=spatial_shape[-1], axis=axes[-1], workers=1)


def bias_field(img: ImageType, coarse_field: np.ndarray, *, is_batch: bool = False) -> ImageType:
    """Upsample a Gaussian log-gain grid, exponentiate it, and multiply image or volume intensities.

    The coarse grid has spatial axes followed by channels. A leading image/volume batch axis in `img`
    shares one generated field; no interpolation occurs across batch or channel axes.
    """
    spatial_shape = img.shape[-coarse_field.ndim : -1]
    field = generate_bias_field(coarse_field, spatial_shape)
    if is_batch and img.dtype == np.uint8:
        result = np.empty_like(img)
        for index, item in enumerate(img):
            result[index] = _apply_bias_gain(item, field)
        return result
    return _apply_bias_gain(img, field)


@float32_io
def _apply_bias_gain(img: ImageType, field: np.ndarray) -> ImageType:
    """Multiply normalized intensities and clip the owned float32 result in place."""
    result = multiply(img, field, inplace=False)
    return clip(result, img.dtype, inplace=True)


def generate_bias_field(coarse_field: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Interpolate a float32 channel-last 2D or 3D log-gain grid and exponentiate it.

    Bilinear/trilinear interpolation preserves the channel count. The input coefficients remain unchanged.
    """
    if coarse_field.shape[:-1] == spatial_shape:
        field = coarse_field.copy()
    elif len(spatial_shape) == 3:
        field = resize3d(coarse_field, (spatial_shape[0], spatial_shape[1], spatial_shape[2]), cv2.INTER_LINEAR)
    else:
        field = fgeometric.resize(coarse_field, (spatial_shape[0], spatial_shape[1]), interpolation=cv2.INTER_LINEAR)
    if field.ndim == 4 and field.size >= _BIAS_FIELD_MIN_CONTIGUOUS_ELEMENTS:
        field = np.ascontiguousarray(field)
    return albucore_exp(field, inplace=True)


@preserve_channel_dim
@float32_io
def rician_noise(
    img: ImageType,
    real_noise: np.ndarray,
    imaginary_noise: np.ndarray,
) -> ImageType:
    """Apply sampled MRI noise fields, fusing large float32 magnitude reconstructions on CPU."""
    result = np.add(img, real_noise)
    if (
        result.size >= _RICIAN_MIN_TORCH_ELEMENTS
        and result.dtype == np.float32
        and imaginary_noise.dtype == np.float32
        and imaginary_noise.flags.writeable
        and all(stride >= 0 for stride in imaginary_noise.strides)
    ):
        working = torch.from_numpy(result)
        imaginary = torch.from_numpy(imaginary_noise)
        with torch.no_grad():
            working.square_().addcmul_(imaginary, imaginary).sqrt_().clamp_(0, 1)
        return result
    np.multiply(result, result, out=result)
    np.add(result, np.square(imaginary_noise), out=result)
    np.sqrt(result, out=result)
    return np.clip(result, 0.0, 1.0, out=result)


@preserve_channel_dim
def k_space_spike(
    img: ImageType,
    spikes: np.ndarray,
    intensity: float,
) -> ImageType:
    """Apply Hermitian MRI Fourier spikes to channel-last images and volumes.

    Each spike adds a real amplitude `intensity * max|F|` at its frequency bin and at the
    conjugate mirror. Large normalized float32 volumes with up to five shared spikes and intensity <= 1
    use cosine waves. Other inputs retain real FFT reconstruction, including the uint8 rounding behavior.
    """
    if intensity == 0 or spikes.size == 0:
        return _k_space_spike_fft(img, [], intensity, 0)

    injections = _spike_injections(spikes)
    if not injections:
        return _k_space_spike_fft(img, injections, intensity, 0)

    if (
        img.dtype == np.float32
        and img.ndim == 4
        and img.shape[0] >= _K_SPACE_MIN_ANALYTIC_DEPTH
        and img.shape[1] >= _K_SPACE_MIN_ANALYTIC_PLANE_SIZE
        and img.shape[2] >= _K_SPACE_MIN_ANALYTIC_PLANE_SIZE
        and spikes.ndim == 2
        and spikes.shape[-1] == 3
        and len(injections) <= 5
        and intensity <= 1
        and all(
            0 <= coord < size for coords, _ in injections for coord, size in zip(coords, img.shape[:3], strict=True)
        )
        and img.size
        and img.min() >= 0
        and img.max() <= 1
    ):
        return _k_space_spike_volume(img, injections, intensity)
    return _k_space_spike_fft(img, injections, intensity, spikes.shape[-1])


def _k_space_spike_volume(
    volume: ImageType,
    injections: list[tuple[tuple[int, ...], int | None]],
    intensity: float,
) -> ImageType:
    """Add shared Fourier spike pairs using one owned output and XY-sized scratch buffers."""
    spatial_shape = volume.shape[:3]
    # The spectral maximum of nonnegative magnitude data is its DC coefficient.
    max_channel_mean = float(volume.mean(axis=(0, 1, 2), dtype=np.float64).max())
    result = volume.copy()
    working = np.empty(spatial_shape[1:], dtype=np.complex64)
    noise = np.empty(spatial_shape[1:], dtype=np.float32)
    for coords, _ in injections:
        factors = [
            np.exp(2j * np.pi * coord * np.arange(size) / size).astype(np.complex64)
            for coord, size in zip(coords, spatial_shape, strict=True)
        ]
        plane = factors[1][:, None] * factors[2][None, :]
        self_conjugate = all((2 * coord) % size == 0 for coord, size in zip(coords, spatial_shape, strict=True))
        amplitude = np.float32(intensity * max_channel_mean * (1 if self_conjugate else 2))
        for depth, depth_factor in enumerate(factors[0]):
            np.multiply(plane, depth_factor, out=working)
            np.multiply(working.real, amplitude, out=noise)
            np.add(result[depth], noise[..., None], out=result[depth])
    return np.clip(result, 0.0, 1.0, out=result)


@float32_io
def _k_space_spike_fft(
    img: ImageType,
    injections: list[tuple[tuple[int, ...], int | None]],
    intensity: float,
    ndim: int,
) -> ImageType:
    """Reconstruct general spike layouts with the original dtype-normalized real FFT operation."""
    if intensity == 0 or not injections:
        return img
    is_batch = img.ndim == ndim + 2
    axes = tuple(range(1, ndim + 1)) if is_batch else tuple(range(ndim))
    axis_sizes = tuple(img.shape[axis] for axis in axes)

    spectrum = cast("np.ndarray", fft.rfftn(np.ascontiguousarray(img), axes=axes, workers=1))
    max_amplitudes = np.abs(spectrum).max(axis=axes, keepdims=True)

    def index_for(coords: tuple[int, ...], channel: int | None = None) -> tuple[Any, ...]:
        prefix = (slice(None),) if is_batch else ()
        return prefix + coords + ((channel,) if channel is not None else ())

    for coords, channel in injections:
        amplitude = _k_space_spike_amplitude(max_amplitudes, intensity, is_batch, img.shape[0], channel)
        for bin_coords in _rfft_spike_bins(coords, axis_sizes):
            spectrum[index_for(bin_coords, channel)] += amplitude

    reconstructed = cast("np.ndarray", fft.irfftn(spectrum, s=axis_sizes, axes=axes, workers=1))
    return np.clip(reconstructed, 0.0, 1.0)


def _rfft_spike_bins(coords: tuple[int, ...], axis_sizes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return the stored rFFT bin or bins representing a real-valued spike pair."""
    stored_coords = _rfft_bin(coords, axis_sizes)
    last_size = axis_sizes[-1]
    is_boundary_plane = coords[-1] == 0 or (last_size % 2 == 0 and coords[-1] == last_size // 2)
    if not is_boundary_plane:
        return (stored_coords,)

    mirror = tuple((size - coord) % size for size, coord in zip(axis_sizes, coords, strict=True))
    stored_mirror = _rfft_bin(mirror, axis_sizes)
    return (stored_coords,) if stored_coords == stored_mirror else (stored_coords, stored_mirror)


def _rfft_bin(coords: tuple[int, ...], axis_sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Map a full-spectrum coordinate to the non-redundant rFFT half-spectrum."""
    if coords[-1] <= axis_sizes[-1] // 2:
        return coords
    return tuple((size - coord) % size for size, coord in zip(axis_sizes, coords, strict=True))


def _k_space_spike_amplitude(
    max_amplitudes: np.ndarray,
    intensity: float,
    is_batch: bool,
    batch_size: int,
    channel: int | None,
) -> float | np.ndarray:
    """Scale each batch item from its own spectrum while sharing the sampled spike."""
    amplitudes = max_amplitudes if channel is None else max_amplitudes[..., channel]
    if not is_batch:
        return intensity * float(amplitudes.max())
    if channel is None:
        return intensity * amplitudes.max(axis=-1).reshape(batch_size, 1)
    return intensity * amplitudes.reshape(batch_size)


def _spike_injections(spikes: np.ndarray) -> list[tuple[tuple[int, ...], int | None]]:
    """Expand shared or per-channel spikes into (coords, channel) pairs."""
    injections: list[tuple[tuple[int, ...], int | None]] = []
    if spikes.ndim == 2:
        injections.extend((tuple(int(index) for index in spike), None) for spike in spikes)
    else:
        for channel in range(spikes.shape[0]):
            injections.extend((tuple(int(index) for index in spike), channel) for spike in spikes[channel])
    return injections


@float32_io
@clipped
def ghosting_artifact(
    volume: ImageType,
    num_ghosts: int,
    intensity: float,
    axis: int,
    restore: float,
) -> ImageType:
    """Attenuate conjugate-symmetric periodic frequency planes outside a protected central band.

    The plane mask depends only on the selected spatial frequency. Transforms of the other axes cancel,
    so one axis-wise real FFT implements the full 3D spectral-plane filter without full-volume complex FFTs.
    """
    length = volume.shape[axis]
    cutoff = int(restore * length / 2)
    start = (cutoff // num_ghosts + 1) * num_ghosts
    if intensity == 0 or start > length // 2:
        return volume

    spectrum = fft.rfft(volume, axis=axis, workers=1)
    index = [slice(None)] * volume.ndim
    index[axis] = slice(start, None, num_ghosts)
    spectrum[tuple(index)] *= 1 - intensity
    reconstructed = fft.irfft(spectrum, n=length, axis=axis, workers=1, overwrite_x=True)
    return np.abs(reconstructed, out=reconstructed)


@float32_io
@clipped
def motion_artifact(
    volume: VolumeType,
    matrices: np.ndarray,
    boundaries: tuple[int, ...],
    axis: int,
    interpolation: int,
) -> VolumeType:
    """Combine sequential centred k-space segments from absolute rigid motion states.

    The first segment is stationary. Each subsequent segment uses one forward `(x, y, z)` matrix.
    Segment masks depend only on the acquisition axis, so the other spatial FFTs cancel with their
    inverses. A full complex FFT along that axis preserves asymmetric, non-Hermitian segments.
    Large writable single-channel volumes use CPU Tensor FFTs. Only one moved volume and its spectrum
    are retained at a time, and the combined spectrum is reused for the inverse.
    """
    identity = np.eye(4, dtype=np.float32)
    if matrices.size == 0 or np.all(matrices == identity):
        return volume

    spatial_shape = volume.shape[:3]
    use_torch = (
        volume.size >= _MOTION_MIN_TORCH_ELEMENTS
        and volume.shape[-1] == 1
        and volume.flags.writeable
        and all(stride >= 0 for stride in volume.strides)
    )
    if use_torch:
        with torch.no_grad():
            spectrum = torch.fft.fft(torch.from_numpy(volume), dim=axis).numpy()
    else:
        spectrum = fft.fft(volume, axis=axis, workers=1)
    for matrix, begin, end in zip(matrices, boundaries, (*boundaries[1:], volume.shape[axis]), strict=True):
        if np.array_equal(matrix, identity):
            continue
        moved = warp_affine3d(
            volume,
            matrix,
            spatial_shape,
            interpolation=interpolation,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0,
        )
        if use_torch:
            with torch.no_grad():
                moved_spectrum = torch.fft.fft(torch.from_numpy(moved), dim=axis).numpy()
        else:
            moved_spectrum = fft.fft(moved, axis=axis, workers=1)
        _replace_motion_segment(spectrum, moved_spectrum, axis, begin, end)
        del moved, moved_spectrum

    if use_torch:
        with torch.no_grad():
            working = torch.from_numpy(spectrum)
            torch.fft.ifft(working, dim=axis, out=working)
        reconstructed = spectrum
    else:
        reconstructed = fft.ifft(spectrum, axis=axis, workers=1, overwrite_x=True)
    return np.abs(reconstructed)


def _replace_motion_segment(
    spectrum: np.ndarray,
    moved_spectrum: np.ndarray,
    axis: int,
    begin: int,
    end: int,
) -> None:
    """Write a centred acquisition interval without allocating a shifted spectrum."""
    length = spectrum.shape[axis]
    start = (begin + (length + 1) // 2) % length
    stop = start + end - begin
    index = [slice(None)] * spectrum.ndim
    for lower, upper in ((start, min(stop, length)), (0, max(0, stop - length))):
        index[axis] = slice(lower, upper)
        spectrum[tuple(index)] = moved_spectrum[tuple(index)]


def get_anisotropy_downsample_shape(
    spatial_shape: tuple[int, int, int],
    axes: tuple[int, ...],
    downscale_factor: float,
) -> tuple[int, int, int]:
    """Derive an anisotropic intermediate shape by scaling selected axes while retaining non-selected axes, ensuring
    every requested spatial dimension remains valid.
    """
    return cast(
        "tuple[int, int, int]",
        tuple(
            max(1, round(axis_size / downscale_factor)) if axis_index in axes else axis_size
            for axis_index, axis_size in enumerate(spatial_shape)
        ),
    )


def anisotropy_3d(
    volume: VolumeType | torch.Tensor,
    downsample_shape: tuple[int, int, int],
    antialias: bool,
) -> VolumeType | torch.Tensor:
    """Simulate thicker or lower-resolution volume acquisition by shrinking selected spatial axes and restoring the
    original shape for 3D robustness training.

    Both routes delegate to Albucore `resize3d`, which resizes only spatial axes and preserves the input representation.
    NumPy applies antialiasing while shrinking; PyTorch does not yet provide 5D trilinear antialiasing, so Tensor input
    uses the non-antialiased native route until upstream support is available.
    """
    source_shape = get_volume_shape(volume)
    if source_shape == downsample_shape:
        return volume

    is_channel_less_numpy_volume = isinstance(volume, np.ndarray) and volume.ndim == NUM_VOLUME_DIMENSIONS - 1
    working_volume = volume[..., np.newaxis] if is_channel_less_numpy_volume else volume
    downsampled = resize3d(
        working_volume,
        downsample_shape,
        interpolation=cv2.INTER_LINEAR,
        antialias=antialias and not isinstance(volume, torch.Tensor),
    )
    restored = resize3d(downsampled, source_shape, interpolation=cv2.INTER_LINEAR)
    return restored[..., 0] if is_channel_less_numpy_volume else restored
