"""Histology stain normalization functional helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from albucore import exp as albucore_exp

from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    ImageType,
    clipped,
    cv2,
    float32_io,
    np,
    reduce_sum,
)

_Cv2InPlaceOp = Callable[..., np.ndarray]
_SPARSE_TISSUE_MAX_FRACTION = 0.4


def rgb_to_optical_density(img: ImageType, eps: float = 1e-6) -> np.ndarray:
    """Convert RGB image to optical density (-log10). eps avoids log(0). Expects uint8 or float32 in
    [0,1]. Returns (N*H*W, 3) float64. For stain normalization.

    This function converts an RGB image to optical density.

    Args:
        img (ImageType): Input image.
        eps (float): Epsilon value.

    Returns:
        np.ndarray: Optical density image.

    """
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    pixel_matrix = np.ascontiguousarray(img.reshape(-1, 3)).astype(np.float32, copy=True)
    multiply = cast("_Cv2InPlaceOp", cv2.multiply)
    max_op = cast("_Cv2InPlaceOp", cv2.max)
    multiply(pixel_matrix, 1.0 / max_value, dst=pixel_matrix)
    max_op(pixel_matrix, eps, dst=pixel_matrix)
    cv2.log(pixel_matrix, dst=pixel_matrix)
    multiply(pixel_matrix, -1.0, dst=pixel_matrix)
    return pixel_matrix


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length (L2). Axis and dtype preserved; 1D or 2D. For stain
    normalization (e.g. Macenko) stain vector normalization.

    This function normalizes vectors.

    Args:
        vectors (np.ndarray): Vectors to normalize.

    Returns:
        np.ndarray: Normalized vectors.

    """
    norms = np.sqrt(reduce_sum(vectors**2, axis=1, keepdims=True))
    return vectors / norms


def get_normalizer(method: Literal["vahadane", "macenko"]) -> StainNormalizer:
    """Get stain normalizer based on method ('vahadane' or 'macenko'). Returns
    VahadaneNormalizer or MacenkoNormalizer instance for histology stain norm.

    This function gets a stain normalizer based on a method.

    Args:
        method (Literal['vahadane', 'macenko']): Method to use for stain normalization.

    Returns:
        StainNormalizer: Stain normalizer.

    """
    return VahadaneNormalizer() if method == "vahadane" else MacenkoNormalizer()


class StainNormalizer:
    """Base class for stain normalizers. Subclass and implement fit/transform for
    histology stain normalization (e.g. Vahadane, Macenko).
    """

    def __init__(self) -> None:
        self.stain_matrix_target: np.ndarray | None = None

    def fit(self, img: ImageType) -> None:
        """Fit the stain normalizer to a reference image. Learns stain matrix from img; call transform
        on target images after. Subclass implements the actual extraction.

        This function fits the stain normalizer to an image.

        Args:
            img (ImageType): Input image.

        """
        raise NotImplementedError


class SimpleNMF:
    """Simple NMF for histology stain separation. Factorizes OD matrix into stain basis and
    concentrations. Iterative multiplicative updates, non-negativity.

    This class implements a simplified version of the Non-negative Matrix Factorization algorithm
    specifically designed for separating Hematoxylin and Eosin (H&E) stains in histopathology images.
    It is used as part of the Vahadane stain normalization method.

    The algorithm decomposes optical density values of H&E stained images into stain color appearances
    (the stain color vectors) and stain concentrations (the density of each stain at each pixel).

    The implementation uses an iterative multiplicative update approach that preserves non-negativity
    constraints, which are physically meaningful for stain separation as concentrations and
    absorption coefficients cannot be negative.

    This implementation is optimized for stability by:
    1. Initializing with standard H&E reference colors from Ruifrok
    2. Using normalized projection for initial concentrations
    3. Applying careful normalization to avoid numerical issues

    Args:
        n_iter (int): Number of iterations for the NMF algorithm. Default: 100

    References:
        - Vahadane, A., et al. (2016): Structure-preserving color normalization and
          sparse stain separation for histological images. IEEE Transactions on
          Medical Imaging, 35(8), 1962-1971.
        - Ruifrok, A. C., & Johnston, D. A. (2001): Quantification of histochemical
          staining by color deconvolution. Analytical and Quantitative Cytology and
          Histology, 23(4), 291-299.

    """

    def __init__(self, n_iter: int = 100):
        self.n_iter = n_iter
        # Initialize with standard H&E colors from Ruifrok
        self.initial_colors = np.array(
            [
                [0.644211, 0.716556, 0.266844],  # Hematoxylin
                [0.092789, 0.954111, 0.283111],  # Eosin
            ],
            dtype=np.float32,
        )

    def fit_transform(self, optical_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit the NMF model to optical density matrix. Learns stain basis and
        concentrations; used internally by VahadaneNormalizer for stain separation.

        This function fits the NMF model to optical density.

        Args:
            optical_density (np.ndarray): Optical density image.

        Returns:
            tuple[np.ndarray, np.ndarray]: Stain concentrations and stain colors.

        """
        # Start with known H&E colors
        stain_colors = self.initial_colors.copy()

        # This gives us a physically meaningful starting point
        stain_colors_normalized = normalize_vectors(stain_colors)

        # Suppress numerical warnings for edge cases (handled by eps)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            stain_concentrations = np.maximum(optical_density @ stain_colors_normalized.T, 0)

            # Iterative updates with careful normalization
            eps = 1e-6
            for _ in range(self.n_iter):
                # Update concentrations
                numerator = optical_density @ stain_colors.T
                denominator = stain_concentrations @ (stain_colors @ stain_colors.T)
                stain_concentrations *= numerator / (denominator + eps)

                stain_concentrations = np.maximum(stain_concentrations, 0)

                # Update colors
                numerator = stain_concentrations.T @ optical_density
                denominator = (stain_concentrations.T @ stain_concentrations) @ stain_colors
                stain_colors *= numerator / (denominator + eps)

                stain_colors = np.maximum(stain_colors, 0)
                stain_colors = normalize_vectors(stain_colors)

        return stain_concentrations, stain_colors


def order_stains_combined(stain_colors: np.ndarray) -> tuple[int, int]:
    """Order stains using a combination of methods (angular and spectral).
    Returns ordered stain matrix for consistent H/E ordering.

    This combines both angular information and spectral characteristics
    for more robust identification.

    Args:
        stain_colors (np.ndarray): Stain colors.

    Returns:
        tuple[int, int]: Hematoxylin and eosin indices.

    """
    # Normalize stain vectors
    stain_colors = normalize_vectors(stain_colors)

    # Calculate angles (Macenko)
    angles = np.mod(np.arctan2(stain_colors[:, 1], stain_colors[:, 0]), np.pi)

    # Calculate spectral ratios (Ruifrok)
    stain_sums = np.asarray(reduce_sum(stain_colors, axis=1), dtype=np.float32) + 1e-6
    blue_ratio = stain_colors[:, 2] / stain_sums
    red_ratio = stain_colors[:, 0] / stain_sums

    # Combine scores
    # High angle and high blue ratio indicates Hematoxylin
    # Low angle and high red ratio indicates Eosin
    scores = angles * blue_ratio - red_ratio

    hematoxylin_idx = int(np.argmax(scores))
    eosin_idx = 1 - hematoxylin_idx

    return hematoxylin_idx, eosin_idx


class VahadaneNormalizer(StainNormalizer):
    """Vahadane stain normalizer for histopathology. NMF-based stain separation;
    fit on reference image, then transform. Used for H&E normalization.

    This class implements the "Structure-Preserving Color Normalization and Sparse Stain Separation
    for Histological Images" method proposed by Vahadane et al. The technique uses Non-negative
    Matrix Factorization (NMF) to separate Hematoxylin and Eosin (H&E) stains in histopathology
    images and then normalizes them to a target standard.

    The Vahadane method is particularly effective for histology image normalization because:
    1. It maintains tissue structure during color normalization
    2. It performs sparse stain separation, reducing color bleeding
    3. It adaptively estimates stain vectors from each image
    4. It preserves biologically relevant information

    This implementation uses SimpleNMF as its core matrix factorization algorithm to extract
    stain color vectors (appearance matrix) and concentration matrices from optical
    density-transformed images. It identifies the Hematoxylin and Eosin stains by their
    characteristic color profiles and spatial distribution.

    References:
        Vahadane, et al., 2016: Structure-preserving color normalization
        and sparse stain separation for histological images. IEEE transactions on medical imaging,
        35(8), pp.1962-1971.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> from albumentations.augmentations.pixel import functional as F
        >>> import cv2
        >>>
        >>> # Load source and target images (H&E stained histopathology)
        >>> source_img = cv2.imread('source_image.png')
        >>> source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        >>> target_img = cv2.imread('target_image.png')
        >>> target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        >>>
        >>> # Create and fit the normalizer to the target image
        >>> normalizer = F.VahadaneNormalizer()
        >>> normalizer.fit(target_img)
        >>>
        >>> # Normalize the source image to match the target's stain characteristics
        >>> normalized_img = normalizer.transform(source_img)

    """

    def fit(self, img: ImageType) -> None:
        """Fit the Vahadane stain normalizer to a reference image. Runs NMF on OD
        matrix; call transform on target images for normalization.

        This function fits the Vahadane stain normalizer to an image.

        Args:
            img (ImageType): Input image.

        """
        optical_density = rgb_to_optical_density(img)

        nmf = SimpleNMF(n_iter=100)
        _, stain_colors = nmf.fit_transform(optical_density)

        # Use combined method for robust stain ordering
        hematoxylin_idx, eosin_idx = order_stains_combined(stain_colors)

        self.stain_matrix_target = np.array(
            [
                stain_colors[hematoxylin_idx],
                stain_colors[eosin_idx],
            ],
        )


class MacenkoNormalizer(StainNormalizer):
    """Macenko stain normalizer with optimized computations. SVD-based stain
    separation; fit on reference, then transform. Used for H&E normalization.
    """

    def __init__(self, angular_percentile: float = 99):
        super().__init__()
        self.angular_percentile = angular_percentile

    def fit(self, img: ImageType, angular_percentile: float = 99) -> None:
        """Fit the Macenko stain normalizer to a reference image. SVD-based;
        call transform on target images for H&E normalization.

        This function fits the Macenko stain normalizer to an image.

        Args:
            img (ImageType): Input image.
            angular_percentile (float): Angular percentile.

        """
        optical_density = rgb_to_optical_density(img)

        od_threshold = 0.05
        threshold_mask = (optical_density > od_threshold).any(axis=1)
        tissue_density = optical_density[threshold_mask]

        if len(tissue_density) < 1:
            raise ValueError(f"No tissue pixels found (threshold={od_threshold})")

        tissue_density = np.ascontiguousarray(tissue_density, dtype=np.float32)
        covariance_mean = np.empty((0,), dtype=np.float32)
        od_covariance = cast(
            "np.ndarray",
            cv2.calcCovarMatrix(
                tissue_density,
                covariance_mean,
                cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE,
            )[0],
        )

        eigenvalues, eigenvectors = cv2.eigen(od_covariance)[1:]
        idx = np.argsort(eigenvalues.ravel())[-2:]
        principal_eigenvectors = np.ascontiguousarray(eigenvectors[:, idx], dtype=np.float32)

        # Add small epsilon to avoid numerical instability
        epsilon = 1e-8
        if np.any(np.abs(principal_eigenvectors) < epsilon):
            # Regularize near-zero entries by assigning ±ε based on original sign
            principal_eigenvectors = np.where(
                np.abs(principal_eigenvectors) < epsilon,
                np.where(principal_eigenvectors < 0, -epsilon, epsilon),
                principal_eigenvectors,
            )

        # Add small epsilon to tissue_density to avoid numerical issues
        safe_tissue_density = tissue_density + epsilon

        # Suppress numerical warnings for edge cases with extreme optical densities
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            plane_coordinates = safe_tissue_density @ principal_eigenvectors

        polar_angles = np.arctan2(
            plane_coordinates[:, 1],
            plane_coordinates[:, 0],
        )

        hematoxylin_angle = np.percentile(polar_angles, 100 - angular_percentile)
        eosin_angle = np.percentile(polar_angles, angular_percentile)

        hem_cos, hem_sin = np.cos(hematoxylin_angle), np.sin(hematoxylin_angle)
        eos_cos, eos_sin = np.cos(eosin_angle), np.sin(eosin_angle)

        angle_to_vector = np.array(
            [[hem_cos, hem_sin], [eos_cos, eos_sin]],
            dtype=np.float32,
        )

        principal_eigenvectors_t = np.ascontiguousarray(principal_eigenvectors.T, dtype=np.float32)
        empty_matrix = np.empty((0,), dtype=np.float32)
        stain_vectors = cast(
            "np.ndarray",
            cv2.gemm(
                angle_to_vector,
                principal_eigenvectors_t,
                1.0,
                empty_matrix,
                0.0,
            ),
        )

        stain_vectors = np.abs(stain_vectors)

        stain_norms = np.sqrt(
            np.asarray(reduce_sum(stain_vectors**2, axis=1, keepdims=True), dtype=np.float32) + epsilon,
        )
        stain_vectors = stain_vectors / stain_norms

        # Step 10: Order vectors as [hematoxylin, eosin]
        self.stain_matrix_target = stain_vectors if stain_vectors[0, 0] > stain_vectors[1, 0] else stain_vectors[::-1]


def get_tissue_mask(img: ImageType, threshold: float = 0.85) -> np.ndarray:
    """Get tissue mask from image (exclude background). threshold for intensity-based masking of
    non-tissue. Returns 1D bool mask.

    Args:
        img (ImageType): Input image
        threshold (float): Threshold for tissue detection. Default: 0.85

    Returns:
        np.ndarray: Binary mask where True indicates tissue regions

    """
    luminosity = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

    # Tissue is darker, so we want pixels below threshold
    mask = luminosity < threshold

    return mask.reshape(-1)


def _build_stain_affine_matrix(
    stain_matrix: np.ndarray,
    deconvolution_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
) -> np.ndarray:
    affine_matrix = np.empty((3, 4), dtype=np.float32)
    affine_matrix[:, :3] = stain_matrix.T @ (scale_factors[:, None] * deconvolution_matrix)
    affine_matrix[:, 3] = stain_matrix.T @ shift_values
    return affine_matrix


def _apply_stain_affine(optical_density: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    result = cv2.transform(optical_density.reshape(-1, 1, 3), affine_matrix)
    result *= -1.0
    return albucore_exp(result, inplace=True).reshape(-1, 3)


def _validate_stain_augmentation_inputs(
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    residual_mode: Literal["project", "preserve", "augment"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stain_matrix = np.ascontiguousarray(stain_matrix, dtype=np.float32)
    if stain_matrix.shape not in {(2, 3), (3, 3)}:
        raise ValueError(f"stain_matrix must have shape (2, 3) or (3, 3), got {stain_matrix.shape}")
    if not np.isfinite(stain_matrix).all():
        raise ValueError("stain_matrix must contain only finite values")
    if residual_mode not in {"project", "preserve", "augment"}:
        raise ValueError(f"Invalid residual_mode: {residual_mode}")
    if stain_matrix.shape == (3, 3):
        if residual_mode == "project":
            raise ValueError("A full stain basis is incompatible with residual_mode='project'")
        if np.linalg.matrix_rank(stain_matrix) < 3:
            raise ValueError("stain_matrix rows must be linearly independent")

    component_count = 2 if residual_mode == "project" else 3
    scale_factors = np.asarray(scale_factors)
    shift_values = np.asarray(shift_values)
    if scale_factors.shape != (component_count,):
        raise ValueError(f"scale_factors must have shape ({component_count},), got {scale_factors.shape}")
    if shift_values.shape != (component_count,):
        raise ValueError(f"shift_values must have shape ({component_count},), got {shift_values.shape}")
    return stain_matrix, scale_factors, shift_values


def _build_residual_stain_matrix(stain_matrix: np.ndarray, tolerance: float) -> np.ndarray | None:
    residual_vector = np.cross(stain_matrix[0], stain_matrix[1])
    residual_norm = np.linalg.norm(residual_vector)
    stain_norm_product = np.linalg.norm(stain_matrix[0]) * np.linalg.norm(stain_matrix[1])
    if (
        not np.isfinite(residual_norm)
        or not np.isfinite(stain_norm_product)
        or residual_norm <= tolerance * stain_norm_product
    ):
        return None

    residual_vector /= residual_norm
    reconstruction_matrix = np.empty((3, 3), dtype=np.float32)
    reconstruction_matrix[:2] = stain_matrix
    reconstruction_matrix[2] = residual_vector
    return reconstruction_matrix


def _apply_project_stain_affine(
    img: ImageType,
    optical_density: np.ndarray,
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    augment_background: bool,
    regularization: float,
) -> np.ndarray:
    stain_correlation = stain_matrix @ stain_matrix.T + regularization * np.eye(2)
    deconvolution_matrix = np.linalg.solve(stain_correlation, stain_matrix)
    augmented_affine = _build_stain_affine_matrix(
        stain_matrix,
        deconvolution_matrix,
        scale_factors,
        shift_values,
    )
    if augment_background:
        return _apply_stain_affine(optical_density, augmented_affine)

    tissue_mask = get_tissue_mask(img)
    if np.all(tissue_mask):
        return _apply_stain_affine(optical_density, augmented_affine)

    base_affine = _build_stain_affine_matrix(
        stain_matrix,
        deconvolution_matrix,
        np.ones_like(scale_factors),
        np.zeros_like(shift_values),
    )
    result = _apply_stain_affine(optical_density, base_affine)
    if np.any(tissue_mask):
        result[tissue_mask] = _apply_stain_affine(optical_density[tissue_mask], augmented_affine)
    return result


def _apply_residual_stain_affine(
    img: ImageType,
    optical_density: np.ndarray,
    reconstruction_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    augment_background: bool,
) -> np.ndarray:
    deconvolution_matrix = np.linalg.inv(reconstruction_matrix).T
    augmented_affine = _build_stain_affine_matrix(
        reconstruction_matrix,
        deconvolution_matrix,
        scale_factors,
        shift_values,
    )
    if augment_background:
        return _apply_stain_affine(optical_density, augmented_affine)

    tissue_mask = get_tissue_mask(img)
    tissue_count = np.count_nonzero(tissue_mask)
    image_matrix = img.reshape(-1, 3)
    if tissue_count == tissue_mask.size:
        return _apply_stain_affine(optical_density, augmented_affine)
    if tissue_count <= tissue_mask.size * _SPARSE_TISSUE_MAX_FRACTION:
        result = image_matrix.copy()
        if tissue_count:
            result[tissue_mask] = _apply_stain_affine(optical_density[tissue_mask], augmented_affine)
        return result

    result = _apply_stain_affine(optical_density, augmented_affine)
    background_mask = ~tissue_mask
    result[background_mask] = image_matrix[background_mask]
    return result


@clipped
@float32_io
def apply_he_stain_augmentation(
    img: ImageType,
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    augment_background: bool,
    residual_mode: Literal["project", "preserve", "augment"] = "project",
) -> ImageType:
    """Perturb stain concentrations in optical-density space with a two-stain H&E basis or an explicit
    three-stain basis for histology augmentation.

    A two-row matrix defines hematoxylin and eosin. Residual modes derive a normalized third vector from their cross
    product. A three-row matrix supplies the third stain directly for full-basis operations such as HED jitter.

    Args:
        img (ImageType): RGB image with values in the dtype's standard range.
        stain_matrix (np.ndarray): Optical-density basis with shape `(2, 3)` for H&E or `(3, 3)` for an explicit
            three-stain basis. A three-row matrix must have full rank.
        scale_factors (np.ndarray): Per-component multiplicative factors. Supply two values for `"project"` and
            three values for the residual modes.
        shift_values (np.ndarray): Per-component additive shifts, with the same length as `scale_factors`.
        augment_background (bool): Whether to adjust concentrations outside the tissue mask.
        residual_mode (Literal['project', 'preserve', 'augment']): Third-component policy. `"project"` reconstructs
            from H&E only and requires a two-row matrix. `"preserve"` keeps the derived or explicit third component
            unchanged. `"augment"` adjusts all three components.

    Returns:
        ImageType: RGB image with the input shape and dtype.

    Raises:
        ValueError: If the stain matrix, residual mode, or adjustment-vector shapes are invalid.

    Note:
        - `"project"` solves the regularized two-stain model used by earlier releases.
        - With a two-row matrix, the residual modes use `R = normalize(cross(H, E))` and solve the full H&E+R basis.
        - With a three-row matrix, the residual modes use the supplied third row without deriving a replacement.
        - If H and E in a two-row matrix are nearly collinear, residual modes fall back to the regularized
          `"project"` model.

    Examples:
        >>> import numpy as np
        >>> from albumentations.augmentations.pixel import functional as fpixel
        >>> image = np.full((8, 8, 3), 0.5, dtype=np.float32)
        >>> stain_matrix = np.array(
        ...     [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
        ...     dtype=np.float32,
        ... )
        >>> result = fpixel.apply_he_stain_augmentation(
        ...     image,
        ...     stain_matrix,
        ...     scale_factors=np.array([1.05, 0.95, 1.02]),
        ...     shift_values=np.array([0.02, -0.01, 0.01]),
        ...     augment_background=True,
        ...     residual_mode="augment",
        ... )

    """
    stain_matrix, scale_factors, shift_values = _validate_stain_augmentation_inputs(
        stain_matrix,
        scale_factors,
        shift_values,
        residual_mode,
    )
    regularization = 1e-6
    if residual_mode == "project":
        reconstruction_matrix = None
    elif stain_matrix.shape == (3, 3):
        reconstruction_matrix = stain_matrix
    else:
        reconstruction_matrix = _build_residual_stain_matrix(stain_matrix, regularization)
    optical_density = rgb_to_optical_density(img)

    # Suppress numerical warnings for edge cases with extreme optical densities
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if reconstruction_matrix is None:
            component_slice = slice(None) if residual_mode == "project" else slice(2)
            rgb_result = _apply_project_stain_affine(
                img,
                optical_density,
                stain_matrix,
                scale_factors[component_slice],
                shift_values[component_slice],
                augment_background,
                regularization,
            )
        else:
            rgb_result = _apply_residual_stain_affine(
                img,
                optical_density,
                reconstruction_matrix,
                scale_factors,
                shift_values,
                augment_background,
            )

    return rgb_result.reshape(img.shape)


__all__ = [
    "MacenkoNormalizer",
    "SimpleNMF",
    "StainNormalizer",
    "VahadaneNormalizer",
    "apply_he_stain_augmentation",
    "get_normalizer",
    "get_tissue_mask",
    "normalize_vectors",
    "order_stains_combined",
    "rgb_to_optical_density",
]
