"""Histology stain normalization functional helpers."""

from __future__ import annotations

from typing import Literal

from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    ImageType,
    clipped,
    cv2,
    float32_io,
    np,
    reduce_sum,
)


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
    cv2.multiply(pixel_matrix, 1.0 / max_value, dst=pixel_matrix)
    cv2.max(pixel_matrix, eps, dst=pixel_matrix)
    cv2.log(pixel_matrix, dst=pixel_matrix)
    cv2.multiply(pixel_matrix, -1.0, dst=pixel_matrix)
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
        self.stain_matrix_target = None

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

        # Initialize concentrations based on projection onto initial colors
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

                # Ensure non-negativity
                stain_concentrations = np.maximum(stain_concentrations, 0)

                # Update colors
                numerator = stain_concentrations.T @ optical_density
                denominator = (stain_concentrations.T @ stain_concentrations) @ stain_colors
                stain_colors *= numerator / (denominator + eps)

                # Ensure non-negativity and normalize
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
    blue_ratio = stain_colors[:, 2] / (reduce_sum(stain_colors, axis=1) + 1e-6)
    red_ratio = stain_colors[:, 0] / (reduce_sum(stain_colors, axis=1) + 1e-6)

    # Combine scores
    # High angle and high blue ratio indicates Hematoxylin
    # Low angle and high red ratio indicates Eosin
    scores = angles * blue_ratio - red_ratio

    hematoxylin_idx = np.argmax(scores)
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
        # Step 1: Convert RGB to optical density (OD) space
        optical_density = rgb_to_optical_density(img)

        # Step 2: Remove background pixels
        od_threshold = 0.05
        threshold_mask = (optical_density > od_threshold).any(axis=1)
        tissue_density = optical_density[threshold_mask]

        if len(tissue_density) < 1:
            raise ValueError(f"No tissue pixels found (threshold={od_threshold})")

        # Step 3: Compute covariance matrix
        tissue_density = np.ascontiguousarray(tissue_density, dtype=np.float32)
        od_covariance = cv2.calcCovarMatrix(
            tissue_density,
            None,
            cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE,
        )[0]

        # Step 4: Get principal components
        eigenvalues, eigenvectors = cv2.eigen(od_covariance)[1:]
        idx = np.argsort(eigenvalues.ravel())[-2:]
        principal_eigenvectors = np.ascontiguousarray(eigenvectors[:, idx], dtype=np.float32)

        # Step 5: Project onto eigenvector plane
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

        # Step 6: Find angles of extreme points
        polar_angles = np.arctan2(
            plane_coordinates[:, 1],
            plane_coordinates[:, 0],
        )

        # Get robust angle estimates
        hematoxylin_angle = np.percentile(polar_angles, 100 - angular_percentile)
        eosin_angle = np.percentile(polar_angles, angular_percentile)

        # Step 7: Convert angles back to RGB space
        hem_cos, hem_sin = np.cos(hematoxylin_angle), np.sin(hematoxylin_angle)
        eos_cos, eos_sin = np.cos(eosin_angle), np.sin(eosin_angle)

        angle_to_vector = np.array(
            [[hem_cos, hem_sin], [eos_cos, eos_sin]],
            dtype=np.float32,
        )

        # Ensure both matrices have the same data type for cv2.gemm
        principal_eigenvectors_t = np.ascontiguousarray(principal_eigenvectors.T, dtype=np.float32)
        stain_vectors = cv2.gemm(
            angle_to_vector,
            principal_eigenvectors_t,
            1,
            None,
            0,
        )

        # Step 8: Ensure non-negativity by taking absolute values
        stain_vectors = np.abs(stain_vectors)

        # Step 9: Normalize vectors to unit length
        stain_vectors = stain_vectors / np.sqrt(reduce_sum(stain_vectors**2, axis=1, keepdims=True) + epsilon)

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
    # Convert to grayscale using RGB weights: R*0.299 + G*0.587 + B*0.114
    luminosity = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

    # Tissue is darker, so we want pixels below threshold
    mask = luminosity < threshold

    return mask.reshape(-1)


@clipped
@float32_io
def apply_he_stain_augmentation(
    img: ImageType,
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    augment_background: bool,
) -> ImageType:
    """Apply HE (hematoxylin-eosin) stain augmentation. Shifts stain concentrations;
    params control strength. Used for histology augmentation. Returns RGB image.

    This function applies HE stain augmentation to an image.

    Args:
        img (ImageType): Input image.
        stain_matrix (np.ndarray): Stain matrix.
        scale_factors (np.ndarray): Scale factors.
        shift_values (np.ndarray): Shift values.
        augment_background (bool): Whether to augment the background.

    Returns:
        ImageType: Augmented image.

    """
    # Step 1: Convert RGB to optical density space
    optical_density = rgb_to_optical_density(img)

    # Step 2: Calculate stain concentrations using regularized pseudo-inverse
    stain_matrix = np.ascontiguousarray(stain_matrix, dtype=np.float32)

    # Add small regularization term for numerical stability
    regularization = 1e-6

    # Suppress numerical warnings for edge cases with extreme optical densities
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        stain_correlation = stain_matrix @ stain_matrix.T + regularization * np.eye(2)
        density_projection = stain_matrix @ optical_density.T

        try:
            # Solve for stain concentrations
            stain_concentrations = np.linalg.solve(stain_correlation, density_projection).T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if direct solve fails
            stain_concentrations = np.linalg.lstsq(
                stain_matrix.T,
                optical_density,
                rcond=regularization,
            )[0].T

        # Step 3: Apply concentration adjustments
        if not augment_background:
            # Only modify tissue regions
            tissue_mask = get_tissue_mask(img).reshape(-1)
            stain_concentrations[tissue_mask] = stain_concentrations[tissue_mask] * scale_factors + shift_values
        else:
            # Modify all pixels
            stain_concentrations = stain_concentrations * scale_factors + shift_values

        # Step 4: Reconstruct RGB image
        optical_density_result = stain_concentrations @ stain_matrix
        rgb_result = np.exp(-optical_density_result)

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
