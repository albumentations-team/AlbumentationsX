"""Module containing utility functions for augmentation operations.

This module provides a collection of helper functions and utilities used throughout
the augmentation pipeline. It includes functions for image loading, type checking,
error handling, mathematical operations, and decorators that add functionality to
other functions in the codebase. These utilities help ensure consistent behavior
and simplify common operations across different augmentation transforms.
"""

from __future__ import annotations

import functools
from functools import wraps
from typing import Any, Callable, TypeVar, cast

import cv2
import numpy as np
from albucore.utils import (
    is_grayscale_image,
    is_multispectral_image,
    is_rgb_image,
)
from typing_extensions import Concatenate, ParamSpec

from albumentations.core.keypoints_utils import angle_to_2pi_range

__all__ = [
    "angle_2pi_range",
    "non_rgb_error",
]

P = ParamSpec("P")
T = TypeVar("T", bound=np.ndarray)
F = TypeVar("F", bound=Callable[..., Any])


def angle_2pi_range(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Decorator to normalize angle values to the range [0, 2π).

    This decorator wraps a function that processes keypoints, ensuring that
    angle values (stored in the 4th column, index 3) are normalized to the
    range [0, 2π) after the wrapped function executes.

    Args:
        func (Callable): Function that processes keypoints and returns a numpy array.
            The function should take a keypoints array as its first parameter.

    Returns:
        Callable: Wrapped function that normalizes angles after processing keypoints.

    """

    @wraps(func)
    def wrapped_function(keypoints: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        result = func(keypoints, *args, **kwargs)
        if len(result) > 0 and result.shape[1] > 3:
            result[:, 3] = angle_to_2pi_range(result[:, 3])
        return result

    return wrapped_function


def non_rgb_error(image: np.ndarray) -> None:
    """Check if the input image is RGB and raise a ValueError if it's not.

    This function is used to ensure that certain transformations are only applied to
    RGB images. It provides helpful error messages for grayscale and multi-spectral images.

    Args:
        image (np.ndarray): The input image to check. Expected to be a numpy array
                            representing an image.

    Raises:
        ValueError: If the input image is not an RGB image (i.e., does not have exactly 3 channels).
                    The error message includes specific instructions for grayscale images
                    and a note about incompatibility with multi-spectral images.

    Note:
        - RGB images are expected to have exactly 3 channels.
        - Grayscale images (1 channel) will trigger an error with conversion instructions.
        - Multi-spectral images (more than 3 channels) will trigger an error stating incompatibility.

    Examples:
        >>> import numpy as np
        >>> rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> non_rgb_error(rgb_image)  # No error raised
        >>>
        >>> grayscale_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> non_rgb_error(grayscale_image)  # Raises ValueError with conversion instructions
        >>>
        >>> multispectral_image = np.random.randint(0, 256, (100, 100, 5), dtype=np.uint8)
        >>> non_rgb_error(multispectral_image)  # Raises ValueError stating incompatibility

    """
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_grayscale_image(image):
            message += "\nYou can convert your grayscale image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        if is_multispectral_image(image):  # Any image with a number of channels other than 1 and 3
            message += "\nThis transformation cannot be applied to multi-spectral images"

        raise ValueError(message)


def check_range(value: tuple[float, float], lower_bound: float, upper_bound: float, name: str | None) -> None:
    """Checks if the given value is within the specified bounds

    Args:
        value (tuple[float, float]): The value to check and convert. Can be a single float or a tuple of floats.
        lower_bound (float): The lower bound for the range check.
        upper_bound (float): The upper bound for the range check.
        name (str | None): The name of the parameter being checked. Used for error messages.

    Raises:
        ValueError: If the value is outside the bounds or if the tuple values are not ordered correctly.

    """
    if not all(lower_bound <= x <= upper_bound for x in value):
        raise ValueError(f"All values in {name} must be within [{lower_bound}, {upper_bound}] for tuple inputs.")
    if not value[0] <= value[1]:
        raise ValueError(f"{name!s} tuple values must be ordered as (min, max). Got: {value}")


class PCA:
    """Principal Component Analysis (PCA) transformer.

    This class provides a wrapper around OpenCV's PCA implementation for
    dimensionality reduction. It can be used to project data onto a lower
    dimensional space while preserving as much variance as possible.

    Args:
        n_components (int | None): Number of components to keep.
            - If None: Keep all components (min of n_samples and n_features)
            - If int: Keep the specified number of components
            Must be greater than 0 if specified.

    Raises:
        ValueError: If n_components is specified and is less than or equal to 0.

    Attributes:
        n_components (int | None): Number of components to keep
        mean (np.ndarray | None): Mean of the training data (set after fitting)
        components_ (np.ndarray | None): Principal components (set after fitting)
        explained_variance_ (np.ndarray | None): Explained variance for each component (set after fitting)

    Examples:
        >>> import numpy as np
        >>> from albumentations.augmentations.utils import PCA
        >>> # Create sample data
        >>> data = np.random.randn(100, 10)  # 100 samples, 10 features
        >>> # Initialize PCA to keep 3 components
        >>> pca = PCA(n_components=3)
        >>> # Fit and transform the data
        >>> transformed = pca.fit_transform(data)
        >>> print(transformed.shape)  # (100, 3)

    """

    def __init__(self, n_components: int | None = None) -> None:
        if n_components is not None and n_components <= 0:
            raise ValueError("Number of components must be greater than zero.")
        self.n_components = n_components
        self.mean: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        """Fit the PCA model to the input data.

        Computes the mean, principal components, and explained variance
        from the input data. The principal components are sorted by
        explained variance in descending order.

        Args:
            x (np.ndarray): Training data of shape (n_samples, n_features).
                Data will be automatically converted to float64 for computation.

        Note:
            - The data is automatically centered (mean-subtracted) during fitting
            - Components are sorted by explained variance (highest first)
            - Uses OpenCV's PCACompute2 for efficient computation

        Examples:
            >>> import numpy as np
            >>> from albumentations.augmentations.utils import PCA
            >>> data = np.random.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> pca.fit(data)
            >>> print(pca.components_.shape)  # (3, 10)

        """
        x = x.astype(np.float64, copy=False)  # avoid unnecessary copy if already float64
        n_samples, n_features = x.shape

        # Determine the number of components if not set
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        self.mean, eigenvectors, eigenvalues = cv2.PCACompute2(x, mean=None, maxComponents=self.n_components)
        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues.flatten()

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data using the fitted PCA model.

        Projects the input data onto the principal components learned during fitting.
        The data is first centered using the mean computed during fitting.

        Args:
            x (np.ndarray): Data to transform of shape (n_samples, n_features).
                Must have the same number of features as the data used for fitting.
                Data will be automatically converted to float64 for computation.

        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
                The transformed data is in the principal component space.

        Raises:
            ValueError: If the model has not been fitted yet (components_ is None).

        Examples:
            >>> import numpy as np
            >>> from albumentations.augmentations.utils import PCA
            >>> # Fit on training data
            >>> train_data = np.random.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> pca.fit(train_data)
            >>> # Transform new data
            >>> test_data = np.random.randn(20, 10)
            >>> transformed = pca.transform(test_data)
            >>> print(transformed.shape)  # (20, 3)

        """
        if self.components_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        x = x.astype(np.float64, copy=False)  # avoid unnecessary copy if already float64
        return cv2.PCAProject(x, self.mean, self.components_)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit the PCA model and transform the data in one step.

        This is equivalent to calling fit(x) followed by transform(x),
        but more convenient. Useful when you want to both learn the
        principal components and transform the same data.

        Args:
            x (np.ndarray): Data to fit and transform of shape (n_samples, n_features).
                Data will be automatically converted to float64 for computation.

        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
                The data projected onto the principal components.

        Examples:
            >>> import numpy as np
            >>> from albumentations.augmentations.utils import PCA
            >>> data = np.random.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> # Fit and transform in one step
            >>> transformed = pca.fit_transform(data)
            >>> print(transformed.shape)  # (100, 3)

        """
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data back to the original space.

        Reconstructs the original data from the principal component representation.
        Note that if n_components < n_features, this will be a lossy reconstruction.

        Args:
            x (np.ndarray): Data in principal component space of shape (n_samples, n_components).
                Must have the same number of components as used during fitting.

        Returns:
            np.ndarray: Reconstructed data of shape (n_samples, n_features).
                The data transformed back to the original feature space.

        Raises:
            ValueError: If the model has not been fitted yet (components_ is None).

        Note:
            - The reconstruction is exact only if all components were kept (n_components = n_features)
            - Otherwise, some information is lost and the reconstruction is approximate
            - The reconstruction adds back the mean that was subtracted during fitting

        Examples:
            >>> import numpy as np
            >>> from albumentations.augmentations.utils import PCA
            >>> # Original data
            >>> data = np.random.randn(100, 10)
            >>> pca = PCA(n_components=3)  # Keep only 3 components
            >>> # Transform and inverse transform
            >>> transformed = pca.fit_transform(data)
            >>> reconstructed = pca.inverse_transform(transformed)
            >>> print(reconstructed.shape)  # (100, 10)
            >>> # Note: reconstructed ≈ data (approximate due to dimensionality reduction)

        """
        if self.components_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        return cv2.PCABackProject(x, self.mean, self.components_)

    def explained_variance_ratio(self) -> np.ndarray:
        """Calculate the proportion of variance explained by each principal component.

        The explained variance ratio indicates how much of the total variance
        in the data is captured by each principal component. Higher values
        indicate components that capture more variance.

        Returns:
            np.ndarray: Array of shape (n_components,) containing the fraction
                of total variance explained by each component. Values sum to <= 1.0,
                with equality when all components are kept.

        Raises:
            ValueError: If the model has not been fitted yet (explained_variance_ is None).

        Note:
            - Values are normalized so they sum to 1.0 if all components are kept
            - The first component always explains the most variance
            - Useful for determining how many components to keep

        Examples:
            >>> import numpy as np
            >>> from albumentations.augmentations.utils import PCA
            >>> data = np.random.randn(100, 10)
            >>> pca = PCA(n_components=5)
            >>> pca.fit(data)
            >>> ratios = pca.explained_variance_ratio()
            >>> print(ratios.shape)  # (5,)
            >>> print(f"First component explains {ratios[0]:.2%} of variance")
            >>> print(f"Total variance explained: {ratios.sum():.2%}")

        """
        if self.explained_variance_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.",
            )
        total_variance = np.sum(self.explained_variance_)
        return self.explained_variance_ / total_variance

    def cumulative_explained_variance_ratio(self) -> np.ndarray:
        """Calculate the cumulative proportion of variance explained.

        Returns the cumulative sum of explained variance ratios. This is useful
        for determining how many components are needed to explain a desired
        amount of variance in the data.

        Returns:
            np.ndarray: Array of shape (n_components,) containing the cumulative
                fraction of variance explained. The i-th element is the total
                variance explained by the first i+1 components.

        Raises:
            ValueError: If the model has not been fitted yet (via explained_variance_ratio).

        Note:
            - Values are monotonically increasing from 0 to <= 1.0
            - Useful for choosing n_components to retain desired variance
            - Common thresholds are 0.95 or 0.99 (95% or 99% of variance)

        Examples:
            >>> import numpy as np
            >>> from albumentations.augmentations.utils import PCA
            >>> data = np.random.randn(100, 10)
            >>> pca = PCA()  # Keep all components
            >>> pca.fit(data)
            >>> cumsum = pca.cumulative_explained_variance_ratio()
            >>> # Find how many components explain 95% of variance
            >>> n_components_95 = np.argmax(cumsum >= 0.95) + 1
            >>> print(f"Need {n_components_95} components for 95% variance")

        """
        return np.cumsum(self.explained_variance_ratio())


def handle_empty_array(param_name: str) -> Callable[[F], F]:
    """Decorator to handle empty array inputs gracefully.

    This decorator wraps a function to check if the specified array parameter
    is empty. If the array is empty, it returns the empty array immediately
    without calling the wrapped function. This prevents errors in functions
    that cannot handle empty arrays.

    Args:
        param_name (str): Name of the parameter that should be checked for emptiness.
            This parameter should be an array-like object with a `len()` method.

    Returns:
        Callable[[F], F]: A decorator function that can be applied to other functions
            to add empty array handling.

    Raises:
        ValueError: If the specified parameter is not provided to the wrapped function.

    Note:
        - The decorator checks for the parameter as both a positional and keyword argument
        - An empty array is defined as one with `len(array) == 0`
        - If the array is empty, the original empty array is returned unmodified
        - This is useful for functions that perform operations on arrays which
          would fail or be meaningless on empty inputs

    Examples:
        >>> import numpy as np
        >>> from albumentations.augmentations.utils import handle_empty_array
        >>>
        >>> @handle_empty_array("points")
        ... def process_points(points):
        ...     # This would fail on empty arrays
        ...     return points.mean(axis=0)
        >>>
        >>> # Empty array is returned immediately
        >>> empty = np.array([])
        >>> result = process_points(empty)
        >>> assert result is empty
        >>>
        >>> # Non-empty arrays are processed normally
        >>> points = np.array([[1, 2], [3, 4]])
        >>> result = process_points(points)
        >>> assert np.array_equal(result, np.array([2., 3.]))

    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if the parameter is passed as positional argument
            if len(args) > 0:
                array = args[0]
            # Check if the parameter is passed as keyword argument
            elif param_name in kwargs:
                array = kwargs[param_name]
            else:
                raise ValueError(f"Missing required argument: {param_name}")

            if len(array) == 0:
                return array
            return func(*args, **kwargs)

        return cast("F", wrapper)

    return decorator
