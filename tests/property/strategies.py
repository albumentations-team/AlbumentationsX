"""Reusable Hypothesis strategies for AlbumentationsX invariants."""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst


def image_shapes(
    min_height: int = 4,
    max_height: int = 24,
    min_width: int = 4,
    max_width: int = 24,
    channels: tuple[int, ...] = (1, 3, 4, 5),
) -> st.SearchStrategy[tuple[int, int, int]]:
    return st.tuples(
        st.integers(min_height, max_height),
        st.integers(min_width, max_width),
        st.sampled_from(channels),
    )


def _elements_for_dtype(dtype: type[np.generic]) -> st.SearchStrategy[Any]:
    if dtype == np.uint8:
        return st.integers(0, 255)
    if dtype == np.float32:
        return st.floats(0.0, 1.0, width=32, allow_nan=False, allow_infinity=False, allow_subnormal=False)
    msg = f"Unsupported dtype strategy: {dtype}"
    raise ValueError(msg)


def image_arrays(
    dtypes: tuple[type[np.generic], ...] = (np.uint8,),
    channels: tuple[int, ...] = (1, 3, 4, 5),
) -> st.SearchStrategy[np.ndarray]:
    return image_shapes(channels=channels).flatmap(
        lambda shape: st.sampled_from(dtypes).flatmap(
            lambda dtype: npst.arrays(dtype=dtype, shape=shape, elements=_elements_for_dtype(dtype)),
        ),
    )


def image_and_mask_arrays() -> st.SearchStrategy[tuple[np.ndarray, np.ndarray]]:
    return image_shapes(channels=(1, 3, 5)).flatmap(
        lambda shape: st.tuples(
            npst.arrays(dtype=np.uint8, shape=shape, elements=st.integers(0, 255)),
            npst.arrays(dtype=np.uint8, shape=shape[:2], elements=st.integers(0, 3)),
        ),
    )


def volume_and_mask3d_arrays() -> st.SearchStrategy[tuple[np.ndarray, np.ndarray]]:
    volume_shapes = st.tuples(
        st.integers(3, 6),
        st.integers(6, 10),
        st.integers(6, 10),
        st.sampled_from((1, 3)),
    )
    return volume_shapes.flatmap(
        lambda shape: st.tuples(
            npst.arrays(dtype=np.uint8, shape=shape, elements=st.integers(0, 255)),
            npst.arrays(dtype=np.uint8, shape=shape[:3], elements=st.integers(0, 3)),
        ),
    )


def compose_seeds() -> st.SearchStrategy[int]:
    return st.sampled_from((137, 17, 2026))
