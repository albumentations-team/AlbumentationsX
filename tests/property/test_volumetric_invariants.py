"""Volumetric shape and alignment invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given

import albumentations as A
from tests.property.strategies import volume_and_mask3d_arrays

pytestmark = pytest.mark.property


@given(data=volume_and_mask3d_arrays())
def test_center_crop3d_preserves_volume_mask3d_alignment(data: tuple[np.ndarray, np.ndarray]) -> None:
    volume, mask3d = data
    transform = A.Compose([A.CenterCrop3D(size=(2, 4, 4), p=1.0)], strict=True)

    result = transform(volume=volume, mask3d=mask3d)

    assert result["volume"].shape[:3] == result["mask3d"].shape
    assert result["volume"].shape[3] == volume.shape[3]
    assert result["volume"].dtype == volume.dtype
    assert result["mask3d"].dtype == mask3d.dtype
