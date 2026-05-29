import numpy as np
import pytest
import albumentations as A

def test_random_snow_float32():
    # Test that float32 inputs are processed correctly and stay float32
    img_float = np.full((100, 100, 3), 0.5, dtype=np.float32)
    
    transform = A.RandomSnow(p=1.0)
    
    # We should run it a few times to ensure both methods ('bleach', 'texture') might be hit if randomized,
    # or just rely on the fact that random seed will eventually hit them.
    # We can also mock or just run it.
    
    for _ in range(5):
        res = transform(image=img_float)['image']
        
        # Check dtype is preserved
        assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
        
        # Check values are still in valid range
        assert res.min() >= 0.0, f"Min value is {res.min()}"
        assert res.max() <= 1.0, f"Max value is {res.max()}"

def test_random_snow_uint8():
    # Verify uint8 still works
    img_uint8 = np.full((100, 100, 3), 128, dtype=np.uint8)
    transform = A.RandomSnow(p=1.0)
    for _ in range(5):
        res = transform(image=img_uint8)['image']
        assert res.dtype == np.uint8, f"Expected uint8, got {res.dtype}"
        assert res.min() >= 0
        assert res.max() <= 255
