import numpy as np
import albumentations as A

def test_hsv_apply_to_images_list():
    img1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
    img2 = np.ones((32, 32, 3), dtype=np.uint8) * 150

    # force apply with p=1.0 and enable apply_to_images
    aug = A.HueSaturationValue(apply_to_images=True, p=1.0)

    out = aug(images=[img1, img2])

    # Support both API styles (some transforms return dict with 'images' key)
    if isinstance(out, dict) and "images" in out:
        images_out = out["images"]
    else:
        images_out = out

    assert isinstance(images_out, (list, tuple))
    assert len(images_out) == 2
    assert images_out[0].shape == img1.shape
    assert images_out[1].shape == img2.shape

    # If two identical images are passed, outputs should be identical because we sample params once
    same1 = np.copy(img1)
    same2 = np.copy(img1)
    out_same = aug(images=[same1, same2])
    if isinstance(out_same, dict) and "images" in out_same:
        images_same = out_same["images"]
    else:
        images_same = out_same

    assert np.array_equal(images_same[0], images_same[1])
