import numpy as np
import pytest

import albumentations as A


class TestFlipMasksCorrectness:
    @pytest.mark.parametrize(
        "transform_class,axis",
        [
            (A.HorizontalFlip, 1),
            (A.VerticalFlip, 0),
        ],
    )
    def test_flip_single_mask_correctness(self, transform_class, axis):
        mask = np.array(
            [
                [[1, 10], [2, 20], [3, 30]],
                [[4, 40], [5, 50], [6, 60]],
            ],
            dtype=np.uint8,
        )

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), mask=mask)

        if axis == 0:
            expected = np.flip(mask, axis=0)
        else:
            expected = np.flip(mask, axis=1)

        np.testing.assert_array_equal(result["mask"], expected)

    @pytest.mark.parametrize(
        "transform_class,axis",
        [
            (A.HorizontalFlip, 2),
            (A.VerticalFlip, 1),
        ],
    )
    def test_flip_masks_batch_correctness(self, transform_class, axis):
        masks = np.array(
            [
                [
                    [[1, 10], [2, 20], [3, 30]],
                    [[4, 40], [5, 50], [6, 60]],
                ],
                [
                    [[7, 70], [8, 80], [9, 90]],
                    [[10, 100], [11, 110], [12, 120]],
                ],
            ],
            dtype=np.uint8,
        )

        transform = transform_class(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), masks=masks)

        expected = np.flip(masks, axis=axis)
        np.testing.assert_array_equal(result["masks"], expected)

    def test_transpose_mask_correctness(self):
        mask = np.array(
            [
                [[1, 10], [2, 20], [3, 30]],
                [[4, 40], [5, 50], [6, 60]],
            ],
            dtype=np.uint8,
        )

        transform = A.Transpose(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), mask=mask)

        expected = np.transpose(mask, (1, 0, 2))
        np.testing.assert_array_equal(result["mask"], expected)

    def test_transpose_masks_batch_correctness(self):
        masks = np.array(
            [
                [
                    [[1, 10], [2, 20], [3, 30]],
                    [[4, 40], [5, 50], [6, 60]],
                ],
                [
                    [[7, 70], [8, 80], [9, 90]],
                    [[10, 100], [11, 110], [12, 120]],
                ],
            ],
            dtype=np.uint8,
        )

        transform = A.Transpose(p=1.0)
        aug = A.Compose([transform])
        result = aug(image=np.zeros((2, 3, 3), dtype=np.uint8), masks=masks)

        expected = np.transpose(masks, (0, 2, 1, 3))
        np.testing.assert_array_equal(result["masks"], expected)


@pytest.mark.pytorch
class TestFlipMasksPyTorchCompatibility:
    def test_horizontal_flip_with_to_tensor_v2(self):
        import torch

        image = np.random.randint(0, 256, (101, 99, 3), dtype=np.uint8)
        masks = np.stack([image[:, :, 0]] * 2)

        transform = A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.ToFloat(max_value=255),
                A.ToTensorV2(),
            ],
            is_check_shapes=False,
            strict=True,
        )

        result = transform(image=image, masks=masks)

        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["masks"], torch.Tensor)
        assert result["masks"].shape[0] == 2


class TestD4MasksSpecific:
    @pytest.mark.parametrize(
        "group_element,should_transpose",
        [
            ("e", False),
            ("r90", True),
            ("r180", False),
            ("r270", True),
            ("v", False),
            ("h", False),
            ("t", True),
            ("hvt", True),
        ],
    )
    def test_d4_mask_dimension_changes(self, group_element, should_transpose):
        mask = np.random.randint(0, 2, (80, 120, 3), dtype=np.uint8)

        result = A.Compose([A.D4(group_element=group_element, p=1.0)])(
            image=np.zeros((80, 120, 3), dtype=np.uint8),
            mask=mask,
        )["mask"]

        if should_transpose:
            assert result.shape == (120, 80, 3), (
                f"D4 with '{group_element}' should swap dimensions but got {result.shape}"
            )
        else:
            assert result.shape == (80, 120, 3), (
                f"D4 with '{group_element}' should preserve dimensions but got {result.shape}"
            )
