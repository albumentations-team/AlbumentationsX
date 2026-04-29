"""OverlayElements mixing transform."""

from typing import Any

from ._transforms_shared import (
    LENGTH_RAW_BBOX,
    BaseTransformInitSchema,
    DualTransform,
    ImageType,
    Targets,
    check_bboxes,
    cv2,
    denormalize_bboxes,
    fgeometric,
    fmixing,
    np,
    random,
)


class OverlayElements(DualTransform):
    """Apply overlay images/masks onto an input image (e.g. stickers, logos). Optional bboxes
    and masks for placement. Uses metadata_key.

    Args:
        metadata_key (str): Additional target key for metadata. Default `overlay_metadata`.
        p (float): Probability of applying the transformation. Default: 0.5.

    Possible Metadata Fields:
        - image (ImageType): The overlay image to be applied. This is a required field.
        - bbox (list[int]): The bounding box specifying the region where the overlay should be applied. It should
                            contain four floats: [y_min, x_min, y_max, x_max]. If `label_id` is provided, it should
                            be appended as the fifth element in the bbox. BBox should be in Albumentations format,
                            that is the same as normalized Pascal VOC format
                            [x_min / width, y_min / height, x_max / width, y_max / height]
        - mask (np.ndarray): An optional mask that defines the non-rectangular region of the overlay image. If not
                             provided, the entire overlay image is used.
        - mask_id (int): An optional identifier for the mask. If provided, the regions specified by the mask will
                         be labeled with this identifier in the output mask.

    Targets:
        image, mask

    Image types:
        uint8, float32

    References:
        doc-augmentation: https://github.com/danaaubakirova/doc-augmentation

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare primary data (base image and mask)
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> mask = np.zeros((300, 300), dtype=np.uint8)
        >>>
        >>> # 1. Create a simple overlay image (a red square)
        >>> overlay_image1 = np.zeros((50, 50, 3), dtype=np.uint8)
        >>> overlay_image1[:, :, 0] = 255  # Red color
        >>>
        >>> # 2. Create another overlay with a mask (a blue circle with transparency)
        >>> overlay_image2 = np.zeros((80, 80, 3), dtype=np.uint8)
        >>> overlay_image2[:, :, 2] = 255  # Blue color
        >>> overlay_mask2 = np.zeros((80, 80), dtype=np.uint8)
        >>> # Create a circular mask
        >>> center = (40, 40)
        >>> radius = 30
        >>> for i in range(80):
        ...     for j in range(80):
        ...         if (i - center[0])**2 + (j - center[1])**2 < radius**2:
        ...             overlay_mask2[i, j] = 255
        >>>
        >>> # 3. Create an overlay with both bbox and mask_id
        >>> overlay_image3 = np.zeros((60, 120, 3), dtype=np.uint8)
        >>> overlay_image3[:, :, 1] = 255  # Green color
        >>> # Create a rectangular mask with rounded corners
        >>> overlay_mask3 = np.zeros((60, 120), dtype=np.uint8)
        >>> cv2.rectangle(overlay_mask3, (10, 10), (110, 50), 255, -1)
        >>>
        >>> # Create the metadata list - each item is a dictionary with overlay information
        >>> overlay_metadata = [
        ...     {
        ...         'image': overlay_image1,
        ...         # No bbox provided - will be placed randomly
        ...     },
        ...     {
        ...         'image': overlay_image2,
        ...         'bbox': [0.6, 0.1, 0.9, 0.4],  # Normalized coordinates [x_min, y_min, x_max, y_max]
        ...         'mask': overlay_mask2,
        ...         'mask_id': 1  # This overlay will update the mask with id 1
        ...     },
        ...     {
        ...         'image': overlay_image3,
        ...         'bbox': [0.1, 0.7, 0.5, 0.9],  # Bottom left placement
        ...         'mask': overlay_mask3,
        ...         'mask_id': 2  # This overlay will update the mask with id 2
        ...     }
        ... ]
        >>>
        >>> # Create the transform
        >>> transform = A.Compose([
        ...     A.OverlayElements(p=1.0),
        ... ])
        >>>
        >>> # Apply the transform
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     overlay_metadata=overlay_metadata  # Pass metadata using the default key
        ... )
        >>>
        >>> # Get results with overlays applied
        >>> result_image = result['image']  # Image with the three overlays applied
        >>> result_mask = result['mask']    # Mask with regions labeled using the mask_id values
        >>>
        >>> # Let's verify the mask contains the specified mask_id values
        >>> has_mask_id_1 = np.any(result_mask == 1)  # Should be True
        >>> has_mask_id_2 = np.any(result_mask == 2)  # Should be True

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        metadata_key: str

    def __init__(
        self,
        metadata_key: str = "overlay_metadata",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key]

    @staticmethod
    def preprocess_metadata(
        metadata: dict[str, Any],
        img_shape: tuple[int, int],
        random_state: random.Random,
    ) -> dict[str, Any]:
        overlay_image = metadata["image"]
        overlay_height, overlay_width = overlay_image.shape[:2]
        image_height, image_width = img_shape[:2]

        if "bbox" in metadata:
            bbox = metadata["bbox"]
            bbox_np = np.array([bbox])
            check_bboxes(bbox_np)
            denormalized_bbox = denormalize_bboxes(bbox_np, img_shape[:2])[0]

            x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])

            if "mask" in metadata:
                mask = metadata["mask"]
                mask = fgeometric.resize(mask, (y_max - y_min, x_max - x_min), cv2.INTER_NEAREST)
            else:
                mask = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8)

            overlay_image = fgeometric.resize(overlay_image, (y_max - y_min, x_max - x_min), cv2.INTER_AREA)
            offset = (y_min, x_min)

            if len(bbox) == LENGTH_RAW_BBOX and "bbox_id" in metadata:
                bbox = [x_min, y_min, x_max, y_max, metadata["bbox_id"]]
            else:
                bbox = (x_min, y_min, x_max, y_max, *bbox[4:])
        else:
            if image_height < overlay_height or image_width < overlay_width:
                overlay_image = fgeometric.resize(overlay_image, (image_height, image_width), cv2.INTER_AREA)
                overlay_height, overlay_width = overlay_image.shape[:2]

            mask = metadata["mask"] if "mask" in metadata else np.ones_like(overlay_image, dtype=np.uint8)

            max_x_offset = image_width - overlay_width
            max_y_offset = image_height - overlay_height

            offset_x = random_state.randint(0, max_x_offset)
            offset_y = random_state.randint(0, max_y_offset)

            offset = (offset_y, offset_x)

            bbox = [
                offset_x,
                offset_y,
                offset_x + overlay_width,
                offset_y + overlay_height,
            ]

            if "bbox_id" in metadata:
                bbox = [*bbox, metadata["bbox_id"]]

        result = {
            "overlay_image": overlay_image,
            "overlay_mask": mask,
            "offset": offset,
            "bbox": bbox,
        }

        if "mask_id" in metadata:
            result["mask_id"] = metadata["mask_id"]

        return result

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        metadata = data[self.metadata_key]
        img_shape = params["shape"]

        if isinstance(metadata, list):
            overlay_data = [self.preprocess_metadata(md, img_shape, self.py_random) for md in metadata]
        else:
            overlay_data = [self.preprocess_metadata(metadata, img_shape, self.py_random)]

        return {
            "overlay_data": overlay_data,
        }

    def apply(
        self,
        img: ImageType,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> ImageType:
        for data in overlay_data:
            overlay_image = data["overlay_image"]
            overlay_mask = data["overlay_mask"]
            offset = data["offset"]
            img = fmixing.copy_and_paste_blend(img, overlay_image, overlay_mask, offset=offset)
        return img

    def apply_to_mask(
        self,
        mask: ImageType,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> ImageType:
        for data in overlay_data:
            if "mask_id" in data and data["mask_id"] is not None:
                overlay_mask = data["overlay_mask"]
                offset = data["offset"]
                mask_id = data["mask_id"]

                y_min, x_min = offset
                y_max = y_min + overlay_mask.shape[0]
                x_max = x_min + overlay_mask.shape[1]

                mask_section = mask[y_min:y_max, x_min:x_max]
                mask_section[overlay_mask > 0] = mask_id

        return mask


__all__ = [
    "OverlayElements",
]
