"""Bounding-box-aware safe crop transforms."""

from typing import Annotated, Any, Literal

from ._transforms_shared import (
    ALL_TARGETS,
    BaseTransformInitSchema,
    Field,
    ImageType,
    cv2,
    denormalize_bboxes,
    fcrops,
    fgeometric,
    np,
    union_of_bboxes,
)
from .base import (
    BaseCrop,
    CropSizeError,
)


class BBoxSafeRandomCrop(BaseCrop):
    """Random crop that keeps all bboxes inside (erosion_rate). Use when losing any object
    is unacceptable. For at least one bbox use AtLeastOneBBoxRandomCrop.

    Similar to AtLeastOneBboxRandomCrop, but with a key difference:
    - BBoxSafeRandomCrop ensures ALL bounding boxes are preserved in the crop when erosion_rate=0.0
    - AtLeastOneBboxRandomCrop ensures AT LEAST ONE bounding box is present in the crop

    This makes BBoxSafeRandomCrop more suitable for scenarios where:
    - You need to preserve all objects in the scene
    - Losing any bounding box would be problematic (e.g., rare object classes)
    - You're training a model that needs to detect multiple objects simultaneously

    The algorithm:
    1. If bounding boxes exist:
        - Computes the union of all bounding boxes
        - Applies erosion based on erosion_rate to this union
        - Clips the eroded union to valid image coordinates [0,1]
        - Randomly samples crop coordinates within the clipped union area
    2. If no bounding boxes exist:
        - Computes crop height based on erosion_rate
        - Sets crop width to maintain original aspect ratio
        - Randomly places the crop within the image

    Args:
        erosion_rate (float): Controls how much the valid crop region can deviate from the bbox union.
            Must be in range [0.0, 1.0].
            - 0.0: crop must contain the exact bbox union (safest option that guarantees all boxes are preserved)
            - 1.0: crop can deviate maximally from the bbox union (increases likelihood of cutting off some boxes)
            Defaults to 0.0.
        p (float, optional): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Raises:
        CropSizeError: If requested crop size exceeds image dimensions

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with erosion_rate parameter
        >>> transform = A.Compose([
        ...     A.BBoxSafeRandomCrop(erosion_rate=0.2),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = result['image']  # Cropped image containing all bboxes
        >>> transformed_mask = result['mask']    # Cropped mask
        >>> transformed_bboxes = result['bboxes']  # All bounding boxes preserved with adjusted coordinates
        >>> transformed_bbox_labels = result['bbox_labels']  # Original labels preserved
        >>> transformed_keypoints = result['keypoints']  # Keypoints with adjusted coordinates
        >>> transformed_keypoint_labels = result['keypoint_labels']  # Original keypoint labels preserved
        >>>
        >>> # Example with a different erosion_rate
        >>> transform_more_flexible = A.Compose([
        ...     A.BBoxSafeRandomCrop(erosion_rate=0.5),  # More flexibility in crop placement
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']))
        >>>
        >>> # Apply transform with only image and bboxes
        >>> result_bboxes_only = transform_more_flexible(
        ...     image=image,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels
        ... )
        >>> transformed_image = result_bboxes_only['image']
        >>> transformed_bboxes = result_bboxes_only['bboxes']  # All bboxes still preserved

    Note:
        - IMPORTANT: Using erosion_rate > 0.0 may result in some bounding boxes being cut off,
          particularly narrow boxes at the boundary of the union area. For guaranteed preservation
          of all bounding boxes, use erosion_rate=0.0.
        - Aspect ratio is preserved only when no bounding boxes are present
        - May be more restrictive in crop placement compared to AtLeastOneBboxRandomCrop
        - The crop size is determined by the bounding boxes when present

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        erosion_rate: float = Field(
            ge=0.0,
            le=1.0,
        )

    def __init__(self, erosion_rate: float = 0.0, p: float = 1.0):
        super().__init__(p=p)
        self.erosion_rate = erosion_rate

    def _get_coords_no_bbox(self, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        image_height, image_width = image_shape

        erosive_h = int(image_height * (1.0 - self.erosion_rate))
        crop_height = image_height if erosive_h >= image_height else self.py_random.randint(erosive_h, image_height)

        crop_width = int(crop_height * image_width / image_height)

        h_start = self.py_random.random()
        w_start = self.py_random.random()

        crop_shape = (crop_height, crop_width)

        return fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]

        if len(data["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            crop_coords = self._get_coords_no_bbox(image_shape)
            return {"crop_coords": crop_coords}

        bbox_union = union_of_bboxes(bboxes=data["bboxes"], erosion_rate=self.erosion_rate)

        if bbox_union is None:
            crop_coords = self._get_coords_no_bbox(image_shape)
            return {"crop_coords": crop_coords}

        x_min, y_min, x_max, y_max = bbox_union

        x_min = np.clip(x_min, 0, 1)
        y_min = np.clip(y_min, 0, 1)
        x_max = np.clip(x_max, x_min, 1)
        y_max = np.clip(y_max, y_min, 1)

        image_height, image_width = image_shape

        crop_x_min = int(x_min * self.py_random.random() * image_width)
        crop_y_min = int(y_min * self.py_random.random() * image_height)

        bbox_xmax = x_max + (1 - x_max) * self.py_random.random()
        bbox_ymax = y_max + (1 - y_max) * self.py_random.random()
        crop_x_max = int(bbox_xmax * image_width)
        crop_y_max = int(bbox_ymax * image_height)

        return {"crop_coords": (crop_x_min, crop_y_min, crop_x_max, crop_y_max)}


class RandomSizedBBoxSafeCrop(BBoxSafeRandomCrop):
    """Random crop keeping every bbox inside, then resize to (height, width). erosion_rate sets
    minimum crop size. Use when no object can be cut off.

    This transform first attempts to crop a random portion of the input image while ensuring that all bounding boxes
    remain within the cropped area. It then resizes the crop to the specified size. This is particularly useful for
    object detection tasks where preserving all objects in the image is crucial while also standardizing the image size.

    Args:
        height (int): Height of the output image after resizing.
        width (int): Width of the output image after resizing.
        erosion_rate (float): A value between 0.0 and 1.0 that determines the minimum allowable size of the crop
            as a fraction of the original image size. For example, an erosion_rate of 0.2 means the crop will be
            at least 80% of the original image height and width. Default: 0.0 (no minimum size).
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - This transform ensures that all bounding boxes in the original image are fully contained within the
          cropped area. If it's not possible to find such a crop (e.g., when bounding boxes are too spread out),
          it will default to cropping the entire image.
        - After cropping, the result is resized to the specified (height, width) size.
        - Bounding box coordinates are adjusted to match the new image size.
        - Keypoints are moved along with the crop and scaled to the new image size.
        - If there are no bounding boxes in the image, it will fall back to a random crop.

    Mathematical Details:
        1. A crop region is selected that includes all bounding boxes.
        2. The crop size is determined by the erosion_rate:
           min_crop_size = (1 - erosion_rate) * original_size
        3. If the selected crop is smaller than min_crop_size, it's expanded to meet this requirement.
        4. The crop is then resized to the specified (height, width) size.
        5. Bounding box coordinates are transformed to match the new image size:
           new_coord = (old_coord - crop_start) * (new_size / crop_size)

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (300, 300), dtype=np.uint8)
        >>>
        >>> # Create bounding boxes with some overlap and separation
        >>> bboxes = np.array([
        ...     [10, 10, 80, 80],    # top-left box
        ...     [100, 100, 200, 200], # center box
        ...     [210, 210, 290, 290]  # bottom-right box
        ... ], dtype=np.float32)
        >>> bbox_labels = ['cat', 'dog', 'bird']
        >>>
        >>> # Create keypoints inside the bounding boxes
        >>> keypoints = np.array([
        ...     [45, 45],    # inside first box
        ...     [150, 150],  # inside second box
        ...     [250, 250]   # inside third box
        ... ], dtype=np.float32)
        >>> keypoint_labels = ['nose', 'eye', 'tail']
        >>>
        >>> # Example 1: Basic usage with default parameters
        >>> transform_basic = A.Compose([
        ...     A.RandomSizedBBoxSafeCrop(height=224, width=224, p=1.0),
        ... ], bbox_params=A.BboxParams(
        ...     format='pascal_voc',
        ...     label_fields=['bbox_labels']
        ... ), keypoint_params=A.KeypointParams(
        ...     format='xy',
        ...     label_fields=['keypoint_labels']
        ... ))
        >>>
        >>> # Apply the transform
        >>> result_basic = transform_basic(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Access the transformed data
        >>> transformed_image = result_basic['image']  # Shape will be (224, 224, 3)
        >>> transformed_mask = result_basic['mask']    # Shape will be (224, 224)
        >>> transformed_bboxes = result_basic['bboxes']  # All original bounding boxes preserved
        >>> transformed_bbox_labels = result_basic['bbox_labels']  # Original labels preserved
        >>> transformed_keypoints = result_basic['keypoints']  # Keypoints adjusted to new coordinates
        >>> transformed_keypoint_labels = result_basic['keypoint_labels']  # Original labels preserved
        >>>
        >>> # Example 2: With erosion_rate for more flexibility in crop placement
        >>> transform_erosion = A.Compose([
        ...     A.RandomSizedBBoxSafeCrop(
        ...         height=256,
        ...         width=256,
        ...         erosion_rate=0.2,  # Allows 20% flexibility in crop placement
        ...         interpolation=cv2.INTER_CUBIC,  # Higher quality interpolation
        ...         mask_interpolation=cv2.INTER_NEAREST,  # Preserve mask edges
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(
        ...     format='pascal_voc',
        ...     label_fields=['bbox_labels'],
        ...     min_visibility=0.3  # Only keep bboxes with at least 30% visibility
        ... ), keypoint_params=A.KeypointParams(
        ...     format='xy',
        ...     label_fields=['keypoint_labels'],
        ...     remove_invisible=True  # Remove keypoints outside the crop
        ... ))
        >>>
        >>> # Apply the transform with erosion
        >>> result_erosion = transform_erosion(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # With erosion_rate=0.2, the crop has more flexibility in placement
        >>> # while still ensuring all bounding boxes are included

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        erosion_rate: float = Field(
            ge=0.0,
            le=1.0,
        )
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

    def __init__(
        self,
        height: int,
        width: int,
        erosion_rate: float = 0.0,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        p: float = 1.0,
    ):
        super().__init__(erosion_rate=erosion_rate, p=p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def apply(
        self,
        img: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        crop = fcrops.crop(img, *crop_coords)
        return fgeometric.resize(crop, (self.height, self.width), self.interpolation)

    def apply_to_mask(
        self,
        mask: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        crop = fcrops.crop(mask, *crop_coords)
        return fgeometric.resize(crop, (self.height, self.width), self.mask_interpolation)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        keypoints = fcrops.crop_keypoints_by_coords(keypoints, crop_coords)

        crop_height = crop_coords[3] - crop_coords[1]
        crop_width = crop_coords[2] - crop_coords[0]

        scale_y = self.height / crop_height
        scale_x = self.width / crop_width
        return fgeometric.keypoints_scale(keypoints, scale_x=scale_x, scale_y=scale_y)


class AtLeastOneBBoxRandomCrop(BaseCrop):
    """Random crop of fixed size that contains at least one bbox. erosion_factor controls
    overlap with reference box. Use when some object loss is acceptable.

    Similar to BBoxSafeRandomCrop, but with a key difference:
    - BBoxSafeRandomCrop ensures ALL bounding boxes are preserved in the crop
    - AtLeastOneBBoxRandomCrop ensures AT LEAST ONE bounding box is present in the crop

    This makes AtLeastOneBBoxRandomCrop more flexible for scenarios where:
    - You want to focus on individual objects rather than all objects
    - You're willing to lose some bounding boxes to get more varied crops
    - The image has many bounding boxes and keeping all of them would be too restrictive

    The algorithm:
    1. If bounding boxes exist:
        - Randomly selects a reference bounding box from available boxes
        - Computes an eroded version of this box (shrunk by erosion_factor)
        - Calculates valid crop bounds that ensure overlap with the eroded box
        - Randomly samples crop coordinates within these bounds
    2. If no bounding boxes exist:
        - Uses full image dimensions as valid bounds
        - Randomly samples crop coordinates within these bounds

    Args:
        height (int): Fixed height of the crop
        width (int): Fixed width of the crop
        erosion_factor (float, optional): Factor by which to erode (shrink) the reference
            bounding box when computing valid crop regions. Must be in range [0.0, 1.0].
            - 0.0 means no erosion (crop must fully contain the reference box)
            - 1.0 means maximum erosion (crop can be anywhere that intersects the reference box)
            Defaults to 0.0.
        p (float, optional): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Raises:
        CropSizeError: If requested crop size exceeds image dimensions

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (300, 300), dtype=np.uint8)
        >>> # Create multiple bounding boxes - the transform will ensure at least one is in the crop
        >>> bboxes = np.array([
        ...     [30, 50, 100, 140],   # first box
        ...     [150, 120, 270, 250], # second box
        ...     [200, 30, 280, 90]    # third box
        ... ], dtype=np.float32)
        >>> bbox_labels = [1, 2, 3]
        >>> keypoints = np.array([
        ...     [50, 70],    # keypoint inside first box
        ...     [190, 170],  # keypoint inside second box
        ...     [240, 60]    # keypoint inside third box
        ... ], dtype=np.float32)
        >>> keypoint_labels = [0, 1, 2]
        >>>
        >>> # Define transform with different erosion_factor values
        >>> transform = A.Compose([
        ...     A.AtLeastOneBBoxRandomCrop(
        ...         height=200,
        ...         width=200,
        ...         erosion_factor=0.2,  # Allows moderate flexibility in crop placement
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']       # Shape: (200, 200, 3)
        >>> transformed_mask = transformed['mask']         # Shape: (200, 200)
        >>> transformed_bboxes = transformed['bboxes']     # At least one bbox is guaranteed
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for the preserved bboxes
        >>> transformed_keypoints = transformed['keypoints']      # Only keypoints in crop are kept
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Their labels
        >>>
        >>> # Verify that at least one bounding box was preserved
        >>> assert len(transformed_bboxes) > 0, "Should have at least one bbox in the crop"
        >>>
        >>> # With erosion_factor=0.0, the crop must fully contain the selected reference bbox
        >>> conservative_transform = A.Compose([
        ...     A.AtLeastOneBBoxRandomCrop(
        ...         height=200,
        ...         width=200,
        ...         erosion_factor=0.0,  # No erosion - crop must fully contain a bbox
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']))
        >>>
        >>> # With erosion_factor=1.0, the crop must only intersect with the selected reference bbox
        >>> flexible_transform = A.Compose([
        ...     A.AtLeastOneBBoxRandomCrop(
        ...         height=200,
        ...         width=200,
        ...         erosion_factor=1.0,  # Maximum erosion - crop only needs to intersect a bbox
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']))

    Note:
        - Uses fixed crop dimensions (height and width)
        - Bounding boxes that end up partially outside the crop will be adjusted
        - Bounding boxes that end up completely outside the crop will be removed
        - If no bounding boxes are provided, acts as a regular random crop

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseCrop.InitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        erosion_factor: Annotated[float, Field(ge=0.0, le=1.0)]

    def __init__(
        self,
        height: int,
        width: int,
        erosion_factor: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.erosion_factor = erosion_factor

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_height, image_width = params["shape"][:2]
        bboxes = data.get("bboxes", [])

        if self.height > image_height or self.width > image_width:
            raise CropSizeError(
                f"Crop size (height, width) exceeds image dimensions (height, width):"
                f" {(self.height, self.width)} vs {image_height, image_width}",
            )

        if len(bboxes) > 0:
            bboxes = denormalize_bboxes(bboxes, shape=(image_height, image_width))

            # Pick a bbox amongst all possible as our reference bbox.
            idx = self.random_generator.integers(0, len(bboxes))
            reference_bbox = bboxes[idx]

            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = reference_bbox[:4]

            # Compute valid crop bounds:
            # erosion_factor = 0.0: crop must fully contain the bbox
            # erosion_factor = 1.0: crop can be anywhere that intersects the bbox
            if self.erosion_factor < 1.0:
                # Regular case: compute eroded box dimensions
                bbox_width = bbox_x2 - bbox_x1
                bbox_height = bbox_y2 - bbox_y1
                eroded_width = bbox_width * (1.0 - self.erosion_factor)
                eroded_height = bbox_height * (1.0 - self.erosion_factor)

                min_crop_x = np.clip(
                    a=bbox_x1 + eroded_width - self.width,
                    a_min=0.0,
                    a_max=image_width - self.width,
                )
                max_crop_x = np.clip(
                    a=bbox_x2 - eroded_width,
                    a_min=0.0,
                    a_max=image_width - self.width,
                )

                min_crop_y = np.clip(
                    a=bbox_y1 + eroded_height - self.height,
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
                max_crop_y = np.clip(
                    a=bbox_y2 - eroded_height,
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
            else:
                # Maximum erosion case: crop can be anywhere that intersects the bbox
                min_crop_x = np.clip(
                    a=bbox_x1 - self.width,  # leftmost position that still intersects
                    a_min=0.0,
                    a_max=image_width - self.width,
                )
                max_crop_x = np.clip(
                    a=bbox_x2,  # rightmost position that still intersects
                    a_min=0.0,
                    a_max=image_width - self.width,
                )

                min_crop_y = np.clip(
                    a=bbox_y1 - self.height,  # topmost position that still intersects
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
                max_crop_y = np.clip(
                    a=bbox_y2,  # bottommost position that still intersects
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
        else:
            # If there are no bboxes, just crop anywhere in the image.
            min_crop_x = 0.0
            max_crop_x = image_width - self.width

            min_crop_y = 0.0
            max_crop_y = image_height - self.height

        # Randomly draw the upper-left corner of the crop.
        crop_x1 = int(self.py_random.uniform(a=min_crop_x, b=max_crop_x))
        crop_y1 = int(self.py_random.uniform(a=min_crop_y, b=max_crop_y))

        crop_x2 = crop_x1 + self.width
        crop_y2 = crop_y1 + self.height

        return {"crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2)}


__all__ = [
    "AtLeastOneBBoxRandomCrop",
    "BBoxSafeRandomCrop",
    "RandomSizedBBoxSafeCrop",
]
