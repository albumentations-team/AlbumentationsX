"""Transform for synthetic annotation and callout artifacts."""

import string
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

from albumentations.augmentations.other import annotation_artifacts_functional as fannotation
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.type_definitions import ImageType

__all__ = ["AnnotationArtifacts"]

AnnotationElementType = Literal["text", "rectangle", "arrow", "line", "callout"]
LineStyle = Literal["solid", "dashed", "dotted"]
Point = tuple[int, int]

TEXT_ALPHABET = string.ascii_uppercase + string.digits
FONT_OPTIONS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
)
LINE_STYLES: tuple[LineStyle, ...] = ("solid", "dashed", "dotted")
LINE_STYLE_WEIGHTS = (0.55, 0.35, 0.1)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


class AnnotationArtifacts(ImageOnlyTransform):
    """Add synthetic text, arrows, boxes, guide lines, and callouts that mimic scientific
    markup. Use to harden models against annotation artifacts.

    This transform simulates sparse human annotation artifacts commonly found in scientific
    figures, medical images, microscopy screenshots, and competition data. It draws short text
    tokens, rectangles, arrows, horizontal or vertical guide lines, and zoom-callout boxes directly
    on the image.

    Args:
        element_types (tuple[Literal["text", "rectangle", "arrow", "line", "callout"], ...]): Artifact types
            to sample. Default: ("text", "rectangle", "arrow", "line", "callout").
        element_probabilities (tuple[float, ...]): Sampling weights matching `element_types`.
            Values must be non-negative and at least one value must be positive.
            Default: (0.35, 0.2, 0.2, 0.15, 0.1).
        count_range (tuple[int, int]): Range for the number of artifacts drawn per image.
            Default: (1, 3).
        text_length_range (tuple[int, int]): Range for generated text token length.
            Text uses uppercase ASCII letters and digits. Default: (1, 5).
        font_scale_range (tuple[float, float]): Range for OpenCV Hershey font scale.
            Default: (0.3, 1.2).
        thickness_range (tuple[int, int]): Range for line, rectangle, arrow, and text thickness.
            Default: (1, 3).
        size_ratio_range (tuple[float, float]): Range for rectangle and callout size as a
            fraction of image width and height. Default: (0.1, 0.35).
        line_length_ratio_range (tuple[float, float]): Range for line and arrow length as a
            fraction of the smaller image dimension. Default: (0.1, 0.8).
        tip_length_range (tuple[float, float]): Range for arrowhead length as a fraction of arrow length.
            Default: (0.2, 0.4).
        corner_prob (float): Probability of placing artifacts near image corners or edges instead
            of uniformly inside the image. Default: 0.6.
        black_white_prob (float): Probability of choosing black or white instead of red for an artifact.
            Default: 0.85.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This is an image-only transform: masks, bounding boxes, and keypoints are not modified.
        - Colors are adapted to the number of channels; black and white affect all channels,
          while red maps to the first channel and pads remaining channels with zero.
        - Random values are sampled before drawing, so replay and deterministic pipelines preserve
          the exact generated artifacts.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> image = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.AnnotationArtifacts(
        ...         element_types=("text", "rectangle", "arrow", "line", "callout"),
        ...         element_probabilities=(0.35, 0.2, 0.2, 0.15, 0.1),
        ...         count_range=(1, 3),
        ...         corner_prob=0.6,
        ...         p=1.0,
        ...     )
        ... ])
        >>> result = transform(image=image)
        >>> augmented_image = result["image"]

    References:
        - Uladzislau Leketush: https://www.linkedin.com/in/leketush/
        - Original augmentation gist: https://gist.github.com/vlad3996/00724aafce45374214e16eb9eb07e893
        - Kaggle 1st place solution: https://github.com/vlad3996/forgeryscope/
        - Competition: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection

    See Also:
        - TextImage: Metadata-driven rendering of text inside known bounding boxes.
        - OverlayElements: Paste supplied overlay images or masks onto an image.
        - CoarseDropout: Remove rectangular regions instead of adding annotation markup.

    """

    class InitSchema(BaseTransformInitSchema):
        element_types: tuple[AnnotationElementType, ...] = Field(min_length=1)
        element_probabilities: tuple[float, ...] = Field(min_length=1)
        count_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        text_length_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        font_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        thickness_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        size_ratio_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        line_length_ratio_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        tip_length_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        corner_prob: float = Field(ge=0, le=1)
        black_white_prob: float = Field(ge=0, le=1)

        @model_validator(mode="after")
        def _validate_element_probabilities(self) -> Self:
            if len(self.element_types) != len(self.element_probabilities):
                raise ValueError("element_types and element_probabilities must have the same length.")

            if any(probability < 0 for probability in self.element_probabilities):
                raise ValueError("element_probabilities must be non-negative.")

            if sum(self.element_probabilities) <= 0:
                raise ValueError("At least one element probability must be positive.")

            return self

    def __init__(
        self,
        element_types: tuple[AnnotationElementType, ...] = ("text", "rectangle", "arrow", "line", "callout"),
        element_probabilities: tuple[float, ...] = (0.35, 0.2, 0.2, 0.15, 0.1),
        count_range: tuple[int, int] = (1, 3),
        text_length_range: tuple[int, int] = (1, 5),
        font_scale_range: tuple[float, float] = (0.3, 1.2),
        thickness_range: tuple[int, int] = (1, 3),
        size_ratio_range: tuple[float, float] = (0.1, 0.35),
        line_length_ratio_range: tuple[float, float] = (0.1, 0.8),
        tip_length_range: tuple[float, float] = (0.2, 0.4),
        corner_prob: float = 0.6,
        black_white_prob: float = 0.85,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.element_types = element_types
        self.element_probabilities = element_probabilities
        self.count_range = count_range
        self.text_length_range = text_length_range
        self.font_scale_range = font_scale_range
        self.thickness_range = thickness_range
        self.size_ratio_range = size_ratio_range
        self.line_length_ratio_range = line_length_ratio_range
        self.tip_length_range = tip_length_range
        self.corner_prob = corner_prob
        self.black_white_prob = black_white_prob

    def apply(
        self,
        img: ImageType,
        artifacts: list[dict[str, Any]],
        **params: Any,
    ) -> ImageType:
        return fannotation.draw_annotation_artifacts(img, artifacts)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        image_height, image_width = params["shape"][:2]
        artifacts = self._generate_artifacts(image_height, image_width)
        self.applied_config = self._get_applied_config(artifacts, image_height, image_width)
        return {"artifacts": artifacts}

    @staticmethod
    def _get_applied_config(
        artifacts: list[dict[str, Any]],
        image_height: int,
        image_width: int,
    ) -> dict[str, Any]:
        min_dimension = min(image_height, image_width)
        return {
            "count_range": len(artifacts),
            "text_length_range": [len(artifact["text"]) for artifact in artifacts if artifact["type"] == "text"],
            "font_scale_range": [artifact["font_scale"] for artifact in artifacts if artifact["type"] == "text"],
            "thickness_range": [artifact["thickness"] for artifact in artifacts if "thickness" in artifact],
            "size_ratio_range": [
                (
                    (artifact["bottom_right"][0] - artifact["top_left"][0]) / image_width,
                    (artifact["bottom_right"][1] - artifact["top_left"][1]) / image_height,
                )
                for artifact in artifacts
                if artifact["type"] in {"rectangle", "callout"}
            ],
            "line_length_ratio_range": [
                np.hypot(
                    artifact["end"][0] - artifact["start"][0],
                    artifact["end"][1] - artifact["start"][1],
                )
                / min_dimension
                for artifact in artifacts
                if artifact["type"] in {"arrow", "line"} and min_dimension > 0
            ],
            "tip_length_range": [artifact["tip_length"] for artifact in artifacts if artifact["type"] == "arrow"],
        }

    def _generate_artifacts(self, image_height: int, image_width: int) -> list[dict[str, Any]]:
        if image_height <= 1 or image_width <= 1:
            return []

        artifact_count = self.py_random.randint(*self.count_range)
        artifact_types = self.py_random.choices(
            self.element_types,
            weights=self.element_probabilities,
            k=artifact_count,
        )

        artifacts = []
        for artifact_type in artifact_types:
            artifact = self._generate_artifact(artifact_type, image_height, image_width)
            if artifact is not None:
                artifacts.append(artifact)

        return artifacts

    def _generate_artifact(
        self,
        artifact_type: AnnotationElementType,
        image_height: int,
        image_width: int,
    ) -> dict[str, Any] | None:
        generators = {
            "text": self._generate_text_artifact,
            "rectangle": self._generate_rectangle_artifact,
            "arrow": self._generate_arrow_artifact,
            "line": self._generate_line_artifact,
            "callout": self._generate_callout_artifact,
        }
        return generators[artifact_type](image_height, image_width)

    def _generate_text_artifact(self, image_height: int, image_width: int) -> dict[str, Any]:
        text_length = self.py_random.randint(*self.text_length_range)
        text = "".join(self.py_random.choices(TEXT_ALPHABET, k=text_length))
        font = self.py_random.choice(FONT_OPTIONS)
        font_scale = self.py_random.uniform(*self.font_scale_range)
        thickness = self.py_random.randint(*self.thickness_range)
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        margin = self._sample_margin(image_height, image_width)
        origin = self._sample_text_origin(image_height, image_width, text_width, text_height, margin)

        return {
            "type": "text",
            "text": text,
            "origin": origin,
            "font": font,
            "font_scale": font_scale,
            "color": self._sample_color(),
            "thickness": thickness,
        }

    def _generate_rectangle_artifact(self, image_height: int, image_width: int) -> dict[str, Any]:
        box_width, box_height = self._sample_box_size(image_height, image_width)
        top_left = self._sample_box_origin(image_height, image_width, box_width, box_height)
        top_left_col, top_left_row = top_left
        bottom_right = (top_left_col + box_width, top_left_row + box_height)

        return {
            "type": "rectangle",
            "top_left": top_left,
            "bottom_right": bottom_right,
            "color": self._sample_color(),
            "thickness": self.py_random.randint(*self.thickness_range),
            "filled": False,
        }

    def _generate_callout_artifact(self, image_height: int, image_width: int) -> dict[str, Any]:
        artifact = self._generate_rectangle_artifact(image_height, image_width)
        artifact["type"] = "callout"
        artifact["style"] = self._sample_line_style()
        artifact["lines"] = self._sample_callout_lines(artifact, image_height, image_width)
        return artifact

    def _generate_line_artifact(self, image_height: int, image_width: int) -> dict[str, Any]:
        margin = self._sample_margin(image_height, image_width)
        is_vertical = self.py_random.choice([True, False])
        line_length = self._sample_line_length(image_height, image_width, margin, is_vertical)

        if is_vertical:
            min_row = margin
            max_row = max(margin, image_height - 1 - margin)
            start_row = self.py_random.randint(min_row, max_row - line_length)
            line_col = self.py_random.randint(margin, max(margin, image_width - 1 - margin))
            start = (line_col, start_row)
            end = (line_col, start_row + line_length)
        else:
            min_col = margin
            max_col = max(margin, image_width - 1 - margin)
            start_col = self.py_random.randint(min_col, max_col - line_length)
            line_row = self.py_random.randint(margin, max(margin, image_height - 1 - margin))
            start = (start_col, line_row)
            end = (start_col + line_length, line_row)

        return {
            "type": "line",
            "start": start,
            "end": end,
            "color": self._sample_color(),
            "thickness": self.py_random.randint(*self.thickness_range),
            "style": self._sample_line_style(),
        }

    def _sample_line_length(
        self,
        image_height: int,
        image_width: int,
        margin: int,
        is_vertical: bool,
    ) -> int:
        max_length = max(0, (image_height if is_vertical else image_width) - 1 - (2 * margin))
        sampled_length = round(self.py_random.uniform(*self.line_length_ratio_range) * min(image_height, image_width))
        return min(max_length, max(1, sampled_length))

    def _generate_arrow_artifact(self, image_height: int, image_width: int) -> dict[str, Any]:
        start = self._sample_arrow_start(image_height, image_width)
        end = self._sample_arrow_end(start, image_height, image_width)

        return {
            "type": "arrow",
            "start": start,
            "end": end,
            "color": self._sample_color(),
            "thickness": self.py_random.randint(*self.thickness_range),
            "tip_length": self.py_random.uniform(*self.tip_length_range),
            "style": self._sample_line_style(),
        }

    def _sample_color(self) -> tuple[int, ...]:
        if self.py_random.random() < self.black_white_prob:
            return self.py_random.choice([BLACK, WHITE])

        return RED

    def _sample_line_style(self) -> LineStyle:
        return self.py_random.choices(LINE_STYLES, weights=LINE_STYLE_WEIGHTS, k=1)[0]

    def _sample_margin(self, image_height: int, image_width: int) -> int:
        max_margin = max(1, min(10, min(image_height, image_width) // 8))
        return self.py_random.randint(1, max_margin)

    def _sample_box_size(self, image_height: int, image_width: int) -> tuple[int, int]:
        min_ratio, max_ratio = self.size_ratio_range
        box_width = max(1, int(self.py_random.uniform(min_ratio, max_ratio) * image_width))
        box_height = max(1, int(self.py_random.uniform(min_ratio, max_ratio) * image_height))
        return min(box_width, image_width - 1), min(box_height, image_height - 1)

    def _sample_box_origin(
        self,
        image_height: int,
        image_width: int,
        box_width: int,
        box_height: int,
    ) -> Point:
        max_origin_col = max(0, image_width - 1 - box_width)
        max_origin_row = max(0, image_height - 1 - box_height)
        margin = self._sample_margin(image_height, image_width)

        if self.py_random.random() < self.corner_prob:
            return self._sample_corner_origin(max_origin_col, max_origin_row, margin)

        return (
            self.py_random.randint(0, max_origin_col),
            self.py_random.randint(0, max_origin_row),
        )

    def _sample_corner_origin(self, max_origin_col: int, max_origin_row: int, margin: int) -> Point:
        corner = self.py_random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
        left_col = min(margin, max_origin_col)
        right_col = max(0, max_origin_col - margin)
        top_row = min(margin, max_origin_row)
        bottom_row = max(0, max_origin_row - margin)

        corner_points = {
            "top_left": (left_col, top_row),
            "top_right": (right_col, top_row),
            "bottom_left": (left_col, bottom_row),
            "bottom_right": (right_col, bottom_row),
        }
        return corner_points[corner]

    def _sample_text_origin(
        self,
        image_height: int,
        image_width: int,
        text_width: int,
        text_height: int,
        margin: int,
    ) -> Point:
        max_origin_col = max(0, image_width - 1 - text_width)
        min_origin_row = min(image_height - 1, text_height)
        max_origin_row = max(min_origin_row, image_height - 1 - margin)

        if self.py_random.random() < self.corner_prob:
            corner = self.py_random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
            origin_col = min(margin, max_origin_col) if "left" in corner else max(0, max_origin_col - margin)
            origin_row = min_origin_row + margin if "top" in corner else max_origin_row
            return (origin_col, min(origin_row, image_height - 1))

        return (
            self.py_random.randint(0, max_origin_col),
            self.py_random.randint(min_origin_row, max_origin_row),
        )

    def _sample_callout_lines(
        self,
        artifact: dict[str, Any],
        image_height: int,
        image_width: int,
    ) -> list[tuple[Point, Point]]:
        top_left_col, top_left_row = artifact["top_left"]
        bottom_right_col, bottom_right_row = artifact["bottom_right"]
        side_count = self.py_random.randint(1, 2)
        sides = self.py_random.sample(["left", "right", "top", "bottom"], side_count)
        side_points = {
            "left": (
                (top_left_col, (top_left_row + bottom_right_row) // 2),
                (0, (top_left_row + bottom_right_row) // 2),
            ),
            "right": (
                (bottom_right_col, (top_left_row + bottom_right_row) // 2),
                (image_width - 1, (top_left_row + bottom_right_row) // 2),
            ),
            "top": (
                ((top_left_col + bottom_right_col) // 2, top_left_row),
                ((top_left_col + bottom_right_col) // 2, 0),
            ),
            "bottom": (
                ((top_left_col + bottom_right_col) // 2, bottom_right_row),
                ((top_left_col + bottom_right_col) // 2, image_height - 1),
            ),
        }
        return [side_points[side] for side in sides]

    def _sample_arrow_start(self, image_height: int, image_width: int) -> Point:
        margin = self._sample_margin(image_height, image_width)
        max_col = max(margin, image_width - 1 - margin)
        max_row = max(margin, image_height - 1 - margin)

        if self.py_random.random() >= self.corner_prob:
            return (
                self.py_random.randint(margin, max_col),
                self.py_random.randint(margin, max_row),
            )

        edge = self.py_random.choice(["left", "right", "top", "bottom"])
        edge_points = {
            "left": (margin, self.py_random.randint(margin, max_row)),
            "right": (max_col, self.py_random.randint(margin, max_row)),
            "top": (self.py_random.randint(margin, max_col), margin),
            "bottom": (self.py_random.randint(margin, max_col), max_row),
        }
        return edge_points[edge]

    def _sample_arrow_end(self, start: Point, image_height: int, image_width: int) -> Point:
        start_col, start_row = start
        center_col = image_width / 2
        center_row = image_height / 2
        base_angle = np.arctan2(center_row - start_row, center_col - start_col)
        angle = base_angle + self.py_random.uniform(-np.pi / 4, np.pi / 4)
        length = int(self.py_random.uniform(*self.line_length_ratio_range) * min(image_width, image_height))
        end_col = int(start_col + length * np.cos(angle))
        end_row = int(start_row + length * np.sin(angle))

        return (
            int(np.clip(end_col, 0, image_width - 1)),
            int(np.clip(end_row, 0, image_height - 1)),
        )
