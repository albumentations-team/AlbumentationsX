"""Transform for synthetic annotation and callout artifacts."""

import string
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

from albumentations.augmentations.other import annotation_artifacts_functional as fannotation
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.type_definitions import ImageType

__all__ = ["AnnotationArtifacts"]

AnnotationElementType = Literal["text", "rectangle", "arrow", "line", "callout"]
LineStyle = Literal["solid", "dashed", "dotted"]
LineGeometry = Literal["axis_aligned", "random_endpoints", "random_angle"]
Point = tuple[int, int]
ColorValue = Annotated[int, Field(ge=0, le=255)]
ArtifactColor = Annotated[tuple[ColorValue, ...], Field(min_length=1)]
ColorPalette = Annotated[tuple[ArtifactColor, ...], Field(min_length=1)]
PixelLengthRange = Annotated[
    tuple[int, int],
    AfterValidator(check_range_bounds(1, None)),
    AfterValidator(nondecreasing),
]

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


def _validate_choice_weights(
    choices: tuple[Any, ...],
    weights: tuple[float, ...],
    choices_name: str,
    weights_name: str,
) -> None:
    if len(choices) != len(weights):
        raise ValueError(f"{choices_name} and {weights_name} must have the same length.")

    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError(f"{weights_name} must contain finite non-negative values.")

    if sum(weights) <= 0:
        raise ValueError(f"At least one weight in {weights_name} must be positive.")


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
        line_geometry (Literal["axis_aligned", "random_endpoints", "random_angle"]): Geometry used for line
            artifacts. `"axis_aligned"` preserves horizontal and vertical lines. `"random_endpoints"` samples
            both endpoints independently. `"random_angle"` samples a start point, angle, and length.
            Default: `"axis_aligned"`.
        line_styles (tuple[Literal["solid", "dashed", "dotted"], ...]): Line styles sampled for lines, arrows,
            and callouts. Default: ("solid", "dashed", "dotted").
        line_style_probabilities (tuple[float, ...]): Sampling weights matching `line_styles`. Values must be
            non-negative and at least one value must be positive. Default: (0.55, 0.35, 0.1).
        random_color_prob (float): Probability of sampling every image channel independently and uniformly from
            `[0, 255]`. The remaining probability uses `color_palette` when provided, otherwise the legacy
            black/white/red policy. Default: 0.0.
        color_palette (tuple[tuple[int, ...], ...] | None): Optional artifact colors in `[0, 255]`. Colors are
            truncated or extended using their last value to match the image channels. Default: None.
        color_palette_probabilities (tuple[float, ...] | None): Optional sampling weights matching
            `color_palette`. When omitted, palette colors are sampled uniformly. Default: None.
        line_length_range (tuple[int, int] | None): Optional length range in pixels for `"random_angle"` lines.
            When omitted, `line_length_ratio_range` controls their length. Default: None.
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
        - Palette and uniformly sampled colors use uint8-scale values. Float32 images are rendered through the
          transform's dtype adapter, which maps those values to `[0, 1]`.
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
        line_geometry: LineGeometry
        line_styles: tuple[LineStyle, ...] = Field(min_length=1)
        line_style_probabilities: tuple[float, ...] = Field(min_length=1)
        random_color_prob: float = Field(ge=0, le=1)
        color_palette: ColorPalette | None
        color_palette_probabilities: tuple[float, ...] | None
        line_length_range: PixelLengthRange | None

        @model_validator(mode="after")
        def _validate_sampling_policies(self) -> Self:
            _validate_choice_weights(
                self.element_types,
                self.element_probabilities,
                "element_types",
                "element_probabilities",
            )
            _validate_choice_weights(
                self.line_styles,
                self.line_style_probabilities,
                "line_styles",
                "line_style_probabilities",
            )

            if self.line_length_range is not None and self.line_geometry != "random_angle":
                raise ValueError("line_length_range requires line_geometry='random_angle'.")

            if self.color_palette_probabilities is not None:
                if self.color_palette is None:
                    raise ValueError("color_palette must be provided with color_palette_probabilities.")

                _validate_choice_weights(
                    self.color_palette,
                    self.color_palette_probabilities,
                    "color_palette",
                    "color_palette_probabilities",
                )

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
        *,
        line_geometry: LineGeometry = "axis_aligned",
        line_styles: tuple[LineStyle, ...] = LINE_STYLES,
        line_style_probabilities: tuple[float, ...] = LINE_STYLE_WEIGHTS,
        random_color_prob: float = 0.0,
        color_palette: ColorPalette | None = None,
        color_palette_probabilities: tuple[float, ...] | None = None,
        line_length_range: PixelLengthRange | None = None,
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
        self.line_geometry = line_geometry
        self.line_styles = line_styles
        self.line_style_probabilities = line_style_probabilities
        self.random_color_prob = random_color_prob
        self.color_palette = color_palette
        self.color_palette_probabilities = color_palette_probabilities
        self.line_length_range = line_length_range
        self._use_legacy_color_policy = random_color_prob == 0 and color_palette is None

    def apply(
        self,
        img: ImageType,
        artifacts: list[dict[str, Any]],
        **params: Any,
    ) -> ImageType:
        return fannotation.draw_annotation_artifacts(img, artifacts)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        shape = params["shape"]
        image_height, image_width = shape[:2]
        num_channels = shape[2] if len(shape) > 2 else 1
        artifacts = self._generate_artifacts(image_height, image_width, num_channels, sampling)
        record_line_length_ratio = self.line_geometry != "random_endpoints" and not (
            self.line_geometry == "random_angle" and self.line_length_range is not None
        )
        sampling.applied_overrides.update(
            self._get_applied_config(
                artifacts,
                image_height,
                image_width,
                record_line_length_ratio=record_line_length_ratio,
            )
        )
        return {"artifacts": artifacts}

    @staticmethod
    def _get_applied_config(
        artifacts: list[dict[str, Any]],
        image_height: int,
        image_width: int,
        *,
        record_line_length_ratio: bool,
    ) -> dict[str, Any]:
        def bounds(values: list[Any]) -> tuple[Any, Any] | None:
            flattened = [item for value in values for item in (value if isinstance(value, tuple) else (value,))]
            return (min(flattened), max(flattened)) if flattened else None

        min_dimension = min(image_height, image_width)
        sampled_values = {
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
            "line_length_ratio_range": (
                [
                    np.hypot(
                        artifact["end"][0] - artifact["start"][0],
                        artifact["end"][1] - artifact["start"][1],
                    )
                    / min_dimension
                    for artifact in artifacts
                    if artifact["type"] in {"arrow", "line"} and min_dimension > 0
                ]
                if record_line_length_ratio
                else []
            ),
            "tip_length_range": [artifact["tip_length"] for artifact in artifacts if artifact["type"] == "arrow"],
        }
        result: dict[str, Any] = {"count_range": len(artifacts)}
        result.update(
            (name, sampled_bounds)
            for name, values in sampled_values.items()
            if (sampled_bounds := bounds(values)) is not None
        )
        return result

    def _generate_artifacts(
        self,
        image_height: int,
        image_width: int,
        num_channels: int,
        sampling: SamplingContext,
    ) -> list[dict[str, Any]]:
        if image_height <= 1 or image_width <= 1:
            return []

        artifact_count = sampling.py_random.randint(*self.count_range)
        artifact_types = sampling.py_random.choices(
            self.element_types,
            weights=self.element_probabilities,
            k=artifact_count,
        )

        artifacts = []
        for artifact_type in artifact_types:
            artifact = self._generate_artifact(artifact_type, image_height, image_width, num_channels, sampling)
            if artifact is not None:
                artifacts.append(artifact)

        return artifacts

    def _generate_artifact(
        self,
        artifact_type: AnnotationElementType,
        image_height: int,
        image_width: int,
        num_channels: int,
        sampling: SamplingContext,
    ) -> dict[str, Any] | None:
        generators = {
            "text": self._generate_text_artifact,
            "rectangle": self._generate_rectangle_artifact,
            "arrow": self._generate_arrow_artifact,
            "line": self._generate_line_artifact,
            "callout": self._generate_callout_artifact,
        }
        return generators[artifact_type](image_height, image_width, num_channels, sampling)

    def _generate_text_artifact(
        self,
        image_height: int,
        image_width: int,
        num_channels: int,
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        text_length = sampling.py_random.randint(*self.text_length_range)
        text = "".join(sampling.py_random.choices(TEXT_ALPHABET, k=text_length))
        font = sampling.py_random.choice(FONT_OPTIONS)
        font_scale = sampling.py_random.uniform(*self.font_scale_range)
        thickness = sampling.py_random.randint(*self.thickness_range)
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        margin = self._sample_margin(image_height, image_width, sampling)
        origin = self._sample_text_origin(image_height, image_width, text_width, text_height, margin, sampling)

        return {
            "type": "text",
            "text": text,
            "origin": origin,
            "font": font,
            "font_scale": font_scale,
            "color": self._sample_color(num_channels, sampling),
            "thickness": thickness,
        }

    def _generate_rectangle_artifact(
        self,
        image_height: int,
        image_width: int,
        num_channels: int,
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        box_width, box_height = self._sample_box_size(image_height, image_width, sampling)
        top_left = self._sample_box_origin(image_height, image_width, box_width, box_height, sampling)
        top_left_col, top_left_row = top_left
        bottom_right = (top_left_col + box_width, top_left_row + box_height)

        return {
            "type": "rectangle",
            "top_left": top_left,
            "bottom_right": bottom_right,
            "color": self._sample_color(num_channels, sampling),
            "thickness": sampling.py_random.randint(*self.thickness_range),
            "filled": False,
        }

    def _generate_callout_artifact(
        self,
        image_height: int,
        image_width: int,
        num_channels: int,
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        artifact = self._generate_rectangle_artifact(image_height, image_width, num_channels, sampling)
        artifact["type"] = "callout"
        artifact["style"] = self._sample_line_style(sampling)
        artifact["lines"] = self._sample_callout_lines(artifact, image_height, image_width, sampling)
        return artifact

    def _generate_line_artifact(
        self,
        image_height: int,
        image_width: int,
        num_channels: int,
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        if self.line_geometry == "axis_aligned":
            margin = self._sample_margin(image_height, image_width, sampling)
            is_vertical = sampling.py_random.choice([True, False])
            line_length = self._sample_line_length(image_height, image_width, margin, is_vertical, sampling)

            if is_vertical:
                min_row = margin
                max_row = max(margin, image_height - 1 - margin)
                start_row = sampling.py_random.randint(min_row, max_row - line_length)
                line_col = sampling.py_random.randint(margin, max(margin, image_width - 1 - margin))
                start = (line_col, start_row)
                end = (line_col, start_row + line_length)
            else:
                min_col = margin
                max_col = max(margin, image_width - 1 - margin)
                start_col = sampling.py_random.randint(min_col, max_col - line_length)
                line_row = sampling.py_random.randint(margin, max(margin, image_height - 1 - margin))
                start = (start_col, line_row)
                end = (start_col + line_length, line_row)
        elif self.line_geometry == "random_endpoints":
            start, end = self._sample_random_endpoints(image_height, image_width, sampling)
        else:
            start, end = self._sample_random_angle_line(image_height, image_width, sampling)

        return {
            "type": "line",
            "start": start,
            "end": end,
            "color": self._sample_color(num_channels, sampling),
            "thickness": sampling.py_random.randint(*self.thickness_range),
            "style": self._sample_line_style(sampling),
        }

    def _sample_random_endpoints(
        self, image_height: int, image_width: int, sampling: SamplingContext
    ) -> tuple[Point, Point]:
        return (
            (sampling.py_random.randint(0, image_width - 1), sampling.py_random.randint(0, image_height - 1)),
            (sampling.py_random.randint(0, image_width - 1), sampling.py_random.randint(0, image_height - 1)),
        )

    def _sample_random_angle_line(
        self, image_height: int, image_width: int, sampling: SamplingContext
    ) -> tuple[Point, Point]:
        start_col = sampling.py_random.randint(0, image_width - 1)
        start_row = sampling.py_random.randint(0, image_height - 1)
        angle = sampling.py_random.uniform(0, 2 * np.pi)
        line_length = self._sample_unbounded_line_length(image_height, image_width, sampling)
        end = (
            round(start_col + line_length * np.cos(angle)),
            round(start_row + line_length * np.sin(angle)),
        )
        return (start_col, start_row), end

    def _sample_line_length(
        self,
        image_height: int,
        image_width: int,
        margin: int,
        is_vertical: bool,
        sampling: SamplingContext,
    ) -> int:
        max_length = max(0, (image_height if is_vertical else image_width) - 1 - (2 * margin))
        sampled_length = round(
            sampling.py_random.uniform(*self.line_length_ratio_range) * min(image_height, image_width)
        )
        return min(max_length, max(1, sampled_length))

    def _sample_unbounded_line_length(self, image_height: int, image_width: int, sampling: SamplingContext) -> int:
        if self.line_length_range is not None:
            return sampling.py_random.randint(*self.line_length_range)

        return max(1, round(sampling.py_random.uniform(*self.line_length_ratio_range) * min(image_height, image_width)))

    def _generate_arrow_artifact(
        self,
        image_height: int,
        image_width: int,
        num_channels: int,
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        start = self._sample_arrow_start(image_height, image_width, sampling)
        end = self._sample_arrow_end(start, image_height, image_width, sampling)

        return {
            "type": "arrow",
            "start": start,
            "end": end,
            "color": self._sample_color(num_channels, sampling),
            "thickness": sampling.py_random.randint(*self.thickness_range),
            "tip_length": sampling.py_random.uniform(*self.tip_length_range),
            "style": self._sample_line_style(sampling),
        }

    def _sample_color(self, num_channels: int, sampling: SamplingContext) -> tuple[int, ...]:
        if self._use_legacy_color_policy:
            if sampling.py_random.random() < self.black_white_prob:
                return sampling.py_random.choice([BLACK, WHITE])

            return RED

        if self.random_color_prob == 1 or (
            self.random_color_prob > 0 and sampling.py_random.random() < self.random_color_prob
        ):
            return tuple(sampling.py_random.randint(0, 255) for _ in range(num_channels))

        if self.color_palette is not None:
            if self.color_palette_probabilities is None:
                return sampling.py_random.choice(self.color_palette)

            return sampling.py_random.choices(
                self.color_palette,
                weights=self.color_palette_probabilities,
                k=1,
            )[0]

        return sampling.py_random.choice([BLACK, WHITE]) if sampling.py_random.random() < self.black_white_prob else RED

    def _sample_line_style(self, sampling: SamplingContext) -> LineStyle:
        return sampling.py_random.choices(self.line_styles, weights=self.line_style_probabilities, k=1)[0]

    def _sample_margin(self, image_height: int, image_width: int, sampling: SamplingContext) -> int:
        max_margin = max(1, min(10, min(image_height, image_width) // 8))
        return sampling.py_random.randint(1, max_margin)

    def _sample_box_size(self, image_height: int, image_width: int, sampling: SamplingContext) -> tuple[int, int]:
        min_ratio, max_ratio = self.size_ratio_range
        box_width = max(1, int(sampling.py_random.uniform(min_ratio, max_ratio) * image_width))
        box_height = max(1, int(sampling.py_random.uniform(min_ratio, max_ratio) * image_height))
        return min(box_width, image_width - 1), min(box_height, image_height - 1)

    def _sample_box_origin(
        self,
        image_height: int,
        image_width: int,
        box_width: int,
        box_height: int,
        sampling: SamplingContext,
    ) -> Point:
        max_origin_col = max(0, image_width - 1 - box_width)
        max_origin_row = max(0, image_height - 1 - box_height)
        margin = self._sample_margin(image_height, image_width, sampling)

        if sampling.py_random.random() < self.corner_prob:
            return self._sample_corner_origin(max_origin_col, max_origin_row, margin, sampling)

        return (
            sampling.py_random.randint(0, max_origin_col),
            sampling.py_random.randint(0, max_origin_row),
        )

    def _sample_corner_origin(
        self, max_origin_col: int, max_origin_row: int, margin: int, sampling: SamplingContext
    ) -> Point:
        corner = sampling.py_random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
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
        sampling: SamplingContext,
    ) -> Point:
        max_origin_col = max(0, image_width - 1 - text_width)
        min_origin_row = min(image_height - 1, text_height)
        max_origin_row = max(min_origin_row, image_height - 1 - margin)

        if sampling.py_random.random() < self.corner_prob:
            corner = sampling.py_random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
            origin_col = min(margin, max_origin_col) if "left" in corner else max(0, max_origin_col - margin)
            origin_row = min_origin_row + margin if "top" in corner else max_origin_row
            return (origin_col, min(origin_row, image_height - 1))

        return (
            sampling.py_random.randint(0, max_origin_col),
            sampling.py_random.randint(min_origin_row, max_origin_row),
        )

    def _sample_callout_lines(
        self,
        artifact: dict[str, Any],
        image_height: int,
        image_width: int,
        sampling: SamplingContext,
    ) -> list[tuple[Point, Point]]:
        top_left_col, top_left_row = artifact["top_left"]
        bottom_right_col, bottom_right_row = artifact["bottom_right"]
        side_count = sampling.py_random.randint(1, 2)
        sides = sampling.py_random.sample(["left", "right", "top", "bottom"], side_count)
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

    def _sample_arrow_start(self, image_height: int, image_width: int, sampling: SamplingContext) -> Point:
        margin = self._sample_margin(image_height, image_width, sampling)
        max_col = max(margin, image_width - 1 - margin)
        max_row = max(margin, image_height - 1 - margin)

        if sampling.py_random.random() >= self.corner_prob:
            return (
                sampling.py_random.randint(margin, max_col),
                sampling.py_random.randint(margin, max_row),
            )

        edge = sampling.py_random.choice(["left", "right", "top", "bottom"])
        edge_points = {
            "left": (margin, sampling.py_random.randint(margin, max_row)),
            "right": (max_col, sampling.py_random.randint(margin, max_row)),
            "top": (sampling.py_random.randint(margin, max_col), margin),
            "bottom": (sampling.py_random.randint(margin, max_col), max_row),
        }
        return edge_points[edge]

    def _sample_arrow_end(self, start: Point, image_height: int, image_width: int, sampling: SamplingContext) -> Point:
        start_col, start_row = start
        center_col = image_width / 2
        center_row = image_height / 2
        base_angle = np.arctan2(center_row - start_row, center_col - start_col)
        angle = base_angle + sampling.py_random.uniform(-np.pi / 4, np.pi / 4)
        length = int(sampling.py_random.uniform(*self.line_length_ratio_range) * min(image_width, image_height))
        end_col = int(start_col + length * np.cos(angle))
        end_row = int(start_row + length * np.sin(angle))

        return (
            int(np.clip(end_col, 0, image_width - 1)),
            int(np.clip(end_row, 0, image_height - 1)),
        )
