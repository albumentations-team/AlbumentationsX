"""Functional helpers for drawing synthetic annotation artifacts."""

from collections.abc import Callable
from typing import Any, Literal

import cv2
import numpy as np
from albucore import preserve_channel_dim, uint8_io

from albumentations.core.type_definitions import ImageType

LineStyle = Literal["solid", "dashed", "dotted"]
Point = tuple[int, int]


def _clip_point(point: Point, image_shape: tuple[int, int]) -> Point:
    image_height, image_width = image_shape
    point_col, point_row = point
    return (
        int(np.clip(point_col, 0, max(image_width - 1, 0))),
        int(np.clip(point_row, 0, max(image_height - 1, 0))),
    )


def _normalize_color(color: tuple[int, ...], num_channels: int) -> tuple[int, ...]:
    if len(color) >= num_channels:
        return color[:num_channels]

    return (*color, *([color[-1]] * (num_channels - len(color))))


def _draw_channelwise(
    image: np.ndarray,
    color: tuple[int, ...],
    draw_fn: Callable[[np.ndarray, int], None],
) -> None:
    if image.ndim == 2:
        draw_fn(image, color[0])
        return

    channel_colors = _normalize_color(color, image.shape[2])
    for channel_index, channel_color in enumerate(channel_colors):
        channel = np.ascontiguousarray(image[..., channel_index])
        draw_fn(channel, channel_color)
        image[..., channel_index] = channel


def _draw_solid_line(
    image: np.ndarray,
    start: Point,
    end: Point,
    color: tuple[int, ...],
    thickness: int,
) -> None:
    image_shape = image.shape[:2]
    clipped_start = _clip_point(start, image_shape)
    clipped_end = _clip_point(end, image_shape)

    def draw(channel: np.ndarray, channel_color: int) -> None:
        cv2.line(channel, clipped_start, clipped_end, channel_color, thickness, cv2.LINE_AA)

    _draw_channelwise(image, color, draw)


def _draw_dashed_line(
    image: np.ndarray,
    start: Point,
    end: Point,
    color: tuple[int, ...],
    thickness: int,
    dash_length: int = 8,
    gap_length: int = 6,
) -> None:
    image_shape = image.shape[:2]
    clipped_start = _clip_point(start, image_shape)
    clipped_end = _clip_point(end, image_shape)
    start_col, start_row = clipped_start
    end_col, end_row = clipped_end
    distance = int(np.hypot(end_col - start_col, end_row - start_row))

    if distance == 0:
        return

    for segment_start in range(0, distance, dash_length + gap_length):
        start_fraction = segment_start / distance
        end_fraction = min(segment_start + dash_length, distance) / distance

        dash_start = (
            int(start_col + (end_col - start_col) * start_fraction),
            int(start_row + (end_row - start_row) * start_fraction),
        )
        dash_end = (
            int(start_col + (end_col - start_col) * end_fraction),
            int(start_row + (end_row - start_row) * end_fraction),
        )

        _draw_solid_line(image, dash_start, dash_end, color, thickness)


def _draw_dotted_line(
    image: np.ndarray,
    start: Point,
    end: Point,
    color: tuple[int, ...],
    thickness: int,
    dot_spacing: int = 8,
) -> None:
    image_shape = image.shape[:2]
    clipped_start = _clip_point(start, image_shape)
    clipped_end = _clip_point(end, image_shape)
    start_col, start_row = clipped_start
    end_col, end_row = clipped_end
    distance = int(np.hypot(end_col - start_col, end_row - start_row))

    if distance == 0:
        return

    radius = max(1, thickness // 2)

    def draw_dot(center: Point) -> None:
        def draw(channel: np.ndarray, channel_color: int) -> None:
            cv2.circle(channel, center, radius, channel_color, -1, cv2.LINE_AA)

        _draw_channelwise(image, color, draw)

    for dot_position in range(0, distance, dot_spacing):
        dot_fraction = dot_position / distance
        dot_center = (
            int(start_col + (end_col - start_col) * dot_fraction),
            int(start_row + (end_row - start_row) * dot_fraction),
        )
        draw_dot(dot_center)


def _draw_styled_line(
    image: np.ndarray,
    start: Point,
    end: Point,
    color: tuple[int, ...],
    thickness: int,
    style: LineStyle,
) -> None:
    if style == "solid":
        _draw_solid_line(image, start, end, color, thickness)
    elif style == "dashed":
        _draw_dashed_line(image, start, end, color, thickness)
    else:
        _draw_dotted_line(image, start, end, color, thickness)


def _draw_arrow_head(
    image: np.ndarray,
    start: Point,
    end: Point,
    color: tuple[int, ...],
    thickness: int,
    tip_length: float,
) -> None:
    start_col, start_row = start
    end_col, end_row = end
    distance = int(np.hypot(end_col - start_col, end_row - start_row))

    if distance == 0:
        return

    arrow_length = max(1.0, distance * tip_length)
    arrow_angle = np.arctan2(end_row - start_row, end_col - start_col)
    left_tip = (
        int(end_col - arrow_length * np.cos(arrow_angle - np.pi / 6)),
        int(end_row - arrow_length * np.sin(arrow_angle - np.pi / 6)),
    )
    right_tip = (
        int(end_col - arrow_length * np.cos(arrow_angle + np.pi / 6)),
        int(end_row - arrow_length * np.sin(arrow_angle + np.pi / 6)),
    )

    _draw_solid_line(image, end, left_tip, color, thickness)
    _draw_solid_line(image, end, right_tip, color, thickness)


def _draw_arrow(
    image: np.ndarray,
    artifact: dict[str, Any],
) -> None:
    start = artifact["start"]
    end = artifact["end"]
    color = artifact["color"]
    thickness = artifact["thickness"]
    style = artifact["style"]
    tip_length = artifact["tip_length"]

    if style == "solid":
        clipped_start = _clip_point(start, image.shape[:2])
        clipped_end = _clip_point(end, image.shape[:2])

        def draw(channel: np.ndarray, channel_color: int) -> None:
            cv2.arrowedLine(channel, clipped_start, clipped_end, channel_color, thickness, cv2.LINE_AA, 0, tip_length)

        _draw_channelwise(image, color, draw)
        return

    _draw_styled_line(image, start, end, color, thickness, style)
    _draw_arrow_head(image, start, end, color, thickness, tip_length)


def _draw_text(
    image: np.ndarray,
    artifact: dict[str, Any],
) -> None:
    origin = _clip_point(artifact["origin"], image.shape[:2])
    text = artifact["text"]
    font = artifact["font"]
    font_scale = artifact["font_scale"]
    color = artifact["color"]
    thickness = artifact["thickness"]

    def draw(channel: np.ndarray, channel_color: int) -> None:
        cv2.putText(channel, text, origin, font, font_scale, channel_color, thickness, cv2.LINE_AA)

    _draw_channelwise(image, color, draw)


def _draw_rectangle(
    image: np.ndarray,
    artifact: dict[str, Any],
) -> None:
    top_left = _clip_point(artifact["top_left"], image.shape[:2])
    bottom_right = _clip_point(artifact["bottom_right"], image.shape[:2])
    color = artifact["color"]
    thickness = artifact["thickness"]
    line_thickness = -1 if artifact.get("filled", False) else thickness

    def draw(channel: np.ndarray, channel_color: int) -> None:
        cv2.rectangle(channel, top_left, bottom_right, channel_color, line_thickness, cv2.LINE_AA)

    _draw_channelwise(image, color, draw)


def _draw_line(
    image: np.ndarray,
    artifact: dict[str, Any],
) -> None:
    _draw_styled_line(
        image,
        artifact["start"],
        artifact["end"],
        artifact["color"],
        artifact["thickness"],
        artifact["style"],
    )


def _draw_callout(
    image: np.ndarray,
    artifact: dict[str, Any],
) -> None:
    _draw_rectangle(image, artifact)

    for start, end in artifact["lines"]:
        _draw_styled_line(image, start, end, artifact["color"], artifact["thickness"], artifact["style"])


@uint8_io
@preserve_channel_dim
def draw_annotation_artifacts(image: ImageType, artifacts: list[dict[str, Any]]) -> ImageType:
    """Draw generated annotation commands with OpenCV primitives. Used by markup-style
    augmentations that add text, arrows, boxes, lines, or callouts.

    Args:
        image (ImageType): Input image.
        artifacts (list[dict[str, Any]]): Artifact drawing commands generated by the transform.

    Returns:
        ImageType: Image with annotation artifacts drawn.

    """
    result = image.copy()
    draw_fns = {
        "text": _draw_text,
        "rectangle": _draw_rectangle,
        "arrow": _draw_arrow,
        "line": _draw_line,
        "callout": _draw_callout,
    }

    for artifact in artifacts:
        draw_fns[artifact["type"]](result, artifact)

    return result
