"""Functional implementations for text manipulation and rendering.

This module provides utility functions for manipulating text in strings and
rendering text onto images. Includes functions for word manipulation, text drawing,
and handling text regions in images.
"""

import random
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from albucore import (
    NUM_RGB_CHANNELS,
    preserve_channel_dim,
    uint8_io,
)

from albumentations.core.type_definitions import PAIR, ImageType

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def delete_random_words(words: list[str], num_words: int, py_random: random.Random) -> str:
    """Delete num_words random words from list. py_random for reproducibility. Returns joined string.
    Used by TextImage (deletion). Empty if num_words >= len(words).

    This function randomly removes words from the input list and joins the remaining
    words with spaces to form a new string.

    Args:
        words (list[str]): List of words to process.
        num_words (int): Number of words to delete.
        py_random (random.Random): Random number generator for reproducibility.

    Returns:
        str: New string with specified words removed. Returns empty string if
             num_words is greater than or equal to the length of words.

    """
    if num_words >= len(words):
        return ""

    indices_to_delete = py_random.sample(range(len(words)), num_words)
    new_words = [word for idx, word in enumerate(words) if idx not in indices_to_delete]
    return " ".join(new_words)


def swap_random_words(words: list[str], num_words: int, py_random: random.Random) -> str:
    """Swap random pairs of words. num_words swaps; py_random. Used by TextImage (swap). Returns
    original if num_words 0 or list has fewer than 2 words.

    This function randomly selects pairs of words and swaps their positions
    a specified number of times.

    Args:
        words (list[str]): List of words to process.
        num_words (int): Number of swaps to perform.
        py_random (random.Random): Random number generator for reproducibility.

    Returns:
        str: New string with words swapped. If num_words is 0 or the list has fewer
             than 2 words, returns the original string.

    """
    if num_words == 0 or len(words) < PAIR:
        return " ".join(words)

    words = words.copy()

    for _ in range(num_words):
        idx1, idx2 = py_random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def insert_random_stopwords(
    words: list[str],
    num_insertions: int,
    stopwords: tuple[str, ...] | None,
    py_random: random.Random,
) -> str:
    """Insert random stopwords into word list. num_insertions, stopwords, py_random. Used by TextImage
    (insertion). Returns string with stopwords at random positions.

    This function randomly inserts stopwords at random positions in the
    list of words a specified number of times.

    Args:
        words (list[str]): List of words to process.
        num_insertions (int): Number of stopwords to insert.
        stopwords (tuple[str, ...] | None): Tuple of stopwords to choose from.
            If None, default stopwords will be used.
        py_random (random.Random): Random number generator for reproducibility.

    Returns:
        str: New string with stopwords inserted.

    """
    if stopwords is None:
        stopwords = ("and", "the", "is", "in", "at", "of")  # Default stopwords if none provided

    for _ in range(num_insertions):
        idx = py_random.randint(0, len(words))
        words.insert(idx, py_random.choice(stopwords))
    return " ".join(words)


def convert_image_to_pil(image: ImageType) -> "PILImage":
    """Convert a NumPy array image (H,W,C) to a PIL Image. Grayscale (C=1) or RGB (C=3). Used by
    render_text for text drawing. Requires Pillow.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is not installed") from ImportError

    # All images now have channel dimension (H, W, C)
    if image.shape[2] == 1:  # Grayscale (height, width, 1)
        return Image.fromarray(image[:, :, 0], mode="L")
    if image.shape[2] == NUM_RGB_CHANNELS:  # RGB (height, width, 3)
        return Image.fromarray(image)

    raise TypeError(f"Unsupported image shape: {image.shape}")


def draw_text_on_pil_image(pil_image: "PILImage", metadata_list: list[dict[str, Any]]) -> "PILImage":
    """Draw text on PIL image from metadata_list (bbox_coords, text, font, font_color). Mutates image.
    Used by render_text for grayscale and RGB. Requires Pillow.
    """
    try:
        from PIL import ImageDraw
    except ImportError:
        raise ImportError("Pillow is not installed") from ImportError

    draw = ImageDraw.Draw(pil_image)
    for metadata in metadata_list:
        bbox_coords = metadata["bbox_coords"]
        text = metadata["text"]
        font = metadata["font"]
        font_color = metadata["font_color"]

        # Adapt font_color based on image mode
        if pil_image.mode == "L":  # Grayscale
            # For grayscale images, use only the first value or average the RGB values
            if isinstance(font_color, tuple):
                if len(font_color) >= 3:
                    # Average RGB values for grayscale
                    font_color = int(sum(font_color[:3]) / 3)
                elif len(font_color) == 1:
                    font_color = int(font_color[0])
        # For RGB and other modes, ensure font_color is a tuple of integers
        elif isinstance(font_color, tuple):
            font_color = tuple(int(c) for c in font_color)

        position = bbox_coords[:2]
        draw.text(position, text, font=font, fill=font_color)
    return pil_image


def draw_text_on_multi_channel_image(image: ImageType, metadata_list: list[dict[str, Any]]) -> ImageType:
    """Draw text on multi-channel image (C>3). Per-channel font_color; returns numpy array. Used by
    render_text when C>3. Requires Pillow.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("Pillow is not installed") from ImportError

    channels = [Image.fromarray(image[:, :, i]) for i in range(image.shape[2])]
    pil_images = [ImageDraw.Draw(channel) for channel in channels]

    for metadata in metadata_list:
        bbox_coords = metadata["bbox_coords"]
        text = metadata["text"]
        font = metadata["font"]
        font_color = metadata["font_color"]

        # Handle font_color as tuple[float, ...]
        # Ensure we have enough color values for all channels
        if len(font_color) < image.shape[2]:
            # If fewer values than channels, pad with zeros
            font_color = tuple(list(font_color) + [0] * (image.shape[2] - len(font_color)))
        elif len(font_color) > image.shape[2]:
            # If more values than channels, truncate
            font_color = font_color[: image.shape[2]]

        # Convert to integers for PIL
        font_color = [int(c) for c in font_color]

        position = bbox_coords[:2]

        # For each channel, use the corresponding color value
        for channel_id, pil_image in enumerate(pil_images):
            # For single-channel PIL images, color must be an integer
            pil_image.text(position, text, font=font, fill=font_color[channel_id])

    # Determine dtype from first channel (PIL Images from uint8 arrays return uint8)
    first_array = np.array(channels[0])
    result = np.empty((*channels[0].size[::-1], len(channels)), dtype=first_array.dtype)
    result[..., 0] = first_array
    for i in range(1, len(channels)):
        result[..., i] = np.array(channels[i])
    return result


@uint8_io
@preserve_channel_dim
def render_text(image: ImageType, metadata_list: list[dict[str, Any]], clear_bg: bool) -> ImageType:
    """Render text onto image from metadata_list (bbox_coords, text, font, font_color). clear_bg: inpaint
    first. Grayscale, RGB, multi-channel. uint8 I/O.

    This function draws text on an image using metadata that specifies text content,
    position, font, and color. It can optionally clear the background before rendering.
    The function handles different image types (grayscale, RGB, multi-channel).

    Args:
        image (ImageType): Image to draw text on.
        metadata_list (list[dict[str, Any]]): List of metadata dictionaries containing:
            - bbox_coords: Bounding box coordinates (x_min, y_min, x_max, y_max)
            - text: Text string to render
            - font: PIL ImageFont object
            - font_color: Color for the text
        clear_bg (bool): Whether to clear (inpaint) the background under the text.

    Returns:
        ImageType: Image with text rendered on it.

    """
    # First clean background under boxes using seamless clone if clear_bg is True
    if clear_bg:
        image = inpaint_text_background(image, metadata_list)

    # All images now have channel dimension (H, W, C)
    if image.shape[-1] in {1, NUM_RGB_CHANNELS}:
        pil_image = convert_image_to_pil(image)
        pil_image = draw_text_on_pil_image(pil_image, metadata_list)
        return np.array(pil_image)

    return draw_text_on_multi_channel_image(image, metadata_list)


def inpaint_text_background(
    image: np.ndarray,
    metadata_list: list[dict[str, Any]],
    method: int = cv2.INPAINT_TELEA,
) -> np.ndarray:
    """Inpaint (clear) regions where text will be rendered. metadata_list bbox regions; method
    INPAINT_TELEA or INPAINT_NS. Before render_text when clear_bg True.

    This function creates a clean background for text by inpainting rectangular
    regions specified in the metadata. It removes any existing content in those
    regions to provide a clean slate for rendering text.

    Args:
        image (np.ndarray): Image to inpaint.
        metadata_list (list[dict[str, Any]]): List of metadata dictionaries containing:
            - bbox_coords: Bounding box coordinates (x_min, y_min, x_max, y_max)
        method (int, optional): Inpainting method to use. Defaults to cv2.INPAINT_TELEA.
            Options include:
            - cv2.INPAINT_TELEA: Fast Marching Method
            - cv2.INPAINT_NS: Navier-Stokes method

    Returns:
        np.ndarray: Image with specified regions inpainted.

    """
    result_image = image.copy()
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for metadata in metadata_list:
        x_min, y_min, x_max, y_max = metadata["bbox_coords"]

        # Black out the region
        result_image[y_min:y_max, x_min:x_max] = 0

        # Update the mask to indicate the region to inpaint
        mask[y_min:y_max, x_min:x_max] = 255

    # Inpaint the blacked-out regions
    return cv2.inpaint(result_image, mask, inpaintRadius=3, flags=method)
