"""Weather and outdoor artifact functional helpers."""

from __future__ import annotations

from typing import Any

from ._functional_color import (
    equalize,
)
from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    ImageType,
    add,
    add_weighted,
    clip,
    clipped,
    cv2,
    float32_io,
    maybe_process_in_chunks,
    non_rgb_error,
    np,
    preserve_channel_dim,
    reduce_sum,
    uint8_io,
)
from ._functional_sharpness import (
    convolve,
)


@uint8_io
def add_snow_bleach(
    img: ImageType,
    snow_point: float,
    brightness_coeff: float,
) -> ImageType:
    """Add a simple snow effect by bleaching out pixels. Brightness increase and
    optional mask; used as a building block for more complex snow augmentations.

    This function simulates a basic snow effect by increasing the brightness of pixels
    that are above a certain threshold (snow_point). It operates in the HLS color space
    to modify the lightness channel.

    Args:
        img (ImageType): Input image. Can be either RGB uint8 or float32.
        snow_point (float): A float in the range [0, 1], scaled and adjusted to determine
            the threshold for pixel modification. Higher values result in less snow effect.
        brightness_coeff (float): Coefficient applied to increase the brightness of pixels
            below the snow_point threshold. Larger values lead to more pronounced snow effects.
            Should be greater than 1.0 for a visible effect.

    Returns:
        ImageType: Image with simulated snow effect. The output has the same dtype as the input.

    Note:
        - This function converts the image to the HLS color space to modify the lightness channel.
        - The snow effect is created by selectively increasing the brightness of pixels.
        - This method tends to create a 'bleached' look, which may not be as realistic as more
          advanced snow simulation techniques.
        - The function automatically handles both uint8 and float32 input images.

    The snow effect is created through the following steps:
    1. Convert the image from RGB to HLS color space.
    2. Adjust the snow_point threshold.
    3. Increase the lightness of pixels below the threshold.
    4. Convert the image back to RGB.

    Mathematical Formulation:
        Let L be the lightness channel in HLS space.
        For each pixel (i, j):
        If L[i, j] < snow_point:
            L[i, j] = L[i, j] * brightness_coeff

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> snowy_image = A.functional.add_snow_v1(image, snow_point=0.5, brightness_coeff=1.5)

    References:
        - HLS Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV
        - Original implementation: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    # Precompute snow_point threshold
    snow_point = (snow_point * max_value / 2) + (max_value / 3)

    # Convert image to HLS color space once and avoid repeated dtype casting
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lightness_channel = image_hls[:, :, 1].astype(np.float32)

    # Utilize boolean indexing for efficient lightness adjustment
    mask = lightness_channel < snow_point
    lightness_channel[mask] *= brightness_coeff

    # Clip the lightness values in place
    lightness_channel = clip(lightness_channel, np.uint8, inplace=True)

    # Update the lightness channel in the original image
    image_hls[:, :, 1] = lightness_channel

    # Convert back to RGB
    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def generate_snow_textures(
    img_shape: tuple[int, int],
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate snow texture and sparkle mask for add_snow_texture. Returns texture
    and mask arrays; uses random generator for reproducibility.

    Args:
        img_shape (tuple[int, int]): Image shape.
        random_generator (np.random.Generator): Random generator to use.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (snow_texture, sparkle_mask) arrays.

    """
    # Generate base snow texture
    snow_texture = random_generator.normal(size=img_shape[:2], loc=0.5, scale=0.3)
    snow_texture = cv2.GaussianBlur(snow_texture, (0, 0), sigmaX=1, sigmaY=1)

    # Generate sparkle mask
    sparkle_mask = random_generator.random(img_shape[:2]) > 0.99

    return snow_texture, sparkle_mask


@uint8_io
def add_snow_texture(
    img: ImageType,
    snow_point: float,
    brightness_coeff: float,
    snow_texture: np.ndarray,
    sparkle_mask: np.ndarray,
) -> ImageType:
    """Add snow effect: texture overlay, sparkle, depth gradient, blue tint. snow_point,
    brightness_coeff; takes precomputed snow_texture and sparkle_mask. uint8 I/O.

    This function simulates snowfall by applying multiple visual effects to the image,
    including brightness adjustment, snow texture overlay, depth simulation, and color tinting.
    The result is a more natural-looking snow effect compared to simple pixel bleaching methods.

    Args:
        img (ImageType): Input image in RGB format.
        snow_point (float): Coefficient that controls the amount and intensity of snow.
            Should be in the range [0, 1], where 0 means no snow and 1 means maximum snow effect.
        brightness_coeff (float): Coefficient for brightness adjustment to simulate the
            reflective nature of snow. Should be in the range [0, 1], where higher values
            result in a brighter image.
        snow_texture (np.ndarray): Snow texture.
        sparkle_mask (np.ndarray): Sparkle mask.

    Returns:
        ImageType: Image with added snow effect. The output has the same dtype as the input.

    Note:
        - The function first converts the image to HSV color space for better control over
          brightness and color adjustments.
        - A snow texture is generated using Gaussian noise and then filtered for a more
          natural appearance.
        - A depth effect is simulated, with more snow at the top of the image and less at the bottom.
        - A slight blue tint is added to simulate the cool color of snow.
        - Random sparkle effects are added to simulate light reflecting off snow crystals.

    The snow effect is created through the following steps:
    1. Brightness adjustment in HSV space
    2. Generation of a snow texture using Gaussian noise
    3. Application of a depth effect to the snow texture
    4. Blending of the snow texture with the original image
    5. Addition of a cool blue tint
    6. Addition of sparkle effects

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> snowy_image = A.functional.add_snow_v2(image, snow_coeff=0.5, brightness_coeff=0.2)

    Note:
        This function works with both uint8 and float32 image types, automatically
        handling the conversion between them.

    References:
        - Perlin Noise: https://en.wikipedia.org/wiki/Perlin_noise
        - HSV Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    # Convert to HSV for better color control
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Increase brightness
    np.multiply(img_hsv[:, :, 2], 1 + brightness_coeff * snow_point, out=img_hsv[:, :, 2])
    np.clip(img_hsv[:, :, 2], 0, max_value, out=img_hsv[:, :, 2])

    # Generate snow texture
    snow_texture = cv2.GaussianBlur(snow_texture, (0, 0), sigmaX=1, sigmaY=1)

    # Create depth effect for snow simulation
    # More snow accumulates at the top of the image, gradually decreasing towards the bottom
    # This simulates natural snow distribution on surfaces
    # The effect is achieved using a linear gradient from 1 (full snow) to 0.2 (less snow)
    rows = img.shape[0]
    depth_effect = np.linspace(1, 0.2, rows)[:, np.newaxis]
    snow_texture *= depth_effect

    # Apply snow texture
    snow_layer = (snow_texture[:, :, np.newaxis] * (max_value * snow_point)).astype(
        np.float32,
    )

    # Blend snow with original image
    img_with_snow = cv2.add(img_hsv, snow_layer)

    # Add a slight blue tint to simulate cool snow color
    blue_tint = np.full_like(img_with_snow, (0.6, 0.75, 1))  # Slight blue in HSV

    img_with_snow = cv2.addWeighted(
        img_with_snow,
        0.85,
        blue_tint,
        0.15 * snow_point,
        0,
    )

    # Convert back to RGB
    img_with_snow = cv2.cvtColor(img_with_snow.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Add some sparkle effects for snow glitter
    img_with_snow[sparkle_mask] = [max_value, max_value, max_value]

    return img_with_snow


@uint8_io
@preserve_channel_dim
def add_rain(
    img: ImageType,
    slant: float,
    drop_length: int,
    drop_width: int,
    drop_color: tuple[int, int, int],
    blur_value: int,
    brightness_coefficient: float,
    rain_drops: np.ndarray,
) -> ImageType:
    """Add rain streaks. slant, drop_length, drop_width, drop_color, blur_value,
    brightness_coefficient, rain_drops. Polylines; optional blur. uint8 I/O.

    This function adds rain to an image by drawing rain drops on the image.
    The rain drops are drawn using the OpenCV function cv2.polylines.

    Args:
        img (ImageType): The image to add rain to.
        slant (float): The slant of the rain drops.
        drop_length (int): The length of the rain drops.
        drop_width (int): The width of the rain drops.
        drop_color (tuple[int, int, int]): The color of the rain drops.
        blur_value (int): The blur value of the rain drops.
        brightness_coefficient (float): The brightness coefficient of the rain drops.
        rain_drops (np.ndarray): The rain drops to draw on the image.

    Returns:
        ImageType: The image with rain added.

    """
    if not rain_drops.size:
        return img.copy()

    img = img.copy()

    # Pre-allocate rain layer
    rain_layer = np.zeros_like(img, dtype=np.uint8)

    # Calculate end points correctly
    end_points = rain_drops + np.array([[slant, drop_length]])  # This creates correct shape

    # Stack arrays properly - both must be same shape arrays
    lines = np.stack((rain_drops, end_points), axis=1)  # Use tuple and proper axis

    cv2.polylines(
        rain_layer,
        lines.astype(np.int32),
        False,
        drop_color,
        drop_width,
        lineType=cv2.LINE_4,
    )

    if blur_value > 1:
        cv2.blur(rain_layer, (blur_value, blur_value), dst=rain_layer)

    cv2.add(img, rain_layer, dst=img)

    if brightness_coefficient != 1.0:
        cv2.multiply(img, brightness_coefficient, dst=img, dtype=cv2.CV_8U)

    return img


def get_fog_particle_radiuses(
    img_shape: tuple[int, int],
    num_particles: int,
    fog_intensity: float,
    random_generator: np.random.Generator,
) -> list[int]:
    """Generate per-particle radius list for add_fog. num_particles, fog_intensity, image size;
    random_generator samples. Returns list[int].

    Args:
        img_shape (tuple[int, int]): Image shape.
        num_particles (int): Number of fog particles.
        fog_intensity (float): Intensity of the fog effect, between 0 and 1.
        random_generator (np.random.Generator): Random generator to use.

    Returns:
        list[int]: List of radiuses for each fog particle.

    """
    height, width = img_shape[:2]
    max_fog_radius = max(2, int(min(height, width) * 0.1 * fog_intensity))
    min_radius = max(1, max_fog_radius // 2)

    return [random_generator.integers(min_radius, max_fog_radius) for _ in range(num_particles)]


@uint8_io
@clipped
@preserve_channel_dim
def add_fog(
    img: ImageType,
    fog_intensity: float,
    alpha_coef: float,
    fog_particle_positions: list[tuple[int, int]],
    fog_particle_radiuses: list[int],
) -> ImageType:
    """Add fog with circular particles and alpha blending. fog_intensity, alpha_coef, positions,
    radiuses (lists from get_fog_particle_radiuses). uint8 I/O, clipped.

    This function adds fog to an image by drawing fog particles on the image.
    The fog particles are drawn using the OpenCV function cv2.circle.

    Args:
        img (ImageType): The image to add fog to.
        fog_intensity (float): The intensity of the fog effect, between 0 and 1.
        alpha_coef (float): The coefficient for the alpha blending.
        fog_particle_positions (list[tuple[int, int]]): The positions of the fog particles.
        fog_particle_radiuses (list[int]): The radiuses of the fog particles.

    Returns:
        ImageType: The image with fog added.

    """
    result = img.copy()

    # Apply fog particles progressively like in old version
    for (x, y), radius in zip(fog_particle_positions, fog_particle_radiuses, strict=True):
        overlay = result.copy()
        cv2.circle(
            overlay,
            center=(x, y),
            radius=radius,
            color=(255, 255, 255),
            thickness=-1,
        )

        # Progressive blending
        alpha = alpha_coef * fog_intensity
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, dst=result)

    # Final subtle blur
    blur_size = max(3, int(min(img.shape[:2]) // 30))
    if blur_size % 2 == 0:
        blur_size += 1

    result = cv2.GaussianBlur(result, (blur_size, blur_size), 0)

    return clip(result, np.uint8, inplace=True)


@uint8_io
@preserve_channel_dim
@maybe_process_in_chunks
def add_sun_flare_overlay(
    img: ImageType,
    flare_center: tuple[float, float],
    src_radius: int,
    src_color: tuple[int, ...],
    circles: list[Any],
) -> ImageType:
    """Add a sun flare effect using a simple overlay. Params: src_radius, num_flare_circles;
    used as helper for physics-based sun flare.

    This function creates a basic sun flare effect by overlaying multiple semi-transparent
    circles of varying sizes and intensities on the input image. The effect simulates
    a simple lens flare caused by bright light sources.

    Args:
        img (ImageType): The input image.
        flare_center (tuple[float, float]): (x, y) coordinates of the flare center
            in pixel coordinates.
        src_radius (int): The radius of the main sun circle in pixels.
        src_color (tuple[int, ...]): The color of the sun, represented as a tuple of RGB values.
        circles (list[Any]): A list of tuples, each representing a circle that contributes
            to the flare effect. Each tuple contains:
            - alpha (float): The transparency of the circle (0.0 to 1.0).
            - center (tuple[int, int]): (x, y) coordinates of the circle center.
            - radius (int): The radius of the circle.
            - color (tuple[int, int, int]): RGB color of the circle.

    Returns:
        ImageType: The output image with the sun flare effect added.

    Note:
        - This function uses a simple alpha blending technique to overlay flare elements.
        - The main sun is created as a gradient circle, fading from the center outwards.
        - Additional flare circles are added along an imaginary line from the sun's position.
        - This method is computationally efficient but may produce less realistic results
          compared to more advanced techniques.

    The flare effect is created through the following steps:
    1. Create an overlay image and output image as copies of the input.
    2. Add smaller flare circles to the overlay.
    3. Blend the overlay with the output image using alpha compositing.
    4. Add the main sun circle with a radial gradient.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> flare_center = (50, 50)
        >>> src_radius = 20
        >>> src_color = (255, 255, 200)
        >>> circles = [
        ...     (0.1, (60, 60), 5, (255, 200, 200)),
        ...     (0.2, (70, 70), 3, (200, 255, 200))
        ... ]
        >>> flared_image = A.functional.add_sun_flare_overlay(
        ...     image, flare_center, src_radius, src_color, circles
        ... )

    References:
        - Alpha compositing: https://en.wikipedia.org/wiki/Alpha_compositing
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare

    """
    overlay = img.copy()
    output = img.copy()

    weighted_brightness = 0.0
    total_radius_length = 0.0

    for alpha, (x, y), rad3, circle_color in circles:
        weighted_brightness += alpha * rad3
        total_radius_length += rad3
        cv2.circle(overlay, (x, y), rad3, circle_color, -1)
        output = add_weighted(overlay, alpha, output, 1 - alpha)

    point = [int(x) for x in flare_center]

    overlay = output.copy()
    num_times = src_radius // 10

    # max_alpha is calculated using weighted_brightness and total_radii_length times 5
    # meaning the higher the alpha with larger area, the brighter the bright spot will be
    # for list of alphas in range [0.05, 0.2], the max_alpha should below 1
    max_alpha = weighted_brightness / total_radius_length * 5
    alpha = np.linspace(0.0, min(max_alpha, 1.0), num=num_times)

    rad = np.linspace(1, src_radius, num=num_times)

    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        output = add_weighted(overlay, alp, output, 1 - alp)

    return output


@uint8_io
@clipped
def add_sun_flare_physics_based(
    img: ImageType,
    flare_center: tuple[int, int],
    src_radius: int,
    src_color: tuple[int, int, int],
    circles: list[Any],
) -> ImageType:
    """Physics-based sun flare: circle, spikes, ghosts, chromatic aberration, screen blend.
    flare_center, src_radius, src_color, circles.

    This function creates a complex sun flare effect by simulating various optical phenomena
    that occur in real camera lenses when capturing bright light sources. The result is a
    more realistic and physically plausible lens flare effect.

    Args:
        img (ImageType): Input image.
        flare_center (tuple[int, int]): (x, y) coordinates of the sun's center in pixels.
        src_radius (int): Radius of the main sun circle in pixels.
        src_color (tuple[int, int, int]): Color of the sun in RGB format.
        circles (list[Any]): List of tuples, each representing a flare circle with parameters:
            (alpha, center, size, color)
            - alpha (float): Transparency of the circle (0.0 to 1.0).
            - center (tuple[int, int]): (x, y) coordinates of the circle center.
            - size (float): Size factor for the circle radius.
            - color (tuple[int, int, int]): RGB color of the circle.

    Returns:
        ImageType: Image with added sun flare effect.

    Note:
        This function implements several techniques to create a more realistic flare:
        1. Separate flare layer: Allows for complex manipulations of the flare effect.
        2. Lens diffraction spikes: Simulates light diffraction in camera aperture.
        3. Radial gradient mask: Creates natural fading of the flare from the center.
        4. Gaussian blur: Softens the flare for a more natural glow effect.
        5. Chromatic aberration: Simulates color fringing often seen in real lens flares.
        6. Screen blending: Provides a more realistic blending of the flare with the image.

    The flare effect is created through the following steps:
    1. Create a separate flare layer.
    2. Add the main sun circle and diffraction spikes to the flare layer.
    3. Add additional flare circles based on the input parameters.
    4. Apply Gaussian blur to soften the flare.
    5. Create and apply a radial gradient mask for natural fading.
    6. Simulate chromatic aberration by applying different blurs to color channels.
    7. Blend the flare with the original image using screen blending mode.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [1000, 1000, 3], dtype=np.uint8)
        >>> flare_center = (500, 500)
        >>> src_radius = 50
        >>> src_color = (255, 255, 200)
        >>> circles = [
        ...     (0.1, (550, 550), 10, (255, 200, 200)),
        ...     (0.2, (600, 600), 5, (200, 255, 200))
        ... ]
        >>> flared_image = A.functional.add_sun_flare_physics_based(
        ...     image, flare_center, src_radius, src_color, circles
        ... )

    References:
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare
        - Diffraction: https://en.wikipedia.org/wiki/Diffraction
        - Chromatic aberration: https://en.wikipedia.org/wiki/Chromatic_aberration
        - Screen blending: https://en.wikipedia.org/wiki/Blend_modes#Screen

    """
    output = img.copy()
    height, width = img.shape[:2]

    # Create a separate flare layer
    flare_layer = np.zeros_like(img, dtype=np.float32)

    # Add the main sun
    cv2.circle(flare_layer, flare_center, src_radius, src_color, -1)

    # Add lens diffraction spikes
    for angle in [0, 45, 90, 135]:
        end_point = (
            int(flare_center[0] + np.cos(np.radians(angle)) * max(width, height)),
            int(flare_center[1] + np.sin(np.radians(angle)) * max(width, height)),
        )
        cv2.line(flare_layer, flare_center, end_point, src_color, 2)

    # Add flare circles
    for _, center, size, color in circles:
        cv2.circle(flare_layer, center, int(size**0.33), color, -1)

    # Apply gaussian blur to soften the flare
    flare_layer = cv2.GaussianBlur(flare_layer, (0, 0), sigmaX=15, sigmaY=15)

    # Create a radial gradient mask
    y, x = np.ogrid[:height, :width]
    mask = np.sqrt((x - flare_center[0]) ** 2 + (y - flare_center[1]) ** 2)
    mask = 1 - np.clip(mask / (max(width, height) * 0.7), 0, 1)
    mask = np.dstack([mask] * 3)

    # Apply the mask to the flare layer
    flare_layer *= mask

    # Add chromatic aberration
    channels = list(cv2.split(flare_layer))
    channels[0] = cv2.GaussianBlur(
        channels[0],
        (0, 0),
        sigmaX=3,
        sigmaY=3,
    )  # Blue channel
    channels[2] = cv2.GaussianBlur(
        channels[2],
        (0, 0),
        sigmaX=5,
        sigmaY=5,
    )  # Red channel
    flare_layer = cv2.merge(channels)

    # Blend the flare with the original image using screen blending
    return 255 - ((255 - output) * (255 - flare_layer) / 255)


@uint8_io
@preserve_channel_dim
def add_shadow(
    img: ImageType,
    vertices_list: list[np.ndarray],
    intensities: np.ndarray,
) -> ImageType:
    """Darken polygonal regions to simulate shadows. vertices_list and intensities per polygon.
    Use for outdoor or synthetic shadow augmentation. uint8 I/O.

    Args:
        img (ImageType): Input image. Multichannel images are supported.
        vertices_list (list[np.ndarray]): List of vertices for shadow polygons.
        intensities (np.ndarray): Array of shadow intensities. Range is [0, 1].

    Returns:
        ImageType: Image with shadows added.

    References:
        Automold--Road-Augmentation-Library: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    img_shadowed = img.copy()
    poly_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for vertices, shadow_intensity in zip(vertices_list, intensities, strict=True):
        poly_mask[:] = 0
        cv2.fillPoly(poly_mask, [vertices], (max_value,))

        shadowed_indices = poly_mask[:, :, 0] == max_value
        darkness = 1 - shadow_intensity
        img_shadowed[shadowed_indices] = clip(
            img_shadowed[shadowed_indices] * darkness,
            np.uint8,
            inplace=True,
        )

    return img_shadowed


@uint8_io
@clipped
@preserve_channel_dim
def add_gravel(img: ImageType, gravels: list[Any]) -> ImageType:
    """Add gravel: write HLS saturation in rectangular regions. gravels: list of
    (min_y, max_y, min_x, max_x, sat). RGB only; uint8 I/O.

    This function adds gravel to an image by drawing gravel particles on the image.
    The gravel particles are drawn using the OpenCV function cv2.circle.

    Args:
        img (ImageType): The image to add gravel to.
        gravels (list[Any]): The gravel particles to draw on the image.

    Returns:
        ImageType: The image with gravel added.

    """
    non_rgb_error(img)
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    for gravel in gravels:
        min_y, max_y, min_x, max_x, sat = gravel
        image_hls[min_y:max_y, min_x:max_x, 1] = sat

    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


@float32_io
@clipped
@preserve_channel_dim
def spatter_rain(img: ImageType, rain: np.ndarray) -> ImageType:
    """Add rain layer using precomputed pattern from get_rain_params. Simulates wet surfaces.
    Used by Spatter. float32 I/O, clipped.

    This function applies spatter rain to an image by adding the rain to the image.

    Args:
        img (ImageType): Input image as a numpy array.
        rain (np.ndarray): Rain image as a numpy array.

    Returns:
        ImageType: The spatter rain applied to the image.

    """
    return add(img, rain, inplace=False)


@float32_io
@clipped
@preserve_channel_dim
def spatter_mud(img: ImageType, non_mud: np.ndarray, mud: np.ndarray) -> ImageType:
    """Spatter mud: blend non_mud and mud layers. non_mud, mud from get_mud_params. Simulates dirt on
    lens/surface. float32 I/O, clipped.

    This function applies spatter mud to an image by adding the mud to the image.

    Args:
        img (ImageType): Input image as a numpy array.
        non_mud (np.ndarray): Non-mud image as a numpy array.
        mud (np.ndarray): Mud image as a numpy array.

    Returns:
        ImageType: The spatter mud applied to the image.

    """
    return add(img * non_mud, mud, inplace=False)


def get_rain_params(
    liquid_layer: np.ndarray,
    color: np.ndarray,
    intensity: float,
) -> dict[str, Any]:
    """Generate parameters for rain effect. liquid_layer, color, intensity. Returns dict with 'drops'
    for add_rain/spatter_rain.

    This function generates parameters for a rain effect.

    Args:
        liquid_layer (np.ndarray): Liquid layer of the image.
        color (np.ndarray): Color of the rain.
        intensity (float): Intensity of the rain.

    Returns:
        dict[str, Any]: Parameters for the rain effect.

    """
    liquid_layer = clip(liquid_layer * 255, np.uint8, inplace=False)

    # Generate distance transform with more defined edges
    dist = 255 - cv2.Canny(liquid_layer, 50, 150)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
    _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)

    # Use separate blur operations for better drop formation
    dist = cv2.GaussianBlur(
        dist,
        ksize=(3, 3),
        sigmaX=1,  # Add slight sigma for smoother drops
        sigmaY=1,
        borderType=cv2.BORDER_REPLICATE,
    )
    dist = clip(dist, np.uint8, inplace=True)

    dist = dist[..., np.newaxis]

    # Enhance contrast in the distance map
    dist = equalize(dist)
    # Modified kernel for more natural drop shapes
    ker = np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.float32,
    )

    # Apply convolution with better precision
    dist = convolve(dist, ker)

    # Final blur with larger kernel for smoother drops
    dist = cv2.GaussianBlur(
        dist,
        ksize=(5, 5),  # Increased kernel size
        sigmaX=1.5,  # Adjusted sigma
        sigmaY=1.5,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    # Calculate final rain mask with better blending
    m = liquid_layer.astype(np.float32) * dist

    # Normalize with better handling of edge cases
    m_max = np.max(m, axis=(0, 1))
    if m_max > 0:
        m *= 1 / m_max
    else:
        m = np.zeros_like(m)

    # Apply color with adjusted intensity for more natural look
    drops = m[:, :, None] * color * (intensity * 0.9)  # Slightly reduced intensity

    return {
        "drops": drops,
    }


def get_mud_params(
    liquid_layer: np.ndarray,
    color: np.ndarray,
    cutout_threshold: float,
    sigma: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> dict[str, Any]:
    """Generate parameters for mud effect. liquid_layer, color, cutout_threshold, sigma, intensity,
    random_generator. Returns dict for spatter_mud.

    This function generates parameters for a mud effect.

    Args:
        liquid_layer (np.ndarray): Liquid layer of the image.
        color (np.ndarray): Color of the mud.
        cutout_threshold (float): Cutout threshold for the mud.
        sigma (float): Sigma for the Gaussian blur.
        intensity (float): Intensity of the mud.
        random_generator (np.random.Generator): Random number generator.

    Returns:
        dict[str, Any]: Parameters for the mud effect.

    """
    height, width = liquid_layer.shape

    # Create initial mask (ensure we have some non-zero values)
    mask = (liquid_layer > cutout_threshold).astype(np.float32)
    if reduce_sum(mask) == 0:  # If mask is all zeros
        # Force minimum coverage of 10%
        num_pixels = height * width
        num_needed = max(1, int(0.1 * num_pixels))  # At least 1 pixel
        flat_indices = random_generator.choice(num_pixels, num_needed, replace=False)
        mask = np.zeros_like(liquid_layer, dtype=np.float32)
        mask.flat[flat_indices] = 1.0

    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        mask = cv2.GaussianBlur(
            mask,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

    # Safe normalization (avoid division by zero)
    mask_max = np.max(mask)
    if mask_max > 0:
        mask = mask / mask_max
    else:
        # If mask is somehow all zeros after blur, force some effect
        mask[0, 0] = 1.0

    # Scale by intensity directly (no minimum)
    mask = mask * intensity

    # Create mud effect array
    mud = np.zeros((height, width, 3), dtype=np.float32)

    # Apply color directly - the intensity scaling is already handled
    for i in range(3):
        mud[..., i] = mask * color[i]

    # Create complementary non-mud array
    non_mud = np.ones_like(mud)
    for i in range(3):
        if color[i] > 0:
            non_mud[..., i] = np.clip((color[i] - mud[..., i]) / color[i], 0, 1)
        else:
            non_mud[..., i] = 1.0 - mask

    return {
        "mud": mud.astype(np.float32),
        "non_mud": non_mud.astype(np.float32),
    }


def apply_atmospheric_fog(
    img: ImageType,
    density: float,
    fog_color: tuple[float, ...],
    depth_map: np.ndarray,
) -> ImageType:
    """Apply depth-aware atmospheric fog using standard scattering. Formula: img *
    exp(-density*depth) + fog_color*(1 - exp(-density*depth)).

    Formula: result = img * exp(-density * depth) + fog_color * (1 - exp(-density * depth))

    Args:
        img (ImageType): Input image (H, W, C).
        density (float): Fog density factor.
        fog_color (tuple[float, ...]): Color of the fog, values in [0, max_val].
        depth_map (np.ndarray): (H, W) float32 array with values in [0, 1], where 1 is farthest.

    Returns:
        ImageType: Image with fog applied.

    """
    num_channels = img.shape[-1]
    transmission = np.exp(-density * depth_map).astype(np.float32)[:, :, np.newaxis]

    fog_array = np.array(fog_color, dtype=np.float32)
    if len(fog_array) < num_channels:
        fog_array = np.pad(fog_array, (0, num_channels - len(fog_array)), mode="edge")
    fog_array = fog_array[:num_channels].reshape(1, 1, -1)

    result = img.astype(np.float32) * transmission + fog_array * (1.0 - transmission)

    return clip(result, img.dtype)


__all__ = [
    "add_fog",
    "add_gravel",
    "add_rain",
    "add_shadow",
    "add_snow_bleach",
    "add_snow_texture",
    "add_sun_flare_overlay",
    "add_sun_flare_physics_based",
    "apply_atmospheric_fog",
    "generate_snow_textures",
    "get_fog_particle_radiuses",
    "get_mud_params",
    "get_rain_params",
    "spatter_mud",
    "spatter_rain",
]
