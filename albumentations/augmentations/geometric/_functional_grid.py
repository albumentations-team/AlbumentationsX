"""Grid, tile, padding, and perspective utility helpers."""

from __future__ import annotations

from typing import Literal

from ._functional_shared import (
    cv2,
    defaultdict,
    math,
    np,
    reduce_sum,
)


def generate_grid(
    image_shape: tuple[int, int],
    steps_x: list[float],
    steps_y: list[float],
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a distorted grid (map_x, map_y) for remap. steps_x, steps_y,
    num_steps control distortion. image_shape (H, W). For GridDistortion.

    This function creates two 2D arrays (map_x and map_y) that represent a distorted version
    of the original image grid. These arrays can be used with OpenCV's remap function to
    apply grid distortion to an image.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        steps_x (list[float]): List of step sizes for the x-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the x direction.
        steps_y (list[float]): List of step sizes for the y-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the y direction.
        num_steps (int): The number of steps to divide each axis into. This determines
            the granularity of the distortion grid.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two 2D numpy arrays:
            - map_x: A 2D array of float32 values representing the x-coordinates
              of the distorted grid.
            - map_y: A 2D array of float32 values representing the y-coordinates
              of the distorted grid.

    Note:
        - The function generates a grid where each cell can be distorted independently.
        - The distortion is controlled by the steps_x and steps_y parameters, which
          determine how much each grid line is shifted.
        - The resulting map_x and map_y can be used directly with cv2.remap() to
          apply the distortion to an image.
        - The distortion is applied smoothly across each grid cell using linear
          interpolation.

    Examples:
        >>> image_shape = (100, 100)
        >>> steps_x = [1.1, 0.9, 1.0, 1.2, 0.95, 1.05]
        >>> steps_y = [0.9, 1.1, 1.0, 1.1, 0.9, 1.0]
        >>> num_steps = 5
        >>> map_x, map_y = generate_grid(image_shape, steps_x, steps_y, num_steps)
        >>> distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    """
    height, width = image_shape[:2]
    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0.0
    for idx, step in enumerate(steps_x):
        x = idx * x_step
        start = int(x)
        end = min(int(x) + x_step, width)
        cur = prev + x_step * step
        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0.0
    for idx, step in enumerate(steps_y):
        y = idx * y_step
        start = int(y)
        end = min(int(y) + y_step, height)
        cur = prev + y_step * step
        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    return np.meshgrid(xx, yy)


def normalize_grid_distortion_steps(
    image_shape: tuple[int, int],
    num_steps: int,
    x_steps: list[float],
    y_steps: list[float],
) -> dict[str, np.ndarray]:
    """Normalize grid distortion steps so distortion stays in image bounds.
    image_shape, num_steps, x_steps, y_steps. Returns dict steps_x, steps_y.

    This function normalizes the grid distortion steps, ensuring that the distortion never leaves the image bounds.
    It compensates for smaller last steps in the source image and normalizes the steps such that the distortion
    never leaves the image bounds.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        num_steps (int): The number of steps to divide each axis into. This determines
            the granularity of the distortion grid.
        x_steps (list[float]): List of step sizes for the x-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the x direction.
        y_steps (list[float]): List of step sizes for the y-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the y direction.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the normalized step sizes for the x and y axes.

    """
    height, width = image_shape[:2]

    # compensate for smaller last steps in source image.
    x_step = width // num_steps
    last_x_step = min(width, ((num_steps + 1) * x_step)) - (num_steps * x_step)
    x_steps[-1] *= last_x_step / x_step

    y_step = height // num_steps
    last_y_step = min(height, ((num_steps + 1) * y_step)) - (num_steps * y_step)
    y_steps[-1] *= last_y_step / y_step

    # now normalize such that distortion never leaves image bounds.
    tx = width / math.floor(width / num_steps)
    ty = height / math.floor(height / num_steps)
    x_steps_arr = np.array(x_steps, dtype=np.float32)
    y_steps_arr = np.array(y_steps, dtype=np.float32)
    x_steps = x_steps_arr * (tx / reduce_sum(x_steps_arr))
    y_steps = y_steps_arr * (ty / reduce_sum(y_steps_arr))

    return {"steps_x": x_steps, "steps_y": y_steps}


def almost_equal_intervals(n: int, parts: int) -> np.ndarray:
    """Generate nearly equal integer intervals that sum to n. parts is count; max diff 1.
    For splitting H or W into grid rows/cols. Returns 1D array of part sizes.

    This function divides the number `n` into `parts` nearly equal parts. It ensures that
    the sum of all parts equals `n`, and the difference between any two parts is at most one.
    This is useful for distributing a total amount into nearly equal discrete parts.

    Args:
        n (int): The total value to be split.
        parts (int): The number of parts to split into.

    Returns:
        np.ndarray: An array of integers where each integer represents the size of a part.

    Examples:
        >>> almost_equal_intervals(20, 3)
        array([7, 7, 6])  # Splits 20 into three parts: 7, 7, and 6
        >>> almost_equal_intervals(16, 4)
        array([4, 4, 4, 4])  # Splits 16 into four equal parts

    """
    part_size, remainder = divmod(n, parts)
    # Create an array with the base part size and adjust the first `remainder` parts by adding 1
    return np.array(
        [part_size + 1 if i < remainder else part_size for i in range(parts)],
    )


def generate_shuffled_splits(
    size: int,
    divisions: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate shuffled splits for a dimension (size, divisions). random_generator
    shuffles interval sizes. Returns cumulative edges. For GridDistortion/Mosaic.

    Args:
        size (int): Total size of the dimension (height or width).
        divisions (int): Number of divisions (rows or columns).
        random_generator (np.random.Generator): The random generator to use for shuffling the splits.
            If None, the splits are not shuffled.

    Returns:
        np.ndarray: Cumulative edges of the shuffled intervals.

    """
    intervals = almost_equal_intervals(size, divisions)
    random_generator.shuffle(intervals)
    return np.insert(np.cumsum(intervals), 0, 0)


def split_uniform_grid(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Split image shape into a uniform grid (rows, cols). Shuffled splits; returns
    tile coords (start_y, start_x, end_y, end_x) per tile. For GridShuffle/Mosaic.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        grid (tuple[int, int]): The grid size as (rows, columns).
        random_generator (np.random.Generator): The random generator to use for shuffling the splits.
            If None, the splits are not shuffled.

    Returns:
        np.ndarray: An array containing the tiles' coordinates in the format (start_y, start_x, end_y, end_x).

    Note:
        The function uses `generate_shuffled_splits` to generate the splits for the height and width of the image.
        The splits are then used to calculate the coordinates of the tiles.

    """
    n_rows, n_cols = grid

    height_splits = generate_shuffled_splits(
        image_shape[0],
        grid[0],
        random_generator=random_generator,
    )
    width_splits = generate_shuffled_splits(
        image_shape[1],
        grid[1],
        random_generator=random_generator,
    )

    # Calculate tiles coordinates
    tiles = [
        (height_splits[i], width_splits[j], height_splits[i + 1], width_splits[j + 1])
        for i in range(n_rows)
        for j in range(n_cols)
    ]

    return np.array(tiles, dtype=np.int16)


def generate_perspective_points(
    image_shape: tuple[int, int],
    scale: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate four perspective corner points for image_shape and scale. Normal
    jitter, modulated to bounds. random_generator. For Perspective transform.

    This function generates perspective points for a given image shape and scale.
    It uses a normal distribution to generate the points, and then modulates them to be within the image bounds.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        scale (float): The scale of the perspective points.
        random_generator (np.random.Generator): The random generator to use for generating the points.

    Returns:
        np.ndarray: The perspective points.

    """
    height, width = image_shape[:2]
    points = random_generator.normal(0, scale, (4, 2))
    points = np.mod(np.abs(points), 0.32)

    # top left -- no changes needed, just use jitter
    # top right
    points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
    # bottom right
    points[2] = 1.0 - points[2]  # w = 1.0 - jitter
    # bottom left
    points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

    points[:, 0] *= width
    points[:, 1] *= height

    return points


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points clockwise: top-left, top-right, bottom-right, bottom-left.
    For perspective transform source/destination quads. pts shape (4, 2).

    This function orders the points in a clockwise manner, ensuring that the points are in the correct
    order for perspective transformation.

    Args:
        pts (np.ndarray): The points to order.

    Returns:
        np.ndarray: The ordered points.

    """
    pts = np.array(sorted(pts, key=lambda x: x[0]))
    left = pts[:2]  # points with smallest x coordinate - left points
    right = pts[2:]  # points with greatest x coordinate - right points

    if left[0][1] < left[1][1]:
        tl, bl = left
    else:
        bl, tl = left

    if right[0][1] < right[1][1]:
        tr, br = right
    else:
        br, tr = right

    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_perspective_params(
    points: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    """Compute perspective params from four points and image_shape. Returns
    (matrix, max_width, max_height). Adjusts dims so transformed image keeps size.

    Computes the perspective transformation matrix and output dimensions for a given
    set of four corner points; call from Perspective or similar transforms.

    Args:
        points (np.ndarray): The points to compute the perspective transformation parameters for.
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[np.ndarray, int, int]: The perspective transformation parameters and the maximum
            dimensions of the transformed image.

    """
    height, width = image_shape
    top_left, top_right, bottom_right, bottom_left = points

    def adjust_dimension(
        dim1: np.ndarray,
        dim2: np.ndarray,
        min_size: int = 2,
    ) -> float:
        size = np.sqrt(reduce_sum((dim1 - dim2) ** 2))
        if size < min_size:
            step_size = (min_size - size) / 2
            dim1[dim1 > dim2] += step_size
            dim2[dim1 > dim2] -= step_size
            dim1[dim1 <= dim2] -= step_size
            dim2[dim1 <= dim2] += step_size
            size = min_size
        return size

    max_width = max(
        adjust_dimension(top_right, top_left),
        adjust_dimension(bottom_right, bottom_left),
    )
    max_height = max(
        adjust_dimension(bottom_right, top_right),
        adjust_dimension(bottom_left, top_left),
    )

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, dst)

    return matrix, int(max_width), int(max_height)


def expand_transform(
    matrix: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    """Expand a transformation matrix to include padding. shape (H, W). Returns
    (expanded_matrix, max_width, max_height). For Perspective with keep_size.

    This function expands a transformation matrix to include padding, ensuring that the transformed
    image retains its original dimensions. It first calculates the destination points of the transformed
    image, then adjusts the matrix to include padding, and finally returns the expanded matrix and the
    maximum dimensions of the transformed image.

    Args:
        matrix (np.ndarray): The transformation matrix to expand.
        shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[np.ndarray, int, int]: The expanded matrix and the maximum dimensions of the transformed image.

    """
    height, width = shape[:2]
    rect = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    dst = cv2.perspectiveTransform(np.array([rect]), matrix)[0]

    dst -= dst.min(axis=0, keepdims=True)
    dst = np.around(dst, decimals=0)

    matrix_expanded = cv2.getPerspectiveTransform(rect, dst)
    max_width, max_height = dst.max(axis=0)
    return matrix_expanded, int(max_width), int(max_height)


def get_dimension_padding(
    current_size: int,
    min_size: int | None,
    divisor: int | None,
) -> tuple[int, int]:
    """Calculate padding (pad_before, pad_after) for one dimension. current_size,
    optional min_size or divisor. For PadIfNeeded / divisible sizes.

    Args:
        current_size (int): Current size of the dimension
        min_size (int | None): Minimum size requirement, if any
        divisor (int | None): Divisor for padding to make size divisible, if any

    Returns:
        tuple[int, int]: (pad_before, pad_after)

    """
    if min_size is not None:
        if current_size < min_size:
            pad_before = int((min_size - current_size) / 2.0)
            pad_after = min_size - current_size - pad_before
            return pad_before, pad_after
    elif divisor is not None:
        remainder = current_size % divisor
        if remainder > 0:
            total_pad = divisor - remainder
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            return pad_before, pad_after

    return 0, 0


def get_padding_params(
    image_shape: tuple[int, int],
    min_height: int | None,
    min_width: int | None,
    pad_height_divisor: int | None,
    pad_width_divisor: int | None,
) -> tuple[int, int, int, int]:
    """Calculate padding (pad_top, pad_bottom, pad_left, pad_right) from image_shape
    and optional min_height, min_width, height/width divisors. For PadIfNeeded.

    Args:
        image_shape (tuple[int, int]): (height, width) of the image
        min_height (int | None): Minimum height requirement, if any
        min_width (int | None): Minimum width requirement, if any
        pad_height_divisor (int | None): Divisor for height padding, if any
        pad_width_divisor (int | None): Divisor for width padding, if any

    Returns:
        tuple[int, int, int, int]: (pad_top, pad_bottom, pad_left, pad_right)

    """
    rows, cols = image_shape[:2]

    h_pad_top, h_pad_bottom = get_dimension_padding(
        rows,
        min_height,
        pad_height_divisor,
    )
    w_pad_left, w_pad_right = get_dimension_padding(cols, min_width, pad_width_divisor)

    return h_pad_top, h_pad_bottom, w_pad_left, w_pad_right


def adjust_padding_by_position(
    h_top: int,
    h_bottom: int,
    w_left: int,
    w_right: int,
    position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"],
    py_random: np.random.RandomState,
) -> tuple[int, int, int, int]:
    """Adjust padding (h_top, h_bottom, w_left, w_right) by position: center,
    top_left, top_right, bottom_*, or random. py_random for random. For PadIfNeeded.
    """
    if position == "center":
        return h_top, h_bottom, w_left, w_right

    if position == "top_left":
        return 0, h_top + h_bottom, 0, w_left + w_right

    if position == "top_right":
        return 0, h_top + h_bottom, w_left + w_right, 0

    if position == "bottom_left":
        return h_top + h_bottom, 0, 0, w_left + w_right

    if position == "bottom_right":
        return h_top + h_bottom, 0, w_left + w_right, 0

    if position == "random":
        h_pad = h_top + h_bottom
        w_pad = w_left + w_right
        h_top = py_random.randint(0, h_pad)
        h_bottom = h_pad - h_top
        w_left = py_random.randint(0, w_pad)
        w_right = w_pad - w_left
        return h_top, h_bottom, w_left, w_right

    raise ValueError(f"Unknown position: {position}")


def swap_tiles_on_image(
    image: np.ndarray,
    tiles: np.ndarray,
    mapping: list[int] | None = None,
) -> np.ndarray:
    """Swap tiles on the image by mapping. tiles (M, 4) [start_y, start_x, end_y, end_x];
    mapping lists new index per tile. For GridShuffle. Returns new image.

    Args:
        image (np.ndarray): Input image.
        tiles (np.ndarray): Array of tiles with each tile as [start_y, start_x, end_y, end_x].
        mapping (list[int] | None): list of new tile indices.

    Returns:
        np.ndarray: Output image with tiles swapped according to the random shuffle.

    """
    # If no tiles are provided, return a copy of the original image
    if tiles.size == 0 or mapping is None:
        return image.copy()

    # Create a copy of the image to retain original for reference
    new_image = np.empty_like(image)
    for num, new_index in enumerate(mapping):
        start_y, start_x, end_y, end_x = tiles[new_index]
        start_y_orig, start_x_orig, end_y_orig, end_x_orig = tiles[num]
        # Assign the corresponding tile from the original image to the new image
        new_image[start_y:end_y, start_x:end_x] = image[
            start_y_orig:end_y_orig,
            start_x_orig:end_x_orig,
        ]

    return new_image


def create_shape_groups(tiles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Group tiles by (height, width) and return dict mapping shape -> list of tile indices.
    For GridShuffle so shuffling happens only within same-shaped tiles.
    """
    shape_groups = defaultdict(list)
    for index, (start_y, start_x, end_y, end_x) in enumerate(tiles):
        shape = (end_y - start_y, end_x - start_x)
        shape_groups[shape].append(index)
    return shape_groups


def shuffle_tiles_within_shape_groups(
    shape_groups: dict[tuple[int, int], list[int]],
    random_generator: np.random.Generator,
) -> list[int]:
    """Shuffles indices within each group of similar shapes and creates a list where each
    index points to the index of the tile it should be mapped to.

    Args:
        shape_groups (dict[tuple[int, int], list[int]]): Groups of tile indices categorized by shape.
        random_generator (np.random.Generator): The random generator to use for shuffling the indices.
            If None, a new random generator will be used.

    Returns:
        list[int]: A list where each index is mapped to the new index of the tile after shuffling.

    """
    # Initialize the output list with the same size as the total number of tiles, filled with -1
    num_tiles = sum(len(indices) for indices in shape_groups.values())
    mapping = [-1] * num_tiles

    # Prepare the random number generator

    for indices in shape_groups.values():
        shuffled_indices = indices.copy()
        random_generator.shuffle(shuffled_indices)

        for old, new in zip(indices, shuffled_indices, strict=True):
            mapping[old] = new

    return mapping


__all__ = [
    "adjust_padding_by_position",
    "almost_equal_intervals",
    "compute_perspective_params",
    "create_shape_groups",
    "expand_transform",
    "generate_grid",
    "generate_perspective_points",
    "generate_shuffled_splits",
    "get_dimension_padding",
    "get_padding_params",
    "normalize_grid_distortion_steps",
    "order_points",
    "shuffle_tiles_within_shape_groups",
    "split_uniform_grid",
    "swap_tiles_on_image",
]
