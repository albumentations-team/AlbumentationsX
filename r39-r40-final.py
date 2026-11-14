def generate_plasma_pattern(
    target_shape: tuple[int, int],
    roughness: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate a plasma pattern using diamond-square algorithm. Returns float32
    (H, W) or (H, W, 1) for plasma-based brightness/contrast and shadow.

    This function generates a plasma pattern using the diamond-square algorithm.

    Args:
        target_shape (tuple[int, int]): The shape of the plasma pattern to generate.
        roughness (float): The roughness of the plasma pattern.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The plasma pattern generated.

    """

    def one_diamond_square_step(current_grid: np.ndarray, noise_scale: float) -> np.ndarray:
        next_height = (current_grid.shape[0] - 1) * 2 + 1
        next_width = (current_grid.shape[1] - 1) * 2 + 1

        grid_shape = (next_height, next_width)
        expanded_grid = np.zeros(grid_shape, dtype=np.float32)
        all_noise = random_generator.uniform(-noise_scale, noise_scale, grid_shape).astype(np.float32)
        np.add(current_grid, all_noise[::2, ::2], out=expanded_grid[::2, ::2])

        interpolation = np.empty_like(expanded_grid)
        mask = np.empty_like(expanded_grid)
        for kernel in (DIAMOND_KERNEL, SQUARE_KERNEL):
            cv2.filter2D(expanded_grid, -1, kernel, dst=interpolation, borderType=cv2.BORDER_CONSTANT)
            cv2.threshold(interpolation, 0, 1, cv2.THRESH_BINARY, dst=mask)
            cv2.add(interpolation, all_noise, dst=interpolation)
            cv2.multiply(interpolation, mask, dst=interpolation)
            cv2.add(expanded_grid, interpolation, dst=expanded_grid)

        # Normalize after each step to prevent value drift.
        cv2.normalize(expanded_grid, expanded_grid, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return expanded_grid

    # Pre-compute noise scales
    max_dimension = max(target_shape)
    power_of_two_size = 2 ** np.ceil(np.log2(max_dimension - 1)) + 1
    total_steps = int(np.log2(power_of_two_size - 1) - 1)
    noise_scales = np.asarray([roughness**i for i in range(total_steps)], dtype=np.float32)

    plasma_grid = random_generator.uniform(-1, 1, (3, 3)).astype(np.float32)

    # Recursively apply diamond-square steps
    for noise_scale in noise_scales:
        plasma_grid = one_diamond_square_step(plasma_grid, cast("float", noise_scale))

    pattern = _normalize_minmax_float32(plasma_grid[: target_shape[0], : target_shape[1]])
    return np.clip(pattern, 0, 1, out=pattern)
