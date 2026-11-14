def _add_channels(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    value: np.ndarray,
    *,
    out: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | None = None,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    if _is_uint8_image(img):
        destination_u8 = cast("NDArray[np.uint8] | None", out)
        if (
            img.ndim == 3
            and img.shape[-1] == NUM_RGB_CHANNELS
            and img.strides[1:] == (NUM_RGB_CHANNELS, 1)
            and img.strides[0] >= img.shape[1] * NUM_RGB_CHANNELS
            and (destination_u8 is None or destination_u8.flags.c_contiguous)
        ):
            red, green, blue = map(float, value)
            if -255 <= red <= 255 and -255 <= green <= 255 and -255 <= blue <= 255:
                # Match the LUT's truncation toward zero before saturating the sum.
                dst = None if destination_u8 is None else destination_u8.reshape(-1).reshape(img.shape)
                result = cv2.add(img, (int(red), int(green), int(blue), 0), dst=dst)
                return (
                    cast("NDArray[np.uint8]", result.reshape(img.shape)) if destination_u8 is None else destination_u8
                )
        return _apply_lut(img, value, "add", out=destination_u8)
    if _is_float16_image(img):
        destination = cast("NDArray[np.float16] | None", out)
        result = _opencv_float16_three_channel_vector(img, value, "add", out=destination)
        if result is None:
            result = _numkong_float16_c9_add_channels(img, value, out=destination)
        if result is None and _numpy_float16_c9_vector_eligible(img, value):
            return _numpy_pointwise(img, value, "add", out=out)
        if result is None:
            result = _numkong_float16_single_channel_scale(img, value, "add", out=destination)
        if result is None:
            result = float16_pointwise_torch(
                img,
                "add",
                value,
                minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
                out=destination,
            )
        if result is not None:
            return result
    return _numpy_pointwise(img, value, "add", out=out)
