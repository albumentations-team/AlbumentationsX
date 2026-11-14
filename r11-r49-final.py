def _adjust_hue_torchvision_uint8(img: ImageUInt8, factor: float) -> ImageUInt8:
    img = cast("ImageUInt8", cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    hue = cast("ImageUInt8", cv2.extractChannel(img, 0))
    apply_uint8_lut(hue, lut, out=hue)
    cv2.insertChannel(hue, img, 0)

    return cast("ImageUInt8", cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
