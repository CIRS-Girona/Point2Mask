import cv2
import numpy as np

from typing import List, Tuple


def enhance_image(
    image: np.ndarray, 
    clahe: cv2.CLAHE, 
    brightness: int = 10, 
    saturation: float = 1.25
) -> np.ndarray:
    """Applies bilateral filtering, color correction, and CLAHE."""
    # Denoise
    image = cv2.bilateralFilter(image, 5, 50.0, 50.0).astype(np.float32)

    max_ch = np.max(image, axis=(0, 1))
    mean_ch = np.mean(image, axis=(0, 1))

    # Log power per channel color correction
    gain = np.mean(mean_ch) / 255
    power_ch = np.log(gain) / np.log(mean_ch / max_ch)

    for i in range(3):
        image[:, :, i] = np.power(image[:, :, i] / max_ch[i], power_ch[i])

    image = np.clip(255 * image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    image[:, :, 2] += brightness
    image[:, :, 1] *= saturation

    image = np.clip(image, 0, 255).astype(np.uint8)
    image[:, :, 2] = clahe.apply(image[:, :, 2])

    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def post_process_mask(
    mask: np.ndarray,
    color: Tuple[int, int, int],
    min_area: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters small components and returns filled mask + colored layer."""
    mask_uint8 = mask.astype(np.uint8)

    # Filter by area
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    filtered_mask = np.zeros_like(mask_uint8)

    # Vectorized approach is harder here due to stats check, looping is fine
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1

    # Morphological closing
    kernel = np.ones((15, 15), np.uint8)
    filled_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    # Create colored layer
    colored_layer = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_layer[filled_mask == 1] = color

    return filled_mask, colored_layer


def render_polygon_mask(segmentation: List[List[float]], height: int, width: int, color: tuple) -> np.ndarray:
    """
    Renders a COCO flat-list segmentation format back into a colored raster mask.
    (Note: Recommend removing this once coordinate validation is complete.)
    """
    # Create a blank RGB mask
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # COCO segmentations can have multiple distinct polygons per object
    for poly in segmentation:
        # Reshape the flat list [x1, y1, x2, y2...] into OpenCV's expected shape (N, 1, 2)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

        # Fill the polygon with the specified color
        cv2.fillPoly(mask, [pts], color)

    return mask