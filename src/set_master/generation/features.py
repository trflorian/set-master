import cv2
import numpy as np

from set_master.models import Color, Shading, Shape


def _squiggle_mask(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, min(round(width * 0.62), round(height * 0.9)))
    t = np.linspace(0, 1, max(24, height))
    x = np.round((width - 1) / 2 + np.sin(t * 2 * np.pi) * width * 0.18).astype(np.int32)
    radius = thickness // 2
    y = np.round(radius + t * (height - 1 - 2 * radius)).astype(np.int32)
    points = np.column_stack((x, y)).reshape((-1, 1, 2))
    cv2.polylines(mask, [points], False, 255, thickness, cv2.LINE_AA)
    cv2.circle(mask, (int(x[0]), int(y[0])), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (int(x[-1]), int(y[-1])), radius, 255, -1, cv2.LINE_AA)
    return mask


def draw_single_feature(
    img: np.ndarray,
    position: tuple[int, int],
    size: tuple[int, int],
    shape: Shape,
    color: Color,
    shading: Shading,
) -> None:
    """Draw a Set symbol, applying its solid, striped, or open shading."""
    width, height = size
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if shape == Shape.RECTANGLE:
        cv2.rectangle(
            mask,
            (position[0] - width // 2, position[1] - height // 2),
            (position[0] + width // 2, position[1] + height // 2),
            255,
            -1,
            cv2.LINE_AA,
        )
    elif shape == Shape.ELLIPSE:
        cv2.ellipse(mask, position, (width // 2, height // 2), 0, 0, 360, 255, -1, cv2.LINE_AA)
    elif shape == Shape.SQUIGGLE:
        local_mask = _squiggle_mask(size)
        x0, y0 = position[0] - width // 2, position[1] - height // 2
        x1, y1 = x0 + width, y0 + height
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1 = width - max(0, x1 - img.shape[1])
        sy1 = height - max(0, y1 - img.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            mask[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)] = local_mask[sy0:sy1, sx0:sx1]
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    if shading == Shading.OPEN:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color.value, max(1, round(min(size) * 0.045)), cv2.LINE_AA)
        return

    img[mask > 0] = color.value
    if shading == Shading.STRIPED:
        border = max(1, round(min(size) * 0.035))
        inner = cv2.erode(mask, np.ones((border * 2 + 1, border * 2 + 1), np.uint8))
        stripe = max(1, round(height / 120))
        gap = max(3, round(height / 28))
        for y in range(position[1] - height // 2, position[1] + height // 2 + 1, stripe + gap):
            y0, y1 = max(0, y), min(img.shape[0], y + stripe)
            if y1 > y0:
                stripe_mask = inner[y0:y1] > 0
                img[y0:y1][stripe_mask] = (255, 255, 255)
