from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from set_master.models import Color, Shading, Shape


@dataclass(frozen=True)
class FeatureStyle:
    stripe_direction: Literal["horizontal", "vertical"] = "horizontal"
    stripe_spacing: int = 8
    line_thickness: int = 2
    squiggle_amplitude: float = 0.12
    squiggle_waves: float = 1.0
    squiggle_thickness: float = 0.58


def _squiggle_mask(size: tuple[int, int], style: FeatureStyle) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, min(round(height * style.squiggle_thickness), round(height * 0.9)))
    t = np.linspace(0, 1, max(24, width))
    radius = thickness // 2
    x = np.round(radius + t * (width - 1 - 2 * radius)).astype(np.int32)
    y = np.round(
        (height - 1) / 2
        + np.sin(t * 2 * np.pi * style.squiggle_waves)
        * height
        * style.squiggle_amplitude
    ).astype(np.int32)
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
    style: FeatureStyle = FeatureStyle(),
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
        local_mask = _squiggle_mask(size, style)
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
        cv2.drawContours(img, contours, -1, color.value, style.line_thickness, cv2.LINE_AA)
        return

    paint_mask = mask.copy()
    if shading == Shading.STRIPED:
        border = max(1, style.line_thickness)
        inner = cv2.erode(mask, np.ones((border * 2 + 1, border * 2 + 1), np.uint8))
        if style.stripe_direction == "horizontal":
            y_start, y_end = position[1] - height // 2, position[1] + height // 2 + 1
            for y in range(y_start, y_end, style.stripe_spacing):
                y0, y1 = max(0, y), min(img.shape[0], y + style.line_thickness)
                paint_mask[y0:y1][inner[y0:y1] > 0] = 0
        else:
            x_start, x_end = position[0] - width // 2, position[0] + width // 2 + 1
            for x in range(x_start, x_end, style.stripe_spacing):
                x0, x1 = max(0, x), min(img.shape[1], x + style.line_thickness)
                paint_mask[:, x0:x1][inner[:, x0:x1] > 0] = 0

    alpha = paint_mask.astype(np.float32)[..., None] / 255
    img[:] = (img * (1 - alpha) + np.asarray(color.value, dtype=np.float32) * alpha).astype(np.uint8)
