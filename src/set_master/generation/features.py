from dataclasses import dataclass
import cv2
import numpy as np

from set_master.models import Color, Shading, ShadingPattern, Shape


@dataclass(frozen=True)
class FeatureStyle:
    shading_pattern: ShadingPattern = ShadingPattern.STRIPED_HORIZONTAL
    stripe_spacing: int = 5
    line_thickness: int = 2
    squiggle_amplitude: float = 0.18
    squiggle_waves: float = 1.0
    squiggle_thickness: float = 0.65


def _squiggle_mask(size: tuple[int, int], style: FeatureStyle) -> np.ndarray:
    width, height = size
    scale = 4
    mask = np.zeros((height * scale, width * scale), dtype=np.uint8)
    # Cubic boundaries reproduce the broad, curved ends and narrow middle in the photos.
    segments = np.asarray(
        [
            [(0.04, 0.76), (-0.08, 0.16), (0.20, 0.02), (0.45, 0.25)],
            [(0.45, 0.25), (0.70, 0.48), (0.78, 0.18), (0.90, 0.12)],
            [(0.90, 0.12), (1.08, 0.04), (1.04, 0.72), (0.85, 0.87)],
            [(0.85, 0.87), (0.62, 1.08), (0.36, 0.62), (0.19, 0.78)],
            [(0.19, 0.78), (0.05, 0.97), (0.02, 0.91), (0.04, 0.76)],
        ]
    )
    t = np.linspace(0, 1, max(24, width // 2))[:, None]
    points = np.concatenate(
        [(1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t**2 * p[2] + t**3 * p[3] for p in segments]
    )
    points[:, 1] = (
        0.5
        + (points[:, 1] - 0.5) * style.squiggle_thickness / 0.65
        + np.sin(points[:, 0] * 2 * np.pi * style.squiggle_waves) * (style.squiggle_amplitude - 0.18)
    )
    points = np.round(points * (width * scale - 1, height * scale - 1)).astype(np.int32)
    cv2.fillPoly(mask, [points], 255, cv2.LINE_AA)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)


def draw_single_feature(
    img: np.ndarray,
    position: tuple[int, int],
    size: tuple[int, int],
    shape: Shape,
    color: Color,
    shading: Shading,
    style: FeatureStyle = FeatureStyle(),
) -> None:
    """Draw a Set symbol with solid, patterned shaded, or open shading."""
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
    elif shape == Shape.DIAMOND:
        x, y = position
        points = np.asarray(
            [(x - width // 2, y), (x, y - height // 2), (x + width // 2, y), (x, y + height // 2)], np.int32
        )
        cv2.fillPoly(mask, [points], 255, cv2.LINE_AA)
    elif shape == Shape.CAPSULE:
        radius = min(width, height) // 2
        x, y = position
        cv2.rectangle(
            mask, (x - width // 2 + radius, y - height // 2), (x + width // 2 - radius, y + height // 2), 255, -1
        )
        for end_x in (x - width // 2 + radius, x + width // 2 - radius):
            cv2.circle(mask, (end_x, y), radius, 255, -1, cv2.LINE_AA)
    elif shape == Shape.SQUIGGLE:
        local_mask = _squiggle_mask(size, style)
        x0, y0 = position[0] - width // 2, position[1] - height // 2
        x1, y1 = x0 + width, y0 + height
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1 = width - max(0, x1 - img.shape[1])
        sy1 = height - max(0, y1 - img.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            mask[max(0, y0) : min(img.shape[0], y1), max(0, x0) : min(img.shape[1], x1)] = local_mask[sy0:sy1, sx0:sx1]
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    if shading == Shading.OPEN:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color.value, style.line_thickness, cv2.LINE_AA)
        return

    paint_mask = mask.copy()
    if shading == Shading.SHADED:
        border = max(1, style.line_thickness)
        inner = cv2.erode(mask, np.ones((border * 2 + 1, border * 2 + 1), np.uint8))
        pattern_mask = np.zeros_like(mask)
        if style.shading_pattern in (ShadingPattern.STRIPED_HORIZONTAL, ShadingPattern.STRIPED_VERTICAL):
            horizontal = style.shading_pattern == ShadingPattern.STRIPED_HORIZONTAL
            if horizontal:
                start, end = position[1] - height // 2, position[1] + height // 2 + 1
            else:
                start, end = position[0] - width // 2, position[0] + width // 2 + 1
            for coordinate in range(start, end, max(1, style.stripe_spacing)):
                if horizontal:
                    y0, y1 = max(0, coordinate), min(img.shape[0], coordinate + border)
                    pattern_mask[y0:y1] = 255
                else:
                    x0, x1 = max(0, coordinate), min(img.shape[1], coordinate + border)
                    pattern_mask[:, x0:x1] = 255
        elif style.shading_pattern == ShadingPattern.DOTTED:
            radius = max(1, border // 2)
            y_start, y_end = position[1] - height // 2, position[1] + height // 2 + 1
            x_start, x_end = position[0] - width // 2, position[0] + width // 2 + 1
            spacing = max(1, style.stripe_spacing)
            for y in range(y_start + spacing // 2, y_end, spacing):
                for x in range(x_start + spacing // 2, x_end, spacing):
                    cv2.circle(pattern_mask, (x, y), radius, 255, -1, cv2.LINE_AA)
        else:
            raise ValueError(f"Unsupported shading pattern: {style.shading_pattern}")
        paint_mask[inner > 0] = 0
        paint_mask[cv2.bitwise_and(inner, pattern_mask) > 0] = 255

    alpha = paint_mask.astype(np.float32)[..., None] / 255
    img[:] = (img * (1 - alpha) + np.asarray(color.value, dtype=np.float32) * alpha).astype(np.uint8)
