import cv2
import numpy as np

from set_master.models import Shape, Color, Shading


def draw_single_feature(
    img: np.ndarray,
    position: tuple[int, int],
    size: tuple[int, int],
    shape: Shape,
    color: Color,
    shading: Shading,
) -> None:
    """
    Draw a single feature on the given image at the specified position.
    """

    if shape == Shape.RECTANGLE:
        cv2.rectangle(
            img,
            (
                position[0] - size[0] // 2,
                position[1] - size[1] // 2,
            ),
            (
                position[0] + size[0] // 2,
                position[1] + size[1] // 2,
            ),
            color.value,
            -1,
        )
    elif shape == Shape.ELLIPSE:
        cv2.ellipse(
            img,
            position,
            (size[0] // 2, size[1] // 2),
            0,
            0,
            360,
            color.value,
            -1,
        )
    elif shape == Shape.SQUIGGLE:
        points = np.array(
            [
                (
                    position[0] - size[0] // 2 + int(0.1 * size[0]),
                    position[1] + int(0.2 * size[1]),
                ),
                (
                    position[0] - size[0] // 4,
                    position[1] - int(0.3 * size[1]),
                ),
                (
                    position[0],
                    position[1] + int(0.3 * size[1]),
                ),
                (
                    position[0] + size[0] // 4,
                    position[1] - int(0.3 * size[1]),
                ),
                (
                    position[0] + size[0] // 2 - int(0.1 * size[0]),
                    position[1] + int(0.2 * size[1]),
                ),
            ],
            np.int32,
        )
        cv2.fillPoly(img, [points], color.value)
