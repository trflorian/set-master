import cv2
import numpy as np

from set_master.generation.features import FeatureStyle, draw_single_feature
from set_master.models import Card


def draw_card_surface(
    img: np.ndarray,
    position: tuple[int, int],
    size: tuple[int, int],
    corner_radius: int = 50,
) -> None:
    """
    Draw a card shape with rounded corners on the given image.

    Args:
        img (np.ndarray): The image on which to draw the card.
        card_width (int): The width of the card.
        card_height (int): The height of the card.
        corner_radius (int): The radius of the rounded corners.
    """
    card_width, card_height = size

    # draw a white card shape with rounded corners
    cv2.rectangle(
        img,
        (
            position[0] - size[0] // 2 + corner_radius,
            position[1] - size[1] // 2,
        ),
        (
            position[0] + size[0] // 2 - corner_radius,
            position[1] + size[1] // 2,
        ),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        img,
        (
            position[0] - size[0] // 2,
            position[1] - size[1] // 2 + corner_radius,
        ),
        (
            position[0] + size[0] // 2,
            position[1] + size[1] // 2 - corner_radius,
        ),
        (255, 255, 255),
        -1,
    )

    dw = card_width // 2 - corner_radius
    dh = card_height // 2 - corner_radius

    # draw corners
    for x in [-1, 1]:
        for y in [-1, 1]:
            # position: center + card_width / 2
            cv2.ellipse(
                img,
                (
                    position[0] + x * dw,
                    position[1] + y * dh,
                ),
                (corner_radius, corner_radius),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=-1,
            )


def draw_card(
    img: np.ndarray,
    card: Card,
    position: tuple[int, int] | None = None,
    size: tuple[int, int] = (200, 300),
    feature_size: tuple[int, int] | None = None,
    symbol_spacing: int | None = None,
    corner_radius: int = 50,
    feature_style: FeatureStyle = FeatureStyle(),
) -> None:
    """
    Draw a card with its features on the given image.
    """

    img_size = img.shape

    if position is None:
        # center the card
        position = (img_size[1] // 2, img_size[0] // 2)

    draw_card_surface(
        img,
        size=size,
        position=position,
        corner_radius=corner_radius,
    )

    num_features = card.number.value

    if feature_size is None:
        feature_size = (size[0] // 2, size[1] // 6)
    spacing = symbol_spacing if symbol_spacing is not None else size[1] // (num_features + 1)

    for i in range(num_features):
        feature_position = (
            position[0],
            position[1] - spacing * (num_features - 1) // 2 + spacing * i,
        )
        draw_single_feature(
            img,
            position=feature_position,
            size=feature_size,
            shape=card.shape,
            color=card.color,
            shading=card.shading,
            style=feature_style,
        )
