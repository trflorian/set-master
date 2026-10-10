from pathlib import Path
import cv2
import numpy as np

from set_master.generation.card import draw_card
from set_master.generation.features import FeatureStyle
from set_master.models import Card, Color, Number, Shading, ShadingPattern, Shape


IMAGE_SIZE = (1440, 1100)


def test_draw_card_variants_for_each_color(output_dir: Path) -> None:
    """Render every shape and shading combination into 1440p review grids."""
    cards_output_dir = output_dir / "cards"
    cards_output_dir.mkdir(parents=True, exist_ok=True)
    label_width = 136
    header_height = 48
    padding = 12
    cell_width = (IMAGE_SIZE[1] - label_width - padding) // len(Shape)
    rows = [(shading, None) for shading in (Shading.SOLID, Shading.OPEN)]
    rows.extend((Shading.SHADED, pattern) for pattern in ShadingPattern)
    cell_height = (IMAGE_SIZE[0] - header_height - padding) // len(rows)
    card_height = min(cell_height - padding, (cell_width - padding) * 3 // 2)
    card_size = (round(card_height * 2 / 3), card_height)

    for color in Color:
        image = np.full((*IMAGE_SIZE, 3), 245, dtype=np.uint8)
        for column, shape in enumerate(Shape):
            heading = shape.value.title()
            text_size, _ = cv2.getTextSize(heading, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            cv2.putText(
                image,
                heading,
                (label_width + column * cell_width + (cell_width - text_size[0]) // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
        for row, (shading, pattern) in enumerate(rows):
            row_center = header_height + row * cell_height + cell_height // 2
            row_labels = pattern.value.split("_")[::-1] if pattern else [shading.value]
            for line, label in enumerate(row_labels):
                cv2.putText(
                    image,
                    label.title(),
                    (padding, row_center + 7 + (line * 24) - (len(row_labels) - 1) * 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )
            for column, shape in enumerate(Shape):
                center = (
                    label_width + column * cell_width + cell_width // 2,
                    row_center,
                )
                card = Card(shape=shape, color=color, shading=shading, number=Number.THREE)
                style = FeatureStyle(shading_pattern=pattern) if pattern else FeatureStyle()
                draw_card(image, card, position=center, size=card_size, feature_style=style)

        output_path = cards_output_dir / f"{color.name.lower()}.png"
        assert cv2.imwrite(str(output_path), image)
        saved_image = cv2.imread(str(output_path))
        assert saved_image is not None and saved_image.shape[:2] == IMAGE_SIZE
