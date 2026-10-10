from pathlib import Path
import unittest

import cv2
import numpy as np

from set_master.generation.card import draw_card
from set_master.models import Card, Color, Number, Shading, Shape


OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_SIZE = (1440, 2560)


class TestGeneratedCardGrids(unittest.TestCase):
    def test_draw_card_variants_for_each_color(self) -> None:
        """Render every shape and shading combination into 1440p review grids."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cell_width = IMAGE_SIZE[1] // 3
        cell_height = IMAGE_SIZE[0] // 3
        card_height = cell_height - 100
        card_size = (min(cell_width - 48, round(card_height * 1.5)), card_height)

        for color in Color:
            with self.subTest(color=color):
                image = np.full((*IMAGE_SIZE, 3), 245, dtype=np.uint8)
                for row, shading in enumerate(Shading):
                    for column, shape in enumerate(Shape):
                        center = (
                            column * cell_width + cell_width // 2,
                            row * cell_height + cell_height // 2,
                        )
                        card = Card(shape=shape, color=color, shading=shading, number=Number.THREE)
                        draw_card(image, card, position=center, size=card_size)
                        cv2.putText(
                            image,
                            f"{shape.value} / {shading.value}",
                            (column * cell_width + 24, row * cell_height + 38),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (30, 30, 30),
                            2,
                            cv2.LINE_AA,
                        )

                output_path = OUTPUT_DIR / f"{color.name.lower()}.png"
                self.assertTrue(cv2.imwrite(str(output_path), image))
                self.assertEqual(cv2.imread(str(output_path)).shape[:2], IMAGE_SIZE)
