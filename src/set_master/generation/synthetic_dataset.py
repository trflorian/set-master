import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from set_master.generation.card import draw_card, draw_card_surface
from set_master.generation.features import FeatureStyle
from set_master.models import Card, Color, Number, Shading, ShadingPattern, Shape


@dataclass(frozen=True)
class DatasetConfig:
    output_root: Path = Path("data/synthetic_datasets")
    run_name: str | None = None
    image_count: int = 100
    image_width: int = 1280
    image_height: int = 720
    cards_per_image: tuple[int, int] = (3, 7)
    card_width: tuple[int, int] = (100, 170)
    card_aspect_ratio: tuple[float, float] = (0.64, 0.69)
    rotation_degrees: tuple[float, float] = (-25.0, 25.0)
    feature_count: tuple[int, int] = (1, 3)
    feature_width_ratio: tuple[float, float] = (0.60, 0.68)
    feature_height_ratio: tuple[float, float] = (0.16, 0.20)
    symbol_spacing_ratio: tuple[float, float] = (0.95, 1.05)
    corner_radius_ratio: tuple[float, float] = (0.035, 0.055)
    stripe_spacing: tuple[int, int] = (4, 6)
    line_thickness: tuple[int, int] = (1, 2)
    squiggle_amplitude: tuple[float, float] = (0.16, 0.20)
    squiggle_waves: tuple[float, float] = (0.95, 1.05)
    squiggle_thickness: tuple[float, float] = (0.60, 0.70)
    background_gray: int = 185
    shapes: tuple[Shape, ...] = tuple(Shape)
    colors: tuple[Color, ...] = tuple(Color)
    shadings: tuple[Shading, ...] = tuple(Shading)
    shading_patterns: tuple[ShadingPattern, ...] = tuple(ShadingPattern)
    seed: int = 42
    placement_attempts: int = 250

    def __post_init__(self) -> None:
        if self.image_count < 1 or self.image_width < 1 or self.image_height < 1:
            raise ValueError("Image count and dimensions must be positive")
        if not 0 <= self.background_gray <= 255:
            raise ValueError("Background gray must be between 0 and 255")
        if self.placement_attempts < 1:
            raise ValueError("Placement attempts must be positive")
        if self.run_name is not None and (not self.run_name or Path(self.run_name).name != self.run_name):
            raise ValueError("Run name must be a non-empty directory name")
        for name in (
            "cards_per_image", "card_width", "card_aspect_ratio", "rotation_degrees",
            "feature_count", "feature_width_ratio", "feature_height_ratio",
            "symbol_spacing_ratio", "corner_radius_ratio", "stripe_spacing",
            "line_thickness", "squiggle_amplitude", "squiggle_waves",
            "squiggle_thickness",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} minimum cannot exceed its maximum")
        if self.card_width[0] < 1 or self.card_aspect_ratio[0] <= 0:
            raise ValueError("Card width and width/height aspect ratio must be positive")
        for name in ("shapes", "colors", "shadings", "shading_patterns"):
            if not getattr(self, name):
                raise ValueError(f"At least one value must be selected for {name}")


def _rotated_card(
    card: Card,
    width: int,
    height: int,
    corner_radius: int,
    feature_size: tuple[int, int],
    symbol_spacing: int,
    style: FeatureStyle,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    sprite = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_card(
        sprite,
        card,
        size=(width, height),
        feature_size=feature_size,
        symbol_spacing=symbol_spacing,
        corner_radius=corner_radius,
        feature_style=style,
    )
    surface = np.zeros_like(sprite)
    draw_card_surface(surface, (width // 2, height // 2), (width, height), corner_radius)
    source_mask = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)

    radians = math.radians(angle)
    rotated_width = math.ceil(height * abs(math.sin(radians)) + width * abs(math.cos(radians)))
    rotated_height = math.ceil(height * abs(math.cos(radians)) + width * abs(math.sin(radians)))
    center = (width / 2, height / 2)
    transform = cv2.getRotationMatrix2D(center, angle, 1.0)
    transform[0, 2] += rotated_width / 2 - center[0]
    transform[1, 2] += rotated_height / 2 - center[1]
    rotated_sprite = cv2.warpAffine(
        sprite, transform, (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask = cv2.warpAffine(
        source_mask, transform, (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return rotated_sprite, rotated_mask


def _annotation_polygon(mask: np.ndarray, offset_x: int, offset_y: int) -> list[float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find a card contour for COCO segmentation")
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
    if len(contour) < 3:
        raise ValueError("Card contour has fewer than three points")
    points = contour.astype(np.float64)
    points[:, 0] += offset_x
    points[:, 1] += offset_y
    return points.flatten().tolist()


def generate_coco_dataset(config: DatasetConfig) -> Path:
    """Create a new synthetic card dataset and return its run directory."""
    rng = random.Random(config.seed)
    run_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = config.output_root / run_name
    (run_dir / "images").mkdir(parents=True, exist_ok=False)
    (run_dir / "masks").mkdir()

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1

    for image_index in range(config.image_count):
        image_id = image_index + 1
        image = np.full(
            (config.image_height, config.image_width, 3),
            config.background_gray,
            dtype=np.uint8,
        )
        binary_mask = np.zeros((config.image_height, config.image_width), dtype=np.uint8)
        occupied = np.zeros((config.image_height, config.image_width), dtype=np.uint8)
        target_count = rng.randint(*config.cards_per_image)
        image_annotations = []

        for _ in range(target_count):
            placed = False
            for _ in range(config.placement_attempts):
                width = rng.randint(*config.card_width)
                height = round(width / rng.uniform(*config.card_aspect_ratio))
                number = Number(rng.randint(*config.feature_count))
                spacing = height * 0.24 * rng.uniform(*config.symbol_spacing_ratio)
                feature_size = (
                    max(8, round(width * rng.uniform(*config.feature_width_ratio))),
                    max(8, round(height * rng.uniform(*config.feature_height_ratio))),
                )
                radius = round(min(width, height) * rng.uniform(*config.corner_radius_ratio))
                style = FeatureStyle(
                    shading_pattern=rng.choice(config.shading_patterns),
                    stripe_spacing=rng.randint(*config.stripe_spacing),
                    line_thickness=rng.randint(*config.line_thickness),
                    squiggle_amplitude=rng.uniform(*config.squiggle_amplitude),
                    squiggle_waves=rng.uniform(*config.squiggle_waves),
                    squiggle_thickness=rng.uniform(*config.squiggle_thickness),
                )
                card = Card(
                    shape=rng.choice(config.shapes),
                    color=rng.choice(config.colors),
                    shading=rng.choice(config.shadings),
                    number=number,
                )
                angle = rng.uniform(*config.rotation_degrees)
                rotated_sprite, rotated_mask = _rotated_card(
                    card, width, height, radius, feature_size, round(spacing), style, angle
                )
                card_height, card_width = rotated_mask.shape
                if card_width > config.image_width or card_height > config.image_height:
                    continue

                x = rng.randint(0, config.image_width - card_width)
                y = rng.randint(0, config.image_height - card_height)
                mask_view = rotated_mask > 0
                occupied_view = occupied[y:y + card_height, x:x + card_width]
                if np.any(occupied_view[mask_view]):
                    continue

                image_view = image[y:y + card_height, x:x + card_width]
                image_view[mask_view] = rotated_sprite[mask_view]
                mask_view_on_image = binary_mask[y:y + card_height, x:x + card_width]
                mask_view_on_image[mask_view] = 255
                occupied_view[mask_view] = 1

                ys, xs = np.nonzero(mask_view)
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x + x_min, y + y_min, x_max - x_min + 1, y_max - y_min + 1],
                    "area": int(np.count_nonzero(mask_view)),
                    "segmentation": [_annotation_polygon(rotated_mask, x, y)],
                    "iscrowd": 0,
                }
                image_annotations.append(annotation)
                annotation_id += 1
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    f"Could not place {target_count} non-overlapping cards in image {image_id}. "
                    "Increase the image size, reduce card sizes/count, or reduce the rotation range."
                )

        image_name = f"{image_id:06d}.jpg"
        mask_name = f"{image_id:06d}.png"
        if not cv2.imwrite(str(run_dir / "images" / image_name), image):
            raise OSError(f"Failed to write image {image_name}")
        if not cv2.imwrite(str(run_dir / "masks" / mask_name), binary_mask):
            raise OSError(f"Failed to write binary mask {mask_name}")

        images.append({
            "id": image_id,
            "width": config.image_width,
            "height": config.image_height,
            "file_name": f"images/{image_name}",
            "mask_file": f"masks/{mask_name}",
        })
        annotations.extend(image_annotations)

    coco = {
        "info": {"description": "Synthetic top-down SET card scenes", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "card", "supercategory": "object"}],
    }
    (run_dir / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return run_dir
