import json
import math
import random
from typing import Any
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from set_master.generation.card import draw_card, draw_card_surface
from set_master.generation.features import FeatureStyle
from set_master.models import BackgroundSplit, BackgroundType, Card, Color, Number, Shading, ShadingPattern, Shape, SceneLayout
from set_master.paths import BACKGROUNDS_DIR, SYNTHETIC_DATASETS_DIR
from set_master.generation.backgrounds import BackgroundRecord, crop_background, load_backgrounds
from set_master.geometry import geometry_annotation
from set_master.annotation_visualization import draw_annotation_geometry


@dataclass(frozen=True)
class DatasetConfig:
    output_root: Path = SYNTHETIC_DATASETS_DIR
    run_name: str | None = None
    image_count: int = 100
    image_width: int = 1280
    image_height: int = 720
    cards_per_image: tuple[int, int] = (5, 15)
    card_width: tuple[int, int] = (45, 215)
    card_scale_jitter: float = 0.10
    layout_orientations: tuple[int, ...] = (0, 90)
    card_aspect_ratio: tuple[float, float] = (0.64, 0.69)
    rotation_degrees: tuple[float, float] = (-180.0, 180.0)
    skew_x_degrees: tuple[float, float] = (-15.0, 15.0)
    skew_y_degrees: tuple[float, float] = (-15.0, 15.0)
    perspective_jitter: tuple[float, float] = (0.0, 0.12)
    layouts: tuple[SceneLayout, ...] = tuple(SceneLayout)
    layout_gap: tuple[float, float] = (0.02, 0.18)
    layout_rotation_jitter: float = 8.0
    scene_perspective: tuple[float, float] = (0.0, 0.4)
    overlap_probability: float = 0.15
    max_overlap_fraction: float = 0.06
    shadow_opacity: tuple[float, float] = (0.08, 0.20)
    shadow_blur: int = 11
    shadow_offset: tuple[int, int] = (2, 5)
    illumination_strength: tuple[float, float] = (0.0, 0.18)
    glare_probability: float = 0.35
    glare_strength: tuple[float, float] = (0.15, 0.50)
    glare_radius_ratio: tuple[float, float] = (0.08, 0.20)
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
    background_type: BackgroundType = BackgroundType.GRAY
    background_dir: Path = BACKGROUNDS_DIR
    background_split: BackgroundSplit = BackgroundSplit.TRAIN
    real_background_probability: float = 0.5
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
        if not 0 <= self.real_background_probability <= 1:
            raise ValueError("Real background probability must be between 0 and 1")
        if not 0 <= self.card_scale_jitter <= 0.3:
            raise ValueError("Card scale jitter must be between zero and 0.3")
        if not self.layout_orientations or any(angle not in (0, 90) for angle in self.layout_orientations):
            raise ValueError("Select portrait (0) and/or landscape (90) layout orientations")
        if not self.layouts:
            raise ValueError("Select at least one scene layout")
        for name in ("scene_perspective", "layout_gap", "shadow_opacity", "illumination_strength", "glare_strength", "glare_radius_ratio"):
            low, high = getattr(self, name)
            if not 0 <= low <= high <= 1:
                raise ValueError(f"{name} must be ordered and between zero and one")
        if not 0 <= self.glare_probability <= 1 or self.glare_radius_ratio[0] <= 0:
            raise ValueError("Glare probability must be in [0, 1] and radius positive")
        if not 0 <= self.overlap_probability <= 1 or not 0 <= self.max_overlap_fraction <= 0.15:
            raise ValueError("Overlap probability must be in [0, 1], maximum overlap in [0, 0.15]")
        if self.shadow_blur < 1 or self.shadow_blur % 2 != 1:
            raise ValueError("Shadow blur must be a positive odd number")
        if not 0 <= self.shadow_offset[0] <= self.shadow_offset[1] or not 0 <= self.layout_rotation_jitter <= 30:
            raise ValueError("Shadow offsets must be nonnegative and layout jitter between 0 and 30")
        if self.placement_attempts < 1:
            raise ValueError("Placement attempts must be positive")
        if self.run_name is not None and (not self.run_name or Path(self.run_name).name != self.run_name):
            raise ValueError("Run name must be a non-empty directory name")
        for name in (
            "cards_per_image", "card_width", "card_aspect_ratio", "rotation_degrees", "perspective_jitter",
            "skew_x_degrees", "skew_y_degrees",
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
        if not 0 <= self.perspective_jitter[0] <= self.perspective_jitter[1] <= 0.3:
            raise ValueError("Perspective jitter must be between 0 and 0.3")
        for name in ("skew_x_degrees", "skew_y_degrees"):
            if not -30 <= getattr(self, name)[0] <= getattr(self, name)[1] <= 30:
                raise ValueError(f"{name} must stay between -30 and 30 degrees")
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
    perspective_offsets: np.ndarray | None = None,
    skew_angles: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    perspective = np.eye(3)
    if perspective_offsets is not None:
        perspective = cv2.getPerspectiveTransform(corners, corners + perspective_offsets)
    skew_x, skew_y = (math.tan(math.radians(value)) for value in skew_angles)
    shear = np.array([[1, skew_x, 0], [skew_y, 1, 0], [0, 0, 1]], dtype=np.float64)
    rotation = np.eye(3)
    rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    transform = rotation @ shear @ perspective
    transformed_corners = cv2.perspectiveTransform(corners.reshape(1, 4, 2), transform)[0]
    minimum = np.floor(transformed_corners.min(axis=0)) - 2
    maximum = np.ceil(transformed_corners.max(axis=0)) + 2
    rotated_width, rotated_height = (maximum - minimum + 1).astype(int).tolist()
    translation = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]])
    transform = translation @ transform
    rotated_sprite = cv2.warpPerspective(
        sprite, transform, (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask = cv2.warpPerspective(
        source_mask, transform, (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    corners = cv2.perspectiveTransform(corners.reshape(1, 4, 2), transform)[0]
    return rotated_sprite, rotated_mask, corners


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


def _finish_scene(
    image: np.ndarray, background: np.ndarray, instances: np.ndarray, annotations: list[dict[str, Any]],
    config: DatasetConfig, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Warp the shared plane and rebuild visible geometry after card occlusion."""
    height, width = image.shape[:2]
    strength = rng.uniform(*config.scene_perspective)
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    inset = strength * width * 0.35
    top = strength * height * 0.5
    destination = np.float32([[inset, top], [width - 1 - inset, top],
                             [width - 1, height - 1], [0, height - 1]])
    if rng.random() < 0.5:
        destination[:, 1] = height - 1 - destination[::-1, 1]
        destination[:, 0] = destination[::-1, 0]
    transform = cv2.getPerspectiveTransform(source, destination)
    warped_background = cv2.warpPerspective(background, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    footprint = cv2.warpPerspective(np.ones((height, width), dtype=np.uint8), transform,
                                   (width, height), flags=cv2.INTER_NEAREST)
    image = cv2.warpPerspective(image, transform, (width, height))
    # Extend only the background: reflecting rendered cards would introduce unlabeled duplicates.
    image[footprint == 0] = warped_background[footprint == 0]
    instances = cv2.warpPerspective(instances, transform, (width, height), flags=cv2.INTER_NEAREST)
    visible_annotations = []
    for annotation in annotations:
        mask = np.uint8(instances == annotation["id"]) * 255
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [cv2.approxPolyDP(contour, 0.75, True).reshape(-1, 2)
                    for contour in contours if cv2.contourArea(contour) >= 1]
        polygons = [polygon for polygon in polygons if len(polygon) >= 3]
        if not polygons:
            raise ValueError("Visible card has no valid segmentation contour")
        corners = np.float32(annotation["keypoints"]).reshape(4, 3)[:, :2]
        corners = cv2.perspectiveTransform(corners[None], transform)[0]
        annotation.update(geometry_annotation(corners, np.concatenate(polygons).astype(np.float32), card_relative=True))
        # Rounded corners need a small neighborhood when checking whether another card covers them.
        keypoints = np.float32(annotation["keypoints"]).reshape(4, 3)
        for point in keypoints:
            x, y = np.rint(point[:2]).astype(int)
            neighborhood = instances[max(0, y - 3):min(height, y + 4), max(0, x - 3):min(width, x + 4)]
            if not np.any(neighborhood == annotation["id"]):
                point[2] = 1
        annotation["keypoints"] = keypoints.flatten().tolist()
        annotation["segmentation"] = [polygon.flatten().astype(float).tolist() for polygon in polygons]
        annotation["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        annotation["area"] = int(len(xs))
        annotation["scene_transform"] = transform.tolist()
        visible_annotations.append(annotation)
    strength = rng.uniform(*config.illumination_strength)
    if strength:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        angle = rng.uniform(0, 2 * math.pi)
        gradient = (xx / width - 0.5) * math.cos(angle) + (yy / height - 0.5) * math.sin(angle)
        image = np.clip(image.astype(np.float32) * (1 + strength * gradient[..., None]), 0, 255).astype(np.uint8)
    for annotation in visible_annotations:
        if rng.random() >= config.glare_probability:
            continue
        corners = np.float32(annotation["keypoints"]).reshape(4, 3)[:, :2]
        center = rng.choice(corners)
        radius = max(2.0, np.linalg.norm(corners[1] - corners[0]) * rng.uniform(*config.glare_radius_ratio))
        center = center + np.float32([rng.uniform(-radius, radius), rng.uniform(-radius, radius)])
        left, top = max(0, int(center[0] - 3 * radius)), max(0, int(center[1] - 3 * radius))
        right, bottom = min(width, int(center[0] + 3 * radius) + 1), min(height, int(center[1] + 3 * radius) + 1)
        yy, xx = np.mgrid[top:bottom, left:right]
        light = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * radius ** 2))
        alpha = light[..., None] * rng.uniform(*config.glare_strength)
        view = image[top:bottom, left:right]
        view[:] = np.clip(view * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8)
    return image, np.uint8(instances > 0) * 255, visible_annotations


def generate_coco_dataset(config: DatasetConfig) -> Path:
    """Create a new synthetic card dataset and return its run directory."""
    rng = random.Random(config.seed)
    background_rng = random.Random(config.seed + 1)
    backgrounds: list[BackgroundRecord] = []
    if config.background_type != BackgroundType.GRAY:
        backgrounds = [record for record in load_backgrounds(config.background_dir)
                       if record.approved and record.split == config.background_split.value]
        if not backgrounds:
            raise ValueError(f"No approved {config.background_split.value} backgrounds in {config.background_dir}. "
                             "Download and review backgrounds in download_backgrounds.ipynb first.")
    license_keys = sorted({(record.license, record.license_url) for record in backgrounds})
    license_ids = {key: index + 1 for index, key in enumerate(license_keys)}
    run_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = config.output_root / run_name
    (run_dir / "images").mkdir(parents=True, exist_ok=False)
    (run_dir / "masks").mkdir()
    (run_dir / "debug").mkdir()

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
        background = None
        if (config.background_type == BackgroundType.REAL_SCENE or
                config.background_type == BackgroundType.MIXED and
                background_rng.random() < config.real_background_probability):
            background = background_rng.choice(backgrounds)
            path = config.background_dir / background.file_name
            source = cv2.imread(str(path))
            if source is None:
                raise FileNotFoundError(f"Could not read background {path}")
            image = crop_background(source, config.image_width, config.image_height,
                                    background_rng.random(), background_rng.random())
        scene_background = image.copy()
        # Float32 supports nearest-neighbor OpenCV warps while preserving these integer instance IDs.
        occupied = np.zeros((config.image_height, config.image_width), dtype=np.float32)
        target_count = rng.randint(*config.cards_per_image)
        image_annotations = []
        layout = rng.choice(config.layouts)
        layout_angle = rng.choice(config.layout_orientations)
        aspect = rng.uniform(*config.card_aspect_ratio)
        surface_ratio = aspect if layout_angle == 0 else 1 / aspect
        columns = max(1, round(math.sqrt(target_count * config.image_width / config.image_height / surface_ratio)))
        rows = max(1, math.ceil(target_count / columns))
        gap = rng.uniform(*config.layout_gap)
        if layout == SceneLayout.CLUSTERED:
            gap += 0.20
        layout_width = round(math.exp(rng.uniform(math.log(config.card_width[0]), math.log(config.card_width[1]))))
        if layout == SceneLayout.SCATTERED and target_count:
            # Random packing needs spare area; dense close-up grids can use much more of the canvas.
            packing_width = math.sqrt(config.image_width * config.image_height * 0.25 * aspect / target_count)
            layout_width = min(layout_width, max(config.card_width[0], round(packing_width)))
        layout_height = round(layout_width / aspect)
        surface_width, surface_height = ((layout_width, layout_height) if layout_angle == 0
                                         else (layout_height, layout_width))
        fit = (1.0 if layout == SceneLayout.SCATTERED else
               min(1.0, config.image_width / (columns * surface_width * (1 + gap) + 10),
                   config.image_height / (rows * surface_height * (1 + gap) + 10)))
        layout_width, layout_height = max(8, round(layout_width * fit)), max(12, round(layout_height * fit))
        surface_width, surface_height = ((layout_width, layout_height) if layout_angle == 0
                                         else (layout_height, layout_width))
        cell_width = surface_width * (1 + gap)
        cell_height = surface_height * (1 + gap)
        layout_left = (config.image_width - columns * cell_width) / 2
        layout_top = (config.image_height - rows * cell_height) / 2
        shadow_strength = rng.uniform(*config.shadow_opacity)
        shadow_dx = rng.randint(*config.shadow_offset) * rng.choice((-1, 1))
        shadow_dy = rng.randint(*config.shadow_offset)
        for card_index in range(target_count):
            allow_overlap = rng.random() < config.overlap_probability
            placed = False
            for _ in range(config.placement_attempts):
                width = round(layout_width * rng.uniform(1 - config.card_scale_jitter, 1 + config.card_scale_jitter))
                width = int(np.clip(width, *config.card_width))
                height = round(width / aspect)
                if layout != SceneLayout.SCATTERED:
                    width, height = layout_width, layout_height
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
                angle = (rng.uniform(*config.rotation_degrees) if layout == SceneLayout.SCATTERED
                         else layout_angle + rng.uniform(-config.layout_rotation_jitter, config.layout_rotation_jitter))
                skew_angles = (rng.uniform(*config.skew_x_degrees), rng.uniform(*config.skew_y_degrees))
                jitter = rng.uniform(*config.perspective_jitter)
                if layout != SceneLayout.SCATTERED:
                    skew_angles = (0.0, 0.0)
                    jitter = 0.0
                offsets = np.float32([
                    [rng.uniform(0, jitter) * width, rng.uniform(0, jitter) * height],
                    [-rng.uniform(0, jitter) * width, rng.uniform(0, jitter) * height],
                    [-rng.uniform(0, jitter) * width, -rng.uniform(0, jitter) * height],
                    [rng.uniform(0, jitter) * width, -rng.uniform(0, jitter) * height],
                ])
                rotated_sprite, rotated_mask, corners = _rotated_card(
                    card, width, height, radius, feature_size, round(spacing), style, angle, offsets, skew_angles,
                )
                card_height, card_width = rotated_mask.shape
                if card_width > config.image_width or card_height > config.image_height:
                    continue

                if layout == SceneLayout.SCATTERED:
                    x = rng.randint(0, config.image_width - card_width)
                    y = rng.randint(0, config.image_height - card_height)
                else:
                    jitter = 0.18 if layout == SceneLayout.CLUSTERED else 0.025
                    ys, xs = np.nonzero(rotated_mask)
                    surface_width, surface_height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
                    if surface_width > cell_width or surface_height > cell_height:
                        continue
                    # Keep the visible surface inside its cell, including after local rotation.
                    jitter_x = min(jitter * cell_width, (cell_width - surface_width) / 2)
                    jitter_y = min(jitter * cell_height, (cell_height - surface_height) / 2)
                    center_x = layout_left + (card_index % columns + 0.5) * cell_width + rng.uniform(-jitter_x, jitter_x)
                    center_y = layout_top + (card_index // columns + 0.5) * cell_height + rng.uniform(-jitter_y, jitter_y)
                    x = int(np.clip(center_x - card_width / 2, 0, config.image_width - card_width))
                    y = int(np.clip(center_y - card_height / 2, 0, config.image_height - card_height))
                mask_view = rotated_mask > 0
                occupied_view = occupied[y:y + card_height, x:x + card_width]
                overlap = occupied_view[mask_view]
                overlap_count = np.count_nonzero(overlap)
                if overlap_count and (not allow_overlap or overlap_count / len(overlap) > config.max_overlap_fraction):
                    continue
                if overlap_count:
                    ids, counts = np.unique(overlap[overlap > 0], return_counts=True)
                    if any(count / np.count_nonzero(occupied == instance_id) > config.max_overlap_fraction
                           for instance_id, count in zip(ids, counts)):
                        continue

                if shadow_strength:
                    padding = config.shadow_blur + max(abs(shadow_dx), abs(shadow_dy))
                    left, top = max(0, x - padding), max(0, y - padding)
                    right, bottom = min(config.image_width, x + card_width + padding), min(config.image_height, y + card_height + padding)
                    shadow = np.zeros((bottom - top, right - left), dtype=np.float32)
                    shadow[y - top:y - top + card_height, x - left:x - left + card_width] = mask_view
                    shadow = cv2.warpAffine(shadow, np.float32([[1, 0, shadow_dx], [0, 1, shadow_dy]]), (right - left, bottom - top))
                    shadow = cv2.GaussianBlur(shadow, (config.shadow_blur, config.shadow_blur), 0)
                    view = image[top:bottom, left:right]
                    view[:] = np.clip(view * (1 - shadow_strength * shadow[..., None]), 0, 255).astype(np.uint8)

                image_view = image[y:y + card_height, x:x + card_width]
                image_view[mask_view] = rotated_sprite[mask_view]
                occupied_view[mask_view] = annotation_id

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
                annotation.update(geometry_annotation(
                    corners + np.float32([x, y]),
                    np.float32(annotation["segmentation"][0]).reshape(-1, 2),
                    card_relative=True,
                ))
                annotation["geometry_source"] = "render_transform"
                annotation["orientation_source"] = "render_transform"
                annotation["render_transform"] = {"rotation_degrees": angle,
                                                  "skew_x_degrees": skew_angles[0],
                                                  "skew_y_degrees": skew_angles[1],
                                                  "perspective_jitter": jitter}
                image_annotations.append(annotation)
                annotation_id += 1
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    f"Could not place {target_count} {layout.value} cards in image {image_id} "
                    f"(scene width {layout_width}, gap {gap:.3f}). "
                    "Increase the image size, reduce card sizes/count, or reduce the rotation range."
                )

        image, binary_mask, image_annotations = _finish_scene(image, scene_background, occupied, image_annotations, config, rng)
        image_name = f"{image_id:06d}.jpg"
        mask_name = f"{image_id:06d}.png"
        if not cv2.imwrite(str(run_dir / "images" / image_name), image):
            raise OSError(f"Failed to write image {image_name}")
        if not cv2.imwrite(str(run_dir / "masks" / mask_name), binary_mask):
            raise OSError(f"Failed to write binary mask {mask_name}")
        if not cv2.imwrite(str(run_dir / "debug" / image_name), draw_annotation_geometry(image, image_annotations)):
            raise OSError(f"Failed to write debug overlay {image_name}")

        image_info: dict[str, object] = {
            "id": image_id,
            "layout": layout.value,
            "layout_orientation": layout_angle if layout != SceneLayout.SCATTERED else None,
            "width": config.image_width,
            "height": config.image_height,
            "file_name": f"images/{image_name}",
            "mask_file": f"masks/{mask_name}",
            "debug_file": f"debug/{image_name}",
        }
        if background is not None:
            image_info["background"] = asdict(background)
            image_info["license"] = license_ids[(background.license, background.license_url)]
        images.append(image_info)
        annotations.extend(image_annotations)

    coco = {
        "info": {"description": "Synthetic top-down SET card scenes", "version": "1.0"},
        "licenses": [{"id": license_ids[key], "name": key[0], "url": key[1]} for key in license_keys],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "card", "supercategory": "object",
                        "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
                        "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]]}],
    }
    (run_dir / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(asdict(config), default=str, indent=2) + "\n")
    return run_dir
