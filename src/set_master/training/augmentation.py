import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor

from set_master.geometry import order_corners, oriented_box


@dataclass(frozen=True)
class AugmentationConfig:
    flip_probability: float = 0.5
    perspective_probability: float = 0.4
    perspective_scale: tuple[float, float] = (0.01, 0.07)
    min_visibility: float = 0.95
    brightness_probability: float = 0.65
    brightness_limit: float = 0.22
    contrast_limit: float = 0.20
    temperature_probability: float = 0.35
    temperature_range: tuple[int, int] = (4000, 8500)
    temperature_strength: float = 0.5
    gamma_probability: float = 0.35
    gamma_range: tuple[int, int] = (80, 130)
    illumination_probability: float = 0.75
    illumination_strength: tuple[float, float] = (0.12, 0.40)
    shadow_probability: float = 0.4
    shadow_strength: tuple[float, float] = (0.15, 0.45)
    shadow_size: tuple[float, float] = (0.12, 0.45)
    glare_probability: float = 0.45
    glare_strength: tuple[float, float] = (0.40, 0.85)
    glare_radius_ratio: tuple[float, float] = (0.12, 0.30)
    glare_spots: tuple[int, int] = (1, 2)
    blur_probability: float = 0.3
    blur_kernel: tuple[int, int] = (3, 7)
    downscale_probability: float = 0.35
    downscale_range: tuple[float, float] = (0.55, 0.90)
    noise_probability: float = 0.3
    noise_std: tuple[float, float] = (0.005, 0.035)
    compression_probability: float = 0.4
    jpeg_quality: tuple[int, int] = (55, 95)

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith('probability') and not 0 <= value <= 1:
                raise ValueError(f'{field.name} must be between zero and one')
            if isinstance(value, tuple) and (len(value) != 2 or value[0] > value[1] or value[0] < 0):
                raise ValueError(f'{field.name} must be an ordered nonnegative pair')
        for name in ('perspective_scale', 'illumination_strength', 'shadow_strength', 'glare_strength',
                     'noise_std', 'downscale_range'):
            if getattr(self, name)[1] > 1:
                raise ValueError(f'{name} must stay within [0, 1]')
        if not 0 < self.downscale_range[0] <= 1 or not 0 < self.min_visibility <= 1:
            raise ValueError('Downscale and minimum visibility must be positive and at most one')
        if not 0 <= self.temperature_strength <= 1:
            raise ValueError('Temperature strength must be in [0, 1]')
        if not 0 <= self.brightness_limit <= 0.5 or not 0 <= self.contrast_limit <= 0.5:
            raise ValueError('Brightness and contrast limits must be in [0, 0.5]')
        if not 3000 <= self.temperature_range[0] <= 6000 <= self.temperature_range[1] <= 15000:
            raise ValueError('Temperature range must include 6000 K and stay within 3000–15000 K')
        if self.blur_kernel[0] < 3 or any(value % 2 != 1 for value in self.blur_kernel):
            raise ValueError('Blur kernels must be odd and at least three')
        if self.glare_spots[0] < 1 or self.glare_radius_ratio[0] <= 0 or self.shadow_size[0] <= 0:
            raise ValueError('Light sizes and glare spot counts must be positive')
        if self.gamma_range[0] < 1 or not 1 <= self.jpeg_quality[0] <= self.jpeg_quality[1] <= 100:
            raise ValueError('Gamma must be positive and JPEG quality between 1 and 100')

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + '\n')

    @classmethod
    def load(cls, path: Path) -> 'AugmentationConfig':
        values = json.loads(path.read_text())
        for field in fields(cls):
            if field.name in values and isinstance(field.default, tuple):
                values[field.name] = tuple(values[field.name])
        return cls(**values)


def _clip_polygon(polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip contour edges to the image rectangle without clamping individual vertices."""
    points = polygon.tolist()
    for axis, edge, lower in ((0, 0, True), (0, width - 1, False),
                              (1, 0, True), (1, height - 1, False)):
        output = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            inside = current[axis] >= edge if lower else current[axis] <= edge
            previous_inside = previous[axis] >= edge if lower else previous[axis] <= edge
            if inside != previous_inside:
                fraction = (edge - previous[axis]) / (current[axis] - previous[axis])
                output.append([previous[d] + fraction * (current[d] - previous[d]) for d in (0, 1)])
            if inside:
                output.append(current)
        points = output
        if not points:
            break
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


class CardAugmentation:
    """Transform contours and corners together, then rebuild visible detection geometry."""

    def __init__(self, seed: int = 42, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        c = self.config
        self.transform = A.Compose([
            A.HorizontalFlip(p=c.flip_probability),
            A.VerticalFlip(p=c.flip_probability),
            # Map an interior quadrilateral onto the canvas, avoiding padded black wedges.
            A.Perspective(scale=c.perspective_scale, fit_output=False, keep_size=True,
                          p=c.perspective_probability),
            A.RandomBrightnessContrast(brightness_limit=c.brightness_limit,
                                       contrast_limit=c.contrast_limit, p=c.brightness_probability),
            A.RandomGamma(gamma_limit=c.gamma_range, p=c.gamma_probability),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False), seed=seed)
        self.temperature_transform = A.PlanckianJitter(temperature_limit=c.temperature_range, p=1)
        self.camera_transform = A.Compose([
            A.Downscale(scale_range=c.downscale_range,
                        interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LINEAR},
                        p=c.downscale_probability),
            A.OneOf([A.GaussianBlur(blur_limit=c.blur_kernel), A.MotionBlur(blur_limit=c.blur_kernel)],
                    p=c.blur_probability),
            A.GaussNoise(std_range=c.noise_std, p=c.noise_probability),
            A.ImageCompression(quality_range=c.jpeg_quality, p=c.compression_probability),
        ], seed=seed + 1)

    def __call__(self, image: np.ndarray, target: dict[str, Tensor],
                 polygons: list[list[np.ndarray]]) -> tuple[np.ndarray, dict[str, Tensor]]:
        lengths = [[len(segment) for segment in polygon] for polygon in polygons]
        points = [point for polygon in polygons for segment in polygon for point in segment.tolist()]
        if 'keypoints' in target:
            points.extend(target['keypoints'][..., :2].reshape(-1, 2).tolist())
        mask_args = {"masks": [mask.numpy() for mask in target['masks']]} if 'masks' in target else {}
        transformed = self.transform(image=image, keypoints=points, **mask_args)
        image = transformed['image']
        coords = np.asarray(transformed['keypoints'], dtype=np.float32).reshape(-1, 2)
        height, width = image.shape[:2]
        boxes, areas, obbs, kept = [], [], [], []
        offset = 0
        for index, segment_lengths in enumerate(lengths):
            segments = []
            for length in segment_lengths:
                segments.append(coords[offset:offset + length])
                offset += length
            original_area = sum(abs(cv2.contourArea(segment)) for segment in segments)
            clipped = [_clip_polygon(segment, width, height) for segment in segments]
            clipped = [segment for segment in clipped if len(segment) >= 3]
            area = sum(abs(cv2.contourArea(segment)) for segment in clipped)
            if area < 16 or area / max(original_area, 1) < self.config.min_visibility:
                continue
            polygon = np.concatenate(clipped)
            lower = polygon.min(axis=0)
            upper = np.minimum(polygon.max(axis=0) + 1, [width, height])
            boxes.append([*lower, *upper])
            areas.append(area)
            obbs.append(oriented_box(polygon))
            kept.append(index)
        selection = torch.tensor(kept, dtype=torch.int64)
        target = {name: value[selection] if value.ndim and value.shape[0] == len(polygons) else value
                  for name, value in target.items()}
        target['boxes'] = torch.tensor(np.asarray(boxes), dtype=torch.float32).reshape(-1, 4)
        target['area'] = torch.tensor(areas, dtype=torch.float32)
        if 'masks' in target:
            masks = np.asarray(transformed['masks'], dtype=np.uint8).reshape(-1, height, width)
            target['masks'] = torch.from_numpy(np.ascontiguousarray(masks[kept]))
        if 'oriented_boxes' in target:
            target['oriented_boxes'] = torch.tensor(obbs, dtype=torch.float32).reshape(-1, 5)
        xy = np.empty((0, 4, 2), dtype=np.float32)
        if 'keypoints' in target:
            corners = coords[offset:].reshape(-1, 4, 2)[kept]
            visibility = target['keypoints'][..., 2].numpy().copy()
            ordered, ordered_visibility = [], []
            for quad, flags in zip(corners, visibility):
                new_quad = order_corners(quad)
                # Flips reverse winding; retain corner zero so the short-edge identity survives.
                start = np.linalg.norm(new_quad - quad[0], axis=1).argmin()
                new_quad = np.roll(new_quad, -int(start), axis=0)
                # A reflection swaps the two neighbors of corner zero. Keep its original
                # short-side neighbor as corner one, allowing a different starting vertex.
                if np.linalg.norm(new_quad[1] - quad[1]) > np.linalg.norm(new_quad[3] - quad[1]):
                    new_quad = np.roll(new_quad, -3, axis=0)
                permutation = np.linalg.norm(new_quad[:, None] - quad[None], axis=2).argmin(axis=1)
                flags = flags[permutation]
                outside = ((new_quad[:, 0] < 0) | (new_quad[:, 0] >= width) |
                           (new_quad[:, 1] < 0) | (new_quad[:, 1] >= height))
                flags[outside] = 0
                ordered.append(new_quad)
                ordered_visibility.append(flags)
            xy = np.asarray(ordered, dtype=np.float32).reshape(-1, 4, 2)
            flags = np.asarray(ordered_visibility, dtype=np.float32).reshape(-1, 4, 1)
            target['keypoints'] = torch.from_numpy(np.concatenate((xy, flags), axis=2))
        rng = self.transform.random_generator
        if rng.random() < self.config.temperature_probability:
            self.temperature_transform.set_random_seed(int(rng.integers(2**31)))
            shifted = self.temperature_transform(image=image)['image']
            image = cv2.addWeighted(image, 1 - self.config.temperature_strength,
                                    shifted, self.config.temperature_strength, 0)
        self._add_lighting(image)
        self._add_glare(image, xy)
        self.camera_transform.set_random_seed(int(self.transform.random_generator.integers(2**31)))
        image = self.camera_transform(image=image)['image']
        return image, target

    def _add_lighting(self, image: np.ndarray) -> None:
        rng = self.transform.random_generator
        c = self.config
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        gain = np.ones((height, width), dtype=np.float32)
        if rng.random() < c.illumination_probability:
            angle = rng.uniform(0, 2 * np.pi)
            gradient = (xx / width - 0.5) * np.cos(angle) + (yy / height - 0.5) * np.sin(angle)
            gradient /= max(float(np.abs(gradient).max()), 1e-6)
            gain *= 1 + rng.uniform(*c.illumination_strength) * gradient
        if rng.random() < c.shadow_probability:
            center_x, center_y = rng.uniform(0.1, 0.9, size=2) * [width, height]
            sigma_x = width * rng.uniform(*c.shadow_size)
            sigma_y = height * rng.uniform(*c.shadow_size)
            shadow = np.exp(-0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2))
            gain *= 1 - rng.uniform(*c.shadow_strength) * shadow
        image[:] = np.clip(image.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)

    def _add_glare(self, image: np.ndarray, corners: np.ndarray) -> None:
        rng = self.transform.random_generator
        c = self.config
        if not len(corners) or rng.random() >= c.glare_probability:
            return
        height, width = image.shape[:2]
        for _ in range(int(rng.integers(c.glare_spots[0], c.glare_spots[1] + 1))):
            quad = corners[int(rng.integers(len(corners)))]
            center = quad[int(rng.integers(4))].copy()
            size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
            radius = max(2, size * rng.uniform(*c.glare_radius_ratio))
            center += rng.uniform(-radius * 0.5, radius * 0.5, size=2)
            left, top = max(0, int(center[0] - radius * 3)), max(0, int(center[1] - radius * 3))
            right, bottom = min(width, int(center[0] + radius * 3) + 1), min(height, int(center[1] + radius * 3) + 1)
            if left >= right or top >= bottom:
                continue
            yy, xx = np.mgrid[top:bottom, left:right]
            alpha = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * radius ** 2))
            alpha = alpha[..., None] * rng.uniform(*c.glare_strength)
            view = image[top:bottom, left:right]
            view[:] = np.clip(view * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8)


HARD_AUGMENTATION_CONFIG = replace(
    AugmentationConfig(), brightness_limit=0.28, contrast_limit=0.25,
    illumination_strength=(0.20, 0.55), shadow_strength=(0.25, 0.60), shadow_probability=0.55,
    glare_probability=0.65, glare_strength=(0.55, 0.95),
    downscale_range=(0.4, 0.75), downscale_probability=0.5,
    noise_std=(0.01, 0.045), jpeg_quality=(40, 85),
)
