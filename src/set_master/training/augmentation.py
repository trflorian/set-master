import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor

from set_master.geometry import order_corners, oriented_box


class CardAugmentation:
    """Transform contours and corners together, then rebuild detection geometry."""

    def __init__(self, seed: int = 42) -> None:
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Perspective(scale=(0.02, 0.08), fit_output=True, keep_size=True, p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.20, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=18, val_shift_limit=10, p=0.4),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
            A.RandomGamma(gamma_limit=(85, 115), p=0.4),
            A.OneOf([
                A.Illumination(mode="linear", intensity_range=(0.05, 0.20)),
                A.Illumination(mode="gaussian", intensity_range=(0.05, 0.20)),
            ], p=0.6),
            A.RandomShadow(num_shadows_limit=(1, 2), shadow_intensity_range=(0.08, 0.25), p=0.2),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9))], p=0.35),
            A.GaussNoise(std_range=(0.01, 0.06), p=0.4),
            A.ImageCompression(quality_range=(45, 95), p=0.4),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False), seed=seed)

    def __call__(self, image: np.ndarray, target: dict[str, Tensor],
                 polygons: list[list[np.ndarray]]) -> tuple[np.ndarray, dict[str, Tensor]]:
        lengths = [[len(segment) for segment in polygon] for polygon in polygons]
        points = [point for polygon in polygons for segment in polygon for point in segment.tolist()]
        if "keypoints" in target:
            points.extend(target["keypoints"][..., :2].reshape(-1, 2).tolist())
        transformed = self.transform(image=image, keypoints=points)
        image = transformed["image"]
        coords = np.asarray(transformed["keypoints"], dtype=np.float32).reshape(-1, 2)
        height, width = image.shape[:2]
        boxes, areas, obbs = [], [], []
        offset = 0
        for segment_lengths in lengths:
            segments = []
            for length in segment_lengths:
                segments.append(coords[offset:offset + length])
                offset += length
            polygon = np.concatenate(segments)
            lower = np.clip(polygon.min(axis=0), [0, 0], [width, height])
            upper = np.clip(polygon.max(axis=0) + 1, [0, 0], [width, height])
            boxes.append([*lower, *upper])
            areas.append(sum(abs(cv2.contourArea(segment)) for segment in segments))
            obbs.append(oriented_box(polygon))
        target = dict(target)
        target["boxes"] = torch.tensor(np.asarray(boxes), dtype=torch.float32).reshape(-1, 4)
        target["area"] = torch.tensor(areas, dtype=torch.float32)
        if "oriented_boxes" in target:
            target["oriented_boxes"] = torch.tensor(obbs, dtype=torch.float32).reshape(-1, 5)
        if "keypoints" in target:
            corners = coords[offset:].reshape(-1, 4, 2)
            ordered = np.asarray([order_corners(quad) for quad in corners], dtype=np.float32).reshape(-1, 4, 2)
            target["keypoints"] = torch.cat((torch.from_numpy(ordered), target["keypoints"][..., 2:]), dim=-1)
        return image, target
