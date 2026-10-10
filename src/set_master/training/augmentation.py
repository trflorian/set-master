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
            A.Perspective(scale=(0.04, 0.18), fit_output=True, keep_size=True, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=45, g_shift_limit=45, b_shift_limit=45, p=0.6),
            A.RandomGamma(gamma_limit=(65, 145), p=0.5),
            A.RandomShadow(p=0.4),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9))], p=0.35),
            A.GaussNoise(std_range=(0.01, 0.06), p=0.4),
            A.ImageCompression(quality_range=(45, 95), p=0.4),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False), seed=seed)

    def __call__(self, image: np.ndarray, target: dict[str, Tensor],
                 polygons: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Tensor]]:
        lengths = [len(polygon) for polygon in polygons]
        points = [point for polygon in polygons for point in polygon.tolist()]
        if "keypoints" in target:
            points.extend(target["keypoints"][..., :2].reshape(-1, 2).tolist())
        transformed = self.transform(image=image, keypoints=points)
        image = transformed["image"]
        coords = np.asarray(transformed["keypoints"], dtype=np.float32).reshape(-1, 2)
        height, width = image.shape[:2]
        boxes, areas, obbs = [], [], []
        offset = 0
        for length in lengths:
            polygon = coords[offset:offset + length]
            offset += length
            lower = np.clip(polygon.min(axis=0), [0, 0], [width, height])
            upper = np.clip(polygon.max(axis=0) + 1, [0, 0], [width, height])
            boxes.append([*lower, *upper])
            areas.append(abs(cv2.contourArea(polygon)))
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
