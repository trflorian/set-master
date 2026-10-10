import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import sigmoid_focal_loss

from set_master.geometry import obb_corners, oriented_box, polygon_iou
from set_master.training.backbone import FrozenDinoFPN
from set_master.training.models.box import CardDetector
from set_master.training.config import DetectorVariant


def rotated_nms(boxes: Tensor, scores: Tensor, threshold: float = 0.3, limit: int = 100) -> Tensor:
    polygons = [obb_corners(box) for box in boxes.detach().cpu().tolist()]
    order = scores.argsort(descending=True).cpu().tolist()
    keep = []
    while order and len(keep) < limit:
        index = order.pop(0)
        keep.append(index)
        order = [other for other in order if polygon_iou(polygons[index], polygons[other]) <= threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


class OrientedDenseDetector(nn.Module):
    """Center-assigned FPN detector with learned short/long sides and a periodic angle."""

    def __init__(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(),
        )
        self.classifier = nn.Conv2d(256, 1, 3, padding=1)
        # dx, dy, log(width), log(height), sin(2 angle), cos(2 angle).
        self.regressor = nn.Conv2d(256, 6, 3, padding=1)
        nn.init.constant_(self.classifier.bias, -math.log(99))
        nn.init.normal_(self.regressor.weight, std=0.01)
        nn.init.zeros_(self.regressor.bias)
        with torch.no_grad():
            self.regressor.bias[5] = 1

    def forward(
        self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        if self.training and targets is None:
            raise ValueError("Oriented training requires targets")
        original_sizes = [image.shape[-2:] for image in images]
        batch, _ = self.transform(images, targets)
        features = list(self.backbone(batch.tensors).values())
        logits, regressions, grids, strides = [], [], [], []
        padded_height, padded_width = batch.tensors.shape[-2:]
        for feature in features:
            shared = self.head(feature)
            height, width = feature.shape[-2:]
            stride = feature.new_tensor([padded_width / width, padded_height / height])
            yy, xx = torch.meshgrid(torch.arange(height, device=feature.device),
                                    torch.arange(width, device=feature.device), indexing="ij")
            grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(feature.dtype)
            grids.append((grid + 0.5) * stride)
            strides.append(stride.expand(len(grid), 2))
            logits.append(self.classifier(shared).flatten(2).squeeze(1))
            regressions.append(self.regressor(shared).flatten(2).transpose(1, 2))
        centers = torch.cat(grids)
        stride_xy = torch.cat(strides)
        level_ids = torch.cat([torch.full((len(grid),), i, device=centers.device) for i, grid in enumerate(grids)])
        scores = torch.cat(logits, dim=1)
        regression = torch.cat(regressions, dim=1)
        if self.training:
            if targets is None:
                raise ValueError("Oriented training requires targets")
            classification_loss = scores.sum() * 0
            geometry_loss = regression.sum() * 0
            for index, target in enumerate(targets):
                source_height, source_width = original_sizes[index]
                height, width = batch.image_sizes[index]
                scale = np.float32([width / source_width, height / source_height])
                boxes = target["oriented_boxes"]
                resized = [oriented_box(obb_corners(box) * scale) for box in boxes.detach().cpu().tolist()]
                boxes = centers.new_tensor(resized).reshape(-1, 5)
                valid = (centers[:, 0] < width) & (centers[:, 1] < height)
                classes = torch.zeros_like(scores[index])
                assigned = torch.full((len(centers),), -1, device=centers.device, dtype=torch.long)
                closest = torch.full((len(centers),), float("inf"), device=centers.device)
                for box_index, box in enumerate(boxes):
                    scale_size = torch.sqrt(box[2] * box[3])
                    reference_sizes = torch.stack([s[0].mean() * 4 for s in strides])
                    level = torch.abs(torch.log2(scale_size / reference_sizes)).argmin()
                    delta = centers - box[:2]
                    radians = torch.deg2rad(box[4])
                    local_x = delta[:, 0] * radians.cos() + delta[:, 1] * radians.sin()
                    local_y = -delta[:, 0] * radians.sin() + delta[:, 1] * radians.cos()
                    distance = (delta / stride_xy).square().sum(dim=1)
                    inside = (local_x.abs() <= box[2] / 2) & (local_y.abs() <= box[3] / 2)
                    candidates = valid & (level_ids == level) & inside & (distance <= 2.25)
                    if not candidates.any():
                        eligible = valid & (level_ids == level)
                        candidates[distance.masked_fill(~eligible, float("inf")).argmin()] = True
                    selected = candidates & (distance < closest)
                    assigned[selected] = box_index
                    closest[selected] = distance[selected]
                positive = assigned >= 0
                classes[positive] = 1
                count = positive.sum().clamp(min=1)
                classification_loss = classification_loss + sigmoid_focal_loss(
                    scores[index][valid], classes[valid], reduction="sum",
                ) / count
                if positive.any():
                    selected_boxes = boxes[assigned[positive]]
                    angles = torch.deg2rad(selected_boxes[:, 4]) * 2
                    expected = torch.cat((
                        (selected_boxes[:, :2] - centers[positive]) / stride_xy[positive],
                        (selected_boxes[:, 2:4] / stride_xy[positive]).log(),
                        torch.stack((angles.sin(), angles.cos()), dim=1),
                    ), dim=1)
                    geometry_loss = geometry_loss + F.smooth_l1_loss(regression[index][positive], expected)
            return {"loss_classifier": classification_loss / len(images),
                    "loss_oriented_geometry": geometry_loss / len(images)}
        predictions = []
        for index, (height, width) in enumerate(batch.image_sizes):
            confidence = scores[index].sigmoid()
            valid = (centers[:, 0] < width) & (centers[:, 1] < height) & (confidence >= 0.05)
            indices = torch.where(valid)[0]
            indices = indices[confidence[indices].argsort(descending=True)[:500]]
            parameters = regression[index][indices]
            positions = centers[indices] + parameters[:, :2] * stride_xy[indices]
            sizes = parameters[:, 2:4].clamp(-2, 6).exp() * stride_xy[indices]
            angles = torch.rad2deg(torch.atan2(parameters[:, 4], parameters[:, 5])) / 2
            # Canonicalize equivalent rectangles before rotated suppression.
            swap = sizes[:, 0] > sizes[:, 1]
            sizes = torch.where(swap[:, None], sizes.flip(1), sizes)
            angles = (angles + swap * 90 + 90) % 180 - 90
            boxes = torch.cat((positions, sizes, angles[:, None]), dim=1)
            confidence = confidence[indices]
            keep = rotated_nms(boxes, confidence)
            boxes, confidence = boxes[keep], confidence[keep]
            source_height, source_width = original_sizes[index]
            scale = np.float32([source_width / width, source_height / height])
            polygons = [obb_corners(box) * scale for box in boxes.cpu().tolist()]
            points = centers.new_tensor(np.asarray(polygons, dtype=np.float32)).reshape(-1, 4, 2)
            original_obbs = centers.new_tensor([oriented_box(p) for p in polygons]).reshape(-1, 5)
            lower, upper = points.amin(dim=1), points.amax(dim=1)
            axis_boxes = torch.cat((lower, upper), dim=1)
            axis_boxes[:, 0::2].clamp_(0, source_width)
            axis_boxes[:, 1::2].clamp_(0, source_height)
            nonempty = (axis_boxes[:, 2] > axis_boxes[:, 0]) & (axis_boxes[:, 3] > axis_boxes[:, 1])
            predictions.append({"boxes": axis_boxes[nonempty], "oriented_boxes": original_obbs[nonempty],
                                "scores": confidence[nonempty], "polygons": points[nonempty],
                                "labels": torch.ones(int(nonempty.sum()), dtype=torch.int64, device=centers.device)})
        return predictions


class OrientedCardDetector(CardDetector):
    variant = DetectorVariant.ORIENTED

    def build_detector(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> nn.Module:
        return OrientedDenseDetector(backbone, min_size, max_size)
