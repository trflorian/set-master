import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.config import DetectorVariant
from set_master.training.models.box import CardDetector


class QuadrilateralRoIHeads(nn.Module):
    """Reuse box-region features to regress four ordered corners directly."""

    def __init__(self, box_heads: RoIHeads) -> None:
        super().__init__()
        self.box_heads = box_heads
        self.regressor = nn.Linear(box_heads.box_predictor.cls_score.in_features, 8)
        nn.init.normal_(self.regressor.weight, std=0.001)
        # Start from the corners of the proposal rather than a collapsed polygon.
        with torch.no_grad():
            self.regressor.bias.copy_(torch.tensor([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5]))

    def forward(
        self, features: dict[str, Tensor], proposals: list[Tensor],
        image_shapes: list[tuple[int, int]], targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        results, losses = self.box_heads(features, proposals, image_shapes, targets)
        if self.training:
            if targets is None:
                raise ValueError('Quadrilateral training requires corner targets')
            sampled, matched, labels, _ = self.box_heads.select_training_samples(proposals, targets)
            positive = [label > 0 for label in labels]
            regions = [boxes[keep] for boxes, keep in zip(sampled, positive)]
            points = torch.cat([target['keypoints'][indices[keep]]
                                for target, indices, keep in zip(targets, matched, positive)])
        else:
            regions = [result['boxes'] for result in results]
        pooled = self.box_heads.box_roi_pool(features, regions, image_shapes)
        offsets = self.regressor(self.box_heads.box_head(pooled)).reshape(-1, 4, 2)
        boxes = torch.cat(regions)
        size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1)
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if self.training:
            expected = (points[..., :2] - center[:, None]) / size[:, None]
            visible = points[..., 2] > 0
            losses['loss_quadrilateral'] = (
                F.smooth_l1_loss(offsets[visible], expected[visible], beta=1 / 9)
                if visible.any() else offsets.sum() * 0
            )
        else:
            corners = offsets * size[:, None] + center[:, None]
            for result, polygon in zip(results, corners.split([len(region) for region in regions])):
                # Reject crossed, degenerate, or nonfinite polygons before evaluation/rectification.
                edges = polygon.roll(-1, dims=1) - polygon
                following = edges.roll(-1, dims=1)
                cross = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
                valid = torch.isfinite(polygon).all(dim=(1, 2)) & (cross > 1e-4).all(dim=1)
                for name, value in result.items():
                    result[name] = value[valid]
                polygon = polygon[valid]
                result['keypoints'] = torch.cat((polygon, torch.ones_like(polygon[..., :1])), dim=2)
        return results, losses


class QuadrilateralCardDetector(CardDetector):
    variant = DetectorVariant.QUADRILATERAL

    def build_detector(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> nn.Module:
        detector = super().build_detector(backbone, min_size, max_size)
        if not isinstance(detector, FasterRCNN):
            raise TypeError('Quadrilateral regression requires the Faster R-CNN box detector')
        detector.roi_heads = QuadrilateralRoIHeads(detector.roi_heads)
        return detector
