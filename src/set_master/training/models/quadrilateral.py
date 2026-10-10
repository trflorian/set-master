import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.config import DetectorVariant
from set_master.training.models.box import CardDetector


def quadrilateral_geometry_loss(offsets: Tensor, expected: Tensor, visible: Tensor) -> Tensor:
    """Match shape independently of which clockwise vertex starts the sequence."""
    if not visible.any():
        return offsets.sum() * 0
    candidates = torch.stack([
        (F.smooth_l1_loss(offsets, expected.roll(shift, dims=1), beta=1 / 9, reduction='none')
         * visible.roll(shift, dims=1)[..., None]).sum(dim=(1, 2))
        for shift in range(4)
    ], dim=1)
    return candidates.amin(dim=1).sum() / (visible.sum() * 2)


def portrait_axis_encoding(points: Tensor) -> Tensor:
    """Encode the axis between short-edge midpoints without a 180-degree seam."""
    axis = (points[:, 2] + points[:, 3] - points[:, 0] - points[:, 1]) / 2
    x, y = axis.unbind(dim=1)
    # cos(2 theta), sin(2 theta): opposite directions share a target, sideways does not.
    return torch.stack((x.square() - y.square(), 2 * x * y), dim=1) / axis.square().sum(dim=1, keepdim=True).clamp(min=1e-8)


def orient_quadrilaterals(points: Tensor, orientation: Tensor) -> Tensor:
    """Choose the short-edge pair while preserving the predicted polygon geometry."""
    candidates = torch.stack((portrait_axis_encoding(points), portrait_axis_encoding(points.roll(-1, dims=1))), dim=1)
    shift = (candidates * orientation[:, None]).sum(dim=2).argmax(dim=1)
    indices = (torch.arange(4, device=points.device)[None] + shift[:, None]) % 4
    return points.gather(1, indices[..., None].expand(-1, -1, 2))


class QuadrilateralRoIHeads(nn.Module):
    """Predict polygon geometry and portrait orientation independently from ROI features."""

    def __init__(self, box_heads: RoIHeads) -> None:
        super().__init__()
        self.box_heads = box_heads
        self.regressor = nn.Linear(box_heads.box_predictor.cls_score.in_features, 8)
        self.orientation_head = nn.Linear(box_heads.box_predictor.cls_score.in_features, 2)
        nn.init.normal_(self.regressor.weight, std=0.001)
        nn.init.normal_(self.orientation_head.weight, std=0.001)
        # Start from the corners of the proposal rather than a collapsed polygon.
        with torch.no_grad():
            self.regressor.bias.copy_(torch.tensor([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5]))
            self.orientation_head.bias.copy_(torch.tensor([-1.0, 0.0]))

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
        region_features = self.box_heads.box_head(pooled)
        offsets = self.regressor(region_features).reshape(-1, 4, 2)
        orientation = self.orientation_head(region_features)
        boxes = torch.cat(regions)
        size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1)
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if self.training:
            expected = (points[..., :2] - center[:, None]) / size[:, None]
            visible = points[..., 2] > 0
            losses['loss_quadrilateral'] = quadrilateral_geometry_loss(offsets, expected, visible)
            axis_targets = portrait_axis_encoding(points[..., :2])
            valid_orientation = visible.all(dim=1) & (axis_targets.square().sum(dim=1) > 0)
            losses['loss_orientation'] = (
                F.mse_loss(orientation[valid_orientation], axis_targets[valid_orientation])
                if valid_orientation.any() else orientation.sum() * 0
            )
        else:
            corners = offsets * size[:, None] + center[:, None]
            counts = [len(region) for region in regions]
            for result, polygon, direction in zip(results, corners.split(counts), orientation.split(counts)):
                # Reject crossed, degenerate, or nonfinite polygons before evaluation/rectification.
                edges = polygon.roll(-1, dims=1) - polygon
                following = edges.roll(-1, dims=1)
                cross = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
                valid = (torch.isfinite(polygon).all(dim=(1, 2)) & (cross > 1e-4).all(dim=1)
                         & torch.isfinite(direction).all(dim=1))
                for name, value in result.items():
                    result[name] = value[valid]
                polygon = orient_quadrilaterals(polygon[valid], direction[valid])
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
