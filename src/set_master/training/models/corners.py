import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    keypoints_to_heatmap,
    keypointrcnn_inference,
)

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.models.box import CardDetector, DEFAULT_REPO, DEFAULT_WEIGHTS
from set_master.training.config import DetectorVariant
from set_master.training.data import DetectionBatch


def cyclic_keypoint_loss(
    logits: Tensor,
    proposals: list[Tensor],
    targets: list[Tensor],
    matched: list[Tensor],
) -> Tensor:
    """Accept 180-degree symmetry while penalizing sideways corner assignments."""
    count, corners, height, width = logits.shape
    if corners != 4 or height != width:
        raise ValueError("Corner heatmaps must have four channels and equal height and width")
    heatmaps, visibility = [], []
    for boxes, points, indices in zip(proposals, targets, matched):
        heatmap, valid = keypoints_to_heatmap(points[indices], boxes, height)
        heatmaps.append(heatmap)
        visibility.append(valid)
    expected = torch.cat(heatmaps)
    valid = torch.cat(visibility).bool()
    if not valid.any():
        return logits.sum() * 0
    flat_logits = logits.reshape(count * corners, height * width)
    candidates = []
    for shift in (0, 2):
        shifted_targets = expected.roll(shift, dims=1)
        shifted_valid = valid.roll(shift, dims=1)
        losses = F.cross_entropy(flat_logits, shifted_targets.reshape(-1), reduction="none")
        candidates.append((losses.reshape(count, corners) * shifted_valid).sum(dim=1))
    return torch.stack(candidates, dim=1).amin(dim=1).sum() / valid.sum()


def order_and_filter_corners(result: dict[str, Tensor]) -> dict[str, Tensor]:
    """Canonicalize corner order and discard nonfinite or nonconvex detections."""
    points = result["keypoints"]
    xy = points[..., :2]
    centered = xy - xy.mean(dim=1, keepdim=True)
    order = torch.atan2(centered[..., 1], centered[..., 0]).argsort(dim=1)
    # Preserve channel zero's edge identity rather than choosing an image-relative start.
    start = (order == 0).to(torch.int64).argmax(dim=1)
    cyclic = (torch.arange(4, device=xy.device)[None] + start[:, None]) % 4
    order = order.gather(1, cyclic)
    points = points.gather(1, order[..., None].expand(-1, -1, points.shape[-1]))
    xy = points[..., :2]
    edges = xy.roll(-1, dims=1) - xy
    following = edges.roll(-1, dims=1)
    cross = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
    area = (xy[..., 0] * xy.roll(-1, dims=1)[..., 1] - xy[..., 1] * xy.roll(-1, dims=1)[..., 0]).sum(dim=1) / 2
    valid = torch.isfinite(points).all(dim=(1, 2)) & (cross > 1e-4).all(dim=1) & (area >= 1)
    ordered = dict(result)
    ordered["keypoints"] = points
    ordered["keypoints_scores"] = result["keypoints_scores"].gather(1, order)
    return {name: value[valid] for name, value in ordered.items()}


class CornerRoIHeads(RoIHeads):
    """Box detection with orientation-aware corner supervision and valid polygons."""

    def forward(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        if self.training:
            if targets is None:
                raise ValueError("Corner training requires keypoint targets")
            proposals, matched, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        results: list[dict[str, Tensor]] = []
        losses: dict[str, Tensor] = {}
        if self.training:
            classifier_loss, box_loss = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": classifier_loss, "loss_box_reg": box_loss}
            positive = [label > 0 for label in labels]
            regions = [boxes[keep] for boxes, keep in zip(proposals, positive)]
            indices = [index[keep] for index, keep in zip(matched, positive)]
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            results = [dict(boxes=box, scores=score, labels=label) for box, score, label in zip(boxes, scores, labels)]
            regions = boxes
        logits = self.keypoint_predictor(self.keypoint_head(self.keypoint_roi_pool(features, regions, image_shapes)))
        if self.training:
            losses["loss_keypoint"] = cyclic_keypoint_loss(
                logits,
                regions,
                [target["keypoints"] for target in targets],
                indices,
            )
        else:
            points, scores = keypointrcnn_inference(logits, regions)
            for result, keypoints, keypoint_scores in zip(results, points, scores):
                result["keypoints"] = keypoints
                result["keypoints_scores"] = keypoint_scores
            results = [order_and_filter_corners(result) for result in results]
        return results, losses


class CornerCardDetector(CardDetector):
    variant = DetectorVariant.CORNERS

    def __init__(
        self,
        weights_path: str = str(DEFAULT_WEIGHTS),
        repo_path: str = str(DEFAULT_REPO),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        min_size: int = 640,
        max_size: int = 1280,
        initialize_pretrained: bool = True,
        variant: str | None = None,
        keypoint_channels: tuple[int, ...] = (128, 128, 128, 128),
    ) -> None:
        if not keypoint_channels or any(channels < 1 for channels in keypoint_channels):
            raise ValueError("Keypoint channels must be a nonempty sequence of positive integers")
        self.keypoint_channels = keypoint_channels
        super().__init__(
            weights_path=weights_path,
            repo_path=repo_path,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            min_size=min_size,
            max_size=max_size,
            initialize_pretrained=initialize_pretrained,
            variant=variant,
        )
        self.save_hyperparameters({"keypoint_channels": keypoint_channels})

    def training_step(self, batch: DetectionBatch, batch_idx: int) -> Tensor:
        images, targets = batch
        expanded = []
        for image, target in zip(images, targets):
            # Keypoint heatmaps need virtual sharp corners to lie inside training ROIs.
            # Keep the dataset's original boxes for the common evaluation metrics.
            copy = dict(target)
            boxes = target["boxes"].clone()
            points = target["keypoints"][..., :2]
            boxes[:, :2] = torch.minimum(boxes[:, :2], points.amin(dim=1)).clamp(min=0)
            boxes[:, 2:] = torch.maximum(boxes[:, 2:], points.amax(dim=1))
            boxes[:, 0::2].clamp_(0, image.shape[-1])
            boxes[:, 1::2].clamp_(0, image.shape[-2])
            copy["boxes"] = boxes
            expanded.append(copy)
        return super().training_step((images, expanded), batch_idx)

    def build_detector(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> nn.Module:
        detector = KeypointRCNN(
            backbone,
            num_classes=2,
            min_size=min_size,
            max_size=max_size,
            keypoint_head=KeypointRCNNHeads(backbone.out_channels, self.keypoint_channels),
            keypoint_predictor=KeypointRCNNPredictor(self.keypoint_channels[-1], 4),
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            ),
            box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2),
            keypoint_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2),
            box_score_thresh=0.05,
        )
        heads = detector.roi_heads
        detector.roi_heads = CornerRoIHeads(
            heads.box_roi_pool,
            heads.box_head,
            heads.box_predictor,
            heads.proposal_matcher.high_threshold,
            heads.proposal_matcher.low_threshold,
            heads.fg_bg_sampler.batch_size_per_image,
            heads.fg_bg_sampler.positive_fraction,
            heads.box_coder.weights,
            heads.score_thresh,
            heads.nms_thresh,
            heads.detections_per_img,
            keypoint_roi_pool=heads.keypoint_roi_pool,
            keypoint_head=heads.keypoint_head,
            keypoint_predictor=heads.keypoint_predictor,
        )
        return detector
