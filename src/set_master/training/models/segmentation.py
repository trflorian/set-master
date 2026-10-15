import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.config import DetectorVariant
from set_master.training.models.box import CardDetector


class SemanticFPN(nn.Module):
    """Union-of-cards decoder; losses exclude the transform's batch padding."""

    def __init__(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.transform = GeneralizedRCNNTransform(
            min_size,
            max_size,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        self.projections = nn.ModuleList(nn.Conv2d(backbone.out_channels, 128, 1) for _ in range(4))
        self.decoder = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()) for _ in range(4)],
            nn.Conv2d(128, 1, 1),
        )

    def forward(
        self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        original_sizes = [image.shape[-2:] for image in images]
        batch, targets = self.transform(images, targets)
        features = list(self.backbone(batch.tensors).values())
        size = features[0].shape[-2:]
        fused = torch.stack(
            [
                F.interpolate(project(feature), size=size, mode="bilinear", align_corners=False)
                for project, feature in zip(self.projections, features)
            ]
        ).sum(0)
        logits = F.interpolate(self.decoder(fused), size=batch.tensors.shape[-2:], mode="bilinear", align_corners=False)
        if self.training:
            if targets is None:
                raise ValueError("Semantic training requires mask targets")
            bce, dice = [], []
            for logit, target, (height, width) in zip(logits, targets, batch.image_sizes):
                logit = logit[0, :height, :width]
                truth = target["masks"].bool().any(0).to(logit.dtype)
                bce.append(F.binary_cross_entropy_with_logits(logit, truth))
                probability = logit.sigmoid()
                dice.append(1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1))
            return {"loss_mask_bce": torch.stack(bce).mean(), "loss_mask_dice": torch.stack(dice).mean()}
        return [
            {
                "semantic_mask": F.interpolate(
                    logit[None, :, :height, :width], size=original, mode="bilinear", align_corners=False
                )[0].sigmoid()
            }
            for logit, (height, width), original in zip(logits, batch.image_sizes, original_sizes)
        ]


class SemanticCardDetector(CardDetector):
    variant = DetectorVariant.SEMANTIC

    def build_detector(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> nn.Module:
        return SemanticFPN(backbone, min_size, max_size)


class InstanceCardDetector(CardDetector):
    variant = DetectorVariant.INSTANCE

    def build_detector(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> nn.Module:
        return MaskRCNN(
            backbone,
            num_classes=None,
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            ),
            box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2),
            box_predictor=FastRCNNPredictor(1024, 2),
            mask_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], 14, 2),
            mask_head=MaskRCNNHeads(backbone.out_channels, [128] * 4, 1),
            mask_predictor=MaskRCNNPredictor(128, 128, 2),
            box_score_thresh=0.05,
        )
