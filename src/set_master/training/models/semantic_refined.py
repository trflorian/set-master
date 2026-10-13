"""Progressive semantic masks with RGB detail and supervision at every scale."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.config import DetectorVariant
from set_master.training.models.box import CardDetector


class MaskRefinement(nn.Module):
    """Predict a residual from spatial features and the preceding mask."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.correction = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, features: Tensor, coarse: Tensor) -> Tensor:
        # Fine-scale supervision must not change the coarser mask predictor.
        expanded = F.interpolate(coarse.detach(), size=features.shape[-2:],
                                 mode='bilinear', align_corners=False)
        return expanded + self.correction(torch.cat((features, expanded.sigmoid()), dim=1))


def pyramid_mask_losses(logits: list[Tensor], truth: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Supervise soft area targets at each scale, excluding batch padding."""
    bce, dice, consistency = [], [], []
    probabilities, validity = [], []
    for logit in logits:
        size = logit.shape[-2:]
        coverage = F.interpolate(valid, size=size, mode='area')
        # Normalize partially valid cells so padding does not become background supervision.
        target = F.interpolate(truth * valid, size=size, mode='area') / coverage.clamp_min(1e-6)
        local_mean = F.avg_pool2d(target, 7, stride=1, padding=3, count_include_pad=False)
        weight = coverage * (1 + 4 * (local_mean - target).abs())
        pixel_loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        bce.append(((pixel_loss * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3)).clamp_min(1)).mean())
        probability = logit.sigmoid()
        intersection = (probability * target * coverage).sum((1, 2, 3))
        denominator = ((probability + target) * coverage).sum((1, 2, 3))
        dice.append((1 - (2 * intersection + 1) / (denominator + 1)).mean())
        probabilities.append(probability)
        validity.append(coverage)
    for coarse, fine, coarse_valid, fine_valid in zip(
        probabilities, probabilities[1:], validity, validity[1:],
    ):
        reduced = F.interpolate(fine * fine_valid, size=coarse.shape[-2:], mode='area')
        reduced_valid = F.interpolate(fine_valid, size=coarse.shape[-2:], mode='area')
        difference = (reduced / reduced_valid.clamp_min(1e-6) - coarse.detach()).abs()
        consistency.append(((difference * coarse_valid).sum((1, 2, 3)) /
                            coarse_valid.sum((1, 2, 3)).clamp_min(1)).mean())
    return {'loss_mask_bce': torch.stack(bce).mean(),
            'loss_mask_dice': torch.stack(dice).mean(),
            'loss_pyramid_consistency': 0.1 * torch.stack(consistency).mean()}


class RefinedSemanticFPN(nn.Module):
    """Refine FPN masks down to input resolution with a compact RGB branch.

    Inspired by inverse saliency pyramids, without InSPyReNet's attention or
    two-resolution blending. Image size limits still apply before refinement.
    """

    def __init__(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        )
        self.projections = nn.ModuleList(nn.Sequential(
            nn.Conv2d(backbone.out_channels, 128, 1), nn.ReLU(),
        ) for _ in range(4))
        self.coarse_mask = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 1, 1),
        )
        self.fpn_refinements = nn.ModuleList(MaskRefinement(128) for _ in range(3))
        self.rgb_half = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU())
        self.rgb_full = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.half_refinement = MaskRefinement(32)
        self.full_refinement = MaskRefinement(16)

    def forward(self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None
                ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        original_sizes = [image.shape[-2:] for image in images]
        batch, targets = self.transform(images, targets)
        features = [project(feature) for project, feature in
                    zip(self.projections, self.backbone(batch.tensors).values())]
        logits = [self.coarse_mask(features[-1])]
        for refine, feature in zip(self.fpn_refinements, reversed(features[:-1])):
            logits.append(refine(feature, logits[-1]))
        logits.append(self.half_refinement(self.rgb_half(batch.tensors), logits[-1]))
        logits.append(self.full_refinement(self.rgb_full(batch.tensors), logits[-1]))
        if self.training:
            if targets is None:
                raise ValueError('Refined semantic training requires mask targets')
            # Loss arithmetic stays in float32 under mixed precision.
            truth = batch.tensors.new_zeros((len(images), 1, *batch.tensors.shape[-2:]), dtype=torch.float32)
            valid = torch.zeros_like(truth)
            for index, (target, (height, width)) in enumerate(zip(targets, batch.image_sizes)):
                truth[index, 0, :height, :width] = target['masks'].bool().any(0)
                valid[index, 0, :height, :width] = 1
            return pyramid_mask_losses([logit.float() for logit in logits], truth, valid)
        return [{'semantic_mask': F.interpolate(logit[None, :, :height, :width], size=original,
                                                mode='bilinear', align_corners=False)[0].sigmoid()}
                for logit, (height, width), original in zip(logits[-1], batch.image_sizes, original_sizes)]


class RefinedSemanticCardDetector(CardDetector):
    variant = DetectorVariant.SEMANTIC_REFINED

    def build_detector(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> nn.Module:
        return RefinedSemanticFPN(backbone, min_size, max_size)
