import torch
from torch import Tensor, nn
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.models.box import CardDetector
from set_master.training.config import DetectorVariant
from set_master.training.data import DetectionBatch


class CornerCardDetector(CardDetector):
    variant = DetectorVariant.CORNERS

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
        return KeypointRCNN(
            backbone, num_classes=2, num_keypoints=4, min_size=min_size, max_size=max_size,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            ),
            box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2),
            keypoint_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2),
            box_score_thresh=0.05,
        )
