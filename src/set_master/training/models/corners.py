import torch
from torch import Tensor, nn
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.models.box import CardDetector, DEFAULT_REPO, DEFAULT_WEIGHTS
from set_master.training.config import DetectorVariant
from set_master.training.data import DetectionBatch


class CornerCardDetector(CardDetector):
    variant = DetectorVariant.CORNERS

    def __init__(
        self, weights_path: str = str(DEFAULT_WEIGHTS), repo_path: str = str(DEFAULT_REPO),
        learning_rate: float = 1e-3, weight_decay: float = 1e-4,
        min_size: int = 640, max_size: int = 1280,
        initialize_pretrained: bool = True, variant: str | None = None,
        keypoint_channels: tuple[int, ...] = (128, 128, 128, 128),
    ) -> None:
        if not keypoint_channels or any(channels < 1 for channels in keypoint_channels):
            raise ValueError('Keypoint channels must be a nonempty sequence of positive integers')
        self.keypoint_channels = keypoint_channels
        super().__init__(
            weights_path=weights_path, repo_path=repo_path,
            learning_rate=learning_rate, weight_decay=weight_decay,
            min_size=min_size, max_size=max_size,
            initialize_pretrained=initialize_pretrained, variant=variant,
        )
        self.save_hyperparameters({'keypoint_channels': keypoint_channels})

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
            backbone, num_classes=2, min_size=min_size, max_size=max_size,
            keypoint_head=KeypointRCNNHeads(backbone.out_channels, self.keypoint_channels),
            keypoint_predictor=KeypointRCNNPredictor(self.keypoint_channels[-1], 4),
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            ),
            box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2),
            keypoint_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2),
            box_score_thresh=0.05,
        )
