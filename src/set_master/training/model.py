from collections import OrderedDict
import importlib
from pathlib import Path
import sys

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

from set_master.training.data import DetectionBatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
DEFAULT_REPO = PROJECT_ROOT / "src/dinov3"


class FrozenDinoFPN(nn.Module):
    out_channels = 256

    def __init__(self, weights_path: str, repo_path: str, initialize_pretrained: bool = True) -> None:
        super().__init__()
        if not Path(repo_path, "hubconf.py").is_file():
            raise FileNotFoundError(f"DINOv3 source checkout not found: {repo_path}")
        if initialize_pretrained and not Path(weights_path).is_file():
            raise FileNotFoundError(f"DINOv3 pretrained weights not found: {weights_path}")
        # Import just the official backbone entry point: hubconf also imports optional
        # segmentation/depth packages that this detector does not use.
        sys.path.insert(0, repo_path)
        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
            self.encoder = backbones.dinov3_convnext_tiny(pretrained=False, patch_size=None)
        finally:
            sys.path.remove(repo_path)
        if initialize_pretrained:
            self.encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], self.out_channels)

    def train(self, mode: bool = True) -> "FrozenDinoFPN":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(images, n=4, reshape=True, norm=True)
        return self.fpn(OrderedDict((str(i), feature) for i, feature in enumerate(features)))


class CardDetector(pl.LightningModule):
    """Frozen DINOv3 features with trainable FPN, proposals, and box prediction."""

    def __init__(
        self, weights_path: str = str(DEFAULT_WEIGHTS), repo_path: str = str(DEFAULT_REPO),
        learning_rate: float = 1e-3, weight_decay: float = 1e-4,
        min_size: int = 640, max_size: int = 1280,
        initialize_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        backbone = FrozenDinoFPN(weights_path, repo_path, initialize_pretrained)
        self.detector = FasterRCNN(
            backbone, num_classes=2, min_size=min_size, max_size=max_size,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            ),
            box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2),
            box_score_thresh=0.05,
        )
        self.box_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", compute_on_cpu=True)

    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        """Predict boxes in original image coordinates; call in evaluation mode."""
        if self.training:
            raise RuntimeError("Call model.eval() before prediction")
        return self.detector(images)

    def training_step(self, batch: DetectionBatch, batch_idx: int) -> Tensor:
        images, targets = batch
        losses = self.detector(images, targets)
        loss = torch.stack(list(losses.values())).sum()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        for name, value in losses.items():
            self.log(f"train/{name}", value, on_step=False, on_epoch=True, batch_size=len(images))
        return loss

    def validation_step(self, batch: DetectionBatch, batch_idx: int) -> None:
        images, targets = batch
        self.box_map.update(self(images), targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.box_map.compute()
        for name in ("map", "map_50", "map_75", "mar_100"):
            self.log(f"val/{name}", metrics[name], prog_bar=name == "map", sync_dist=True)
        self.box_map.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
        )
