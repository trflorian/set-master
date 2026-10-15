import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from set_master.training.backbone import FrozenDinoFPN
from set_master.training.data import DetectionBatch
from set_master.training.config import DetectorVariant
from set_master.training.geometry_metrics import GeometryMetrics
from set_master.training.segmentation_metrics import SemanticMetrics, update_mask_ap
from set_master.paths import DINOV3_REPO_DIR, MODELS_DIR

DEFAULT_WEIGHTS = MODELS_DIR / "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
DEFAULT_REPO = DINOV3_REPO_DIR


class CardDetector(pl.LightningModule):
    """Frozen DINOv3 features with trainable task heads and validation metrics."""

    variant = DetectorVariant.BOX

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
    ) -> None:
        super().__init__()
        if variant is not None and variant != self.variant.value:
            raise ValueError(f"Checkpoint variant {variant} does not match {self.variant}")
        self.save_hyperparameters()
        self.hparams.variant = self.variant.value
        backbone = FrozenDinoFPN(weights_path, repo_path, initialize_pretrained)
        self.detector = self.build_detector(backbone, min_size, max_size)
        self.box_map = (
            None
            if self.variant.is_semantic
            else MeanAveragePrecision(box_format="xyxy", iou_type="bbox", compute_on_cpu=True)
        )
        self.geometry_metrics = (
            GeometryMetrics(self.variant)
            if self.variant in (DetectorVariant.ORIENTED, DetectorVariant.CORNERS, DetectorVariant.QUADRILATERAL)
            else None
        )
        self.semantic_metrics = SemanticMetrics() if self.variant.is_semantic else None
        self.mask_map = MeanAveragePrecision(iou_type="segm") if self.variant == DetectorVariant.INSTANCE else None

    def build_detector(self, backbone: FrozenDinoFPN, min_size: int, max_size: int) -> nn.Module:
        return FasterRCNN(
            backbone,
            num_classes=2,
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            ),
            box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2),
            box_score_thresh=0.05,
        )

    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        """Predict task outputs in original image coordinates; call in evaluation mode."""
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

    def validation_step(self, batch: DetectionBatch, batch_idx: int) -> list[dict[str, Tensor]]:
        images, targets = batch
        predictions = self(images)
        if self.box_map is not None:
            self.box_map.update(predictions, targets)
        if self.semantic_metrics is not None:
            self.semantic_metrics.update(predictions, targets)
        if self.mask_map is not None:
            update_mask_ap(self.mask_map, predictions, targets)
        if self.geometry_metrics is not None:
            self.geometry_metrics.update(predictions, targets)
        return predictions

    def on_validation_epoch_end(self) -> None:
        self._log_detection_metrics("val")

    def test_step(self, batch: DetectionBatch, batch_idx: int) -> list[dict[str, Tensor]]:
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._log_detection_metrics("test")

    def _log_detection_metrics(self, prefix: str) -> None:
        for metric, label in ((self.box_map, ""), (self.mask_map, "mask_")):
            if metric is not None:
                metrics = metric.compute()
                for name in ("map", "map_50", "map_75", "mar_100"):
                    self.log(f"{prefix}/{label}{name}", metrics[name], prog_bar=name == "map", sync_dist=True)
                metric.reset()
        if self.semantic_metrics is not None:
            for name, value in self.semantic_metrics.compute().items():
                self.log(f"{prefix}/{name}", value, prog_bar=name == "foreground_iou")
            self.semantic_metrics.reset()
        if self.geometry_metrics is not None:
            for name, value in self.geometry_metrics.compute().items():
                self.log(f"{prefix}/{name}", value, prog_bar=name == "geometry_map")
            self.geometry_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
