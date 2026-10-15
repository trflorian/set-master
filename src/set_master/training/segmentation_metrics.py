import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class SemanticMetrics(Metric):
    """Pixel counts aggregated across images, including background-only scenes."""

    counts: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("counts", default=torch.zeros(3, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, predictions: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]) -> None:
        for prediction, target in zip(predictions, targets):
            foreground = prediction["semantic_mask"][0] >= 0.5
            truth = target["masks"].cpu().bool().any(0).to(foreground.device)
            counts = torch.stack([(foreground & truth).sum(), (foreground & ~truth).sum(), (~foreground & truth).sum()])
            self.counts += counts.to(self.counts)

    def compute(self) -> dict[str, Tensor]:
        tp, fp, fn = self.counts.unbind()
        return {
            "foreground_iou": torch.where(tp + fp + fn > 0, tp / (tp + fp + fn), 1.0),
            "dice": torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 1.0),
            "pixel_precision": torch.where(tp + fp > 0, tp / (tp + fp), 1.0),
            "pixel_recall": torch.where(tp + fn > 0, tp / (tp + fn), 1.0),
        }


def update_mask_ap(
    metric: MeanAveragePrecision, predictions: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
) -> None:
    metric.update(
        [
            {
                "masks": pred["masks"][:, 0].detach().cpu() >= 0.5,
                "scores": pred["scores"].detach().cpu(),
                "labels": pred["labels"].cpu(),
            }
            for pred in predictions
        ],
        [
            {
                "masks": target["masks"].cpu().bool(),
                "labels": target["labels"].cpu(),
                "iscrowd": target["iscrowd"].cpu(),
            }
            for target in targets
        ],
    )
