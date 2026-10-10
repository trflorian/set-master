import json
from pathlib import Path

import cv2
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor

from set_master.training.data import CocoCardDataset, DetectionBatch
from set_master.training.evaluation import evaluate_detector, prediction_overlay
from set_master.training.models.box import CardDetector


class EpochEvaluation(pl.Callback):
    def __init__(
        self, run_dir: Path, image_logger: TensorBoardLogger,
        test_dataset: CocoCardDataset | None,
        validation_overlay_ids: list[int] | None = None,
        test_overlay_ids: list[int] | None = None,
        score_threshold: float = 0.75, iou_threshold: float = 0.5,
    ) -> None:
        self.run_dir = run_dir
        self.image_logger = image_logger
        self.test_dataset = test_dataset
        self.validation_overlay_ids = validation_overlay_ids
        self.test_overlay_ids = test_overlay_ids
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: CardDetector,
        outputs: list[dict[str, Tensor]], batch: DetectionBatch,
        batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        images, targets = batch
        directory = self.run_dir / "epoch_evaluations" / f"epoch_{trainer.current_epoch + 1:04d}" / "validation"
        for image, target, prediction in zip(images, targets, outputs):
            image_id = int(target["image_id"].item())
            if self.validation_overlay_ids is not None and image_id not in self.validation_overlay_ids:
                continue
            preview = prediction_overlay(image, target, prediction, self.score_threshold)
            directory.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(directory / f"overlay_{image_id:06d}.jpg"), preview):
                raise OSError(f"Could not write validation overlay in {directory}")
            self.image_logger.experiment.add_image(
                f"validation/image_{image_id:06d}", cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                global_step=trainer.global_step, dataformats="HWC",
            )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: CardDetector) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        directory = self.run_dir / "epoch_evaluations" / f"epoch_{trainer.current_epoch + 1:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        metrics = {name: float(value.item()) for name, value in trainer.callback_metrics.items()
                   if name.startswith("val/")}
        if self.test_dataset is not None:
            overlay_ids = (self.test_overlay_ids if self.test_overlay_ids is not None else
                           [info["id"] for info in self.test_dataset.images])
            result = evaluate_detector(
                pl_module, self.test_dataset, directory / "real",
                overlay_image_ids=tuple(overlay_ids), batch_size=1, num_workers=0,
                score_threshold=self.score_threshold, iou_threshold=self.iou_threshold, progress=False,
            )
            metrics.update({f"real/{name}": value for name, value in result.metrics.items()})
            for image_id in overlay_ids:
                preview = cv2.imread(str(directory / "real" / f"overlay_{image_id:06d}.jpg"))
                if preview is None:
                    raise FileNotFoundError(f"Missing real overlay for image {image_id}")
                self.image_logger.experiment.add_image(
                    f"real/image_{image_id:06d}", cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    global_step=trainer.global_step, dataformats="HWC",
                )
        record = {"epoch": trainer.current_epoch + 1, "global_step": trainer.global_step, "metrics": metrics}
        (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        logged = {name: value for name, value in metrics.items() if name.startswith("real/")}
        for logger in trainer.loggers:
            logger.log_metrics({**logged, "epoch": trainer.current_epoch}, step=trainer.global_step)
            logger.save()
        self.image_logger.experiment.flush()
        summary = " | ".join(f"{name}: {metrics[name]:.3f}" for name in
                             ("val/map", "val/geometry_map", "real/map", "real/precision", "real/recall")
                             if name in metrics)
        pl_module.print(f"Epoch {trainer.current_epoch + 1} | {summary} | {directory}")
