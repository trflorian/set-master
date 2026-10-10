import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from set_master.training.data import CardDataModule, CocoCardDataset, DetectionBatch
from set_master.training.evaluation import evaluate_detector, prediction_overlay
from set_master.training.models.box import CardDetector


class EpochEvaluation(pl.Callback):
    def __init__(
        self, run_dir: Path, image_logger: WandbLogger,
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

    def on_fit_start(self, trainer: pl.Trainer, pl_module: CardDetector) -> None:
        if not trainer.is_global_zero:
            return
        data = trainer.datamodule
        if not isinstance(data, CardDataModule):
            raise TypeError("Data previews require CardDataModule")
        directory = self.run_dir / "data_previews"
        directory.mkdir(parents=True, exist_ok=True)
        for name, source in (("training", data.train_dataset), ("validation", data.val_dataset),
                             ("test", data.test_dataset)):
            if source is None:
                continue
            # Preview sampling must not consume the training augmentation RNG.
            dataset = deepcopy(source)
            count = min(6, len(dataset))
            grid = np.full((((count + 2) // 3) * 320, 3 * 480, 3), 235, dtype=np.uint8)
            for slot, index in enumerate(np.linspace(0, len(dataset) - 1, count, dtype=int)):
                image, target = dataset[int(index)]
                empty = {"boxes": torch.empty((0, 4)), "scores": torch.empty(0),
                         "keypoints": torch.empty((0, 4, 3))}
                if "oriented_boxes" in target:
                    empty["oriented_boxes"] = torch.empty((0, 5))
                preview = prediction_overlay(image, target, empty, 0)
                height, width = preview.shape[:2]
                scale = min(480 / width, 290 / height)
                preview = cv2.resize(preview, (round(width * scale), round(height * scale)))
                x, y = slot % 3 * 480, slot // 3 * 320
                grid[y + 30:y + 30 + preview.shape[0], x:x + preview.shape[1]] = preview
                cv2.putText(grid, f"{name} / image {target['image_id'].item()}", (x + 8, y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            if not cv2.imwrite(str(directory / f"{name}.jpg"), grid):
                raise OSError(f"Could not write {name} preview")
            self.image_logger.log_image(key=f"data/{name}", images=[cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)],
                                        step=0)
        pl_module.print(f"Data preview grids: {directory}")

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
            self.image_logger.log_image(
                key=f"validation/image_{image_id:06d}", images=[cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)],
                step=trainer.global_step,
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
                self.image_logger.log_image(
                    key=f"real/image_{image_id:06d}", images=[cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)],
                    step=trainer.global_step,
                )
        record = {"epoch": trainer.current_epoch + 1, "global_step": trainer.global_step, "metrics": metrics}
        (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        logged = {name: value for name, value in metrics.items() if name.startswith("real/")}
        for logger in trainer.loggers:
            logger.log_metrics({**logged, "epoch": trainer.current_epoch}, step=trainer.global_step)
            logger.save()
        summary = " | ".join(f"{name}: {metrics[name]:.3f}" for name in
                             ("val/map", "val/geometry_map", "real/map", "real/precision", "real/recall")
                             if name in metrics)
        pl_module.print(f"Epoch {trainer.current_epoch + 1} | {summary} | {directory}")
