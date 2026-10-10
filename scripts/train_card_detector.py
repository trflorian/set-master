import argparse
import json
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from set_master.training.data import CardDataModule
from set_master.training.model import DEFAULT_REPO, DEFAULT_WEIGHTS
from set_master.training.config import DetectorConfig, DetectorVariant
from set_master.training.models.factory import create_detector
from set_master.paths import REAL_TESTSET_DIR, RUNS_DIR
from set_master.training.callbacks import EpochEvaluation
from set_master.training.logging import create_wandb_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen DINOv3 ConvNeXt card detector")
    parser.add_argument("--train-data", type=Path, required=True, help="COCO dataset root containing annotations.json")
    parser.add_argument("--variant", choices=list(DetectorVariant), default="box")
    parser.add_argument("--experiment-name", help="Defaults to detector type and timestamp")
    parser.add_argument("--wandb-url", default="http://localhost:8080")
    parser.add_argument("--wandb-project", default="set-master")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--val-data", type=Path, help="Separate validation dataset; otherwise use a seeded image split")
    parser.add_argument("--test-data", type=Path, default=REAL_TESTSET_DIR)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--score-threshold", type=float, default=0.75)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--validation-overlay-ids", type=int, nargs="*", default=None)
    parser.add_argument("--test-overlay-ids", type=int, nargs="*", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--train-images-per-epoch", type=int, default=200)
    parser.add_argument("--val-images-per-epoch", type=int, default=20)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "card_detector")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--precision", default="32-true", choices=("32-true", "16-mixed", "bf16-mixed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fast-dev-run", action="store_true", help="Verify one training and validation batch")
    args = parser.parse_args()
    variant = DetectorVariant(args.variant)
    if args.experiment_name is None:
        args.experiment_name = f"{variant.value}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if not args.experiment_name or Path(args.experiment_name).name != args.experiment_name:
        parser.error("experiment-name must be a directory name")
    experiment_dir = args.output_dir / variant.value / args.experiment_name
    if args.batch_size < 1 or args.num_workers < 0 or not 0 < args.min_size <= args.max_size:
        parser.error("Require batch-size > 0, num-workers >= 0 and 0 < min-size <= max-size")

    pl.seed_everything(args.seed, workers=True)
    data = CardDataModule(
        args.train_data, args.val_data, args.batch_size, args.num_workers, args.val_fraction, args.seed,
        variant=variant, test_root=args.test_data, augment=not args.no_augmentation,
        train_images_per_epoch=args.train_images_per_epoch, val_images_per_epoch=args.val_images_per_epoch,
    )
    data.setup("fit")
    model = create_detector(
        DetectorConfig(variant, args.learning_rate, args.weight_decay, args.min_size, args.max_size),
        weights_path=args.weights.expanduser().resolve(), repo_path=args.dinov3_repo.expanduser().resolve(),
        initialize_pretrained=args.resume is None,
    )
    logger = CSVLogger(str(experiment_dir), name="")
    logger.log_hyperparams(model.hparams)
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = create_wandb_logger(
        run_dir, args.experiment_name, variant,
        args.wandb_url, args.wandb_project, args.wandb_entity,
    )
    tracker.log_hyperparams({**dict(model.hparams), **json.loads(json.dumps(vars(args), default=str))})
    print(f"W&B dashboard: {tracker.experiment.url}")
    (run_dir / "split.json").write_text(json.dumps(data.split_manifest, indent=2) + "\n")
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    monitor = "val/map" if variant == DetectorVariant.BOX else "val/geometry_map"
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", monitor=monitor, mode="max",
        save_top_k=1, save_last=True, filename="epoch-{epoch:02d}", auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(monitor=monitor, mode="max", patience=args.early_stopping_patience,
                                   min_delta=args.early_stopping_min_delta, check_on_train_epoch_end=False, verbose=True)
    epoch_evaluation = EpochEvaluation(
        run_dir, tracker, data.test_dataset,
        validation_overlay_ids=args.validation_overlay_ids, test_overlay_ids=args.test_overlay_ids,
        score_threshold=args.score_threshold, iou_threshold=args.iou_threshold,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1, precision=args.precision,
        logger=[logger, tracker], callbacks=[epoch_evaluation, early_stopping, LearningRateMonitor(logging_interval="epoch"), checkpoint], gradient_clip_val=1.0,
        log_every_n_steps=1,
        fast_dev_run=args.fast_dev_run, default_root_dir=str(experiment_dir),
    )
    trainer.fit(model, datamodule=data, ckpt_path=args.resume)
    if checkpoint.best_model_path:
        print(f"Best checkpoint: {checkpoint.best_model_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
