import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from set_master.training.data import CardDataModule
from set_master.training.model import CardDetector, DEFAULT_REPO, DEFAULT_WEIGHTS
from set_master.paths import RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen DINOv3 ConvNeXt card detector")
    parser.add_argument("--train-data", type=Path, required=True, help="COCO dataset root containing annotations.json")
    parser.add_argument("--val-data", type=Path, help="Separate validation dataset; otherwise use a seeded image split")
    parser.add_argument("--val-fraction", type=float, default=0.2)
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
    if args.batch_size < 1 or args.num_workers < 0 or not 0 < args.min_size <= args.max_size:
        parser.error("Require batch-size > 0, num-workers >= 0 and 0 < min-size <= max-size")

    pl.seed_everything(args.seed, workers=True)
    data = CardDataModule(
        args.train_data, args.val_data, args.batch_size, args.num_workers, args.val_fraction, args.seed,
    )
    data.setup("fit")
    model = CardDetector(
        weights_path=str(args.weights.expanduser().resolve()), repo_path=str(args.dinov3_repo.expanduser().resolve()),
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        min_size=args.min_size, max_size=args.max_size,
    )
    logger = CSVLogger(str(args.output_dir), name="")
    logger.log_hyperparams(model.hparams)
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps(data.split_manifest, indent=2) + "\n")
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", monitor="val/map", mode="max",
        save_top_k=1, save_last=True, filename="epoch-{epoch:02d}", auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1, precision=args.precision,
        logger=logger, callbacks=[checkpoint], gradient_clip_val=1.0,
        log_every_n_steps=1,
        fast_dev_run=args.fast_dev_run, default_root_dir=str(args.output_dir),
    )
    trainer.fit(model, datamodule=data, ckpt_path=args.resume)
    if checkpoint.best_model_path:
        print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
