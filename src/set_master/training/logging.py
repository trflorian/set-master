from pathlib import Path

import wandb
from lightning.pytorch.loggers import WandbLogger

from set_master.training.config import DetectorVariant


def create_wandb_logger(
    run_dir: Path,
    experiment_name: str,
    variant: DetectorVariant,
    server_url: str,
    project: str,
    entity: str | None = None,
) -> WandbLogger:
    if not wandb.login(host=server_url, prompt=False):
        raise RuntimeError(
            f"Log in to the local W&B server first: uv run --extra training wandb login --host {server_url}"
        )
    # Creating the run explicitly avoids Lightning's deprecated anonymous argument.
    run = wandb.init(
        name=experiment_name,
        project=project,
        entity=entity,
        dir=str(run_dir),
        tags=[variant.value],
        settings=wandb.Settings(base_url=server_url, mode="online"),
    )
    return WandbLogger(experiment=run, save_dir=str(run_dir), log_model=False)
