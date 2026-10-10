from pathlib import Path

import torch

from set_master.paths import DINOV3_REPO_DIR
from set_master.training.config import DetectorConfig, DetectorVariant
from set_master.training.models.box import CardDetector, DEFAULT_WEIGHTS
from set_master.training.models.corners import CornerCardDetector
from set_master.training.models.oriented import OrientedCardDetector


def detector_class(variant: DetectorVariant) -> type[CardDetector]:
    return {
        DetectorVariant.BOX: CardDetector,
        DetectorVariant.ORIENTED: OrientedCardDetector,
        DetectorVariant.CORNERS: CornerCardDetector,
    }[variant]


def create_detector(
    config: DetectorConfig, weights_path: Path = DEFAULT_WEIGHTS,
    repo_path: Path = DINOV3_REPO_DIR, initialize_pretrained: bool = True,
) -> CardDetector:
    return detector_class(config.variant)(
        weights_path=str(weights_path), repo_path=str(repo_path),
        initialize_pretrained=initialize_pretrained,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        min_size=config.min_size, max_size=config.max_size,
    )


def load_detector(checkpoint_path: Path, repo_path: Path = DINOV3_REPO_DIR) -> CardDetector:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    variant = DetectorVariant(checkpoint.get("hyper_parameters", {}).get("variant", "box"))
    del checkpoint
    return detector_class(variant).load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=True,
        repo_path=str(repo_path), initialize_pretrained=False,
    ).eval()
