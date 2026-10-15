from pathlib import Path

import torch

from set_master.training.config import DetectorConfig, DetectorVariant
from set_master.training.models.box import DEFAULT_WEIGHTS, CardDetector
from set_master.training.models.corners import CornerCardDetector
from set_master.training.models.oriented import OrientedCardDetector
from set_master.training.models.quadrilateral import QuadrilateralCardDetector
from set_master.training.models.segmentation import InstanceCardDetector, SemanticCardDetector
from set_master.training.models.semantic_refined import RefinedSemanticCardDetector


def detector_class(variant: DetectorVariant) -> type[CardDetector]:
    return {
        DetectorVariant.BOX: CardDetector,
        DetectorVariant.ORIENTED: OrientedCardDetector,
        DetectorVariant.CORNERS: CornerCardDetector,
        DetectorVariant.QUADRILATERAL: QuadrilateralCardDetector,
        DetectorVariant.SEMANTIC: SemanticCardDetector,
        DetectorVariant.SEMANTIC_REFINED: RefinedSemanticCardDetector,
        DetectorVariant.INSTANCE: InstanceCardDetector,
    }[variant]


def create_detector(
    config: DetectorConfig,
    weights_path: Path = DEFAULT_WEIGHTS,
    initialize_pretrained: bool = True,
) -> CardDetector:
    return detector_class(config.variant)(
        weights_path=str(weights_path),
        initialize_pretrained=initialize_pretrained,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        min_size=config.min_size,
        max_size=config.max_size,
    )


def load_detector(checkpoint_path: Path) -> CardDetector:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    variant = DetectorVariant(hyperparameters.get("variant", "box"))
    # Older corner checkpoints used Torchvision's eight-layer, 512-channel head.
    legacy_corner_head = variant == DetectorVariant.CORNERS and "keypoint_channels" not in hyperparameters
    del checkpoint
    if legacy_corner_head:
        return CornerCardDetector.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            initialize_pretrained=False,
            keypoint_channels=(512,) * 8,
        ).eval()
    return (
        detector_class(variant)
        .load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            initialize_pretrained=False,
        )
        .eval()
    )
