from enum import StrEnum
from dataclasses import dataclass


class DetectorVariant(StrEnum):
    BOX = "box"
    ORIENTED = "oriented"
    CORNERS = "corners"


@dataclass(frozen=True)
class DetectorConfig:
    variant: DetectorVariant = DetectorVariant.BOX
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_size: int = 640
    max_size: int = 1280
