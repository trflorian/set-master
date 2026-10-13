from enum import StrEnum
from dataclasses import dataclass


class DetectorVariant(StrEnum):
    BOX = "box"
    ORIENTED = "oriented"
    CORNERS = "corners"
    QUADRILATERAL = "quadrilateral"
    SEMANTIC = "semantic"
    SEMANTIC_REFINED = "semantic_refined"
    INSTANCE = "instance"

    @property
    def is_semantic(self) -> bool:
        return self in (self.SEMANTIC, self.SEMANTIC_REFINED)

    @property
    def selection_metric(self) -> str:
        if self.is_semantic:
            return "foreground_iou"
        return {self.BOX: "map", self.INSTANCE: "mask_map"}.get(self, "geometry_map")


@dataclass(frozen=True)
class DetectorConfig:
    variant: DetectorVariant = DetectorVariant.BOX
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_size: int = 640
    max_size: int = 1280
