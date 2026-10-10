from dataclasses import dataclass
from enum import Enum, StrEnum


class BackgroundType(StrEnum):
    GRAY = "gray"
    REAL_SCENE = "real_scene"
    MIXED = "mixed"


class BackgroundSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


class Shape(Enum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    DIAMOND = "diamond"
    CAPSULE = "capsule"
    SQUIGGLE = "squiggle"


class Color(Enum):
    RED = (85, 100, 170)
    GREEN = (88, 155, 49)
    PURPLE = (146, 119, 119)


class Shading(Enum):
    SOLID = "solid"
    SHADED = "shaded"
    OPEN = "open"


class ShadingPattern(Enum):
    STRIPED_HORIZONTAL = "striped_horizontal"
    STRIPED_VERTICAL = "striped_vertical"
    DOTTED = "dotted"


class Number(Enum):
    ONE = 1
    TWO = 2
    THREE = 3


@dataclass
class Card:
    shape: Shape
    color: Color
    shading: Shading
    number: Number
