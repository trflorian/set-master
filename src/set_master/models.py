from dataclasses import dataclass
from enum import Enum


class Shape(Enum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    SQUIGGLE = "squiggle"


class Color(Enum):
    RED = (85, 100, 170)
    GREEN = (88, 155, 49)
    PURPLE = (146, 119, 119)


class Shading(Enum):
    SOLID = "solid"
    STRIPED = "striped"
    OPEN = "open"


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
