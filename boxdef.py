from enum import Enum


class BodyType(Enum):
    AGENT = 0
    STATIC_OBSTACLE = 1
    MOVING_OBSTACLE = 2
    STATIC_ZONE = 3
    MOVING_ZONE = 4
    BORDER = 5
    DEFAULT = 6


class BodyShape(Enum):
    BOX = 0
    CIRCLE = 1
