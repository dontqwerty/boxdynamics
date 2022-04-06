from enum import Enum, IntEnum


class BodyType(IntEnum):
    AGENT = 0
    STATIC_OBSTACLE = 1
    MOVING_OBSTACLE = 2
    STATIC_ZONE = 3
    MOVING_ZONE = 4
    BORDER = 5
    DEFAULT = 6


class BodyShape(IntEnum):
    BOX = 0
    CIRCLE = 1
