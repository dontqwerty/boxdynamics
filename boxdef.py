from enum import IntEnum


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


class EffectType(IntEnum):
    SET_VELOCITY = 0 # sets given body variable
    APPLY_FORCE = 1
    DONE = 2 # sets self.done in BoxEnv to True
    RESET = 3 # calls BoxEnv().reset()