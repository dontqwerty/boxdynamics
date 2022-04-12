from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

from Box2D import b2Vec2

import boxcolors


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
    SET_VELOCITY = 0  # sets given body variable
    APPLY_FORCE = 1
    DONE = 2  # sets self.done in BoxEnv to True
    RESET = 3  # calls BoxEnv().reset()


class UIMode(IntEnum):
    NONE = 0
    RESIZE = 1
    ROTATE = 2
    MOVE = 8
    SIMULATION = 3
    INPUT_SAVE = 4
    INPUT_LOAD = 5
    QUIT_CONFIRMATION = 6
    USE_CONFIRMATION = 7


@dataclass
class ScreenLayout:
    width: int = 1000  # pixels
    height: int = 800
    size: b2Vec2 = b2Vec2(width, height)
    simulation_xshift: int = width / 4
    simulation_yshift: int = 0
    simulation_pos: b2Vec2 = b2Vec2(simulation_xshift, simulation_yshift)
    simulation_size: b2Vec2 = b2Vec2(
        width - simulation_xshift, height - simulation_yshift)
    board_pos: b2Vec2 = b2Vec2(0, simulation_size.y)
    board_size: b2Vec2 = b2Vec2(width, height) * 0
    popup_size: b2Vec2 = b2Vec2(150, 60)
    popup_pos: b2Vec2 = (b2Vec2(width, height) - popup_size / 2) / 2
    small_font: int = 14
    normal_font: int = 20
    big_font: int = 32


@dataclass
class DesignData:
    selected: bool = False  # TODO: selectable

    # center and radius for circles
    points: List[b2Vec2] = field(default_factory=list)
    # all four vertices for rectangles
    vertices: List[b2Vec2] = field(default_factory=list)
    # body rotation
    rotated: bool = False
    init_vertices: List[b2Vec2] = field(default_factory=list)
    initial_angle: float = 0.0
    delta_angle: float = 0.0

    shape: IntEnum = BodyShape.BOX  # TODO: circles
    # TODO: toggle color
    color: tuple = field(default=boxcolors.STATIC_OBSTACLE)

    params: Dict = field(default_factory=dict)
    # indicates which param to currently change
    params_ix: int = 0
    float_inc: float = 0.1

    effect: Dict = field(default_factory=dict)
