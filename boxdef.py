from dataclasses import dataclass, field
from enum import IntEnum
from turtle import width
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


class EffectType(IntEnum):
    SET_VELOCITY = 0  # sets given body variable
    APPLY_FORCE = 1
    DONE = 2  # TODO: sets self.done in BoxEnv to True
    RESET = 3  # TODO: calls BoxEnv().reset()


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
    valid: bool = False
    points: List[b2Vec2] = field(default_factory=list)
    vertices: List[b2Vec2] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0
    delta_angle: float = 0.0
    moved: bool = False
    rotated: bool = False

    # TODO: toggle color
    color: tuple = field(default=boxcolors.STATIC_OBSTACLE)

    dict_ix: int = 0

    params: Dict = field(default_factory=dict)
    # indicates which param to currently change
    params_ix: int = 0

    effect: Dict = field(default_factory=dict)
    effect_ix: int = 0

    dicts: List = field(default_factory=list)

    float_inc: float = 0.1