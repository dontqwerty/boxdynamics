from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import Dict, List

from Box2D import b2Vec2, b2Body

import boxcolors


@unique
class BodyType(IntEnum):
    AGENT = 0
    STATIC_OBSTACLE = 1
    DYNAMIC_OBSTACLE = 2
    KINEMATIC_OBSTACLE = 3
    STATIC_ZONE = 4
    DYNAMIC_ZONE = 5
    KINEMATIC_ZONE = 6
    BORDER = 7
    DEFAULT = 8


@unique
class EffectWhen(IntEnum):
    DURING_CONTACT = 0
    ON_CONTACT = 1
    OFF_CONTACT = 2


@unique
class EffectWho(IntEnum):
    AGENT = 0
    OTHER = 1
    BOTH = 2


@unique
class EffectType(IntEnum):
    NONE = 5  # params: none
    BOUNCE = 10  # params: coeff
    INVERT_VELOCITY = 4  # params: none
    SET_VELOCITY = 0  # sets given body variable # params: angle, mag
    APPLY_FORCE = 1  # params: angle, mag
    SET_LIN_DAMP = 6  # params: coeff
    SET_ANG_DAMP = 7  # params: coeff
    SET_FRICTION = 8  # params: coeff
    SET_MAX_ACTION = 9  # params: coeff
    DONE = 2  # TODO: sets self.done in BoxEnv to True # params: none
    RESET = 3  # TODO: calls BoxEnv().reset() # params: none


@unique
class SetType(IntEnum):
    DEFAULT = 0
    PREVIOUS = 1
    RANDOM = 2


@unique
class ParamGroup(IntEnum):
    PHYSIC = 0
    EFFECT = 1


@unique
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
    SELECT = 9


@dataclass
class BodyData:
    type: IntEnum = BodyType.DEFAULT
    color: tuple = boxcolors.WHITE
    # list of bodies in contact with this body
    contacts: List[b2Body] = field(default_factory=list)
    reward: float = 0  # reward when agents hits the object
    level: int = 0
    effect: Dict = field(default_factory=dict)


@dataclass
class ScreenLayout:
    width: int = 1000  # pixels
    height: int = 800
    simulation_xshift: int = width / 4
    simulation_yshift: int = 0
    border: int = 10
    popup_size: b2Vec2 = b2Vec2(150, 60)
    big_dot_radius: int = 5
    normal_dot_radius: int = 3
    small_dot_radius: int = 2
    small_font: int = 14
    normal_font: int = 20
    big_font: int = 32
    size: b2Vec2 = b2Vec2(width, height)
    simulation_pos: b2Vec2 = b2Vec2(simulation_xshift, simulation_yshift)
    simulation_size: b2Vec2 = b2Vec2(
        width - simulation_xshift, height - simulation_yshift)
    board_pos: b2Vec2 = b2Vec2(0, simulation_size.y)
    popup_pos: b2Vec2 = (b2Vec2(width, height) - popup_size / 2) / 2
    ndigits: int = 4


@dataclass
class DesignData:
    valid: bool = False
    points: List[b2Vec2] = field(default_factory=list)
    vertices: List[b2Vec2] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    normal_plane: bool = True  # for rotation after inverse resize
    zero_angle: float = 0.0
    angle: float = 0.0
    moved: bool = False
    rotated: bool = False

    # TODO: toggle color
    color: tuple = field(default=boxcolors.STATIC_OBSTACLE)

    physic: Dict = field(default_factory=dict)
    # indicates which param to currently change
    physic_ix: int = 0

    effect: Dict = field(default_factory=dict)
    effect_ix: int = 0

    param_group: ParamGroup = ParamGroup.PHYSIC

    float_inc: float = 0.1
