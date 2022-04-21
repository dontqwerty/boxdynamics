import logging
import math
import random
from dataclasses import is_dataclass
from enum import IntEnum
from logging import debug
from typing import Dict

from Box2D import b2Vec2

from boxdef import (BodyType, DesignData, EffectType, EffectWhen, EffectWho,
                    SetType)


def toggle_enum(e, skip=[], increase=True):
    try:
        enum_list = list(type(e))
    except TypeError as e:
        logging.error("{}".format(e, type(e)))
        return
    if increase:
        new_ix = (enum_list.index(e) + 1) % len(enum_list)
    else:
        new_ix = (enum_list.index(e) - 1) % len(enum_list)
    new_e = enum_list[new_ix]

    # TODO: check for undless loop
    if new_e in skip:
        new_e = toggle_enum(new_e, skip, increase)

    return new_e


def check_design_validity(design: DesignData):
    # TODO: add other checks (?)
    if all(v is not None for v in design.vertices):
        # checking for vertices
        return True
    else:
        logging.warning("Invalid design {}".format(design))
        return False


def get_effect(type: EffectType, param_0=0.0, param_1=0.0, who=EffectWho.AGENT, when=EffectWhen.DURING_CONTACT):
    # TODO: angle in degrees not radians
    effect = {"type": type, "who": who, "when": when,
              "param_0": param_0, "param_1": param_1}
    return effect


def get_design_data(set_type=SetType.DEFAULT, current_design: DesignData = DesignData()):

    design_data = DesignData()
    design_data.points = [None] * 2
    design_data.vertices = [None] * 4

    # use copiable data types
    if set_type == SetType.DEFAULT:
        design_data.physic = {"type": BodyType.STATIC_OBSTACLE,
                              "reward": 0.0,
                              "level": 0,
                              "lin_velocity": 10.0,
                              "lin_velocity_angle": 0.0,
                              "ang_velocity": 0.0,
                              "density": 1.0,
                              "inertia": 0.0,
                              "friction": 0.0,
                              "lin_damping": 0.0,
                              "ang_damping": 0.0}
        design_data.effect = get_effect(EffectType.NONE)
    elif set_type == SetType.PREVIOUS:
        # indexes
        design_data.param_group = current_design.param_group
        design_data.physic_ix = current_design.physic_ix
        design_data.effect_ix = current_design.effect_ix

        design_data.physic = current_design.physic.copy()
        design_data.effect = current_design.effect.copy()
    elif set_type == SetType.RANDOM:
        types = list(BodyType)
        types.remove(BodyType.AGENT)
        types.remove(BodyType.BORDER)
        types.remove(BodyType.DEFAULT)
        design_data.physic = {"type": random.choice(types),
                              "reward": random.uniform(-1, 1),
                              "level": 0,
                              "lin_velocity": random.uniform(0, 10),
                              "lin_velocity_angle": random.uniform(0, 2*math.pi),
                              "ang_velocity": random.uniform(-5, 5),
                              "density": 0.5,
                              # TODO: remove inertia
                              "inertia": 0,
                              "friction": random.uniform(0, 0.001),
                              "lin_damping": random.uniform(0, 0.001),
                              "ang_damping": random.uniform(0, 0.001)}
        design_data.effect = get_effect(type=random.choice(list(EffectType)), param_0=random.uniform(
            0, 2*math.pi), param_1=random.uniform(-10000, 10000), who=random.choice(list(EffectWho)), when=random.choice(list(EffectWhen)))

    return design_data


def copy_design_data(design: DesignData):
    design_dict: Dict[str,
                      DesignData] = design.__dict__  # pointer
    design_data: Dict[str, DesignData] = dict()

    # checking for fields that need to be copied manually
    for key in list(design_dict.keys()):
        # b2Vec2
        if type(design_dict[key]) is b2Vec2:
            design_data[key] = (design_dict['key'].copy())
        # list of b2Vec2
        elif isinstance(design_dict[key], list) and all(isinstance(val, b2Vec2) for val in design_dict[key]):
            design_data[key] = list()
            for val in design_dict[key]:
                design_data[key].append(val.copy())
        # dictionary
        elif isinstance(design_dict[key], dict):
            design_data[key] = design_dict[key].copy()
        else:
            design_data[key] = design_dict[key]
    design_data = DesignData(**design_data)
    return design_data


def copy_design_bodies(design_bodies):
    design_copy = list()

    for body in design_bodies:
        if body.valid:
            design_copy.append(copy_design_data(design=body))

            # TODO: remove or make proper function
            assert design_copy[-1] == body and "Must equal"
            design_copy[-1].points[0].x += 1
            assert design_copy != body and "Must differ"
            design_copy[-1].points[0].x -= 1
            assert design_copy[-1] == body and "Must equal 2"

    return design_copy


def dataclass_to_dict(data):
    dump_db = list()
    for name, value in list(data.__dict__.items()):
        dump_db.append(list())
        dump_db[-1].append(name)
        # getting actual value of dataclass field
        if is_dataclass(value):
            # recursive for nested dataclasses
            debug("dataclass {} : {}".format(name, value))
            dump_db[-1].append(dataclass_to_dict(value))
        elif isinstance(value, dict):
            # using copy() dict method
            dump_db[-1].append(value.copy())
            debug("dict {} : {}".format(name, value))
        elif isinstance(value, IntEnum):
            # copying enum
            debug("IntEnum {} : {}".format(name, value.name))
            dump_db[-1].append(type(value)(value))
        # TODO: support other types of lists
        elif isinstance(value, list):
            # appending list for values
            debug("list {}".format(name))
            dump_db[-1].append(list())
            for sub_value in value:
                # appending every value in list
                if is_dataclass(sub_value):
                    # list of dataclasses
                    dump_db[-1][1].append(dataclass_to_dict(sub_value))
                elif isinstance(sub_value, b2Vec2):
                    # list of b2Vec2
                    dump_db[-1][1].append(list(sub_value))
                elif isinstance(sub_value, (int, float, str)):
                    # list of strings
                    dump_db[-1][1].append(sub_value)
                elif isinstance(sub_value, dict):
                    dump_db[-1][1].append(sub_value.copy())
                else:
                    logging.error("Unexpected type in dataclass {} {}".format(
                        name, type(sub_value)))
                    assert False
        elif isinstance(value, b2Vec2):
            debug("b2Vec2 {} : {}".format(name, value))
            dump_db[-1].append(list(value))
        else:
            # TODO: tuples (and other similar stuff inside data) might be dangerous
            debug("normal {} : {}".format(name, value))
            dump_db[-1].append(value)

    dump_db = dict(dump_db)
    return dump_db

# TODO: change with get_point_angle everywhere


def anglemag_to_vec(angle, magnitude):
    return b2Vec2(math.cos(angle), math.sin(angle)) * magnitude


def get_point_angle(angle: float, length: float = 1, from_point: b2Vec2 = b2Vec2(0, 0)) -> b2Vec2:
    return from_point + b2Vec2(math.cos(angle), math.sin(angle)) * length


def get_line_eq(point1: b2Vec2, point2: b2Vec2):
    try:
        m = (point1.y - point2.y) / (point1.x - point2.x)
    except ZeroDivisionError:
        if point1.y > point2.y:
            m = math.tan(math.pi/2)
        else:
            m = math.tan(math.pi*3/2)

    n = point1[1] - (m*point1[0])
    return m, n


def get_line_eq_angle(point: b2Vec2, angle: float):
    # m*x + n = y
    m = math.tan(angle)
    n = point.y - m*point.x

    return m, n


def get_intersection(line1, line2):
    intersection = [-1, -1]

    a1 = line1[0]
    b1 = -1
    c1 = line1[1]

    a2 = line2[0]
    b2 = -1
    c2 = line2[1]

    try:
        intersection[0] = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
        intersection[1] = (c1*a2 - c2*a1) / (a1*b2 - a2*b1)
    except ZeroDivisionError:
        # TODO: handle this
        debug("Intersection calculation failed")
        pass

    return b2Vec2(intersection)
