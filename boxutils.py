import logging
import math
from dataclasses import is_dataclass
from enum import IntEnum
from logging import debug

from Box2D import b2Vec2


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
                        logging.error("Unexpected type in dataclass".format(name, type(sub_value)))
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
