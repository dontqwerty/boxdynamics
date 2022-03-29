import math
from Box2D import b2Vec2

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

    intersection[0] = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
    intersection[1] = (c1*a2 - c2*a1) / (a1*b2 - a2*b1)

    return b2Vec2(intersection)