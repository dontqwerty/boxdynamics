from enum import Enum

# colors
BLACK = (0, 0, 0, 0)
GREY = (128, 128, 128, 255)
PURPLE = (128, 0, 255, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)
BROWN = (139, 69, 19, 255)
TURQUOISE = (0, 255, 255, 255)
RED = (255, 0, 0, 255)
MAGENTA = (255, 0, 255, 255)
ORANGE = (255, 69, 0, 255)
YELLOW = (255, 255, 0, 255)
WHITE = (255, 255, 255, 255)

BACK = BLACK
INFO_BACK = GREY
AGENT = MAGENTA
STATIC_OBSTACLE = BROWN
MOVING_OBSTACLE = YELLOW
STATIC_ZONE = TURQUOISE
MOVING_ZONE = ORANGE
BORDER = WHITE
ACTION = GREEN
OBSERVATION = YELLOW
INTERSECTION = RED