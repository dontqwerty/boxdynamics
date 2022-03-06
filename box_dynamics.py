import math
import random

from enum import Enum
from dataclasses import dataclass
from turtle import position
import numpy as np
import gym
import pygame
from Box2D import b2World, b2PolygonShape, b2ContactListener

# TODO: if higher PPM, possible bugs with intersections
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800  # pixels
PPM = 10  # pixel per meter


# colors
COLOR_RED = (255, 0, 0, 255)
COLOR_YELLOW = (255, 255, 0, 255)
COLOR_GREEN = (0, 255, 0, 255)
COLOR_TURQUOISE = (0, 255, 255, 255)
COLOR_BLUE = (0, 0, 255, 255)
COLOR_PURPLE = (127, 0, 255, 255)
COLOR_MAGENTA = (255, 0, 255, 255)
COLOR_GREY = (128, 128, 128, 255)
COLOR_BLACK = (0, 0, 0, 0)
COLOR_WHITE = (255, 255, 255, 255)

BACKGROUND_COLOR = COLOR_BLACK
AGENT_COLOR = COLOR_PURPLE
STATIC_OBSTACLE_COLOR = COLOR_GREY
MOVING_OBSTACLE_COLOR = COLOR_WHITE
STATIC_ZONE_COLOR = COLOR_TURQUOISE
MOVING_ZONE_COLOR = COLOR_BLUE
BORDER_COLOR = COLOR_WHITE
ACTION_COLOR = COLOR_GREEN
OBSERVATION_COLOR = COLOR_YELLOW
INTERSECTION_COLOR = COLOR_RED

# world
# TODO: the resulting world sizes must be multiple of 10 or even 100
# to avoid problems when calculating intersections
WORLD_WIDTH = SCREEN_WIDTH / PPM  # meters
WORLD_HEIGHT = SCREEN_HEIGHT / PPM

BOUNDARIES_WIDTH = 10  # meters

FLOAT_PRECISION = 9  # number of decimal digits to use for most calculations

# space in meters between agent edges and its head
# it avoids wrong measures when the intersection
# points are very close to the agent edges
AGENT_HEAD_INSIDE = 10**(-1)

# action space
MIN_ANGLE = -math.pi * 3 / 4  # radians
MAX_ANGLE = math.pi * 3 / 4
# MIN_ANGLE = 0  # radians
# MAX_ANGLE = 2*math.pi

MIN_FORCE = 0  # newtons
MAX_FORCE = 10

# observation space
OBSERVATION_RANGE = math.pi - math.pi/4
OBSERVATION_NUM = 5  # number of distance vectors
MAX_DISTANCE = math.sqrt(WORLD_WIDTH**2 + WORLD_HEIGHT**2)  # meters

# agent frictions
AGENT_ANGULAR_DAMPING = 2
AGENT_LINEAR_DAMPING = 0.5


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


@dataclass
class BodyData:
    type: Enum = BodyType.DEFAULT
    color: tuple = COLOR_BLACK
    shape: Enum = BodyShape.BOX
    # level of deepness when drawing screen (0 is above everything else)
    # if multiple object share same level, first created objects are below others
    level: int = 0


class ContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact):
        print("begin {} {}".format(contact.fixtureA.body.userData.type,
              contact.fixtureB.body.userData.type))
        pass

    def EndContact(self, contact):
        print("end {} {}".format(contact.fixtureA.body.userData.type,
              contact.fixtureB.body.userData.type))
        pass

    def PreSolve(self, contact, oldMainfold):
        pass

    def PostSolve(self, contact, impulse):
        pass


class BoxEnv(gym.Env):
    def __init__(self) -> None:
        # initializing base class
        super(BoxEnv, self).__init__()

        # world keeps track of objects and physics
        self.world = b2World(gravity=(0, 0), doSleep=True,
                             contactListener=ContactListener())

        # action space
        # relative agent_angle, force
        self.action_space = gym.spaces.Box(
            np.array([MIN_ANGLE, MIN_FORCE]).astype(np.float32),
            np.array([MAX_ANGLE, MAX_FORCE]).astype(np.float32)
        )

        # observation space
        # 8 distance vectors for the agent to know
        # where he is
        self.observation_space = gym.spaces.Box(
            low=0, high=MAX_DISTANCE, shape=(OBSERVATION_NUM,)
        )

        self.intersections = list()

        # pygame setup
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Box Dynamics')
        self.clock = pygame.time.Clock()
        pygame.font.init()  # to render text

        # adding world boundaries
        self.__create_boundaries()

        # adding dynamic body for RL agent
        # self.__create_agent(agent_angle=2, agent_pos=[25, 20])
        # TODO: support agent parameters from ouside class
        self.__create_agent()

        # defining own polygon draw function
        b2PolygonShape.draw = self.__draw_polygon

        self.prev_state = None

    def reset(self):
        # resetting base class
        super().reset()

        # resetting reward
        self.reward = 0.0

        # returning state after step with None action
        return self.step(None)[0]

    def step(self, action) -> tuple:
        # calculate agent head point
        # TODO: support circles
        # TODO: support agent_head definition from outside class
        self.agent_head = [
            (self.agent_body.position[0] + math.cos(
                self.agent_body.angle) * (self.agent_size[0] - AGENT_HEAD_INSIDE)),
            (self.agent_body.position[1] + math.sin(
                self.agent_body.angle) * (self.agent_size[0] - AGENT_HEAD_INSIDE))]
        # self.agent_head = self.agent_body.position

        if action is not None:
            # calculating where to apply force
            # target -> "head" of agent
            self.action = [action[1] *
                           math.cos(action[0] + self.agent_body.angle),
                           action[1] *
                           math.sin(action[0] + self.agent_body.angle)]

            self.agent_body.ApplyForce(
                force=self.action, point=self.agent_head, wake=True)

        # Make Box2D simulate the physics of our world for one step.
        # vel_iters, pos_iters = 6, 2
        # world.Step(timeStep, vel_iters, pos_iters)
        self.world.Step(TIME_STEP, 10, 10)

        # clear forces or they will stay permanently
        self.world.ClearForces()

        self.state = self.__get_observations()

        self.prev_state = self.state

        # changing moving direction of moving bodies if they touch the borders
        # for body in self.world.bodies:
        #     if body.userData.type == BodyType.MOVING_OBSTACLE or body.userData.type == BodyType.MOVING_ZONE:
        #         r = math.sqrt(body.size[0]**2 + body.size[1]**2)
        #         if body.position[1] + r > WORLD_HEIGHT or body.position[1] - r < 0:
        #             body.linearVelocity = self.__point_mult(
        #                 body.linearVelocity, -1)

        step_reward = 0
        done = False
        info = {}
        return self.state, step_reward, done, info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                # The user closed the window or pressed escape
                self.__destroy()
                return False

        # background for render screen
        self.screen.fill(BACKGROUND_COLOR)

        # Draw the world
        bodies_levels = [[i, b.userData.level]
                         for i, b in enumerate(self.world.bodies)]
        bodies_levels.sort(key=lambda x: x[1], reverse=True)

        for bix, level in bodies_levels:
            for fixture in self.world.bodies[bix].fixtures:
                fixture.shape.draw(fixture, self.world.bodies[bix])

        # for body in self.world.bodies:
        #     for fixture in body.fixtures:
        #         fixture.shape.draw(fixture, body)
        self.__draw_action()
        self.__draw_observations()

        pygame.display.flip()
        self.clock.tick(TARGET_FPS)
        return True

    def __destroy(self):
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        pygame.quit()

    def __create_boundaries(self):
        inside = 1  # defines how much of the borders are visible

        self.bottom_boundary = self.world.CreateStaticBody(
            position=(WORLD_WIDTH / 2, inside - (BOUNDARIES_WIDTH / 2)),
            shapes=b2PolygonShape(box=(WORLD_WIDTH / 2, BOUNDARIES_WIDTH / 2)),
            userData=BodyData(BodyType.BORDER, BORDER_COLOR, 1)
        )
        self.top_boundary = self.world.CreateStaticBody(
            position=(WORLD_WIDTH / 2, WORLD_HEIGHT -
                      inside + (BOUNDARIES_WIDTH / 2)),
            shapes=b2PolygonShape(box=(WORLD_WIDTH / 2, BOUNDARIES_WIDTH / 2)),
            userData=BodyData(BodyType.BORDER, BORDER_COLOR, 1)
        )
        self.left_boundary = self.world.CreateStaticBody(
            position=(inside - (BOUNDARIES_WIDTH / 2), WORLD_HEIGHT / 2),
            shapes=b2PolygonShape(
                box=(BOUNDARIES_WIDTH / 2, WORLD_HEIGHT / 2)),
            userData=BodyData(BodyType.BORDER, BORDER_COLOR, 1)
        )
        self.right_boundary = self.world.CreateStaticBody(
            position=(WORLD_WIDTH - inside +
                      (BOUNDARIES_WIDTH / 2), WORLD_HEIGHT / 2),
            shapes=b2PolygonShape(
                box=(BOUNDARIES_WIDTH / 2, WORLD_HEIGHT / 2)),
            userData=BodyData(BodyType.BORDER, BORDER_COLOR, 1)
        )

    def __create_agent(self, agent_size=(1, 1), agent_pos=None, agent_angle=None):
        # TODO: check for agent size
        if agent_size is None:
            pass

        agent_width, agent_height = agent_size

        # setting random initial position
        if agent_pos is None:
            # TODO: support float initial position
            r = math.sqrt(agent_width**2 + agent_height**2)
            x = random.randint(int(r), int(WORLD_WIDTH - r))
            y = random.randint(int(r), int(WORLD_HEIGHT - r))
            agent_pos = (x, y)

        # setting random initial angle
        if agent_angle is None:
            agent_angle = random.random() * (2*math.pi)

        self.agent_size = (agent_width, agent_height)

        mass = 0.1  # 1 kg for the agent
        area = agent_width * agent_height

        self.agent_body = self.world.CreateDynamicBody(
            position=agent_pos, angle=agent_angle)
        self.agent_fix = self.agent_body.CreatePolygonFixture(
            box=(agent_width, agent_height), density=mass/area, friction=0.3)

        self.agent_body.userData = BodyData(BodyType.AGENT, AGENT_COLOR)

        self.agent_fix.body.angularDamping = AGENT_ANGULAR_DAMPING
        self.agent_fix.body.linearDamping = AGENT_LINEAR_DAMPING

    def __get_observations(self):
        observations = list()
        self.intersections.clear()

        for delta_angle in range(OBSERVATION_NUM):
            # absolute angle for the observation vector
            # based on OBSERVATION_RANGE
            angle = self.agent_body.angle - \
                (OBSERVATION_RANGE / 2) + (OBSERVATION_RANGE /
                                           (OBSERVATION_NUM - 1) * delta_angle)

            # line equation coefficients for observation vector
            m1, n1 = self.__get_line_eq_angle(self.agent_head, angle)

            # finding closest intersection for current observation vector
            shorter_distance = MAX_DISTANCE
            shorter_intersection = [-1, -1]
            for bix, body in enumerate(self.world.bodies):
                if body.userData.type == BodyType.AGENT:
                    continue
                for fix, fixture in enumerate(body.fixtures):
                    # check for intersection between observation vector and
                    # every line created by adjacent polygon vertices
                    for vix in range(len(fixture.shape.vertices)):
                        # finding points of segment
                        point1 = self.__point_add(
                            body.position, fixture.shape.vertices[vix])
                        point2 = self.__point_add(
                            body.position, fixture.shape.vertices[(vix + 1)
                                                                  % len(fixture.shape.vertices)])

                        # line equation coefficients for current segment
                        m2, n2 = self.__get_line_eq(point1, point2)

                        # get intersection point between observation vector
                        # and segment line
                        intersection = self.__get_intersection(
                            (m1, n1), (m2, n2))

                        # check if intersection inside segment
                        valid_intersection = self.__check_intersection(
                            point1, point2, intersection)

                        if valid_intersection:
                            # calculate distance of intersection point from agent head
                            distance = self.__euclidean_distance(
                                self.agent_head, intersection)

                            # calculating where the intersection point should be
                            # based on current angle and distance
                            # this avoids finding an intersection behind the agent
                            # when the observation vector is actually going "forward"
                            # from the agent perspective
                            precision = 2
                            point_x = round(self.agent_head[0] +
                                            distance * math.cos(angle), precision)
                            point_y = round(self.agent_head[1] +
                                            distance * math.sin(angle), precision)
                            if point_x == round(intersection[0], precision) and \
                                    point_y == round(intersection[1], precision):
                                # look for the closest valid intersection point
                                # the rest should not be visible to the agent
                                if distance < shorter_distance:
                                    shorter_distance = distance
                                    shorter_intersection = intersection
            # no check here because since the world has boundaries, there should
            # always be at least one valid intersection point
            self.intersections.append(shorter_intersection)
            observations.append(shorter_distance)

        return observations

    # user functions
    def get_world_size(self):
        return WORLD_WIDTH, WORLD_HEIGHT

    def create_body(self, shape, pos, size, angle=0, moving=False, solid=False, level=1):
        assert(shape in BodyShape)

        if moving:
            body = self.world.CreateDynamicBody(
                position=pos, angle=angle, 
            )

        if shape == BodyShape.BOX:
            pass
        elif shape == BodyShape.CIRCLE:
            pass

    def create_static_obstacle(self, pos, size, angle=0, level=1):
        body = self.world.CreateStaticBody(
            position=pos, angle=angle
        )
        fixture = body.CreatePolygonFixture(
            box=size)

        body.userData = BodyData(
            BodyType.STATIC_OBSTACLE, STATIC_OBSTACLE_COLOR, level)

    def create_moving_obstacle(self, pos, size, velocity, angle=0, level=1):
        # body = self.world.CreateKinematicBody(
        body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity,
            bullet=False)
        _ = body.CreatePolygonFixture(
            box=size)

        body.userData = BodyData(
            BodyType.MOVING_OBSTACLE, MOVING_OBSTACLE_COLOR, level)

    def create_static_zone(self, pos, size, angle=0, level=1):
        body = self.world.CreateStaticBody(
            position=pos, angle=angle)
        fixture = body.CreatePolygonFixture(
            box=size)
        fixture.sensor = True

        body.userData = BodyData(
            BodyType.STATIC_ZONE, STATIC_ZONE_COLOR, level)

    def create_moving_zone(self, pos, size, velocity, angle=0, level=1):
        # body = self.world.CreateKinematicBody(
        body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity,
            bullet=False)
        fixture = body.CreatePolygonFixture(
            box=size)

        fixture.sensor = True

        body.userData = BodyData(
            BodyType.MOVING_ZONE, MOVING_ZONE_COLOR, level)

    # render functions
    def __draw_polygon(self, polygon, body):
        vertices = [(body.transform * v) *
                    PPM for v in polygon.shape.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, body.userData.color, vertices)

    def __draw_action(self):
        action_start = self.__pygame_coord(self.agent_head)
        action_end = self.__pygame_coord(
            self.__point_add(self.agent_head, self.action))

        pygame.draw.line(self.screen, ACTION_COLOR, action_start, action_end)

    def __draw_observations(self):
        text_font = pygame.font.SysFont('Comic Sans MS', 20)
        start_point = self.__pygame_coord(self.agent_head)
        for iix, intersection in enumerate(self.intersections):
            end_point = self.__pygame_coord(intersection)

            # drawing observation vectors
            pygame.draw.line(self.screen, OBSERVATION_COLOR,
                             start_point, end_point)
            # drawing intersection points
            pygame.draw.circle(self.screen, INTERSECTION_COLOR, end_point, 3)

            # drawing distance text
            distance = round(self.state[iix], 2)

            text_point = self.__pygame_coord(
                self.__point_add(
                    self.agent_head, self.__point_div(
                        self.__point_sub(intersection, self.agent_head), 2
                    )
                )
            )

            text_surface = text_font.render(
                str(distance), False, COLOR_BLACK, COLOR_WHITE)
            self.screen.blit(text_surface, text_point)
        fps = round(self.clock.get_fps())
        text_surface = text_font.render(
            str(fps), False, COLOR_BLACK, COLOR_WHITE)
        self.screen.blit(text_surface, (20, 30))

    # transform point in world coordinates to point in pygame coordinates
    def __pygame_coord(self, point):
        return [point[0] * PPM, SCREEN_HEIGHT - (point[1] * PPM)]

    # utilities functions
    def __euclidean_distance(self, point1, point2):
        return round(
            math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2),
            FLOAT_PRECISION)

    def __check_intersection(self, point1, point2, intersection):
        x1 = round(point1[0], FLOAT_PRECISION)
        y1 = round(point1[1], FLOAT_PRECISION)
        x2 = round(point2[0], FLOAT_PRECISION)
        y2 = round(point2[1], FLOAT_PRECISION)

        xi = round(intersection[0], FLOAT_PRECISION)
        yi = round(intersection[1], FLOAT_PRECISION)

        x_inside = False
        y_inside = False

        if (xi >= x1 and xi <= x2) or (xi <= x1 and xi >= x2):
            x_inside = True

        if (yi >= y1 and yi <= y2) or (yi <= y1 and yi >= y2):
            y_inside = True

        return x_inside and y_inside

    def __get_intersection(self, line1, line2):
        intersection = [-1, -1]

        a1 = line1[0]
        b1 = -1
        c1 = line1[1]

        a2 = line2[0]
        b2 = -1
        c2 = line2[1]

        intersection[0] = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
        intersection[1] = (c1*a2 - c2*a1) / (a1*b2 - a2*b1)

        return intersection

    def __get_line_eq_angle(self, point, angle):
        # m*x + n = y
        m = math.tan(angle)
        n = point[1] - m*point[0]

        return m, n

    def __get_line_eq(self, point1, point2):
        try:
            m = (point1[1] - point2[1]) / (point1[0] - point2[0])
        except ZeroDivisionError:
            if point1[1] > point2[1]:
                m = math.tan(math.pi/2)
            else:
                m = math.tan(math.pi*3/2)

        n = point1[1] - (m*point1[0])
        return m, n

    def __point_add(self, point1, point2):
        return [point1[0] + point2[0], point1[1] + point2[1]]

    def __point_sub(self, point1, point2):
        return [point1[0] - point2[0], point1[1] - point2[1]]

    def __point_mult(self, point, scalar):
        return [point[0] * scalar, point[1] * scalar]

    def __point_div(self, point, scalar):
        return [point[0] / scalar, point[1] / scalar]
