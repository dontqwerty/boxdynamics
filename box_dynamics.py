import math
import random

from enum import Enum
from dataclasses import dataclass
from typing import Any
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
AGENT_COLOR = COLOR_MAGENTA
STATIC_OBSTACLE_COLOR = COLOR_GREY
MOVING_OBSTACLE_COLOR = COLOR_YELLOW
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

# space in meters between agent body edges and its head
# it avoids wrong measures when the intersection
# points are very close to the agent edges
# TODO: check if needed
AGENT_HEAD_INSIDE = 10**(-1)

# action space
MIN_ANGLE = -math.pi * 3 / 4  # radians
MAX_ANGLE = math.pi * 3 / 4
# MIN_ANGLE = 0  # radians
# MAX_ANGLE = 2*math.pi

MIN_FORCE = 0  # newtons
MAX_FORCE = 10

# observation space
OBSERVATION_RANGE = math.pi - math.pi/4  # 2*math.pi
OBSERVATION_NUM = 5  # number of distance vectors

# agent frictions
AGENT_ANGULAR_DAMPING = 2
AGENT_LINEAR_DAMPING = 0.5

# objects
AGENT_MASS = 0.2 # kg
MOVING_OBSTACLE_DENSITY = 1000000000 # kg/(m2)


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
    velocity: tuple = (0, 0)  # used for moving bodies
    on_contact: Any = None  # defines what happens when the body hits another body
    off_contact: Any = None  # defines what happens when the body finish hitting another body
    agent_contact: bool = False  # if the body is currently in contact with the agent
    reward: float = 0  # reward when agents hits the object
    # level of deepness when drawing screen (0 is above everything else)
    # if multiple object share same level, first created objects are below others
    level: int = 0

@dataclass
class Observation:
    index: int = -1
    angle: float = 0
    intersection: tuple((float, float)) = (np.inf, np.inf)
    distance: float = np.inf
    body: BodyData = None


class ContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact):
        try:
            contact.fixtureA.body.userData.on_contact(
                contact, contact.fixtureA.body, contact.fixtureB.body)
        except TypeError:
            pass
        try:
            contact.fixtureB.body.userData.on_contact(
                contact, contact.fixtureB.body, contact.fixtureA.body)
        except TypeError:
            pass

        pass

    def EndContact(self, contact):
        try:
            contact.fixtureA.body.userData.off_contact(
                contact, contact.fixtureA.body, contact.fixtureB.body)
        except TypeError:
            pass
        try:
            contact.fixtureB.body.userData.off_contact(
                contact, contact.fixtureB.body, contact.fixtureA.body)
        except TypeError:
            pass

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

        self.observation_space = gym.spaces.Dict({
            "distances": gym.spaces.Box(low=0, high=np.inf, shape=(OBSERVATION_NUM,)),
            "types": gym.spaces.Tuple((
                [gym.spaces.Discrete(len(BodyType))]*OBSERVATION_NUM
            )),
            "position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "linear_velocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        })

        # it's like the observation space but with more informations
        self.data = list() # list of Observation dataclasses

        # pygame setup
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Box Dynamics')
        self.clock = pygame.time.Clock()
        pygame.font.init()  # to render text

        # adding world borders
        self.__create_borders()

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
        self.__update_agent_head()

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

        assert self.state in self.observation_space

        self.prev_state = self.state

        step_reward = 0

        for body in self.world.bodies:
            if body.userData.agent_contact:
                step_reward += body.userData.reward

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
        self.__draw_text()

        pygame.display.flip()
        self.clock.tick(TARGET_FPS)
        return True

    def __destroy(self):
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        pygame.quit()

    def __create_borders(self):
        inside = 0  # defines how much of the borders are visible

        # TODO: add fixtures (?)

        self.bottom_border = self.world.CreateStaticBody(
            position=(WORLD_WIDTH / 2, inside - (BOUNDARIES_WIDTH / 2)),
            shapes=b2PolygonShape(
                box=(WORLD_WIDTH / 2 + BOUNDARIES_WIDTH, BOUNDARIES_WIDTH / 2)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, on_contact=self.__on_contact_border,
                              off_contact=self.__off_contact, level=1)
        )
        self.top_border = self.world.CreateStaticBody(
            position=(WORLD_WIDTH / 2, WORLD_HEIGHT -
                      inside + (BOUNDARIES_WIDTH / 2)),
            shapes=b2PolygonShape(
                box=(WORLD_WIDTH / 2 + BOUNDARIES_WIDTH, BOUNDARIES_WIDTH / 2)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, on_contact=self.__on_contact_border,
                              off_contact=self.__off_contact, level=1)
        )

        self.left_border = self.world.CreateStaticBody(
            position=(inside - (BOUNDARIES_WIDTH / 2), WORLD_HEIGHT / 2),
            shapes=b2PolygonShape(
                box=(BOUNDARIES_WIDTH / 2, WORLD_HEIGHT / 2 + BOUNDARIES_WIDTH)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, on_contact=self.__on_contact_border,
                              off_contact=self.__off_contact, level=1)
        )
        self.right_border = self.world.CreateStaticBody(
            position=(WORLD_WIDTH - inside +
                      (BOUNDARIES_WIDTH / 2), WORLD_HEIGHT / 2),
            shapes=b2PolygonShape(
                box=(BOUNDARIES_WIDTH / 2, WORLD_HEIGHT / 2 + BOUNDARIES_WIDTH)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, on_contact=self.__on_contact_border,
                              off_contact=self.__off_contact, level=1)
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

        area = agent_width * agent_height

        self.agent_body = self.world.CreateDynamicBody(
            position=agent_pos, angle=agent_angle)
        self.agent_fix = self.agent_body.CreatePolygonFixture(
            box=(agent_width, agent_height), density=AGENT_MASS/area)

        self.agent_body.userData = BodyData(
            type=BodyType.AGENT, color=AGENT_COLOR, on_contact=None, level=0)

        self.agent_fix.body.angularDamping = AGENT_ANGULAR_DAMPING
        self.agent_fix.body.linearDamping = AGENT_LINEAR_DAMPING

    def __update_agent_head(self):
        self.agent_head = [
            (self.agent_body.position[0] + math.cos(
                self.agent_body.angle) * (self.agent_size[0] - AGENT_HEAD_INSIDE)),
            (self.agent_body.position[1] + math.sin(
                self.agent_body.angle) * (self.agent_size[0] - AGENT_HEAD_INSIDE))]
        # self.agent_head = self.agent_body.position

    def __get_agent_velocity(self):
        return math.sqrt(self.agent_body.linearVelocity[0]**2 + self.agent_body.linearVelocity[1]**2)

    def __get_observation_angle(self, delta_angle):
        return self.agent_body.angle - (OBSERVATION_RANGE / 2) + (OBSERVATION_RANGE /
                                           (OBSERVATION_NUM - 1) * delta_angle)

    def __get_observations(self):
        self.data.clear()

        for delta_angle in range(OBSERVATION_NUM):
            # absolute angle for the observation vector
            # based on OBSERVATION_RANGE
            angle = self.__get_observation_angle(delta_angle)

            # line equation coefficients for observation vector
            m1, n1 = self.__get_line_eq_angle(self.agent_head, angle)

            # finding closest intersection for current observation vector
            state = dict()
            observation = Observation()
            observation.distance = np.inf
            for bix, body in enumerate(self.world.bodies):
                if body.userData.type == BodyType.AGENT:
                    continue
                for fix, fixture in enumerate(body.fixtures):
                    # check for intersection between observation vector and
                    # every line created by adjacent polygon vertices
                    for vix in range(len(fixture.shape.vertices)):
                        point1 = body.transform * fixture.shape.vertices[vix]
                        point2 = body.transform * \
                            fixture.shape.vertices[(
                                vix + 1) % len(fixture.shape.vertices)]

                        # line equation coefficients for current segment
                        m2, n2 = self.__get_line_eq(point1, point2)

                        # get intersection point between observation vector
                        # and segment line
                        intersection = self.__get_intersection(
                            (m1, n1), (m2, n2))

                        # calculate distance of intersection point from agent head
                        distance = self.__euclidean_distance(
                            self.agent_head, intersection)

                        # check if intersection is valid
                        valid_intersection = self.__check_intersection(
                            point1, point2, intersection, angle, distance)

                        # look for the closest valid intersection point
                        # the rest should not be visible to the agent
                        if distance < observation.distance and valid_intersection:
                            observation.index = delta_angle
                            observation.angle = angle
                            observation.body = body
                            observation.distance = distance
                            observation.intersection = intersection

            # no check here because since the world has borders, there should
            # always be at least one valid intersection point
            self.data.append(observation)

            # filter info and return observation space
            # speed should not go down by iterating again here
            state["distances"] = np.array([observation.distance for observation in self.data], dtype=np.float32)
            state["types"] = np.array([observation.body.userData.type.value for observation in self.data])
            state["position"] = np.array(self.agent_body.position, dtype=np.float32)
            state["linear_velocity"] = [self.__get_agent_velocity()]

        return state

    # on contact functions
    def __on_contact_border(self, contact, this_body, other_body):
        if other_body.userData.type == BodyType.AGENT:
            this_body.userData.agent_contact = True

    def __on_contact_static_obstacle(self, contact, this_body, other_body):
        if other_body.userData.type == BodyType.AGENT:
            this_body.userData.agent_contact = True

    def __on_contact_moving_obstacle(self, contact, this_body, other_body):
        if other_body.userData.type == BodyType.AGENT:
            this_body.linearVelocity = this_body.userData.velocity
            this_body.userData.agent_contact = True
        elif other_body.userData.type == BodyType.MOVING_ZONE or other_body.userData.type == BodyType.STATIC_ZONE:
            return
        else:
            this_body.linearVelocity = this_body.userData.velocity = self.__point_mult(
                this_body.userData.velocity, -1)

    def __on_contact_static_zone(self, contact, this_body, other_body):
        if other_body.userData.type == BodyType.AGENT:
            this_body.userData.agent_contact = True
            other_body.linearDamping = 1
            # other_body.angularDamping = 0

    def __on_contact_moving_zone(self, contact, this_body, other_body):
        if other_body.userData.type == BodyType.AGENT:
            this_body.userData.agent_contact = True
        if other_body.userData.type == BodyType.BORDER:
            this_body.linearVelocity = this_body.userData.velocity = self.__point_mult(
                this_body.userData.velocity, -1)

    def __off_contact(self, contact, this_body, other_body):
        if other_body.userData.type == BodyType.AGENT:
            this_body.userData.agent_contact = False

        if this_body.userData.type == BodyType.MOVING_OBSTACLE and other_body.userData.type == BodyType.AGENT:
            pass

        if this_body.userData.type == BodyType.STATIC_ZONE and other_body.userData.type == BodyType.AGENT:
            other_body.angularDamping = AGENT_LINEAR_DAMPING
            other_body.angularDamping = AGENT_ANGULAR_DAMPING

    # user functions

    def get_world_size(self):
        return WORLD_WIDTH, WORLD_HEIGHT

    def create_static_obstacle(self, pos, size, angle=0, reward=-1, level=3):
        body = self.world.CreateStaticBody(
            position=pos, angle=angle
        )
        fixture = body.CreatePolygonFixture(
            box=size)

        body.userData = BodyData(
            type=BodyType.STATIC_OBSTACLE, color=STATIC_OBSTACLE_COLOR,
            on_contact=self.__on_contact_static_obstacle,
            off_contact=self.__off_contact,
            reward=reward, level=level)

    def create_moving_obstacle(self, pos, size, velocity, angle=0, reward=-2, level=2):
        body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False, )
        _ = body.CreatePolygonFixture(
            box=size, density=MOVING_OBSTACLE_DENSITY)

        body.userData = BodyData(
            type=BodyType.MOVING_OBSTACLE, color=MOVING_OBSTACLE_COLOR,
            velocity=velocity, on_contact=self.__on_contact_moving_obstacle,
            off_contact=self.__off_contact, reward=reward, level=level)

    def create_static_zone(self, pos, size, angle=0, reward=1, level=3):
        body = self.world.CreateStaticBody(
            position=pos, angle=angle)
        fixture = body.CreatePolygonFixture(
            box=size)
        fixture.sensor = True

        body.userData = BodyData(
            type=BodyType.STATIC_ZONE, color=STATIC_ZONE_COLOR,
            on_contact=self.__on_contact_static_zone,
            off_contact=self.__off_contact, reward=reward, level=level)

    def create_moving_zone(self, pos, size, velocity, angle=0, reward=2, level=1):
        body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False)
        fixture = body.CreatePolygonFixture(
            box=size)

        fixture.sensor = True

        body.userData = BodyData(
            type=BodyType.MOVING_ZONE, color=MOVING_ZONE_COLOR, velocity=velocity,
            on_contact=self.__on_contact_moving_zone,
            off_contact=self.__off_contact, reward=reward, level=level)

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

    def __draw_text(self):
        text_font = pygame.font.SysFont('Comic Sans MS', 20)

        # drawing distance text
        for observation in self.data:
            distance = round(observation.distance, 1)

            text_point = self.__pygame_coord(
                self.__point_add(
                    self.agent_head, self.__point_div(
                        self.__point_sub(observation.intersection, self.agent_head), 2
                    )
                )
            )
            text_surface = text_font.render(
                str(distance), False, COLOR_BLACK, COLOR_WHITE)
            self.screen.blit(text_surface, text_point)

        # fps
        fps = round(self.clock.get_fps())
        fps_point = (20, 30)
        text_surface = text_font.render(
            str(fps), True, COLOR_BLACK, COLOR_WHITE)
        self.screen.blit(text_surface, fps_point)

        # agent info
        velocity = self.__get_agent_velocity()
        agent_info = "Velocity: {}".format(round(velocity, 1))
        text_surface = text_font.render(
            agent_info, True, COLOR_BLACK, COLOR_WHITE
        )
        self.screen.blit(text_surface, (20,700))

    def __draw_observations(self):
        start_point = self.__pygame_coord(self.agent_head)
        for observation in self.data:
            end_point = self.__pygame_coord(observation.intersection)

            # drawing observation vectors
            pygame.draw.line(self.screen, observation.body.userData.color,
                             start_point, end_point)
            # drawing intersection points
            pygame.draw.circle(self.screen, INTERSECTION_COLOR, end_point, 3)

    # transform point in world coordinates to point in pygame coordinates
    def __pygame_coord(self, point):
        return [point[0] * PPM, SCREEN_HEIGHT - (point[1] * PPM)]

    # utilities functions
    def __euclidean_distance(self, point1, point2):
        return round(
            math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2),
            FLOAT_PRECISION)

    def __check_intersection(self, point1, point2, intersection, angle, distance):
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

        # calculating where the intersection point should be
        # based on current angle and distance
        # this avoids finding an intersection behind the agent
        # when the observation vector is actually going "forward"
        # from the agent perspective
        correct_direction = False
        precision = 2
        point_x = round(self.agent_head[0] +
                        distance * math.cos(angle), precision)
        point_y = round(self.agent_head[1] +
                        distance * math.sin(angle), precision)
        if point_x == round(intersection[0], precision) and point_y == round(intersection[1], precision):
            correct_direction = True
            pass

        return x_inside and y_inside and correct_direction

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
