from cmath import rect
import math
import random

from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import List
import warnings
import numpy as np
import gym
import pygame
from Box2D import b2Vec2, b2World, b2Body, b2PolygonShape, b2ContactListener, b2RayCastCallback

# colors
COLOR_BLACK = (0, 0, 0, 0)
COLOR_GREY = (128, 128, 128, 255)
COLOR_PURPLE = (128, 0, 255, 255)
COLOR_BLUE = (0, 0, 255, 255)
COLOR_GREEN = (0, 255, 0, 255)
COLOR_TURQUOISE = (0, 255, 255, 255)
COLOR_RED = (255, 0, 0, 255)
COLOR_MAGENTA = (255, 0, 255, 255)
COLOR_YELLOW = (255, 255, 0, 255)
COLOR_WHITE = (255, 255, 255, 255)

BACK_COLOR = COLOR_BLACK
INFO_BACK_COLOR = COLOR_GREY
AGENT_COLOR = COLOR_MAGENTA
STATIC_OBSTACLE_COLOR = COLOR_GREY
MOVING_OBSTACLE_COLOR = COLOR_YELLOW
STATIC_ZONE_COLOR = COLOR_TURQUOISE
MOVING_ZONE_COLOR = COLOR_BLUE
BORDER_COLOR = COLOR_WHITE
ACTION_COLOR = COLOR_GREEN
OBSERVATION_COLOR = COLOR_YELLOW
INTERSECTION_COLOR = COLOR_RED

TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800  # pixels
PPM = 10  # pixel per meter

# screen layout
INFO_HEIGHT = SCREEN_HEIGHT * 0.2  # pixels
INFO_FONT_SIZE = 18

DESIGN_SLEEP = 0.01 # delay in seconds while designing world

# world
WORLD_WIDTH = SCREEN_WIDTH / PPM  # meters
WORLD_HEIGHT = (SCREEN_HEIGHT - INFO_HEIGHT) / PPM

BOUNDARIES_WIDTH = 10  # meters

# velocity and position iterations
# higher values improve precision in
# physics simulations
VEL_ITER = 6
POS_ITER = 2

# action space
MIN_ANGLE = -math.pi * 3 / 4  # radians
MAX_ANGLE = math.pi * 3 / 4
# MIN_ANGLE = 0  # radians
# MAX_ANGLE = 2*math.pi

MIN_FORCE = 0  # newtons
MAX_FORCE = 10

# observation space
OBSERVATION_RANGE = math.pi - math.pi/4  # 2*math.pi
OBSERVATION_MAX_DISTANCE = 200  # how far can the agent see
OBSERVATION_NUM = 10  # number of distance vectors

# agent frictions
AGENT_ANGULAR_DAMPING = 2
AGENT_LINEAR_DAMPING = 0.5

# corrections
AGENT_HEAD_INSIDE = 0.2

# objects
AGENT_MASS = 0.2  # kg
MOVING_OBSTACLE_DENSITY = 1000000000  # kg/(m*m)


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
    color: tuple = COLOR_WHITE
    shape: Enum = BodyShape.BOX
    # list of bodies in contact with this body
    contact_bodies: List = field(default_factory=list)
    reward: float = 0  # reward when agents hits the object
    # level of deepness when drawing screen (0 is above everything else)
    # if multiple object share same level, first created objects are below others
    level: int = 0


@dataclass
class Observation:
    valid: bool = False
    index: int = -1
    angle: float = 0
    intersection: b2Vec2 = (np.inf, np.inf)
    distance: float = np.inf
    body: b2Body = None

@dataclass
class DesignShapeData:
    shape: Enum = BodyShape.BOX
    points: List = field(default_factory=list) # center and radius for circles
    type: Enum = BodyType.STATIC_OBSTACLE
    color: tuple = STATIC_OBSTACLE_COLOR


class ContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)
        self.contact_bodies = list()

    def BeginContact(self, contact):
        contact.fixtureA.body.userData.contact_bodies.append(
            contact.fixtureB.body)
        contact.fixtureB.body.userData.contact_bodies.append(
            contact.fixtureA.body)
        self.contact_bodies.append(
            {"bodyA": contact.fixtureA.body, "bodyB": contact.fixtureB.body})

        pass

    def EndContact(self, contact):
        for c in self.contact_bodies:
            if c["bodyA"] == contact.fixtureA.body and c["bodyB"] == contact.fixtureB.body:
                contact.fixtureA.body.userData.contact_bodies.remove(
                    contact.fixtureB.body)
                contact.fixtureB.body.userData.contact_bodies.remove(
                    contact.fixtureA.body)
                self.contact_bodies.remove(
                    {"bodyA": contact.fixtureA.body, "bodyB": contact.fixtureB.body})
                break

        pass

    def PreSolve(self, contact, oldMainfold):
        pass

    def PostSolve(self, contact, impulse):
        pass


class RayCastClosestCallback(b2RayCastCallback):
    """This callback finds the closest hit"""

    def __repr__(self):
        return 'Closest hit'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        '''
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        '''
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        return fraction


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

        self.user_action = None
        self.manual_mode = False

        # TODO: inside_zone
        # the observation spaces is defined by observation_keys
        self.observation_keys = ["distances",
                                 "body_types", "position", "inside_zone", "body_velocities", "linear_velocity", "velocity_mag"]

        self.observation_space = gym.spaces.Dict(self.__get_observation_dict())

        # it's like the observation space but with more informations
        self.data = list()  # list of Observation dataclasses

        # pygame setup
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Box Dynamics')
        self.clock = pygame.time.Clock()
        pygame.font.init()  # to render text

        # adding world borders
        self.__create_borders()

        # adding dynamic body for RL agent
        # TODO: support agent parameters from ouside class
        self.__create_agent()

        # defining own polygon draw function
        # b2PolygonShape.draw = self.__draw_polygon

        self.prev_state = None

        self.screen_infos = list()
        self.design_shapes = list()

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
        self.action = b2Vec2(0, 0)

        if action is not None:
            # calculating where to apply force
            # target -> "head" of agent
            self.action = b2Vec2(action[1] *
                                 math.cos(action[0] + self.agent_body.angle),
                                 action[1] *
                                 math.sin(action[0] + self.agent_body.angle))

        # can override action
        done = self.__user_input()

        self.agent_body.ApplyForce(
            force=self.action, point=self.agent_head, wake=True)

        # Make Box2D simulate the physics of our world for one step.
        self.world.Step(TIME_STEP, VEL_ITER, POS_ITER)

        # clear forces or they will stay permanently
        self.world.ClearForces()

        self.state = self.__get_observations()

        assert self.state in self.observation_space

        # TODO: rewards and effects
        self.__contact_effects()

        self.prev_state = self.state

        step_reward = 0

        self.__get_screen_infos()

        self.render()

        info = {}

        return self.state, step_reward, done, info

    def __contact_effects(self):
        pass

    def render(self):
        # background for render screen
        self.screen.fill(BACK_COLOR)

        # drawing bodies based on their level
        self.__draw_bodies(BodyType)

        # drawing agent actions
        self.__draw_action()

        # drawing agent observations
        self.__draw_observations()

        # drawing text distances
        self.__draw_distances()

        # last for a reason, drawing infos
        self.__draw_simulation_infos()

        pygame.display.flip()
        self.clock.tick(TARGET_FPS)

    def __destroy(self):
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        pygame.quit()

    def box_design(self):
        first_set = False
        second_set = False

        box = DesignShapeData()

        box.points = [b2Vec2(0, 0) for _ in range(4)]
        self.design_shapes.append(box)

        while not (first_set and second_set):
            # drawing borders and info box
            self.__render_design()

            mouse_pos = b2Vec2(pygame.mouse.get_pos())

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if first_set:
                        second_set = True
                        end_pos = mouse_pos
                    else:
                        first_set = True
                        start_pos = mouse_pos

            self.design_shapes.remove(box)
            # only update box from here

            if first_set:
                box.points[0] = start_pos
                box.points[1] = b2Vec2(mouse_pos.x, start_pos.y)
                box.points[2] = mouse_pos
                box.points[3] = b2Vec2(start_pos.x, mouse_pos.y)
            if first_set and second_set:
                box.points[0] = start_pos
                box.points[1] = b2Vec2(end_pos.x, start_pos.y)
                box.points[2] = end_pos
                box.points[3] = b2Vec2(start_pos.x, end_pos.y)

            # only update box until here
            self.design_shapes.append(box)

            sleep(DESIGN_SLEEP)

    def world_design(self):
        circle_mode = False

        finished = False

        while not finished:
            # drawing borders, infos and currently designed shapes
            self.__render_design()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                        (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    # The user closed the window or pressed escape
                    self.__destroy()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.box_design()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    circle_mode = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    finished = True

            sleep(DESIGN_SLEEP)
        
        self.__create_from_design_shapes()
        pass

    def __create_from_design_shapes(self):
        for shape in self.design_shapes:
            points = shape.points

            pos = b2Vec2(points[0].x + (points[1].x - points[0].x) / 2, points[0].y + (points[3].y - points[0].y) / 2)
            size = b2Vec2(abs(points[1].x - points[0].x) / 2, abs(points[3].y - points[0].y) / 2)

            pos = self.__world_coord(pos)
            size = size / PPM
            self.create_static_obstacle(pos, size)

    def __user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                # The user closed the window or pressed escape
                self.__destroy()
                return True
            # TODO: add commands description
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.manual_mode = not self.manual_mode
        if self.manual_mode:
            mouse_pos = b2Vec2(pygame.mouse.get_pos())
            self.action = self.__world_coord(
                mouse_pos) - self.agent_body.position
        return False

    # returns a dictionary which can then be converted to a gym.spaces.Dict
    # defines min, max, shape and of each observation key
    def __get_observation_dict(self):
        observation_dict = dict()

        for key in self.observation_keys:
            partial_dict = dict()
            if key == "distances":
                partial_dict = ({"distances": gym.spaces.Box(
                    low=0, high=np.inf, shape=(OBSERVATION_NUM,))})
            elif key == "body_types":
                partial_dict = ({"body_types": gym.spaces.Tuple(
                    ([gym.spaces.Discrete(len(BodyType))]*OBSERVATION_NUM))})
            elif key == "body_velocities":
                partial_dict = ({"body_velocities": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(OBSERVATION_NUM, 2, ))})
            elif key == "position":
                partial_dict = ({"position": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,))})
            elif key == "linear_velocity":
                partial_dict = ({"linear_velocity": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,))})
            elif key == "velocity_mag":
                partial_dict = ({"velocity_mag": gym.spaces.Box(
                    low=0, high=np.inf, shape=(1,))})
            elif key == "inside_zone":
                # partial_dict = ({"inside_zone": gym.spaces.Tuple(
                #     ([gym.spaces.Discrete(2)] * ZONES_NUM))})
                partial_dict = ({"inside_zone": gym.spaces.Discrete(2)})

            observation_dict.update(partial_dict)

        return observation_dict

    # returns a state wich is an "instance" of observation_space
    # here is defined how each observation key in the observation space
    # dict should be set
    def __set_observation_dict(self):
        state = dict()
        for key in self.observation_keys:
            if key == "distances":
                state["distances"] = np.array(
                    [observation.distance for observation in self.data], dtype=np.float32)
            elif key == "body_types":
                state["body_types"] = tuple()
                for observation in self.data:
                    if observation.valid:
                        state["body_types"] += (
                            observation.body.userData.type.value,)
                    else:
                        state["body_types"] += (BodyType.DEFAULT.value,)
            elif key == "body_velocities":
                state["body_velocities"] = list()
                for observation in self.data:
                    if observation.valid:
                        state["body_velocities"].append(
                            observation.body.linearVelocity)
                    else:
                        state["body_velocities"].append(b2Vec2(0, 0))
                state["body_velocities"] = np.array(
                    state["body_velocities"], dtype=np.float32)
            elif key == "position":
                state["position"] = np.array(
                    self.agent_body.position, dtype=np.float32)
            elif key == "linear_velocity":
                state["linear_velocity"] = np.array(
                    self.agent_body.linearVelocity, dtype=np.float32)
            elif key == "velocity_mag":
                state["velocity_mag"] = np.array(
                    [self.agent_body.linearVelocity.length], dtype=np.float32)
            elif key == "inside_zone":
                state["inside_zone"] = False

        return state

    def __get_observations(self):
        self.data.clear()

        for delta_angle in range(OBSERVATION_NUM):
            # absolute angle for the observation vector
            # based on OBSERVATION_RANGE
            angle = self.__get_observation_angle(delta_angle)

            observation_end = (self.agent_head.x + math.cos(angle) * OBSERVATION_MAX_DISTANCE,
                               self.agent_head.y + math.sin(angle) * OBSERVATION_MAX_DISTANCE)

            callback = RayCastClosestCallback()

            self.world.RayCast(callback, self.agent_head, observation_end)

            observation = Observation()
            observation.valid = False
            observation.index = delta_angle
            observation.angle = angle
            if callback.hit:
                observation.valid = True
                if hasattr(callback, "point"):
                    observation.intersection = callback.point
                    observation.distance = self.__euclidean_distance(
                        self.agent_head, callback.point)
                else:
                    observation.valid = False

                if hasattr(callback, "fixture"):
                    observation.body = callback.fixture.body
                else:
                    observation.valid = False

            self.data.append(observation)

        # filter info and return observation space
        state = self.__set_observation_dict()

        return state

    def __get_screen_infos(self):
        self.screen_infos.clear()

        self.screen_infos.append({"name": "agent position", "value": (
            round(self.agent_body.position.x, 1), round(self.agent_body.position.y, 1))})
        self.screen_infos.append({"name": "agent velocity magnitude", "value": round(
            self.agent_body.linearVelocity.length, 1)})
        self.screen_infos.append(
            {"name": "agent angular velocity", "value": round(self.agent_body.angularVelocity, 1)})

        for cix, agent_contact in enumerate(self.agent_body.userData.contact_bodies):
            self.screen_infos.append({"name": "contact {}".format(
                cix), "value": agent_contact.userData.type})

    def __update_agent_head(self):
        # TODO: define agent head types and let the user choose
        # TODO: use AGENT_HEAD_INSIDE only of needed aka
        # use it only if agent head is on agent edges
        self.agent_head = b2Vec2(
            (self.agent_body.position.x + math.cos(
                self.agent_body.angle) * (self.agent_size.x - AGENT_HEAD_INSIDE)),
            (self.agent_body.position.y + math.sin(
                self.agent_body.angle) * (self.agent_size.x - AGENT_HEAD_INSIDE)))
        # self.agent_head = self.agent_body.position

    def __get_observation_angle(self, delta_angle):
        try:
            return self.agent_body.angle - (OBSERVATION_RANGE / 2) + (OBSERVATION_RANGE / (OBSERVATION_NUM - 1) * delta_angle)
        except ZeroDivisionError:
            return self.agent_body.angle

    def __create_borders(self):
        inside = 0.5  # defines how much of the borders are visible

        # TODO: add fixtures (?)

        self.bottom_border = self.world.CreateStaticBody(
            position=(WORLD_WIDTH / 2, inside -
                      (BOUNDARIES_WIDTH / 2) + INFO_HEIGHT / PPM),
            shapes=b2PolygonShape(
                box=(WORLD_WIDTH / 2 + BOUNDARIES_WIDTH, BOUNDARIES_WIDTH / 2)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, level=1)
        )

        self.top_border = self.world.CreateStaticBody(
            position=(WORLD_WIDTH / 2, WORLD_HEIGHT + (INFO_HEIGHT / PPM) -
                      inside + (BOUNDARIES_WIDTH / 2)),
            shapes=b2PolygonShape(
                box=(WORLD_WIDTH / 2 + BOUNDARIES_WIDTH, BOUNDARIES_WIDTH / 2)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, level=1)
        )

        self.left_border = self.world.CreateStaticBody(
            position=(inside - (BOUNDARIES_WIDTH / 2),
                      WORLD_HEIGHT / 2 + (INFO_HEIGHT / PPM)),
            shapes=b2PolygonShape(
                box=(BOUNDARIES_WIDTH / 2, WORLD_HEIGHT / 2 + BOUNDARIES_WIDTH)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, level=1)
        )
        self.right_border = self.world.CreateStaticBody(
            position=(WORLD_WIDTH - inside +
                      (BOUNDARIES_WIDTH / 2), WORLD_HEIGHT / 2 + (INFO_HEIGHT / PPM)),
            shapes=b2PolygonShape(
                box=(BOUNDARIES_WIDTH / 2, WORLD_HEIGHT / 2 + BOUNDARIES_WIDTH)),
            userData=BodyData(type=BodyType.BORDER,
                              color=BORDER_COLOR, level=1)
        )

    def __create_agent(self, agent_size=(1, 1), agent_pos=None, agent_angle=None):
        agent_width, agent_height = agent_size
        self.agent_size = b2Vec2(agent_width, agent_height)

        # setting random initial position
        if agent_pos is None:
            r = self.agent_size.length
            try:
                x = random.randint(int(r), int(WORLD_WIDTH - r))
                y = random.randint(int(r + (INFO_HEIGHT / PPM)),
                                   int(WORLD_HEIGHT + (INFO_HEIGHT / PPM) - r))
            except ValueError:
                assert False and "There is no space to spawn the agent, modify world sizes"
            agent_pos = b2Vec2(x, y)

        # setting random initial angle
        if agent_angle is None:
            agent_angle = random.random() * (2*math.pi)

        area = agent_width * agent_height

        self.agent_body = self.world.CreateDynamicBody(
            position=agent_pos, angle=agent_angle)
        self.agent_fix = self.agent_body.CreatePolygonFixture(
            box=(agent_width, agent_height), density=AGENT_MASS/area)

        self.agent_body.userData = BodyData(
            type=BodyType.AGENT, color=AGENT_COLOR, level=0)

        self.agent_fix.body.angularDamping = AGENT_ANGULAR_DAMPING
        self.agent_fix.body.linearDamping = AGENT_LINEAR_DAMPING

    def create_static_obstacle(self, pos, size, angle=0, reward=-1, level=3):
        body = self.world.CreateStaticBody(
            position=pos, angle=angle
        )
        fixture = body.CreatePolygonFixture(
            box=size)

        body.userData = BodyData(
            type=BodyType.STATIC_OBSTACLE, color=STATIC_OBSTACLE_COLOR,
            reward=reward, level=level)

    def create_moving_obstacle(self, pos, size, velocity, angle=0, reward=-2, level=2):
        body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False, )
        _ = body.CreatePolygonFixture(
            box=size, density=MOVING_OBSTACLE_DENSITY)

        body.userData = BodyData(
            type=BodyType.MOVING_OBSTACLE, color=MOVING_OBSTACLE_COLOR,
            reward=reward, level=level)

    def create_static_zone(self, pos, size, angle=0, reward=1, level=3):
        body = self.world.CreateStaticBody(
            position=pos, angle=angle)
        fixture = body.CreatePolygonFixture(
            box=size)
        fixture.sensor = True

        body.userData = BodyData(
            type=BodyType.STATIC_ZONE, color=STATIC_ZONE_COLOR,
            reward=reward, level=level)

    def create_moving_zone(self, pos, size, velocity, angle=0, reward=2, level=1):
        body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False)
        fixture = body.CreatePolygonFixture(
            box=size)

        fixture.sensor = True

        body.userData = BodyData(
            type=BodyType.MOVING_ZONE, color=MOVING_ZONE_COLOR,
            reward=reward, level=level)

    # user functions
    def get_world_size(self):
        return WORLD_WIDTH, WORLD_HEIGHT

    # render functions
    def __render_design(self):
        # draws borders, design infos and currently designed shapes

        self.screen.fill(BACK_COLOR)
        self.__draw_bodies([BodyType.BORDER])
        self.__draw_design_infos()
        self.__draw_design_shapes()
        pygame.display.flip()

    def __draw_design_shapes(self):
        for shape in self.design_shapes:
            pygame.draw.polygon(self.screen, COLOR_GREEN, shape.points)

    def __draw_bodies(self, types):
        # Draw the world based on bodies levels
        bodies_levels = [[b, b.userData.level]
                         for b in self.world.bodies]
        bodies_levels.sort(key=lambda x: x[1], reverse=True)

        for body, level in bodies_levels:
            if body.userData.type in types:
                for fixture in body.fixtures:
                    vertices = [self.__pygame_coord(body.transform * v) for v in fixture.shape.vertices]
                    pygame.draw.polygon(self.screen, body.userData.color, vertices)
        pass

    def __draw_action(self):
        action_start = self.__pygame_coord(self.agent_head)
        action_end = self.__pygame_coord(self.agent_head + self.action)

        pygame.draw.line(self.screen, ACTION_COLOR, action_start, action_end)

    def __draw_distances(self):
        text_font = pygame.font.SysFont('Comic Sans MS', 16)

        # drawing distance text
        # TODO: only draw them if there is enough space
        freq = 1  # defines "distance text" quantity
        for oix, observation in enumerate(self.data):
            if oix % freq == 0 and observation.valid:
                distance = round(observation.distance, 1)

                text_point = self.__pygame_coord(
                    self.agent_head + (observation.intersection - self.agent_head) / 2)
                text_surface = text_font.render(
                    str(distance), False, COLOR_BLACK, COLOR_WHITE)
                self.screen.blit(text_surface, text_point)

    def __draw_observations(self):
        start_point = self.__pygame_coord(self.agent_head)
        for observation in self.data:
            if observation.valid:
                end_point = self.__pygame_coord(observation.intersection)

                # drawing observation vectors
                pygame.draw.line(self.screen, observation.body.userData.color,
                                 start_point, end_point)
                # drawing intersection points
                pygame.draw.circle(
                    self.screen, INTERSECTION_COLOR, end_point, 3)

    def __draw_infos_box(self):
        info_vertices = [(0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT), (
            SCREEN_WIDTH, SCREEN_HEIGHT - INFO_HEIGHT), (0, SCREEN_HEIGHT - INFO_HEIGHT)]
        pygame.draw.polygon(self.screen, INFO_BACK_COLOR, info_vertices)

    def __draw_design_infos(self):
        self.__draw_infos_box()

        text_font = pygame.font.SysFont('Comic Sans MS', INFO_FONT_SIZE)

        shapes = [{"key": "R", "description": "create rectangle"}, {"key": "C", "description": "create circle"}]

        max_len = 0
        for six, shape in enumerate(shapes):
            s = "{}: {}".format(shape["key"], shape["description"])
            text_surface = text_font.render(s, True, COLOR_BLACK, COLOR_WHITE)

            pos, max_len = self.__get_info_pos(six, len(s), max_len)
            self.screen.blit(text_surface, pos)

    def __draw_simulation_infos(self):
        self.__draw_infos_box()

        text_font = pygame.font.SysFont('Comic Sans MS', INFO_FONT_SIZE)

        # fps
        fps = round(self.clock.get_fps())
        fps_point = (0, 0)
        fps_color = COLOR_GREEN if abs(fps - TARGET_FPS) < 10 else COLOR_RED
        text_surface = text_font.render(
            str(fps), True, COLOR_BLACK, fps_color)
        self.screen.blit(text_surface, fps_point)

        max_len = 0
        for iix, info in enumerate(self.screen_infos):
            info_str = "> {}: {}".format(info["name"], info["value"])
            text_surface = text_font.render(
                info_str, True, COLOR_BLACK, COLOR_WHITE
            )

            pos, max_len = self.__get_info_pos(iix, len(info_str), max_len)

            self.screen.blit(text_surface, pos)

    def __get_info_pos(self, str_ix, str_len, max_str_len):
        border = 10  # info border pixels
        # first info coordinate
        info_coord = b2Vec2(border, SCREEN_HEIGHT - INFO_HEIGHT + border)
        y_space = INFO_HEIGHT - border*2  # available vertical space
        y_inc = 20  # vertical space between infos
        x_inc = 0
        y_slots = y_space / y_inc

        pos_inc = b2Vec2(x_inc, (str_ix % y_slots)*y_inc)
        pos = info_coord + pos_inc

        if str_len > max_str_len:
            max_str_len = str_len
        if (str_ix + 1) % y_slots == 0:
            x_inc += max_str_len * 7
            max_str_len = 0
        if x_inc + max_str_len > SCREEN_WIDTH:
            warnings.warn(
                "Some infos are not visible, increase INFO_HEIGHT")

        return pos, max_str_len

    # transform point in world coordinates to point in pygame coordinates
    def __pygame_coord(self, point):
        return b2Vec2(point.x * PPM, SCREEN_HEIGHT - (point.y * PPM))

    def __world_coord(self, point):
        return b2Vec2(point.x / PPM, (SCREEN_HEIGHT - point.y) / PPM)

    # utilities functions
    def __euclidean_distance(self, point1, point2):
        return (point1 - point2).length
