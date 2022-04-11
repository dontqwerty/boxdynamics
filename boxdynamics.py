import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Dict, List

import gym
import numpy as np
import pygame as pg
from Box2D import (b2Body, b2Fixture, b2PolygonShape, b2Vec2, b2World)

import boxcolors as color
from boxcontacts import ContactListener
from boxdef import BodyShape, BodyType, EffectType
from boxraycast import RayCastClosestCallback
from boxui import BoxUI, DesignData, Mode, ScreenLayout
from boxutils import get_intersection, get_line_eq


@dataclass
class EnvConfig:
    target_fps: float = 60
    time_step: float = 1.0 / target_fps
    ppm: float = 10  # pixel per meter

    # world
    boundaries_width: float = 10  # meters

    # velocity and position iterations
    # higher values improve precision in
    # physics simulations
    vel_iter: int = 6
    pos_iter: int = 2

    # action space
    min_angle: float = -math.pi * 3 / 4  # radians
    max_angle: float = math.pi * 3 / 4

    min_force: float = 0  # newtons
    max_force: float = 8

    # observation space
    observation_range: float = math.pi - math.pi/8  # 2*math.pi
    observation_max_distance: float = 1000  # how far can the agent see
    observation_num: int = 20  # number of distance vectors

    # agent frictions
    agent_angular_damping: float = 2
    agent_linear_damping: float = 0.1

    # corrections
    agent_head_inside: float = 0.2

    # objects
    agent_mass: float = 0.2  # kg


@dataclass
class BodyData:
    type: IntEnum = BodyType.DEFAULT
    color: tuple = color.WHITE
    shape: IntEnum = BodyShape.BOX
    # list of bodies in contact with this body
    contacts: List[b2Body] = field(default_factory=list)
    reward: float = 0  # reward when agents hits the object
    level: int = 0
    effect: Dict = field(default_factory=dict)


@dataclass
class Observation:
    valid: bool = False
    index: int = -1
    angle: float = 0
    intersection: b2Vec2 = (np.inf, np.inf)
    distance: float = np.inf
    body: b2Body = None


class BoxEnv(gym.Env):
    def __init__(self) -> None:
        # initializing base class
        super(BoxEnv, self).__init__()

        self.conf = EnvConfig()

        # world keeps track of objects and physics
        self.world = b2World(gravity=(0, 0), doSleep=True,
                             contactListener=ContactListener(self))

        # action space
        # relative agent_angle, force
        self.action_space = gym.spaces.Box(
            np.array([self.conf.min_angle, self.conf.min_force]
                     ).astype(np.float32),
            np.array([self.conf.max_angle, self.conf.max_force]
                     ).astype(np.float32)
        )

        # to signal that the training is done
        self.done = False

        self.user_action = None
        self.manual_mode = False

        # TODO: contacts
        # the observation spaces is defined by observation_keys
        self.observation_keys = ["distances",
                                 "body_types", "position", "contacts", "body_velocities", "linear_velocity", "velocity_mag"]

        self.observation_keys = ["distances"]

        self.observation_space = gym.spaces.Dict(self.__get_observation_dict())

        self.screen_layout = ScreenLayout()
        self.ui = BoxUI(self, self.screen_layout,
                        self.conf.ppm, self.conf.target_fps)

        self.screen_width = self.screen_layout.width
        self.screen_height = self.screen_layout.height

        self.world_width = self.screen_layout.simulation_size.x / self.conf.ppm
        self.world_height = self.screen_layout.simulation_size.y / self.conf.ppm
        self.world_pos = b2Vec2(self.screen_layout.simulation_pos.x, self.screen_height -
                                self.screen_layout.simulation_pos.y) / self.conf.ppm - b2Vec2(0, self.world_height)

        # it's like the observation space but with more informations
        # list of Observation dataclasses
        self.data: List[Observation] = list()

        # creates world borders and agent
        self.create_world()

        self.prev_state = None

    def reset(self):
        # resetting base class
        super().reset()

        # resetting reward
        # TODO: give reward to agent
        self.reward = 0.0

        # returning state after step with None action
        return self.step(None)[0]

    def step(self, action) -> tuple:
        # calculate agent head point
        # TODO: support circles
        # TODO: support agent_head definition from outside class
        self.__update_agent_head()

        self.perform_action(action)

        # Make Box2D simulate the physics of our world for one step.
        self.world.Step(self.conf.time_step,
                        self.conf.vel_iter, self.conf.pos_iter)

        # clear forces or they will stay permanently
        self.world.ClearForces()

        # current and previous state
        self.state = self.__get_observations()
        assert self.state in self.observation_space
        self.prev_state = self.state

        # agent rewards
        step_reward = self.set_reward()

        info = {}

        done = self.done

        self.render()
        self.ui.user_input()

        return self.state, step_reward, done, info

    def render(self):
        self.ui.render()

    def destroy(self):
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        self.ui.quit()

    def save_conf(self, filename="config.json"):
        with open(filename, "w") as f:
            json.dump(asdict(self.conf), f)
        pass

    def load_conf(self, filename="config.json"):
        with open(filename, "r") as f:
            self.conf = EnvConfig(**json.load(f))
        pass

    def set_reward(self):
        step_reward = 0
        for agent_contacts in self.agent_body.userData.contacts:
            step_reward += agent_contacts.userData.reward
        return step_reward

    def perform_action(self, action):
        self.action = b2Vec2(0, 0)
        if self.manual_mode:
            mouse_pos = b2Vec2(pg.mouse.get_pos())
            self.action = self.__world_coord(
                mouse_pos) - self.agent_body.position
        elif action is not None:
            # calculating where to apply force
            # target is "head" of agent
            self.action = b2Vec2(action[1] *
                                 math.cos(action[0] + self.agent_body.angle),
                                 action[1] *
                                 math.sin(action[0] + self.agent_body.angle))

        self.agent_body.ApplyForce(
            force=self.action, point=self.agent_head, wake=True)

    def world_design(self):
        self.ui.set_mode(Mode.RESIZE)
        while self.ui.mode != Mode.SIMULATION:
            self.ui.user_input()
            self.ui.ui_sleep()
            self.render()
        self.__create_from_design_data()

    def __create_from_design_data(self):
        for design in self.ui.design_bodies:
            points = [self.__world_coord(point) for point in design.vertices]
            try:
                line1 = get_line_eq(points[0], points[2])
                line2 = get_line_eq(points[1], points[3])
            except IndexError:
                # the body is a single point (?)
                continue
            pos = get_intersection(line1, line2)
            width = (points[0] - points[1]).length / 2
            height = (points[1] - points[2]).length / 2
            size = (width, height)
            angle = -design.delta_angle

            self.__create_body(pos, size, angle, design)

        self.ui.design_bodies.clear()

    def set_body_params(self, body: b2Body, design_data: DesignData):
        body.angularDamping = design_data.params["ang_damping"]
        body.angularVelocity = design_data.params["ang_velocity"]
        body.linearDamping = design_data.params["lin_damping"]
        design_data.params["lin_velocity_angle"] = design_data.params["lin_velocity_angle"] * (
            2 * math.pi / 360)
        body.linearVelocity = b2Vec2(math.cos(design_data.params["lin_velocity_angle"]), math.sin(
            design_data.params["lin_velocity_angle"])) * design_data.params["lin_velocity"]
        body.inertia = design_data.params["inertia"]
        body.fixtures[0].density = design_data.params["density"]
        body.fixtures[0].friction = design_data.params["friction"]

    def __create_body(self, pos, size, angle, design_data: DesignData):
        type = design_data.params["type"]
        if type == BodyType.STATIC_OBSTACLE:
            body = self.create_static_obstacle(pos, size, angle=angle)
        elif type == BodyType.MOVING_OBSTACLE:
            body = self.create_moving_obstacle(pos, size, angle=angle)
            self.set_body_params(body, design_data)
        elif type == BodyType.STATIC_ZONE:
            body = self.create_static_zone(pos, size, angle=angle)
        elif type == BodyType.MOVING_ZONE:
            body = self.create_moving_zone(pos, size, angle=angle)
            self.set_body_params(body, design_data)

        body.userData.reward = design_data.params["reward"]
        body.userData.level = design_data.params["level"]

        body.userData.effect = design_data.effect.copy()

    # returns a dictionary which can then be converted to a gym.spaces.Dict
    # defines min, max, shape and of each observation key
    def __get_observation_dict(self):
        observation_dict = dict()

        for key in self.observation_keys:
            partial_dict = dict()
            if key == "distances":
                partial_dict = ({"distances": gym.spaces.Box(
                    low=0, high=np.inf, shape=(self.conf.observation_num,))})
            elif key == "body_types":
                partial_dict = ({"body_types": gym.spaces.Tuple(
                    ([gym.spaces.Discrete(len(BodyType))]*self.conf.observation_num))})
            elif key == "body_velocities":
                partial_dict = ({"body_velocities": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.conf.observation_num, 2, ))})
            elif key == "position":
                partial_dict = ({"position": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,))})
            elif key == "linear_velocity":
                partial_dict = ({"linear_velocity": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,))})
            elif key == "velocity_mag":
                partial_dict = ({"velocity_mag": gym.spaces.Box(
                    low=0, high=np.inf, shape=(1,))})
            elif key == "contacts":
                # partial_dict = ({"contacts": gym.spaces.Tuple(
                #     ([gym.spaces.Discrete(2)] * ZONES_NUM))})
                partial_dict = ({"contacts": gym.spaces.Discrete(2)})

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
            elif key == "contacts":
                state["contacts"] = False

        return state

    def __get_observations(self):
        self.data.clear()

        for delta_angle in range(self.conf.observation_num):
            # absolute angle for the observation vector
            # based on self.conf.observation_range
            angle = self.__get_observation_angle(delta_angle)

            observation_end = (self.agent_head.x + math.cos(angle) * self.conf.observation_max_distance,
                               self.agent_head.y + math.sin(angle) * self.conf.observation_max_distance)

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
                    observation.distance = (
                        self.agent_head - callback.point).length
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

    def __update_agent_head(self):
        # TODO: define agent head types and let the user choose
        # TODO: use self.conf.agent_head_inside only of needed aka
        # use it only if agent head is on agent edges
        self.agent_head = b2Vec2(
            (self.agent_body.position.x + math.cos(
                self.agent_body.angle) * (self.agent_size.x - self.conf.agent_head_inside)),
            (self.agent_body.position.y + math.sin(
                self.agent_body.angle) * (self.agent_size.x - self.conf.agent_head_inside)))
        # self.agent_head = self.agent_body.position

    def __get_observation_angle(self, delta_angle):
        try:
            return self.agent_body.angle - (self.conf.observation_range / 2) + (self.conf.observation_range / (self.conf.observation_num - 1) * delta_angle)
        except ZeroDivisionError:
            return self.agent_body.angle

    def create_world(self):
        # adding world borders
        self.__create_borders()

        # adding dynamic body for RL agent
        # TODO: support agent parameters from ouside class
        self.__create_agent()

    def __create_borders(self):
        inside = 0.0  # defines how much of the borders are visible

        # TODO: border reward

        pos = b2Vec2(self.world_width / 2, inside -
                     (self.conf.boundaries_width / 2))
        self.bottom_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.world_width / 2 + self.conf.boundaries_width, self.conf.boundaries_width / 2)),
            userData=self.get_border_data()
        )

        pos = b2Vec2(self.world_width / 2, self.world_height -
                     inside + (self.conf.boundaries_width / 2))
        self.top_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.world_width / 2 + self.conf.boundaries_width, self.conf.boundaries_width / 2)),
            userData=self.get_border_data()
        )

        pos = b2Vec2(inside - (self.conf.boundaries_width / 2),
                     self.world_height / 2)
        self.left_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.conf.boundaries_width / 2, self.world_height / 2 + self.conf.boundaries_width)),
            userData=self.get_border_data()
        )

        pos = b2Vec2(self.world_width - inside +
                     (self.conf.boundaries_width / 2), self.world_height / 2)
        self.right_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.conf.boundaries_width / 2, self.world_height / 2 + self.conf.boundaries_width)),
            userData=self.get_border_data()
        )

    def get_data(self, type: BodyType):
        if type == BodyType.BORDER:
            return self.get_border_data()
        elif type == BodyType.AGENT:
            return self.get_agent_data()
        elif type == BodyType.STATIC_OBSTACLE:
            return self.get_static_obstacle_data()
        elif type == BodyType.MOVING_OBSTACLE:
            return self.get_moving_obstacle_data()
        elif type == BodyType.STATIC_ZONE:
            return self.get_static_zone_data()
        elif type == BodyType.MOVING_ZONE:
            return self.get_moving_zone_data()
        else:
            # TODO: explain
            assert False

    def get_border_data(self) -> BodyData:
        # todo border effect
        effect = {"type": EffectType.APPLY_FORCE, "value": [100, 100]}
        return BodyData(type=BodyType.BORDER, color=color.BORDER, effect=effect)

    def get_agent_data(self) -> BodyData:
        return BodyData(type=BodyType.AGENT, color=color.AGENT, level=np.inf)

    def get_static_obstacle_data(self) -> BodyData:
        return BodyData(type=BodyType.STATIC_OBSTACLE, color=color.STATIC_OBSTACLE)

    def get_moving_obstacle_data(self) -> BodyData:
        return BodyData(
            type=BodyType.MOVING_OBSTACLE, color=color.MOVING_OBSTACLE)

    def get_static_zone_data(self) -> BodyData:
        return BodyData(
            type=BodyType.STATIC_ZONE, color=color.STATIC_ZONE)

    def get_moving_zone_data(self) -> BodyData:
        return BodyData(
            type=BodyType.MOVING_ZONE, color=color.MOVING_ZONE)

    def __create_agent(self, agent_size=(1, 1), agent_pos=None, agent_angle=None):
        agent_width, agent_height = agent_size
        self.agent_size = b2Vec2(agent_width, agent_height)

        # setting random initial position
        if agent_pos is None:
            r = self.agent_size.length
            try:
                x = random.randint(int(r), int(self.world_width - r))
                y = random.randint(int(r),
                                   int(self.world_height - r))
            except ValueError:
                assert False and "There is no space to spawn the agent, modify world sizes"
            agent_pos = b2Vec2(x, y)

        # setting random initial angle
        if agent_angle is None:
            agent_angle = random.random() * (2*math.pi)

        area = agent_width * agent_height

        self.agent_body: b2Body = self.world.CreateDynamicBody(
            position=agent_pos, angle=agent_angle)
        self.agent_fix: b2Fixture = self.agent_body.CreatePolygonFixture(
            box=(agent_width, agent_height), density=self.conf.agent_mass/area)

        self.agent_body.userData = self.get_agent_data()

        self.agent_fix.body.angularDamping = self.conf.agent_angular_damping
        self.agent_fix.body.linearDamping = self.conf.agent_linear_damping

    # TODO: support colors, circles
    def create_static_obstacle(self, pos, size, angle=0):
        body: b2Body = self.world.CreateStaticBody(
            position=pos, angle=angle
        )
        fixture: b2Fixture = body.CreatePolygonFixture(
            box=size)

        body.userData = self.get_static_obstacle_data()
        return body

    def create_moving_obstacle(self, pos, size, velocity=b2Vec2(1, 1), angle=0):
        body: b2Body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False, )
        _: b2Fixture = body.CreatePolygonFixture(
            box=size, density=1)

        body.userData = self.get_moving_obstacle_data()
        return body

    def create_static_zone(self, pos, size, angle=0):
        body: b2Body = self.world.CreateStaticBody(
            position=pos, angle=angle)
        fixture: b2Fixture = body.CreatePolygonFixture(
            box=size)
        fixture.sensor = True

        body.userData = self.get_static_zone_data()
        return body

    def create_moving_zone(self, pos, size, velocity=b2Vec2(1, 1), angle=0):
        body: b2Body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False)
        fixture: b2Fixture = body.CreatePolygonFixture(
            box=size)

        fixture.sensor = True

        body.userData = self.get_moving_zone_data()
        return body

    # user functions
    def get_world_size(self) -> tuple:
        return self.world_width, self.world_height

    def __world_coord(self, point: b2Vec2) -> b2Vec2:
        return b2Vec2(point.x / self.conf.ppm, (self.screen_height - point.y) / self.conf.ppm) - self.world_pos
