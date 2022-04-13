import json
import math
import random
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from logging import info, warn
from typing import Dict, List

import gym
import numpy as np
import pygame as pg
from Box2D import b2Body, b2Fixture, b2PolygonShape, b2Vec2, b2World

import boxcolors as color
from boxcontacts import ContactListener
from boxdef import (BodyShape, BodyType, DesignData, EffectType, ScreenLayout,
                    UIMode)
from boxraycast import RayCastClosestCallback
from boxui import BoxUI
from boxutils import get_intersection, get_line_eq


class AgentHeadType(IntEnum):
    CENTER = 0
    EDGE = 1


@dataclass
class AgentCfg:
    size: tuple = (1, 1)
    pos: tuple = None  # if None then random
    angle: float = None  # if None then random

    density: float = 0.2  # kg

    # agent frictions
    ang_damp: float = 2
    lin_damp: float = 0.1
    friction: float = 0.0

    # head
    head_type: AgentHeadType = AgentHeadType.EDGE

    # corrections
    head_inside: float = 0.2

    # action space
    min_angle: float = -math.pi * 3 / 4  # radians
    max_angle: float = math.pi * 3 / 4

    min_force: float = 0  # newtons
    max_force: float = 8

    # observation space
    obs_range: float = math.pi - math.pi/8  # 2*math.pi
    obs_max_dist: float = 1000  # how far can the agent see
    obs_num: int = 20  # number of distance vectors
    # TODO: set obs keys function
    obs_keys: List = field(default_factory=list)


@dataclass
class EnvCfg:
    target_fps: float = 60
    time_step: float = 1.0 / target_fps
    ppm: float = 10  # pixel per meter

    screen_layout: ScreenLayout = ScreenLayout()

    world_width: float = screen_layout.simulation_size.x / ppm
    world_height: float = screen_layout.simulation_size.y / ppm
    world_pos: b2Vec2 = b2Vec2(screen_layout.simulation_pos.x, screen_layout.height -
                               screen_layout.simulation_pos.y) / ppm - b2Vec2(0, world_height)

    # world
    create_borders: bool = True
    borders_width: float = 10  # meters
    border_inside: float = 0

    # velocity and position iterations
    # higher values improve precision in
    # physics simulations
    vel_iter: int = 6
    pos_iter: int = 2

    agent_cfg: AgentCfg = AgentCfg()
    design_bodies: List = field(default_factory=list)


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


# represents the data for each ray coming out of the agent
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

        # world keeps track of objects and physics
        self.world = b2World(gravity=(0, 0), doSleep=True,
                             contactListener=ContactListener(self))

        # setting configuration based on deafults
        # needed before using self.cfg
        self.default_cfg()

        # initializing UI
        self.ui = BoxUI(self, self.cfg.screen_layout,
                        self.cfg.ppm, self.cfg.target_fps)

        # action space
        # relative agent_angle, force
        self.action_space = gym.spaces.Box(
            np.array([self.cfg.agent_cfg.min_angle, self.cfg.agent_cfg.min_force]
                     ).astype(np.float32),
            np.array([self.cfg.agent_cfg.max_angle, self.cfg.agent_cfg.max_force]
                     ).astype(np.float32)
        )

        # to signal that the training is done
        self.done = False
        # to overwrite agent actions with user manual actions
        self.manual_mode = False
        self.agent_head = b2Vec2(0, 0)
        self.action = b2Vec2(0, 0)

        # defining observation spaces based on cfg.agent_cfg.obs_keys
        self.observation_space = gym.spaces.Dict(self.get_observation_dict())

        # list of Observation dataclasses used to set some
        # observations on each step
        self.data: List[Observation] = list()

        # creates world borders and agent
        self.create_world()

    def reset(self):
        # resetting base class
        super().reset()

        # resetting reward
        # TODO: give reward to agent
        self.reward = 0.0
        self.done = False
        self.manual_mode = False

        # returning state after step with None action
        return self.step(None)[0]

    def step(self, action) -> tuple:
        # calculate agent head point
        # TODO: support circles
        self.update_agent_head()

        self.perform_action(action)

        # Make Box2D simulate the physics of our world for one step.
        self.world.Step(self.cfg.time_step,
                        self.cfg.vel_iter, self.cfg.pos_iter)

        # clear forces or they will stay permanently
        self.world.ClearForces()

        # current and previous state
        self.state = self.get_observations()
        assert self.state in self.observation_space

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

    def default_cfg(self):
        self.cfg = EnvCfg()
        self.cfg.agent_cfg.obs_keys = ["distances", "body_types", "position",
                                       "contacts", "body_velocities", "linear_velocity", "velocity_mag"]
        pass

    def save_conf(self, filename="config.json"):
        print(self.ui.dataclass_to_dict(self.cfg))
        with open(filename, "w") as f:
            json.dump(self.ui.dataclass_to_dict(self.cfg), f)
        pass

    def load_conf(self, filename="config.json"):
        with open(filename, "r") as f:
            self.cfg = EnvCfg(**json.load(f))
        if isinstance(self.cfg.design_bodies, list):
            for bix, body in enumerate(self.cfg.design_bodies):
                if len(body["points"]) == 2:  # checking for valid body
                    self.cfg.design_bodies[bix] = DesignData(**body)
        # if there is only one design it will the above loop would iterate
        # through the keys instead of the designs
        elif isinstance(self.cfg.design_bodies, dict):
            if len(self.cfg.design_bodies["points"]) == 2:  # checking for valid body
                self.cfg.design_bodies[bix] = DesignData(**body)
        self.cfg.screen_layout = ScreenLayout(**self.cfg.screen_layout)
        self.cfg.agent_cfg = AgentCfg(**self.cfg.agent_cfg)

        self.create_bodies()
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
            self.action = self.world_coord(
                mouse_pos) - self.agent_body.position
        elif action is not None:
            # calculating where to apply force
            # target is "head" of agent
            self.action = b2Vec2(action[1] *
                                 math.cos(action[0] + self.agent_body.angle),
                                 action[1] *
                                 math.sin(action[0] + self.agent_body.angle))
        else:
            self.action = b2Vec2(0, 0)

        self.agent_body.ApplyForce(
            force=self.action, point=self.agent_head, wake=True)

    def create_world(self):
        if self.cfg.create_borders:
            # adding world borders
            self.create_borders()
        self.create_agent()

    # let's the user create the world interactively
    # when finished it copies the created world
    # inside the self.cfg.design
    def world_design(self):
        self.ui.set_mode(UIMode.RESIZE)
        while self.ui.mode != UIMode.SIMULATION:
            self.ui.user_input()
            self.ui.ui_sleep()
            self.render()

        self.cfg.design_bodies = self.ui.copy_design_bodies()
        self.create_bodies()

    def create_bodies(self):
        # creates all the bodies in the enviroment configuration
        if isinstance(self.cfg.design_bodies, list):
            for design in self.cfg.design_bodies:
                self.create_body(design)

    def create_body(self, design_data: DesignData):

        print(design_data.delta_angle)

        points = [self.world_coord(b2Vec2(point))
                  for point in design_data.vertices]
        try:
            line1 = get_line_eq(points[0], points[2])
            line2 = get_line_eq(points[1], points[3])
        except IndexError:
            # the body is a single point (?)
            warn("Can not create body from design {}".format(design_data))
            return
        pos = get_intersection(line1, line2)
        width = (points[0] - points[1]).length / 2
        height = (points[1] - points[2]).length / 2
        size = (width, height)
        angle = -design_data.delta_angle

        type = design_data.params["type"]
        if type == BodyType.AGENT:
            pass
        elif type == BodyType.BORDER:
            pass
        elif type == BodyType.STATIC_OBSTACLE:
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

    def create_borders(self):
        inside = self.cfg.border_inside  # defines how much of the borders are visible

        # TODO: border reward

        pos = b2Vec2(self.cfg.world_width / 2, inside -
                     (self.cfg.borders_width / 2))
        self.bottom_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.cfg.world_width / 2 + self.cfg.borders_width, self.cfg.borders_width / 2)),
            userData=self.get_border_data()
        )

        pos = b2Vec2(self.cfg.world_width / 2, self.cfg.world_height -
                     inside + (self.cfg.borders_width / 2))
        self.top_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.cfg.world_width / 2 + self.cfg.borders_width, self.cfg.borders_width / 2)),
            userData=self.get_border_data()
        )

        pos = b2Vec2(inside - (self.cfg.borders_width / 2),
                     self.cfg.world_height / 2)
        self.left_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.cfg.borders_width / 2, self.cfg.world_height / 2 + self.cfg.borders_width)),
            userData=self.get_border_data()
        )

        pos = b2Vec2(self.cfg.world_width - inside +
                     (self.cfg.borders_width / 2), self.cfg.world_height / 2)
        self.right_border: b2Body = self.world.CreateStaticBody(
            position=pos,
            shapes=b2PolygonShape(
                box=(self.cfg.borders_width / 2, self.cfg.world_height / 2 + self.cfg.borders_width)),
            userData=self.get_border_data()
        )

    def create_agent(self):
        agent_width, agent_height = self.cfg.agent_cfg.size
        self.agent_size = b2Vec2(agent_width, agent_height)
        agent_angle = self.cfg.agent_cfg.angle

        # setting random initial position
        if self.cfg.agent_cfg.pos is None:
            r = self.agent_size.length
            try:
                x = random.randint(int(r), int(
                    self.cfg.world_width - r))
                y = random.randint(int(r),
                                   int(self.cfg.world_height - r))
            except ValueError:
                assert False and "There is no space to spawn the agent, modify world sizes"
            agent_pos = b2Vec2(x, y)
        else:
            agent_pos = b2Vec2(self.cfg.agent_cfg.pos)

        # setting random initial angle
        if self.cfg.agent_cfg.angle is None:
            agent_angle = random.random() * (2*math.pi)

        self.agent_body: b2Body = self.world.CreateDynamicBody(
            position=agent_pos, angle=agent_angle)
        self.agent_fix: b2Fixture = self.agent_body.CreatePolygonFixture(
            box=(agent_width, agent_height), density=self.cfg.agent_cfg.density, friction=self.cfg.agent_cfg.friction)

        self.agent_body.userData = self.get_agent_data()

        self.agent_fix.body.angularDamping = self.cfg.agent_cfg.ang_damp
        self.agent_fix.body.linearDamping = self.cfg.agent_cfg.lin_damp

        info("Created agent")

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

    # returns a dictionary which can then be converted to a gym.spaces.Dict
    # defines min, max, shape and of each observation key
    def get_observation_dict(self):
        observation_dict = dict()

        for key in self.cfg.agent_cfg.obs_keys:
            partial_dict = dict()
            if key == "distances":
                partial_dict = ({"distances": gym.spaces.Box(
                    low=0, high=np.inf, shape=(self.cfg.agent_cfg.obs_num,))})
            elif key == "body_types":
                partial_dict = ({"body_types": gym.spaces.Tuple(
                    ([gym.spaces.Discrete(len(BodyType))]*self.cfg.agent_cfg.obs_num))})
            elif key == "body_velocities":
                partial_dict = ({"body_velocities": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.cfg.agent_cfg.obs_num, 2, ))})
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
    def set_observation_dict(self):
        state = dict()
        for key in self.cfg.agent_cfg.obs_keys:
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

    def get_observations(self):
        self.data.clear()

        for delta_angle in range(self.cfg.agent_cfg.obs_num):
            # absolute angle for the observation vector
            # based on self.cfg.agent_cfg.obs_range
            angle = self.get_observation_angle(delta_angle)

            observation_end = (self.agent_head.x + math.cos(angle) * self.cfg.agent_cfg.obs_max_dist,
                               self.agent_head.y + math.sin(angle) * self.cfg.agent_cfg.obs_max_dist)

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
        state = self.set_observation_dict()

        return state

    def update_agent_head(self):
        if self.cfg.agent_cfg.head_type == AgentHeadType.EDGE:
            self.agent_head = b2Vec2(
                (self.agent_body.position.x + math.cos(
                    self.agent_body.angle) * (self.agent_size.x - self.cfg.agent_cfg.head_inside)),
                (self.agent_body.position.y + math.sin(
                    self.agent_body.angle) * (self.agent_size.x - self.cfg.agent_cfg.head_inside)))
        elif self.cfg.agent_cfg.head_type == AgentHeadType.CENTER:
            self.agent_head = self.agent_body.position

    def get_observation_angle(self, delta_angle):
        try:
            return self.agent_body.angle - (self.cfg.agent_cfg.obs_range / 2) + (self.cfg.agent_cfg.obs_range / (self.cfg.agent_cfg.obs_num - 1) * delta_angle)
        except ZeroDivisionError:
            return self.agent_body.angle

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

    # user functions
    def get_world_size(self) -> tuple:
        return self.cfg.world_width, self.cfg.world_height

    def world_coord(self, point: b2Vec2) -> b2Vec2:
        return b2Vec2(point.x / self.cfg.ppm, (self.cfg.screen_layout.height - point.y) / self.cfg.ppm) - self.cfg.world_pos
