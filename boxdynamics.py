import json
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum, unique
import logging
from string import hexdigits
from turtle import heading
from typing import Dict, List

import gym
import numpy as np
import pygame as pg
from Box2D import b2Body, b2Fixture, b2PolygonShape, b2Vec2, b2World

import boxcolors as color
from boxcontacts import ContactListener
from boxdef import (BodyType, BodyData, DesignData, EffectType, EffectWhen, ScreenLayout,
                    UIMode)
from boxraycast import RayCastClosestCallback
from boxui import BoxUI
from boxutils import anglemag_to_vec, dataclass_to_dict, get_intersection, get_line_eq, copy_design_bodies, get_effect


@unique
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

    screen: ScreenLayout = ScreenLayout()

    world_width: float = screen.simulation_size.x / ppm
    world_height: float = screen.simulation_size.y / ppm
    world_pos: b2Vec2 = b2Vec2(screen.simulation_pos.x, screen.height -
                               screen.simulation_pos.y) / ppm - b2Vec2(0, world_height)

    # world
    create_borders: bool = True
    render_distances: bool = True  # render text distances for observations
    borders_width: float = 10  # meters
    border_inside: float = 0

    # velocity and position iterations
    # higher values improve precision in
    # physics simulations
    vel_iter: int = 6
    pos_iter: int = 2

    agent: AgentCfg = AgentCfg()
    design_bodies: List = field(default_factory=list)


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
        # TODO: logging level in config file
        logging.basicConfig(
            format='%(levelname)s: %(asctime)s: %(message)s', level=logging.DEBUG)

        # TODO: give possibility to delete body if out of world

        # initializing base class
        super(BoxEnv, self).__init__()

        # world keeps track of objects and physics
        self.contact_listener = ContactListener(self)
        self.world = b2World(gravity=(0, 0), doSleep=True,
                             contactListener=self.contact_listener)

        # setting configuration based on deafults
        # needed before using self.cfg
        self.load_conf("config.json")

        # initializing UI
        self.ui = BoxUI(self, self.cfg.screen,
                        self.cfg.ppm, self.cfg.target_fps)

        # action space
        # relative agent_angle, force
        self.action_space = gym.spaces.Box(
            np.array([self.cfg.agent.min_angle, self.cfg.agent.min_force]
                     ).astype(np.float32),
            np.array([self.cfg.agent.max_angle, self.cfg.agent.max_force]
                     ).astype(np.float32)
        )

        # to signal that the training is done
        self.done = False
        # to overwrite agent actions with user manual actions
        self.manual_mode = False
        self.agent_head = b2Vec2(0, 0)
        self.action = b2Vec2(0, 0)

        # defining observation spaces based on cfg.agent.obs_keys
        self.observation_space = gym.spaces.Dict(self.get_observation_dict())
        self.state = dict()

        # list of Observation dataclasses used to set some
        # observations on each step
        self.data: List[Observation] = list()

        # creates world borders and agent
        self.create_world()

        # reward to zero
        self.total_reward = 0.0

    def reset(self):
        logging.info("Resetting enviroment")
        # resetting base class
        super().reset()

        # resetting reward
        # TODO: give reward to agent
        self.total_reward = 0.0
        self.done = False
        self.manual_mode = False

        # returning state after step with None action
        return self.step(None)[0]

    def step(self, action) -> tuple:
        # calculate agent head point
        # TODO: support circles
        self.update_agent_head()

        self.perform_action(action)

        # effects with EffectWhen.DURING_CONTACT
        for bodyA in self.world.bodies:
            dataA: BodyData = bodyA.userData
            if dataA.effect["type"] != EffectType.NONE:
                for bodyB in dataA.contacts:
                    self.contact_listener.contact_effect(bodyA, bodyB, EffectWhen.DURING_CONTACT)

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
        self.total_reward += step_reward

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
            json.dump(dataclass_to_dict(self.cfg), f)
        pass

    def load_conf(self, filename="config.json"):
        try:
            with open(filename, "r") as f:
                self.cfg = EnvCfg(**json.load(f))
        except Exception as e:
            logging.warn(
                "Unable to read config file: {} with exception: {}".format(filename, e))
            logging.info("Using default config")
            self.cfg = EnvCfg()
            return

        # design bodies
        for bix, body in enumerate(self.cfg.design_bodies):
            # TODO: warning!! Here the type of the fields is not
            # guaranteed to be the same as the hint in the dataclass
            # definition
            # Example: points which is a list of b2Vec2, here is just
            # a list of 2-elements lists
            self.cfg.design_bodies[bix] = DesignData(**body)
            self.cfg.design_bodies[bix].points = [
                b2Vec2(p) for p in self.cfg.design_bodies[bix].points]
            self.cfg.design_bodies[bix].vertices = [
                b2Vec2(p) for p in self.cfg.design_bodies[bix].vertices]

        # screen layout
        self.cfg.screen = ScreenLayout(**self.cfg.screen)
        self.cfg.screen.size = b2Vec2(self.cfg.screen.size)
        self.cfg.screen.simulation_pos = b2Vec2(self.cfg.screen.simulation_pos)
        self.cfg.screen.simulation_size = b2Vec2(
            self.cfg.screen.simulation_size)
        self.cfg.screen.board_pos = b2Vec2(self.cfg.screen.board_pos)
        self.cfg.screen.popup_size = b2Vec2(self.cfg.screen.popup_size)
        self.cfg.screen.popup_pos = b2Vec2(self.cfg.screen.popup_pos)

        # agent
        self.cfg.agent = AgentCfg(**self.cfg.agent)

        logging.info("Config file loaded correctly")
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
            self.action = anglemag_to_vec(
                angle=action[0] + self.agent_body.angle, magnitude=action[1])
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
        for design in copy_design_bodies(self.cfg.design_bodies):
            self.ui.design_bodies.append(design)

        self.ui.set_mode(UIMode.RESIZE)
        while self.ui.mode != UIMode.SIMULATION:
            self.ui.user_input()
            self.ui.ui_sleep()
            self.render()

        self.cfg.design_bodies = copy_design_bodies(
            self.ui.design_bodies)

        self.create_bodies()

    def create_bodies(self):
        # creates all the bodies in the enviroment configuration
        if isinstance(self.cfg.design_bodies, list):
            for design in self.cfg.design_bodies:
                if design.valid:
                    logging.debug("Creating object from design")
                    self.create_body(design)
                else:
                    logging.debug("Invalud object from design")

    def create_body(self, design_data: DesignData):
        points = [self.world_coord(b2Vec2(point))
                  for point in design_data.vertices]
        try:
            line1 = get_line_eq(points[0], points[2])
            line2 = get_line_eq(points[1], points[3])
        except IndexError:
            # the body is a single point (?)
            logging.warning(
                "Can not create body from design {}".format(design_data))
            return
        pos = get_intersection(line1, line2)
        # size for Box2D is half the real size
        width = (design_data.width) / (2*self.cfg.ppm)
        height = (design_data.height) / (2*self.cfg.ppm)
        size = (width, height)
        angle = -design_data.angle

        type = design_data.physic["type"]
        if type == BodyType.AGENT:
            pass
        elif type == BodyType.BORDER:
            pass
        elif type == BodyType.STATIC_OBSTACLE:
            body = self.create_static_obstacle(pos, size, angle=angle)
        elif type == BodyType.DYNAMIC_OBSTACLE:
            body = self.create_dynamic_obstacle(pos, size, angle=angle)
            self.set_body_params(body, design_data)
        elif type == BodyType.KINEMATIC_OBSTACLE:
            body = self.create_kinematic_obstacle(pos, size, angle=angle)
            self.set_body_params(body, design_data)
        elif type == BodyType.STATIC_ZONE:
            body = self.create_static_zone(pos, size, angle=angle)
        elif type == BodyType.DYNAMIC_ZONE:
            body = self.create_dynamic_zone(pos, size, angle=angle)
            self.set_body_params(body, design_data)
        elif type == BodyType.KINEMATIC_ZONE:
            body = self.create_kinematic_zone(pos, size, angle=angle)
            self.set_body_params(body, design_data)
        else:
            assert False and "Unknown body type"

        body.userData.reward = design_data.physic["reward"]
        body.userData.level = design_data.physic["level"]

        body.userData.effect = design_data.effect.copy()

    def create_borders(self):
        inside = self.cfg.border_inside  # defines how much of the borders is visible

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
        logging.info("Created borders")

    def create_agent(self):
        agent_width, agent_height = self.cfg.agent.size
        self.agent_size = b2Vec2(agent_width, agent_height)
        agent_angle = self.cfg.agent.angle

        # setting random initial position
        if self.cfg.agent.pos is None:
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
            agent_pos = b2Vec2(self.cfg.agent.pos)

        # setting random initial angle
        if self.cfg.agent.angle is None:
            agent_angle = random.random() * (2*math.pi)

        self.agent_body: b2Body = self.world.CreateDynamicBody(
            position=agent_pos, angle=agent_angle)
        self.agent_fix: b2Fixture = self.agent_body.CreatePolygonFixture(
            box=(agent_width, agent_height), density=self.cfg.agent.density, friction=self.cfg.agent.friction)

        self.agent_body.userData = self.get_agent_data()

        self.agent_fix.body.angularDamping = self.cfg.agent.ang_damp
        self.agent_fix.body.linearDamping = self.cfg.agent.lin_damp

        logging.info("Created agent")

    # TODO: support colors, circles
    def create_static_obstacle(self, pos, size, angle=0):
        body: b2Body = self.world.CreateStaticBody(
            position=pos, angle=angle
        )
        fixture: b2Fixture = body.CreatePolygonFixture(
            box=size)

        body.userData = self.get_static_obstacle_data()
        return body

    def create_dynamic_obstacle(self, pos, size, velocity=b2Vec2(1, 1), angle=0):
        body: b2Body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False, )
        _: b2Fixture = body.CreatePolygonFixture(
            box=size, density=1)

        body.userData = self.get_dynamic_obstacle_data()
        return body

    def create_kinematic_obstacle(self, pos, size, velocity=b2Vec2(1, 1), angle=0):
        body: b2Body = self.world.CreateKinematicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0)
        _: b2Fixture = body.CreatePolygonFixture(
            box=size)

        body.userData = self.get_kinematic_obstacle_data()
        return body

    def create_static_zone(self, pos, size, angle=0):
        body: b2Body = self.world.CreateStaticBody(
            position=pos, angle=angle)
        fixture: b2Fixture = body.CreatePolygonFixture(
            box=size)
        fixture.sensor = True

        body.userData = self.get_static_zone_data()
        return body

    def create_dynamic_zone(self, pos, size, velocity=b2Vec2(1, 1), angle=0):
        body: b2Body = self.world.CreateDynamicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0,
            bullet=False)
        fixture: b2Fixture = body.CreatePolygonFixture(
            box=size)

        fixture.sensor = True

        body.userData = self.get_dynamic_zone_data()
        return body

    def create_kinematic_zone(self, pos, size, velocity=b2Vec2(1, 1), angle=0):
        body: b2Body = self.world.CreateKinematicBody(
            position=pos, angle=angle, linearVelocity=velocity, angularVelocity=0)
        fixture: b2Fixture = body.CreatePolygonFixture(
            box=size)

        fixture.sensor = True

        body.userData = self.get_kinematic_zone_data()
        return body

    def set_body_params(self, body: b2Body, design_data: DesignData):
        body.angularDamping = design_data.physic["ang_damping"]
        body.angularVelocity = design_data.physic["ang_velocity"]
        body.linearDamping = design_data.physic["lin_damping"]
        design_data.physic["lin_velocity_angle"] = design_data.physic["lin_velocity_angle"] * (
            2 * math.pi / 360)
        body.linearVelocity = anglemag_to_vec(
            angle=design_data.physic["lin_velocity_angle"], magnitude=design_data.physic["lin_velocity"])
        body.inertia = design_data.physic["inertia"]
        body.fixtures[0].density = design_data.physic["density"]
        body.fixtures[0].friction = design_data.physic["friction"]

    # returns a dictionary which can then be converted to a gym.spaces.Dict
    # defines min, max, shape and of each observation key
    def get_observation_dict(self):
        observation_dict = dict()

        for key in self.cfg.agent.obs_keys:
            partial_dict = dict()
            if key == "distances":
                partial_dict = ({"distances": gym.spaces.Box(
                    low=0, high=np.inf, shape=(self.cfg.agent.obs_num,))})
            elif key == "body_types":
                partial_dict = ({"body_types": gym.spaces.Tuple(
                    ([gym.spaces.Discrete(len(BodyType))]*self.cfg.agent.obs_num))})
            elif key == "body_velocities":
                partial_dict = ({"body_velocities": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.cfg.agent.obs_num, 2, ))})
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
        for key in self.cfg.agent.obs_keys:
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

        for delta_angle in range(self.cfg.agent.obs_num):
            # absolute angle for the observation vector
            # based on self.cfg.agent.obs_range
            angle = self.get_observation_angle(delta_angle)

            observation_end = (self.agent_head.x + math.cos(angle) * self.cfg.agent.obs_max_dist,
                               self.agent_head.y + math.sin(angle) * self.cfg.agent.obs_max_dist)

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
        if self.cfg.agent.head_type == AgentHeadType.EDGE:
            self.agent_head = b2Vec2(
                (self.agent_body.position.x + math.cos(
                    self.agent_body.angle) * (self.agent_size.x - self.cfg.agent.head_inside)),
                (self.agent_body.position.y + math.sin(
                    self.agent_body.angle) * (self.agent_size.x - self.cfg.agent.head_inside)))
        elif self.cfg.agent.head_type == AgentHeadType.CENTER:
            self.agent_head = self.agent_body.position

    def get_observation_angle(self, delta_angle):
        try:
            return self.agent_body.angle - (self.cfg.agent.obs_range / 2) + (self.cfg.agent.obs_range / (self.cfg.agent.obs_num - 1) * delta_angle)
        except ZeroDivisionError:
            return self.agent_body.angle

    def get_data(self, type: BodyType):
        if type == BodyType.BORDER:
            return self.get_border_data()
        elif type == BodyType.AGENT:
            return self.get_agent_data()
        elif type == BodyType.STATIC_OBSTACLE:
            return self.get_static_obstacle_data()
        elif type == BodyType.DYNAMIC_OBSTACLE:
            return self.get_dynamic_obstacle_data()
        elif type == BodyType.KINEMATIC_OBSTACLE:
            return self.get_kinematic_obstacle_data()
        elif type == BodyType.STATIC_ZONE:
            return self.get_static_zone_data()
        elif type == BodyType.DYNAMIC_ZONE:
            return self.get_dynamic_zone_data()
        elif type == BodyType.KINEMATIC_ZONE:
            return self.get_kinematic_zone_data()
        else:
            assert False and "Unknown body type"

    def get_border_data(self) -> BodyData:
        # todo border effect
        effect = get_effect(EffectType.NONE)
        return BodyData(type=BodyType.BORDER, color=color.BORDER, effect=effect)

    def get_agent_data(self) -> BodyData:
        effect = get_effect(EffectType.NONE)
        return BodyData(type=BodyType.AGENT, color=color.AGENT, level=np.inf, effect=effect)

    def get_static_obstacle_data(self) -> BodyData:
        return BodyData(type=BodyType.STATIC_OBSTACLE, color=color.STATIC_OBSTACLE)

    def get_dynamic_obstacle_data(self) -> BodyData:
        return BodyData(
            type=BodyType.DYNAMIC_OBSTACLE, color=color.DYNAMIC_OBSTACLE)

    def get_kinematic_obstacle_data(self) -> BodyData:
        return BodyData(
            type=BodyType.KINEMATIC_OBSTACLE, color=color.KINEMATIC_OBSTACLE)

    def get_static_zone_data(self) -> BodyData:
        return BodyData(
            type=BodyType.STATIC_ZONE, color=color.STATIC_ZONE)

    def get_dynamic_zone_data(self) -> BodyData:
        return BodyData(
            type=BodyType.DYNAMIC_ZONE, color=color.DYNAMIC_ZONE)

    def get_kinematic_zone_data(self) -> BodyData:
        return BodyData(
            type=BodyType.KINEMATIC_ZONE, color=color.KINEMATIC_ZONE)

    # user functions
    def get_world_size(self) -> tuple:
        return self.cfg.world_width, self.cfg.world_height

    def world_coord(self, point: b2Vec2) -> b2Vec2:
        return b2Vec2(point.x / self.cfg.ppm, (self.cfg.screen.height - point.y) / self.cfg.ppm) - self.cfg.world_pos
