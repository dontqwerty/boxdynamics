
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from logging import debug
from multiprocessing.sharedctypes import Value
from time import sleep
from tracemalloc import start
from typing import List

import numpy as np
import pygame
from Box2D import b2Vec2

import boxcolors as color
from boxdef import BodyShape, BodyType
from boxutils import get_intersection, get_line_eq_angle

DESIGN_SLEEP = 0.01  # delay in seconds while designing world


class Mode(Enum):
    NONE = 0
    WORLD_DESIGN = 1
    SIMULATION = 2
    SET_REWARD = 3
    SET_LEVEL = 4
    SET_PHYSICS = 5
    ROTATE = 6


@dataclass
class ScreenLayout:
    width: int = 800  # pixels
    height: int = 800
    size: b2Vec2 = b2Vec2(width, height)
    simulation_xshift = 200
    simulation_yshift = 0
    simulation_pos: b2Vec2 = b2Vec2(simulation_xshift, simulation_yshift)
    simulation_size: b2Vec2 = b2Vec2(600, height - simulation_yshift)
    board_pos: b2Vec2 = b2Vec2(0, simulation_size.y)
    board_size: b2Vec2 = b2Vec2(width, height) * 0
    small_font: int = 14
    normal_font: int = 20
    big_font: int = 32


@dataclass
class DesignData:
    shape: Enum = BodyShape.BOX  # used if creating
    selected: bool = False  # TODO: selectable

    # center and radius for circles
    points: List[b2Vec2] = field(default_factory=list)
    # all four vertices for rectangles
    vertices: List[b2Vec2] = field(default_factory=list)
    type: Enum = BodyType.STATIC_OBSTACLE
    color: Enum = color.BROWN
    reward: float = 0.0
    level: int = 0

    # body rotation
    rotated: bool = False
    rotation_begin: bool = False
    init_vertices: List[b2Vec2] = field(default_factory=list)
    initial_angle: float = 0.0
    delta_angle: float = 0.0

    # physics
    physics = {"lin_velocity": 0.0,
               "lin_velocity_angle": 0.0,
               "ang_velocity": 0.0,
               "friction": 0.0,
               "density": 1.0,
               "inertia": 0.0,
               "lin_damping": 0.0,
               "ang_damping": 0.0}
    # indicates which param to currently change
    physics_param_ix = 0
    float_inc: float = 0.1


class BoxUI():
    def __init__(self, env, screen_layout: ScreenLayout, ppm, target_fps: float) -> None:
        self.env = env
        self.layout = screen_layout
        self.ppm = ppm
        self.target_fps = target_fps

        # only setting mode this way here
        # use set_mode everywhere else
        self.mode = Mode.NONE
        self.prev_mode = Mode.NONE

        self.design_bodies: List[DesignData] = list()

        # setting up design variables
        self.design_data = DesignData()

        # pygame setup
        self.screen = pygame.display.set_mode(
            (self.layout.width, self.layout.height), 0, 32)
        pygame.display.set_caption('Box Dynamics')
        self.clock = pygame.time.Clock()
        pygame.font.init()  # to render text

        # border when drawing rectangles that contain text
        self.border_width = 10

        # will be updated on runtime based on the real dimensions
        self.title_surface_height = 0
        self.design_surface_height = 0
        self.commands_surface_height = 0

        logging.basicConfig(level=logging.DEBUG)
        debug("BoxUI created")

        pass

    def ui_sleep(self):
        sleep(DESIGN_SLEEP)

    def __destroy(self):
        pass

    def quit(self):
        pygame.quit()
        # TODO: signal quitting to BoxEnv
        exit()

    def set_mode(self, mode: Mode):
        self.prev_mode = self.mode
        self.mode = mode
        debug("New mode {}, old mode {}".format(self.mode, self.prev_mode))

    def shape_design(self):
        if self.design_data.shape == BodyShape.BOX:
            self.box_design()
        elif self.design_data.shape == BodyShape.CIRCLE:
            # TODO: circle design function
            pass

    def render(self):
        self.screen.fill(color.BACK)

        self.render_world()

        # title
        text_font = pygame.font.SysFont(
            'Comic Sans MS', self.layout.big_font)
        s = "{}".format(self.mode.name)
        text_surface = text_font.render(
            s, True, color.BLACK, color.GREEN)
        self.title_surface_height = text_surface.get_height()
        pos = b2Vec2(0, 0)
        self.screen.blit(text_surface, pos)

        self.render_commands()

        if self.mode == Mode.SIMULATION:
            self.render_action()
            self.render_observations()
            self.draw_distances()

        elif self.mode != Mode.SIMULATION and self.mode != Mode.NONE:
            self.render_design()


        pygame.display.flip()
        self.clock.tick(self.target_fps)
        pass

    def user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # TODO: quit signal to BoxEnv
                self.quit()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DELETE:
                # abort current changes to body
                try:
                    self.design_bodies.pop()
                except IndexError:
                    pass
                self.design_data = DesignData()

            elif self.mode == Mode.WORLD_DESIGN:
                # TODO: check for delete key and cancel creating
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.design_data.shape = BodyShape.BOX
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    # TODO: circle
                    self.design_data.shape = BodyShape.CIRCLE
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    # TODO: save function
                    self.set_mode(Mode.SIMULATION)
                    pass
                elif event.type == pygame.KEYDOWN and (event.key == pygame.K_u or
                                                       event.key == pygame.K_RETURN):
                    # use created world
                    # TODO: check for saving if not saved
                    self.set_mode(Mode.SIMULATION)
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    # toggle body type
                    self.design_data.type = self.toggle_enum(
                        self.design_data.type, [BodyType.AGENT, BodyType.BORDER, BodyType.DEFAULT])
                    self.design_data.color = self.env.get_data(
                        self.design_data.type).color
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                    # reward
                    self.set_mode(Mode.SET_REWARD)
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                    # level
                    self.set_mode(Mode.SET_LEVEL)
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    # angle
                    points_num = len(self.design_data.points)
                    if points_num == 2 and self.design_data.shape == BodyShape.BOX:
                        self.design_data.rotation_begin = True
                        self.set_mode(Mode.ROTATE)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    if self.design_data.type == BodyType.MOVING_OBSTACLE or \
                            self.design_data.type == BodyType.MOVING_ZONE:
                        self.set_mode(Mode.SET_PHYSICS)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # TODO: check for mouse pos and perform action like select bodies
                    points_num = len(self.design_data.points)
                    if points_num == 0:
                        mouse_pos = pygame.mouse.get_pos()
                        self.design_data.points.append(b2Vec2(mouse_pos))
                    elif points_num == 2:
                        # reset design data
                        self.design_data = DesignData()
                elif event.type == pygame.MOUSEMOTION:
                    points_num = len(self.design_data.points)
                    if points_num > 0:
                        # mouse motion after one point has been fixed
                        mouse_pos = b2Vec2(pygame.mouse.get_pos())

                        # removing old body
                        try:
                            self.design_bodies.remove(self.design_data)
                        except ValueError:
                            pass

                        if points_num == 2:
                            self.design_data.vertices = self.get_vertices(
                                self.design_data)
                            # replacing old point with new point on mouse position
                            self.design_data.points.pop()

                        self.design_data.points.append(b2Vec2(mouse_pos))

                        # new updated body
                        self.design_bodies.append(self.design_data)
                    pass

            elif self.mode == Mode.ROTATE:
                # TODO: change angle with arrows (like rewards)
                if self.design_data.rotation_begin:
                    # the user just changed mode to rotate body
                    # saving useful info now

                    self.design_data.rotated = True

                    # initial mouse position
                    initial_mouse = b2Vec2(pygame.mouse.get_pos())

                    # initial angle
                    self.design_data.initial_angle = self.get_angle(
                        self.design_data.points[0], initial_mouse)

                    # vertices when starting the rotation
                    self.design_data.init_vertices = self.design_data.vertices.copy()

                    self.design_data.rotation_begin = False

                if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    self.set_mode(Mode.WORLD_DESIGN)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.design_bodies.pop()  # removing old body
                    # adding new updated body
                    self.design_bodies.append(self.design_data)
                    self.design_data = DesignData()  # reset of design data for new body
                    self.set_mode(Mode.WORLD_DESIGN)

                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = b2Vec2(pygame.mouse.get_pos())

                    self.design_data.delta_angle = self.get_angle(
                        self.design_data.points[0], mouse_pos) - self.design_data.initial_angle

                    # rotating every vertex
                    for vix, vertex in enumerate(self.design_data.vertices):
                        distance = (vertex - self.design_data.points[0]).length

                        init_angle = self.get_angle(
                            self.design_data.points[0], self.design_data.init_vertices[vix])

                        final_angle = init_angle + self.design_data.delta_angle
                        self.design_data.vertices[vix] = self.design_data.points[0] + (
                            b2Vec2(math.cos(final_angle), math.sin(final_angle)) * distance)

            elif self.mode == Mode.SET_REWARD:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                    # return to world design mode
                    self.set_mode(Mode.WORLD_DESIGN)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    # increase reward by inc
                    self.design_data.reward += self.design_data.float_inc
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    # decrease reward by inc
                    self.design_data.reward -= self.design_data.float_inc
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    # increase inc by factor 10
                    self.design_data.float_inc = self.design_data.float_inc * 10
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    # decrease inc by factor 10
                    self.design_data.float_inc = self.design_data.float_inc / 10
                    pass

            elif self.mode == Mode.SET_LEVEL:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                    # return to world design mode
                    self.set_mode(Mode.WORLD_DESIGN)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    # increase level by inc
                    self.design_data.level += 1
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    # decrease level by inc
                    self.design_data.level -= 1
                    pass

            elif self.mode == Mode.SET_PHYSICS:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    self.set_mode(Mode.WORLD_DESIGN)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    # toggle physics parameter to change
                    self.design_data.physics_param_ix = (
                        self.design_data.physics_param_ix + 1) % len(list(self.design_data.physics))
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    # increase reward by inc
                    self.design_data.physics[list(self.design_data.physics)[
                        self.design_data.physics_param_ix]] += self.design_data.float_inc
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    # decrease reward by inc
                    self.design_data.physics[list(self.design_data.physics)[
                        self.design_data.physics_param_ix]] -= self.design_data.float_inc
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    # increase inc by factor 10
                    self.design_data.float_inc = self.design_data.float_inc * 10
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    # decrease inc by factor 10
                    self.design_data.float_inc = self.design_data.float_inc / 10
                    pass
                pass

            elif self.mode == Mode.SIMULATION:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    self.env.manual_mode = not self.env.manual_mode
                    pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # TODO: check for mouse pos and perform action
                    # show body properties when clicked
                    # let user change level
                    pass

    def get_angle(self, pivot: b2Vec2, point: b2Vec2):
        delta = point - pivot
        # first quadrant
        if point.x >= pivot.x and point.y >= pivot.y:
            try:
                angle = math.atan(delta.y / delta.x)
            except ZeroDivisionError:
                angle = math.atan(np.inf)
            pass
        # second quadrant
        elif point.x <= pivot.x and point.y >= pivot.y:
            try:
                angle = math.pi + math.atan(delta.y / delta.x)
            except ZeroDivisionError:
                angle = math.atan(-np.inf)
            pass
        # third quadrant
        elif point.x <= pivot.x and point.y <= pivot.y:
            try:
                angle = math.pi + math.atan(delta.y / delta.x)
            except ZeroDivisionError:
                angle = math.atan(-np.inf)
            pass
        # fourth quadrant
        elif point.x >= pivot.x and point.y <= pivot.y:
            try:
                angle = math.atan(delta.y / delta.x)
            except ZeroDivisionError:
                angle = math.atan(np.inf)
            pass

        return angle

    def toggle_enum(self, e, skip=[]):
        enum_list = list(type(e))
        new_ix = (enum_list.index(e) + 1) % len(enum_list)
        new_e = enum_list[new_ix]

        # TODO: check for undless loop
        if new_e in skip:
            new_e = self.toggle_enum(new_e, skip)

        return new_e

    def render_world(self):
        # Draw the world based on bodies levels
        bodies_levels = [[b, b.userData.level]
                         for b in self.env.world.bodies]
        bodies_levels.sort(key=lambda x: x[1], reverse=False)

        for body, _ in bodies_levels:
            for fixture in body.fixtures:
                vertices = [self.__pygame_coord(
                    body.transform * v) for v in fixture.shape.vertices]
                pygame.draw.polygon(self.screen, body.userData.color, vertices)

        # Filling screen not used by world
        vertices = [(0, 0),
                    (self.layout.simulation_xshift, 0),
                    (self.layout.simulation_xshift, self.layout.height),
                    (0, self.layout.height)]
        pygame.draw.polygon(self.screen, color.WHITE, vertices)

        vertices = [(0, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.simulation_yshift),
                    (0, self.layout.simulation_yshift)]
        pygame.draw.polygon(self.screen, color.WHITE, vertices)

        vertices = [(self.layout.simulation_pos.x + self.layout.simulation_size.x, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.height),
                    (self.layout.simulation_pos.x + self.layout.simulation_size.x, self.layout.height)]
        pygame.draw.polygon(self.screen, color.WHITE, vertices)

        vertices = [(0, self.layout.simulation_pos.y + self.layout.simulation_size.y),
                    (self.layout.width, self.layout.simulation_pos.y +
                     self.layout.simulation_size.y),
                    (self.layout.width, self.layout.height),
                    (0, self.layout.height)]
        pygame.draw.polygon(self.screen, color.WHITE, vertices)

    def render_design(self, types=BodyType):
        # Draw the world based on bodies levels
        bodies_levels: List[tuple(DesignData, int)] = [[b, b.level]
                                                       for b in self.design_bodies]
        bodies_levels.sort(key=lambda x: x[1], reverse=False)

        for body, _ in bodies_levels:
            if body.type in types:
                if body.shape == BodyShape.BOX:
                    try:
                        pygame.draw.polygon(
                            self.screen, body.color, body.vertices)
                    except ValueError:
                        # wait another cycle for the vertices to be there
                        pass
                elif body.shape == BodyShape.CIRCLE:
                    radius = (body.points[0] - body.points[1]).length
                    pygame.draw.circle(
                        self.screen, body.color, body.points[0], radius)

        self.render_design_data()

    def render_design_data(self):
        # rendering design data when user is designin shape
        text_font = pygame.font.SysFont(
            'Comic Sans MS', self.layout.big_font)

        design_pos = b2Vec2(0, self.commands_surface_height + self.border_width * 3)
        pos = design_pos.copy()

        pygame.draw.rect(self.screen, color.YELLOW,
                            pygame.Rect(pos.x, pos.y, self.layout.simulation_xshift, self.design_surface_height))
        pygame.draw.rect(self.screen, color.BLACK,
                            pygame.Rect(pos.x - self.border_width,
                            pos.y,
                            self.layout.simulation_xshift + self.border_width * 2,
                            self.design_surface_height), width=self.border_width)

        # title
        pos += b2Vec2(self.border_width, self.border_width)
        s = "DESIGN DATA"
        text_surface = text_font.render(
            s, True, color.BLACK, color.YELLOW)
        self.screen.blit(text_surface, pos)

        text_font = pygame.font.SysFont(
            'Comic Sans MS', self.layout.normal_font)

        data = ["Shape: {}".format(self.design_data.shape.name),
                "Type: {}".format(self.design_data.type.name),
                "Angle: {}".format(round(self.design_data.delta_angle, 3)),
                "Reward: {}".format(round(self.design_data.reward, 3)),
                "Level: {}".format(self.design_data.level),
                "Parameter increment: {}".format((self.design_data.float_inc))]

        for s in data:
            pos += b2Vec2(0, text_surface.get_height())
            text_surface = text_font.render(
                s, True, color.BLACK, color.YELLOW)
            self.screen.blit(text_surface, pos)

        if self.design_data.type == BodyType.MOVING_OBSTACLE or \
                self.design_data.type == BodyType.MOVING_ZONE:
            physic_params = ["Velocity: {}".format(round(self.design_data.physics["lin_velocity"], 3)),
                             "Velocity angle: {}".format(
                                 round(self.design_data.physics["lin_velocity_angle"], 3)),
                             "Angular velocity: {}".format(
                                 round(self.design_data.physics["ang_velocity"], 3)),
                             "Friction: {}".format(
                                 round(self.design_data.physics["friction"], 3)),
                             "Density: {}".format(
                                 round(self.design_data.physics["density"], 3)),
                             "Inertia: {}".format(
                                 round(self.design_data.physics["inertia"], 3)),
                             "Linear damping: {}".format(
                                 round(self.design_data.physics["lin_damping"], 3)),
                             "Angular damping: {}".format(round(self.design_data.physics["ang_damping"], 3))]

            for ix, s in enumerate(physic_params):
                pos += b2Vec2(0, text_surface.get_height())
                if ix == self.design_data.physics_param_ix:
                    text_surface = text_font.render(
                        s, True, color.BLACK, color.GREEN)
                else:
                    text_surface = text_font.render(
                        s, True, color.BLACK, color.YELLOW)
                self.screen.blit(text_surface, pos)
        self.design_surface_height = pos.y - design_pos.y + self.border_width * 3

    def render_commands(self):
        pos = b2Vec2(0, self.title_surface_height + self.border_width)
        pygame.draw.rect(self.screen, color.YELLOW,
                            pygame.Rect(pos.x, pos.y, self.layout.simulation_xshift, self.commands_surface_height))
        pygame.draw.rect(self.screen, color.BLACK,
                            pygame.Rect(pos.x - self.border_width,
                                    pos.y,
                                    self.layout.simulation_xshift + self.border_width * 2,
                                    self.commands_surface_height), width=self.border_width)

        text_font = pygame.font.SysFont(
            'Comic Sans MS', self.layout.big_font)

        # title
        pos += b2Vec2(self.border_width, self.border_width)
        text_surface = text_font.render(
            "COMMANDS", True, color.BLACK, color.YELLOW)
        self.screen.blit(text_surface, pos)

        text_font = pygame.font.SysFont(
            'Comic Sans MS', self.layout.normal_font)

        commands = list()
        if self.mode == Mode.WORLD_DESIGN:
            commands = [{"key": "mouse click", "description": "fix point"},
                        {"key": "R", "description": "rectangle"},
                        {"key": "C", "description": "circle"},
                        {"key": "T", "description": "type"},
                        {"key": "W", "description": "reward"},
                        {"key": "L", "description": "level"},
                        {"key": "U", "description": "use"},
                        {"key": "S", "description": "save"},
                        {"key": "ESC", "description": "exit"}]
            if self.design_data.type == BodyType.MOVING_OBSTACLE or \
                    self.design_data.type == BodyType.MOVING_ZONE:
                commands.append({"key": "P", "description": "physics"})

            if self.design_data.shape == BodyShape.BOX:
                commands.append({"key": "A", "description": "rotate"})
        if self.mode == Mode.ROTATE:
            commands = [{"key": "A", "description": "finish rotating"},
                        {"key": "mouse movement", "description": "rotate"},
                        {"key": "mouse click", "description": "fix point"},
                        {"key": "ESC", "description": "exit"}]
        elif self.mode == Mode.SET_REWARD:
            commands = [{"key": "W", "description": "finish set reward"},
                        {"key": "UP", "description": "increase reward"},
                        {"key": "DOWN", "description": "decrease reward"},
                        {"key": "RIGHT", "description": "increase reward inc"},
                        {"key": "LEFT", "description": "decrease reward inc"},
                        {"key": "ESC", "description": "exit"}]
        elif self.mode == Mode.SET_LEVEL:
            commands = [{"key": "L", "description": "finish set level"},
                        {"key": "UP", "description": "increase level"},
                        {"key": "DOWN", "description": "decrease level"},
                        {"key": "ESC", "description": "exit"}]
        elif self.mode == Mode.SET_PHYSICS:
            commands = [{"key": "P", "description": "finish set physics"},
                        {"key": "SPACE", "description": "toggle parameter"},
                        {"key": "UP", "description": "increase parameter"},
                        {"key": "DOWN", "description": "decrease parameter"},
                        {"key": "RIGHT", "description": "increase parameter inc"},
                        {"key": "LEFT", "description": "decrease parameter inc"},
                        {"key": "ESC", "description": "exit"}]
        elif self.mode == Mode.SIMULATION:
            commands = [{"key": "M", "description": "manual"},
                        {"key": "ESC", "description": "exit"}]

        # pos += b2Vec2(self.border_width, self.border_width)
        for command in commands:
            pos += b2Vec2(0, text_surface.get_height())
            s = "- {}: {}".format(command["key"], command["description"])
            text_surface = text_font.render(s, True, color.BACK, color.YELLOW)
            self.screen.blit(text_surface, pos)

        self.commands_surface_height = pos.y

    def render_action(self):
        p1 = self.__pygame_coord(self.env.agent_head)
        p2 = self.__pygame_coord(self.env.agent_head + self.env.action)

        pygame.draw.line(self.screen, color.ACTION, p1, p2)

    def draw_distances(self):
        text_font = pygame.font.SysFont(
            'Comic Sans MS', self.layout.small_font)

        # drawing distance text
        # TODO: only draw them if there is enough space
        freq = 1  # defines "distance text" quantity
        for oix, observation in enumerate(self.env.data):
            if oix % freq == 0 and observation.valid:
                distance = round(observation.distance, 1)

                text_point = self.__pygame_coord(
                    self.env.agent_head + (observation.intersection - self.env.agent_head) / 2)
                text_surface = text_font.render(
                    str(distance), False, color.BLACK, color.WHITE)
                self.screen.blit(text_surface, text_point)

    def render_observations(self):
        start_point = self.__pygame_coord(self.env.agent_head)
        for observation in self.env.data:
            if observation.valid:
                end_point = self.__pygame_coord(observation.intersection)

                # drawing observation vectors
                pygame.draw.line(self.screen, observation.body.userData.color,
                                 start_point, end_point)
                # drawing intersection points
                pygame.draw.circle(
                    self.screen, color.INTERSECTION, end_point, 3)

    # def draw_infos_box(self, title):
    #     info_vertices = [(self.layout.board_pos.x, self.layout.board_pos.y + self.layout.board_size.y),
    #                      (self.layout.board_pos.x + self.layout.board_size.x,
    #                       self.layout.board_pos.y + self.layout.board_size.y),
    #                      (self.layout.board_pos.x +
    #                       self.layout.board_size.x, self.layout.board_pos.y),
    #                      self.layout.board_pos]
    #     pygame.draw.polygon(self.screen, color.BACK, info_vertices)

    #     text_font = pygame.font.SysFont('Comic Sans MS', self.title_font_size)
    #     s = "{}".format(title)
    #     text_surface = text_font.render(
    #         s, True, color.BLACK.value, color.GREEN.value)

    #     # TODO: get center function
    #     pos = ((self.layout.width - text_surface.get_width()) / 2,
    #            self.layout.height - self.layout.board_size.y)
    #     self.screen.blit(text_surface, pos)
    #     return text_surface.get_height()

    # def draw_design(self):
    #     title_height = self.draw_infos_box("CREATE MODE")

    #     text_font = pygame.font.SysFont('Comic Sans MS', self.font_size)

    #     # design commands
    #     for cix, cmd in enumerate(self.design_commands):
    #         s = "{}: {}".format(cmd["key"], cmd["description"])
    #         text_surface = text_font.render(
    #             s, True, color.BLACK.value, color.WHITE.value)

    #         pos = self.get_info_pos(0, cix, title_height)
    #         self.screen.blit(text_surface, pos)

    #     # design infos
    #     for iix, info in enumerate(self.design_infos):
    #         s = "{}: {}".format(info["name"], info["value"])
    #         text_surface = text_font.render(
    #             s, True, color.BLACK.value, color.WHITE.value)

    #         pos = self.get_info_pos(1,
    #                                   iix, title_height)
    #         self.screen.blit(text_surface, pos)

    # def draw_simulation_infos(self):
    #     title_height = self.draw_infos_box("SIMULATION MODE")

    #     text_font = pygame.font.SysFont('Comic Sans MS', self.font_size)

    #     # fps
    #     fps = round(self.clock.get_fps())
    #     fps_point = (0, 0)
    #     fps_color = color.GREEN.value if abs(
    #         fps - self.target_fps) < 10 else color.RED.value
    #     text_surface = text_font.render(
    #         str(fps), True, color.BLACK.value, fps_color)
    #     self.screen.blit(text_surface, fps_point)

    #     max_len = 0
    #     for iix, info in enumerate(self.screen_infos):
    #         info_str = "> {}: {}".format(info["name"], info["value"])
    #         text_surface = text_font.render(
    #             info_str, True, color.BLACK.value, color.WHITE.value
    #         )

    #         if info["name"][0:len("contact")] == "contact":
    #             pos = self.get_info_pos(1, iix, title_height)
    #         else:
    #             pos = self.get_info_pos(0, iix, title_height)

    #         self.screen.blit(text_surface, pos)

    # def get_title_pos(self, width, height):
    #     pass

    # def get_info_pos(self, x_ix, y_ix, title_height):
    #     x_border = self.layout.width / (INFO_X_SLOTS + 2)
    #     y_border = self.layout.board_size.y / (INFO_Y_SLOTS + 2)

    #     # TODO: check out of slots
    #     x = x_ix*(x_border + 1)
    #     y = y_ix*(y_border + 1) + title_height

    #     return b2Vec2(x, self.layout.height - self.layout.board_size.y + y)

    # def get_vertices(self, body: DesignData):
    #     p1 = body.points[0]
    #     p3 = body.points[1]
    #     line11 = get_line_eq_angle(
    #         p1, body.beta + (body.alpha - body.angle))
    #     line12 = get_line_eq_angle(
    #         p1, body.gamma + (body.alpha - body.angle))

    #     line21 = get_line_eq_angle(
    #         p3, body.gamma + (body.alpha - body.angle))
    #     line22 = get_line_eq_angle(
    #         p3, body.beta + (body.alpha - body.angle))

    #     p2 = get_intersection(line11, line21)
    #     p4 = get_intersection(line12, line22)

    #     return [p1, p2, p3, p4]

    def get_vertices(self, body: DesignData):
        p1 = body.points[0]
        p3 = body.points[1]
        if body.rotated:

            init_angle1 = self.get_angle(p1, self.design_data.init_vertices[1])
            final_angle1 = init_angle1 + self.design_data.delta_angle

            init_angle2 = self.get_angle(p1, self.design_data.init_vertices[3])
            final_angle2 = init_angle2 + self.design_data.delta_angle

            line11 = get_line_eq_angle(
                p1, final_angle1)
            line12 = get_line_eq_angle(
                p1, final_angle2)

            line21 = get_line_eq_angle(
                p3, final_angle2)
            line22 = get_line_eq_angle(
                p3, final_angle1)

            p2 = get_intersection(line11, line21)
            p4 = get_intersection(line12, line22)

        else:
            p2 = b2Vec2(p3.x, p1.y)
            p4 = b2Vec2(p1.x, p3.y)

        return [p1, p2, p3, p4]

    # transform point in world coordinates to point in pygame coordinates
    def __pygame_coord(self, point):
        point = b2Vec2(point.x * self.ppm, (self.env.world_height -
                       point.y) * self.ppm) + self.layout.simulation_pos
        return point
