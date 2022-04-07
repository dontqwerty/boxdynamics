
import enum
import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from logging import debug, info
from time import sleep
from typing import Dict, List

import numpy as np
import pygame as pg
from Box2D import b2Body, b2Vec2

import boxcolors as color
from boxdef import BodyShape, BodyType
from boxutils import get_intersection, get_line_eq_angle

DESIGN_SLEEP = 0.01  # delay in seconds while designing world


class Mode(Enum):
    NONE = 0
    DESIGN = 1
    ROTATE = 2
    SIMULATION = 3
    QUIT_CONFIRMATION = 4 # TODO: confirmation
    INPUT_SAVE = 5
    INPUT_LOAD = 6


@dataclass
class ScreenLayout:
    width: int = 1000  # pixels
    height: int = 800
    size: b2Vec2 = b2Vec2(width, height)
    simulation_xshift = 200
    simulation_yshift = 0
    simulation_pos: b2Vec2 = b2Vec2(simulation_xshift, simulation_yshift)
    simulation_size: b2Vec2 = b2Vec2(width - simulation_xshift, height - simulation_yshift)
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
    color: tuple = field(default=color.STATIC_OBSTACLE)

    # body rotation
    rotated: bool = False
    init_vertices: List[b2Vec2] = field(default_factory=list)
    initial_angle: float = 0.0
    delta_angle: float = 0.0

    params: Dict = field(default_factory=dict)
    # indicates which param to currently change
    params_ix: int = 0
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
        self.default_design_data()

        # pg setup
        self.screen = pg.display.set_mode(
            (self.layout.width, self.layout.height), 0, 32)
        pg.display.set_caption('Box Dynamics')
        self.clock = pg.time.Clock()
        pg.font.init()  # to render text

        self.commands = list()

        # font
        self.font = 'Comic Sans MS'

        # border when drawing rectangles that contain text
        self.border_width = 10

        # will be updated on runtime based on the real dimensions
        # once the text is rendered
        self.title_surface_height = 0
        self.commands_surface_height = 0
        self.design_surface_height = 0

        # input text from user
        self.text_input = ""

        logging.basicConfig(level=logging.INFO)
        info("BoxUI created")

        pass

    def ui_sleep(self):
        sleep(DESIGN_SLEEP)

    def quit(self):
        pg.quit()
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

        self.render_back()

        # title
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)
        s = "{}".format(self.mode.name)
        text_surface = text_font.render(
            s, True, color.BLACK, color.INFO_BACK)
        self.title_surface_height = text_surface.get_height()
        pos = b2Vec2(self.border_width, self.border_width / 2)
        self.screen.blit(text_surface, pos)

        self.render_commands()

        self.render_world()

        if self.mode == Mode.SIMULATION:
            self.render_action()
            self.render_observations()
            self.draw_distances()
        elif self.mode not in (Mode.SIMULATION, Mode.NONE):
            self.render_design()

        pg.display.flip()
        self.clock.tick(self.target_fps)
        pass

    def default_design_data(self):
        self.design_data = DesignData()
        self.design_data.params = {"reward": 0,
                                   "level": 0.0,
                                   "lin_velocity": 0.0,
                                   "lin_velocity_angle": 0.0,
                                   "ang_velocity": 0.0,
                                   "density": 1.0,
                                   "inertia": 0.0,
                                   "friction": 0.0,
                                   "lin_damping": 0.0,
                                   "ang_damping": 0.0}

    def user_confirmation(self) -> bool:
        # TODO: render confirmation screen
        while True:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    return False
                elif event.type == pg.KEYDOWN and event.key == pg.K_RETURN:
                    return True

    def user_input(self):
        for event in pg.event.get():
            if self.mode in (Mode.INPUT_LOAD, Mode.INPUT_SAVE) and event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    if self.mode == Mode.INPUT_SAVE:
                        self.save_design(self.text_input)
                    elif self.mode == Mode.INPUT_LOAD:
                        self.load_design(self.text_input)
                    self.text_input = ""
                    self.set_mode(Mode.DESIGN)
                    # return
                elif event.key == pg.K_BACKSPACE:
                    self.text_input = self.text_input[:-1]
                else:
                    self.text_input += event.unicode
                pass

            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                # TODO: quit signal to BoxEnv
                # TODO: uncomment for confirmation
                # if self.user_confirmation():
                # self.quit()
                if self.mode == Mode.QUIT_CONFIRMATION:
                    # not confirmed, back to prev mode
                    self.set_mode(self.prev_mode)
                else:
                    # asking user for confirmation
                    self.set_mode(Mode.QUIT_CONFIRMATION)

            elif event.type == pg.KEYDOWN and event.key == pg.K_RETURN:
                if self.mode == Mode.QUIT_CONFIRMATION:
                    self.quit()

            elif event.type == pg.QUIT:
                # exit
                # here no confirmation
                self.quit()

            elif event.type == pg.KEYDOWN and event.key == pg.K_DELETE:
                # abort current changes to body
                # TODO: ask for confirmation (?)
                try:
                    self.design_bodies.pop()
                except IndexError:
                    pass
                self.default_design_data()

            elif event.type == pg.KEYDOWN and event.key == pg.K_c:
                if self.mode in (Mode.DESIGN, Mode.ROTATE):
                    self.design_data.shape = BodyShape.BOX
                    pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_s:
                if self.mode == Mode.DESIGN:
                    # asking to input file name
                    self.set_mode(Mode.INPUT_SAVE)
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_l:
                if self.mode == Mode.DESIGN:
                    self.set_mode(Mode.INPUT_LOAD)
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_u:
                if self.mode == Mode.DESIGN:
                    # use created world
                    # TODO: check for saving if not saved
                    self.set_mode(Mode.SIMULATION)
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_t:
                if self.mode in (Mode.DESIGN, Mode.ROTATE):
                    # toggle body type
                    self.toggle_body_type()
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_a:
                if self.mode == Mode.DESIGN:
                    # angle
                    points_num = len(self.design_data.points)
                    if points_num == 2 and self.design_data.shape == BodyShape.BOX:
                        self.rotate(first=True)
                        self.set_mode(Mode.ROTATE)
                elif self.mode == Mode.ROTATE:
                    self.set_mode(Mode.DESIGN)
                pass
            elif event.type == pg.MOUSEBUTTONDOWN:
                mouse_pos = b2Vec2(pg.mouse.get_pos())
                if self.mode == Mode.DESIGN:
                    # TODO: check for mouse pos and perform action like select bodies
                    points_num = len(self.design_data.points)
                    if points_num == 0:
                        self.design_data.points.append(mouse_pos)
                    elif points_num == 2:
                        # reset design data
                        self.default_design_data()
                elif self.mode == Mode.ROTATE:
                    self.design_bodies.pop()  # removing old body
                    # adding new updated body
                    self.design_bodies.append(self.design_data)
                    self.default_design_data()
                    self.set_mode(Mode.DESIGN)
                elif self.mode == Mode.SIMULATION:
                    pass
                # TODO: check for mouse pos and perform action
                # show body properties when clicked
                # let user change level
                pass
            elif event.type == pg.MOUSEMOTION:
                mouse_pos = b2Vec2(pg.mouse.get_pos())
                if self.mode == Mode.DESIGN:
                    points_num = len(self.design_data.points)
                    if points_num > 0:
                        # removing old body
                        try:
                            self.design_bodies.remove(self.design_data)
                        except ValueError:
                            # there where no bodies to remove
                            pass
                        if points_num == 2:
                            self.design_data.vertices = self.get_vertices(
                                self.design_data)
                            # replacing old point with new point on mouse position
                            self.design_data.points.pop()
                        # mouse motion after one point has been fixed
                        self.design_data.points.append(mouse_pos)
                        # new updated body
                        self.design_bodies.append(self.design_data)
                elif self.mode == Mode.ROTATE:
                    self.rotate(first=False)
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_m:
                if self.mode == Mode.SIMULATION:
                    self.env.manual_mode = not self.env.manual_mode
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                if self.mode == Mode.DESIGN:
                    param_name = list(self.design_data.params)[
                        self.design_data.params_ix]
                    if param_name == "level":
                        self.design_data.params[param_name] += 1
                    else:
                        self.design_data.params[param_name] += self.design_data.float_inc
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                if self.mode == Mode.DESIGN:
                    param_name = list(self.design_data.params)[
                        self.design_data.params_ix]
                    if param_name == "level":
                        self.design_data.params[param_name] -= 1
                    else:
                        self.design_data.params[param_name] -= self.design_data.float_inc
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                if self.mode == Mode.DESIGN:
                    # increase inc by factor 10
                    self.design_data.float_inc = self.design_data.float_inc * 10
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                if self.mode == Mode.DESIGN:
                    # decrease inc by factor 10
                    self.design_data.float_inc = self.design_data.float_inc / 10
                pass
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                if self.mode == Mode.DESIGN:
                    # toggle parameter to change
                    # TODO: shift space to toggle -1
                    if self.design_data.type == BodyType.MOVING_OBSTACLE or \
                        self.design_data.type == BodyType.MOVING_ZONE:
                        self.design_data.params_ix = (
                            self.design_data.params_ix + 1) % len(list(self.design_data.params))
                    else:
                        # TODO: only toggle static bodies params not hardcoded
                        self.design_data.params_ix = (
                            self.design_data.params_ix + 1) % 2
                pass

    def rotate(self, first: bool):
        if first:
            self.design_data.rotated = True

            # initial mouse position
            initial_mouse = b2Vec2(pg.mouse.get_pos())

            # initial angle
            self.design_data.initial_angle = self.get_angle(
                self.design_data.points[0], initial_mouse)

            # vertices when starting the rotation
            self.design_data.init_vertices = self.design_data.vertices.copy()
        else:
            mouse_pos = b2Vec2(pg.mouse.get_pos())

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

    def toggle_body_type(self):
        self.design_data.type = self.toggle_enum(
            self.design_data.type, [BodyType.AGENT, BodyType.BORDER, BodyType.DEFAULT])
        self.design_data.color = self.env.get_data(
            self.design_data.type).color

    def save_design(self, filename="test.json"):
        # db as design bodies since it's just a different
        # format for the same thing
        db = [list(body.__dict__.items()) for body in (self.design_bodies)]

        # design bodies to be dumpes as json
        dump_db = list()
        for bix, body in enumerate(db):
            dump_db.append(list())
            for dix, _ in enumerate(body):
                dump_db[bix].append(list())
                name = db[bix][dix][0]  # name of dataclass field
                value = db[bix][dix][1]  # value of dataclass field
                # appending name of data field
                dump_db[bix][dix].append(name)
                # getting actual value of dataclass field
                if isinstance(value, Enum):
                    # appending value of data field
                    dump_db[bix][dix].append(str(value.name))
                elif isinstance(value, list) and all(isinstance(i, b2Vec2) for i in value):
                    # TODO: check if actually all lists contain only b2Vec2 (yes for now)
                    # appending list for values
                    dump_db[bix][dix].append(list())
                    for vix, v in enumerate(value):
                        # appending every value in list
                        dump_db[bix][dix][1].append(
                            list(b2Vec2(value[vix].x, value[vix].y)))
                else:
                    # TODO: tuples (and other similar stuff inside DesignData) might be dangerous
                    dump_db[bix][dix].append(value)

        dump_db = [dict(body) for body in dump_db]

        # TODO: better name for json
        try:
            with open(filename, "w") as f:
                json.dump(dump_db, f)
        except FileNotFoundError:
            info("File not found: {}".format(filename))
            return

        info("Saved design {}".format(filename))

    def load_design(self, filename="test.json"):
        try:
            with open(filename, "r") as f:
                j = json.load(f)
        except FileNotFoundError:
            info("File not found: {}".format(filename))
            return

        # TODO: append to existing design or overwrite?
        # design_bodies = list()
        for body in j:
            design = DesignData(**body)
            design.shape = BodyShape[design.shape]
            design.points = [b2Vec2(p) for p in design.points]
            design.vertices = [b2Vec2(p) for p in design.vertices]
            design.type = BodyType[design.type]
            design.init_vertices = [b2Vec2(p) for p in design.init_vertices]
            # design_bodies.append(design)
            self.design_bodies.append(design)

        info("Loaded design {}".format(filename))

    def get_angle(self, pivot: b2Vec2, point: b2Vec2):
        delta: b2Vec2 = point - pivot
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
        bodies_levels: List[tuple(b2Body, int)] = self.get_sorted_bodies()

        for body, _ in bodies_levels:
            # skipping border rendering cause not necessary
            if body.userData.type == BodyType.BORDER:
                continue
            for fixture in body.fixtures:
                vertices = [self.__pg_coord(
                    body.transform * v) for v in fixture.shape.vertices]
                pg.draw.polygon(self.screen, body.userData.color, vertices)

    def render_back(self):
        # Filling screen not used by world
        vertices = [(0, 0),
                    (self.layout.simulation_xshift, 0),
                    (self.layout.simulation_xshift, self.layout.height),
                    (0, self.layout.height)]
        pg.draw.polygon(self.screen, color.INFO_BACK, vertices)

        vertices = [(0, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.simulation_yshift),
                    (0, self.layout.simulation_yshift)]
        pg.draw.polygon(self.screen, color.INFO_BACK, vertices)

        vertices = [(self.layout.simulation_pos.x + self.layout.simulation_size.x, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.height),
                    (self.layout.simulation_pos.x + self.layout.simulation_size.x, self.layout.height)]
        pg.draw.polygon(self.screen, color.INFO_BACK, vertices)

        vertices = [(0, self.layout.simulation_pos.y + self.layout.simulation_size.y),
                    (self.layout.width, self.layout.simulation_pos.y +
                     self.layout.simulation_size.y),
                    (self.layout.width, self.layout.height),
                    (0, self.layout.height)]
        pg.draw.polygon(self.screen, color.INFO_BACK, vertices)

    def render_simulation_data(self):
        # text_font = pg.font.SysFont(self.font, self.font_size)

        # # fps
        # fps = round(self.clock.get_fps())
        # fps_point = (0, 0)
        # fps_color = color.GREEN.value if abs(
        #     fps - self.target_fps) < 10 else color.RED.value
        # text_surface = text_font.render(
        #     str(fps), True, color.BLACK.value, fps_color)
        # self.screen.blit(text_surface, fps_point)
        pass

    def render_design(self, types=BodyType):
        # Draw the world based on bodies levels
        bodies_levels: List[tuple(DesignData, int)
                            ] = self.get_sorted_bodies(design=True)

        for body, _ in bodies_levels:
            if body.type in types:
                if body.shape == BodyShape.BOX:
                    try:
                        pg.draw.polygon(
                            self.screen, body.color, body.vertices)
                    except ValueError:
                        # wait another cycle for the vertices to be there
                        pass
                elif body.shape == BodyShape.CIRCLE:
                    radius = (body.points[0] - body.points[1]).length
                    pg.draw.circle(
                        self.screen, body.color, body.points[0], radius)

        self.render_design_data()

    def get_sorted_bodies(self, design=False):
        # TODO: have them already sorted for performance
        if design:
            bodies_levels: List[tuple(DesignData, int)] = [[b, b.params["level"]]
                                                           for b in self.design_bodies]
            bodies_levels.sort(key=lambda x: x[1], reverse=False)
        else:
            bodies_levels = [[b, b.userData.level]
                             for b in self.env.world.bodies]
            bodies_levels.sort(key=lambda x: x[1], reverse=False)

        return bodies_levels

    def render_design_data(self):
        # rendering design data when user is designin shape
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        design_pos = b2Vec2(
            0, self.commands_surface_height + self.border_width * 3)
        pos = design_pos.copy()

        pg.draw.rect(self.screen, color.BLACK,
                         pg.Rect(pos.x - self.border_width,
                                     pos.y,
                                     self.layout.simulation_xshift + self.border_width * 2,
                                     self.design_surface_height), width=self.border_width)

        # title
        pos += b2Vec2(self.border_width, self.border_width)
        s = "DESIGN DATA"
        text_surface = text_font.render(
            s, True, color.BLACK, color.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        data = ["Type: {}".format(self.design_data.type.name),
                "Shape: {}".format(self.design_data.shape.name),
                "Angle: {}".format(
                    round(-360 * self.design_data.delta_angle / (2 * math.pi), 3))]

        for ix, s in enumerate(data):
            pos += b2Vec2(0, text_surface.get_height())
            # TODO: not ix hardcoded
            if self.mode == Mode.ROTATE and ix == 2:
                text_surface = text_font.render(
                    s, True, color.BLACK, color.GREEN)
            else:
                text_surface = text_font.render(
                    s, True, color.BLACK, color.INFO_BACK)
            self.screen.blit(text_surface, pos)

        params = list()
        params.append("Reward: {}".format(
            round(self.design_data.params["reward"], 3)))
        params.append("Level: {}".format(round(self.design_data.params["level"])))

        if self.design_data.type == BodyType.MOVING_OBSTACLE or \
                self.design_data.type == BodyType.MOVING_ZONE:
            params.append("Velocity: {}".format(
                round(self.design_data.params["lin_velocity"], 3)))
            params.append("Velocity angle: {}".format(
                round(self.design_data.params["lin_velocity_angle"], 3)))
            params.append("Angular velocity: {}".format(
                round(self.design_data.params["ang_velocity"], 3)))
            params.append("Density: {}".format(
                round(self.design_data.params["density"], 3)))
            params.append("Inertia: {}".format(
                round(self.design_data.params["inertia"], 3)),)
            params.append("Friction: {}".format(
                round(self.design_data.params["friction"], 3)))
            params.append("Linear damping: {}".format(
                round(self.design_data.params["lin_damping"], 3)))
            params.append("Angular damping: {}".format(
                round(self.design_data.params["ang_damping"], 3)))

        params.append("Increment: {}".format((self.design_data.float_inc)))

        for ix, s in enumerate(params):
            pos += b2Vec2(0, text_surface.get_height())
            if self.mode == Mode.DESIGN and ix == self.design_data.params_ix:
                text_surface = text_font.render(
                    s, True, color.BLACK, color.GREEN)
            else:
                text_surface = text_font.render(
                    s, True, color.BLACK, color.INFO_BACK)
            self.screen.blit(text_surface, pos)
        self.design_surface_height = pos.y - design_pos.y + self.border_width * 3

    def render_commands(self):
        pos = b2Vec2(0, self.title_surface_height + self.border_width)
        pg.draw.rect(self.screen, color.BLACK,
                         pg.Rect(pos.x - self.border_width,
                                     pos.y,
                                     self.layout.simulation_xshift + self.border_width * 2,
                                     self.commands_surface_height), width=self.border_width)

        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        # title
        pos += b2Vec2(self.border_width, self.border_width)
        text_surface = text_font.render(
            "COMMANDS", True, color.BLACK, color.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        self.set_commands()
        for command in self.commands:
            pos += b2Vec2(0, text_surface.get_height())
            s = "- {}: {}".format(command["key"], command["description"])
            text_surface = text_font.render(
                s, True, color.BACK, color.INFO_BACK)
            self.screen.blit(text_surface, pos)

        self.commands_surface_height = pos.y

    def set_commands(self):
        self.commands.clear()
        if self.mode == Mode.DESIGN:
            self.commands = [{"key": "mouse click", "description": "fix point"},
                             {"key": "R", "description": "rectangle"},
                             {"key": "C", "description": "circle"},
                             {"key": "T", "description": "type"},
                             {"key": "U", "description": "use"},
                             {"key": "S", "description": "save"},
                             {"key": "L", "description": "load"},
                             {"key": "SPACE", "description": "toggle parameter"},
                             {"key": "UP", "description": "increase parameter"},
                             {"key": "DOWN", "description": "decrease parameter"},
                             {"key": "RIGHT", "description": "increase parameter inc"},
                             {"key": "LEFT", "description": "decrease parameter inc"}]
            if self.design_data.shape == BodyShape.BOX:
                self.commands.append({"key": "A", "description": "rotate"})
        elif self.mode == Mode.ROTATE:
            self.commands = [{"key": "A", "description": "finish rotating"},
                             {"key": "mouse movement", "description": "rotate"},
                             {"key": "mouse click", "description": "fix point"}]
        elif self.mode == Mode.SIMULATION:
            self.commands = [{"key": "M", "description": "manual"}]

        if self.mode not in (Mode.SIMULATION, Mode.NONE):
            self.commands.append({"key": "DEL", "description": "delete object"})

        self.commands.append({"key": "ESC", "description": "exit"})

    def render_action(self):
        p1 = self.__pg_coord(self.env.agent_head)
        p2 = self.__pg_coord(self.env.agent_head + self.env.action)

        pg.draw.line(self.screen, color.ACTION, p1, p2)

    def draw_distances(self):
        text_font = pg.font.SysFont(
            self.font, self.layout.small_font)

        # drawing distance text
        # TODO: only draw them if there is enough space
        freq = 1  # defines "distance text" quantity
        for oix, observation in enumerate(self.env.data):
            if oix % freq == 0 and observation.valid:
                distance = round(observation.distance, 1)

                text_point = self.__pg_coord(
                    self.env.agent_head + (observation.intersection - self.env.agent_head) / 2)
                text_surface = text_font.render(
                    str(distance), False, color.BLACK, color.WHITE)
                self.screen.blit(text_surface, text_point)

    def render_observations(self):
        start_point = self.__pg_coord(self.env.agent_head)
        for observation in self.env.data:
            if observation.valid:
                end_point = self.__pg_coord(observation.intersection)

                # drawing observation vectors
                pg.draw.line(self.screen, observation.body.userData.color,
                                 start_point, end_point)
                # drawing intersection points
                pg.draw.circle(
                    self.screen, color.INTERSECTION, end_point, 3)

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

    # transform point in world coordinates to point in pg coordinates
    def __pg_coord(self, point):
        point = b2Vec2(point.x * self.ppm, (self.env.world_height -
                       point.y) * self.ppm) + self.layout.simulation_pos
        return point
