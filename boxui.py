import json
import logging
import math
import random
from dataclasses import is_dataclass
from enum import IntEnum
from time import sleep
from typing import Dict, List

import numpy as np
import pygame as pg
from Box2D import b2Body, b2Vec2

import boxcolors
from boxdef import (BodyType, DesignData, EffectType, ScreenLayout,
                    UIMode)
from boxutils import dataclass_to_dict, get_intersection, get_line_eq_angle

DESIGN_SLEEP = 0.001  # delay in seconds while designing world


class SetType(IntEnum):
    DEFAULT = 0
    PREVIOUS = 1
    RANDOM = 2


class BoxUI():
    def __init__(self, env, screen_layout: ScreenLayout, ppm, target_fps: float, mode: UIMode = UIMode.SIMULATION) -> None:
        self.env = env
        self.layout = screen_layout
        self.ppm = ppm
        self.target_fps = target_fps

        # only setting mode this way here
        # use set_mode everywhere else
        self.mode = mode
        self.prev_mode = UIMode.NONE

        self.design_bodies: List[DesignData] = list()

        # setting up design variables
        self.design_data = self.get_design_data(set_type=SetType.DEFAULT)
        # self.set_type = SetType.DEFAULT
        self.set_type = SetType.PREVIOUS
        # self.set_type = SetType.RANDOM

        # pg setup
        pg.init()
        if self.layout.size == b2Vec2(0, 0): # fullscreen
            info = pg.display.Info()
            self.screen = pg.display.set_mode((info.current_w, info.current_h), pg.DOUBLEBUF | pg.FULLSCREEN, 32)
        else:
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
        self.popup_msg_size = b2Vec2(0, 0)

        # input text from user
        self.text_input = ""

        # design_bodies index when selecting
        self.design_body_ix = 0

        self.reverse_toggle = False
        self.prev_mouse_pos = None

        # initially empty list of design_data values that the user
        # can save and use using keys from 0 to 9
        self.saved_designs: List[DesignData] = [None]*10

        logging.info("BoxUI created")

        pass

    def ui_sleep(self):
        sleep(DESIGN_SLEEP)

    def quit(self):
        pg.quit()
        # TODO: signal quitting to BoxEnv
        exit()

    def set_mode(self, mode: UIMode):
        self.prev_mode = self.mode
        self.mode = mode
        logging.info("Old mode {}, new mode {}".format(
            self.prev_mode.name, self.mode.name))

    def render(self):
        self.screen.fill(boxcolors.BACK)

        self.render_back()

        # title
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)
        s = "Mode {}".format(self.mode.name)
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.title_surface_height = text_surface.get_height()
        pos = b2Vec2(self.border_width, self.border_width / 2)
        self.screen.blit(text_surface, pos)

        # TODO: render commands
        # self.render_commands()

        # TODO: fix that when designing and quit confirmation is asked
        # the observation vectors of the agente can be seen
        self.render_world()
        if self.mode == UIMode.SIMULATION or (self.prev_mode == UIMode.SIMULATION and self.mode == UIMode.QUIT_CONFIRMATION):
            self.render_action()
            self.render_observations()
            self.draw_distances()
        if self.mode in (UIMode.RESIZE,
                         UIMode.MOVE,
                         UIMode.ROTATE,
                         UIMode.INPUT_LOAD,
                         UIMode.INPUT_SAVE,
                         UIMode.USE_CONFIRMATION) or \
            (self.prev_mode == UIMode.RESIZE and
             self.mode == UIMode.QUIT_CONFIRMATION):
            self.render_design()
            if self.mode in (UIMode.INPUT_LOAD, UIMode.INPUT_SAVE):
                self.render_input()

        if self.mode == UIMode.QUIT_CONFIRMATION:
            self.render_confirmation("QUIT")
        elif self.mode == UIMode.USE_CONFIRMATION:
            self.render_confirmation("USE")
        pg.display.flip()
        self.clock.tick(self.target_fps)
        pass

    def render_popup(self):
        pg.draw.rect(self.screen, boxcolors.INFO_BACK,
                     pg.Rect(self.layout.popup_pos.x,
                             self.layout.popup_pos.y,
                             self.layout.popup_size.x,
                             self.layout.popup_size.y), width=4)

    def render_confirmation(self, msg=""):
        self.render_popup()

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        # text
        pos = self.layout.popup_pos + \
            (self.layout.popup_size - self.popup_msg_size) / 2
        s = "CONFIRM " + str(msg)
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.popup_msg_size = b2Vec2(text_surface.get_size())
        self.screen.blit(text_surface, pos)

    def render_input(self):
        self.render_popup()

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        # text
        pos: b2Vec2 = self.layout.popup_pos + \
            (self.layout.popup_size - self.popup_msg_size) / 2
        s = "ENTER NAME"
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.popup_msg_size = b2Vec2(text_surface.get_size())
        self.screen.blit(text_surface, pos)

        pos += b2Vec2(text_surface.get_width() / 4, text_surface.get_height())
        s = self.text_input
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.popup_msg_size.y += text_surface.get_height()
        b2Vec2(text_surface.get_size())
        self.screen.blit(text_surface, pos)

    def copy_design_bodies(self):
        design_copy = list()

        count = 0
        for body in self.design_bodies:
            design_dict: Dict[str,
                            DesignData] = body.__dict__  # pointer
            design_data: Dict[str, DesignData] = dict()

            # checking for fields that need to be copied manually
            for key in list(design_dict.keys()):
                # b2Vec2
                if type(design_dict[key]) is b2Vec2:
                    design_data[key] = (design_dict['key'].copy())
                # list of b2Vec2
                elif isinstance(design_dict[key], list) and all(isinstance(val, b2Vec2) for val in design_dict[key]):
                    design_data[key] = list()
                    for val in design_dict[key]:
                        design_data[key].append(val.copy())
                # dictionary
                elif isinstance(design_dict[key], dict):
                    design_data[key] = design_dict[key].copy()
                else:
                    design_data[key] = design_dict[key]


            design_data = DesignData(**design_data)
            design_copy.append(design_data)

            # TODO: remove or make proper function
            assert design_copy[-1] == body and "Must equal"
            design_copy[-1].points[0].x += 1
            assert design_copy != body and "Must differ"
            design_copy[-1].points[0].x -= 1
            assert design_copy[-1] == body and "Must equal 2"

        return design_copy

    def get_design_data(self, set_type=SetType.DEFAULT):
        design_data = DesignData()
        # use copiable data types
        if set_type == SetType.DEFAULT:
            design_data.params = {"type": BodyType.STATIC_OBSTACLE,
                                  "reward": 0.0,
                                  "level": 0,
                                  "lin_velocity": 10.0,
                                  "lin_velocity_angle": 0.0,
                                  "ang_velocity": 0.0,
                                  "density": 1.0,
                                  "inertia": 0.0,
                                  "friction": 0.0,
                                  "lin_damping": 0.0,
                                  "ang_damping": 0.0}
            design_data.effect = {"type": EffectType.APPLY_FORCE,
                                  "value": [0, 0]}  # TODO: let value be choosen at runtime
        elif set_type == SetType.PREVIOUS:
            # indexes
            design_data.dict_ix = self.design_data.dict_ix
            design_data.params_ix = self.design_data.params_ix
            design_data.effect_ix = self.design_data.effect_ix
            
            design_data.params = self.design_data.params.copy()
            design_data.effect = self.design_data.effect.copy()
        elif set_type == SetType.RANDOM:
            types = [BodyType.STATIC_OBSTACLE, BodyType.MOVING_OBSTACLE,
                     BodyType.STATIC_ZONE, BodyType.MOVING_ZONE]
            design_data.params = {"type": random.choice(types),
                                  "reward": random.uniform(-1, 1),
                                  "level": 0,
                                  "lin_velocity": random.uniform(0, 10),
                                  "lin_velocity_angle": random.uniform(0, 2*math.pi),
                                  "ang_velocity": random.uniform(-5, 5),
                                  "density": 0.5,
                                  "inertia": 0,
                                  "friction": random.uniform(0, 0.001),
                                  "lin_damping": random.uniform(0, 0.001),
                                  "ang_damping": random.uniform(0, 0.001)}
            design_data.effect = {"type": EffectType.APPLY_FORCE,
                                  "value": [1000000, 1000000]}
        design_data.dicts = [design_data.params, design_data.effect]
        return design_data

    def user_input(self):
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if self.mode in (UIMode.INPUT_LOAD, UIMode.INPUT_SAVE):
                    if event.key == pg.K_RETURN:
                        if self.mode == UIMode.INPUT_SAVE:
                            self.save_design(self.text_input)
                        elif self.mode == UIMode.INPUT_LOAD:
                            self.load_design(self.text_input)
                        self.text_input = ""
                        self.set_mode(self.prev_mode)
                        # return
                    elif event.key == pg.K_BACKSPACE:
                        self.text_input = self.text_input[:-1]
                    elif event.key == pg.K_ESCAPE:
                        self.text_input = ""
                        self.set_mode(self.prev_mode)
                    else:
                        self.text_input += event.unicode
                    pass

                elif event.key == pg.K_ESCAPE:
                    # TODO: quit signal to BoxEnv
                    # TODO: uncomment for confirmation
                    # if self.user_confirmation():
                    # self.quit()
                    if self.mode in (UIMode.QUIT_CONFIRMATION, UIMode.USE_CONFIRMATION):
                        # not confirmed, back to prev mode
                        self.set_mode(self.prev_mode)
                    else:
                        # asking user for confirmation
                        self.set_mode(UIMode.QUIT_CONFIRMATION)

                elif event.key == pg.K_RETURN:
                    if self.mode == UIMode.QUIT_CONFIRMATION:
                        # quit confirmed
                        self.quit()
                    elif self.mode == UIMode.USE_CONFIRMATION:
                        # use confirmed
                        self.set_mode(UIMode.SIMULATION)

                elif event.type == pg.QUIT:
                    # exit
                    # here no confirmation
                    self.quit()

                elif event.key == pg.K_DELETE:
                    # abort current changes to body
                    # TODO: ask for confirmation (?)
                    try:
                        self.design_bodies.pop()
                    except IndexError:
                        pass
                    self.design_data = self.get_design_data(
                        set_type=self.set_type)
                elif event.key == pg.K_LSHIFT:
                    self.reverse_toggle = True
                    pass
                elif event.key == pg.K_s:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # asking to input file name
                        self.set_mode(UIMode.INPUT_SAVE)
                    pass
                elif event.key == pg.K_l:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        self.set_mode(UIMode.INPUT_LOAD)
                    pass
                elif event.key == pg.K_u:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # use created world
                        self.set_mode(UIMode.USE_CONFIRMATION)
                    pass
                elif event.key == pg.K_m:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE):
                        # moving body
                        self.move(first=True)
                    elif self.mode == UIMode.MOVE:
                        self.set_mode(UIMode.RESIZE)
                    elif self.mode == UIMode.SIMULATION:
                        self.env.manual_mode = not self.env.manual_mode
                    pass
                    pass
                elif event.key == pg.K_a:
                    if self.mode in (UIMode.RESIZE, UIMode.MOVE):
                        # rotating body
                        points_num = len(self.design_data.points)
                        if points_num == 2:
                            self.rotate(first=True)
                            self.set_mode(UIMode.ROTATE)
                    elif self.mode == UIMode.ROTATE:
                        self.set_mode(UIMode.RESIZE)
                    pass
                elif event.key == pg.K_e:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # TODO: shift for reverse toggle
                        self.design_data.dict_ix = (self.design_data.dict_ix + 1) % len(self.design_data.dicts)
                        pass
                    pass
                elif event.key == pg.K_UP:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase currently selected param
                        self.modify_param(increase=True)
                    pass
                elif event.key == pg.K_DOWN:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # decrease currently selected param
                        self.modify_param(increase=False)
                    pass
                elif event.key == pg.K_RIGHT:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase inc by factor 10
                        self.design_data.float_inc = self.design_data.float_inc * 10
                    pass
                elif event.key == pg.K_LEFT:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # decrease inc by factor 10
                        self.design_data.float_inc = self.design_data.float_inc / 10
                    pass
                elif event.key == pg.K_SPACE:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # toggle parameter to change
                        self.toggle_param()
                    pass
            elif event.type == pg.KEYUP:
                if event.key == pg.K_LSHIFT:
                    self.reverse_toggle = False
                pass
            # pg mouse buttons
            # 2 - middle click
            elif event.type == pg.MOUSEBUTTONDOWN:
                # 1 - left click
                if event.button == 1:
                    if self.mode == UIMode.RESIZE:
                        self.set_points(from_rotate=True)
                    elif self.mode in (UIMode.ROTATE, UIMode.MOVE):
                        self.set_points(from_rotate=False)
                        self.set_mode(UIMode.RESIZE)
                    elif self.mode == UIMode.SIMULATION:
                        # TODO: check for mouse pos and perform action
                        # show body properties when clicked
                        # let user change level
                        pass
                # 3 - right click
                elif event.button == 3:
                    # toggles between already created design bodies
                    # in order to modify them
                    # TODO: make it better
                    if self.mode in (UIMode.ROTATE, UIMode.MOVE, UIMode.RESIZE):
                        self.toggle_design_body()
                        if self.mode != UIMode.RESIZE:
                            self.set_mode(UIMode.RESIZE)
                # 4 - scroll up
                elif event.button == 4:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase currently selected param
                        self.modify_param(increase=True)
                # 5 - scroll down
                elif event.button == 5:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase currently selected param
                        self.modify_param(increase=False)
                pass
            elif event.type == pg.MOUSEMOTION:
                if self.mode == UIMode.RESIZE:
                    self.resize()
                elif self.mode == UIMode.ROTATE:
                    self.rotate(first=False)
                elif self.mode == UIMode.MOVE:
                    self.move(first=False)
                    pass
                pass

    def toggle_design_body(self):
        if len(self.design_bodies):
            if self.reverse_toggle == False:
                inc = 1
            else:
                inc = -1
            if self.design_data == self.design_bodies[self.design_body_ix]:
                self.design_body_ix = (
                    self.design_body_ix + inc) % len(self.design_bodies)
            self.design_data = self.design_bodies[self.design_body_ix]

    def set_points(self, from_rotate: bool):
        if from_rotate:
            mouse_pos = b2Vec2(pg.mouse.get_pos())
            # TODO: check for mouse pos and perform action like select bodies
            points_num = len(self.design_data.points)
            if points_num == 0:
                self.design_data.points.append(mouse_pos)
            elif points_num == 2:
                # reset design data
                self.design_data = self.get_design_data(set_type=self.set_type)
        else:
            self.design_bodies.pop()  # removing old body
            # adding new updated body
            self.design_bodies.append(self.design_data)
            self.design_data = self.get_design_data(set_type=self.set_type)

    def resize(self):
        mouse_pos = b2Vec2(pg.mouse.get_pos())
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

    def move(self, first: bool):
        mouse_pos = b2Vec2(pg.mouse.get_pos())
        if first:
            points_num = len(self.design_data.points)
            if points_num == 2:
                self.prev_mouse_pos = b2Vec2(pg.mouse.get_pos())
                self.set_mode(UIMode.MOVE)
        else:
            delta_mouse = mouse_pos - self.prev_mouse_pos
            # if self.prev_mouse_pos:
            #     delta_mouse = mouse_pos - self.prev_mouse_pos
            # else:
            #     self.prev_mouse_pos = mouse_pos
            #     delta_mouse = mouse_pos
            for point in self.design_data.vertices:
                point += delta_mouse
            self.prev_mouse_pos = mouse_pos

            self.design_data.points[0] = self.design_data.vertices[0]
            self.design_data.points[1] = self.design_data.vertices[2]
            self.design_data.init_vertices = self.design_data.vertices.copy()

    def toggle_param(self):
        if self.reverse_toggle == False:
            inc = 1
        else:
            inc = -1
        if self.design_data.dicts[self.design_data.dict_ix] == self.design_data.params:
            # we are toggling design params
            if self.design_data.params["type"] == BodyType.MOVING_OBSTACLE or \
                    self.design_data.params["type"] == BodyType.MOVING_ZONE:
                self.design_data.params_ix = (
                    self.design_data.params_ix + inc) % len(list(self.design_data.params))
            else:
                # TODO: only toggle static bodies params not hardcoded
                self.design_data.params_ix = (
                    self.design_data.params_ix + inc) % 3
        elif self.design_data.dicts[self.design_data.dict_ix] == self.design_data.effect:
            # we are toggling effect params
            self.design_data.effect_ix = (self.design_data.effect_ix + inc) % len(self.design_data.effect)
            pass
        else:
            assert False and "Added something to self.design_data.dicts?"

    def modify_param(self, increase=True):
        param_name = list(self.design_data.params)[self.design_data.params_ix]
        if param_name == "type":
            if increase:
                self.toggle_body_type(increase)
            else:
                self.toggle_body_type(increase)
        elif param_name == "level":
            if increase:
                self.design_data.params[param_name] += 1
            else:
                self.design_data.params[param_name] -= 1
        else:
            if increase:
                self.design_data.params[param_name] += self.design_data.float_inc
            else:
                self.design_data.params[param_name] -= self.design_data.float_inc
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

    def toggle_body_type(self, increase=True):
        self.design_data.params["type"] = self.toggle_enum(
            self.design_data.params["type"], skip=[BodyType.AGENT, BodyType.BORDER, BodyType.DEFAULT], increase=increase)
        self.design_data.color = self.env.get_data(
            self.design_data.params["type"]).color

    def save_design(self, filename):
        dump_db = [dataclass_to_dict(body) for body in self.design_bodies]
        try:
            with open(filename, "w") as f:
                json.dump(dump_db, f)
        except FileNotFoundError:
            logging.info("File not found: {}".format(filename))
            return

        logging.info("Saved design {}".format(filename))

    def load_design(self, filename):
        try:
            with open(filename, "r") as f:
                loaded_db = json.load(f)
        except FileNotFoundError:
            logging.info("File not found: {}".format(filename))
            return

        # TODO: append to existing design or overwrite?
        # currently appending
        for body in loaded_db:
            design = DesignData(**body)
            design.points = [b2Vec2(p) for p in design.points]
            design.vertices = [b2Vec2(p) for p in design.vertices]
            design.init_vertices = [b2Vec2(p) for p in design.init_vertices]
            self.design_bodies.append(design)

        logging.info("Loaded design {}".format(filename))

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

    def toggle_enum(self, e, skip=[], increase=True):
        enum_list = list(type(e))
        if increase:
            new_ix = (enum_list.index(e) + 1) % len(enum_list)
        else:
            new_ix = (enum_list.index(e) - 1) % len(enum_list)
        new_e = enum_list[new_ix]

        # TODO: check for undless loop
        if new_e in skip:
            new_e = self.toggle_enum(new_e, skip, increase)

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
        pg.draw.polygon(self.screen, boxcolors.INFO_BACK, vertices)

        vertices = [(0, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.simulation_yshift),
                    (0, self.layout.simulation_yshift)]
        pg.draw.polygon(self.screen, boxcolors.INFO_BACK, vertices)

        vertices = [(self.layout.simulation_pos.x + self.layout.simulation_size.x, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.height),
                    (self.layout.simulation_pos.x + self.layout.simulation_size.x, self.layout.height)]
        pg.draw.polygon(self.screen, boxcolors.INFO_BACK, vertices)

        vertices = [(0, self.layout.simulation_pos.y + self.layout.simulation_size.y),
                    (self.layout.width, self.layout.simulation_pos.y +
                     self.layout.simulation_size.y),
                    (self.layout.width, self.layout.height),
                    (0, self.layout.height)]
        pg.draw.polygon(self.screen, boxcolors.INFO_BACK, vertices)

    def render_simulation_data(self):
        # text_font = pg.font.SysFont(self.font, self.font_size)

        # # fps
        # fps = round(self.clock.get_fps())
        # fps_point = (0, 0)
        # fps_color = boxcolors.GREEN.value if abs(
        #     fps - self.target_fps) < 10 else boxcolors.RED.value
        # text_surface = text_font.render(
        #     str(fps), True, boxcolors.BLACK.value, fps_color)
        # self.screen.blit(text_surface, fps_point)
        pass

    def render_design(self):
        # Draw the world based on bodies levels
        bodies_levels: List[tuple(DesignData, int)
                            ] = self.get_sorted_bodies(design=True)

        for body, _ in bodies_levels:
            color = self.env.get_data(body.params["type"]).color
            try:
                pg.draw.polygon(
                    self.screen, color, body.vertices)
            except ValueError:
                # wait another cycle for the vertices to be there
                pass
            if body == self.design_data:
                pg.draw.circle(self.screen, boxcolors.YELLOW,
                               body.points[0], 5)

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
            0, self.border_width * 3)
        pos = design_pos.copy()
        pos = b2Vec2(self.border_width, self.title_surface_height + self.border_width)

        # pg.draw.rect(self.screen, boxcolors.BLACK,
        #              pg.Rect(pos.x - self.border_width,
        #                      pos.y,
        #                      self.layout.simulation_xshift + self.border_width * 2,
        #                      self.design_surface_height), width=self.border_width)

        # title
        # pos += b2Vec2(self.border_width, self.border_width)
        s = "DESIGN DATA"
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        # can't be toggled like params
        data = ["Angle: {}".format(
                    round(-360 * self.design_data.delta_angle / (2 * math.pi), 3))]

        for ix, s in enumerate(data):
            pos += b2Vec2(0, text_surface.get_height())
            # TODO: not ix hardcoded
            if self.mode == UIMode.ROTATE and ix == 2:
                text_surface = text_font.render(
                    s, True, boxcolors.BLACK, boxcolors.GREEN)
            else:
                text_surface = text_font.render(
                    s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
            self.screen.blit(text_surface, pos)

        params = list()
        params.append("Type: {}".format(
            BodyType(self.design_data.params["type"]).name))
        params.append("Reward: {}".format(
            round(self.design_data.params["reward"], 3)))
        params.append("Level: {}".format(
            round(self.design_data.params["level"])))

        if self.design_data.params["type"] == BodyType.MOVING_OBSTACLE or \
                self.design_data.params["type"] == BodyType.MOVING_ZONE:
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
            if ix == self.design_data.params_ix:
                text_surface = text_font.render(
                    s, True, boxcolors.BLACK, boxcolors.GREEN)
            else:
                text_surface = text_font.render(
                    s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
            self.screen.blit(text_surface, pos)
        self.design_surface_height = pos.y - design_pos.y + self.border_width * 3

    def render_commands(self):
        pos = b2Vec2(0, self.title_surface_height + self.border_width)
        # pg.draw.rect(self.screen, boxcolors.BLACK,
        #              pg.Rect(pos.x - self.border_width,
        #                      pos.y,
        #                      self.layout.simulation_xshift + self.border_width * 2,
        #                      self.commands_surface_height), width=self.border_width)

        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        # title
        pos += b2Vec2(self.border_width, self.border_width)
        text_surface = text_font.render(
            "COMMANDS", True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        self.set_commands()
        for command in self.commands:
            pos += b2Vec2(0, text_surface.get_height())
            s = "- {}: {}".format(command["key"], command["description"])
            text_surface = text_font.render(
                s, True, boxcolors.BACK, boxcolors.INFO_BACK)
            self.screen.blit(text_surface, pos)

        self.commands_surface_height = pos.y

    def set_commands(self):
        self.commands.clear()
        if self.mode == UIMode.RESIZE:
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
                             {"key": "LEFT", "description": "decrease parameter inc"},
                             {"key": "A", "description": "rotate"}]
        elif self.mode == UIMode.ROTATE:
            self.commands = [{"key": "A", "description": "finish rotating"},
                             {"key": "mouse movement", "description": "rotate"},
                             {"key": "mouse click", "description": "fix point"}]
        elif self.mode == UIMode.SIMULATION:
            self.commands = [{"key": "M", "description": "manual"}]

        if self.mode not in (UIMode.SIMULATION, UIMode.NONE):
            self.commands.append(
                {"key": "DEL", "description": "delete object"})

        self.commands.append({"key": "ESC", "description": "exit"})

    def render_action(self):
        p1 = self.__pg_coord(self.env.agent_head)
        p2 = self.__pg_coord(self.env.agent_head + self.env.action)

        pg.draw.line(self.screen, boxcolors.ACTION, p1, p2)

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
                    str(distance), False, boxcolors.BLACK, boxcolors.WHITE)
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
                    self.screen, boxcolors.INTERSECTION, end_point, 3)

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
        point = b2Vec2(point.x * self.ppm, (self.env.cfg.world_height -
                       point.y) * self.ppm) + self.layout.simulation_pos
        return point
