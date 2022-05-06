import json
import logging
import math
from time import sleep
from typing import List

import numpy as np
import pygame as pg
from Box2D import b2Body, b2Vec2

import boxcolors
from boxdef import (BodyType, DesignData, EffectType, EffectWhen, EffectWho,
                    ParamGroup, ScreenLayout, SetType, UIMode)
from boxutils import (anglemag_to_vec, dataclass_to_dict, get_design_data,
                      get_intersection, get_line_eq_angle, get_point_angle, copy_design_data, check_design_validity, toggle_enum)

DESIGN_SLEEP = 0.001  # delay in seconds while designing world


class BoxUI():
    def __init__(self, env, screen_layout: ScreenLayout, ppm, target_fps: float, mode: UIMode = UIMode.SIMULATION) -> None:
        self.env = env
        self.layout = screen_layout
        self.ppm = ppm
        self.target_fps = target_fps

        # only setting mode this way here
        # use set_mode() everywhere else
        self.mode = mode
        self.prev_mode = UIMode.NONE

        self.design_bodies: List[DesignData] = list()

        # setting up design variables
        self.design_data = get_design_data(set_type=SetType.DEFAULT)
        self.design_bodies.append(self.design_data)
        # TODO: include in config
        # self.set_type = SetType.DEFAULT
        self.set_type = SetType.PREVIOUS
        # self.set_type = SetType.RANDOM

        # pygame setup
        pg.init()
        pg.font.init()  # to render text
        if self.layout.size == b2Vec2(0, 0):  # fullscreen
            info = pg.display.Info()
            self.screen = pg.display.set_mode(
                (info.current_w, info.current_h), pg.DOUBLEBUF | pg.FULLSCREEN, 32)
        else:
            self.screen = pg.display.set_mode(
                (self.layout.width, self.layout.height), pg.DOUBLEBUF, 32)

        # window title
        pg.display.set_caption('Box Dynamics')
        self.clock = pg.time.Clock()

        # font
        self.font = 'Comic Sans MS'

        # will be updated on runtime based on the real dimensions
        # once the text is rendered
        self.board_y_shift = 0
        self.popup_msg_size = b2Vec2(0, 0)

        # input text from user
        self.text_input = ""

        # design_bodies index when selecting
        self.design_body_ix = 0

        # help variables
        self.shift_pressed = False
        self.prev_mouse_pos = None

        self.commands = list()

        # initially empty list of design_data values that the user
        # can save and use using keys from 0 to 9
        # TODO: implement
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

        self.render_world()
        self.render_back()

        # title
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)
        s = "Mode {}".format(self.mode.name)
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.board_y_shift = text_surface.get_height()
        pos = b2Vec2(self.layout.border, self.layout.border)
        self.screen.blit(text_surface, pos)  # TODO: blit only once

        # TODO: fix that when designing and quit confirmation is asked
        # the observation vectors of the agente can be seen
        if self.mode == UIMode.SIMULATION or (self.prev_mode == UIMode.SIMULATION and self.mode == UIMode.QUIT_CONFIRMATION):
            self.render_action()
            self.render_observations()
            self.render_simulation_data()
            # self.render_state()
        if self.mode in (UIMode.SELECT,
                         UIMode.RESIZE,
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

        # TODO: render commands
        self.render_commands()

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
                    elif self.mode == UIMode.SELECT:
                        self.design_data.points[1] = b2Vec2(pg.mouse.get_pos())
                        self.set_mode(UIMode.RESIZE)

                elif event.type == pg.QUIT:
                    # exit
                    # here no confirmation
                    self.quit()

                elif event.key == pg.K_DELETE:
                    # abort current changes to body
                    if self.shift_pressed:
                        # TODO: ask for confirmation
                        # remove all designs
                        self.design_bodies.clear()
                        pass
                    else:
                        try:
                            self.design_bodies.remove(self.design_data)
                        except IndexError:
                            logging.debug("Can't remove design")
                            pass
                    self.new_design()
                    self.set_mode(UIMode.RESIZE)
                elif event.key == pg.K_LSHIFT:
                    self.shift_pressed = True
                    pass
                elif event.key == pg.K_s:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # asking to input file name
                        self.set_mode(UIMode.INPUT_SAVE)
                    pass
                elif event.key == pg.K_l:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        self.set_mode(UIMode.INPUT_LOAD)
                    pass
                elif event.key == pg.K_u:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # use created world
                        self.set_mode(UIMode.USE_CONFIRMATION)
                    pass
                elif event.key == pg.K_m:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE):
                        # moving body
                        points_num = len(self.design_data.points)
                        if points_num == 2:
                            self.move()
                            self.set_mode(UIMode.MOVE)
                    elif self.mode == UIMode.MOVE:
                        self.set_mode(UIMode.RESIZE)
                    elif self.mode == UIMode.SIMULATION:
                        self.env.manual_mode = not self.env.manual_mode
                    pass
                    pass
                elif event.key == pg.K_a:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.MOVE):
                        # rotating body
                        points_num = len(self.design_data.points)
                        if points_num == 2:
                            self.rotate()
                            self.set_mode(UIMode.ROTATE)
                    elif self.mode == UIMode.ROTATE:
                        self.set_mode(UIMode.RESIZE)
                    pass
                elif event.key == pg.K_e:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # TODO: shift for reverse toggle
                        self.design_data.param_group = toggle_enum(ParamGroup(self.design_data.param_group), skip=[], increase=self.shift_pressed)
                        pass
                elif event.key == pg.K_n:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        self.set_type = toggle_enum(SetType(self.set_type), skip=[], increase=self.shift_pressed)
                        pass
                    pass
                elif event.key == pg.K_UP:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase currently selected param
                        self.modify_param(increase=True)
                    pass
                elif event.key == pg.K_DOWN:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # decrease currently selected param
                        self.modify_param(increase=False)
                    pass
                elif event.key == pg.K_RIGHT:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase inc by factor 10
                        self.design_data.float_inc = self.design_data.float_inc * 10
                    pass
                elif event.key == pg.K_LEFT:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # decrease inc by factor 10
                        self.design_data.float_inc = self.design_data.float_inc / 10
                    pass
                elif event.key == pg.K_SPACE:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # toggle parameter to change
                        self.toggle_param()
                    pass
            elif event.type == pg.KEYUP:
                if event.key == pg.K_LSHIFT:
                    self.shift_pressed = False
                pass
            # pg mouse buttons
            # 2 - middle click
            elif event.type == pg.MOUSEBUTTONDOWN:
                # 1 - left click
                if event.button == 1:
                    if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        self.set_points()
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
                    if self.mode in (UIMode.SELECT, UIMode.ROTATE, UIMode.MOVE, UIMode.RESIZE):
                        self.toggle_design_body()
                        if self.mode != UIMode.SELECT:
                            self.set_mode(UIMode.SELECT)
                # 4 - scroll up
                elif event.button == 4:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase currently selected param
                        self.modify_param(increase=True)
                # 5 - scroll down
                elif event.button == 5:
                    if self.mode in (UIMode.SELECT, UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
                        # increase currently selected param
                        self.modify_param(increase=False)
                pass
            elif event.type == pg.MOUSEMOTION:
                if self.first_exists():
                    if self.mode == UIMode.RESIZE:
                        self.resize()
                    elif self.mode == UIMode.ROTATE:
                        self.rotate()
                    elif self.mode == UIMode.MOVE:
                        self.move()
                        pass
                pass

        self.update_design()

        # post-mode stuff
        if self.prev_mode == UIMode.MOVE:
            self.design_data.moved = False
        elif self.prev_mode == UIMode.ROTATE:
            self.design_data.rotated = False

    def first_exists(self):
        return self.design_data.points[0] is not None

    def second_exists(self):
        return self.design_data.points[1] is not None

    def set_points(self):
        mouse_pos = b2Vec2(pg.mouse.get_pos())
        if not self.first_exists():
            # setting first body point
            self.design_data.points[0] = mouse_pos
        else:
            # setting second body point and
            # getting ready for new design
            self.design_data.points[1] = mouse_pos
            self.design_data.valid = check_design_validity(self.design_data)
            # removing design pointer
            self.design_bodies.remove(self.design_data)
            # appending design copy
            self.design_bodies.append(copy_design_data(self.design_data))
            self.new_design()

    def save_design(self, filename):
        filename = "designs/{}".format(filename)
        dump_db = list()
        for body in self.design_bodies:
            if body.valid:
                dump_db.append(dataclass_to_dict(body))
        try:
            with open(filename, "w") as f:
                json.dump(dump_db, f)
        except FileNotFoundError:
            logging.info("File not found: {}".format(filename))
            return

        logging.info("Saved design {}".format(filename))

    def load_design(self, filename):
        # TODO: bugfix when after loading design
        # and changing loaded bodies
        # the angle changes by 90 deg when creating objects
        # TODO: hardcoded folder name
        filename = "designs/{}".format(filename)
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
            design.physic["type"] = BodyType(design.physic["type"])
            design.effect["type"] = EffectType(design.effect["type"])
            design.effect["who"] = EffectWho(design.effect["who"])
            design.effect["when"] = EffectWhen(design.effect["when"])
            self.design_bodies.append(design)

        logging.info("Loaded design {}".format(filename))

    def new_design(self):
        self.design_data = get_design_data(set_type=self.set_type, current_design=self.design_data)
        self.design_bodies.append(self.design_data)

    def edit_design(self, design: DesignData):
        design_ix = self.design_bodies.index(design)
        self.design_data = self.design_bodies[design_ix]

    def update_design(self):
        def get_primary_vertex():
            return get_point_angle(self.design_data.angle, self.design_data.width, self.design_data.points[0])

        def get_secondary_vertex():
            delta_angle = math.pi / 2
            return get_point_angle(self.design_data.angle + delta_angle, self.design_data.height, self.design_data.points[0])

        def normal_plane_check():
            if self.design_data.normal_plane:
                supposed_p2 = get_primary_vertex()
                supposed_p4 = get_secondary_vertex()
            else:
                supposed_p2 = get_secondary_vertex()
                supposed_p4 = get_primary_vertex()

            old_p2 = self.design_data.vertices[1]
            old_p4 = self.design_data.vertices[3]
            epsilon = 0.1
            d2 = (old_p2 - supposed_p2).length
            d4 = (old_p4 - supposed_p4).length
            if (d2 > epsilon and d4 > epsilon) or (d2 <= epsilon and d4 <= epsilon):
                return True
            else:
                return False

        if self.first_exists() and self.second_exists():
            # dummy variables
            p1 = self.design_data.points[0]
            p3 = self.design_data.points[1]

            # calculating line equations
            line11 = get_line_eq_angle(
                p1, self.design_data.angle)
            line12 = get_line_eq_angle(
                p1, self.design_data.angle + (math.pi/2))
            line21 = get_line_eq_angle(
                p3, self.design_data.angle + (math.pi/2))
            line22 = get_line_eq_angle(
                p3, self.design_data.angle)

            if all(v is not None for v in self.design_data.vertices):
                if self.mode == UIMode.RESIZE:
                    if normal_plane_check():
                        # logging.debug("Resizing in normal plane")
                        # calculating vertices with lines intersection
                        p2 = get_intersection(line11, line21)
                        p4 = get_intersection(line12, line22)

                        # calculating dimensions
                        self.design_data.width = (p1 - p2).length
                        self.design_data.height = (p1 - p4).length

                        self.design_data.normal_plane = True
                    else:
                        # logging.debug("Resizing in abnormal plane")
                        # calculating vertices with lines intersection
                        p2 = get_intersection(line12, line22)
                        p4 = get_intersection(line11, line21)

                        # calculating dimensions
                        self.design_data.width = (p1 - p4).length
                        self.design_data.height = (p1 - p2).length

                        self.design_data.normal_plane = False
                elif self.design_data.normal_plane:
                    # logging.debug("Setting vertices in normal plane")
                    # calculating vertices with lines intersection
                    p2 = get_intersection(line11, line21)
                    p4 = get_intersection(line12, line22)

                    # calculating dimensions
                    self.design_data.width = (p1 - p2).length
                    self.design_data.height = (p1 - p4).length
                else:
                    logging.debug("Setting vertices in abnormal plane")
                    # calculating vertices with lines intersection
                    p2 = get_intersection(line12, line22)
                    p4 = get_intersection(line11, line21)

                    # calculating dimensions
                    self.design_data.width = (p1 - p4).length
                    self.design_data.height = (p1 - p2).length
            else:
                # this only runs the first time a bidy is created
                # and the vertices calculation has never been done
                # the angle should be zero at this point
                logging.debug("Setting vertices for the first time")
                if (p1.x < p3.x and p1.y < p3.y) or (p1.x >= p3.x and p1.y >= p3.y):
                    # calculating vertices with lines intersection
                    p2 = get_intersection(line11, line21)
                    p4 = get_intersection(line12, line22)

                    # calculating dimensions
                    self.design_data.width = (p1 - p2).length
                    self.design_data.height = (p1 - p4).length
                else:
                    # calculating vertices with lines intersection
                    p2 = get_intersection(line12, line22)
                    p4 = get_intersection(line11, line21)

                    # calculating dimensions
                    self.design_data.width = (p1 - p4).length
                    self.design_data.height = (p1 - p2).length

            self.design_data.vertices = [p1, p2, p3, p4]
        pass

    def resize(self):
        assert self.first_exists() and "Must exist one point to resize"
        # setting second point
        mouse_pos = b2Vec2(pg.mouse.get_pos())
        self.design_data.points[1] = mouse_pos

    def move(self):
        mouse_pos = b2Vec2(pg.mouse.get_pos())
        if self.design_data.moved is False:
            self.prev_mouse_pos = b2Vec2(pg.mouse.get_pos())
            self.design_data.moved = True
        else:
            delta_mouse = mouse_pos - self.prev_mouse_pos
            for point in self.design_data.vertices:
                point += delta_mouse
            self.prev_mouse_pos = mouse_pos

            self.design_data.points[0] = self.design_data.vertices[0]
            self.design_data.points[1] = self.design_data.vertices[2]

    def rotate(self):
        mouse_pos = b2Vec2(pg.mouse.get_pos())
        if self.design_data.rotated is False:
            if self.design_data.normal_plane:
                self.design_data.zero_angle = - \
                    math.atan(self.design_data.height /
                              (self.design_data.width))
            else:
                self.design_data.zero_angle = math.atan(
                    self.design_data.height / (self.design_data.width))
            self.design_data.rotated = True
        else:
            diagonal = (
                self.design_data.points[1] - self.design_data.points[0]).length
            diagonal_angle = self.get_angle(
                self.design_data.points[0], mouse_pos)
            self.design_data.angle = diagonal_angle + self.design_data.zero_angle

            self.design_data.points[1] = self.design_data.points[0] + \
                anglemag_to_vec(angle=diagonal_angle, magnitude=diagonal)

    def modify_param(self, increase=True):
        if self.design_data.param_group == ParamGroup.PHYSIC:
            # currently changin parameters
            name = list(self.design_data.physic)[self.design_data.physic_ix]
            if name == "type":
                self.toggle_body_type(increase)
            elif name == "level":
                if increase:
                    self.design_data.physic[name] += 1
                else:
                    self.design_data.physic[name] -= 1
            else:
                if increase:
                    self.design_data.physic[name] += self.design_data.float_inc
                else:
                    self.design_data.physic[name] -= self.design_data.float_inc
        # TODO: could be more than two groups!!
        elif self.design_data.param_group == ParamGroup.EFFECT:
            name = list(self.design_data.effect)[self.design_data.effect_ix]
            if name in ("type", "who", "when"):
                self.design_data.effect[name] = toggle_enum(
                    self.design_data.effect[name], [], increase)
            else:
                if increase:
                    self.design_data.effect[name] += self.design_data.float_inc
                else:
                    self.design_data.effect[name] -= self.design_data.float_inc
        pass

    def toggle_design_body(self):
        if len(self.design_bodies):
            if self.shift_pressed == False:
                inc = 1
            else:
                inc = -1
            if self.design_data == self.design_bodies[self.design_body_ix]:
                self.design_body_ix = (
                    self.design_body_ix + inc) % len(self.design_bodies)
            self.design_data = self.design_bodies[self.design_body_ix]
            # self.design_data.points[1] = b2Vec2(pg.mouse.get_pos())

    def toggle_body_type(self, increase=True):
        self.design_data.physic["type"] = toggle_enum(
            BodyType(self.design_data.physic["type"]), skip=[BodyType.AGENT, BodyType.BORDER, BodyType.DEFAULT], increase=increase)
        self.design_data.color = self.env.get_data(
            self.design_data.physic["type"]).color

    def set_commands(self):
        self.commands.clear()
        # TODO: shift commands
        if self.mode in (UIMode.RESIZE, UIMode.ROTATE, UIMode.MOVE):
            self.commands = [{"key": "left click", "description": "fix point"},
                             {"key": "right click", "description": "toggle body"},
                             {"key": "A", "description": "rotate"},
                             {"key": "M", "description": "move"},
                             {"key": "U", "description": "use"},
                             {"key": "S", "description": "save"},
                             {"key": "L", "description": "load"},
                             {"key": "SPACE", "description": "toggle parameter"},
                             {"key": "(roll) UP",
                              "description": "increase parameter"},
                             {"key": "(roll) DOWN",
                              "description": "decrease parameter"},
                             {"key": "RIGHT", "description": "increase parameter inc"},
                             {"key": "LEFT", "description": "decrease parameter inc"},
                             {"key": "DEL", "description": "delete object"},
                             {"key": "ESC", "description": "exit"}]
        elif self.mode in (UIMode.INPUT_LOAD, UIMode.INPUT_SAVE, UIMode.QUIT_CONFIRMATION, UIMode.USE_CONFIRMATION):
            self.commands = [{"key": "ENTER", "description": "confirm"},
                             {"key": "ESC", "description": "cancel"}]
        elif self.mode == UIMode.SIMULATION:
            self.commands = [{"key": "M", "description": "manual"}]

    def toggle_param(self):
        if self.shift_pressed == False:
            inc = 1
        else:
            inc = -1
        if self.design_data.param_group == ParamGroup.PHYSIC:
            # we are toggling design physic
            if self.design_data.physic["type"] in (BodyType.DYNAMIC_OBSTACLE,
                                                   BodyType.DYNAMIC_ZONE,
                                                   BodyType.KINEMATIC_OBSTACLE,
                                                   BodyType.KINEMATIC_ZONE):
                self.design_data.physic_ix = (
                    self.design_data.physic_ix + inc) % len(list(self.design_data.physic))
            else:
                # TODO: only toggle static bodies physic not hardcoded
                self.design_data.physic_ix = (
                    self.design_data.physic_ix + inc) % 3
        elif self.design_data.param_group == ParamGroup.EFFECT:
            # we are toggling effect physic
            if self.design_data.effect["type"] in (EffectType.APPLY_FORCE,
                                                   EffectType.SET_VELOCITY):
                # all effect keys needed
                self.design_data.effect_ix = (
                    self.design_data.effect_ix + inc) % len(self.design_data.effect)
            elif self.design_data.effect["type"] in (EffectType.SET_LIN_DAMP,
                                                     EffectType.SET_ANG_DAMP,
                                                     EffectType.SET_FRICTION,
                                                     EffectType.SET_MAX_ACTION,
                                                     EffectType.BOUNCE):
                # key param_1 not needed
                self.design_data.effect_ix = (
                    self.design_data.effect_ix + inc) % (len(self.design_data.effect) - 1)
            elif self.design_data.effect["type"] in (EffectType.DONE,
                                                     EffectType.RESET,
                                                     EffectType.INVERT_VELOCITY):
                # keys param_0 and param_1 not needed
                self.design_data.effect_ix = (
                    self.design_data.effect_ix + inc) % (len(self.design_data.effect) - 2)
            elif self.design_data.effect["type"] == EffectType.NONE:
                # only key type needed
                self.design_data.effect_ix = (
                    self.design_data.effect_ix + inc) % 1
            pass
        else:
            assert False and "Unknown parameters group"

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

    def render_world(self):
        # Draw the world based on bodies levels
        sorted_bodies: List[b2Body] = self.get_sorted_bodies(design=False)

        for body in sorted_bodies:
            # skipping border rendering cause not necessary
            if body.userData.type == BodyType.BORDER:
                # continue
                pass
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

    def get_simulation_data(self):
        def get_fps():
            return round(self.clock.get_fps())
        def get_time():
            # TODO: start time when starting simulation and not during design
            return round(pg.time.get_ticks() / 1000, 1)
        def get_total_reward():
            return round(self.env.total_reward, self.layout.ndigits)
        def get_agent_lin_damp():
            return round(self.env.agent_body.linearDamping, self.layout.ndigits)
        def get_agent_ang_damp():
            return round(self.env.agent_body.angularDamping, self.layout.ndigits)
        def get_agent_friction():
            return round(self.env.agent_body.fixtures[0].friction, self.layout.ndigits)

        data = [{"name": "FPS", "value": get_fps()},
                {"name": "TIME", "value": get_time()},
                {"name": "REWARD", "value": get_total_reward()},
                {"name": "LIN DAMP", "value": get_agent_lin_damp()},
                {"name": "ANG DAMP", "value": get_agent_ang_damp()},
                {"name": "FRICTION", "value": get_agent_friction()}]
        
        return data

    def render_simulation_data(self):
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        # title
        pos = b2Vec2(self.layout.border, self.board_y_shift +
                     (2*self.layout.border))
        text_surface = text_font.render(
            "SIMULATION DATA", True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)
        for data in self.get_simulation_data():
            pos += b2Vec2(0, text_surface.get_height())
            s = "{}: {}".format(data["name"], data["value"])
            if data["name"] == "FPS":
                back_color = boxcolors.GREEN if abs(data["value"] - self.target_fps) < 10 else boxcolors.RED
            elif data["name"] == "REWARD":
                back_color = boxcolors.GREEN if data["value"] >= 0 else boxcolors.RED
            else:
                back_color = boxcolors.INFO_BACK
            text_surface = text_font.render(
                s, True, boxcolors.BLACK, back_color)
            self.screen.blit(text_surface, pos)

        self.board_y_shift = pos.y
        pass

    def render_state(self):
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        # title
        pos = b2Vec2(self.layout.border, self.board_y_shift +
                     (2*self.layout.border))
        text_surface = text_font.render(
            "STATE", True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)
        for data in self.env.state:
            pos += b2Vec2(0, text_surface.get_height())
            s = "{}".format(self.env.state[data])
            text_surface = text_font.render(
                s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
            self.screen.blit(text_surface, pos)

        self.board_y_shift = pos.y
        pass

    def render_design(self):
        # Draw the world based on bodies levels
        sorted_bodies: List[DesignData] = self.get_sorted_bodies(design=True)

        body_vertices = [b2Vec2(-1, -1)]*4

        for body in sorted_bodies:
            color = self.env.get_data(body.physic["type"]).color
            try:
                pg.draw.polygon(
                    self.screen, color, body.vertices)
                # showing circle on current design
                if body == self.design_data:
                    for vix, v in enumerate(body.vertices):
                        body_vertices[vix] = v.copy()
            except TypeError:
                # wait another cycle for the vertices to be there
                pass

        for vix, v in enumerate(body_vertices):
            pg.draw.line(self.screen, boxcolors.YELLOW, v,
                         body_vertices[(vix + 1) % len(body_vertices)])
            if vix == 0:
                pg.draw.circle(self.screen, boxcolors.YELLOW,
                               v, self.layout.big_dot_radius)
            else:
                pg.draw.circle(self.screen, boxcolors.YELLOW,
                               v, self.layout.small_dot_radius)

        self.render_design_data()
        self.render_effects()

    def render_design_data(self):
        # rendering design data when user is designin shape
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        pos = b2Vec2(self.layout.border,
                     self.board_y_shift + (2*self.layout.border))

        # title
        s = "DESIGN DATA"
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        # data can't be toggled like physic
        data = ["Next: {}".format(self.set_type.name),
                "Angle: {}".format(
                round(-360 * self.design_data.angle / (2 * math.pi), 3))]

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

        # NOTICE: parameters must be in the same order as
        # when initialized in 
        physic = list()
        physic.append("Type: {}".format(
            BodyType(self.design_data.physic["type"]).name))
        physic.append("Reward: {}".format(
            round(self.design_data.physic["reward"], self.layout.ndigits)))
        physic.append("Level: {}".format(
            round(self.design_data.physic["level"])))

        if self.design_data.physic["type"] in (BodyType.DYNAMIC_OBSTACLE,
                                               BodyType.DYNAMIC_ZONE,
                                               BodyType.KINEMATIC_OBSTACLE,
                                               BodyType.KINEMATIC_ZONE):
            physic.append("Velocity: {}".format(
                round(self.design_data.physic["lin_velocity"], self.layout.ndigits)))
            physic.append("Velocity angle: {}".format(
                round(self.design_data.physic["lin_velocity_angle"], self.layout.ndigits)))
            physic.append("Angular velocity: {}".format(
                round(self.design_data.physic["ang_velocity"], self.layout.ndigits)))
            physic.append("Density: {}".format(
                round(self.design_data.physic["density"], self.layout.ndigits)))
            physic.append("Inertia: {}".format(
                round(self.design_data.physic["inertia"], self.layout.ndigits)),)
            physic.append("Friction: {}".format(
                round(self.design_data.physic["friction"], self.layout.ndigits)))
            physic.append("Linear damping: {}".format(
                round(self.design_data.physic["lin_damping"], self.layout.ndigits)))
            physic.append("Angular damping: {}".format(
                round(self.design_data.physic["ang_damping"], self.layout.ndigits)))

        # increment always as last parameter
        physic.append("Increment: {}".format((self.design_data.float_inc)))

        for ix, s in enumerate(physic):
            pos += b2Vec2(0, text_surface.get_height())
            if self.design_data.param_group == ParamGroup.PHYSIC and ix == self.design_data.physic_ix:
                # highlight current param
                text_surface = text_font.render(
                    "* {}".format(s), True, boxcolors.BLACK, boxcolors.GREEN)
            else:
                text_surface = text_font.render(
                    "- {}".format(s), True, boxcolors.BLACK, boxcolors.INFO_BACK)
            self.screen.blit(text_surface, pos)

        self.board_y_shift = pos.y

    def render_effects(self):
        # rendering design data when user is designin shape
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        pos = b2Vec2(self.layout.border,
                     self.board_y_shift + (2*self.layout.border))

        # title
        s = "EFFECTS"
        text_surface = text_font.render(
            s, True, boxcolors.BLACK, boxcolors.INFO_BACK)
        self.screen.blit(text_surface, pos)

        text_font = pg.font.SysFont(
            self.font, self.layout.normal_font)

        effect = list()
        effect.append("Type: {}".format(
            EffectType(self.design_data.effect["type"]).name))

        if self.design_data.effect["type"] in (EffectType.APPLY_FORCE,
                                               EffectType.SET_VELOCITY):
            effect.append("Who: {}".format(
                EffectWho(self.design_data.effect["who"]).name))
            effect.append("When: {}".format(EffectWhen(
                self.design_data.effect["when"]).name))
            effect.append("Param A: {}".format(
                round(self.design_data.effect["param_0"], self.layout.ndigits)))
            effect.append("Param B: {}".format(
                round(self.design_data.effect["param_1"], self.layout.ndigits)))
        elif self.design_data.effect["type"] in (EffectType.SET_LIN_DAMP,
                                                 EffectType.SET_ANG_DAMP,
                                                 EffectType.SET_FRICTION,
                                                 EffectType.SET_MAX_ACTION,
                                                 EffectType.BOUNCE):
            effect.append("Who: {}".format(
                EffectWho(self.design_data.effect["who"]).name))
            effect.append("When: {}".format(EffectWhen(
                self.design_data.effect["when"]).name))
            effect.append("Param A: {}".format(
                round(self.design_data.effect["param_0"], self.layout.ndigits)))
        elif self.design_data.effect["type"] in (EffectType.DONE,
                                                 EffectType.RESET,
                                                 EffectType.INVERT_VELOCITY):
            effect.append("Who: {}".format(
                EffectWho(self.design_data.effect["who"]).name))
            effect.append("When: {}".format(EffectWhen(
                self.design_data.effect["when"]).name))

        for ix, s in enumerate(effect):
            if self.design_data.param_group == ParamGroup.EFFECT and ix == self.design_data.effect_ix:
                # highlight current param
                back_color = boxcolors.GREEN
                s = "* {}".format(s)
            else:
                back_color = boxcolors.INFO_BACK
                s = "- {}".format(s)

            pos += b2Vec2(0, text_surface.get_height())
            text_surface = text_font.render(
                s, True, boxcolors.BLACK, back_color)
            self.screen.blit(text_surface, pos)

        self.board_y_shift = pos.y

    def render_commands(self):
        text_font = pg.font.SysFont(
            self.font, self.layout.big_font)

        # title
        pos = b2Vec2(self.layout.border, self.board_y_shift +
                     (2*self.layout.border))
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

        self.board_y_shift = pos.y

    def render_action(self):
        p1 = self.__pg_coord(self.env.agent_head)
        p2 = self.__pg_coord(self.env.agent_head + self.env.action)

        pg.draw.line(self.screen, boxcolors.ACTION, p1, p2)

    def render_distances(self):
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
                    self.screen, boxcolors.INTERSECTION, end_point, self.layout.normal_dot_radius)
        # drawing text distances
        if self.env.cfg.render_distances:
            self.render_distances()


    def get_sorted_bodies(self, design=False):
        # TODO: have them already sorted for performance
        if design:
            bodies_levels: List[tuple(DesignData, int)] = [[b, b.physic["level"]]
                                                           for b in self.design_bodies]
            bodies_levels.sort(key=lambda x: x[1], reverse=False)
        else:
            bodies_levels = [[b, b.userData.level]
                             for b in self.env.world.bodies]
            bodies_levels.sort(key=lambda x: x[1], reverse=False)

        return [body[0] for body in bodies_levels]


    # transform point in world coordinates to point in pg coordinates
    def __pg_coord(self, point):
        point = b2Vec2(point.x * self.ppm, (self.env.cfg.world_height -
                       point.y) * self.ppm) + self.layout.simulation_pos
        return point
