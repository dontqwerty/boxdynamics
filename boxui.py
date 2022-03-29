
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from logging import debug
from multiprocessing.sharedctypes import Value
from time import sleep
from typing import List

import pygame
from Box2D import b2Vec2

import boxcolors as color
from boxdef import BodyShape, BodyType
from boxutils import get_intersection, get_line_eq_angle

DESIGN_SLEEP = 0.01  # delay in seconds while designing world


class Mode(Enum):
    NONE = 0
    WORLD_DESIGN = 1  # create box, circle, save, use
    SIMULATION = 2


@dataclass
class ScreenLayout:
    width: int = 800  # pixels
    height: int = 800
    size: b2Vec2 = b2Vec2(width, height)
    simulation_xshift = 100
    simulation_yshift = 70
    simulation_pos: b2Vec2 = b2Vec2(simulation_xshift, simulation_yshift)
    simulation_size: b2Vec2 = b2Vec2(600, 600)
    board_pos: b2Vec2 = b2Vec2(0, simulation_size.y)
    board_size: b2Vec2 = b2Vec2(width, height) * 0
    small_font: int = 14
    normal_font: int = 16
    big_font: int = 24


@dataclass
class DesignData:
    shape: Enum = BodyShape.BOX  # used if creating
    selected: bool = False  # TODO: selectable

    points: List[b2Vec2] = field(default_factory=list)  # center and radius for circles
    vertices: List[b2Vec2] = field(default_factory=list) # all four vertices for rectangles
    rotated = False
    alpha: float = 0
    beta: float = 0
    angle: float = 0
    gamma: float = 0
    mouse_radius: float = None
    rotating: bool = False  # used to rotate the object
    type: Enum = BodyType.STATIC_OBSTACLE
    color: Enum = color.BROWN
    level: int = 0


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

        if self.mode == Mode.WORLD_DESIGN:
            self.render_design()

        self.render_world()

        if self.mode == Mode.SIMULATION:
            self.render_action()
            self.render_observations()
            self.draw_distances()

        pygame.display.flip()
        self.clock.tick(self.target_fps)
        pass

    def user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # TODO: quit signal to BoxEnv
                self.quit()
                return

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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_u:
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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    # angle
                    points_num = len(self.design_data.points)
                    if points_num > 0:
                        self.design_data.rotating = not self.design_data.rotating
                        if self.design_data.rotating is False:
                            self.design_data.mouse_radius = None
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    # level up
                    self.design_data.level += 1
                    pass
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    # level down
                    self.design_data.level -= 1
                    pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # TODO: check for mouse pos and perform action like select bodies
                    mouse_pos = pygame.mouse.get_pos()

                    points_num = len(self.design_data.points)
                    if points_num == 0:
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

                        if self.design_data.rotating:
                            # deltas between p1 and mouse pos
                            delta_x = abs(
                                mouse_pos.x - self.design_data.points[0].x)
                            delta_y = abs(
                                mouse_pos.y - self.design_data.points[0].y)

                            if self.design_data.rotated is False:
                                # first time rotating body
                                self.design_data.rotated = True
                                self.design_data.mouse_radius = (
                                    self.design_data.points[0] - mouse_pos).length

                                # first quadrant
                                if mouse_pos.x >= self.design_data.points[0].x and mouse_pos.y >= self.design_data.points[0].y:
                                    self.design_data.angle = math.atan(
                                        delta_y / delta_x)
                                    self.design_data.beta = 0
                                    self.design_data.gamma = math.pi / 2
                                    pass
                                # second quadrant
                                elif mouse_pos.x <= self.design_data.points[0].x and mouse_pos.y >= self.design_data.points[0].y:
                                    self.design_data.angle = (
                                        math.pi) - math.atan(delta_y / delta_x)
                                    self.design_data.beta = math.pi / 2
                                    self.design_data.gamma = math.pi
                                    pass
                                # third quadrant
                                elif mouse_pos.x <= self.design_data.points[0].x and mouse_pos.y <= self.design_data.points[0].y:
                                    self.design_data.angle = (
                                        math.pi) + math.atan(delta_y / delta_x)
                                    self.design_data.beta = math.pi
                                    self.design_data.gamma = 3 * math.pi / 2
                                    pass
                                # fourth quadrant
                                elif mouse_pos.x >= self.design_data.points[0].x and mouse_pos.y <= self.design_data.points[0].y:
                                    self.design_data.angle = - \
                                        math.atan(delta_y / delta_x)
                                    self.design_data.beta = 3 * math.pi / 2
                                    self.design_data.gamma = 2 * math.pi
                                    pass

                                self.design_data.alpha = self.design_data.angle

                            else:
                                # second or more time rotating
                                # first quadrant
                                if mouse_pos.x > self.design_data.points[0].x and mouse_pos.y > self.design_data.points[0].y:
                                    self.design_data.alpha = math.atan(
                                        delta_y / delta_x)
                                    pass
                                # second quadrant
                                elif mouse_pos.x < self.design_data.points[0].x and mouse_pos.y > self.design_data.points[0].y:
                                    self.design_data.alpha = (
                                        math.pi) - math.atan(delta_y / delta_x)
                                    pass
                                # third quadrant
                                elif mouse_pos.x < self.design_data.points[0].x and mouse_pos.y < self.design_data.points[0].y:
                                    self.design_data.alpha = (
                                        math.pi) + math.atan(delta_y / delta_x)
                                    pass
                                # fourth quadrant
                                elif mouse_pos.x > self.design_data.points[0].x and mouse_pos.y < self.design_data.points[0].y:
                                    self.design_data.alpha = - \
                                        math.atan(delta_y / delta_x)
                                    pass


                            # TODO: second round of rotating
                            if self.design_data.mouse_radius is None:
                                self.design_data.angle = self.design_data.alpha
                                self.design_data.mouse_radius = (
                                    self.design_data.points[0] - mouse_pos).length

                            mouse_pos = self.design_data.points[0] + b2Vec2(math.cos(self.design_data.alpha), math.sin(
                                self.design_data.alpha)) * self.design_data.mouse_radius
                            pass

                        if points_num == 2:
                            self.design_data.vertices = self.get_vertices(self.design_data)
                            # replacing old point with new point on mouse position
                            self.design_data.points.pop()

                        self.design_data.points.append(b2Vec2(mouse_pos))

                        # new updated body
                        self.design_bodies.append(self.design_data)
                    pass

            elif self.mode == Mode.SIMULATION:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    # TODO: toggle manual mode
                    self.env.manual_mode = not self.env.manual_mode
                    pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # TODO: check for mouse pos and perform action
                    # show body properties when clicked
                    pass

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
        pygame.draw.polygon(self.screen, color.GREY, vertices)

        vertices = [(0, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.simulation_yshift),
                    (0, self.layout.simulation_yshift)]
        pygame.draw.polygon(self.screen, color.GREY, vertices)

        vertices = [(self.layout.simulation_pos.x + self.layout.simulation_size.x, 0),
                    (self.layout.width, 0),
                    (self.layout.width, self.layout.height),
                    (self.layout.simulation_pos.x + self.layout.simulation_size.x, self.layout.height)]
        pygame.draw.polygon(self.screen, color.GREY, vertices)

        vertices = [(0, self.layout.simulation_pos.y + self.layout.simulation_size.y),
                    (self.layout.width, self.layout.simulation_pos.y +
                     self.layout.simulation_size.y),
                    (self.layout.width, self.layout.height),
                    (0, self.layout.height)]
        pygame.draw.polygon(self.screen, color.GREY, vertices)

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

        self.render_corner_text()

        # render text: commands, current body properties

    def render_corner_text(self):
        # rendering infos when user is designin shape
        text_font = pygame.font.SysFont(
            'Comic Sans MS', self.layout.normal_font)

        pos = self.layout.simulation_pos
        s = "Shape: {}".format(self.design_data.shape)
        text_surface = text_font.render(
            s, True, color.BLACK, color.GREEN)
        self.screen.blit(text_surface, pos)

        pos = self.layout.simulation_pos + b2Vec2(0, text_surface.get_height())
        s = "Type: {}".format(self.design_data.type)
        text_surface = text_font.render(
            s, True, color.BLACK, color.GREEN)
        self.screen.blit(text_surface, pos)

        pos = self.layout.simulation_pos + \
            b2Vec2(0, 2*text_surface.get_height())
        s = "Level: {}".format(self.design_data.level)
        text_surface = text_font.render(
            s, True, color.BLACK, color.GREEN)
        self.screen.blit(text_surface, pos)

        pass

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

    def get_vertices(self, body: DesignData):
        p1 = body.points[0]
        p3 = body.points[1]
        if body.rotated:

            line11 = get_line_eq_angle(
                p1, body.beta + (body.alpha - body.angle))
            line12 = get_line_eq_angle(
                p1, body.gamma + (body.alpha - body.angle))

            line21 = get_line_eq_angle(
                p3, body.gamma + (body.alpha - body.angle))
            line22 = get_line_eq_angle(
                p3, body.beta + (body.alpha - body.angle))

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
