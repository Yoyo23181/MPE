from mpe2._mpe_utils.simple_env import SimpleEnv

import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.core import Landmark, World

from agent import IA2CAgent

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class mySimpleEnv(SimpleEnv):
    def __init__(
            self, scenario, world, max_cycles,
            render_mode = None, continuous_actions = False, local_ratio = None,
            dynamic_rescaling = False, ):
        super().__init__(scenario, world, max_cycles,
                render_mode , continuous_actions , local_ratio,
                dynamic_rescaling )

    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        if agent.movable:
            # act = np.array([(action[0][0]-0.5)*2, (action[0][1]-0.5)*2])
            # act = act / np.linalg.norm(act)
            # create a vector with the direction of the action
            dir = (action[0][0]-0.5)*2*np.pi
            act = np.array([np.cos(dir), np.sin(dir)])


            agent.action.u= act



        #
        #     # physical action
        #     agent.action.u = np.zeros(self.world.dim_p)
        #     if self.continuous_actions:
        #         # Process continuous action as in OpenAI MPE
        #         # Note: this ordering preserves the same movement direction as in the discrete case
        #         agent.action.u[0] += action[0][2] - action[0][1]
        #         agent.action.u[1] += action[0][4] - action[0][3]
        #     else:
        #         # process discrete action
        #         if action[0] == 1:
        #             agent.action.u[0] = -1.0
        #         if action[0] == 2:
        #             agent.action.u[0] = +1.0
        #         if action[0] == 3:
        #             agent.action.u[1] = -1.0
        #         if action[0] == 4:
        #             agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = 0
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

        def draw(self):
            return
            # clear screen
            self.screen.fill((255, 255, 255))

            # update bounds to center around agent
            all_poses = [entity.state.p_pos for entity in self.world.entities]
            cam_range = 20

            # The scaling factor is used for dynamic rescaling of the rendering - a.k.a Zoom In/Zoom Out effect
            # The 0.9 is a factor to keep the entities from appearing "too" out-of-bounds
            scaling_factor = 0.9 * self.original_cam_range / cam_range

            # update geometry and text positions
            text_line = 0
            for e, entity in enumerate(self.world.entities):
                # geometry
                x, y = entity.state.p_pos
                y *= (
                    -1
                )  # this makes the display mimic the old pyglet setup (ie. flips image)
                x = (
                        (x / cam_range) * self.width // 2 * 0.9
                )  # the .9 is just to keep entities from appearing "too" out-of-bounds
                y = (y / cam_range) * self.height // 2 * 0.9
                x += self.width // 2
                y += self.height // 2

                # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
                if self.dynamic_rescaling:
                    radius = entity.size * 350 * scaling_factor
                else:
                    radius = entity.size * 350

                pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)  # borders
                assert (
                        0 < x < self.width and 0 < y < self.height
                ), f"Coordinates {(x, y)} are out of bounds."

                # text
                if isinstance(entity, Agent):
                    if entity.silent:
                        continue
                    if np.all(entity.state.c == 0):
                        word = "_"
                    elif self.continuous_actions:
                        word = (
                                "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                        )
                    else:
                        word = alphabet[np.argmax(entity.state.c)]

                    message = entity.name + " sends " + word + "   "
                    message_x_pos = self.width * 0.05
                    message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                    self.game_font.render_to(
                        self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                    )
                    text_line += 1


class MyScenario(BaseScenario):
    def __init__(self, number_agent=1, number_landmark=1):
        self.number_agent = number_agent
        self.number_landmark = number_landmark
    def make_world(self):
        world = World()
        world.dim_c = 0 # make communication trought c possible?
        world.collaborative = False
        # add agents
        # world.agents = [Agent() for i in range(number_agent)]

        num_lm = 1
        dim_p = world.dim_p
        dim_c = world.dim_c
        # print("weight", dim_p, dim_c)
        state_dim = 2 + 2 + num_lm*dim_p + num_lm + dim_c
        action_dim = dim_p + dim_c

        world.agents = [IA2CAgent(state_dim=state_dim, action_dim=action_dim) for i in range(self.number_agent)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = False
            agent.weight = round(np.random.uniform(1, 1))
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.number_landmark)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.weight = round(np.random.uniform(1, 1))
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        weight_to_color = {
            1: np.array([1.0, 0.0, 0.0]),  # red
            2: np.array([1.0, 0.5, 0.0]),  # orange
            3: np.array([1.0, 1.0, 0.0]),  # yellow
            4: np.array([0.0, 1.0, 0.0]),  # green
            5: np.array([0.0, 0.0, 1.0]),  # blue
        }
        default_color = np.array([0.75, 0.75, 0.75])

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            agent.state.p_pos = np.random.uniform(-5, +5, world.dim_p)
            agent.state.p_vel = np.array([0.0, 0.0])
            agent.state.c = np.zeros(world.dim_c)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.75, 0.75, 0.75])
            landmark.color = weight_to_color.get(landmark.weight, default_color)
            place = False
            while not place:
                can_place = True
                for agent in world.agents:
                    landmark.state.p_pos = np.random.uniform(-8, +8, world.dim_p)
                    if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) < 3:
                        can_place = False
                if can_place:
                    place = True

            # landmark.state.p_vel = np_random.uniform(-0.5, 0.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        print("World reset")