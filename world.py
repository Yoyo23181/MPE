import sys
import os

import torch

import pyglet
from pyglet.text import Label

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from MySimpleEnv import mySimpleEnv
from mpe2._mpe_utils.core import Agent
from pettingzoo.utils.conversions import parallel_wrapper_fn
from agent import IA2CAgent


class my_raw_env(mySimpleEnv, EzPickle):
    def __init__(self, max_cycles=2500, continuous_actions=True, render_mode="human",
                 eval_mode=False,
                 food_positions=None):
        dynamic_rescaling = True
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            eval_mode=eval_mode,
            food_positions=food_positions
        )

        scenario = Scenario(eval_mode=eval_mode, food_positions=food_positions)
        world = scenario.make_world()

        world.eval_mode = eval_mode
        world.food_positions = (np.array(food_positions, dtype=np.float32)
                                if food_positions is not None else None)
        world.food_idx = 0

        mySimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_v3"



number_agent = 1
number_landmark = 1


class Scenario(BaseScenario):
    def __init__(self, number_agent = 1, number_landmark = 1,
                 eval_mode=False, food_positions=None):
        self.number_agent = number_agent
        self.number_landmark = number_landmark
        self.eval_mode = eval_mode
        self.food_positions = food_positions or []


    def make_world(self, training = True):
        world = World()
        world.dim_c = 0
        world.collaborative = False

        num_lm = self.number_landmark
        dim_p = world.dim_p
        dim_c = world.dim_c
        state_dim = 2
        action_dim = dim_p + dim_c

        world.agents = [Agent() for i in range(self.number_agent)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = False
            agent.weight = round(np.random.uniform(1, 1))
        # add landmarks
        world.landmarks = [Landmark() for i in range(number_landmark)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.weight = round(np.random.uniform(1, 1))
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        weight_to_color = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([1.0, 0.5, 0.0]),
            3: np.array([1.0, 1.0, 0.0]),
            4: np.array([0.0, 1.0, 0.0]),
            5: np.array([0.0, 0.0, 1.0]),
        }
        default_color = np.array([0.75, 0.75, 0.75])

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

            if getattr(world, "eval_mode", False):
                agent.state.p_pos = np.array([0.0, 0.0])
            else:
                agent.state.p_pos = np.random.uniform(-5, +5, world.dim_p)
            agent.state.p_vel = np.array([0.0, 0.0])
            agent.state.c = np.zeros(world.dim_c)

            agent.prev_distance = None
            agent.previous_theta = None
            agent.entity_speed = 0
            agent._consumed_this_step = False

        # random or fixed properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = weight_to_color.get(landmark.weight, default_color)
            if getattr(world, "eval_mode", False) and getattr(world, "food_positions", None) is not None and len(world.food_positions) > 0:

                idx = getattr(world, "food_idx", 0) % len(world.food_positions)
                xy = world.food_positions[idx]
                landmark.state.p_pos = np.array([xy[0], xy[1]], dtype=np.float32)
            else:
                place = False
                while not place:
                    can_place = True
                    for agent in world.agents:
                        landmark.state.p_pos = np.random.uniform(-8, +8, world.dim_p)
                        if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) < 4:
                            can_place = False
                    if can_place:
                        place = True
            landmark.state.p_vel = np.zeros(world.dim_c)

        print("World reset:")

    def reward(self, agent, world):
        reward = 0
        radius = 1.0
        punishment_distance = 10.0
        consumed = False

        for landmark in world.landmarks:
            current_distance = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)



            if agent.prev_distance is not None:
                speed = agent.prev_distance - current_distance
                agent.entity_speed = speed
                reward += speed*10
            agent.prev_distance = current_distance

            nearby_agents = [a for a in world.agents if np.linalg.norm(a.state.p_pos - landmark.state.p_pos) < radius]
            total_weight = sum([a.weight for a in nearby_agents])

            if current_distance <= radius:
                reward += 15.0
                agent.prev_distance = None  # reset the previous distance
                consumed = True
                agent._consumed_this_step = True

                if getattr(world, "eval_mode", False) and getattr(world, "food_positions", None) is not None and len(world.food_positions) > 0:
                    world.food_idx = (getattr(world, "food_idx", 0) + 1) % len(world.food_positions)
                    xy = world.food_positions[world.food_idx]
                    landmark.state.p_pos = np.array([xy[0], xy[1]], dtype=np.float32)
                    landmark.state.p_vel = np.zeros(world.dim_p)
                else:
                    # original random respawn
                    while True:
                        landmark.state.p_pos = np.random.uniform(-5, +5, world.dim_p)
                        if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) > 4:
                            break
                    landmark.state.p_vel = np.zeros(world.dim_p)
                break

        if not consumed:
            agent._consumed_this_step = False

        return reward

    def observation(self, agent, world):
        entity_pos = []

        for entity in world.landmarks:
            pos = entity.state.p_pos - agent.state.p_pos
            entity_pos.append(pos)
        return np.concatenate([np.array(entity_pos).flatten(),])
