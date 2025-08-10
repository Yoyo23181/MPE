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
from MyCustomEnv import mySimpleEnv
from mpe2._mpe_utils.core import Agent
# from agilerl.algorithms.matd3 import MATD3


from pettingzoo.utils.conversions import parallel_wrapper_fn
from new_agent import IA2CAgent
# from agents.mimic_agent import Agent



class my_raw_env(mySimpleEnv, EzPickle):
    def __init__(self, max_cycles=2500, continuous_actions=True, render_mode="human"):

        # self.render_mode = render_mode
        dynamic_rescaling = True
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )

        scenario = Scenario()
        world = scenario.make_world()

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



# env = make_env(raw_env)
# parallel_env = parallel_wrapper_fn(env)

number_agent = 1
number_landmark = 1


class Scenario(BaseScenario):
    def __init__(self, number_agent = 1, number_landmark = 1):
        self.number_agent = number_agent
        self.number_landmark = number_landmark
    def make_world(self, training = True):
        world = World()
        world.dim_c = 0 # make communication trought c possible?
        world.collaborative = False
        # add agents
        # world.agents = [Agent() for i in range(number_agent)]

        num_lm = self.number_landmark
        dim_p = world.dim_p
        dim_c = world.dim_c
        # print("weight", dim_p, dim_c)
        state_dim = 2 + 2 + num_lm*dim_p + num_lm + dim_c
        state_dim = 2
        action_dim = dim_p + dim_c
        # print("weight", state_dim, action_dim)
#         learning_rate = 0.001
#         n_episodes = 1_000_000
#         start_epsilon = 1.0
#         epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
#         final_epsilon = 0.1
#         world.agents = [IA2CAgent(env=env,learning_rate=learning_rate,initial_epsilon=start_epsilon,epsilon_decay=epsilon_decay,final_epsilon=final_epsilon,state_dim=state_dim,action_dim=action_dim,
# ) for i in range(number_agent)]
        world.agents = [Agent() for i in range(self.number_agent)]

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # path = "./models/MATD3/MATD3_trained_agent.pt"
        # matd3 = MATD3.load(path, device)

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
            agent.prev_distance = None
            agent.previous_food_pos = None
            agent.entity_speed=0
            agent.previous_theta =None
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.75, 0.75, 0.75])
            landmark.color = weight_to_color.get(landmark.weight, default_color)
            place = False
            while not place:
                can_place = True
                for agent in world.agents:
                    landmark.state.p_pos = np.random.uniform(-8, +8, world.dim_p)
                    if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) < 4:
                        can_place = False
                if can_place:
                    place = True


            # landmark.state.p_vel = np_random.uniform(-0.5, 0.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        print("World reset:")
    # def reward(self, agent, world):
    #     reward = 0
    #     radius = 1.0
    #
    #     for landmark in world.landmarks:
    #
    #         nearby_agents = [a for a in world.agents if np.linalg.norm(a.state.p_pos - landmark.state.p_pos) < radius]
    #
    #         total_weight = sum([a.weight for a in nearby_agents])
    #
    #         current_distance = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
    #
    #         if hasattr(agent, "prev_distance"):
    #             reward += agent.prev_distance - current_distance  # positive if getting closer
    #
    #
    #         agent.prev_distance = current_distance
    #         if current_distance > 10:
    #             reward -= 1
    #
    #
    #         if total_weight >= landmark.weight and current_distance < radius:
    #             reward += 10  # Reward for eating the landmark
    #             world.landmarks.remove(landmark)  # Remove the landmark
    #
    #             new_lm = Landmark()
    #             new_lm.name = f"landmark {len(world.landmarks)}"
    #             new_lm.collide = False
    #             new_lm.movable = False
    #             new_lm.weight = round(np.random.uniform(1, 1))
    #             new_lm.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
    #             new_lm.state.p_vel = np.zeros(world.dim_p)
    #             new_lm.state.c = np.zeros(world.dim_c)
    #             new_lm.color = np.array([0.75, 0.75, 0.75])
    #
    #             world.landmarks.append(new_lm) # add a new landmark since the other one was removed
    #
    #             break
    #
    #     return reward

    def reward(self, agent, world):
        reward = 0
        radius = 1.0
        punishment_distance = 10.0
        for landmark in world.landmarks:
            current_distance = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)

            if agent.prev_distance is not None:
                speed = agent.prev_distance - current_distance
                agent.entity_speed = speed
                # shaping = speed*10
                # reward += speed*10
                # shaping = np.clip(shaping, -1, 1)
                # reward += 10/(current_distance-radius)
            agent.prev_distance = current_distance

            # if np.linalg.norm(agent.state.p_pos) > punishment_distance:
            #     reward = -200


            nearby_agents = [a for a in world.agents if np.linalg.norm(a.state.p_pos - landmark.state.p_pos) < radius]
            total_weight = sum([a.weight for a in nearby_agents])


            # if total_weight >= landmark.weight and current_distance < radius:
            if current_distance <= radius:
                reward += 10
                agent.prev_distance = None  # reset the previous distance
                world.landmarks.remove(landmark)
                new_lm = Landmark()
                new_lm.name = f"landmark {len(world.landmarks)}"
                new_lm.collide = False
                new_lm.movable = False
                new_lm.weight = round(np.random.uniform(1, 1))
                while True:
                    new_lm.state.p_pos = np.random.uniform(-5, +5, world.dim_p)
                    if np.linalg.norm(agent.state.p_pos - new_lm.state.p_pos) > 4:
                        break
                new_lm.state.p_vel = np.zeros(world.dim_p)
                new_lm.state.c = np.zeros(world.dim_c)
                new_lm.color = np.array([0.75, 0.75, 0.75])
                world.landmarks.append(new_lm)
                break

        return reward


    def observation(self, agent, world):
        entity_pos = []
        entity_weights = []

        for entity in world.landmarks:
            pos = entity.state.p_pos - agent.state.p_pos
            R = np.linalg.norm(agent.state.p_pos - entity.state.p_pos)/10
            R = np.clip(R, 0, 1)  # clip distance to avoid large values
            s = agent.entity_speed*5
            s = np.clip(s, -1, 1)/2 + 0.5  # clip speed to avoid large values
            # s= np.clip(s, 0, 1)

            theta_2 = (np.arctan2(entity.state.p_pos[1] - agent.state.p_pos[1], entity.state.p_pos[0] - agent.state.p_pos[0])+np.pi)/(2*np.pi)
            theta = np.arctan2(entity.state.p_pos[1] - agent.state.p_pos[1], entity.state.p_pos[0] - agent.state.p_pos[0])
            if agent.previous_theta is not None:
                theta_speed = (theta - agent.previous_theta)*40
                theta_speed = np.clip(theta_speed, -1, 1)/2 + 0.5
            else:
                theta_speed = 0
            agent.previous_theta = theta

            # current_pos = entity.state.p_pos - agent.state.p_pos
            # if agent.previous_food_pos is not None:
            #     entity_velocity = agent.previous_food_pos - current_pos
            # else:
            #     entity_velocity = np.zeros_like(current_pos)
            # agent.previous_food_pos = current_pos
            # entity_pos.append(np.array([s, theta_speed ]))
            # entity_pos.append(np.array([R, s, theta_2]))
            entity_pos.append(pos)
            # entity_weights.append(entity.weight)
        # print(agent.state.c)
        return np.concatenate(
            [
                # agent.state.p_pos,
                # agent.state.p_vel,
                np.array(entity_pos).flatten(),
                # np.array(entity_weights).flatten(),
                # agent.state.c
            ]
        )