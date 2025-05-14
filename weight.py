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

# from agilerl.algorithms.matd3 import MATD3


from pettingzoo.utils.conversions import parallel_wrapper_fn
from agent import IA2CAgent
# from agents.mimic_agent import Agent

class raw_env(SimpleEnv, EzPickle):
    def __init__(self, max_cycles=2500, continuous_actions=False, render_mode="human"):

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

        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_v3"

        # self._weight_labels = []

        # super().render()

    # class raw_env(SimpleEnv, EzPickle):
        # … your __init__ stays the same …

    #     class raw_env(SimpleEnv, EzPickle):
    #         # … your __init__ stays the same …
    #
    #         def render(self):
    #             # 1) draw the world and create self.viewer if needed
    #             super().render()
    #
    #             # 2) if there’s no viewer yet, bail
    #             if self.viewer is None:
    #                 return
    #
    #             # 3) clear old labels
    #             for lbl in getattr(self, "_weight_labels", []):
    #                 lbl.delete()
    #             self._weight_labels = []
    #
    #             # 4) draw new ones in BLACK
    #             w, h = self.viewer.width, self.viewer.height
    #             lim = getattr(self.world, "boundary", 1.0)
    #
    #             for entity in list(self.world.agents) + list(self.world.landmarks):
    #                 wx, wy = entity.state.p_pos
    #                 sx = (wx + lim) / (2 * lim) * w
    #                 sy = (wy + lim) / (2 * lim) * h
    #                 lbl = Label(
    #                     text=str(entity.weight),
    #                     x=sx, y=sy,
    #                     anchor_x="center", anchor_y="center",
    #                     font_size=14,
    #                     color=(0, 0, 0, 255),  # <— draw in black now
    #                     batch=self.viewer.batch,
    #                 )
    #                 self._weight_labels.append(lbl)
    #
    # def _entity_to_screen(self, pos):
    #     """
    #     Helper to map world coords [-1,1] to screen pixels.
    #     SimpleEnv stores viewer.width/height in self.viewer
    #     and sets self.scale internally.
    #     """
    #     # these attributes exist on the built-in viewer
    #     w, h = self.viewer.width, self.viewer.height
    #     # world limits are [-lim,lim] where lim=self.world.boundary or 1.0 by default
    #     lim = self.world.boundary if hasattr(self.world, "boundary") else 1.0
    #
    #     # map x from [-lim,+lim] → [0,w]; same for y but invert Y if needed
    #     screen_x = (pos[0] + lim) / (2 * lim) * w
    #     screen_y = (pos[1] + lim) / (2 * lim) * h
    #     return screen_x, screen_y



env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

number_agent = 1
number_landmark = 1


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 3 # make communication trought c possible?
        world.collaborative = True
        # add agents
        # world.agents = [Agent() for i in range(number_agent)]

        num_lm = number_landmark
        dim_p = world.dim_p
        dim_c = world.dim_c

        state_dim = 2 + 2 + num_lm*dim_p + num_lm + dim_c
        action_dim = dim_p + dim_c

        world.agents = [IA2CAgent(state_dim,action_dim) for i in range(number_agent)]

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
            landmark.weight = round(np.random.uniform(1, 3))
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
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np_random.uniform(-0.5, 0.5, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.75, 0.75, 0.75])
            landmark.color = weight_to_color.get(landmark.weight, default_color)
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            # landmark.state.p_vel = np_random.uniform(-0.5, 0.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        reward = 0
        # pos = agent.state.p_pos
        # # if x or y outside the box
        # if (pos[0] < -10.0) or (pos[0] > 10.0) or (pos[1] < -10.0) or (pos[1] > 10.0):
        #     reward -= 30.0
        #     # optional: clamp position back into the box so the agent can’t wander off
        #     agent.state.p_pos = np.clip(pos, -10.0, 10.0)
        for landmark in world.landmarks:
            # Find nearby agents around each landmark (within a distance of 0.2)
            nearby_agents = [a for a in world.agents if np.linalg.norm(a.state.p_pos - landmark.state.p_pos) < 0.4]
            # for i, lm in enumerate(world.agents):
            #     print(f"[Agent {i}] pos={lm.state.p_pos}, vel={lm.state.p_vel}")
            # Calculate total weight of nearby agents
            total_weight = sum([a.weight for a in nearby_agents])

            # Reward for moving toward the landmark (based on proximity)
            distance_to_landmark = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            reward += 1 / (1 + distance_to_landmark)  # Higher reward for being closer to the landmark

            # If agents' combined weight is enough to "eat" the landmark
            if total_weight >= landmark.weight:
                reward += 10  # Reward for eating the landmark
                world.landmarks.remove(landmark)  # Remove the landmark

                new_lm = Landmark()
                new_lm.name = f"landmark {len(world.landmarks)}"
                new_lm.collide = False
                new_lm.movable = False
                new_lm.weight = round(np.random.uniform(2, 4))
                new_lm.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                new_lm.state.p_vel = np.zeros(world.dim_p)
                new_lm.state.c = np.zeros(world.dim_c)
                new_lm.color = np.array([0.75, 0.75, 0.75])

                world.landmarks.append(new_lm) # add a new landmark since the other one was removed

                break  # Only remove one landmark at a time, break after handling it

        return reward

    def observation(self, agent, world):
        entity_pos = []
        entity_weights = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_weights.append(entity.weight)
        # print(agent.state.c)
        return np.concatenate(
            [
                agent.state.p_pos,
                agent.state.p_vel,
                np.array(entity_pos).flatten(),
                np.array(entity_weights).flatten(),
                agent.state.c
            ]
        )

    # if __name__ == "__main__":
    #     # 1. Create the environment
    #     env = parallel_env()
    #     observations = env.reset()
    #
    #     # 2. Use one observation to determine input/output sizes
    #     example_obs = observations[env.agents[0]]
    #     state_dim = example_obs.shape[0]
    #     action_dim = env.action_space(env.agents[0]).n
    #
    #     # 3. Create one IA2CAgent per agent in the environment
    #     agents = {
    #         agent_id: IA2CAgent(state_dim, action_dim)
    #         for agent_id in env.agents
    #     }
    #
    #     # 4. Run loop to let IA2C agents interact with environment
    #     for step in range(100):  # Or max_cycles
    #         actions = {}
    #
    #         # Each IA2C agent decides its own action
    #         for agent_id in env.agents:
    #             obs = torch.FloatTensor(observations[agent_id])
    #             action = agents[agent_id].get_action(obs)
    #             actions[agent_id] = action
    #
    #         # 5. Step environment with selected actions
    #         observations, rewards, terminations, truncations, infos = env.step(actions)
    #
    #         # Optional: print debug info
    #         print(f"Step {step}: Actions: {actions}, Rewards: {rewards}")
    #
    #         # 6. Stop early if all agents are done
    #         if all(terminations.values()) or all(truncations.values()):
    #             break
