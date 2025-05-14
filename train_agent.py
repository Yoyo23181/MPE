from agent_for_training import *
from tqdm import tqdm
from mpe2 import simple_v3
from matplotlib import pyplot as plt
import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn
from weight import *

# hyperparameters
learning_rate = 0.001
n_episodes = 1_000_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

# env = gym.make("Blackjack-v1", sab=False)
# env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
# env = simple_v3.parallel_env(
#     max_cycles=25,
#     continuous_actions=True,
#     dynamic_rescaling=True
# )
# obs, info = env.reset(seed=42)

# env = raw_env(render_mode="human", continuous_actions=True)
env = parallel_wrapper_fn(raw_env)(render_mode="human", continuous_actions=True)

obs, info = env.reset(seed=42)

first_obs = obs[env.agents[0]]
state_dim = first_obs.shape[0]
action_dim = env.action_space(env.agents[0]).shape[0]

agent = Agent_train(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    state_dim=state_dim,
    action_dim=action_dim,
)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset(seed=42)
    done = False
    # play one episode
    while not done:
        # action = agent.get_action(obs)

        agent_id = env.agents[0]  # assuming single agent
        state = torch.FloatTensor(obs[agent_id])
        action = agent.get_action(state)

        actions = {agent_id: action}

        next_obs, reward, terminated, truncated, info = env.step(actions)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()



def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# # Smooth over a 500 episode window
# rolling_length = 500
# fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
#
# axs[0].set_title("Episode rewards")
# reward_moving_average = get_moving_avgs(
#     env.return_queue,
#     rolling_length,
#     "valid"
# )
# axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
#
# axs[1].set_title("Episode lengths")
# length_moving_average = get_moving_avgs(
#     env.length_queue,
#     rolling_length,
#     "valid"
# )
# axs[1].plot(range(len(length_moving_average)), length_moving_average)
#
# axs[2].set_title("Training Error")
# training_error_moving_average = get_moving_avgs(
#     agent.training_error,
#     rolling_length,
#     "same"
# )
# axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
# plt.tight_layout()
# plt.show()

# from torch.cuda import memory

# import simple_v3
# from weight import raw_env
# from idp_agent import IA2CAgent
# import torch
#
# def run():
#     env = raw_env(render_mode="human", continuous_actions=True)
#     env.reset(seed=42)
#
#     first_obs = env.observe(env.agents[0])
#     state_dim = first_obs.shape[0]
#     action_dim = env.action_space(env.agents[0]).shape[0]
#
#     my_agents = {
#         agent_id: IA2CAgent(state_dim, action_dim)
#         for agent_id in env.agents
#     }
#
#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last() # return the result of that agent's previous env.step(action).
#                                                                         # Observation is a numpy array what the current agent see.
#                                                                             # Return atm
#                                                                                 # - coordinates (x,y) of the agent
#                                                                                 # - velocity of this agent (x,y)
#                                                                                 # - distance in x and y from landmarks
#                                                                                 # - weight of each landmarks
#                                                                         # Reward is a scalar giving the immediate reward that the agent received as result of the previous action
#                                                                         # termination if true, agent has reached a terminal state
#                                                                         # truncation if true, agent hit a time limit
#
#         if termination or truncation:
#             print(f"{agent} is done. No more actions.")
#             action = None
#         else:
#             # this is where you would insert your policy
#             # action = env.action_space(agent).sample() # this is a random action
#             obs = torch.FloatTensor(observation)
#             # print(observation)
#             # print(type(obs))  # <class 'torch.Tensor'>
#             # print(obs.dtype)  # torch.float32
#             # print(obs.shape)  # torch.Size([state_dim])
#             # print(obs)        # prints the tensor values
#             action = my_agents[agent].get_action(obs) # action is a 5 dimension vector it gives
#                                                         # - 2 first numbers are the physical forces u_x and u_y what we use to move
#                                                         # - others 3 are communication channels that can be used to communicate between agents
#             # print(f"{agent} takes action: {action}")
#         env.step(action)
#         # env.render()
#     env.close()


