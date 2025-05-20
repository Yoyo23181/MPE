from agent_for_training import *
from tqdm import tqdm
from mpe2 import simple_v3
from matplotlib import pyplot as plt
import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn
from weight import *
from agent import *

# hyperparameters
learning_rate = 0.0001
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1


max_steps_per_episode = 1000

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
# print("First obs shape:", obs[env.agents[0]].shape)

first_obs = obs[env.agents[0]]
state_dim = first_obs.shape[0]
# action_dim = env.action_space(env.agents[0]).shape[0]
action_dim = 2
# print("train_agent", action_dim)

agent = IA2CAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    state_dim=state_dim,
    action_dim=action_dim,
    training=True
)

episode_rewards = []
episode_lengths = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset(seed=42)
    # obs, info = env.reset(seed=np.random.randint(1_000_000))
    done = False
    total_reward = 0
    step_count = 0


    # play one episode
    while not done and step_count < max_steps_per_episode:
        # action = agent.get_action(obs)

        agent_id = env.agents[0]  # assuming single agent
        state = torch.FloatTensor(obs[agent_id])
        action_env, action_nn = agent.get_action(state)
        # print("Action:", action)

        actions = {agent_id: action_env}

        next_obs, reward, terminated, truncated, info = env.step(actions)
        # print("next_obs[agent_id] shape:", next_obs[agent_id].shape)



        agent.update(
            obs[agent_id],  # must be shape (10,)
            action_nn,  # shape (5,) after fix
            reward[agent_id],
            terminated[agent_id],
            next_obs[agent_id]  # must be shape (10,)
        )
        agent_pos = env.aec_env.world.agents[0].state.p_pos
        print("Agent position:", agent_pos)
        for i, landmark in enumerate(env.aec_env.world.landmarks):
            print(f"Landmark {i} position:", landmark.state.p_pos)

        print(total_reward)
        print("Physical action (u):", action_nn)
        # update if the environment is done and the current obs
        done = False
        for agent_id_ter in terminated:
            done = done or terminated[agent_id_ter]
            done = done or truncated[agent_id_ter]
        obs = next_obs

        total_reward += reward[agent_id]
        step_count += 1
        if total_reward < -25000:
            done = True

        if episode % 100 == 0 and step_count == 0:
            print("action_nn:", action_nn)
    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    episode_lengths.append(step_count)


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

torch.save(agent.state_dict(), "trained_agent_test_1.pth")

# Smooth over a 500 episode window
# rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# axs[0].set_title("Episode rewards")
# reward_moving_average = get_moving_avgs(
#     episode_rewards,
#     rolling_length,
#     "valid"
# )
# axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
#
# axs[1].set_title("Episode lengths")
# length_moving_average = get_moving_avgs(
#     episode_lengths,
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

# Plot rewards per episode
axs[0].set_title("Episode rewards")
axs[0].plot(episode_rewards)

# Plot steps per episode
axs[1].set_title("Episode lengths")
axs[1].plot(episode_lengths)

# Plot training error per step
axs[2].set_title("Training Error")
axs[2].plot(agent.training_error)

plt.tight_layout()
plt.show()
