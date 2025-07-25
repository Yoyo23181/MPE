from agent_for_training import *
from tqdm import tqdm
from mpe2 import simple_v3
from matplotlib import pyplot as plt
import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn
from weight import *
from agent import *

# hyperparameters
learning_rate = 0
n_episodes = 0
start_epsilon = 0
epsilon_decay = 0  # reduce the exploration over time
final_epsilon = 0


max_steps_per_episode = 5000

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
    training=False
)
agent.load_state_dict(torch.load("trained_agent_test_1.pth"))
agent.eval()
print("Loaded trained model.")
test_episodes = 5  # Adjust this number as needed
reward_during_time = []

for episode in range(test_episodes):
    obs, info = env.reset(seed=np.random.randint(1_000_000))
    done = False
    total_reward = 0
    step_count = 0

    print(f"\n--- Episode {episode + 1} ---")

    while not done and step_count < max_steps_per_episode:
        agent_id = env.agents[0]  # Assuming single agent
        state = torch.FloatTensor(obs[agent_id])
        action_env, action_nn = agent.get_action(state)

        actions = {agent_id: action_env}
        next_obs, reward, terminated, truncated, info = env.step(actions)

        reward_during_time.append(reward)
        print("Reward:", reward)
        # Check done status
        done = any(terminated.values()) or any(truncated.values())
        obs = next_obs
        total_reward += reward[agent_id]
        step_count += 1
        done = False
        for agent_id_ter in terminated:
            done = done or terminated[agent_id_ter]
            done = done or truncated[agent_id_ter]

        if total_reward < -10000:
            done = True

    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps taken: {step_count}")


fig, axs = plt.subplots(ncols=1, figsize=(12, 5))

axs.set_title("Reward during time")
axs.plot([reward['agent_0'] for reward in reward_during_time])
plt.tight_layout()
plt.show()