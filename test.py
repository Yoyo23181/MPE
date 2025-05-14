from mpe2 import simple_v3
from pettingzoo.utils.conversions import parallel_wrapper_fn
from stable_baselines3 import PPO
import gymnasium as gym

# Convert PettingZoo env to Gymnasium-compatible single-agent env
env_fn = parallel_wrapper_fn(lambda: simple_v3.parallel_env())
env = env_fn()

# Train using Stable-Baselines3 PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_mpe2_simple")



# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#
#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()
#
#     env.step(action)
# env.close()