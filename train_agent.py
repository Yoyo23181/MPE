from agent_for_training import *
from tqdm import tqdm
from mpe2 import simple_v3
from matplotlib import pyplot as plt
import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn
from weight import *
from agent import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QT5Agg")  # Use a non-interactive backend for matplotlib
import time

plt.ion()  # Turn on interactive mode


import torch


class train_agent:
    def __init__(self,  n_episodes=1000, episode_steps = 1000, warmup_eps = 0, networkpath=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.networkpath = networkpath
        self.n_episodes = n_episodes
        self.episode_steps = episode_steps
        self.warmup_eps = warmup_eps
        self.action_dim = 1
        self.episode = 0

        self.init_agent()
        self.init_env()


        self.actor_loss_list = []
        self.critic_loss_list = []
        self.advantage_list = []
        self.total_rewards_list = []


    def init_agent(self):

        self.nn_agent = IA2CAgent(
            learning_rate_actor=1e-5,
            learning_rate_critic=1e-5,
            state_dim=2,
            action_dim=self.action_dim,
            training=True
            ).to(self.device)

        if self.networkpath is not None and os.path.exists(self.networkpath):
            self.nn_agent.load_state_dict(torch.load(self.networkpath, map_location=self.device))
            print(f"Loaded network from {self.networkpath}")
        else:
            print(f"Network path {self.networkpath} does not exist. Starting with a new network.")

    def init_env(self):
        self.env = parallel_wrapper_fn(my_raw_env)(render_mode="rgb_array", continuous_actions=True)

    def reset_lists(self):
        self.agent_posx_list  = []
        self.agent_posy_list = []
        self.food_pos_x_list = []
        self.food_pos_y_list = []
        self.local_total_rewards_list = []
        self.local_rewards = []
        self.action_list = []
        self.obs_list = []
        self.prob_list = []
        self.mu_list = []
        self.entropy_list = []

    def init_trianing_lists(self):
        self.episode_actions =[]
        self.episode_obs = []
        self.episode_rewards = []

    def reset_env(self):
        self.init_trianing_lists()
        obs, _ = self.env.reset(seed=42)
        self.episode_obs.append(obs["agent_0"])  # assuming a single agent for simplicity

    def log_step(self,obs,  reward , action):
        for i, agent in enumerate(self.env.aec_env.world.agents):
            self.agent_posx_list.append(agent.state.p_pos[0])
            self.agent_posy_list.append(agent.state.p_pos[1])
            self.food_pos_x_list.append(self.env.aec_env.world.landmarks[0].state.p_pos[0])
            self.food_pos_y_list.append(self.env.aec_env.world.landmarks[0].state.p_pos[1])
            self.local_rewards.append(reward)
            self.local_total_rewards_list.append(np.sum(np.array(self.local_rewards)))
            self.obs_list.append(obs)  # assuming a single agent for simplicity
            self.action_list.append(action)
            # self.prob_list.append(log_prob)
            # self.mu_list.append(mu)
            # self.entropy_list.append(entropy)

    def log_training_batch(self, obs , reward,action):
        self.episode_actions.append(action)  # convert to numpy array and append to the list
        self.episode_obs.append(obs)
        self.episode_rewards.append(reward)

    def run_episode(self):
        self.reset_env()
        self.nn_agent.init_hidden()
        self.nn_agent.reset_episode_tensors()
        for step in range(self.episode_steps):
            obs, reward, action= self.step()

            self.log_step(obs, reward, action)
        self.get_internal_state()

    def get_internal_state(self):
        self.mu_list = self.nn_agent.mu_list.detach().cpu().numpy()
        self.prob_list = self.nn_agent.log_prob_list.detach().cpu().numpy()
        self.entropy_list = self.nn_agent.entropy_list.detach().cpu().numpy()


    def update_agent(self):
        actor_loss, critic_loss, avantage = self.nn_agent.batch_update_new()
        self.actor_loss_list.append(actor_loss)
        self.critic_loss_list.append(critic_loss)
        self.advantage_list.append(avantage)
        self.total_rewards_list.append(self.local_total_rewards_list[-1])

    def step(self):
        actions = {}
        agent_id = f"agent_0"  # assuming a single agent for simplicity

        # for i, agent in enumerate(self.env.aec_env.world.agents):
        if self.episode < self.warmup_eps:
            # action_nn = self.env.action_space(agent_id).sample()
            # action= [(action_nn[2] - action_nn[1])/2 +0.5 , (action_nn[4] - action_nn[3])/2 +0.5]  # convert to [0, 1] range
            action = np.random.uniform(0, 1, size=(self.action_dim)) # random action in [0, 1] range
            # action_dir =
            action_nn = np.array(action)
            mu = np.zeros_like(action_nn)
            log_prob = np.zeros_like(action_nn)
            entropy = np.zeros_like(action_nn)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(self.episode_obs[-1]).to(self.device)  # convert to tensor and move to device
            action_nn = self.nn_agent.get_train_action_new(state)
            action_nn = action_nn.cpu().numpy()

        actions[agent_id] = action_nn

        next_obs, reward, terminated, truncated, info = self.env.step(actions)
        self.nn_agent.get_reward(reward[agent_id], next_obs[agent_id])  # store the reward and next observation for training
        return next_obs[agent_id], reward[agent_id], action_nn  # return the observation, reward and action for the agent
        # self.obs = next_obs  # update the observation for the next step
        #
        #
        #
        # # print("next_obs[agent_id] shape:", next_obs[agent_id].shape)
        #
        # reward_during_time.append(reward)
        # local_rewards.append(reward[agent_id])

    def train(self):
        self.init_plots()
        for self.episode in tqdm(range(self.n_episodes)):
            self.reset_lists()
            self.run_episode()
            self.update_agent()
            self.plot_update()

        self.save_network("nn_agent.pth")
        self.final_plot()

    def save_network(self, filename):
        if os.path.exists(filename):
            filename = filename.split(".pth")[0] + "1.pth"
        torch.save(self.nn_agent.state_dict(), filename)

    def init_plots(self):
        self.fig = plt.figure(figsize=(12, 10))

        self.ax_total_reward = self.fig.add_subplot(321)

        self.ax_reward_local = self.fig.add_subplot(322)
        self.ax_obs = self.ax_reward_local.twinx()

        self.ax_critic = self.fig.add_subplot(323)
        self.ax_adv = self.ax_critic.twinx()

        self.ax_action = self.fig.add_subplot(324)
        self.ax_h = self.ax_action.twinx()
        self.ax_plot = self.fig.add_subplot(325)
        self.ax_reward = self.fig.add_subplot(326)

        plt.show()

    def plot_update(self):
        self.fig.suptitle(f"{self.episode}/{n_episodes}")
        self.ax_reward_local.clear()
        self.ax_total_reward.clear()
        self.ax_action.clear()
        self.ax_h.clear()
        self.ax_critic.clear()
        self.ax_plot.clear()
        self.ax_reward.clear()
        self.ax_obs.clear()
        self.ax_adv.clear()
        self.ax_reward_local.set_ylabel("Reward")

        self.ax_total_reward.set_ylabel("Total Reward")
        self.ax_action.set_ylabel("Actor loss")
        self.ax_reward_local.set_ylim(0,1)
        # self.ax_reward_local.plot(self.action_list, label="action")
        # ax_h.set_ylabel("Action NN")

        self.ax_critic.set_ylabel("Critic loss")
        # self.ax_reward_local.plot(self.local_rewards, c="tab:purple")
        self.ax_reward_local.plot(self.obs_list, label="obs")
        self.ax_reward_local.legend(loc="upper left")
        self.ax_total_reward.plot(self.local_total_rewards_list)
        self.ax_total_reward.plot(self.local_rewards)
        self.ax_critic.plot(self.actor_loss_list)
        self.ax_critic.plot(self.critic_loss_list)
        self.ax_adv.plot(self.advantage_list, c="tab:orange", label="advantage")
        self.ax_plot.plot(self.agent_posx_list, self.agent_posy_list, c="tab:blue", label="agent")
        self.ax_plot.plot(self.food_pos_x_list, self.food_pos_y_list, c="tab:orange", marker="x", label="food")
        self.ax_plot.plot([self.agent_posx_list[-1]], [self.agent_posy_list[-1]], c="tab:blue", marker="o", label="current position")

        self.ax_reward.plot(self.total_rewards_list)

        self.ax_action.plot(self.action_list)
        self.ax_action.plot(self.mu_list, c="tab:orange", label="mu")
        self.ax_action.plot(self.prob_list, c="tab:green", label="log_prob")
        self.ax_action.plot(self.entropy_list, c="tab:red", label="entropy")
        self.ax_action.set_xlabel("step")
        # self.ax_action.set_ylabel("Actor loss")
        self.ax_plot.set_xlabel("plot_x")
        self.ax_plot.set_ylabel("plot_y")
        self.ax_reward_local.set_xlabel("step")
        self.ax_reward_local.set_ylabel("Reward")
        self.ax_critic.set_xlabel("step")
        self.ax_critic.set_ylabel("critic loss")
        self.ax_total_reward.set_xlabel("step")
        self.ax_total_reward.set_ylabel("Total Reward")
        self.ax_total_reward.set_xlabel("step")
        self.ax_total_reward.set_ylabel("Total Reward")

        plt.pause(0.001)

    def final_plot(self):
        plt.close(self.fig)
        plt.ioff()
        fig = plt.figure(figsize=(12, 5))
        titles = {"Actor Loss":self.actor_loss_list,
                  "Critic Loss":self.critic_loss_list,
                  "Total Rewards": self.total_rewards_list,
                  "Advantage":self.advantage_list}
        for i, title in enumerate(titles):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel(title)
            ax.plot(titles[title], label=title)

        plt.show()




if __name__=="__main__":
    n_episodes = 1000
    episode_steps = 200
    warmup_eps = 0  # number of episodes to explore randomly before training
    network_path = "nn_agent_dir_good.pth"
    network_path = None
    trainer = train_agent(n_episodes=n_episodes, episode_steps=episode_steps, warmup_eps=warmup_eps, networkpath=network_path)
    trainer.train()

