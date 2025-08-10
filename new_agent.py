import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from mpe2._mpe_utils.core import Agent, Entity, AgentState, Action
import torch.nn.functional as F

class IA2CAgent(nn.Module):
    def __init__( self, learning_rate_actor=1e-4, learning_rate_critic=1e-3,
                  # initial_epsilon=1.0, epsilon_decay=0.0, final_epsilon=0.1, discount_factor=0.99,
                  state_dim=0, action_dim=0,  training=True, ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Agent.__init__(self)
        super().__init__()
        self.previous_distance = None
        self.previous_food_pos = None
        self.entity_speed = 0
        # self.epsilon = initial_epsilon
        # self.epsilon_decay = epsilon_decay
        # self.final_epsilon = final_epsilon
        # self.discount_factor = discount_factor
        self.training_error = []
        self.state_dim = state_dim
        self.action_dim = action_dim

        # self.action_callback = None
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(self.state_dim, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 128),
        #     nn.Tanh()
        # )
        #
        # self.fc = nn.Sequential(
        #     self._layer_init(nn.Linear(self.state_dim, 256)),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     self._layer_init(nn.Linear(256, 128)),
        #     nn.ReLU()
        # )

        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # self.fc = nn.LSTM(input_size=state_dim, hidden_size=128)
        # self.init_hidden(hidden_size=128, batch_size=1)
        self.actor = nn.Sequential(nn.Linear(128, self.action_dim), nn.Sigmoid())
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32, device=self.device))
        self.critic = nn.Sequential(nn.Linear(128, 1))

        # self.log_std= torch.tensor([0.1], dtype=torch.float32).to(self.device)

        if training:
            self.optimizer = torch.optim.Adam(
                list(self.fc.parameters()) +
                list(self.actor.parameters()) +
                [self.log_std] +
                list(self.critic.parameters()),
                lr=learning_rate_actor,
                eps=1e-5
            )

        else:
            self.optimizer = None  # avoid empty parameter list error

        self.test_layers()
        self.reset_episode_tensors()

    def reset_episode_tensors(self):
        self.log_prob_list = torch.empty((0, self.action_dim)).to(self.device)
        self.mu_list = torch.empty((0,self.action_dim)).to(self.device)
        self.entropy_list = torch.empty((0,self.action_dim)).to(self.device)
        self.state_list = torch.empty((0,self.state_dim)).to(self.device)
        self.reward_list = torch.empty((0, 1)).to(self.device)
        self.value_list = torch.empty((0, 1)).to(self.device)


    def test_layers(self):
        self.fc.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        obs = torch.rand(1, self.state_dim).to(self.device)  # Random observation for testing
        out = self.fc(obs)
        action = self.actor(out)
        value = self.critic(out)
        print(f"Test: Actor output: {action.detach()}, Value output: {value.detach()}")

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.1):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


    # def forward(self, state):
    #     x = self.fc(state)
    #     # action_probs = torch.softmax(self.actor(x), dim=-1)
    #     val_critic = self.critic(x)
    #     val_actor = self.actor(x)
    #     return val_actor, val_critic

    def init_hidden(self,hidden_size=128, batch_size=1, num_layers=1):
        h_0 = torch.zeros(num_layers, batch_size, hidden_size).to(self.device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size).to(self.device)
        self.hidden =  (h_0, c_0)

    def detach_hidden(self):
        if self.hidden is not None:
            h_0, c_0 = self.hidden
            self.hidden = (h_0.detach(), c_0.detach())


    def forward(self, state):
        out = self.fc(state)
        action = self.actor(out)
        value = self.critic(out)
        return action, value

    def get_action(self, state):
        a, v, h = self.forward(state)
        return a.detach()


    def get_train_action(self, state, sigma=0.2):

        out = self.fc(state)
        mu = self.actor(out)
        # e.g., sigmoid output in [0,1] or tanh
        # std = torch.exp(self.log_std)  # assuming you added this
        # dist = torch.distributions.Normal(mu, std)
        # action = dist.sample()  # ← critical!
        # action = action *0.5 + 0.5

        dist = torch.distributions.Normal(mu, torch.ones_like(mu) * sigma)
        a = dist.sample()  # <-- sample, not mu
        return a.clamp(0.0, 1.0).detach()
        # return mu.detach()

    def get_reward(self, reward, next_state):
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)  # ensure reward is a tensor
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.reward_list = torch.cat((self.reward_list, reward.unsqueeze(0)), dim=0)
        self.state_list = torch.cat((self.state_list, next_state), dim=0)


    def get_train_action_new(self, state):
        out = self.fc(state)
        value = self.critic(out)
        mu = self.actor(out)  # ensure in [-1, 1]
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()  # <--- reparameterized sample

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        scaled_action = (action * 0.5 + 0.5).clamp(0.0, 1.0)

        self.mu_list =  torch.cat((self.mu_list, mu.unsqueeze(0).to(self.device)), dim=0)
        self.log_prob_list = torch.cat((self.log_prob_list, log_prob.unsqueeze(0).to(self.device)), dim=0)
        self.entropy_list = torch.cat((self.entropy_list, entropy.unsqueeze(0).to(self.device)), dim=0)
        self.value_list = torch.cat((self.value_list, value.unsqueeze(0).to(self.device)), dim=0)

        return mu.clone().detach()

    def get_value(self, state):
        _, value = self(state)
        return value

    def batch_update(self, episode_obs, episode_actions, episode_rewards,
                     gamma=0.99, entropy_coef=0.01, critic_coef=0.5, sigma=0.2):

        device = self.device

        # T x obs_dim / act_dim / 1
        states = episode_obs.to(device)  # [T, S]
        actions = episode_actions.to(device)  # [T, A]
        rewards = episode_rewards.to(device)  # [T]

        # 1) Compute discounted returns
        with torch.no_grad():
            G = 0.0
            rets = []
            for r in reversed(rewards.tolist()):
                G = r + gamma * G
                rets.append(G)
            returns = torch.tensor(list(reversed(rets)), dtype=torch.float32, device=device).unsqueeze(1)  # [T,1]

        # 2) Critic values & advantages
        x = self.fc(states)  # [T,128]
        values = self.critic(x)  # [T,1]
        advantages = returns - values.detach()  # [T,1]

        # --- normalize advantages (critical for stability) ---
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std

        # 3) Actor loss (Gaussian policy over actions in [0,1])
        mu = self.actor(x)  # [T,A]
        dist = torch.distributions.Normal(mu, torch.ones_like(mu) * sigma)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)  # [T,1]
        entropy = dist.entropy().sum(dim=-1, keepdim=True)  # [T,1]

        actor_loss = -(log_probs * advantages).mean() - entropy_coef * entropy.mean()
        critic_loss = critic_coef * torch.nn.functional.smooth_l1_loss(values, returns)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        # Logging so you can see if the signal is alive
        with torch.no_grad():
            print(
                f"Batch update - Actor: {actor_loss.item():.4f}, "
                f"Critic: {critic_loss.item():.4f}, "
                f"Adv μ: {adv_mean.item():.4e}, σ: {adv_std.item():.4e}, "
                f"logπ μ: {log_probs.mean().item():.4f}, H: {entropy.mean().item():.4f}"
            )

        return actor_loss.item(), critic_loss.item(), advantages.mean().item()

    def batch_update_new(self,  gamma=0.9):


        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.reward_list):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device)

        advantages = returns - self.value_list.clone().detach()
        actor_loss = (-self.log_prob_list * advantages - 0.01 * self.entropy_list).mean()

        # Critic loss
        critic_loss = F.smooth_l1_loss(self.value_list, returns)
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optimizer.step()

        print(f"Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
        return actor_loss.item(), critic_loss.item(), advantages.mean().item()

