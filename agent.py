import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mpe2._mpe_utils.core import Agent, Entity, AgentState, Action


class IA2CAgent(Agent, nn.Module):
    def __init__(
        self,
        env=None,
        learning_rate=1e-3,
        initial_epsilon=1.0,
        epsilon_decay=0.0,
        final_epsilon=0.1,
        state_dim=0,
        action_dim=0,
        discount_factor=0.99,
        training=False,
    ):
        Agent.__init__(self)
        nn.Module.__init__(self)

        self.env = env
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.training_error = []

        # Physical agent properties (from Agent)
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.c_noise = None
        self.u_range = 1.0
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 128),
            nn.ReLU(inplace=False)
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)


        if training:
            self.actor_optimizer = torch.optim.Adam(
                list(self.fc.parameters()) + list(self.actor.parameters()),
                lr=learning_rate
            )
            self.critic_optimizer = torch.optim.Adam(
                list(self.fc.parameters()) + list(self.critic.parameters()),
                lr=learning_rate
            )
        else:
            self.optimizer = None  # avoid empty parameter list error


    def forward(self, state):
        x = self.fc(state)
        # action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return self.actor(x), value


    def get_action(self, state):
        # action_probs, _ = self(state)
        # action = torch.multinomial(action_probs, 1).item()
        x = self.fc(state)
        y= self.actor(x)
        # action = (torch.tanh(self.actor(x)) + 1) / 2  # scale to [0;1]
        y = torch.tanh(y)  # scale to [0;1]
        action = np.zeros(5)

        action[1] = torch.relu(-y[0]).item()
        action[2] = torch.relu(y[0]).item()
        action[3] = torch.relu(-y[1]).item()
        action[4] = torch.relu(y[1]).item()


        return action, y.detach().numpy()


    def get_value(self, state):
        _, value = self(state)
        return value


    def update(self, obs, action, reward, terminated, next_obs):
        state = torch.FloatTensor(obs)
        next_state = torch.FloatTensor(next_obs)

        value = self.critic(self.fc(state))
        next_value = self.critic(self.fc(next_state)).detach()

        target = reward + (0 if terminated else self.discount_factor * next_value)
        advantage = target - value

        mu = torch.tanh(self.actor(self.fc(state))).unsqueeze(-1)
        dist = torch.distributions.Normal(mu, torch.ones_like(mu) * 0.1)
        log_prob = dist.log_prob(torch.FloatTensor(action)).sum()

        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        if self.training:
            print("Updating agent...")
            print("Actor loss:", actor_loss.item())
            print("Critic loss:", critic_loss.item())

        before = self.actor.weight.clone()

        # Update actor
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward(retain_graph=True)
        # self.actor_optimizer.step()
        #
        # after = self.actor.weight
        # print("Actor weights changed:", not torch.equal(before, after))
        #
        #     # Update critic
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward(retain_graph=True)
        # self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.training_error.append(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
