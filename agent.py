import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class EntityState:  # Physical state (position, velocity)
    def __init__(self):
        self.p_pos = None
        self.p_vel = None


class AgentState(EntityState):  # Adds communication state
    def __init__(self):
        super().__init__()
        self.c = None


class Entity:
    def __init__(self):
        self.name = ""
        self.size = 0.050
        self.movable = False
        self.collide = True
        self.density = 25.0
        self.color = None
        self.max_speed = None
        self.accel = None
        self.state = EntityState()
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Action:
    def __init__(self):
        self.u = None  # physical action
        self.c = None  # communication action


class IA2CAgent(Entity, nn.Module):
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
        Entity.__init__(self)
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
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

        if training:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
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
        action = (torch.tanh(self.actor(x)) + 1)/2 # scale to [0;1]
        # action = torch.tanh(self.actor(x)) # scale to [-1;1]
        # print(f"[IA2C] Action output: {action.detach().numpy()}")
        return action.detach().numpy()

    def get_value(self, state):
        _, value = self(state)
        return value

    def update(self, obs, action, reward, terminated, next_obs):
        state = torch.FloatTensor(obs)
        next_state = torch.FloatTensor(next_obs)

        value = self.get_value(state)
        next_value = self.get_value(next_state)

        target = reward + (0 if terminated else self.discount_factor * next_value.item())
        loss = (target - value) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_error.append(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# class Agent(Entity):  # Simulation agent (non-learned)
#     def __init__(self):
#         super().__init__()
#         self.movable = True
#         self.silent = False
#         self.blind = False
#         self.u_noise = None
#         self.c_noise = None
#         self.u_range = 1.0
#         self.state = AgentState()
#         self.action = Action()
#         self.action_callback = None





