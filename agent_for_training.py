from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class Agent_train(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        state_dim,
        action_dim,
        discount_factor: float = 0.95,
    ):
        nn.Module.__init__(self)
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        self.fc = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    # def get_action(self, obs: tuple[int, int, bool]) -> int:
    #     """
    #     Returns the best action with probability (1 - epsilon)
    #     otherwise a random action with probability epsilon to ensure exploration.
    #     """
    #     # with probability epsilon return a random action to explore the environment
    #     if np.random.random() < self.epsilon:
    #         return self.env.action_space.sample()
    #     # with probability (1 - epsilon) act greedily (exploit)
    #     else:
    #         return int(np.argmax(self.q_values[obs]))

    def get_action(self, state):

    # action_probs, _ = self(state)
            # action = torch.multinomial(action_probs, 1).item()
            x = self.fc(state)
            action = (torch.tanh(self.actor(x)) + 1)/2 # scale to [0;1]
            # action = torch.tanh(self.actor(x)) # scale to [-1;1]
            # print(f"[IA2C] Action output: {action.detach().numpy()}")
            return action.detach().numpy()

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        # future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # temporal_difference = (
        #     reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        # )
        #
        # self.q_values[obs][action] = (
        #     self.q_values[obs][action] + self.lr * temporal_difference
        # )
        # self.training_error.append(temporal_difference)
        obs_key = tuple(obs)
        next_obs_key = tuple(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs_key][action]
        )

        self.q_values[obs_key][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
#
#
#
# class EntityState:  # Physical state (position, velocity)
#     def __init__(self):
#         self.p_pos = None
#         self.p_vel = None
#
#
# class AgentState(EntityState):  # Adds communication state
#     def __init__(self):
#         super().__init__()
#         self.c = None
#
#
# class Entity:
#     def __init__(self):
#         self.name = ""
#         self.size = 0.050
#         self.movable = False
#         self.collide = True
#         self.density = 25.0
#         self.color = None
#         self.max_speed = None
#         self.accel = None
#         self.state = EntityState()
#         self.initial_mass = 1.0
#
#     @property
#     def mass(self):
#         return self.initial_mass
#
#
# class Action:
#     def __init__(self):
#         self.u = None  # physical action
#         self.c = None  # communication action
#
#
# class IA2CAgent(Entity, nn.Module):
#     def __init__(self, state_dim, action_dim):
#         Entity.__init__(self)
#         nn.Module.__init__(self)
#
#         # Physical agent properties (from Agent)
#         self.movable = True
#         self.silent = False
#         self.blind = False
#         self.u_noise = None
#         self.c_noise = None
#         self.u_range = 1.0
#         self.state = AgentState()
#         self.action = Action()
#         self.action_callback = None
#
#         # Neural network
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#         self.actor = nn.Linear(128, action_dim)
#         self.critic = nn.Linear(128, 1)
#
#     def forward(self, state):
#         x = self.fc(state)
#         # action_probs = torch.softmax(self.actor(x), dim=-1)
#         value = self.critic(x)
#         return self.actor(x), value
#
#     def get_action(self, state):
#         # action_probs, _ = self(state)
#         # action = torch.multinomial(action_probs, 1).item()
#         x = self.fc(state)
#         action = (torch.tanh(self.actor(x)) + 1)/2 # scale to [0;1]
#         # action = torch.tanh(self.actor(x)) # scale to [-1;1]
#         # print(f"[IA2C] Action output: {action.detach().numpy()}")
#         return action.detach().numpy()
#
#     def get_value(self, state):
#         _, value = self(state)
#         return value

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




# class IA2CAgent(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(IA2CAgent, self).__init__()
#         # Simple fully connected network for the actor and critic
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#         self.actor = nn.Linear(128, action_dim)  # Output action logits
#         self.critic = nn.Linear(128, 1)  # Output state value
#
#     def forward(self, state):
#         x = self.fc(state)
#         # action_probs = torch.softmax(self.actor(x), dim=-1)
#         value = self.critic(x)
#         return self.actor(x), value
#
#     def get_action(self, state):
#         # action_probs, _ = self(state)
#         # action = torch.multinomial(action_probs, 1).item()
#         x = self.fc(state)
#         action = torch.tanh(self.actor(x))
#         print(f"[IA2C] Action output: {action.detach().numpy()}")
#         return action.detach().numpy()
#
#     def get_value(self, state):
#         _, value = self(state)
#         return value
