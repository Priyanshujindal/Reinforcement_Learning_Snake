from torch.nn.modules import loss
import random
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer

import random
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self):

        # State and action sizes
        self.state_size = 11
        self.action_size = 3

        # Hyperparameters
        self.gamma = 0.9
        self.learning_rate = 0.001

        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Count environment steps for epsilon decay
        self.epsilon_steps = 0

        # Training batch size
        self.batch_size = 64

        # Policy network
        self.policy_network = DQN(
            self.state_size,
            self.action_size
        )

        # Target network
        self.target_network = DQN(
            self.state_size,
            self.action_size
        )

        # Initially make both networks identical
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        # Loss function
        self.loss_function = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(100_000)

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return torch.argmax(q_values, dim=1).item()

    def train_step(self):

        # Don't train until we have enough experiences
        if len(self.memory) < self.batch_size:
            return None

        # Sample experiences
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(
            states,
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # Current Q-values
        current_q_values = self.policy_network(states)

        # Q-value for the action that was actually taken
        current_q_values = current_q_values.gather(
            1,
            actions
        ).squeeze(1)

        # Calculate target Q-values
        with torch.no_grad():

            next_q_values = self.target_network(next_states)

            max_next_q_values = next_q_values.max(
                dim=1
            )[0]

            target_q_values = rewards + (
                1 - dones
            ) * self.gamma * max_next_q_values

        # Calculate loss
        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):

        self.epsilon_steps += 1

        # Decrease epsilon every 10,000 environment steps
        if self.epsilon_steps % 10_000 == 0:

            if self.epsilon > self.epsilon_min:

                self.epsilon *= self.epsilon_decay

                if self.epsilon < self.epsilon_min:
                    self.epsilon = self.epsilon_min

if __name__=="__main__":

    agent = DQNAgent()

    state = [
        0, 0, 0,
        0, 1, 0, 0,
        1, 0, 0, 1
    ]

    action = agent.choose_action(state)

    print("Selected action:", action)
    print("Epsilon:", agent.epsilon)

    for _ in range(10_000):
        agent.update_epsilon()
    print("Epsilon after 10,000 steps:", agent.epsilon)
        

