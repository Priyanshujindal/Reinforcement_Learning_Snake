from matplotlib import path
import random
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn import DQN
from .replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self):

        # -------------------------
        # Device
        # ------------------------- 

        self.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", self.device) 

        # -------------------------
        # Environment information
        # -------------------------
        self.state_size = 11
        self.action_size = 3

        # -------------------------
        # DQN hyperparameters
        # -------------------------
        self.gamma = 0.9
        self.learning_rate = 0.001

        # -------------------------
        # Epsilon-greedy
        # -------------------------
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_steps = 0

        # -------------------------
        # Replay buffer
        # -------------------------
        self.batch_size = 64

        self.memory = ReplayBuffer(100_000)

        # -------------------------
        # Neural networks
        # -------------------------
        self.policy_network = DQN(
            self.state_size,
            self.action_size
        ).to(self.device)

        self.target_network = DQN(
            self.state_size,
            self.action_size
        ).to(self.device)   

        # Initially both networks have the same weights
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        # -------------------------
        # Optimizer
        # -------------------------
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        # -------------------------
        # Loss function
        # -------------------------
        self.loss_function = nn.MSELoss()

    # =====================================================
    # Choose an action using epsilon-greedy
    # =====================================================

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        return torch.argmax(
            q_values,
            dim=1
        ).item()


    # =====================================================
    # Train the policy network
    # =====================================================

    def train_step(self):

        # Don't train until we have enough experiences
        if len(self.memory) < self.batch_size:
            return None

        # ---------------------------------------------
        # Sample random experiences
        # ---------------------------------------------

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # ---------------------------------------------
        # Convert to tensors
        # ---------------------------------------------

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # ---------------------------------------------
        # Current Q-values
        # ---------------------------------------------

        current_q_values = self.policy_network(states)

        current_q_values = current_q_values.gather(
            1,
            actions
        ).squeeze(1)

        # ---------------------------------------------
        # Target Q-values
        # ---------------------------------------------

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            )

            max_next_q_values = next_q_values.max(
                dim=1
            )[0]

            target_q_values = rewards + (
                1 - dones
            ) * self.gamma * max_next_q_values

        # ---------------------------------------------
        # Calculate loss
        # ---------------------------------------------

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()


    # =====================================================
    # Update target network
    # =====================================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )


    # =====================================================
    # Update epsilon
    # =====================================================

    def update_epsilon(self):

        self.epsilon_steps += 1

        if self.epsilon_steps % 3_000 == 0:

            if self.epsilon > self.epsilon_min:

                self.epsilon *= self.epsilon_decay

                if self.epsilon < self.epsilon_min:
                    self.epsilon = self.epsilon_min
    
    def save_checkpoint(self, path, episode, total_steps):

        checkpoint = {
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),

            "replay_buffer": list(self.memory.buffer),

            "epsilon": self.epsilon,
            "epsilon_steps": self.epsilon_steps,

            "episode": episode,
            "total_steps": total_steps
        }

        torch.save(checkpoint, path)

        print(f"Checkpoint saved: {path}")


    def load_checkpoint(self, path):

        checkpoint = torch.load(
            path,
            weights_only=False
        )

        self.policy_network.load_state_dict(
            checkpoint["policy_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.memory.buffer.extend(
            checkpoint["replay_buffer"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.epsilon_steps = checkpoint["epsilon_steps"]

        episode = checkpoint["episode"]

        total_steps = checkpoint["total_steps"]

        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from episode: {episode}")
        print(f"Epsilon: {self.epsilon:.4f}")
        print(f"Total steps: {total_steps}")
        print(f"Replay buffer: {len(self.memory)}")

        return episode, total_steps

    def load_model(self, path):

        self.policy_network.load_state_dict(
            torch.load(
                path,
                weights_only=True
            )
        )

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        print(f"Existing model loaded: {path}")


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

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

    print(
        "Epsilon after 10,000 steps:",
        agent.epsilon
    )