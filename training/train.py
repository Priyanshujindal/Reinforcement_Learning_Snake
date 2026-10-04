import sys
import os
import torch

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import matplotlib.pyplot as plt

from game.snake import SnakeGame
from agent.agent import DQNAgent


def train():

    # -----------------------------------------
    # Create environment and agent
    # -----------------------------------------

    game = SnakeGame()
    agent = DQNAgent()

    # -----------------------------------------
    # File paths
    # -----------------------------------------

    checkpoint_path = "models/snake_checkpoint.pth"
    model_path = "models/snake_dqn.pth"

    # -----------------------------------------
    # Default training state
    # -----------------------------------------

    start_episode = 0
    total_steps = 0

    # -----------------------------------------
    # Load previous training state
    # -----------------------------------------

    if os.path.exists(checkpoint_path):

        print("Full checkpoint found.")

        start_episode, total_steps = (
            agent.load_checkpoint(
                checkpoint_path
            )
        )

    # -----------------------------------------
    # If no checkpoint exists
    # -----------------------------------------

    elif os.path.exists(model_path):

        print("No checkpoint found.")
        print("Loading existing trained model...")

        agent.load_model(model_path)

    else:

        print("No model or checkpoint found.")
        print("Starting training from scratch.")

    # -----------------------------------------
    # Training settings
    # -----------------------------------------

    num_episodes = 10000

    warmup_size = 1000 

    target_update_frequency = 1000

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    scores = []
    rewards = []
    losses = []
    epsilons = []
    episode_lengths = []

    # -----------------------------------------
    # Training loop
    # -----------------------------------------

    for episode in range(
        start_episode + 1,
        start_episode + num_episodes + 1
    ):

        state = game.reset()

        total_reward = 0

        episode_loss = []

        episode_steps = 0

        done = False

        # -------------------------------------
        # One episode
        # -------------------------------------

        while not done:

            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, done = game.step(action)

            # Store experience
            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            # ---------------------------------
            # Train only after warm-up
            # ---------------------------------

            if len(agent.memory) >= warmup_size:

                loss = agent.train_step()

                if loss is not None:
                    episode_loss.append(loss)

            # Move to next state
            state = next_state

            # Update epsilon
            agent.update_epsilon()

            # Count environment steps
            total_steps += 1

            episode_steps += 1

            # ---------------------------------
            # Update target network
            # ---------------------------------

            if total_steps % target_update_frequency == 0:

                agent.update_target_network()

            # Add reward
            total_reward += reward

        # -------------------------------------
        # Save metrics
        # -------------------------------------

        scores.append(game.score)

        rewards.append(total_reward)

        episode_lengths.append(episode_steps)

        epsilons.append(agent.epsilon)

        # -------------------------------------
        # Average loss
        # -------------------------------------

        if len(episode_loss) > 0:

            average_loss = (
                sum(episode_loss)
                / len(episode_loss)
            )

        else:

            average_loss = 0

        losses.append(average_loss)

        # -------------------------------------
        # Print progress
        # -------------------------------------

        print(
            f"Episode: {episode:4d} | "
            f"Score: {game.score:3d} | "
            f"Reward: {total_reward:6.1f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Steps: {episode_steps} | "
            f"Buffer: {len(agent.memory)}"
        )

    # =========================================
    # Training finished
    # =========================================

    print("\nTraining complete!")

    # =========================================
    # Save COMPLETE checkpoint
    # =========================================

    os.makedirs("models", exist_ok=True)

    agent.save_checkpoint(
        checkpoint_path,
        episode,
        total_steps
    )  

    # Save policy network separately
    torch.save(
        agent.policy_network.state_dict(),
        model_path
    ) 

    print(
        f"Complete checkpoint saved to: "
        f"{checkpoint_path}"
    )

    # =========================================
    # Score
    # =========================================

    plt.figure()

    plt.plot(scores)

    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.title("Snake Score")

    plt.grid()

    plt.show()

    # =========================================
    # Moving average
    # =========================================

    window = 20

    if len(scores) >= window:

        moving_average = []

        for i in range(
            window - 1,
            len(scores)
        ):

            average = sum(
                scores[
                    i - window + 1:i + 1
                ]
            ) / window

            moving_average.append(average)

        plt.figure()

        plt.plot(
            range(
                window,
                len(scores) + 1
            ),
            moving_average
        )

        plt.xlabel("Episode")
        plt.ylabel("Average Score")

        plt.title(
            "20-Episode Moving Average"
        )

        plt.grid()

        plt.show()

    # =========================================
    # Loss
    # =========================================

    plt.figure()

    plt.plot(losses)

    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid()

    plt.show()

    # =========================================
    # Epsilon
    # =========================================

    plt.figure()

    plt.plot(epsilons)

    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.title("Epsilon Decay")

    plt.grid()

    plt.show()

    # =========================================
    # Episode length
    # =========================================

    plt.figure()

    plt.plot(episode_lengths)

    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.title("Episode Length")

    plt.grid()

    plt.show()


if __name__ == "__main__":
    train()