import sys
import os
import torch
import pygame

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from game.snake import SnakeGame
from agent.agent import DQNAgent


# =========================================
# Settings
# =========================================

MODEL_PATH = "models/snake_dqn.pth"

# False = fast evaluation without visualization
# True  = watch the Snake play
VISUALIZE = False

# 100 games for reliable statistics
# 1 game when visualization is enabled
NUM_GAMES = 1 if VISUALIZE else 100


def evaluate():

    # =========================================
    # Create environment and agent
    # =========================================

    game = SnakeGame()
    agent = DQNAgent()

    # =========================================
    # Load trained policy network
    # =========================================

    agent.policy_network.load_state_dict(
        torch.load(
            MODEL_PATH,
            weights_only=True
        )
    )

    # No exploration during evaluation
    agent.epsilon = 0.0

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Evaluation games: {NUM_GAMES}")
    print(f"Visualization: {VISUALIZE}")

    # =========================================
    # Metrics
    # =========================================

    scores = []
    episode_lengths = []

    # =========================================
    # Evaluation loop
    # =========================================

    for game_number in range(1, NUM_GAMES + 1):

        state = game.reset()

        done = False
        steps = 0

        while not done:

            # Choose best action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, done = game.step(action)

            state = next_state

            steps += 1

            # ---------------------------------
            # Visualization
            # ---------------------------------

            if VISUALIZE:

                # Handle window events
                for event in pygame.event.get():

                    if event.type == pygame.QUIT:

                        pygame.quit()
                        return

                game.draw()

                game.clock.tick(10)

        # -------------------------------------
        # Save results
        # -------------------------------------

        scores.append(game.score)
        episode_lengths.append(steps)

        print(
            f"Game: {game_number:3d} | "
            f"Score: {game.score:3d} | "
            f"Steps: {steps}"
        )

    # =========================================
    # Evaluation results
    # =========================================

    average_score = (
        sum(scores) / len(scores)
    )

    maximum_score = max(scores)

    minimum_score = min(scores)

    average_steps = (
        sum(episode_lengths)
        / len(episode_lengths)
    )

    # =========================================
    # Print results
    # =========================================

    print("\n" + "=" * 45)

    print("EVALUATION RESULTS")

    print("=" * 45)

    print(
        f"Games evaluated: {NUM_GAMES}"
    )

    print(
        f"Average score:   {average_score:.2f}"
    )

    print(
        f"Maximum score:   {maximum_score}"
    )

    print(
        f"Minimum score:   {minimum_score}"
    )

    print(
        f"Average steps:   {average_steps:.2f}"
    )

    print("=" * 45)

    pygame.quit()


if __name__ == "__main__":
    evaluate()