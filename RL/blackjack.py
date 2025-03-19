import argparse
from collections import defaultdict
import time
from contextlib import closing
import pickle

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class BlackjackAgent:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon."""
        self.q_values = defaultdict(lambda: np.zeros(2))  # 2 actions for Blackjack
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)  # Random action
        return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_values), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_values = defaultdict(lambda: np.zeros(2), pickle.load(f))


def train_agent(n_episodes=100_0000, learning_rate=0.001,
                start_epsilon=1.0, final_epsilon=0.1):
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    epsilon_decay = start_epsilon / (n_episodes / 2)

    agent = BlackjackAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    env.close()
    return agent


def visualize_results(agent, env):
    def get_moving_avgs(arr, window, mode):
        return np.convolve(np.array(arr).flatten(), np.ones(window), mode) / window

    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Plotting code remains the same
    # [Add your plotting code here]

    plt.savefig("training_results.png")
    plt.close()


def evaluate_agent(agent):
    with closing(gym.make("Blackjack-v1", render_mode="human", sab=False)) as env:
        for episode in range(5):  # Run multiple games
            observation, _ = env.reset()
            episode_over = False
            print(f"\n=== Starting Game {episode} ===")

            while not episode_over:
                action = agent.get_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)

                print(f"Observation: {observation} | Action: {action} | Reward: {reward}")
                time.sleep(1)
                episode_over = terminated or truncated

            print(f"=== Game {episode} finished ===")
            time.sleep(3)  # Pause between games

        time.sleep(5) # Pause before closing the window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate Blackjack RL agent')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agent')
    args = parser.parse_args()

    if args.train:
        agent = train_agent()
        agent.save("blackjack_agent.pkl")
        print("Training completed and model saved.")

    if args.evaluate:
        agent = BlackjackAgent(
            learning_rate=0.01,
            initial_epsilon=0.1,
            epsilon_decay=0,
            final_epsilon=0.1
        )
        agent.load("blackjack_agent.pkl")
        evaluate_agent(agent)