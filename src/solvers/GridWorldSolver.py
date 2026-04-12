from typing import Any, Optional
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.agents.QLambdaLearner import QLambdaLearner

class GridWorldSolver:
    def __init__(
            self, 
            environment: gym.Env, 
            agent_class: type[QLambdaLearner],
            agent_kwargs: dict[str, Any],
            verbose: bool = False
            ):
        self.verbose = verbose
        self.environment = environment
        self.agent = agent_class(environment, **agent_kwargs)

    def train_one_epoch(self, start_obs: int):
        obs = start_obs
        reward_game = 0
        done = False

        while not done:
            action = self.agent.actuate(obs)
            next_obs, reward, terminated, truncated, info = self.environment.step(action)
            self.agent.percept(
                s=obs, 
                a=int(action), 
                s_prime=next_obs, 
                r=float(reward)
            )
            reward_game += float(reward)

            obs = next_obs # Update state
            if terminated or truncated:
                done = True

        # Apply intra-episode update actions (like epsilon decay)
        self.agent.update_episode(epoch_total_reward=reward_game)
        return reward_game

    def train(self, n_epochs: int) -> tuple[int, np.ndarray, np.ndarray]:
        reward_history = np.zeros(n_epochs)
        total_reward_history = np.zeros(n_epochs)
        total_reward = 0

        for i in tqdm(range(n_epochs), desc="Epoch", disable=(not self.verbose)):
            obs, info = self.environment.reset()
            reward_episode = self.train_one_epoch(start_obs=obs)
            total_reward += reward_episode
            reward_history[i] = reward_episode
            total_reward_history[i] = total_reward
        if self.verbose:
            print(f'Total reward = {total_reward}')

        return n_epochs, reward_history, total_reward_history

    def plot_train_history(
            self, 
            reward_history: np.ndarray, 
            total_reward_history: np.ndarray,
            dpi: int = 120
            ):
        fig, axes = plt.subplots(2, 1, figsize=(5, 4), dpi=dpi, sharex='all')
        axes[0].plot(np.arange(len(total_reward_history)), total_reward_history,
                        alpha=0.7, color='#d62728', label=r'$\xi$ = ' + f'{self.agent.xi}')
        axes[0].set_ylabel('Total rewards')
        axes[0].legend(loc='best')
        axes[1].plot(np.arange(len(reward_history)), reward_history, marker='o', markersize=2,
                        alpha=0.7, color='#2ca02c', linestyle='none')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Reward from\na single game')
        # axes[1].set_ylim(-1000, 100)
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axes[0].grid(axis='x')
        axes[1].grid(axis='x')
        plt.tight_layout()
        plt.show()