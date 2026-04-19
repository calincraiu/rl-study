from typing import Any, Optional

from tqdm import tqdm
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import wandb

from src.agents.Agent import Agent
from src.solvers.Solver import Solver


class GridWorldSolver(Solver):
    def __init__(
            self, 
            environment: gym.Env, 
            agent_class: type[Agent],
            agent_kwargs: dict[str, Any],
            verbose: bool = False,
            **kwargs
            ):
        super().__init__(name=type(self).__name__, **kwargs)

        self.verbose = verbose
        self.environment = environment
        self.agent = agent_class(
            observation_space=environment.observation_space, 
            action_space=environment.action_space, 
            **agent_kwargs
            )
        self.__agent_kwargs = agent_kwargs # Used for logging

    def train(self, n_epochs: int) -> None:
        # Init wandb for logging
        wandb.init(
            project=type(self).__name__,
            config={
                "n_epochs": n_epochs,
                "agent_type": self.agent.__class__.__name__,
                **self.__agent_kwargs
            }
        )

        # Iterate epochs
        for i in tqdm(range(n_epochs), desc="Epoch", disable=(not self.verbose)):
            obs, info = self.environment.reset()
            done = False
            
            # Iterate steps in epoch
            while not done:
                action = self.agent.actuate(obs)
                next_obs, reward, terminated, truncated, info = self.environment.step(action)

                self.agent.percept(
                    s=obs, 
                    a=int(action), 
                    s_prime=next_obs, 
                    r=float(reward),
                    done=bool(terminated or truncated)
                )
                obs = next_obs # Update state
                if terminated or truncated:
                    done = True

                    # Log Environment Metrics (Automatically computed by Gymnasium Wrapper)
                    if "episode" in info:
                        wandb.log({
                            "env/episode_return": info["episode"]["r"],
                            "env/episode_length": info["episode"]["l"],
                            # Success metric: Assuming 1.0 is the reward for the target cell
                            "env/success": 1 if reward == 1.0 else 0 
                        }, step=self.agent.steps_done)

            # Apply intra-episode update actions (like epsilon decay)
            self.agent.update_episode()

            # Log Agent/Training Metrics at the end of every episode
            metrics = self.agent.get_metrics()
            metrics["epoch"] = i # Add solver-level info
            wandb.log(metrics, step=self.agent.steps_done)

        # Close the wandb run
        wandb.finish()

    def plot_train_history(
            self, 
            reward_history: np.ndarray, 
            total_reward_history: np.ndarray,
            dpi: int = 120
            ):
        fig, axes = plt.subplots(2, 1, figsize=(5, 4), dpi=dpi, sharex='all')
        axes[0].plot(np.arange(len(total_reward_history)), total_reward_history,
                        alpha=0.7, color='#d62728', label=r'$\xi$ = ' + f'{getattr(self.agent, "xi", "unspecified_xi")}')
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