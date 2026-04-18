import copy
import gymnasium as gym
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.agents.Agent import Agent


def play_game(agent: Agent, env: gym.Env, num_episodes: int = 1):
    env_copy = copy.copy(env)
    agent_policy = agent.get_policy()
    step_fn: Callable
    if type(agent_policy) == np.ndarray:
        step_fn = lambda x, i: x[i]
    elif isinstance(agent_policy, torch.nn.Module):
        step_fn = lambda x, i: x(
            torch.nn.functional.one_hot(
                torch.tensor(
                    i, 
                    dtype=torch.long, 
                    device=next(x.parameters()).device
                )
            ).float()
        ).argmax().item()

    for episode in range(num_episodes):
        obs, info = env_copy.reset()
        terminated = False
        truncated = False
        total_reward = 0
        num_steps = 0

        print(f"--- Episode {episode + 1} Starting ---")
        while not (terminated or truncated):
            # 1. Choose the best action (Greedy)
            # Use the policy directly to avoid the random epsilon-check in actuate()
            action = step_fn(agent_policy, obs)

            # 2. Take the action
            next_obs, reward, terminated, truncated, info = env_copy.step(action)
            total_reward += float(reward)
            
            # 3. Update current observation
            obs = next_obs

            num_steps += 1

        print(f"Episode Finished. Num steps: {num_steps}; Total reward: {total_reward:.2f}")

    env_copy.close()