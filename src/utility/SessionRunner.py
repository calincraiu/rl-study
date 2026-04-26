import gymnasium as gym
from typing import Callable, Any

import numpy as np
import torch
import torch.nn as nn

from src.agents.Agent import Agent


def play_game(agent: Agent, env: gym.Env, num_episodes: int = 1):
    agent_policy = agent.get_policy()
    env_obs_space = env.observation_space

    def _policy_action(observation: Any) -> int:
        if isinstance(agent_policy, torch.nn.Module):
            device = next(agent_policy.parameters()).device
            if isinstance(env_obs_space, gym.spaces.Discrete):
                obs_tensor = torch.nn.functional.one_hot(
                    torch.tensor(observation, dtype=torch.long, device=device),
                    num_classes=int(env_obs_space.n)
                ).float().unsqueeze(0)
            else:
                obs_array = np.asarray(observation, dtype=np.float32)
                obs_tensor = torch.tensor(obs_array, device=device).flatten().unsqueeze(0)
            with torch.no_grad():
                return int(agent_policy(obs_tensor).argmax().item())

        if isinstance(agent_policy, np.ndarray):
            return int(agent_policy[observation])

        if hasattr(agent, "actuate"):
            epsilon_backup = None
            if hasattr(agent, "epsilon"):
                epsilon_backup = getattr(agent, "epsilon")
                setattr(agent, "epsilon", 0.0)
            try:
                return int(agent.actuate(observation))
            finally:
                if epsilon_backup is not None:
                    setattr(agent, "epsilon", epsilon_backup)

        raise ValueError("Unable to derive a greedy action from the provided agent policy.")

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        num_steps = 0

        print(f"--- Episode {episode + 1} Starting ---")
        while not (terminated or truncated):
            action = _policy_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            num_steps += 1

        print(f"Episode Finished. Num steps: {num_steps}; Total reward: {total_reward:.2f}")
