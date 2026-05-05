from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn

from src.agents.Agent import Agent
from src.environments.Gridworld import GridWorldEnv, Cells

_ACTION_VECTORS = {
    0: (1.0, 0.0),   # RIGHT
    1: (0.0, -1.0),  # UP
    2: (-1.0, 0.0),  # LEFT
    3: (0.0, 1.0),   # DOWN
}


def _apply_observation_wrappers(env: gym.Env, observation: Any) -> Any:
    if isinstance(env, gym.ObservationWrapper):
        observation = _apply_observation_wrappers(env.env, observation)
        return env.observation(observation)
    if isinstance(env, gym.Wrapper):
        return _apply_observation_wrappers(env.env, observation)
    return observation


def _observation_to_tensor(observation: Any, env: gym.Env, device: torch.device) -> torch.Tensor:
    if isinstance(observation, dict):
        if "agent" in observation:
            observation = observation["agent"]
        elif len(observation) == 1:
            observation = next(iter(observation.values()))
        else:
            raise ValueError(f"Unsupported dict observation keys: {list(observation.keys())}")

    if isinstance(env.observation_space, gym.spaces.Discrete):
        idx = int(np.asarray(observation).item())
        return torch.nn.functional.one_hot(
            torch.tensor(idx, dtype=torch.long, device=device),
            num_classes=int(env.observation_space.n),
        ).float().unsqueeze(0)

    if isinstance(env.observation_space, gym.spaces.Box):
        obs_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        return torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)

    if isinstance(env.observation_space, gym.spaces.Dict):
        if not isinstance(observation, dict) or "agent" not in observation:
            raise ValueError("Dict observation space expects an {'agent': ...} observation.")
        obs_array = np.asarray(observation["agent"], dtype=np.float32).reshape(-1)
        return torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)

    raise ValueError(f"Unsupported observation space: {type(env.observation_space)}")


def _get_q_values(agent: Agent, observation: Any, goal: Any, env: gym.Env):
    agent_policy = agent.get_policy()

    if not isinstance(agent_policy, torch.nn.Module):
        raise ValueError("This plotting utility currently supports only torch-based agents.")

    device = next(agent_policy.parameters()).device
    state_tensor = _observation_to_tensor(observation, env, device)
    goal_tensor = _observation_to_tensor(goal, env, device)

    with torch.no_grad():
        return agent_policy(state_tensor, goal_tensor)


def _greedy_action(agent: Agent, observation: Any, goal: Any, env: gym.Env) -> int:
    agent_policy = agent.get_policy()

    if not isinstance(agent_policy, torch.nn.Module):
        raise ValueError("This plotting utility currently supports only torch-based agents.")

    device = next(agent_policy.parameters()).device
    obs_tensor = _observation_to_tensor(observation, env, device)
    goal_tensor = _observation_to_tensor(goal, env, device)

    with torch.no_grad():
        return int(agent_policy(obs_tensor, goal_tensor).argmax(dim=-1).item())

    policy = agent.get_policy()
    if isinstance(policy, torch.nn.Module):
        device = next(policy.parameters()).device
        obs_tensor = _observation_to_tensor(observation, env, device)
        with torch.no_grad():
            return int(policy(obs_tensor).argmax().item())

    if isinstance(policy, np.ndarray):
        return int(policy[observation])

    raise ValueError("Unable to derive a greedy action for this agent.")


def plot_value_and_policy(
    agent: Agent,
    env: gym.Env,
    goal: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (14, 6),
    show: bool = True,
    save_path: Optional[Path | str] = None,
    dpi: int = 150,
):
    """Plot the Gridworld value and policy surfaces for a trained agent."""
    if not isinstance(env.unwrapped, GridWorldEnv):
        raise ValueError("plot_value_and_policy currently supports GridWorldEnv environments.")

    base_env = env.unwrapped
    if goal is None:
        goal = base_env.goal_position
    goal = np.asarray(goal, dtype=np.int64)

    grid = base_env.grid
    n_rows, n_cols = grid.shape

    value_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    policy_grid = np.full((n_rows, n_cols), -1, dtype=np.int32)

    raw_goal = base_env.build_observation(goal)
    transformed_goal = _apply_observation_wrappers(env, raw_goal)

    for row in range(n_rows):
        for col in range(n_cols):
            if grid[row, col] == Cells.WALL.value:
                continue

            raw_obs = base_env.build_observation(np.array([row, col], dtype=np.int64))
            observation = _apply_observation_wrappers(env, raw_obs)

            try:
                q_values = _get_q_values(agent, observation, transformed_goal, env)
                q_np = q_values.squeeze(0).detach().cpu().numpy()
                value_grid[row, col] = float(np.max(q_np))
                policy_grid[row, col] = int(np.argmax(q_np))
            except ValueError:
                value_grid[row, col] = np.nan
                policy_grid[row, col] = _greedy_action(agent, observation, transformed_goal, env)

    fig, (ax_grid, ax_value, ax_policy) = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))

    grid_cmap = mcolors.ListedColormap(["white", "green", "red", "black"])
    grid_bounds = [0, 1, 2, 3, 4]
    grid_norm = mcolors.BoundaryNorm(grid_bounds, grid_cmap.N)
    ax_grid.imshow(grid, origin="upper", cmap=grid_cmap, norm=grid_norm)
    ax_grid.set_title("Grid Layout")
    ax_grid.set_xlabel("Column")
    ax_grid.set_ylabel("Row")
    ax_grid.set_xticks(np.arange(n_cols))
    ax_grid.set_yticks(np.arange(n_rows))
    ax_grid.set_xticklabels([])
    ax_grid.set_yticklabels([])
    ax_grid.set_aspect("equal")
    ax_grid.grid(color="white", linestyle="--", linewidth=0.5)
    ax_grid.scatter(int(goal[1]), int(goal[0]), s=120, marker="*", color="yellow", edgecolors="black", linewidths=1.5)

    masked = np.ma.masked_invalid(value_grid)
    cmap = plt.cm.get_cmap("viridis")
    cmap.set_bad(color="lightgray")
    im = ax_value.imshow(masked, origin="upper", cmap=cmap)
    ax_value.set_title("Value Function")
    ax_value.set_xlabel("Column")
    ax_value.set_ylabel("Row")
    fig.colorbar(im, ax=ax_value, shrink=0.8)

    ax_policy.imshow(grid == Cells.WALL.value, origin="upper", cmap="gray", alpha=0.3)
    for row in range(n_rows):
        for col in range(n_cols):
            if grid[row, col] == Cells.WALL.value:
                continue
            action = policy_grid[row, col]
            if action < 0:
                continue
            dx, dy = _ACTION_VECTORS[action]
            ax_policy.quiver(
                col,
                row,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=4,
                pivot="middle",
                color="black",
                width=0.01,
                headwidth=4,
                headlength=5,
            )

    ax_policy.set_title("Greedy Policy")
    ax_policy.set_xlabel("Column")
    ax_policy.set_ylabel("Row")
    ax_policy.set_xticks(np.arange(n_cols))
    ax_policy.set_yticks(np.arange(n_rows))
    ax_policy.set_xticklabels([])
    ax_policy.set_yticklabels([])
    ax_policy.set_aspect("equal")
    ax_policy.grid(color="white", linestyle="--", linewidth=0.5)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.tight_layout()
        plt.show()

    return fig
