from enum import Enum
from pathlib import Path
from typing import Optional, Any, Union, Callable, cast
from collections import deque
from builtins import getattr as builtin_getattr
import copy

import math
import warnings
import numpy as np
import gymnasium as gym
import pygame

PlotCallback = Callable[..., Any]


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class Cells(Enum):
    TILE = 0
    TARGET = 1
    PITFALL = 2
    WALL = 3


class GridWorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 4,
        "pygame_window_size": 512
    }

    __m_default_grid: np.ndarray = np.array((
        [0,0,0,1],
        [0,3,0,2],
        [0,0,0,0]
    ))
    """0 = regular gruid square, 1 = target, 2 = bad termination condition, 3 = obstacle"""

    __m_default_reward: dict = {
        Cells.TILE.value: -0.04, # reward for regular tile
        Cells.TARGET.value: 1.0, # reward for target / good ending
        Cells.PITFALL.value: -1.0, # reward for bad ending
        Cells.WALL.value: -0.1 # reward for hitting obstacle
    }

    __m_colors = {
        Cells.TILE.value: (255, 255, 255),
        Cells.WALL.value: (0, 0, 0),
        Cells.TARGET.value: (0, 255, 0),
        Cells.PITFALL.value: (255, 0, 0),
    }

    __action_to_direction = {
        Actions.UP.value: np.array([-1, 0]),
        Actions.DOWN.value: np.array([1, 0]),
        Actions.LEFT.value: np.array([0, -1]),
        Actions.RIGHT.value: np.array([0, 1]),
    }

    def __init__(
            self, 
            grid: Optional[np.ndarray] = None, 
            reward: Optional[dict] = None, 
            render_mode: Optional[str] = None,
            wind_p: float = 0.2, # probability that the 'wind' will blow the agent off-course to a lateral step
            step_limit: int = 1000, # num steps permitted per game before truncation
            pad_type: Cells = Cells.WALL # pad grid boundary with this tile type
            ):
        super().__init__()

        self.wind_p = wind_p
        
        grid = grid if grid is not None else self.__m_default_grid
        self.grid = self.__pad_grid(grid, pad_type)
        self.rewards: dict = reward if reward is not None else self.__m_default_reward
        self.num_rows, self.num_cols = self.grid.shape
        
        self.__target_position: np.ndarray = np.argwhere(self.grid == Cells.TARGET.value)[0]
        self.distance_map = self.__compute_distance_map()
    
        self.max_distance_from_target = np.max(self.distance_map[np.isfinite(self.distance_map)]) 

        self.observation_space = gym.spaces.Dict(
            {
                "agent" : gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.num_rows - 1, self.num_cols - 1]),
                    shape=(2,),
                    dtype=np.int64
                )
            }
        )

        self.action_space = gym.spaces.Discrete(len(self.__action_to_direction))
        self.__agent_position: Optional[np.ndarray] = None
        self.__step_limit = step_limit
        self.__current_step = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.__window = None
        self.__clock = None

    def __pad_grid(self, grid: np.ndarray, pad_type: Cells) -> np.ndarray:
        return np.pad(grid, constant_values=pad_type.value, pad_width=1)

    def build_observation(self, agent_position: np.ndarray) -> dict:
        return {
            "agent": np.asarray(agent_position, dtype=np.int64)
        }

    @property
    def goal_position(self) -> np.ndarray:
        return self.__target_position.copy()
    
    def __get_obs(self) -> dict:
        return {
            "agent": self.__agent_position.copy() if self.__agent_position is not None else None,
        }

    def __get_info(self) -> dict:
        return {
            "distance": np.linalg.norm(
                self.__agent_position - self.__target_position, ord=1
            ),
            "goal": self.__target_position.copy(),
        }

    def __set_agent_position(self, position: Optional[np.ndarray] = None) -> None:
        if position is not None:
            self.__agent_position = np.asarray(position, dtype=np.int64).copy()
        else:
            valid_positions = np.argwhere(self.grid == Cells.TILE.value)
            idx = self.np_random.integers(len(valid_positions))
            self.__agent_position = valid_positions[idx].copy()

    def __set_agent_position_by_curriculum(self, curriculum_distance: int, curriculum_threshold):
        if curriculum_threshold < 0.0: curriculum_threshold = 0.0
        if curriculum_threshold > 1.0: curriculum_threshold = 1.0
        if curriculum_distance < 0: curriculum_distance = 0
        if curriculum_distance > self.max_distance_from_target: curriculum_distance = self.max_distance_from_target

        if self.np_random.random() < curriculum_threshold:
            positions = self.__get_positions_within_distance(curriculum_distance)
        else:
            positions = self.__get_positions_within_distance(self.max_distance_from_target)

        idx = self.np_random.integers(len(positions))
        self.__agent_position = positions[idx]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed) 

        # --- Goal placement ---
        # Note: Moved before agent placement so curriculum logic relies on the newly updated distance map
        if options:
            goal_placement_strategy: Optional[str] = options.get("goal_placement_strategy", None)
            if goal_placement_strategy is not None:
                self.__set_goal_position(goal_placement_strategy)

        # --- Agent placement ---
        if options:
            agent_start_position_strategy: str = options.get("agent_start_position_strategy", "random")
            position_strategy_params: dict = options.get("position_strategy_params", {})

            if agent_start_position_strategy == "curriculum":
                curriculum_distance = position_strategy_params.get("curriculum_distance", self.distance_map.max())
                curriculum_threshold = position_strategy_params.get("curriculum_threshold", 0.8)
                self.__set_agent_position_by_curriculum(curriculum_distance, curriculum_threshold)
            elif agent_start_position_strategy == "fixed_position":
                agent_fixed_position = position_strategy_params.get("agent_fixed_position", np.array([0, 0]))
                self.__set_agent_position(agent_fixed_position)
            else:
                self.__set_agent_position()
        else:
            self.__set_agent_position()

        self.__current_step = 0

        if self.render_mode == "human":
            self.render()

        return self.__get_obs(), self.__get_info()
    
    def step(self, action: Union[int, Actions]) -> tuple[dict, float, bool, bool, dict]:
        if isinstance(action, Actions):
            action = action.value 

        if self.np_random.random() < self.wind_p:
            if self.np_random.random() < 0.5:
                actual_action = (action + 1) % 4  
            else:
                actual_action = (action - 1) % 4  
        else:
            actual_action = action

        direction = self.__action_to_direction[actual_action]

        new_agent_position = np.clip(
            self.__agent_position + direction, [0, 0], [self.num_rows - 1, self.num_cols - 1]
        )
        
        cell_type = self.grid[new_agent_position[0], new_agent_position[1]]

        if cell_type == Cells.WALL.value:
            new_agent_position = self.__agent_position 
        
        terminated = cell_type == Cells.TARGET.value or cell_type == Cells.PITFALL.value
        self.__set_agent_position(new_agent_position)
        reward = self.rewards[cell_type]
        self.__current_step += 1
        truncated = self.__current_step >= self.__step_limit

        if self.render_mode == "human":
            self.render()

        return self.__get_obs(), reward, terminated, truncated, self.__get_info()
    
    def get_max_abs_reward(self) -> float:
        return np.max(np.abs(list(self.rewards.values())))
    
    def __compute_distance_map(self) -> np.ndarray:
        distances = np.full((self.num_rows, self.num_cols), np.inf)
        visited = np.zeros_like(distances, dtype=bool)

        queue = deque()
        goal = tuple(self.__target_position)

        queue.append((goal, 0))
        distances[goal] = 0
        visited[goal] = True

        while queue:
            (r, c), dist = queue.popleft()

            for direction in self.__action_to_direction.values():
                nr, nc = r + direction[0], c + direction[1]

                if not (0 <= nr < self.num_rows and 0 <= nc < self.num_cols):
                    continue
                if self.grid[nr, nc] == Cells.WALL.value:
                    continue

                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    distances[nr, nc] = dist + 1
                    queue.append(((nr, nc), dist + 1))

        return distances
    
    def __get_positions_within_distance(self, max_distance: int) -> np.ndarray:
        mask = (
            (self.distance_map <= max_distance) &
            np.isfinite(self.distance_map) &
            (self.grid != Cells.WALL.value) &
            (self.grid != Cells.PITFALL.value) &
            (self.grid != Cells.TARGET.value)
        )
        return np.argwhere(mask)
    
    def __set_goal_position(self, strategy: str) -> None:
        self.grid[tuple(self.__target_position)] = Cells.TILE.value

        if strategy == "random":
            options = np.argwhere(self.grid == Cells.TILE.value)
            if options.size == 0:
                raise RuntimeError("No valid goal positions available for random goal placement")
            goal_idx = self.np_random.integers(len(options))
            goal_loc = options[goal_idx]
            self.grid[tuple(goal_loc)] = Cells.TARGET.value
        else:
            raise ValueError(f"Unsupported goal placement strategy: {strategy}")

        self.__target_position = np.argwhere(self.grid == Cells.TARGET.value)[0].copy()

        self.distance_map = self.__compute_distance_map()
        self.max_distance_from_target = np.max(self.distance_map[np.isfinite(self.distance_map)])
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self.__render_frame()
        elif self.render_mode == "human":
            self.__render_frame()

    def __render_frame(self):
        agent_pos = self.__agent_position
        assert agent_pos is not None

        window_size = self.metadata["pygame_window_size"]
        aspect_ratio = float(self.num_rows) / float(self.num_cols)

        if aspect_ratio > 1.0:
            window_height = window_size
            window_width = int(window_size / aspect_ratio)
        else:
            window_width = window_size
            window_height = int(window_size * aspect_ratio)

        if self.__window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.__window = pygame.display.set_mode((window_width, window_height))

        if self.__clock is None and self.render_mode == "human":
            self.__clock = pygame.time.Clock()

        canvas = pygame.Surface((window_width, window_height))
        canvas.fill((255, 255, 255))

        pix_square_size = min(window_width / self.num_cols, window_height / self.num_rows)

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell = self.grid[row, col]
                color = self.__m_colors[cell]
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        col * pix_square_size,
                        row * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (
                (agent_pos[1] + 0.5) * pix_square_size,
                (agent_pos[0] + 0.5) * pix_square_size,
            ),
            pix_square_size / 3,
        )

        for x in range(self.num_cols + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (x * pix_square_size, 0),
                (x * pix_square_size, window_height),
                width=2,
            )

        for y in range(self.num_rows + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, y * pix_square_size),
                (window_width, y * pix_square_size),
                width=2,
            )

        if self.render_mode == "human":
            assert self.__window is not None
            assert self.__clock is not None
            self.__window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.__clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2),
            )
    
    def close(self):
        if self.__window is not None:
            pygame.display.quit()
            pygame.quit()


class DiscreteGridWorldWrapper(gym.ObservationWrapper):
    def __init__(self, env: GridWorldEnv):
        super().__init__(env)
        self.observation_space = gym.spaces.Discrete(env.num_rows * env.num_cols)

    def observation(self, obs):
        row, col = obs["agent"]
        return int(row * self.get_wrapper_attr("num_cols") + col)
    

class NormalizedCoordWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.rows = self.get_wrapper_attr("num_rows")
        self.cols = self.get_wrapper_attr("num_cols")

    def observation(self, obs):
        r, c = obs["agent"]
        return np.array([r / max(1, self.rows - 1), c / max(1, self.cols - 1)], dtype=np.float32)
    
    
class PatienceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, patience: int = 5):
        super().__init__(env)
        self.patience = patience
        self.stuck_counter = 0
        self.last_obs: Any = None

    # FIX: Ensure it correctly compares observation states independent of nested dictionary layouts.
    def _extract_obs_state(self, obs: Any) -> Any:
        if isinstance(obs, dict) and "agent" in obs:
            return obs["agent"].copy()
        elif isinstance(obs, np.ndarray):
            return obs.copy()
        else:
            return copy.deepcopy(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.stuck_counter = 0
        self.last_obs = self._extract_obs_state(obs) 
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_obs = self._extract_obs_state(obs)
        
        if np.array_equal(current_obs, self.last_obs):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_obs = current_obs
            
        if self.stuck_counter >= self.patience:
            truncated = True
            info["patience_exceeded"] = True 
            
        return obs, reward, terminated, truncated, info


class PlotSavingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        save_path: str | Path,
        agent: Optional[Any] = None,
        plot_callback: Optional[PlotCallback] = None,
        save_every_n_episodes: Optional[int] = None,
        save_every_n_steps: Optional[int] = None,
        show: bool = False,
        dpi: int = 150,
    ):
        super().__init__(env)
        self.agent = agent
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.plot_callback = plot_callback
        self.save_every_n_episodes = save_every_n_episodes
        self.save_every_n_steps = save_every_n_steps
        self.show = show
        self.dpi = dpi
        self.episode_id = -1
        self.step_id = 0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.episode_id += 1
        self.step_id = 0
        self._maybe_save_plot()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_id += 1
        self._maybe_save_plot()
        return obs, reward, terminated, truncated, info

    def _maybe_save_plot(self):
        if self.plot_callback is None or self.agent is None:
            return

        # FIX: Check `self.step_id == 0` so episode triggers don't duplicate loop across the whole episode.
        if self.save_every_n_episodes is not None and self.episode_id >= 0 and self.step_id == 0:
            if self.episode_id % self.save_every_n_episodes == 0:
                self._save_plot(self.episode_id)
                return

        if self.save_every_n_steps is not None and self.step_id > 0:
            if self.step_id % self.save_every_n_steps == 0:
                self._save_plot(self.episode_id, self.step_id)

    def _save_plot(self, episode_id: int, step_id: Optional[int] = None):
        if self.plot_callback is None:
            return

        file_name = f"gridworld_plot_ep{episode_id:05d}"
        if step_id is not None:
            file_name += f"_step{step_id:05d}"
        file_name += ".png"
        file_path = self.save_path / file_name

        try:
            fig = self.plot_callback(
                self.agent,
                self,
                goal=getattr(self.unwrapped, "goal_position", None),  # type: ignore[attr-defined]
                show=self.show,
                save_path=file_path,
                dpi=self.dpi,
            )
            if fig is not None and not self.show:
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Failed to save plot at episode={episode_id}, step={step_id}: {exc}", stacklevel=2)

    def set_agent(self, agent: Any) -> None:
        self.agent = agent