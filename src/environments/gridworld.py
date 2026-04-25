from enum import Enum
from typing import Optional, Any, Union
from collections import deque

import math
import numpy as np
import gymnasium as gym
import pygame


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
        
        self.__target_position: np.ndarray = np.argwhere(self.grid == Cells.TARGET.value)[0] # Compute target position for distance map
        self.distance_map = self.__compute_distance_map() # Distance map (BFS) for agent position scheduling
        self.max_distance_from_target = self.max_distance_from_target = np.max(self.distance_map[np.isfinite(self.distance_map)]) # Maximum distance from the goal from anywhere on the grid (taking into account obstacles)

        # Observation space - can be multipart - here it includes only the agent position
        # But it can include any observations that we deep relevant. Examples: a field of vision,
        # the location of the target (could be considered cheating), distance to nearest wall, etc.
        self.observation_space = gym.spaces.Dict(
            {
                "agent" : gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.num_rows, self.num_cols]),
                    shape=(2,),
                    dtype=np.int64
                )
            }
        )

        # Action space
        self.action_space = gym.spaces.Discrete(len(self.__action_to_direction))

        # Agent
        self.__agent_position: Optional[np.ndarray] = None

        # Truncation condition
        self.__step_limit = step_limit
        self.__current_step = 0

        # Environment rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.__window = None
        self.__clock = None

    def __pad_grid(self, grid: np.ndarray, pad_type: Cells) -> np.ndarray:
        """Pads the grid with a specified cell type."""
        return np.pad(grid, constant_values=pad_type.value, pad_width=1)

    def __get_obs(self) -> dict:
        """
        Convert internal state to observation format. This corresponds to the observation_space structure.
        """
        return {
            "agent" : self.__agent_position,
        }
    
    def __get_info(self) -> dict:
        """
        Compute auxiliary information.

        Returns:
            dict: Info with Manhattan distance between agent and target (for debugging).
        """
        return {
            "distance": np.linalg.norm(
                self.__agent_position - self.__target_position, ord=1
            )
        }
    
    def __set_agent_position(self, position: Optional[np.ndarray] = None) -> None:
        """
        Set the agent position on the grid.

        Args:
            position (np.ndarray | None): This can be a specific position given as [x, y] coordinates or None.
                If None, a random valid position on the grid will be assigned. 
        """
        if position is not None: 
            self.__agent_position = position # Specific position
        else:
            valid_positions = np.argwhere(self.grid == Cells.TILE.value) # All regular grid positions / tiles
            idx = np.random.randint(len(valid_positions)) # Random choice
            self.__agent_position = valid_positions[idx]

    def __set_agent_position_by_curriculum(self, curriculum_distance: int, curriculum_threshold):
        """
        Set an agent position within a specific distance to the target.

        Args:
            curriculum_distance (int): Set the position of the agent randomly within curriculum_distance of the target.
            curriculum_threshold (float): The threshold for deciding whether to use curriculum learning or random placement.
                Higher threshold means more curriculum. Full curriculum learning may lead to overfitting.
        """
        if curriculum_threshold < 0.0: curriculum_threshold = 0.0
        if curriculum_threshold > 1.0: curriculum_threshold = 1.0
        if curriculum_distance < 0: curriculum_distance = 0
        if curriculum_distance > self.max_distance_from_target: curriculum_distance = self.max_distance_from_target

        if self.np_random.random() < curriculum_threshold: # curriculum
            positions = self.__get_positions_within_distance(curriculum_distance)
        else: # random / max
            positions = self.__get_positions_within_distance(self.max_distance_from_target)

        idx = self.np_random.integers(len(positions))
        self.__agent_position = positions[idx]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        """
        Start a new episode.

        Args:
            seed (Optional[int]): Random seed for reproducible episodes
            options Optional[dict]: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        # --- Seed reset
        super().reset(seed=seed) # Must call this first to seed the random number generator

        # --- Agent placement
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
                self.__set_agent_position() # random
        else:
            self.__set_agent_position() # no options -> random

        # --- Truncation condition
        self.__current_step = 0

        # --- Rendering
        if self.render_mode == "human":
            self.render()

        # --- Return
        return self.__get_obs(), self.__get_info()
    
    def step(self, action: Union[int, Actions]) -> tuple[dict, float, bool, bool, dict]:
        """Execute one timestep within the environment.

        Args:
            action (int): The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # --- Act in the environment
        if isinstance(action, Actions):
            action = action.value # get int from Actions

        # --- Stochastic Mechanic (Wind) ---
        # Roll for wind: 0 = No wind, 1 = Left wind, 2 = Right wind
        if self.np_random.random() < self.wind_p:
            # 50/50 chance to blow left or right relative to intended direction
            if self.np_random.random() < 0.5:
                actual_action = (action + 1) % 4  # Relative Left
            else:
                actual_action = (action - 1) % 4  # Relative Right
        else:
            actual_action = action

        # Map the discrete action (0-3) to a movement direction
        direction = self.__action_to_direction[actual_action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        new_agent_position = np.clip(
            self.__agent_position + direction, [0, 0], [self.num_rows - 1, self.num_cols - 1]
        )
        # Verify cell type at the new position
        cell_type = self.grid[new_agent_position[0], new_agent_position[1]]

        # Verify if it hit an obstruction
        if cell_type == Cells.WALL.value:
            new_agent_position = self.__agent_position # Go back to the previous cell
        
        # Check if the agent reached a termination state
        terminated = cell_type == Cells.TARGET.value or cell_type == Cells.PITFALL.value

        # Update agent 
        self.__set_agent_position(new_agent_position)

        # --- Truncation (optional)
        truncated = self.__current_step >= self.__step_limit
        self.__current_step += 1

        # --- Assign reward
        reward = self.rewards[cell_type]

        # --- Rendering
        if self.render_mode == "human":
            self.render()

        # --- Return
        observation = self.__get_obs()
        info = self.__get_info()

        return observation, reward, terminated, truncated, info
    
    def get_max_abs_reward(self) -> float:
        """
        Get the maximum absolut reward value from the environment's reward function.
        This is useful for doing reward scaling so function-approximation agents don't have exploding gradients.
        """
        return np.max(np.abs(list(self.rewards.values())))
    
    def __compute_distance_map(self) -> np.ndarray:
        """
        Compute shortest-path distance from every reachable tile to the goal.
        Walls are treated as impassable.
        
        Returns:
            np.ndarray of shape (rows, cols) with distances (inf if unreachable)
        """
        distances = np.full((self.num_rows, self.num_cols), np.inf)
        visited = np.zeros_like(distances, dtype=bool)

        queue = deque()
        goal = tuple(self.__target_position)

        # Start from goal
        queue.append((goal, 0))
        distances[goal] = 0
        visited[goal] = True

        while queue:
            (r, c), dist = queue.popleft()

            for direction in self.__action_to_direction.values():
                nr, nc = r + direction[0], c + direction[1]

                # Bounds check
                if not (0 <= nr < self.num_rows and 0 <= nc < self.num_cols):
                    continue

                # Skip walls
                if self.grid[nr, nc] == Cells.WALL.value:
                    continue

                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    distances[nr, nc] = dist + 1
                    queue.append(((nr, nc), dist + 1))

        return distances
    
    def __get_positions_within_distance(self, max_distance: int) -> np.ndarray:
        """
        Get all valid positions within a given shortest-path distance from the goal.

        Args:
            max_distance (int): Maximum distance from goal

        Returns:
            np.ndarray of positions [[r, c], ...]
        """
        mask = (
            (self.distance_map <= max_distance) & # is within distance
            np.isfinite(self.distance_map) & # is finite distance (reachable)
            (self.grid != Cells.WALL.value) & # is not a wall
            (self.grid != Cells.PITFALL.value) & # is not a pitfall
            (self.grid != Cells.TARGET.value) # is not the target itself
        )
        return np.argwhere(mask)
    
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
        window_width: int = 0
        window_height: int = 0
        if aspect_ratio > 1.0:
            window_width = int(aspect_ratio * window_size)
            window_height = window_size
        else:
            window_height = int(aspect_ratio * window_size)
            window_width = window_size

        if self.__window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.__window = pygame.display.set_mode((window_width, window_height))

        if self.__clock is None and self.render_mode == "human":
            self.__clock = pygame.time.Clock()

        canvas = pygame.Surface((window_width, window_height))
        canvas.fill((255, 255, 255))

        pix_square_size = window_size / max(self.num_rows, self.num_cols)

        # --- Draw grid cells
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

        # --- Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (
                (agent_pos[1] + 0.5) * pix_square_size,
                (agent_pos[0] + 0.5) * pix_square_size,
            ),
            pix_square_size / 3,
        )

        # --- Grid lines
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
        # Update the space so the Agent knows the new total state count
        self.observation_space = gym.spaces.Discrete(env.num_rows * env.num_cols)

    def observation(self, obs):
        # Convert {"agent": [r, c]} -> single integer index
        row, col = obs["agent"]
        return int(row * self.get_wrapper_attr("num_cols") + col)
    

class NormalizedCoordWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Define the new space: 2 coordinates (row, col) normalized between 0 and 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Access unwrapped env for grid dimensions
        self.rows = self.get_wrapper_attr("num_rows")
        self.cols = self.get_wrapper_attr("num_cols")

    def observation(self, obs):
        # Extract [row, col] from the "agent" dict
        r, c = obs["agent"]
        # Normalize to [0, 1] range for better MLP performance
        return np.array([r / (self.rows - 1), c / (self.cols - 1)], dtype=np.float32)
    
    
class PatienceWrapper(gym.Wrapper):
    """
    Truncates the episode if the agent remains in the exact same position 
    for a specified number of consecutive steps.
    """
    def __init__(self, env: gym.Env, patience: int = 5):
        super().__init__(env)
        self.patience = patience
        self.stuck_counter = 0
        self.last_position: Any = None

    def reset(self, **kwargs):
        # Reset the environment and clear our tracking variables
        obs, info = self.env.reset(**kwargs)
        self.stuck_counter = 0
        
        # Store the initial position. 
        # We assume this is applied to the base GridWorldEnv where obs is a Dict
        self.last_position = obs["agent"].copy() 
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_position = obs["agent"]
        
        # Check if the agent actually moved
        if np.array_equal(current_position, self.last_position):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_position.copy()
            
        # If the agent has been stuck for too long, truncate the episode
        if self.stuck_counter >= self.patience:
            truncated = True
            info["patience_exceeded"] = True # Optional: Good for debugging/logging
            
        return obs, reward, terminated, truncated, info