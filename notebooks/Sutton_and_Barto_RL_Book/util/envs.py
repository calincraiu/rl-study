import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from IPython.display import HTML, display
from typing import Optional, Sequence, Tuple, Union


State = tuple[int, int]
Action = tuple[int, int]
ActionInput = Union[int, Action]


class BaseEnv:
    """Minimal notebook environment interface."""

    def reset(self):
        raise NotImplementedError("reset() must be implemented by subclasses.")

    def step(self, action):
        raise NotImplementedError("step() must be implemented by subclasses.")


class GridEnvBase(BaseEnv):
    """Shared behavior for simple 2D grid environments."""

    DEFAULT_ACTION_SET: Sequence[Action] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
            self,
            grid_shape: tuple[int, int],
            start_position: State,
            goal_position: State,
            action_set: Optional[Sequence[Action]] = None,
    ):
        self.num_rows, self.num_cols = grid_shape
        self.start_position = start_position
        self.goal_position = goal_position
        self.action_set = tuple(action_set) if action_set is not None else tuple(self.DEFAULT_ACTION_SET)
        self.current_position = self.start_position

    def reset(self) -> State:
        self.current_position = self.start_position
        return self.current_position

    def step(self, action: ActionInput) -> tuple[float, State, bool]:
        delta = self._normalize_action(action)
        candidate = (
            self.current_position[0] + delta[0],
            self.current_position[1] + delta[1],
        )
        candidate = self._clip_to_bounds(candidate)

        if self._is_blocked(candidate):
            candidate = self.current_position

        done = self._is_terminal(candidate)
        reward = self._get_reward(candidate, self.current_position, action)
        self.current_position = candidate

        return reward, self.current_position, done

    def _normalize_action(self, action: ActionInput) -> Action:
        if isinstance(action, int):
            if action < 0 or action >= len(self.action_set):
                raise ValueError(f"Action index {action} is out of range.")
            return self.action_set[action]
        if isinstance(action, tuple) and len(action) == 2:
            return action
        raise ValueError("Action must be an integer index or a (dr, dc) tuple.")

    def _clip_to_bounds(self, position: State) -> State:
        row, col = position
        row = max(0, min(self.num_rows - 1, row))
        col = max(0, min(self.num_cols - 1, col))
        return row, col

    def _is_blocked(self, position: State) -> bool:
        return False

    def _is_terminal(self, position: State) -> bool:
        return position == self.goal_position

    def _get_reward(self, position: State, previous_position: State, action: ActionInput) -> float:
        return 0.0

    def plot_training_metrics(
            self,
            episode_lengths,
            episode_returns,
            smoothing_window=10
    ):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(episode_lengths, alpha=0.4)
        if len(episode_lengths) >= smoothing_window:
            smoothed = np.convolve(
                episode_lengths,
                np.ones(smoothing_window) / smoothing_window,
                mode="valid"
            )
            ax[0].plot(
                range(smoothing_window - 1, len(episode_lengths)),
                smoothed,
                linewidth=2
            )

        ax[0].set_title("Episode Length")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Steps to Goal")

        ax[1].plot(episode_returns, alpha=0.4)
        if len(episode_returns) >= smoothing_window:
            smoothed = np.convolve(
                episode_returns,
                np.ones(smoothing_window) / smoothing_window,
                mode="valid"
            )
            ax[1].plot(
                range(smoothing_window - 1, len(episode_returns)),
                smoothed,
                linewidth=2
            )

        ax[1].set_title("Episode Return")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Return")

        plt.tight_layout()
        plt.show()

    def plot_time_steps_per_episode(self, episode_lengths):
        cumulative_steps = np.cumsum(episode_lengths)
        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_steps, np.arange(len(episode_lengths)))
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.title("Learning Curve")
        plt.grid(True)
        plt.show()

    def plot_policy(self, Q_pi, action_set: Optional[Sequence[Action]] = None):
        action_set = action_set if action_set is not None else self.action_set

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.dpi = 120

        ax.set_xlim(-0.5, self.num_cols - 1 + 0.5)
        ax.set_ylim(self.num_rows - 1 + 0.5, -0.5)
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(self.num_cols))
        ax.set_yticks(np.arange(self.num_rows))
        ax.set_xticks(np.arange(-0.5, self.num_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.num_rows, 1), minor=True)
        ax.grid(which="minor", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        X, Y, U, V = [], [], [], []

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                s = (row, col)
                
                # --- NEW WALL LOGIC ---
                if self._is_blocked(s):
                    # Draw a dark gray rectangle. Offset by -0.5 to snap to grid lines.
                    rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor='dimgray')
                    ax.add_patch(rect)
                    continue # Skip drawing an arrow for walls
                # ----------------------

                if s == self.goal_position:
                    continue

                best_action_idx = int(np.argmax(Q_pi[s]))
                d_row, d_col = action_set[best_action_idx]

                X.append(col)
                Y.append(row)
                U.append(d_col)
                V.append(d_row)

        ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=3.0, pivot="middle")

        start_row, start_col = self.start_position
        goal_row, goal_col = self.goal_position

        ax.scatter(start_col, start_row, marker="s", s=250, label="Start")
        ax.scatter(goal_col, goal_row, marker="*", s=350, label="Goal")

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Greedy Policy")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def animate_episode_with_trail(
            self,
            Q_pi,
            action_set: Optional[Sequence[Action]] = None,
            starting_state: Optional[State] = None,
            max_steps=500,
            interval=150,
    ):
        action_set = action_set if action_set is not None else self.action_set

        states = []
        starting_state = starting_state if starting_state else self.reset()
        self.current_position = starting_state
        s = starting_state
        states.append(s)

        for _ in range(max_steps):
            action_idx = int(np.argmax(Q_pi[s]))
            _, s_prime, done = self.step(action_set[action_idx])
            states.append(s_prime)
            s = s_prime
            if done:
                break

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-0.5, self.num_cols - 1 + 0.5)
        ax.set_ylim(self.num_rows - 1 + 0.5, -0.5)
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(self.num_cols))
        ax.set_yticks(np.arange(self.num_rows))
        ax.set_xticks(np.arange(-0.5, self.num_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.num_rows, 1), minor=True)
        ax.grid(which="minor")
        ax.tick_params(which="minor", bottom=False, left=False)

        # --- NEW WALL LOGIC ---
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if self._is_blocked((row, col)):
                    rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor='dimgray')
                    ax.add_patch(rect)
        # ----------------------

        start_row, start_col = starting_state
        goal_row, goal_col = self.goal_position

        ax.scatter(start_col, start_row, marker="s", s=250, label="Start")
        ax.scatter(goal_col, goal_row, marker="*", s=350, label="Goal")

        trail, = ax.plot([], [], linewidth=2)
        agent = ax.scatter([], [], s=150)
        title = ax.set_title("Step 0")

        def update(frame):
            rows = [state[0] for state in states[:frame + 1]]
            cols = [state[1] for state in states[:frame + 1]]
            trail.set_data(cols, rows)
            agent.set_offsets([[cols[-1], rows[-1]]])
            title.set_text(f"Step {frame}")
            return trail, agent

        anim = FuncAnimation(
            fig,
            update,
            frames=len(states),
            interval=interval,
            blit=True,
            repeat=False,
        )

        display(HTML(anim.to_jshtml()))
        plt.close(fig)
        return anim


class RandomWalkEnv(BaseEnv):

    def __init__(self, n_states: int):
        self.n_states = n_states
        self.s = self.n_states // 2

    def reset(self) -> int:
        self.s = self.n_states // 2
        return self.s

    def step(self, a: int) -> tuple[float, int, bool]:
        reward = 0
        s_prime = self.s + a
        if s_prime == 0:
            reward = -1
        if s_prime == self.n_states - 1:
            reward = 1
        self.s = s_prime
        finished = s_prime == 0 or s_prime == self.n_states - 1
        return reward, s_prime, finished


class WindyGridWorldEnv(GridEnvBase):

    def __init__(
            self,
            grid_shape: tuple[int, int],
            goal_position: State,
            start_position: State,
            wind_per_column: Sequence[int],
            stochastic_wind: bool,
            action_set: Optional[Sequence[Action]] = None,
    ):
        super().__init__(grid_shape, start_position, goal_position, action_set)
        self.wind_per_column = tuple(wind_per_column)
        self.stochastic_wind = stochastic_wind

    def _apply_stochastic_wind(self, col: int) -> int:
        wind = self.wind_per_column[col]
        p = np.random.random()
        if p <= 1 / 3:
            wind -= 1
        elif p >= 2 / 3:
            wind += 1
        return -wind

    def step(self, a: Action) -> tuple[float, State, bool]:
        old_row, old_col = self.current_position
        wind = self._apply_stochastic_wind(old_col) if self.stochastic_wind else -self.wind_per_column[old_col]

        h_move = a[1]
        v_move = a[0] + wind
        candidate = (
            old_row + v_move,
            old_col + h_move,
        )
        candidate = self._clip_to_bounds(candidate)

        self.current_position = candidate
        done = self._is_terminal(candidate)
        return -1.0, candidate, done


class GridMazeEnv(GridEnvBase):
    """Simple maze environment driven by a numpy grid.

    The maze grid should include exactly one start marker and one goal marker.
    By default the encoding is:
      0 = free tile
      1 = wall / obstacle
      2 = start
      3 = goal
    """

    def __init__(
            self,
            grid: np.ndarray,
            free_value: int = 0,
            wall_value: int = 1,
            start_value: int = 2,
            goal_value: int = 3,
            action_set: Optional[Sequence[Action]] = None,
            step_reward: float = 0.0,
            goal_reward: float = 1.0,
    ):
        self.grid = np.asarray(grid, dtype=int)
        self.free_value = free_value
        self.wall_value = wall_value
        self.start_value = start_value
        self.goal_value = goal_value
        self.step_reward = step_reward
        self.goal_reward = goal_reward

        start_position = self._find_unique_marker(self.start_value, "start")
        goal_position = self._find_unique_marker(self.goal_value, "goal")

        super().__init__((self.grid.shape[0], self.grid.shape[1]), start_position, goal_position, action_set)

    def __repr__(self):
        return str(self.grid).replace(
            str(self.free_value), "□").replace(
            str(self.wall_value), "■").replace(
            str(self.start_value), "S").replace(
            str(self.goal_value), "G")

    def _find_unique_marker(self, marker_value: int, label: str) -> State:
        positions = np.argwhere(self.grid == marker_value)
        if positions.shape[0] != 1:
            raise ValueError(
                f"Grid must contain exactly one {label} marker value {marker_value}."
            )
        position = positions[0]
        return int(position[0]), int(position[1])

    def _is_blocked(self, position: State) -> bool:
        return self.grid[position[0], position[1]] == self.wall_value

    def _get_reward(self, position: State, previous_position: State, action: ActionInput) -> float:
        if self._is_terminal(position):
            return self.goal_reward
        return self.step_reward