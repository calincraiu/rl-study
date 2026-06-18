import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


class RandomWalkEnv:

    def __init__(self, 
            n_states: int,
            ):
        self.n_states = n_states # including terminal states
        self.s = self.n_states // 2 # start at middle

    def reset(self) -> int:
        self.s = self.n_states // 2
        return self.s

    def step(self, a: int) -> tuple[float, int, bool]:
        reward = 0
        s_prime = self.s + a
        if s_prime == 0: # terminal state on the left
            reward = -1
        if s_prime == self.n_states - 1: # terminal state on the right
            reward = 1
        self.s = s_prime
        finished = (s_prime == self.n_states - 1) or (s_prime == 0)
        return reward, s_prime, finished
    

class WindyGridWorldEnv:

    def __init__(self, 
            grid_shape: tuple[int, int], 
            goal_position: tuple[int, int],
            start_position: tuple[int, int],
            wind_per_column: list[int],
            stochastic_wind: bool
            ):
        self.num_rows, self.num_cols = grid_shape[0], grid_shape[1]
        self.start_position = start_position
        self.goal_position = goal_position
        self.wind_per_column = wind_per_column
        self.stochastic_wind = stochastic_wind

        self.s = self.start_position

    def reset(self) -> tuple[int, int]:
        self.s = self.start_position
        return self.s

    def _apply_stochastic_wind(self, col: int):
        wind = self.wind_per_column[col]
        p = np.random.random()
        if p <= 1/3:
            wind -= 1
        elif p >= 2/3:
            wind += 1
        return -wind # negative - moving 'up' is a decrease in row

    def step(self, a: tuple[int, int]) -> tuple[float, tuple[int, int], bool]:

        reward = -1.0

        old_row = int(self.s[0])
        old_col = int(self.s[1])
        wind = self._apply_stochastic_wind(old_col) if self.stochastic_wind else -self.wind_per_column[old_col]

        h_move = a[1]
        v_move = a[0] + wind

        new_row = max(min(old_row + v_move, self.num_rows - 1), 0)
        new_col = max(min(old_col + h_move, self.num_cols - 1), 0)

        s_prime = (new_row, new_col)
        self.s = s_prime

        finished = s_prime == self.goal_position
        return reward, s_prime, finished
    
    def plot_training_metrics(
        self,
        episode_lengths,
        episode_returns,
        smoothing_window=10
    ):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Episode length
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

        # Returns
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
        plt.title("Windy Gridworld Learning Curve")
        plt.grid(True)
        plt.show()

    def plot_policy(self, Q_pi, action_set):

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.dpi = 120

        # ------------------------------------------------------------------
        # Grid geometry
        # ------------------------------------------------------------------

        ax.set_xlim(-0.5, self.num_cols - 0.5)
        ax.set_ylim(self.num_rows - 0.5, -0.5)

        ax.set_aspect("equal")

        # Major ticks = cell centers
        ax.set_xticks(np.arange(self.num_cols))
        ax.set_yticks(np.arange(self.num_rows))

        # Minor ticks = cell boundaries
        ax.set_xticks(np.arange(-0.5, self.num_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.num_rows, 1), minor=True)

        ax.grid(which="minor", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # ------------------------------------------------------------------
        # Build quiver vectors
        # ------------------------------------------------------------------

        X = []
        Y = []
        U = []
        V = []

        for row in range(self.num_rows):
            for col in range(self.num_cols):

                s = (row, col)

                if s == self.goal_position:
                    continue

                best_action_idx = np.argmax(Q_pi[s])
                d_row, d_col = action_set[best_action_idx]

                X.append(col)
                Y.append(row)
                U.append(d_col)
                V.append(d_row)

        ax.quiver(
            X,
            Y,
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=3.0,
            pivot="middle"
        )

        # ------------------------------------------------------------------
        # Start / Goal markers
        # ------------------------------------------------------------------

        start_row, start_col = self.start_position
        goal_row, goal_col = self.goal_position

        ax.scatter(
            start_col,
            start_row,
            marker="s",
            s=250,
            label="Start"
        )

        ax.scatter(
            goal_col,
            goal_row,
            marker="*",
            s=350,
            label="Goal"
        )

        # ------------------------------------------------------------------
        # Wind annotation
        # ------------------------------------------------------------------

        for col, wind in enumerate(self.wind_per_column):

            ax.text(
                col,
                -0.8,
                f"{wind}",
                ha="center",
                va="center",
                fontsize=10
            )

        ax.text(
            self.num_cols / 2,
            -1.3,
            "Wind Strength",
            ha="center",
            fontsize=12
        )

        # ------------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------------

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Greedy Policy Learned by SARSA")

        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def animate_episode_with_trail(
        self,
        Q_pi,
        action_set,
        max_steps=500,
        interval=150,
    ):

        # ----------------------------------------------------------
        # Generate greedy rollout
        # ----------------------------------------------------------

        states = []

        s = self.reset()
        states.append(s)

        for _ in range(max_steps):

            action_idx = np.argmax(Q_pi[s])

            _, s_prime, done = self.step(
                action_set[action_idx]
            )

            states.append(s_prime)

            s = s_prime

            if done:
                break

        # ----------------------------------------------------------
        # Create figure
        # ----------------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlim(-0.5, self.num_cols - 0.5)
        ax.set_ylim(self.num_rows - 0.5, -0.5)

        ax.set_aspect("equal")

        # Cell centers
        ax.set_xticks(np.arange(self.num_cols))
        ax.set_yticks(np.arange(self.num_rows))

        # Cell boundaries
        ax.set_xticks(
            np.arange(-0.5, self.num_cols, 1),
            minor=True
        )

        ax.set_yticks(
            np.arange(-0.5, self.num_rows, 1),
            minor=True
        )

        ax.grid(which="minor")
        ax.tick_params(which="minor", bottom=False, left=False)

        # ----------------------------------------------------------
        # Wind labels
        # ----------------------------------------------------------

        for col, wind in enumerate(self.wind_per_column):

            ax.text(
                col,
                -0.8,
                str(wind),
                ha="center",
                va="center",
                fontsize=10,
            )

        ax.text(
            self.num_cols / 2,
            -1.3,
            "Wind Strength",
            ha="center",
        )

        # ----------------------------------------------------------
        # Start / Goal
        # ----------------------------------------------------------

        start_row, start_col = self.start_position
        goal_row, goal_col = self.goal_position

        ax.scatter(
            start_col,
            start_row,
            marker="s",
            s=250,
            label="Start",
        )

        ax.scatter(
            goal_col,
            goal_row,
            marker="*",
            s=350,
            label="Goal",
        )

        # ----------------------------------------------------------
        # Animated objects
        # ----------------------------------------------------------

        trail, = ax.plot(
            [],
            [],
            linewidth=2,
        )

        agent = ax.scatter(
            [],
            [],
            s=150,
        )

        title = ax.set_title("Step 0")

        # ----------------------------------------------------------
        # Animation callback
        # ----------------------------------------------------------

        def update(frame):

            rows = [
                state[0]
                for state in states[:frame + 1]
            ]

            cols = [
                state[1]
                for state in states[:frame + 1]
            ]

            trail.set_data(cols, rows)

            agent.set_offsets(
                [[cols[-1], rows[-1]]]
            )

            title.set_text(
                f"Step {frame}"
            )

            return trail, agent

        # ----------------------------------------------------------
        # Animate
        # ----------------------------------------------------------

        anim = FuncAnimation(
            fig,
            update,
            frames=len(states),
            interval=interval,
            blit=True,
            repeat=False,
        )

        display(
            HTML(
                anim.to_jshtml()
            )
        )

        plt.close(fig)

        return anim
