from typing import Any, Optional, Tuple
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from src.agents.Agent import Agent


class ReplayBuffer:
    """Simple circular buffer for experience replay."""
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s_prime: np.ndarray, done: bool):
        self.buffer.append((s, a, r, s_prime, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, done = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(s_prime), np.array(done))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Simple MLP that maps state -> Q-values for each action.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(Agent):
    """
    Deep Q-Network with experience replay + target network.
    Compatible with your existing Agent interface.
    Works for both Discrete and Box observation spaces.
    """

    r_scaling: float
    """
    Reward scaling ( r = r * r_scaling ). Scaling rewards might help depending on whether they are sparse or dense.
    """

    target_update_freq: int
    """
    Steps between target network updates. Recommended smaller for simpler problems / smaller environments and larger for more complex ones.
    """

    train_every_n: int
    """
    Number of percept calls between each Q network training. Recommended smaller for simpler problems / smaller environments and larger for more complex ones.
    """

    batch_size: int = 64
    """
    Number of experiences to use in a batch for training the Q network during experience replay.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        hidden_dim: int = 128,
        alpha: float = 1e-3, # learning rate
        gamma: float = 0.99,
        epsilon: float = 1.0,
        xi: float = 0.995, # epsilon decay
        warmup_episodes: int = 100, # warmup counter - number of episodes before xi gets enabled to start decaying epsilon (exporation threshold)
        r_scaling: float = 1.0, # reward scaling
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        train_every_n: int = 4,
        name: Optional[str] = None,
    ):
        super().__init__(name=name if name else type(self).__name__)

        # --- Spaces ---
        if isinstance(observation_space, gym.spaces.Discrete):
            self.state_dim: int = int(observation_space.n)
            self._state_to_tensor = lambda s: torch.nn.functional.one_hot(
                torch.tensor(s, dtype=torch.long), num_classes=self.state_dim
            ).float().to(device)
        elif isinstance(observation_space, gym.spaces.Box):
            self.state_dim: int = int(np.prod(observation_space.shape))
            self._state_to_tensor = lambda s: torch.tensor(s, dtype=torch.float32).flatten().to(device)
        else:
            raise ValueError(f"Unsupported observation space: {type(observation_space)}")

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("DQN currently supports only Discrete action spaces")
        self.action_dim: int = int(action_space.n)

        # --- Networks ---
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Load the q_net into the target_net (using 2 nets for stability)
        self.target_net.eval() # Freeze target net

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)

        # --- Hyperparameters ---
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi
        self.r_scaling = r_scaling
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_every_n = train_every_n
        self.warmup_episodes = warmup_episodes

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(buffer_capacity)

        # --- Counters & Trackers ---
        self.latest_loss: float = 0.0
        self.latest_max_q: float = 0.0
        # steps_done comes from base class
        # episode_count comes from base class

    def _get_q_values(self, s: Any) -> torch.Tensor:
        """
        Return Q(s, ·) as a tensor (shape: [1, action_dim]).
        """
        state_tensor = self._state_to_tensor(s).unsqueeze(0)  # add batch dim
        return self.q_net(state_tensor)

    def actuate(self, s: Any) -> int:
        """
        ε-greedy using the current Q-network.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # explore

        with torch.no_grad():
            q_values = self._get_q_values(s)
        return int(q_values.argmax().item())

    def percept(self, s: Any, a: int, s_prime: Any, r: float, done: bool = False) -> None:
        """
        Store transition in replay buffer.
        """
        self.buffer.push(s, a, r, s_prime, done)
        self.steps_done += 1
        
        # Train every n steps (once the buffer has enough elements for a batch)
        if len(self.buffer) >= self.batch_size and self.steps_done % self.train_every_n == 0:
            self._train_step()

    def _train_step(self) -> None:
        if len(self.buffer) < self.batch_size:
            return

        # --- Sample batch ---
        s, a, r, s_prime, done = self.buffer.sample(self.batch_size)

        # --- Reward scaling ---
        r *= self.r_scaling

        # --- Convert to tensors (vectorized) ---
        device = next(self.q_net.parameters()).device

        if isinstance(s[0], (int, np.integer)):  # Discrete state space
            s = torch.nn.functional.one_hot(
                torch.tensor(s, dtype=torch.long, device=device),
                num_classes=self.state_dim
            ).float()

            s_prime = torch.nn.functional.one_hot(
                torch.tensor(s_prime, dtype=torch.long, device=device),
                num_classes=self.state_dim
            ).float()
        else:  # Box space
            s = torch.tensor(s, dtype=torch.float32, device=device)
            s_prime = torch.tensor(s_prime, dtype=torch.float32, device=device)

        a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)

        # --- Current Q-values ---
        q_values: torch.Tensor = self.q_net(s).gather(1, a)

        # --- Target Q-values ---
        with torch.no_grad():
            next_q = self.target_net(s_prime).max(1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - done)

        # --- Loss (Huber is more stable than MSE) ---
        loss = nn.functional.smooth_l1_loss(q_values, target)

        # --- Optimize ---
        self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0) # Gradient clipping (stabilizes training)
        self.optimizer.step()

        # --- Track metrics ---
        self.latest_loss = loss.item()
        self.latest_max_q = q_values.max().item()

    def update_episode(self, **kwargs: Any) -> None:
        """
        Called at the end of each episode.
        """
        # Decay exploration
        decay_factor = self.xi if self.episode_count >= self.warmup_episodes else 1.0 # warmup
        self.epsilon = max(0.01, self.epsilon * decay_factor)

        # Train on a batch (you can call this more frequently if you prefer)
        self._train_step()

        # Periodically copy Q-network -> target network. Frequency based on num steps done
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.episode_count += 1

    def get_policy(self) -> nn.Module:
        """
        Get the agent's policy.
        """
        return self.q_net

    def get_metrics(self) -> dict:
        return {
            "train/epsilon": self.epsilon,
            "train/loss": self.latest_loss,
            "train/max_q": self.latest_max_q,
        }
