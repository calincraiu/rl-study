from typing import Any, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from src.agents.Agent import Agent
from src.utility.ReplayBuffers import ReplayBuffer
from src.nn.QNetwork import QNetwork


class BasicDQNAgent(Agent):
    """
    Deep Q-Network with experience replay + target network.
    Works for both Discrete and Box observation spaces.
    """

    target_update_freq: int
    """
    Steps between target network updates. Recommended smaller for simpler problems / smaller environments and larger for more complex ones.
    """

    train_every_n: int
    """
    Number of percept calls between each Q network training. Recommended smaller for simpler problems / smaller environments and larger for more complex ones.
    """

    batch_size: int
    """
    Number of experiences to use in a batch for training the Q network during experience replay.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        replay_buffer: ReplayBuffer,
        hidden_dim: int = 128,
        alpha: float = 1e-3, # learning rate
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_floor: float = 0.1, # The minimum value of epsilon during training
        xi: float = 0.995, # epsilon decay
        batch_size: int = 64,
        target_update_freq: int = 100,
        train_every_n: int = 4,
        name: Optional[str] = None,
    ):
        super().__init__(name=name if name else type(self).__name__)

        # --- Device ---
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # --- Spaces ---

        # Observation space
        if isinstance(observation_space, gym.spaces.Discrete):
            self.state_dim: int = int(observation_space.n)

            def _state_to_tensor(s: Any) -> torch.Tensor:
                x = torch.tensor(s, dtype=torch.long, device=device)
                if x.ndim == 0:
                    x = x.unsqueeze(0)
                return torch.nn.functional.one_hot(x, num_classes=self.state_dim).float().to(device)

            self._state_to_tensor = _state_to_tensor
        elif isinstance(observation_space, gym.spaces.Box):
            self.state_dim: int = int(np.prod(observation_space.shape))
            state_shape = observation_space.shape

            def _state_to_tensor(s: Any) -> torch.Tensor:
                x = torch.as_tensor(s, dtype=torch.float32, device=device)
                if x.ndim == len(state_shape):
                    return x.flatten()
                if x.ndim == len(state_shape) + 1 and tuple(x.shape[1:]) == tuple(state_shape):
                    return x.reshape(x.shape[0], -1)
                raise ValueError(
                    f"Unsupported state tensor shape {tuple(x.shape)} for observation shape {tuple(state_shape)}"
                )

            self._state_to_tensor = _state_to_tensor
        else:
            raise ValueError(f"Unsupported observation space: {type(observation_space)}")

        # Action space
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("DQN currently supports only Discrete action spaces")
        self.action_dim: int = int(action_space.n)

        # --- Networks ---
        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Load the q_net into the target_net (using 2 nets for stability)
        self.target_net.eval() # Freeze target net
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)

        # --- Hyperparameters ---
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_floor = epsilon_floor
        self.xi = xi
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_every_n = train_every_n

        # --- Replay buffer ---
        self.buffer = replay_buffer

        # --- Counters & Trackers ---
        self.latest_loss: float = 0.0
        self.latest_max_q: float = 0.0
        self.latest_mean_q: float = 0.0
        self.latest_q_std: float = 0.0
        self.latest_td_error: float = 0.0
        self.grad_norm_unclipped: float = 0.0
        self.grad_norm_clipped: float = 0.0

    def _get_q_values(self, s: Any) -> torch.Tensor:
        """
        Return Q(s, ·) as a tensor (shape: [1, action_dim]).
        """
        state_tensor = self._state_to_tensor(s)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)  # add batch dim for a single state
        return self.q_net(state_tensor)

    def actuate(self, s: Any) -> int:
        """
        ε-greedy using the current Q-network.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # explore

        with torch.no_grad(): # exploit
            q_values = self._get_q_values(s)
        return int(q_values.argmax().item())

    def percept(self, s: Any, a: int, s_prime: Any, r: float, done: bool = False) -> None:
        """
        Store transition in replay buffer.
        """

        # --- Push to buffer ---
        self.buffer.push(s=s, a=a, s_prime=s_prime, r=r, done=done)
        
        # --- Training ---
        # Train every n steps (once the buffer has enough elements for a batch)
        if len(self.buffer) >= self.batch_size and self.steps_done % self.train_every_n == 0:
            self._train_step()

        # --- Updating target ---
        # Periodically copy Q-network -> target network. Frequency based on num steps done
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # --- Housekeeping ---
        self.steps_done += 1

    def _train_step(self) -> None:

        # --- Ensure training condition ---
        if len(self.buffer) < self.batch_size:
            return # Don't train until we have enough data

        # --- Sample batch ---
        s, a, r, s_prime, done = self.buffer.sample(self.batch_size)

        # --- Convert to tensors (vectorized) ---
        device = next(self.q_net.parameters()).device

        s = self._state_to_tensor(s)
        s_prime = self._state_to_tensor(s_prime)
        a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)

        # --- Current Q-values ---
        q_values: torch.Tensor = self.q_net(s).gather(1, a).squeeze(1)

        # --- Target Q-values ---
        with torch.no_grad():
            next_q = self.target_net(s_prime).max(1)[0]
            target = r.squeeze(1) + self.gamma * next_q * (1 - done.squeeze(1))

        # --- Loss (Huber is more stable than MSE) ---
        loss = nn.functional.smooth_l1_loss(q_values, target, reduction='none')
        loss_mean = loss.mean()

        # --- Optimize ---
        self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
        loss_mean.backward()
        self.grad_norm_unclipped = self.__get_network_gradient_norm() # Tracking
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0) # Gradient clipping (stabilizes training)
        self.grad_norm_clipped = self.__get_network_gradient_norm() # Tracking
        self.optimizer.step()

        # --- Track metrics ---
        self.latest_loss = loss_mean.item()
        self.latest_max_q = q_values.max().item()
        self.latest_mean_q = q_values.mean().item()
        self.latest_q_std = q_values.std().item()
        self.latest_td_error = (q_values - target).abs().mean().item()

    def update_episode(self, **kwargs: Any) -> None:
        """
        Called at the end of each episode.
        """
        # Decay exploration
        self.epsilon = max(self.epsilon_floor, self.epsilon * self.xi)

        # Train on a batch (you can call this more frequently if you prefer)
        self._train_step()

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
            "train/mean_q": self.latest_mean_q,
            "train/q_std": self.latest_q_std,
            "train/td_error": self.latest_td_error,
            "train/grad_norm_unclipped": self.grad_norm_unclipped,
            "train/grad_norm_clipped": self.grad_norm_clipped,
            "train/param_norm": self.__get_network_parameter_norm(),
        }
    
    def __get_network_gradient_norm(self) -> float:
        """
        Get the gradient norm of the network. Used for tracking vanishing/exploding gradients.
        """
        grad_norms = [
            p.grad.norm(2)
            for p in self.q_net.parameters()
            if p.grad is not None
        ]
        if not grad_norms:
            return 0.0
        total_grad_norm = torch.norm(torch.stack(grad_norms), 2).item()
        return total_grad_norm
    
    def __get_network_parameter_norm(self) -> float:
        """
        Gets the parameter norm of the network. Used to track runaway weights.
        """
        param_norms = [p.data.norm(2) for p in self.q_net.parameters()]
        if not param_norms:
            return 0.0
        
        total_param_norm = torch.norm(torch.stack(param_norms), 2).item()
        return total_param_norm
