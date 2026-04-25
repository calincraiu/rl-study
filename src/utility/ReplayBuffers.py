from abc import ABC, abstractmethod
import random
from collections import deque
from typing import Tuple, Any

import torch
import numpy as np


class ReplayBuffer(ABC):
    """Abstract base class for replay buffers."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def push(self, *args, **kwargs) -> None:
        """
        Push base method. Used for pushing transitions into the replay buffer. Typically includes
        state, action, reward, next state and a termination flag.
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: int, *args, **kwargs) -> Any:
        """
        Sample base method. Used for retrieving transitions from the replay buffer.

        Args:
            batch_size (int): number of samples to retrieve.

        Returns:
            Any: batch of transition replay samples.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of elements present in the replay buffer.
        """
        pass


class DequeReplayBuffer(ReplayBuffer):
    """
    Simple circular buffer for experience replay.
    """
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


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full)
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.ptr = 0

    def push(self, transition, priority):
        # New transitions start with maximum priority to ensure they are seen at least once
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = transition
        self.update(idx, priority)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        # Force priority to be a scalar float. 
        # This prevents "sequence" errors if it's a 1-element array/tensor.
        priority = float(priority) 
        
        p = priority ** self.alpha
        change = p - self.tree[idx]
        self.tree[idx] = p
        
        # Update the tree sums
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list[int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], torch.Tensor]:
        """
        Samples a batch of experiences based on priorities.
        Returns: (indices, transitions, importance_sampling_weights)
        """
        batch_indices = []
        batch_priorities = []
        transitions = []
        
        # Divide the total priority sum into 'batch_size' equal segments
        segment = self.tree[0] / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # Pick a random value within this segment
            s = random.uniform(a, b)
            
            # Find the index in the tree corresponding to this priority value
            idx, p, data = self._get_index(s)
            
            batch_indices.append(idx)
            batch_priorities.append(p)
            transitions.append(data)

        # --- Importance Sampling Weights ---
        # Probability P(i) = priority_i / sum_priorities
        probs = np.array(batch_priorities) / self.tree[0]
        
        # Weight w_i = (1/N * 1/P(i))^beta
        weights = (self.n_entries * probs) ** -beta
        
        # Normalize weights so the max weight is 1.0 (for stability)
        weights /= weights.max()
        
        # Unzip transitions (s, a, r, s_prime, done)
        s, a, r, s_prime, done = zip(*transitions)
        
        return (
            batch_indices, 
            (np.array(s), np.array(a), np.array(r), np.array(s_prime), np.array(done)), 
            torch.FloatTensor(weights)
        )

    def _get_index(self, v):
        """
        Helper to find a leaf node index given a priority value v.
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we've reached the bottom of the tree (leaf nodes)
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Decide to go left or right
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def __len__(self):
        """Allows calling len(buffer_instance)"""
        return self.n_entries