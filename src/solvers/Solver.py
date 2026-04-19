from typing import Any
from abc import ABC, abstractmethod

import gymnasium as gym


class Solver(ABC):
    """
    Abstract base class for Solver / Trainer within the RL framework.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train_one_epoch(**kwargs) -> float:
        """
        Train a single epoch.

        Returns:
            - float: Reward obtained during the epoch.
        """
        pass

    @abstractmethod
    def train(**kwargs) -> Any:
        """
        Perform an entire training session.

        Returns:
            - Any: Training results / history.
        """
        pass

    @abstractmethod
    def plot_train_history() -> None:
        """
        Plot the training history.
        """
        pass