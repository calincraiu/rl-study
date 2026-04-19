from typing import Any
from abc import ABC, abstractmethod

import gymnasium as gym


class Solver(ABC):
    """
    Abstract base class for Solver / Trainer within the RL framework.
    """

    name: str
    """Name of the solver."""

    def __init__(self, **kwargs):
        self.name=kwargs.get("name", "Solver")

    @abstractmethod
    def train(self, **kwargs) -> Any:
        """
        Perform an entire training session.

        Returns:
            - Any: Training results / history.
        """
        pass