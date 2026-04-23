from typing import Any
from abc import ABC, abstractmethod

import gymnasium as gym


class Agent(ABC):
    """
    Abstract base class for Agent / Learner within the RL framework.
    """
    
    name: str
    """Name of the agent."""

    steps_done: int
    """Steps performed by the agent."""

    episode_count: int
    """Number of episodes experienced by the agent."""

    def __init__(self, **kwargs):
        self.name=kwargs.get("name", "Agent")
        self.steps_done = 0
        self.episode_count = 0

    @abstractmethod
    def percept(self, s: Any, a: Any, s_prime: Any, r: Any, **kwargs) -> None:
        """
        Percept base method. Used for updating agent state (e.g. value function) during training.

        Args:
            - s (Any) : Previous state / observation
            - a (Any) : Action that the agent took
            - s_prime (Any) : New state / observation
            - r (Any) : Reward obtained from the environment
            - kwargs : Any other arguments.
        """
        pass

    @abstractmethod
    def actuate(self, s: Any) -> Any:
        """
        Actuate base method. Used for choosing actions based on a state / observation.

        Args:
            - s (Any) : Previous state / observation

        Returns:
            - Any: chosen action.
        """
        pass

    @abstractmethod
    def update_episode(self, **kwargs: Any) -> None:
        """
        Used for updating the agent between episodes during training.
        """
        pass

    @abstractmethod
    def get_policy(self) -> Any:
        """
        Used for getting the agent's policy for external solvers. This can serve as a best-policy retirever.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Returns a dictionary of internal metrics for logging.
        Override this in subclasses to log specific data.
        """
        return {}