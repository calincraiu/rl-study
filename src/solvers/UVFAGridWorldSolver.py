import typing
from typing import Any

from tqdm import tqdm
import gymnasium as gym
import wandb

from src.agents.Agent import Agent
from src.solvers.Solver import Solver
from src.environments.Gridworld import Cells, GridWorldEnv


def _apply_observation_wrappers(env: gym.Env, observation: Any) -> Any:
    """
    Apply every ObservationWrapper in the same order Gymnasium does.
    This is important for transforming goals exactly the same way as observations.
    """
    if isinstance(env, gym.ObservationWrapper):
        observation = _apply_observation_wrappers(env.env, observation)
        return env.observation(observation)
    if isinstance(env, gym.Wrapper):
        return _apply_observation_wrappers(env.env, observation)
    return observation


class UVFAGridWorldSolver(Solver):

    __environment_options: dict = {
        "goal_placement_strategy" : "random"
    }
    """
    These are the options passed as additional configuration to the environment reset function.
    """

    def __init__(
            self, 
            environment: gym.Env, 
            agent_class: type[Agent],
            agent_kwargs: dict[str, Any],
            verbose: bool = False,
            **kwargs
            ):
        super().__init__(name=type(self).__name__, **kwargs)
        
        self.verbose = verbose
        self.environment = environment
        self.agent = agent_class(
            observation_space=environment.observation_space, 
            action_space=environment.action_space, 
            **agent_kwargs
            )
        self.__agent_kwargs = agent_kwargs # Used for logging

    def train(self, n_epochs: int) -> None:
        wandb.init(
            project=type(self).__name__,
            config={
                "n_epochs": n_epochs,
                "agent_type": self.agent.__class__.__name__,
                **self.__agent_kwargs,
            },
        )

        for i in tqdm(range(n_epochs), desc="Epoch", disable=(not self.verbose)):
            obs, info = self.environment.reset(options=self.__environment_options)

            # Transform the raw goal through the exact same wrapper stack as the observation.
            raw_goal = info["goal"]
            goal = _apply_observation_wrappers(
                self.environment,
                typing.cast(GridWorldEnv, self.environment.unwrapped).build_observation(raw_goal),
            )

            done = False
            episode_success = 0

            while not done:
                # IMPORTANT: use the transformed goal, not the raw goal.
                action = self.agent.actuate(obs, g=goal)

                next_obs, reward, terminated, truncated, info = self.environment.step(action)

                self.agent.percept(
                    s=obs,
                    a=int(action),
                    s_prime=next_obs,
                    r=float(reward),
                    g=goal,
                    done=bool(terminated or truncated),
                )

                obs = next_obs

                if terminated or truncated:
                    done = True
                    episode_success = int(
                        reward == getattr(self.environment.unwrapped, "rewards")[Cells.TARGET.value]
                    )

                    if "episode" in info:
                        wandb.log(
                            {
                                "env/episode_return": info["episode"]["r"],
                                "env/episode_length": info["episode"]["l"],
                                "env/success": episode_success,
                            },
                            step=self.agent.steps_done,
                        )

            self.agent.update_episode()

            metrics = self.agent.get_metrics()
            metrics["epoch"] = i
            wandb.log(metrics, step=self.agent.steps_done)

        wandb.finish()