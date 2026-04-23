from typing import Any, Optional

from tqdm import tqdm
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import wandb

from src.agents.Agent import Agent
from src.solvers.Solver import Solver
from src.environments.Gridworld import GridWorldEnv


class GridWorldSolver(Solver):

    __environment_options: dict = {
        "agent_start_position_strategy" : "curriculum",
        "position_strategy_params" : {
            "curriculum_distance" : 1,
            "curriculum_threshold" : 0.8
        }
    }
    """
    These are the options passed as additional configuration to the environment reset function.
    This can contain information about the type of strategy to be used for positioning the agent (random, curriculum, fixed),
    and additional parameters.
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
        # Init wandb for logging
        wandb.init(
            project=type(self).__name__,
            config={
                "n_epochs": n_epochs,
                "agent_type": self.agent.__class__.__name__,
                **self.__agent_kwargs
            }
        )
        env_max_distance_from_target: int = 1
        if hasattr(self.environment.unwrapped, "max_distance_from_target"):
            env_max_distance_from_target: int = getattr(self.environment.unwrapped, "max_distance_from_target")
        else:
            raise Exception("The GridWorld environment should have a max_distance_from_target attribute!")

        # Iterate epochs
        for i in tqdm(range(n_epochs), desc="Epoch", disable=(not self.verbose)):
            
            # --- Agent Placement ---
            # Create a curriculum distance - a random placement of the agent within a certain maximum distance of the target
            # This starts very close to the target at early epochs and gradually increases. This is meant to show the agent
            # simple cases first, and gradually increase difficulty. This is meant to help provide a positive reward signal
            # early on in the training - complete random initial positioning might lead to the agent never (or very late in
            # the epochs) encounter the target.
            curriculum_distance = max(1, int(env_max_distance_from_target * (i / n_epochs))) # minimum 1 - never start on the actual target
            self.__environment_options["position_strategy_params"]["curriculum_distance"] = curriculum_distance

            # --- Environment Reset ---
            obs, info = self.environment.reset(options=self.__environment_options)
            done = False
            
            # --- Iterate steps in epoch ---
            while not done:
                action = self.agent.actuate(obs)
                next_obs, reward, terminated, truncated, info = self.environment.step(action)

                self.agent.percept(
                    s=obs, 
                    a=int(action), 
                    s_prime=next_obs, 
                    r=float(reward),
                    done=bool(terminated or truncated)
                )
                obs = next_obs # Update state
                if terminated or truncated:
                    done = True

                    # Log Environment Metrics (Automatically computed by Gymnasium Wrapper)
                    if "episode" in info:
                        wandb.log({
                            "env/episode_return": info["episode"]["r"],
                            "env/episode_length": info["episode"]["l"],
                            # Success metric: Assuming 1.0 is the reward for the target cell
                            "env/success": 1 if reward == 1.0 else 0 
                        }, step=self.agent.steps_done)

            # Apply intra-episode update actions (like epsilon decay)
            self.agent.update_episode()

            # Log Agent/Training Metrics at the end of every episode
            metrics = self.agent.get_metrics()
            metrics["epoch"] = i # Add solver-level info
            wandb.log(metrics, step=self.agent.steps_done)

        # Close the wandb run
        wandb.finish()