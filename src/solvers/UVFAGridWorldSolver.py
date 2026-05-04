from typing import Any

from tqdm import tqdm
import gymnasium as gym
import wandb

from src.agents.Agent import Agent
from src.solvers.Solver import Solver
from src.environments.Gridworld import Cells


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
        # Init wandb for logging
        wandb.init(
            project=type(self).__name__,
            config={
                "n_epochs": n_epochs,
                "agent_type": self.agent.__class__.__name__,
                **self.__agent_kwargs
            }
        )

        # Iterate epochs
        for i in tqdm(range(n_epochs), desc="Epoch", disable=(not self.verbose)):

            # --- Environment Reset ---
            obs, info = self.environment.reset(options=self.__environment_options)
            done: bool = False
            episode_success: int = 0
            
            # --- Iterate steps in epoch ---
            while not done:
                action = self.agent.actuate(obs, g=info.get("goal"))
                next_obs, reward, terminated, truncated, info = self.environment.step(action)

                self.agent.percept(
                    s=obs, 
                    a=int(action), 
                    s_prime=next_obs, 
                    r=float(reward),
                    g=info.get("goal"),
                    done=bool(terminated)
                )
                obs = next_obs # Update state
                if terminated or truncated:
                    done = True
                    episode_success = int(reward == getattr(self.environment.unwrapped, "rewards")[Cells.TARGET.value])

                    # Log Environment Metrics (Automatically computed by Gymnasium Wrapper)
                    if "episode" in info:
                        wandb.log({
                            "env/episode_return": info["episode"]["r"],
                            "env/episode_length": info["episode"]["l"],
                            "env/success": episode_success,
                        }, step=self.agent.steps_done)

            # Apply intra-episode update actions (like epsilon decay)
            self.agent.update_episode()

            # Log Agent/Training Metrics at the end of every episode
            metrics = self.agent.get_metrics()
            metrics["epoch"] = i # Add solver-level info
            wandb.log(metrics, step=self.agent.steps_done)

        # Close the wandb run
        wandb.finish()