import time
import argparse
import yaml
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation, HumanRendering, RecordEpisodeStatistics, RecordVideo

from src.environments.Gridworld import GridWorldEnv, DiscreteGridWorldWrapper, NormalizedCoordWrapper, PatienceWrapper, Cells

from src.agents.QLambdaAgent import QLambdaAgent
from src.agents.DQNAgent import DQNAgent
from src.agents.BasicDQNAgent import BasicDQNAgent

from src.solvers.GridWorldSolver import GridWorldSolver
from src.solvers.BasicGridWorldSolver import BasicGridWorldSolver

from src.utility.ReplayBuffers import DequeReplayBuffer, PrioritizedReplayBuffer
from src.utility.SessionRunner import play_game
from src.utility.General import find_repo_root


# --- Setup ---

REPO_ROOT = find_repo_root(Path(__file__).resolve())

# --- Grid Templates ---

medium_grid = np.array((
    [0,0,0,0,0,2],
    [0,3,3,3,0,0],
    [0,0,0,0,0,0],
    [0,2,3,0,3,0],
    [0,0,3,1,3,0],
    [0,0,0,0,0,0],
))


huge_grid = np.array((
    [0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,3,0,0,0,0],
    [0,0,0,0,0,0,3,0,3,3,3,3,3,0,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,0],
    [0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,3,0,3,0,3,0,0],
    [0,0,0,0,0,0,3,0,0,0,0,0,3,0,0,3,3,3,0,0,0,0,0,0,0,3,3,3,3,3,0,0],
    [0,0,0,0,0,0,3,3,3,3,3,0,3,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,3,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,3,3,3,3,3,0,0,0,0,0,0,0,3,0,0],
    [0,0,0,0,0,0,0,3,3,3,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0],
    [0,0,0,0,0,0,0,3,0,3,0,0,3,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0],
    [0,0,0,0,0,0,0,3,0,3,3,3,3,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0],
    [0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0],
    [0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,0,0,3,3,3,3,3,0,0,0,0,0,0,3,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,3,3,3,3,3,0,0,0,3,3,3,3,0,0],
    [3,3,3,3,0,3,3,3,3,3,3,3,3,3,0,3,0,0,3,3,3,3,3,0,0,0,3,0,0,0,0,0],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,2,2,3,3,3,3,3,0,0,0,3,0,0,0,0,0],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0,0,3,3,3,3,3,0,0,0,3,0,0,1,0,0],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,3,3,3,3,3,0,0,0,3,0,0,0,0,0],
    [3,3,0,3,3,0,3,3,3,0,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,3,3,3,3,3,0],
    [0,0,0,0,0,0,0,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,2,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0]
))

GRIDS = {
    "medium": medium_grid,
    "huge": huge_grid
}

# --- Logic ---

def load_config(path: str = f"{Path(__file__).resolve().parent.as_posix()}/config.yaml") -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def run_basic(
        config: dict, 
        grid: np.ndarray, 
        video_output_dir: Path = REPO_ROOT / "runtime" / "videos"
        ):
    
    # --- Config Parsing ---
    video_output_dir = video_output_dir / f"basic_trial_{time.strftime('%X_%x').replace(':', '').replace('/', '')}"

    tile_reward = config["rewards"]["TILE"]
    target_reward = config["rewards"]["TARGET"]
    pitfall_reward = config["rewards"]["PITFALL"]
    wall_reward = config["rewards"]["WALL"]

    env_wind_p = config["env"]["wind_p"]
    env_step_limit = config["env"]["step_limit"]

    replay_buffer_capacity = config["replay_buffer"]["capacity"]
    logging_record_every = config["logging"]["record_every"]

    agent_gamma = config["agent"]["gamma"]
    agent_alpha = config["agent"]["alpha"]
    agent_epsilon = config["agent"]["epsilon"]
    agent_xi = config["agent"]["xi"]

    agent_nn_hidden_dim = config["agent_nn"]["hidden_dim"]
    agent_nn_batch_size = config["agent_nn"]["batch_size"]
    agent_nn_target_update_freq = config["agent_nn"]["target_update_freq"]
    agent_nn_train_every_n = config["agent_nn"]["train_every_n"]

    # ---Rewards ---

    reward = {
        Cells.TILE.value: tile_reward,
        Cells.TARGET.value: target_reward,
        Cells.PITFALL.value: pitfall_reward,
        Cells.WALL.value: wall_reward,
    }

    # --- Env ---
    env = GridWorldEnv(grid=grid, reward=reward, wind_p=env_wind_p, step_limit=env_step_limit, render_mode="rgb_array")
    wrapped_env = NormalizedCoordWrapper(env)
    wrapped_env = RecordVideo(
        wrapped_env,
        video_folder=video_output_dir.as_posix(),
        episode_trigger=lambda episode_id: episode_id % logging_record_every == 0,
        disable_logger=True # Keeps the console clean
    )

    #  --- Agent ---
    replay_buffer = DequeReplayBuffer(capacity=replay_buffer_capacity)
    learner_kwargs = {
        "replay_buffer": replay_buffer,
        "gamma": agent_gamma,
        "alpha": agent_alpha,
        "epsilon": agent_epsilon,
        "xi": agent_xi,
        # QNetwork
        "hidden_dim": agent_nn_hidden_dim,  
        "batch_size": agent_nn_batch_size,
        "target_update_freq": agent_nn_target_update_freq,
        "train_every_n": agent_nn_train_every_n,
    }

    # --- Solver ---
    solver = BasicGridWorldSolver(wrapped_env, BasicDQNAgent, learner_kwargs, verbose=True)

    # --- Train ---
    solver.train(n_epochs=2000)


def run_curriculum_PER(
    config: dict, 
    grid: np.ndarray, 
    video_output_dir: Path = REPO_ROOT / "runtime" / "videos"
    ):
    raise NotImplementedError


SOLVERS = {
    "basicDQN": run_basic,
    "curPerDQN": run_curriculum_PER,
}


def run(solver_name: str, grid_name: str):
    config = load_config()
    grid = GRIDS[grid_name]
    SOLVERS[solver_name](config, grid)

# --- Agparse ---

parser = argparse.ArgumentParser()

parser.add_argument("--grid", help="The grid to use", type=str, choices=GRIDS.keys(), default=list(GRIDS.keys())[0])
parser.add_argument(
    "--solver", help="The type of agent and solver involved in the gridworld problem.", 
    type=str, choices=["basicDQN", "curPerDQN"],
    default="basicDQN"
    )

args = parser.parse_args()

# --- Main ---

if __name__ == "__main__":
    run(args.solver, args.grid)