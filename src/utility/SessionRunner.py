import copy
import gymnasium as gym

from src.agents.Agent import Agent

def play_game(agent: Agent, env: gym.Env, num_episodes: int = 1):
    env_copy = copy.copy(env)
    for episode in range(num_episodes):
        obs, info = env_copy.reset()
        terminated = False
        truncated = False
        total_reward = 0
        num_steps = 0

        print(f"--- Episode {episode + 1} Starting ---")
        agent_policy = agent.get_policy()

        while not (terminated or truncated):
            # 1. Choose the best action (Greedy)
            # Use the policy directly to avoid the random epsilon-check in actuate()
            action = agent_policy[obs] 

            # 2. Take the action
            next_obs, reward, terminated, truncated, info = env_copy.step(action)
            total_reward += float(reward)
            
            # 3. Update current observation
            obs = next_obs

            num_steps += 1

        print(f"Episode Finished. Num steps: {num_steps}; Total reward: {total_reward:.2f}")

    env_copy.close()