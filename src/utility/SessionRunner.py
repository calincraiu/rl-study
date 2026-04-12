import gymnasium as gym

def play_game(agent, env: gym.Env, num_episodes: int = 1):
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        print(f"--- Episode {episode + 1} Starting ---")

        while not (terminated or truncated):
            # 1. Choose the best action (Greedy)
            # Use the policy directly to avoid the random epsilon-check in actuate()
            action = agent.best_policy[obs] 

            # 2. Take the action
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            
            # 3. Update current observation
            obs = next_obs

            # The 'human' render_mode handles drawing automatically in env.step()
            # but we can add a small delay if it moves too fast for you to see
            # import time; time.sleep(0.1)

        print(f"Episode Finished. Total Reward: {total_reward:.2f}")
    env.close()