import numpy as np
import gymnasium as gym

class QLambdaLearner:
    """
    Backward-view Q-Learning with TD(λ)-equivalence. Maintains eligibility traces that are used to update all (encountered) Q values.

    λ = 0       ->  Q-learning, no traces
    λ ≈ 0.7-0.9 ->  practical sweet spot - propagating into the past a reasonably long amount
    λ = 1	    ->  Monte Carlo-like (long credit chains from start to finish)

    """
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.9,
        xi: float = 0.99,
        lam: float = 0.8 # λ parameter ; Reflects how far back credit assignment goes
    ):
        self.env = env
        
        # --- Num states in environment observation space
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.num_states = env.observation_space.n
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.num_states = np.prod(env.observation_space.shape)
        else:
            # Fallback for complex spaces
            raise ValueError(f"Unsupported observation space: {type(env.observation_space)}")
        
        # --- Num actions in environment
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = env.action_space.n
        else:
            self.num_actions = getattr(env.action_space, 'n', 0)

        # --- Hyperparams
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi
        self.lam = lam

        # --- Values, traces, policy
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.e_trace = np.zeros((self.num_states, self.num_actions)) # Eligibility traces 
        self.cur_policy = np.random.randint(self.num_actions, size=self.num_states)

        # Keep track the of best policy. Note - Q always improves after every game, but the policy itself may oscilate due to stochasticity
        self.best_policy: np.ndarray | None = None 
        self.best_reward: float = -np.inf

    def reset_traces(self):
        self.e_trace.fill(0) # Resetting traces is required - as traces describe only what happened during ONE game / epoch

    def percept(self, s: int, a: int, s_prime: int, r: float):
        # Increase trace for visited (s, a)
        self.e_trace[s, a] += 1
        
        # TD error (same as Q-learning)
        q_prime = np.max(self.q_table[s_prime])
        delta = r + self.gamma * q_prime - self.q_table[s, a]

        # Update ALL Q-values
        self.q_table += self.alpha * delta * self.e_trace

        # Decay traces + apply Watkins's rule: only propagate credit backward as long as actions are greedy. The moment you take a non-greedy action, reset all traces.
        # Intuition - Watkins’ Q(λ) prevents accidentally learning that bad exploratory decisions are good just because the policy (or exploration) fixed them later.
        greedy_action = np.argmax(self.q_table[s])
        if a != greedy_action:
            self.reset_traces() # If we just took an exploratory action, reset traces
        else:
            self.e_trace *= self.gamma * self.lam # Otherwise, keep traces and decay them

        # Update policy
        self.cur_policy[s] = np.argmax(self.q_table[s]) # Current policy


    def actuate(self, s):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.num_actions) # Exploration
        return int(self.cur_policy[s]) # Exploitation

    def update_episode(self, epoch_total_reward: float):
        self.epsilon *= self.xi # Decay exploration threhsold
        self.reset_traces()  # Reset traces every episode!
        if epoch_total_reward >= self.best_reward: # Keep track of the best policy between episode updates
            self.best_reward = epoch_total_reward
            self.best_policy = self.cur_policy.copy()