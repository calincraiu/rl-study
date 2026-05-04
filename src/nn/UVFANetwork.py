import torch
import torch.nn as nn


class UVFANetwork(nn.Module):
    """
    Two-stream Universal Value Function Approximator.
    """
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Stream 1: Embeds the current state
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stream 2: Embeds the goal
        self.goal_net = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final layer maps the combined embedding to action Q-values
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_net(state)
        goal_emb = self.goal_net(goal)
        
        # Combine using element-wise multiplication / Hadamard product
        # This allows the goal embedding to gate or activate specific state features
        combined = state_emb * goal_emb 
        
        return self.fc(combined)