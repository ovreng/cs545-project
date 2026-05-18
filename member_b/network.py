"""
MLP Q-Network for BrainBlock (Member B).
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    MLP network for DQN.
    Takes flattened board grid (40) and inventory vector (10),
    concatenates them, and outputs 320 Q-values.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # grid = 5*8 = 40, vec = 10 -> total input = 50
        input_dim = 40 + 10
        output_dim = 320

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        grid: (B, 1, 5, 8)
        vec:  (B, 10)
        """
        batch_size = grid.shape[0]
        # Flatten grid from (B, 1, 5, 8) -> (B, 40)
        grid_flat = grid.view(batch_size, -1)
        
        # Concat -> (B, 50)
        x = torch.cat([grid_flat, vec], dim=1)
        
        # Output Q-values
        q_values = self.net(x)
        return q_values
