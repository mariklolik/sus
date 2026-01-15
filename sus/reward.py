import torch
import torch.nn as nn
from .encoder import StrategyEncoder


class SuSReward(nn.Module):
    """Strategy-aware Surprise intrinsic reward module."""
    
    def __init__(
        self,
        input_dim: int,
        strategy_dim: int = 128,
        lambda_ss: float = 1.0,
        lambda_sus: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = StrategyEncoder(input_dim, strategy_dim, dropout)
        self.lambda_ss = lambda_ss
        self.lambda_sus = lambda_sus
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state into strategy embedding."""
        return self.encoder(state)
    
    def compute_ss(self, z_pre: torch.Tensor, z_post: torch.Tensor) -> torch.Tensor:
        """Compute Strategy Stability: cosine similarity between embeddings."""
        return (z_pre * z_post).sum(dim=-1)
    
    def compute_sus(
        self, 
        z_pre: torch.Tensor, 
        z_post: torch.Tensor, 
        prediction_error: torch.Tensor
    ) -> torch.Tensor:
        """Compute Strategy Surprise: prediction error weighted by strategy shift."""
        ss = self.compute_ss(z_pre, z_post)
        return prediction_error * (1 - ss)
    
    def compute_reward(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined intrinsic reward."""
        ss = self.compute_ss(z_pre, z_post)
        sus = prediction_error * (1 - ss)
        return self.lambda_ss * ss + self.lambda_sus * sus
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: encode states and compute reward."""
        z_pre = self.encode(state)
        z_post = self.encode(next_state)
        return self.compute_reward(z_pre, z_post, prediction_error)
