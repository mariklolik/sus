import torch
import torch.nn as nn


class StrategyEncoder(nn.Module):
    """Encodes states into strategy embeddings."""
    
    def __init__(self, input_dim: int, strategy_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, strategy_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.encoder(x), dim=-1)
