from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

def as_distribution(vector: torch.Tensor) -> torch.Tensor:
    clamped = vector.clamp(min=torch.finfo(vector.dtype).eps)
    return clamped / clamped.sum(dim=-1, keepdim=True)

def log_odds(distribution: torch.Tensor) -> torch.Tensor:
    '''Log-odds against the final outcome.'''
    dist = as_distribution(distribution)
    
    return dist[..., :-1].log() - dist[..., -1:].log()

WRITE_SPACES = {
    'log_odds' : lambda t, c: log_odds(t) - log_odds(c),
    'probabilities' : lambda t, c: as_distribution(t) - as_distribution(c),
}

class SemanticController(ABC, nn.Module):
    """Map a requested value distribution to a carrier-coordinate edit."""
    
    def __init__(
        self,
        test_dim: int,
        carrier_dim: int,
        *,
        write_space: Literal['log_odds', 'probabilities'] = 'log_odds',      
    ) -> None:
        super().__init__()
        
        self.test_dim = test_dim
        self.carrier_dim = carrier_dim
        
        self.residual = WRITE_SPACES[write_space]
    
    @abstractmethod
    def forward(
        self,
        target: torch.Tensor,
        current: torch.Tensor,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        ...

class GainController(SemanticController):
    """One learned gain from an outcome-space residual to a carrier edit."""
    
    
    @classmethod
    def from_config(cls, config: dict) -> "GainController":
        return cls(
            test_dim=int(config["test_dim"]),
            carrier_dim=int(config["carrier_dim"]),
            write_space=config.get("write_space", "log_odds"),
        )
    
    def __init__(
        self,
        test_dim: int,
        carrier_dim: int,
        *,
        write_space: Literal['log_odds', 'probabilities'] = 'log_odds',
    ) -> None:
        super().__init__(test_dim, carrier_dim, write_space=write_space)
        
        out_dim = test_dim - 1 if write_space == 'log_odds' else test_dim
        
        self.gain = nn.Parameter(torch.zeros(carrier_dim, out_dim))
    
    def forward(
        self,
        target: torch.Tensor,
        current: torch.Tensor,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual(target, current)
        
        return residual @ self.gain.T

class AffineController(GainController):
    """A learned affine map from an outcome-space residual to a carrier edit."""
    
    @classmethod
    def from_config(cls, config: dict) -> "AffineController":
        return cls(
            test_dim=int(config["test_dim"]),
            carrier_dim=int(config["carrier_dim"]),
            write_space=config.get("write_space", "log_odds"),
        )
    
    def __init__(
        self,
        test_dim: int,
        carrier_dim: int,
        *,
        write_space: Literal['log_odds', 'probabilities'] = 'log_odds',
    ) -> None:
        super().__init__(test_dim, carrier_dim, write_space=write_space)
        
        self.linear = nn.Linear(carrier_dim, carrier_dim)
        
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(
        self,
        target: torch.Tensor,
        current: torch.Tensor,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        residual = super().forward(target, current, carrier)
        
        return residual + self.linear(carrier)