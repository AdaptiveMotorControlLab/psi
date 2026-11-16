from typing import Literal

import torch
from torch import Tensor
from math import log, isfinite
from collections.abc import Mapping
from dataclasses import dataclass, field

MassModes = Literal['full', 'conditional', 'reference']

def log1mexp(x: Tensor) -> Tensor:
    """Compute log(1 - exp(x)) in a numerically stable way."""
    THR = log(0.5)
    
    upper = torch.log(-torch.expm1(x.clamp(max=-1e-7)))
    lower = torch.log1p(-torch.exp(x.clamp(max=THR)))
    
    return torch.where(x > THR, upper, lower)

def cross_entropy(
    log_prob: Tensor,
    target: Tensor,
    mode: MassModes = 'full',
    reference: Tensor | None = None,
) -> torch.Tensor:
    """Compute the cross-entropy loss for a batch of logits and targets."""
    log_mass = log_prob.logsumexp(dim=-1, keepdim=True)
    
    match mode:
        case 'full':
            return -(target * log_prob).sum(dim=-1)
        case 'conditional':
            return -(target * (log_prob - log_mass)).sum(dim=-1)
        case 'reference':
            if reference is None:
                raise ValueError("Reference distribution must be provided for 'reference' mode.")
            return -(
                reference * (target * log_prob).sum(dim=-1)
                + (1. - reference) * log1mexp(log_mass.squeeze(-1))
            )
        case _:
            raise ValueError(f"Unknown cross_entropy mode: {mode}")

@dataclass(slots=True, frozen=True)
class LossWeights:
    donor: float = 1.0
    control: float = 0.0
    nuisance: float = 0.25
    calibration: float = 1.0

    def __post_init__(self) -> None:
        values = (self.donor, self.nuisance, self.calibration, self.control)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("Loss weights must be finite and nonnegative.")
        if not any(values):
            raise ValueError("At least one loss weight must be positive.")


@dataclass(slots=True, frozen=True)
class ObjectiveResult:
    donor_loss: Tensor
    control_loss: Tensor
    nuisance_loss: Tensor
    calibration_loss: Tensor
    loss_weights: LossWeights
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)

    @property
    def loss(self) -> Tensor:
        return (
            self.loss_weights.calibration * self.calibration_loss
            + self.loss_weights.donor * self.donor_loss
            + self.loss_weights.nuisance * self.nuisance_loss
            + self.loss_weights.control * self.control_loss
        )

    @property
    def metrics(self) -> Mapping[str, Tensor]:
        return {
            "loss/calibration": self.calibration_loss,
            "loss/donor": self.donor_loss,
            "loss/control": self.control_loss,
            "loss/nuisance": self.nuisance_loss,
            **{f'support/{name}' : value for name, value in self.diagnostics.items()},
        }
