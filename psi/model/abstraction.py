"""Invertible activation abstractions."""

from abc import ABC, abstractmethod

from einops import rearrange
import torch
from torch import Tensor, nn


class Abstraction(nn.Module, ABC):
    """Split an activation into state, carrier, and nuisance coordinates."""

    @classmethod
    def from_config(cls, config: dict) -> "Abstraction":
        match kind := config.get("kind", "linear"):
            case "linear":
                return LinearAbstraction.from_config(config)
            case "sitewise_linear":
                return SitewiseLinearAbstraction.from_config(config)
            case _:
                raise ValueError(f"Unknown abstraction kind: {kind!r}.")

    def __init__(
        self,
        carrier_dim: int,
        activation_dim: int,
    ) -> None:
        super().__init__()
        if not 0 < carrier_dim <= activation_dim:
            raise ValueError(
                "Dimensions must satisfy 0 < carrier_dim <= activation_dim."
            )
        self.carrier_dim = int(carrier_dim)
        self.activation_dim = int(activation_dim)

    def interchange(self, base: Tensor, source: Tensor) -> Tensor:
        """transport the learned causal carrier from source to base."""
        src_carrier, _ = self.lift(source)
        _, base_nuisance = self.lift(base)

        return self.compose(
            src_carrier,
            base_nuisance
        )

    @abstractmethod
    def lift(self, activation: Tensor) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def compose(self, carrier: Tensor, nuisance: Tensor) -> Tensor: ...


class LinearAbstraction(Abstraction):
    """An orthogonal activation-to-carrier map with an exact complement."""

    @classmethod
    def from_config(cls, config: dict) -> "LinearAbstraction":
        return cls(
            carrier_dim=int(config["carrier_rank"]),
            activation_dim=int(config["activation_dim"]),
        )

    def __init__(
        self,
        carrier_dim: int,
        activation_dim: int,
    ) -> None:
        super().__init__(carrier_dim, activation_dim)
        basis = torch.empty(self.carrier_dim, self.activation_dim)
        nn.init.orthogonal_(basis)
        self.raw_basis = nn.Parameter(basis)

    def basis(self) -> Tensor:
        return torch.linalg.qr(self.raw_basis.T, mode="reduced").Q.T

    def lift(self, activation: Tensor) -> tuple[Tensor, Tensor]:
        if activation.shape[-1] != self.activation_dim:
            raise ValueError(
                f"Expected activation width {self.activation_dim}, "
                f"got {activation.shape[-1]}."
            )
        basis = self.basis()
        carrier = activation @ basis.T
        nuisance = activation - carrier @ basis
        return carrier, nuisance

    def compose(self, carrier: Tensor, nuisance: Tensor) -> Tensor:
        if carrier.shape[-1] != self.carrier_dim:
            raise ValueError(f"Expected carrier width {self.carrier_dim}.")
        if nuisance.shape[-1] != self.activation_dim:
            raise ValueError(f"Expected nuisance width {self.activation_dim}.")
        return nuisance + carrier @ self.basis()

class SitewiseLinearAbstraction(Abstraction):
    """A linear abstraction that is applied independently to each site."""

    @classmethod
    def from_config(cls, config: dict) -> "SitewiseLinearAbstraction":
        return cls(
            site_dim=int(config["site_dim"]),
            num_sites=int(config["num_sites"]),
            num_positions=int(config["num_positions"]),
            carrier_dim_per_site=int(config["carrier_rank_per_site"]),
        )

    def __init__(
        self,
        site_dim: int,
        num_sites: int,
        num_positions: int,
        carrier_dim_per_site: int,
    ) -> None:
        self.site_dim = site_dim
        self.num_sites = num_sites
        self.num_positions = num_positions
        self.carrier_dim_per_site = carrier_dim_per_site
        
        carrier_dim = num_sites * num_positions * carrier_dim_per_site
        activation_dim = num_sites * num_positions * site_dim

        super().__init__(carrier_dim, activation_dim)

        basis = torch.empty((num_sites, self.carrier_dim_per_site, site_dim))
        for i in range(num_sites):
            nn.init.orthogonal_(basis[i])

        self.raw_basis = nn.Parameter(basis)

    def basis(self) -> Tensor:
        return torch.linalg.qr(
            self.raw_basis.transpose(-1, -2), mode="reduced"
        ).Q.transpose(-1, -2)

    def lift(self, activation: Tensor) -> tuple[Tensor, Tensor]:
        if activation.shape[-1] != self.activation_dim:
            raise ValueError(
                f"Expected activation width {self.activation_dim}, "
                f"got {activation.shape[-1]}."
            )

        # Interpret activation as shape (batch, num_sites, num_positions, site_dim)
        activation = rearrange(
            activation, 'b (s p d) -> b s p d',
            s=self.num_sites,
            p=self.num_positions,
            d=self.site_dim
        )

        basis = self.basis() # (s, c, d)

        carrier = activation @ basis.transpose(-1, -2) # (b, s, p, c)
        nuisance = activation - carrier @ basis

        carrier = rearrange(carrier, 'b s p c -> b (s p c)')
        nuisance = rearrange(nuisance, 'b s p d -> b (s p d)')

        return carrier, nuisance

    def compose(self, carrier: Tensor, nuisance: Tensor) -> Tensor:
        if carrier.shape[-1] != self.carrier_dim:
            raise ValueError(f"Expected causal-carrier width {self.carrier_dim}.")
        if nuisance.shape[-1] != self.activation_dim:
            raise ValueError(f"Expected nuisance width {self.activation_dim}.")

        carrier = rearrange(
            carrier, 'b (s p c) -> b s p c',
            s=self.num_sites,
            p=self.num_positions,
            c=self.carrier_dim_per_site
        )

        causal = carrier @ self.basis() # (b, s, p, d)

        return nuisance + causal.flatten(1) # (b, s * p * d)
