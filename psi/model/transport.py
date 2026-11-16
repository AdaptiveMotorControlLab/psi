
from typing import Protocol

from torch import Tensor
import torch


class SemanticTransport(Protocol):
    """
    An object that implements the semantic transport mechanism.
    It takes in a source state and a base behaviour, and outputs
    a transported behaviour.
    """

    @classmethod
    def from_name(cls, name: str) -> "SemanticTransport":
        """Create a SemanticTransport from a configuration dictionary."""
        match name:
            case "full_change":
                return FullChangeTransport()
            case "gated_change":
                return GatedTransport()
            case "no_change":
                return NoChangeTransport()
            case "table":
                return TableTransport()
            case _:
                raise ValueError(f"Unknown transport kind: {name!r}")

    def __call__(
        self,
        source_state: Tensor,
        base_behaviour: Tensor,
        transport_gate: Tensor | None,
    ) -> Tensor:
        """
        Perform the semantic transport operation.

        Args:
            source_state: The source state tensor.
            base_behaviour: The base behaviour tensor.
            transport_gate: The transport gate tensor.

        Returns:
            A tensor representing the transported behaviour.
        """
        ...

class FullChangeTransport(SemanticTransport):
    """
    A simple implementation of the SemanticTransport protocol that
    returns the source state, indicating a full change in behaviour.
    """

    def __call__(
        self,
        source_state: Tensor,
        base_behaviour: Tensor,
        transport_gate: Tensor | None,
    ) -> Tensor:
        """
        Return the source state.

        Args:
            source_state: The source state tensor.
            base_behaviour: The base behaviour tensor (unused).
            transport_gate: The transport gate tensor (unused).

        Returns:
            The source state tensor.
        """
        return source_state

class NoChangeTransport(SemanticTransport):
    """
    A simple implementation of the SemanticTransport protocol that
    returns the base behaviour unchanged, effectively acting as an identity function.
    """

    def __call__(
        self,
        source_state: Tensor,
        base_behaviour: Tensor,
        transport_gate: Tensor | None,
    ) -> Tensor:
        """
        Return the base behaviour unchanged.

        Args:
            source_state: The source state tensor (unused).
            base_behaviour: The base behaviour tensor.
            transport_gate: The transport gate tensor (unused).

        Returns:
            The base behaviour tensor unchanged.
        """
        return base_behaviour

class GatedTransport(SemanticTransport):
    """
    A simple implementation of the SemanticTransport protocol that
    returns a gated combination of the source state and base behaviour.
    """

    def __call__(
        self,
        source_state: Tensor,
        base_behaviour: Tensor,
        transport_gate: Tensor,
    ) -> Tensor:
        """
        Return a gated combination of the source state and base behaviour.

        Args:
            source_state: The source state tensor.
            base_behaviour: The base behaviour tensor.
            transport_gate: The transport gate tensor.

        Returns:
            A tensor representing the gated combination of source state and base behaviour.
        """
        if transport_gate is None:
            raise ValueError("transport_gate must be provided for GatedTransport.")

        gate = transport_gate.bool().unsqueeze(-1)  # (num_examples, 1)

        return torch.where(gate, source_state, base_behaviour)

class TableTransport(SemanticTransport):
    """
    A generic implementation for a semantic transport represented as a table
    of shape (num_examples, num_outputs, num_outputs), where the first output 
    is the semantic source and the second output is the semantic target.
    """
    
    def __call__(
        self,
        source_state: Tensor,
        base_behaviour: Tensor,
        transport_gate: Tensor,
    ) -> Tensor:
        if transport_gate is None:
            raise ValueError("transport_gate must be provided for TableTransport.")
        
        batch = torch.arange(len(source_state), device=source_state.device)
        
        # Find the indices of the semantic target and source for each example
        source_class = source_state.argmax(dim=-1)  # (num_examples,)
        
        return transport_gate[batch, source_class]