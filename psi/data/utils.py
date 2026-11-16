
import torch

from einops import rearrange
from psi.data.types import TaskInputs
from psi.model.subject import LMSubject

def record_activations(
    inputs: TaskInputs,
    subject: LMSubject,
    positions: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    """Record the activations of the subject model on the task's dataset.

    Args:
        inputs (TaskInputs): The prepared task inputs.
        subject (LMSubject): The subject model to record activations from.
        positions (torch.Tensor): The cut positions for the task inputs of
            shape (num_examples, num_positions, num_cut_points)
        batch_size (int, optional): The batch size for processing. Defaults to 32.
    """

    if len(positions) != len(inputs):
        raise ValueError(
            f"Length of positions ({len(positions)}) must match length of inputs ({len(inputs)})."
        )

    activations = []
    device = subject.device

    for start in range(0, len(inputs), batch_size):
        end = start + batch_size
        batch = {
            "input_ids": inputs.input_ids[start:end].to(device),
            "attention_mask": inputs.attention_mask[start:end].to(device),
            "position_ids": inputs.position_ids[start:end].to(device),
            "cut_positions": positions[start:end].flatten(1).to(device) # (b, n * s)
        }

        with torch.inference_mode():
            activation = subject.extract(batch)

        activations.append(activation.cpu())

    # Flatten the cut positions
    b, n, s = positions.shape

    activations = torch.cat(activations, dim=0) # (b, n * h)
    activations = rearrange(activations.cpu(), "b (n h) -> b n h", n=n)

    return activations

def record_behavior(
    inputs: TaskInputs,
    subject: LMSubject,
    outcome_ids: torch.Tensor,
    outcome_mask: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    """Record the behavior of the subject model on the task's dataset."""

    behaviors = []
    device = subject.device

    for start in range(0, len(inputs.input_ids), batch_size):
        end = start + batch_size
        batch = {
            "input_ids": inputs.input_ids[start:end].to(device),
            "attention_mask": inputs.attention_mask[start:end].to(device),
            "position_ids": inputs.position_ids[start:end].to(device),
            'outcome_ids': outcome_ids[start:end].to(device),
            'outcome_mask': outcome_mask[start:end].to(device),
        }

        with torch.inference_mode():
            behavior = subject.behavior(batch)

        behaviors.append(behavior.cpu())

    behaviors = torch.cat(behaviors, dim=0) # (b, n * h)

    return behaviors
