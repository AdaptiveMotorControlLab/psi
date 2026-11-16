
import torch
from torch import Tensor
from pathlib import Path
from dataclasses import dataclass

from typing import Mapping

import yaml

TaskInfo = Mapping[str, int | str | float | bool]

@dataclass(frozen=True)
class TaskObservations:
    """A generic task observation. Specific task will define their
    own observation dataclass with the relevant fields."""
    prompts: list[str]

@dataclass(frozen=True)
class TaskInputs:
    """A collection of prepared (shared) input tensors for a task.

    NOTE: num_examples is the total number of examples in the dataset,
    before any variable-specfic correct-gating is applied.
    """
    input_ids: Tensor # (num_examples, seq_len)
    position_ids: Tensor # (num_examples, seq_len)
    attention_mask: Tensor # (num_examples, seq_len)
    offset_mapping: list[list[tuple[int, int]]] | None = None

    def __post_init__(self):
        if not (self.input_ids.shape == self.position_ids.shape == self.attention_mask.shape):
            raise ValueError(
                "All input tensors must have the same shape. "
                f"Got input_ids: {self.input_ids.shape}, "
                f"position_ids: {self.position_ids.shape}, "
                f"attention_mask: {self.attention_mask.shape}."
            )

    def __len__(self) -> int:
        return self.input_ids.shape[0]

@dataclass(frozen=True)
class TaskActivations:
    """A collection of prepared activations for a task."""
    names: tuple[str, ...]

    values: Tensor # (num_examples, num_positions, hidden_dim)
    positions: Tensor # (num_examples, num_positions, num_cut_points)

    def position(self, name: str) -> Tensor:
        if name not in self.names:
            raise ValueError(f"Position '{name}' not found.")
        return self.positions[:, self.names.index(name)] # (num_examples, num_cut_points)

    def activation(self, name: str) -> Tensor:
        if name not in self.names:
            raise ValueError(f"Position '{name}' not found.")
        return self.values[:, self.names.index(name)] # (num_examples, hidden_dim)

@dataclass(frozen=True)
class TaskOutcomes:
    """A collection of prepared variable-specific outcomes for a task.

    NOTE: num_examples is the number of examples in the dataset that pass
    the correct-gating for the variable of interest. `correct_indices`
    represent the indices of these examples in the original dataset and
    may be used to align input with outputs via:
    ```
        inputs = task.inputs(split)
        outcomes = task.outcomes(split)
        correct_inputs = inputs[outcomes.correct_indices]
    ```
    """
    outcome_ids: Tensor # (num_examples, num_outcomes, out_len)
    outcome_mask: Tensor # (num_examples, num_outcomes, out_len)

    correct_indices: Tensor # (num_examples, )
    state_features: Tensor # (num_examples, state_dim)
    base_behaviour: Tensor # (num_examples, behaviour_dim)

    gold_state_features: Tensor | None = None
    transport_gate: Tensor | None = None
    transport_group: Tensor | None = None

@dataclass(frozen=True)
class TaskPair:
    """A single task pair.

    base_index: which observation receives the intervention
    source_index: which observation donates the activation
    family_id: which counterfactual condition this pair belongs to
    example_id: original dataset-row identity
    """

    family_id: int    # long, (num_pairs,)
    example_id: int   # long, (num_pairs,)
    base_index: int   # long, (num_pairs,)
    source_index: int # long, (num_pairs,)

@dataclass(frozen=True)
class TaskPairs:
    """An index-based referential representation of task pairs"""

    family_id: Tensor    # which counterfactual condition this pair belongs to
    example_id: Tensor   # original dataset-row identity
    base_index: Tensor   # which observation receives the intervention
    source_index: Tensor # which observation donates the activation

    def __iter__(self):
        for family_id, example_id, base_index, source_index in zip(
            self.family_id,
            self.example_id,
            self.base_index,
            self.source_index,
        ):
            yield TaskPair(
                family_id=int(family_id.item()),
                example_id=int(example_id.item()),
                base_index=int(base_index.item()),
                source_index=int(source_index.item()),
            )

@dataclass(frozen=True)
class InterventionPairs(TaskPairs):
    """A collection of intervention pairs for a task

    [TaskPairs]

    target: the correct expected outcome in the base observation's outcome space
    """

    target: Tensor # (num_pairs,)

@dataclass(frozen=True, slots=True)
class TaskCache:
    """A collection of prepared data for a task, including train, validation, and test splits."""

    root: Path
    splits: frozenset[str]
    manifest: Mapping[str, object]

    position: str
    variable: str
    cut_fingerprint: str

    @property
    def activation_path(self) -> Path:
        """Return the path to the cached activations for this task."""
        return self.root / "activations" / self.cut_fingerprint

    def total_examples(self, split: str) -> int:
        """Return the total number of examples in the given split."""
        # Load split manifest
        manifest_path = self.root / self.variable / "manifest.yaml"

        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
        return manifest.get(split, {}).get("total_examples", 0)

    def inputs(self, split: str) -> TaskInputs:
        """Load and return the prepared task split corresponding to the given name."""
        payload = torch.load(
            self.root / split / "inputs.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        return TaskInputs(**payload)

    def outcomes(self, split: str) -> TaskOutcomes:
        """Load and return the prepared task split corresponding to the given name."""
        payload = torch.load(
            self.root / self.variable / split / "outcomes.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        return TaskOutcomes(**payload)

    def activations(self, split: str) -> TaskActivations:
        """Load and return the cached activations for the given split."""
        payload = torch.load(
            self.activation_path / f"{split}.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        return TaskActivations(**payload)

    def interventions(self, split: str) -> InterventionPairs:
        """Load and return the intervention pairs for the given split."""
        pairs = torch.load(
            self.root / split / "pairs.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        targets = torch.load(
            self.root / self.variable / split / "targets.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        outcomes = self.outcomes(split)
        correct = outcomes.correct_indices

        raw_to_record = torch.full(
            (self.total_examples(split),),
            -1,
            dtype=torch.long,
        )
        raw_to_record[correct] = torch.arange(len(correct))

        base = raw_to_record[pairs["base_index"]]
        source = raw_to_record[pairs["source_index"]]

        keep = (base >= 0) & (source >= 0)

        return InterventionPairs(
            target=targets[keep],
            family_id=pairs["family_id"][keep],
            example_id=pairs["example_id"][keep],
            base_index=base[keep],
            source_index=source[keep],
        )

    def records(self, split: str) -> Mapping[str, Tensor]:
        inputs = self.inputs(split)
        outcomes = self.outcomes(split)
        activations = self.activations(split)

        input_correct = outcomes.correct_indices
        positions = activations.position(self.position)
        activation = activations.activation(self.position)

        return {
            "input_ids": inputs.input_ids[input_correct],
            "position_ids": inputs.position_ids[input_correct],
            "attention_mask": inputs.attention_mask[input_correct],

            # Select one named cut location.
            "activation": activation[input_correct],
            "cut_positions": positions[input_correct],

            "outcome_ids": outcomes.outcome_ids,
            "outcome_mask": outcomes.outcome_mask,
            "state_features": outcomes.state_features,
            "base_behaviour": outcomes.base_behaviour,
            **(
                {"gold_state_features": outcomes.gold_state_features}
                if outcomes.gold_state_features is not None
                else {}
            ),
            **(
                {"transport_gate": outcomes.transport_gate}
                if outcomes.transport_gate is not None
                else {}
            ),
            **(
                {"transport_group": outcomes.transport_group}
                if outcomes.transport_group is not None
                else {}
            ),
        }
