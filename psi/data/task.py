from psi.data.types import TaskActivations, TaskCache, TaskInfo, TaskInputs, TaskObservations, TaskOutcomes, TaskPairs
from psi.data.utils import record_activations
from psi.model.subject import LMSubject
from psi.utils.hf import get_hf_dataset


import torch
import yaml
from torch import Tensor
from transformers import TokenizersBackend


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class TaskPreparation(ABC):
    """A task preparation is a procedure that defines the necessary
    data-preparation steps for a given task, including:
    1. Definition of the semantic program: the declared answers the variable can produce
    2. Loading of raw examples and their conversion to model input tensors
    3. Measurement of the model's frozen behaviour on the prepared examples
    4. Quality gating of the resulting model behaviour to ensure the task is well-defined and feasible
    """

    task_name: str

    def __init__(self, config: dict[str, Any], tokenizer: TokenizersBackend):
        self.tokenizer = tokenizer

        if not self.tokenizer.is_fast:
            raise ValueError(
                "This analysis requires a fast tokenizer with offset mappings."
            )
        
        

        self.root = Path(config['cache_root'])
        self.subset = config.get('subset')
        self.dataset = config['dataset']
        self.variable = config['variable']
        self.max_length = int(config['max_length'])
        self.quality_gate = float(config['quality_gate'])
        self.cache_version = str(config.get('cache_version', '1.0'))
        self.counterfactuals = tuple(config['counterfactuals'])
        self.record_batch_size = int(config.get('record_batch_size', 64))
        self.correctness_filter = config.get('correctness_filter', True)
        
        if not isinstance(self.correctness_filter, bool):
            raise ValueError(
                f"correctness_filter must be a boolean, got {type(self.correctness_filter)}"
            )

        if not self.variable in self.variable_names:
            raise ValueError(f"Invalid variable: {self.variable}. Must be one of {self.variable_names}.")

    @property
    @abstractmethod
    def variable_names(self) -> tuple[str, ...]:
        """Return the names of the variables in this task."""
        ...

    @property
    @abstractmethod
    def position_names(self) -> tuple[str, ...]:
        """Return the names of the cut positions in this task."""
        ...

    @property
    @abstractmethod
    def num_outcomes(self) -> int:
        """Return the number of outcomes in this task."""
        ...
    
    @property
    def filtered(self) -> str:
        """Return a string indicating whether the task is filtered or unfiltered."""
        return "filtered" if self.correctness_filter else "unfiltered"

    @property
    def task_metadata(self) -> dict[str, Any]:
        """Return additional metadata for the task."""
        return {}

    @property
    def raw_splits(self) -> Mapping[str, str]:
        """Return the names of the raw dataset splits for this task."""
        return {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }

    @property
    def task_observation(self) -> type[TaskObservations]:
        """Return the type of the task observations for this task."""
        return TaskObservations

    @abstractmethod
    def identity(self, subject: LMSubject) -> str:
        """A unique identifier for the task preparation, which may depend on the subject."""
        ...

    def manifest(self, root: str | Path, manifest: Mapping[str, object]) -> None:
        """Write a manifest file for the prepared data."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        manifest_path = root / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.safe_dump(dict(manifest), f, sort_keys=False)

    def materialize(self, subject : LMSubject) -> TaskCache:
        """Materialize the task preparation into a concrete set of
        prepared data, including the model's behaviour on the task.
        If the prepared data already exists in the root directory,
        it will be loaded and not recomputed.

        Args:
            subject: The model subject to be evaluated on the task.

        Returns:
            A tuple containing the prepared data and the intervention pairs.
        """
        """Materialize the task into its input, activations, outcomes and interventions."""
        cache_path = self.root / self.identity(subject)
        activation_cache = cache_path / "activations"

        # Check whether shared inputs are already prepared
        # if not, generate and cache them for all splits
        if not all(
            (cache_path / split / name).exists()
            for split in ('train', 'validation', 'test')
            for name in ("inputs.pt", "observations.pt", "pairs.pt")
        ):
            raw = get_hf_dataset(self.dataset, subset=self.subset)

            for cache_split, raw_split in self.raw_splits.items():
                input_path = cache_path / cache_split
                input_path.mkdir(parents=True, exist_ok=True)

                inputs, observations, pairs = self._prepare_input(raw[raw_split])

                torch.save(vars(pairs), input_path / "pairs.pt")
                torch.save(vars(inputs), input_path / "inputs.pt")
                torch.save(vars(observations), input_path / "observations.pt")

        # Materialize the activation cache if it doesn't already exist
        for split in ('train', 'validation', 'test'):
            activation_path = activation_cache / subject.cut.fingerprint / f"{split}.pt"

            if not activation_path.exists():
                activation_path.parent.mkdir(parents=True, exist_ok=True)
                Observations = self.task_observation
                inputs = TaskInputs(**torch.load(
                    cache_path / split / "inputs.pt",
                    map_location="cpu",
                    weights_only=True,
                ))
                observations = Observations(**torch.load(
                    cache_path / split / "observations.pt",
                    map_location="cpu",
                    weights_only=True,
                ))
                activations = self._prepare_activations(inputs, subject, observations)
                torch.save(vars(activations), activation_path)

        # Check if observations and interventions are already prepared
        # for all splits, if not, generate and cache them for all splits
        if not all(
            (cache_path / self.variable / split / name).exists()
            for split in ('train', 'validation', 'test')
            for name in ('outcomes.pt', 'targets.pt')
        ):
            # Prepare variable-specific observations and interventions
            var_path = cache_path / self.variable
            var_path.mkdir(parents=True, exist_ok=True)

            info = {
                'variable': self.variable,
                'num_outcomes': self.num_outcomes,
                'quality_gate': self.quality_gate,
            }
            for split in ('train', 'validation', 'test'):
                data_path = var_path / split
                data_path.mkdir(parents=True, exist_ok=True)

                # Load the shared input data for this split
                Observations = self.task_observation

                inp = TaskInputs(**torch.load(
                    cache_path / split / "inputs.pt",
                    map_location="cpu",
                    weights_only=True,
                ))
                obs = Observations(**torch.load(
                    cache_path / split / "observations.pt",
                    map_location="cpu",
                    weights_only=True,
                ))
                pairs = TaskPairs(**torch.load(
                    cache_path / split / "pairs.pt",
                    map_location="cpu",
                    weights_only=True,
                ))
                outcomes, targets, split_info = self._prepare_outcomes(inp, subject, obs, pairs)

                torch.save(targets, data_path / "targets.pt")
                torch.save(vars(outcomes), data_path / "outcomes.pt")
                info[split] = split_info

            self.manifest(var_path, info)

        # Write global manifest
        manifest = {
            "task": self.task_name,
            "subset": self.subset,
            "dataset": self.dataset,
            "subject": subject.model_fingerprint,
            "max_length": self.max_length,
            "counterfactuals": list(self.counterfactuals),
            "variable_names": list(self.variable_names),
            "position_names": list(self.position_names),
            **self.task_metadata,
        }
        if not (cache_path / "manifest.yaml").exists():
            self.manifest(cache_path, manifest)

        splits = frozenset(('train', 'validation', 'test'))
        return TaskCache(
            root=cache_path,
            splits=splits,
            manifest=manifest,
            variable=self.variable,
            position=subject.cut.position,
            cut_fingerprint=subject.cut.fingerprint
        )

    @abstractmethod
    def _prepare_input(self, raw_split: Any) -> tuple[TaskInputs, TaskObservations, TaskPairs]:
        """Prepare the shared input tensors for the task from the raw dataset split."""
        ...

    @abstractmethod
    def _prepare_activations(self, inputs: TaskInputs, subject: LMSubject, observations: TaskObservations) -> TaskActivations:
        """Prepare the model's activations for the task from the shared inputs."""
        ...

    @abstractmethod
    def _prepare_outcomes(self, inputs: TaskInputs, subject: LMSubject, observations: TaskObservations, pairs: TaskPairs) -> tuple[TaskOutcomes, Tensor, TaskInfo]:
        """Prepare the variable-specific outcomes for the task from the shared inputs and observations."""
        ...