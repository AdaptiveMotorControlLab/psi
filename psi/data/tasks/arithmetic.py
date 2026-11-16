import re
import torch

from typing import Any, Mapping

from dataclasses import dataclass
from transformers import TokenizersBackend

from psi.data.task import TaskPreparation
from psi.data.types import (
    TaskActivations,
    TaskObservations,
    TaskInfo,
    TaskPairs,
    TaskInputs,
    TaskOutcomes,
)
from psi.data.utils import record_activations, record_behavior
from psi.model.subject import LMSubject

@dataclass(frozen=True)
class ArithmeticObservation:
    """A single observation for the Arithmetic task."""

    prompt: str
    operand1: int
    operand2: int

    @property
    def carry(self) -> int:
        """Returns the carry for the addition of operand1 and operand2."""
        return int(self.operand1 % 10 + self.operand2 % 10 > 9)

@dataclass(frozen=True)
class ArithmeticObservations(TaskObservations):
    """A collection of Arithmetic observations for a given dataset split."""
    prompts: list[str]
    operand1: list[int]
    operand2: list[int]

    answers: list[int]

    def __iter__(self):
        for prompt, operand1, operand2 in zip(self.prompts, self.operand1, self.operand2):
            yield ArithmeticObservation(prompt, operand1, operand2)

class ArithmeticTask(TaskPreparation):
    """An arithmetic task where the model is asked to add two numbers together (two-digit)."""

    task_name = "arithmetic"

    def __init__(self, config: dict[str, Any], tokenizer: TokenizersBackend):
        super().__init__(config, tokenizer)

        self.num_digits = int(config.get("num_digits", 2))

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return the names of the variables in this task."""
        return ('x_carry',)

    @property
    def position_names(self) -> tuple[str, ...]:
        """Return the names of the positions in this task."""
        return ('op2_last', 'last',)

    @property
    def num_outcomes(self) -> int:
        """Return the number of possible outcomes for this task."""
        return 2  # Carry can be either 0 or 1

    @property
    def task_metadata(self) -> dict[str, Any]:
        """Return additional metadata for the task."""
        return {
            "num_digits": self.num_digits,
        }

    @property
    def task_observation(self) -> type[ArithmeticObservations]:
        """Return the type of the task observations for this task."""
        return ArithmeticObservations

    def identity(self, subject: LMSubject) -> str:
        """Return a unique identity string for this task configuration."""
        subject_hash = subject.model_fingerprint
        counterfactuals = "_".join(self.counterfactuals)

        return f"{self.task_name}_{self.dataset}_{self.num_digits}_{self.max_length}_{counterfactuals}_{subject_hash}_v{self.cache_version}"

    def _prepare_input(self, raw_data: Any) -> tuple[TaskInputs, ArithmeticObservations, TaskPairs]:
        """Prepare the shared input tensors for the task from the raw dataset split."""
        def read_arithmetic_row(row: dict[str, Any]) -> ArithmeticObservation:
            """Read a single row from the raw dataset and return an ArithmeticObservation."""
            prompt = row['prompt']
            operand1 = int(row['operand1'])
            operand2 = int(row['operand2'])

            if int(row['label']) != (operand1 + operand2):
                raise ValueError(f"Label mismatch: {row['label']} != {operand1} + {operand2}")

            return ArithmeticObservation(
                prompt=prompt,
                operand1=operand1,
                operand2=operand2,
            )

        _pairs: list[Mapping[str, int]] = []
        _observations: list[ArithmeticObservation] = []
        for example_id, row in enumerate(raw_data):
            # Only use rows that have `num_digits = 2`
            if int(row['num_digit']) != self.num_digits:
                continue

            # Import the base prompt and its labels
            obs = read_arithmetic_row(row)
            base_index = len(_observations)
            _observations.append(obs)

            # Import the task counterfactuals
            for family_id, cf in enumerate(self.counterfactuals):
                source = read_arithmetic_row(row[f'{cf}_counterfactual'])

                source_index = len(_observations)
                _observations.append(source)

                # Create an intervention pair
                _pairs.append(dict(
                    family_id=family_id,
                    example_id=example_id,
                    base_index=base_index,
                    source_index=source_index,
                ))

        prompts = [obs.prompt for obs in _observations]
        operand1 = [obs.operand1 for obs in _observations]
        operand2 = [obs.operand2 for obs in _observations]
        ops_carry = [obs.carry for obs in _observations]

        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # * Prepare input data
        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        inputs = TaskInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            offset_mapping=offset_mapping,
        )

        # * Prepare pairs data
        observations = ArithmeticObservations(
            prompts=prompts,
            operand1=operand1,
            operand2=operand2,
            answers=ops_carry,
        )

        pairs = TaskPairs(
            family_id=torch.tensor([p['family_id'] for p in _pairs]),
            example_id=torch.tensor([p['example_id'] for p in _pairs]),
            base_index=torch.tensor([p['base_index'] for p in _pairs]),
            source_index=torch.tensor([p['source_index'] for p in _pairs]),
        )

        return inputs, observations, pairs

    def _prepare_activations(
        self,
        inputs: TaskInputs,
        subject: LMSubject,
        observations: ArithmeticObservations
    ) -> TaskActivations:
        """Prepare the model's activations for the task from the shared inputs."""
        # Build the two declared cut positions
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        offset_mapping = inputs.offset_mapping

        if offset_mapping is None:
            raise ValueError("Offset mapping is required to compute cut positions for the Arithmetic task.")

        operand_token_ids = []
        for observation, offsets in zip(observations, offset_mapping):
            matches = list(re.finditer(r"\b\d+\b", observation.prompt))
            if len(matches) < 2:
                raise ValueError(f"Could not find two operands in prompt: {observation.prompt}")

            match = matches[-1]

            if int(match.group()) != observation.operand2:
                raise ValueError(f"Operand2 mismatch: {match.group()} != {observation.operand2}")

            operand_token_idx = next(
                i
                for i, (start, end) in enumerate(offsets)
                if start <= match.end() - 1 < end
            )
            operand_token_ids.append(operand_token_idx)

        cut_second_operand = torch.tensor(operand_token_ids, dtype=torch.long).unsqueeze(-1)

        sequence_indices = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        last_tok_positions = sequence_indices.masked_fill(
            ~attention_mask.bool(), -1
        ).amax(dim=-1, keepdim=True) # (b, 1)

        positions = {
            "op2_last": cut_second_operand,
            "last": last_tok_positions,
        }
        cut_positions = torch.stack([
            positions[name] for name in self.position_names
        ], dim=1) # (b, 2, 1)

        activations = record_activations(
            inputs,
            subject,
            cut_positions,
            self.record_batch_size
        )

        return TaskActivations(
            names=self.position_names,
            positions=cut_positions,
            values=activations,
        )

    def _prepare_outcomes(
        self,
        inputs: TaskInputs,
        subject: LMSubject,
        observations: ArithmeticObservations,
        pairs: TaskPairs
    ) -> tuple[TaskOutcomes, torch.Tensor, TaskInfo]:
        """Prepare the outcomes for the arithmetic task."""

        def _add(op1: int, op2: int, carry: int) -> str:
            """Returns the result of the addition of operand1 and operand2 with carry control."""
            tens = op1 // 10 + op2 // 10
            ones = (op1 + op2) % 10
            return str(10 * (tens + carry) + ones)

        # Build the two possible outcome labels: [ops no-carry, ops with-carry]
        labels = [
            lbl
            for op1, op2 in zip(observations.operand1, observations.operand2)
            for lbl in [_add(op1, op2, carry=0), _add(op1, op2, carry=1)]
        ]

        answers = torch.tensor(observations.answers, dtype=torch.long)
        targets = answers[pairs.source_index]

        # * Prepare outcomes data
        encoded = self.tokenizer(
            labels,
            padding=True,
            padding_side="right",
            add_special_tokens=False,
            return_tensors="pt",
        )

        horizon = encoded["input_ids"].shape[1]
        outcome_ids = encoded["input_ids"].view(-1, self.num_outcomes, horizon)
        outcome_mask = encoded["attention_mask"].view(-1, self.num_outcomes, horizon)

        # Record the subject behaviour
        behaviour = record_behavior(
            inputs,
            subject,
            outcome_ids,
            outcome_mask,
            self.record_batch_size
        )

        correct = behaviour.argmax(dim=-1) == answers

        if correct.sum() < self.quality_gate * len(correct):
            raise ValueError(f"Quality gate failed: {correct.sum().item()} / {len(correct)} examples passed the quality gate, which is below the threshold of {self.quality_gate * 100:.1f}%")

        # NOTE: For arithmetic state_features = base_behaviour
        outcomes = TaskOutcomes(
            outcome_ids=outcome_ids[correct],
            outcome_mask=outcome_mask[correct],
            state_features=behaviour[correct],
            base_behaviour=behaviour[correct],
            correct_indices=correct.nonzero(as_tuple=True)[0],
        )

        info = {
            'total_examples': len(correct),
            'correct_examples': correct.sum().item(),
            'num_filtered': (~correct).sum().item(),
        }

        return outcomes, targets, info