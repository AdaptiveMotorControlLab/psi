
import re
from typing import Any, Mapping

import torch
from dataclasses import dataclass
from torch.nn.functional import one_hot
from transformers import TokenizersBackend

from psi.data.task import TaskPreparation
from psi.data.types import (
    TaskActivations,
    TaskObservations,
    TaskPairs,
    TaskInputs,
    TaskOutcomes,
    TaskInfo,
)

from psi.model.subject import LMSubject
from psi.data.utils import record_activations, record_behavior

@dataclass(frozen=True)
class MCQAObservation:
    """A (programmable) single observation for the MCQA task.

    prompt: The input question prompt to the model
    choices: The vocabulary of possible answer choices for the question
    answer_position: The position of the correct answer choice in the choices tuple
    """

    prompt: str
    choices: tuple[str, ...]
    answer_position: int

    @property
    def answer_symbol(self) -> str:
        """Return the correct answer symbol for this observation."""
        return self.choices[self.answer_position]

@dataclass(frozen=True)
class MCQAObservations(TaskObservations):
    """A collection of MCQA observations for a given dataset split."""
    prompts: list[str]
    answers: list[int]
    symbols: list[str]
    choices: list[tuple[str, ...]]

    def __iter__(self):
        for prompt, choices, answer in zip(self.prompts, self.choices, self.answers):
            yield MCQAObservation(prompt, choices, answer)

class MCQATask(TaskPreparation):
    """Multiple Choice Question Answering (MCQA) task preparation."""

    task_name = 'mcqa'

    ALPHABET = tuple(" " + letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self, config: dict[str, Any], tokenizer: TokenizersBackend):
        super().__init__(config, tokenizer)

        self.num_choices = int(config['num_choices'])

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return the names of the variables in this task."""
        return ('x_order', 'o_answer')

    @property
    def position_names(self) -> tuple[str, ...]:
        """Return the names of the cut positions in this task."""
        return ('last_token', 'correct_symbol', 'correct_symbol_period')

    @property
    def task_metadata(self) -> dict[str, Any]:
        """Return additional metadata for the task."""
        return {
            "num_choices": self.num_choices,
        }

    @property
    def task_observation(self) -> type[MCQAObservations]:
        """Return the type of the task observations for this task."""
        return MCQAObservations

    @property
    def num_outcomes(self) -> int:
        """The number of possible outcomes for the chosen variable."""
        if self.variable == 'x_order':
            return self.num_choices
        elif self.variable == 'o_answer':
            return len(self.ALPHABET)
        else:
            raise ValueError(f"Invalid variable: {self.variable}. Must be 'x_order' or 'o_answer'.")

    def identity(self, subject: LMSubject) -> str:
        """Return a unique fingerprint for this task preparation and subject."""
        subject_hash = subject.model_fingerprint
        counterfactuals = "_".join(self.counterfactuals)

        return f"{self.task_name}_{self.dataset}_{self.subset}_{self.num_choices}_{self.max_length}_{counterfactuals}_{subject_hash}_{self.filtered}_v{self.cache_version}"

    def _prepare_input(self, raw_data) -> tuple[TaskInputs, TaskObservations, TaskPairs]:
        """Prepare the shared input data for a single split of the dataset."""
        def read_mcqa_row(row) -> MCQAObservation:
            """Read a single row of the MCQA dataset and return the prompt, choices, and answer position."""
            prompt = row['prompt']
            choices = tuple(' ' + str(lbl) for lbl in row['choices']['label'])
            answer_position = int(row['answerKey'])

            # Validate the data loading
            assert len(choices) == self.num_choices, f"Expected {self.num_choices} choices, but got {len(choices)}"
            assert 0 <= answer_position < self.num_choices, f"Answer position {answer_position} is out of bounds for {self.num_choices} choices"

            return MCQAObservation(prompt, choices, answer_position)

        _pairs: list[Mapping[str, int]] = []
        _observations: list[MCQAObservation] = []
        for example_id, row in enumerate(raw_data):
            # Filter out rows that do not have the expected number of choices
            if len(row['choices']['label']) != self.num_choices:
                continue

            # First let's import the base prompt and true labels/answer-positions
            base = read_mcqa_row(row)
            base_index = len(_observations)
            _observations.append(base)

            # Then let's import the counterfactuals
            for family_id, cf in enumerate(self.counterfactuals):
                source = read_mcqa_row(row[f'{cf}_counterfactual'])

                source_index = len(_observations)
                _observations.append(source)

                # Create an intervention pair
                _pairs.append(dict(
                    base_index=base_index,
                    source_index=source_index,
                    family_id=family_id,
                    example_id=example_id
                ))

        prompts = [obs.prompt for obs in _observations]
        choices = [obs.choices for obs in _observations]
        symbols = [obs.answer_symbol for obs in _observations]
        answers = [obs.answer_position for obs in _observations]

        encoded = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
        )

        # * Prepare input data
        input_ids = encoded['input_ids']
        offset_mapping = encoded['offset_mapping']
        attention_mask = encoded['attention_mask']
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 0)

        inputs = TaskInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            offset_mapping=offset_mapping,
        )

        # * Prepare pairs data
        observations = MCQAObservations(
            prompts=prompts,
            choices=choices,
            answers=answers,
            symbols=symbols,
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
        observations: MCQAObservations,
    ) -> TaskActivations:
        """Prepare the cached activations for a single split of the dataset."""
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        offset_mapping = inputs.offset_mapping

        if offset_mapping is None:
            raise ValueError("Offset mapping is required to compute cut positions for the MCQA task.")

        # Prepare cut token positions
        sequence_indices = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        last_tok_positions = sequence_indices.masked_fill(
            ~attention_mask.bool(), -1
        ).amax(dim=-1, keepdim=True) # (b, 1)

        # Find the token positions of the correct answer symbols in the prompts
        symbol_token_ids = []
        for observation, offsets in zip(observations, offset_mapping):
            symbol = observation.answer_symbol.strip()
            match = next(
                (
                    candidate
                    for candidate in re.finditer(r"\b[A-Z]\b", observation.prompt)
                    if candidate.group() == symbol
                ),
                None,
            )

            if match is None:
                raise ValueError(f"Could not find symbol {symbol} in prompt: {observation.prompt}")

            operand_token_idx = next(
                i
                for i, (start, end) in enumerate(offsets)
                if start <= match.end() - 1 < end
            )
            symbol_token_ids.append(operand_token_idx)

        correct_symb_positions = torch.tensor(symbol_token_ids, dtype=torch.long).unsqueeze(-1)
        correct_symb_period_positions = correct_symb_positions + 1
        if (
            (correct_symb_period_positions >= input_ids.shape[1]).any()
            or not attention_mask.gather(1, correct_symb_period_positions).bool().all()
        ):
            raise ValueError("Correct-symbol period position is outside an MCQA prompt.")

        positions = {
            "last_token": last_tok_positions,
            "correct_symbol": correct_symb_positions,
            "correct_symbol_period": correct_symb_period_positions
        }

        cut_positions = torch.stack([
            positions[name] for name in self.position_names
        ], dim=1) # (b, 3, 1)

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
        observations: MCQAObservations,
        pairs: TaskPairs
    ) -> tuple[TaskOutcomes, torch.Tensor, TaskInfo]:
        """Prepare a single split of the dataset for the chosen variable."""
        # * Program the declared answers and targets for the variable of interest
        src_idx = pairs.source_index

        symbols = observations.symbols
        answers = observations.answers

        if self.variable == 'x_order':
            outcomes = observations.choices
            answer_indices = torch.tensor(answers)
            targets = answer_indices[src_idx]

        elif self.variable == 'o_answer':
            outcomes = [self.ALPHABET for _ in range(len(symbols))]
            answer_indices = torch.tensor([
                self.ALPHABET.index(sym)
                for sym in symbols
            ])
            targets = answer_indices[src_idx]
        else:
            raise ValueError(f"Invalid variable: {self.variable}. Must be 'x_order' or 'o_answer'.")

        labels = [lbl for out in outcomes for lbl in out]

        # * Prepare outcomes data
        encoded = self.tokenizer(
            labels, # type: ignore
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        horizon = encoded['input_ids'].shape[1]
        num_outcomes = self.num_outcomes
        outcome_ids = encoded['input_ids'].view(len(outcomes), num_outcomes, horizon)
        outcome_mask = encoded['attention_mask'].view(len(outcomes), num_outcomes, horizon)

        # Record the subject activations and behaviour for the input data
        behaviour = record_behavior(
            inputs,
            subject,
            outcome_ids,
            outcome_mask,
            self.record_batch_size
        )

        # Quality gate: filter out incorrect behaviour from the observations
        correct = behaviour.argmax(dim=-1) == answer_indices
        
        if self.correctness_filter:
            if correct.sum() < self.quality_gate * len(correct):
                raise ValueError(f"Quality gate failed: {correct.sum().item()} / {len(correct)} examples passed the quality gate, which is below the threshold of {self.quality_gate * 100:.1f}%")
            keep = correct
        else:
            keep = torch.ones_like(correct, dtype=torch.bool)
        
        gold = one_hot(
            answer_indices,
            num_classes=self.num_outcomes
        ).to(behaviour.dtype)

        # NOTE: For MCQA state_features = base_behaviour
        outcomes = TaskOutcomes(
            outcome_ids=outcome_ids[keep],
            outcome_mask=outcome_mask[keep],
            state_features=behaviour[keep],
            base_behaviour=behaviour[keep],
            correct_indices=keep.nonzero(as_tuple=True)[0],
            gold_state_features=gold[keep],
        )

        info = {
            'total_examples': len(keep),
            'correct_examples': correct.sum().item(),
            'num_filtered': (~keep).sum().item(),
            'correctness_filter': self.correctness_filter,
        }

        return outcomes, targets, info
