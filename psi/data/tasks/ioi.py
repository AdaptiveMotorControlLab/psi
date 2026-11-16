import re
from typing import Any, Mapping

import torch

from dataclasses import dataclass
from transformers import TokenizersBackend
from torch.nn.functional import one_hot

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
class IOIObservation:
    """A single observation for the IOI task."""
    prompt: str
    answer_token: str
    names_in_stable_order: tuple[str, str]
    names_in_prompt_order: tuple[str, str]
    answer_position_in_prompt_order: int
    answer_position_in_stable_order: int

@dataclass(frozen=True)
class IOIObservations(TaskObservations):
    """Observations for the IOI task."""
    answers_token: list[str]
    names_in_stable_order: list[tuple[str, str]]
    names_in_prompt_order: list[tuple[str, str]]
    answers_position_in_prompt_order: list[int]
    answers_position_in_stable_order: list[int]

class IOITask(TaskPreparation):
    """The IOI task (Indirect Object Identification) is a task where
    the model is given a sentence with two names and must identify
    which name is the indirect object of the sentence.
    """

    task_name = "ioi"

    def __init__(self, config: dict[str, Any], tokenizer: TokenizersBackend):
        super().__init__(config, tokenizer)

        # Make sure that max_length is 32 for this task
        if self.max_length != 32:
            raise ValueError(f"IOI task requires max_length=32, got {self.max_length}.")

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return the names of the variables in this task."""
        return ("output_position", "output_token")

    @property
    def position_names(self) -> tuple[str, ...]:
        """The IOI tasks records from all positions in the prompt."""
        return ('all',)

    @property
    def task_observation(self) -> type[TaskObservations]:
        """Return the dataclass for the task observations."""
        return IOIObservations

    @property
    def num_outcomes(self) -> int:
        """The IOI semantics are:
            - [p(first name | x), p(second name | x)]
            - [p(first position | x), p(second position | x)]
        """
        return 2

    def identity(self, subject: LMSubject) -> str:
        """Return a unique fingerprint for this task preparation and subject."""
        subject_hash = subject.model_fingerprint
        counterfactuals = "_".join(self.counterfactuals)

        return f"{self.task_name}_{self.dataset}_{self.subset}_{self.max_length}_{counterfactuals}_{subject_hash}_v{self.cache_version}"

    def _prepare_input(self, raw_data: list[dict[str, Any]]) -> tuple[TaskInputs, TaskObservations, TaskPairs]:
        """Prepare the inputs for the IOI task."""

        def read_ioi_row(row: dict[str, Any]) -> IOIObservation:
            """Read a single observation from the raw data."""
            prompt = row["prompt"]
            choices = tuple(row["choices"])
            answer_key = row["answerKey"]
            answer_tok = ' ' + choices[answer_key]

            # Find the character position of the two names in the prompt and
            # store them in the order they appear in the prompt.
            name_pattern = re.compile(r"\b(" + "|".join(map(re.escape, choices)) + r")\b")
            first, second, *rest = name_pattern.findall(prompt)

            # Input validation
            if first == second:
                raise ValueError(f"Expected different first/second names in prompt: {prompt}. Got: {first} and {second}.")

            # NOTE: Add leading spaces to the names to ensure that they are tokenized correctly
            names_in_prompt_order = (' ' + first, ' ' + second)
            names_in_stable_order = tuple(sorted(names_in_prompt_order))

            answer_position_in_prompt_order = names_in_prompt_order.index(answer_tok)
            answer_position_in_stable_order = names_in_stable_order.index(answer_tok)

            return IOIObservation(
                prompt=prompt,
                answer_token=answer_tok,
                names_in_stable_order=names_in_stable_order,
                names_in_prompt_order=names_in_prompt_order,
                answer_position_in_prompt_order=answer_position_in_prompt_order,
                answer_position_in_stable_order=answer_position_in_stable_order,
            )

        _pairs: list[Mapping[str, int]] = []
        _observations: list[IOIObservation] = []
        for example_id, row in enumerate(raw_data):
            # First let's import the base prompt and true labels/answer-positions
            base = read_ioi_row(row)
            base_index = len(_observations)
            _observations.append(base)

            # Then let's import the counterfactuals
            for family_id, cf in enumerate(self.counterfactuals):
                source = read_ioi_row(row[f'{cf}_counterfactual'])

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
        answers_token = [obs.answer_token for obs in _observations]
        names_in_stable_order = [obs.names_in_stable_order for obs in _observations]
        names_in_prompt_order = [obs.names_in_prompt_order for obs in _observations]
        answers_position_in_prompt_order = [obs.answer_position_in_prompt_order for obs in _observations]
        answers_position_in_stable_order = [obs.answer_position_in_stable_order for obs in _observations]

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
        observations = IOIObservations(
            prompts=prompts,
            answers_token=answers_token,
            names_in_stable_order=names_in_stable_order,
            names_in_prompt_order=names_in_prompt_order,
            answers_position_in_prompt_order=answers_position_in_prompt_order,
            answers_position_in_stable_order=answers_position_in_stable_order,
        )

        pairs = TaskPairs(
            family_id=torch.tensor([p['family_id'] for p in _pairs]),
            example_id=torch.tensor([p['example_id'] for p in _pairs]),
            base_index=torch.tensor([p['base_index'] for p in _pairs]),
            source_index=torch.tensor([p['source_index'] for p in _pairs]),
        )

        return inputs, observations, pairs

    def _prepare_activations(self,
        inputs: TaskInputs,
        subject: LMSubject,
        observations: IOIObservations,
    ) -> TaskActivations:
        """Prepare the cached activations for a single split of the dataset."""
        input_ids = inputs.input_ids

        # NOTE: For the IOI task we record from all prompt positions
        cut_positions = torch.arange(
            input_ids.shape[1]
        ).expand_as(input_ids).unsqueeze(1) # (b, 1, max_length = 32)

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
        observations: IOIObservations,
        pairs: TaskPairs
    ) -> tuple[TaskOutcomes, torch.Tensor, TaskInfo]:
        pair_count = len(pairs.base_index)

        if self.variable == 'output_position':
            # The semantic program for this variable is:
            #  - [p(first position | x), p(second position | x)]
            outcomes = observations.names_in_prompt_order
            answers = torch.tensor(observations.answers_position_in_prompt_order)

            token_signal = torch.ones(pair_count)
            position_signal = torch.where(
                answers[pairs.base_index] == answers[pairs.source_index],
                +1.0,
                -1.0,
            )

            # NOTE: For the `position` semantics there is one stable semantic frame
            # for all examples/counterfacturls, so every donor is eligible for every recipient.
            transport_group = None

        elif self.variable == 'output_token':
            # The semantic program for this variable is:
            #  - [p(first name | x), p(second name | x)]
            outcomes = observations.names_in_stable_order
            answers = torch.tensor(observations.answers_position_in_stable_order)

            token_condition = torch.tensor([
                observations.answers_token[pair.base_index] == observations.answers_token[pair.source_index]
                for pair in pairs
            ])

            position_signal = torch.ones(pair_count)
            token_signal = torch.where(
                token_condition,
                +1.0,
                -1.0,
            )

            # NOTE: For the `token` semantics there are many semantic frame of references,
            # one for every unique pair of names. So we group the examples by their stable name-pair.
            # Buld a mapping from name-pair to group index
            name_pair_to_group = {pair: i for i, pair in enumerate(sorted(set(outcomes)))}
            transport_group = torch.tensor([name_pair_to_group[pair] for pair in outcomes])

        else:
            raise ValueError(f"Unknown variable {self.variable}.")

        # NOTE: MIB evaluation check for the `Name` without a leading space, which
        # results in different tokens for GPT-2. Hence we prepare two types of labels:
        # one with the leading space for correctness-gate, and one without for evaluation.
        semantic_labels = [lbl for out in outcomes for lbl in out]
        official_labels = [lbl.strip() for out in outcomes for lbl in out]
        
        targets = self._ioi_target_model(token_signal, position_signal)

        # * Prepare outcomes data
        semantic_encoded = self.tokenizer(
            semantic_labels,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        official_encoded = self.tokenizer(
            official_labels,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        num_outcomes = self.num_outcomes
        
        semantic_horizon = semantic_encoded['input_ids'].shape[1]
        semantic_outcome_ids = semantic_encoded['input_ids'].view(len(outcomes), num_outcomes, semantic_horizon)
        semantic_outcome_mask = semantic_encoded['attention_mask'].view(len(outcomes), num_outcomes, semantic_horizon)

        official_horizon = official_encoded['input_ids'].shape[1]
        official_outcome_ids = official_encoded['input_ids'].view(len(outcomes), num_outcomes, official_horizon)
        official_outcome_mask = official_encoded['attention_mask'].view(len(outcomes), num_outcomes, official_horizon)

        # The state features are used to calibrate the reading q(x),
        # hence we use the proper semantic labels
        state_features = record_behavior(
            inputs,
            subject,
            semantic_outcome_ids,
            semantic_outcome_mask,
            self.record_batch_size
        )
        
        # Base behaviour is used during interventions, hence we use the
        # MIB official labels for correctness evaluation.
        base_behaviour = record_behavior(
            inputs,
            subject,
            official_outcome_ids,
            official_outcome_mask,
            self.record_batch_size
        )

        # Quality gate: filter out incorrect behaviour from the observations
        greedy_token = subject.greedy(
            {
                'input_ids': inputs.input_ids,
                'position_ids': inputs.position_ids,
                'attention_mask': inputs.attention_mask,
            },
            self.record_batch_size,
        )

        # Correctness gate uses the semantic labels since the model
        # is expected to answer with a leading space for the names.
        expected_token = semantic_outcome_ids[
            torch.arange(len(semantic_outcome_ids)),
            answers,
            0
        ]

        correct = greedy_token.eq(expected_token)
        accuracy = correct.float().mean().item()

        if accuracy < self.quality_gate:
            raise ValueError(
                f"Subject accuracy {accuracy:.2%} is below the quality gate {self.quality_gate:.2%}."
            )

        eligible = correct

        # Make sure no groups gets reduced to a singleton group after
        # filtering out incorrect examples
        if transport_group is not None:
            present, counts = transport_group[correct].unique(return_counts=True)
            valid_groups = present[counts > 1]
            eligible = correct & torch.isin(transport_group, valid_groups)
        
        # Construct a transport gate to represent tau(x_b, q_s), i.e. a tensor of
        # shape (num_examples, num_outputs, num_outputs), with num_outputs = 2, which
        # concretely represents which semantic target to expect for each donor.
        ones = torch.tensor(1.0)
        same_margin = self._ioi_target_model(ones, ones)
        
        if self.variable == 'output_position':
            changed_marging = self._ioi_target_model(ones, -ones)
        elif self.variable == 'output_token':
            changed_marging = self._ioi_target_model(-ones, ones)
        else:
            raise ValueError(f"Unknown variable {self.variable}.")
        source_classes = torch.arange(2).unsqueeze(0) # (1, 2)
        answer_is_same = source_classes == answers[:, None] # (num_examples, 2)
        
        io_marging = torch.where(answer_is_same, same_margin, changed_marging)
        p_io = io_marging.sigmoid().unsqueeze(-1) # (num_examples, 2, 1)
        base_io = one_hot(answers, num_classes=2).float()[:, None, :] # (num_examples, 1, 2)
        
        transport_gate = (
            base_io * p_io +
            (1 - base_io) * (1 - p_io)
        ) # (num_examples, 2, 2) representing tau(x_b, q_s)

        # NOTE: For MCQA state_features = base_behaviour
        outcomes = TaskOutcomes(
            outcome_ids=official_outcome_ids[eligible],
            outcome_mask=official_outcome_mask[eligible],
            state_features=state_features[eligible],
            base_behaviour=base_behaviour[eligible],
            correct_indices=eligible.nonzero(as_tuple=True)[0],
            transport_gate=transport_gate[eligible],
            transport_group=transport_group[eligible] if transport_group is not None else None,
        )

        info = {
            'total_examples': len(correct),
            'correct_examples': correct.sum().item(),
            'eligible_examples': eligible.sum().item(),
            'num_filtered': (~eligible).sum().item(),
        }

        return outcomes, targets, info

    def _ioi_target_model(self, token_signal: torch.Tensor, position_signal: torch.Tensor) -> torch.Tensor:
        """Hard-coded target model for the IOI task."""
        BIAS = 0.048
        TOKEN_COEFF = 0.768
        POSITION_COEFF = 2.005
        return BIAS + TOKEN_COEFF * token_signal + POSITION_COEFF * position_signal
