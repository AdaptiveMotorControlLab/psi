
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from datasets import Dataset
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
class RavelObservation:
    """A single observation for the RAVEL task."""

    prompt: str
    entity: str

    country: str
    language: str
    continent: str
    attribute: str

    def value(self, attribute: str) -> str:
        match attribute:
            case 'Country':
                return self.country
            case 'Language':
                return self.language
            case 'Continent':
                return self.continent
            case _:
                raise ValueError(f"Invalid attribute: {attribute}. Must be one of 'Country', 'Language', or 'Continent'.")

    @property
    def answer(self) -> str | None:
        """Return the correct answer for this observation."""
        if self.attribute == 'wikipedia':
            return None # Only used for counterfactuals

        return self.value(self.attribute)

@dataclass(frozen=True)
class RavelObservations(TaskObservations):
    """A collection of RAVEL observations."""

    entities: list[str]
    countries: list[str]
    languages: list[str]
    continents: list[str]
    attributes: list[str]

    answers: list[str | None]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> RavelObservation:
        return RavelObservation(
            prompt=self.prompts[index],
            entity=self.entities[index],
            country=self.countries[index],
            language=self.languages[index],
            continent=self.continents[index],
            attribute=self.attributes[index],
        )

    def __iter__(self):
        for prompt, entity, country, language, continent, attribute in zip(
            self.prompts,
            self.entities,
            self.countries,
            self.languages,
            self.continents,
            self.attributes,
        ):
            yield RavelObservation(
                prompt=prompt,
                entity=entity,
                country=country,
                language=language,
                continent=continent,
                attribute=attribute,
            )

@dataclass(frozen=True)
class RavelOutcomeLibrary:
    tokens: torch.Tensor # (num_outcomes, )
    value_to_class: Mapping[str, int]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def class_of(self, value: str) -> int:
        return self.value_to_class[value]

    def token_of(self, value: str) -> int:
        return int(self.tokens[self.class_of(value)])

    def expand(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the expanded outcome library for a given number of examples."""
        ids = self.tokens[None, :, None].expand(count, -1, -1)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return ids, mask

class RavelTask(TaskPreparation):
    """The RAVEL task query the models to answer questions about
    the geographical and linguistical properties of given cities
    in the world. An example prompt might ask:
    People in New York City speak English. People in Tirana speak [correct ans: Albanian]

    There are no provided answer choices, and the model is expected to generate the correct answer.
    """

    task_name = "ravel"

    ATTRIBUTES = {
        "a_country": "Country",
        "a_continent": "Continent",
        "a_language": "Language",
    }

    def __init__(self, config: dict[str, Any], tokenizer: TokenizersBackend):
        super().__init__(config, tokenizer)

        self.entity_file = Path(config['entity_file'])
        self.max_examples = int(config.get('max_examples', 10000))
        self.selection_seed = int(config.get('selection_seed', 42))

        # Build Ravel outcome library
        entities = json.loads(self.entity_file.read_text())

        values = sorted({
            row[attribute]
            for row in entities.values()
            for attribute in self.ATTRIBUTES.values()
        })

        value_to_token = {
            value: self.tokenizer.encode(
                " " + value.strip(),
                add_special_tokens=False,
            )[0] for value in values
        }

        tokens = torch.tensor(
            sorted(set(value_to_token.values())),
            dtype=torch.long,
        )
        token_to_class = {
            int(token) : index
            for index, token in enumerate(tokens)
        }
        value_to_class = {
            value: token_to_class[token]
            for value, token in value_to_token.items()
        }

        self.library = RavelOutcomeLibrary(
            tokens=tokens,
            value_to_class=value_to_class,
        )

    @property
    def variable_names(self) -> tuple[str, ...]:
        return ("a_country", "a_continent", "a_language")

    @property
    def position_names(self) -> tuple[str, ...]:
        return ("last_token", "entity_last_token")

    @property
    def num_outcomes(self) -> int:
        """Return the number of outcomes in this task."""
        # The RAVEL task has a fixed set of possible outcomes, which are the unique values of the attributes in the dataset.
        return self.library.size

    @property
    def raw_splits(self) -> dict[str, str]:
        """Return a mapping from cache split names to raw dataset split names."""
        return {
            "train": "train",
            "validation": "val",
            "test": "test",
        }

    @property
    def task_observation(self) -> type[TaskObservations]:
        return RavelObservations

    @property
    def task_metadata(self) -> dict[str, Any]:
        """Return additional metadata for the task."""
        return {
            'entity_file': str(self.entity_file),
            'max_examples': self.max_examples,
            'selection_seed': self.selection_seed,
        }

    def identity(self, subject: LMSubject) -> str:
        """Return a unique identity string for this task configuration."""
        subject_hash = subject.model_fingerprint
        counterfactuals = "_".join(self.counterfactuals)

        return f"{self.task_name}_{self.dataset}_{self.max_length}_{self.max_examples}_{self.selection_seed}_{counterfactuals}_{subject_hash}_v{self.cache_version}"

    def _prepare_input(self, raw_data: Dataset) -> tuple[TaskInputs, RavelObservations, TaskPairs]:

        def read_ravel_row(row: dict[str, Any]) -> RavelObservation:
            prompt = row['prompt']
            entity = row['entity']
            country = row['Country']
            language = row['Language']
            continent = row['Continent']
            attribute = row['attribute']

            valid_attributes = set(self.ATTRIBUTES.values()) | {"wikipedia"}
            assert attribute in valid_attributes, f"Invalid attribute: {attribute}. Must be one of {list(valid_attributes)}."

            return RavelObservation(
                prompt=prompt,
                entity=entity,
                country=country,
                language=language,
                continent=continent,
                attribute=attribute,
            )

        _pairs: list[Mapping[str, int]] = []
        _observations: list[RavelObservation] = []

        # Extract max_examples from the raw data using the provided selection seed for reproducibility
        if self.max_examples is not None and len(raw_data) > self.max_examples:
            rng = torch.Generator().manual_seed(self.selection_seed)
            selected_indices = torch.randperm(len(raw_data), generator=rng)[:self.max_examples]
            raw_data = raw_data.select(selected_indices.tolist())

        row: dict
        for example_id, row in enumerate(raw_data): # type: ignore
            # filter out any rows that do not have all the required attributes defined
            if any(row[attr] is None for attr in self.ATTRIBUTES.values()):
                continue

            base = read_ravel_row(row)
            base_index = len(_observations)
            _observations.append(base)

            # Import the task counterfactuals: `attribute` & `wikipedia`
            for family_id, cf in enumerate(self.counterfactuals):
                source_row = row[f'{cf}_counterfactual']
                # A donor needs defined values for every supported variable;
                # otherwise its intervention target cannot be constructed.
                if any(source_row[attr] is None for attr in self.ATTRIBUTES.values()):
                    continue
                source = read_ravel_row(source_row)

                source_index = len(_observations)
                _observations.append(source)

                _pairs.append(dict(
                    family_id=family_id,
                    example_id=example_id,
                    base_index=base_index,
                    source_index=source_index,
                ))

        prompts = [obs.prompt for obs in _observations]
        answers = [obs.answer for obs in _observations]
        entities = [obs.entity for obs in _observations]
        countries = [obs.country for obs in _observations]
        languages = [obs.language for obs in _observations]
        continents = [obs.continent for obs in _observations]
        attributes = [obs.attribute for obs in _observations]

        encoded = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
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

        observations = RavelObservations(
            prompts=prompts,
            answers=answers,
            entities=entities,
            countries=countries,
            languages=languages,
            continents=continents,
            attributes=attributes,
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
        observations: RavelObservations,
    ) -> TaskActivations:
        """Prepare the model's activations for the task from the shared inputs."""
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        offset_mapping = inputs.offset_mapping

        if offset_mapping is None:
            raise ValueError("Offset mapping is required to compute cut positions for the RAVEL task.")

        # Prepare cut token positions
        sequence_indices = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        last_tok_positions = sequence_indices.masked_fill(
            ~attention_mask.bool(), -1
        ).amax(dim=-1, keepdim=True) # (b, 1)

        # Find the entity last token position
        entity_token_ids = []
        for observation, offsets in zip(observations, offset_mapping):
            match = next(
                (
                    candidate
                    for candidate in re.finditer(rf"\b{observation.entity}\b", observation.prompt)
                    if candidate.group() == observation.entity
                ),
                None,
            )

            if match is None:
                raise ValueError(f"Could not find entity {observation.entity} in prompt: {observation.prompt}")

            character = match.end() - 1
            token_index = next(
                index
                for index, (start, end) in enumerate(offsets)
                if start <= character < end
            )

            if token_index is None:
                raise ValueError(f"Entity '{observation.entity}' not found in prompt '{observation.prompt}'.")

            # Take the last token index of the entity
            entity_token_ids.append(token_index)
        entity_last_tok_positions = torch.tensor(entity_token_ids, dtype=torch.long).unsqueeze(-1) # (b, 1)

        positions = {
            "last_token": last_tok_positions,
            "entity_last_token": entity_last_tok_positions,
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
        observations: RavelObservations,
        pairs: TaskPairs,
    ) -> tuple[TaskOutcomes, torch.Tensor, TaskInfo]:
        """Prepare the outcomes for the task from the model's outputs."""
        # Measure the model outcome library for the provided subject
        outcome_ids, outcome_mask = self.library.expand(len(inputs))

        variable_attribute = self.ATTRIBUTES[self.variable]

        # * Measure the model's reading of the attribute
        # Evaluate the model on standardized feature prompts
        feature_prompts = [
            f'Q: What is the {variable_attribute.lower()} of {entity}?\nA:'
            for entity in observations.entities
        ]

        feat_encoded = self.tokenizer(
            feature_prompts,
            padding=True,
            return_tensors="pt",
        )

        feat_input_ids = feat_encoded["input_ids"]
        feat_attention_mask = feat_encoded["attention_mask"]
        feat_position_ids = feat_attention_mask.long().cumsum(dim=-1) - 1
        feat_position_ids.masked_fill_(feat_attention_mask == 0, 0)

        feat_inputs = TaskInputs(
            input_ids=feat_input_ids,
            attention_mask=feat_attention_mask,
            position_ids=feat_position_ids,
        )

        state_features = record_behavior(
            feat_inputs,
            subject,
            outcome_ids,
            outcome_mask,
            self.record_batch_size
        )

        # * Measure the base behavior of the model
        base_behaviour = record_behavior(
            inputs,
            subject,
            outcome_ids,
            outcome_mask,
            self.record_batch_size
        )

        # The transport gate is not trivial: we only map the base behavior
        # to the source behavior when the queried attribute is the same as the counterfactual attribute. Otherwise, we set the transport gate to 0.
        transport_gate = torch.tensor([
            attribute == variable_attribute
            for attribute in observations.attributes
        ])

        # Compute the quality gate by checking that the model's
        # first-token prediction matches the expected answer
        greedy_token = subject.greedy(
            {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "position_ids": inputs.position_ids,
            },
            batch_size=self.record_batch_size
        )

        expected_token = torch.tensor([
            self.library.token_of(answer)
            if answer is not None else -1
            for answer in observations.answers
        ])

        special = expected_token < 0
        checkable = ~special
        correct = special | (greedy_token == expected_token)

        if not checkable.any():
            raise ValueError("RAVEL has no observations with checkable answers.")

        accuracy = correct[checkable].float().mean().item()
        if accuracy < self.quality_gate:
            raise ValueError(
                f"RAVEL task quality gate failed: {accuracy:.2%} < {self.quality_gate:.2%}. "
                "The model's base behavior is not sufficiently accurate to perform the task."
            )

        outcomes = TaskOutcomes(
            outcome_ids=outcome_ids[correct],
            outcome_mask=outcome_mask[correct],
            state_features=state_features[correct],
            base_behaviour=base_behaviour[correct],
            correct_indices=correct.nonzero(as_tuple=True)[0],
            transport_gate=transport_gate[correct]
        )

        # * Prepare the targets for the intervention pairs
        expected: list[str] = []
        for pair in pairs:
            base = observations[pair.base_index]
            source = observations[pair.source_index]

            exp_val = source.value(variable_attribute)\
                if base.attribute == variable_attribute else\
                    base.value(base.attribute)
            expected.append(exp_val)

        # Get the first-token representation of the expected outcomes
        targets = torch.tensor([
            self.library.class_of(value)
            for value in expected
        ]).long()

        info = {
            'total_examples': len(correct),
            'correct_examples': correct.sum().item(),
            'num_filtered': (~correct).sum().item(),
        }

        return outcomes, targets, info
