"""Materialize declared prompt programs through the existing task contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re

import torch
import yaml
from torch.nn.functional import one_hot

from psi.data.task import TaskPreparation
from psi.data.types import (
    TaskInputs,
    TaskObservations,
    TaskPairs,
    TaskActivations,
    TaskOutcomes,
    TaskCache,
)
from psi.data.utils import record_activations, record_behavior
from psi.model.transport import SemanticTransport
from psi.program import At, Gated, Table, Program, column, digest
from psi.prompts import Prompts


@dataclass(frozen=True)
class ProgramObservations(TaskObservations):
    rows: list[dict]


class ProgramTask(TaskPreparation):
    """One task implementation for prompt text, declared answers and a rule."""

    task_name = "program"

    @classmethod
    def from_objects(
        cls,
        prompts,
        program,
        tokenizer,
        *,
        position = 'last_token',
        **config,
    ):
        """Construct a task from in-memory declarations and preparation options."""
        return cls(
            config, tokenizer, prompts=prompts, program=program, position=position
        )

    @classmethod
    def prepare_query(
        cls,
        row,
        program,
        tokenizer,
        subject,
        *,
        position = 'last_token',
        max_length = 128,
        capture = False,
    ):
        """Prepare a real one-row query using the same encoding as task caches.

        Reading needs the continuation batch. Swaps and steering also request
        the selected activation through capture=True.
        """
        inputs = cls.tokenize([row], tokenizer, max_length)
        ids, mask = cls.encode_answers(
            program.answers.resolve([row]),
            tokenizer,
            token_ids=program.compatibility.get("token_ids"),
        )
        batch = {
            "input_ids": inputs.input_ids,
            "position_ids": inputs.position_ids,
            "attention_mask": inputs.attention_mask,
            "outcome_ids": ids,
            "outcome_mask": mask,
        }
        if capture:
            positions = cls.resolve_positions(inputs, [row], {"selected": position})
            batch["cut_positions"] = positions[:, 0]
            batch["activation"] = record_activations(inputs, subject, positions, 1)[
                :, 0
            ]

        return {name: value.to(subject.device) for name, value in batch.items()}

    @property
    def variable_names(
        self,
    ):
        """The generic task exposes one declared program variable."""
        return ("program",)

    @property
    def position_names(
        self,
    ):
        """Names of the sites retained in this task's activation bank."""
        return tuple(self.positions)

    @property
    def num_outcomes(
        self,
    ):
        """Number of answer slots validated during construction."""
        return self._num_outcomes

    @property
    def task_observation(
        self,
    ):
        """Observation container carrying prompt text and metadata."""
        return ProgramObservations

    def __init__(
        self,
        config,
        tokenizer,
        *,
        prompts = None,
        program = None,
        position = 'last_token',
    ):
        """Configure generic preparation from YAML or in-memory declarations.

        The data configuration sets cache_root, max_length, quality_gate,
        correctness_filter and record_batch_size. With supplied Prompts and
        Program objects, no dataset is downloaded. The tokenizer must be fast.
        """
        self.prompts = prompts or Prompts.from_hf(
            config["dataset"],
            config.get("subset"),
            text=config.get("text", "prompt"),
            options=config.get("options"),
            gold=config.get("gold"),
            splits=config.get("splits"),
            seed=config.get("seed", 11),
            limit=config.get("limit"),
        )
        self.program = program or Program.from_dict(config)
        if self.program.gold is None and self.prompts.gold is not None:
            self.program = replace(self.program, gold=self.prompts.gold)
        self.position = position
        self.positions = dict(self.prompts.positions)
        if position != "last_token":
            self.positions["selected"] = position
        self.selected_position = (
            "last_token" if position == "last_token" else "selected"
        )
        defaults = dict(
            cache_root="data/psi",
            dataset="declared-prompts",
            variable="program",
            max_length=128,
            quality_gate=0.0,
            correctness_filter=False,
            counterfactuals=self.prompts.families,
        )
        defaults.update(config)
        super().__init__(defaults, tokenizer)
        if self.correctness_filter and not (self.program.gold or self.prompts.gold):
            raise ValueError("correctness_filter=True requires a gold column.")
        answers = self.program.answers.resolve(self.prompts.all_rows)
        self._num_outcomes = len(answers[0])

    def identity(
        self,
        subject,
    ):
        """Hash data, rules, positions, tokenizer, filters and frozen weights.

        Weight content identifies anonymous models and local weight edits. The
        tokenizer identity excludes transient call settings and retains the
        declared padding, truncation and special-token choices.
        """
        model = subject.model
        hasher = hashlib.sha256()
        for name, parameter in model.state_dict().items():
            tensor = parameter.detach().cpu().contiguous()
            hasher.update(name.encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

        tokenizer = json.loads(self.tokenizer.backend_tokenizer.to_str())
        tokenizer.pop("padding", None)
        tokenizer.pop("truncation", None)

        positions = {}
        for name, position in self.positions.items():
            if isinstance(position, At):
                positions[name] = {"regex": position.regex}
            elif callable(position):
                positions[name] = [
                    position(row["prompt"]) for row in self.prompts.all_rows
                ]
            else:
                positions[name] = position

        state = {
            "positions": positions,
            "prompts": self.prompts.fingerprint,
            "program": self.program.fingerprint(self.prompts.all_rows),
            "tokenizer": tokenizer,
            "truncation_side": self.tokenizer.truncation_side,
            "pad_token_id": self.tokenizer.pad_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "padding_side": self.tokenizer.padding_side,
            "model": hasher.hexdigest(),
            "model_config": model.config.to_dict(),
            "max_length": self.max_length,
            "quality_gate": self.quality_gate,
            "correctness_filter": self.correctness_filter,
            "cache_version": self.cache_version,
        }
        return "program_" + digest(state)

    def materialize(
        self,
        subject,
    ):
        """Prepare each split through the shared task cache contract.

        Inputs and outcomes are shared across layers. Each residual layer gets
        its own activation bank. Existing files in this task's identity are
        reused; evaluation pairs retain their original row identities.
        """
        root = self.root / self.identity(subject)
        activation_root = root / "activations" / subject.cut.fingerprint
        info_path = root / self.variable / "manifest.yaml"
        info = yaml.safe_load(info_path.read_text()) if info_path.exists() else {}
        for split, rows in self.prompts.splits.items():
            shared = root / split
            variable = root / self.variable / split
            shared.mkdir(parents=True, exist_ok=True)
            variable.mkdir(parents=True, exist_ok=True)
            activation_root.mkdir(parents=True, exist_ok=True)

            inputs, obs, pairs = self._prepare_input(rows, split=split)
            for name, value in (
                ("inputs", inputs),
                ("observations", obs),
                ("pairs", pairs),
            ):
                if not (shared / f"{name}.pt").exists():
                    torch.save(vars(value), shared / f"{name}.pt")

            activation_path = activation_root / f"{split}.pt"
            if not activation_path.exists():
                torch.save(
                    vars(self._prepare_activations(inputs, subject, obs)),
                    activation_path,
                )

            if not all(
                (variable / name).exists() for name in ("outcomes.pt", "targets.pt")
            ):
                outcomes, targets, details = self._prepare_outcomes(
                    inputs, subject, obs, pairs
                )
                info[split] = dict(details)
                torch.save(targets, variable / "targets.pt")
                torch.save(vars(outcomes), variable / "outcomes.pt")
            info.setdefault(split, {"total_examples": len(rows)})

        manifest = {
            "task": "program",
            "counterfactuals": list(self.prompts.families),
            "program_key": self.program.fingerprint(self.prompts.all_rows),
        }
        self.manifest(root, manifest)
        self.manifest(root / self.variable, info)
        return TaskCache(
            root=root,
            splits=frozenset(self.prompts.splits),
            manifest=manifest,
            variable=self.variable,
            position=self.selected_position,
            cut_fingerprint=subject.cut.fingerprint,
        )

    def encode_batch(
        self,
        inputs,
        observations,
        *,
        answers = None,
    ):
        """Encode complete answers with right-padded continuation masks.

        Paper presets can supply a tokenizer-specific token library explicitly
        through their compatibility declaration.
        """
        token_ids = None
        if answers is None:
            answers = self.program.answers.resolve(observations.rows)
            token_ids = self.program.compatibility.get("token_ids")

        ids, mask = self.encode_answers(answers, self.tokenizer, token_ids=token_ids)
        batch = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            position_ids=inputs.position_ids,
            outcome_ids=ids,
            outcome_mask=mask,
        )
        return batch, ids, mask

    def gold_indices(
        self,
        observations,
        answers,
    ):
        """Resolve optional gold labels as slot indices or exact answer strings."""
        name = self.program.gold or self.prompts.gold
        if name is None:
            return None
        result = []
        for i, (row, options) in enumerate(zip(observations.rows, answers)):
            value = column(row, name)
            if isinstance(value, int) and not isinstance(value, bool):
                index = value
            elif value in options:
                index = options.index(value)
            elif isinstance(value, str) and value.isdigit():
                index = int(value)
            else:
                raise ValueError(
                    f"Gold value {value!r} in row {i} is neither a slot index nor an exact answer string."
                )
            if not 0 <= index < len(options):
                raise ValueError(
                    f"Gold slot {index} in row {i} is outside the answer list."
                )
            result.append(index)
        return torch.tensor(result)

    @staticmethod
    def tokenize(
        rows,
        tokenizer,
        max_length,
    ):
        """Tokenize prompt records with offsets and left-padding position IDs."""
        texts = [row["prompt"] for row in rows]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        attention = encoded["attention_mask"]
        if not attention.bool().any(-1).all():
            raise ValueError("A prompt has no tokens after tokenization.")
        positions = attention.long().cumsum(-1) - 1
        positions.masked_fill_(~attention.bool(), 0)
        inputs = TaskInputs(
            encoded["input_ids"], positions, attention, encoded["offset_mapping"]
        )
        return inputs

    @staticmethod
    def resolve_positions(
        inputs,
        rows,
        selectors,
    ):
        """Map each named character selector to its token position tensor."""
        indices = torch.arange(inputs.input_ids.shape[1]).expand_as(inputs.input_ids)
        last = indices.masked_fill(~inputs.attention_mask.bool(), -1).amax(-1)
        collected = []
        for name, specification in selectors.items():
            if specification == "last_token":
                collected.append(last[:, None])
                continue

            if specification == "all":
                collected.append(indices)
                continue

            tokens = []
            for row_index, (row, offsets) in enumerate(
                zip(rows, inputs.offset_mapping)
            ):
                prompt = row["prompt"]
                spec = specification
                if isinstance(spec, dict):
                    spec = At(column(row, spec["regex_column"]))
                if isinstance(spec, At):
                    match = re.search(spec.regex, prompt)
                    if match is None:
                        raise ValueError(
                            f"Position {name}: no regex match in row {row_index}."
                        )
                    character = match.end() - 1
                elif callable(spec):
                    character = spec(prompt)
                else:
                    raise ValueError(f"Unknown position: {spec!r}.")
                token = next(
                    (
                        i
                        for i, (start, end) in enumerate(offsets)
                        if int(start) <= character < int(end)
                    ),
                    None,
                )
                if token is None:
                    raise ValueError(
                        f"Position {name} in row {row_index} is outside the tokenized prompt; check truncation."
                    )
                token += (
                    specification.get("token_shift", 0)
                    if isinstance(specification, dict)
                    else 0
                )
                if (
                    token >= inputs.input_ids.shape[1]
                    or not inputs.attention_mask[row_index, token]
                ):
                    raise ValueError(
                        f"Position {name} in row {row_index} is outside the prompt."
                    )
                tokens.append(token)
            resolved = torch.tensor(tokens)[:, None]
            collected.append(resolved)

        positions = torch.stack(collected, dim=1)
        return positions

    @staticmethod
    def encode_answers(
        answers,
        tokenizer,
        *,
        token_ids = None,
    ):
        """Encode one answer matrix with right-padded continuation masks."""
        if token_ids is not None:
            ids = torch.tensor(token_ids)[None, :, None].expand(len(answers), -1, -1)
            return ids, torch.ones_like(ids, dtype=torch.bool)

        encoded = tokenizer(
            [answer for row in answers for answer in row],
            padding=True,
            padding_side="right",
            add_special_tokens=False,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].reshape(len(answers), len(answers[0]), -1)
        mask = encoded["attention_mask"].reshape_as(ids)
        if not mask[..., 0].bool().all():
            raise ValueError(
                "An answer encodes to zero tokens. Check the tokenizer and answer strings."
            )
        return ids, mask

    def _prepare_input(
        self,
        rows,
        *,
        split = None,
    ):
        """Tokenize prompt text and construct seeded or explicit evaluation pairs."""
        rows = list(rows)
        inputs = self.tokenize(rows, self.tokenizer, self.max_length)

        if split in self.prompts.pairs:
            records = self.prompts.pairs[split]
        else:
            order = torch.randperm(
                len(rows), generator=torch.Generator().manual_seed(self.prompts.seed)
            )
            source = torch.empty_like(order)
            source[order] = order.roll(1)
            records = [
                dict(
                    base_index=i, source_index=int(source[i]), family_id=0, example_id=i
                )
                for i in range(len(rows))
            ]
        for record in records:
            if not 0 <= record["family_id"] < len(self.prompts.families):
                raise ValueError("A user-given pair has an unknown family ID.")
            if any(
                not 0 <= record[key] < len(rows)
                for key in ("base_index", "source_index")
            ):
                raise ValueError(
                    "A user-given pair references a row outside its split."
                )
        pairs = TaskPairs(
            **{
                key: torch.tensor([r[key] for r in records], dtype=torch.long)
                for key in ("base_index", "source_index", "family_id", "example_id")
            }
        )
        observations = ProgramObservations(
            prompts=[row["prompt"] for row in rows],
            rows=rows,
        )
        return inputs, observations, pairs

    def _prepare_activations(
        self,
        inputs,
        subject,
        observations,
    ):
        """Resolve character spans and record residual activations at each site."""
        positions = self.resolve_positions(inputs, observations.rows, self.positions)
        activations = record_activations(
            inputs, subject, positions, self.record_batch_size
        )
        return TaskActivations(self.position_names, activations, positions)

    def _prepare_outcomes(
        self,
        inputs,
        subject,
        observations,
        pairs,
    ):
        """Record readings, apply optional gold filtering and compile targets."""
        batch, ids, mask = self.encode_batch(inputs, observations)
        behaviour = record_behavior(inputs, subject, ids, mask, self.record_batch_size)
        features = behaviour
        rows = observations.rows
        compat = self.program.compatibility
        if compat.get("feature_column"):
            encoded = self.tokenizer(
                [column(row, compat["feature_column"]) for row in rows],
                padding=True,
                return_tensors="pt",
            )
            attention = encoded["attention_mask"]
            pos = attention.long().cumsum(-1) - 1
            pos.masked_fill_(attention == 0, 0)
            features = record_behavior(
                TaskInputs(encoded["input_ids"], pos, attention),
                subject,
                ids,
                mask,
                self.record_batch_size,
            )

        answers = self.program.answers.resolve(rows)
        gold = self.gold_indices(observations, answers)
        correct = (
            torch.ones(len(rows), dtype=torch.bool)
            if gold is None
            else behaviour.argmax(-1) == gold
        )
        checkable = torch.ones_like(correct)
        if compat.get("greedy_filter"):
            expected = torch.tensor([row["_expected_token"] for row in rows])
            checkable = expected >= 0
            if not checkable.any():
                raise ValueError("RAVEL has no observations with checkable answers.")
            correct = (expected < 0) | (
                subject.greedy(batch, self.record_batch_size) == expected
            )
        if (
            self.correctness_filter
            and float(correct[checkable].float().mean()) < self.quality_gate
        ):
            raise ValueError(
                f"Quality gate failed: {int(correct.sum())} / {len(correct)} prompts agree with gold."
            )
        keep = correct if self.correctness_filter else torch.ones_like(correct)

        gate = None
        group = None
        if isinstance(self.program.rule, Gated):
            values = [column(row, self.program.rule.column) for row in rows]
            if any(not isinstance(value, bool) for value in values):
                raise ValueError("Gated requires a boolean metadata column.")
            gate = torch.tensor(values, dtype=torch.bool)
        if isinstance(self.program.rule, Table):
            if compat.get("table_column"):
                tables = [column(row, compat["table_column"]) for row in rows]
            else:
                tables = self.program.rule.compile(rows, self.num_outcomes)
            gate = torch.tensor(tables, dtype=behaviour.dtype)

        if compat.get("group_column"):
            values = [column(row, compat["group_column"]) for row in rows]
            lookup = {v: i for i, v in enumerate(sorted(set(values)))}
            group = torch.tensor([lookup[v] for v in values])
            present, counts = group[keep].unique(return_counts=True)
            keep &= torch.isin(group, present[counts > 1])

        state_gold = gold
        if compat.get("state_gold_column"):
            state_gold = torch.tensor(
                [column(row, compat["state_gold_column"]) for row in rows]
            )
        source = (
            features
            if state_gold is None
            else one_hot(state_gold, self.num_outcomes).to(behaviour)
        )
        base = (
            behaviour
            if gold is None
            else one_hot(gold, self.num_outcomes).to(behaviour)
        )
        targets = SemanticTransport.from_name(self.program.transport)(
            source[pairs.source_index],
            base[pairs.base_index],
            None if gate is None else gate[pairs.base_index],
        ).argmax(-1)
        outcomes = TaskOutcomes(
            outcome_ids=ids[keep],
            outcome_mask=mask[keep],
            correct_indices=keep.nonzero(as_tuple=True)[0],
            state_features=features[keep],
            base_behaviour=behaviour[keep],
            gold_state_features=(
                None
                if state_gold is None
                else one_hot(state_gold, self.num_outcomes).to(behaviour)[keep]
            ),
            transport_gate=None if gate is None else gate[keep],
            transport_group=None if group is None else group[keep],
        )
        return (
            outcomes,
            targets,
            {
                "total_examples": len(rows),
                "correct_examples": int(correct.sum()),
                "num_filtered": int((~keep).sum()),
            },
        )
