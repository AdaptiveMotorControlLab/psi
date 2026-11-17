"""Declarations of answers and the rule used to transport them."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import warnings
from typing import Any, Mapping


def column(
    row,
    name,
):
    """Read a column, including dotted paths into nested records."""
    if name in row:
        return row[name]
    value = row
    for part in name.split("."):
        value = value[part]
    return value


def digest(
    value,
):
    """Hash a declaration after deterministic JSON serialization."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Values:
    """One answer list shared by every prompt.

    ``Values([" A", " B"])`` studies the answer letter. A donor choosing
    ``" B"`` asks the base for that same string. Entries are nonempty,
    distinct strings used verbatim as continuations.
    """

    values: tuple[str, ...]

    def __init__(
        self,
        values,
    ):
        values = tuple(values)
        _validate_answers(values)
        object.__setattr__(self, "values", values)

    def resolve(
        self,
        rows,
    ):
        """Repeat the shared answer tuple once for each prompt record."""
        return [self.values for _ in rows]


@dataclass(frozen=True)
class Slots:
    """Per-prompt options whose shared meaning is the chosen slot index.

    ``Slots("choices.label")`` reads a dotted metadata column. A list of
    per-prompt lists is also accepted. Every row needs the same slot count;
    its answer strings and their token lengths may differ.
    """

    column: str | tuple[tuple[str, ...], ...]

    def __init__(
        self,
        column,
    ):
        object.__setattr__(
            self,
            "column",
            column if isinstance(column, str) else tuple(tuple(v) for v in column),
        )

    def resolve(
        self,
        rows,
    ):
        """Read each row's options and validate their common slot count.

        Explicit per-prompt lists use the original indices stored by Prompts,
        so a reproducible split retains its original answer list.
        """
        if isinstance(self.column, str):
            result = [tuple(column(row, self.column)) for row in rows]
        else:
            result = []
            for index, row in enumerate(rows):
                if "_psi_options" in row:
                    result.append(tuple(row["_psi_options"]))
                    continue

                if "_psi_index" not in row and len(rows) != len(self.column):
                    raise ValueError(
                        "Slots with per-prompt lists needs original row metadata or explicit options=."
                    )
                original_index = row.get("_psi_index", index)
                result.append(self.column[original_index])
        if not result:
            raise ValueError("Prompts must contain at least one row.")
        k = len(result[0])
        bad = [
            row.get("_psi_index", i)
            for i, (row, answers) in enumerate(zip(rows, result))
            if len(answers) != k
        ]
        if bad:
            raise ValueError(
                f"Slots needs the same K in every row; expected {k}, offending rows: {bad}."
            )
        for answers in result:
            _validate_answers(answers)
        return result


@dataclass(frozen=True)
class At:
    """Read the last token of the first regex match, as in At(r'Paris')."""

    regex: str


@dataclass(frozen=True)
class Gated:
    """Copy when a boolean column is true and keep the base reading elsewhere."""

    column: str


@dataclass(frozen=True)
class Table:
    """Map donor slots to target slots, as in Table({0: 1, 1: 0}).

    A callable receives a donor slot. A row-dependent callable receives both
    the donor slot and the base row. Each result is a target slot or a
    probability vector. Tables use the existing transport's donor argmax rule.
    """

    mapping: Any

    def compile(
        self,
        rows,
        k,
    ):
        """Build one source-by-target probability table per base prompt.

        A mapping supplies a target slot or probability vector for each source
        slot. A callable accepts the source slot and optionally the base row.
        The existing table transport selects the donor's argmax row.
        """
        import inspect

        result = []
        fn = self.mapping
        arity = len(inspect.signature(fn).parameters) if callable(fn) else 0
        for row in rows:
            table = []
            for source in range(k):
                if not callable(fn):
                    target = fn.get(source, fn.get(str(source)))
                elif arity > 1:
                    target = fn(source, row)
                else:
                    target = fn(source)
                if isinstance(target, int) and not isinstance(target, bool):
                    if not 0 <= target < k:
                        raise ValueError(f"Table target {target} is outside [0, {k}).")
                    target = [float(j == target) for j in range(k)]
                if (
                    target is None
                    or len(target) != k
                    or any(not math.isfinite(float(v)) or float(v) < 0 for v in target)
                    or abs(sum(target) - 1) > 1e-6
                ):
                    raise ValueError(
                        "Every Table row must specify a target slot or a probability vector summing to one."
                    )
                table.append(list(target))
            result.append(table)
        return result


@dataclass(frozen=True)
class CheckReport:
    """Pre-flight mass, answer counts, optional gold agreement and spelling check."""

    mean_answer_mass: float
    lowest_answer_mass: float
    argmax_counts: tuple[int, ...]
    gold_agreement: float | None
    leading_space_mass: float
    warnings: tuple[str, ...]

    def __repr__(
        self,
    ):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())


@dataclass(frozen=True)
class Program:
    """Declare a variable's answers and the rule for an interchange.

    Parameters
    ----------
    answers
        Values for a shared list of continuation strings, or Slots for each
        prompt's own options. Leading spaces are part of every answer.
    rule
        "copy" takes the donor reading, "keep" retains the base reading,
        Gated(column) chooses between them, and Table(mapping) applies an
        explicit source-slot rule.
    gold
        Optional metadata column containing exact answer strings or slot
        indices. Prompts.gold is used when this field is absent.
    compatibility
        Reserved declarations used by paper presets for their feature
        readings, token libraries and semantic groups.

    Examples
    --------
    Program(Values([" yes", " no"]), rule="copy")
    Program(Slots("choices.label"), gold="answerKey")
    """

    answers: Values | Slots
    rule: str | Gated | Table = "copy"
    gold: str | None = None
    # Preset compatibility records the paper tasks' additional reading contracts.
    compatibility: Mapping = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(
        cls,
        value,
    ):
        """Read the values/slots and copy/keep/gated/table declaration schema."""
        answers = value["answers"]
        rule = value.get("rule", "copy")
        if isinstance(rule, dict):
            rule = Gated(rule["gated"]) if "gated" in rule else Table(rule["table"])
        return cls(
            (
                Values(answers["values"])
                if "values" in answers
                else Slots(answers["slots"])
            ),
            rule,
            value.get("gold"),
            value.get("compatibility", {}),
        )

    @property
    def transport(
        self,
    ):
        """Name of the existing transport implementation used by this rule."""
        if isinstance(self.rule, Gated):
            return "gated_change"
        if isinstance(self.rule, Table):
            return "table"
        return {"copy": "full_change", "keep": "no_change"}[self.rule]

    def to_dict(
        self,
        rows = None,
    ):
        """Return the portable answer and rule declaration.

        Rows are needed to compile callable tables. A callable whose tables
        vary by row needs an explicit metadata declaration before saving.
        """
        answers = (
            {"values": list(self.answers.values)}
            if isinstance(self.answers, Values)
            else {"slots": self.answers.column}
        )
        rule = self.rule
        if isinstance(self.rule, Gated):
            rule = {"gated": self.rule.column}

        if isinstance(self.rule, Table):
            mapping = self.rule.mapping
            if callable(mapping):
                if rows is None:
                    raise ValueError(
                        "A callable Table needs prompt rows to compile for serialization."
                    )

                count = len(self.answers.resolve(rows)[0])
                tables = self.rule.compile(rows, count)
                if any(table != tables[0] for table in tables[1:]):
                    raise ValueError(
                        "Saving a row-dependent callable Table requires an explicit metadata column."
                    )
                mapping = dict(enumerate(tables[0]))

            rule = {"table": mapping}

        return {
            "answers": answers,
            "rule": rule,
            "gold": self.gold,
            "compatibility": dict(self.compatibility),
        }

    def fingerprint(
        self,
        rows,
    ):
        """Hash the declaration, compiling callable rules on their prompt domain."""
        if isinstance(self.rule, Table) and callable(self.rule.mapping):
            serial = replace(self, rule="copy").to_dict()
            serial["rule"] = self.rule.compile(rows, len(self.answers.resolve(rows)[0]))
        else:
            serial = self.to_dict()
        return digest(serial)

    def check(
        self,
        model,
        prompts,
    ):
        """Measure answer mass and the model's answers before fitting."""
        import torch
        from psi.data.tasks.program import ProgramTask

        task = ProgramTask.from_objects(
            prompts,
            self,
            model.tokenizer,
            position=model.position,
            max_length=model.max_length,
            correctness_filter=False,
        )
        rows = prompts.all_rows
        key = task.identity(model.subject)
        checks = getattr(model, "_preflight_results", {})
        if key in checks:
            return checks[key]

        inputs, obs, _ = task._prepare_input(rows)
        batch, _, _ = task.encode_batch(inputs, obs)
        masses, predictions = [], []
        alternatives = []
        answer_rows = self.answers.resolve(rows)
        spaced = [
            [" " + answer.lstrip() for answer in answers] for answers in answer_rows
        ]
        spelling_changed = spaced != [list(answers) for answers in answer_rows]
        alt_batch = batch
        if spelling_changed:
            alt_batch, _, _ = task.encode_batch(inputs, obs, answers=spaced)

        with torch.inference_mode():
            for start in range(0, len(rows), model.batch_size):
                b = {
                    key: value[start : start + model.batch_size].to(
                        model.subject.device
                    )
                    for key, value in batch.items()
                }
                alt = {
                    key: value[start : start + model.batch_size].to(
                        model.subject.device
                    )
                    for key, value in alt_batch.items()
                }
                logp = model.subject.outcome_log_probability(b)
                masses.append(logp.logsumexp(-1).exp().cpu())
                predictions.append(logp.argmax(-1).cpu())
                alternate_mass = logp.logsumexp(-1).exp()
                if spelling_changed:
                    alternate_mass = model.subject.behavior_mass(alt)
                alternatives.append(alternate_mass.cpu())

        mass, predicted = torch.cat(masses), torch.cat(predictions)
        gold = task.gold_indices(obs, answer_rows)
        messages = []
        if float(mass.mean()) < 0.1:
            messages.append(
                "Mean answer mass is low. Check answer spellings and prompt endings."
            )
        alt_mass = float(torch.cat(alternatives).mean())
        if alt_mass > float(mass.mean()):
            messages.append(
                "The alternate leading-space spelling carries more answer mass."
            )

        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=2)

        report = CheckReport(
            float(mass.mean()),
            float(mass.min()),
            tuple(torch.bincount(predicted, minlength=len(answer_rows[0])).tolist()),
            None if gold is None else float((predicted == gold).float().mean()),
            alt_mass,
            tuple(messages),
        )
        checks[key] = report
        model._preflight_results = checks
        return report

    def __post_init__(
        self,
    ):
        if not isinstance(self.answers, (Values, Slots)):
            raise TypeError("answers must be Values or Slots.")
        if not isinstance(self.rule, (Gated, Table)) and self.rule not in (
            "copy",
            "keep",
        ):
            raise ValueError("rule must be copy, keep, Gated or Table.")

    def __hash__(
        self,
    ):
        # Python containers use structural equality, including 1 == 1.0.
        # Persistent cache identities use fingerprint(rows) separately.
        rule = self.rule.mapping if isinstance(self.rule, Table) else self.rule
        return hash(
            (self.answers, _hashable(rule), self.gold, _hashable(self.compatibility))
        )


def _validate_answers(
    answers,
):
    if len(answers) < 2 or any(not isinstance(a, str) or not a for a in answers):
        raise ValueError(
            "Declare at least two nonempty answer strings, used verbatim as continuations."
        )
    if len(set(answers)) != len(answers):
        raise ValueError("Answer strings within a row must be distinct.")


def _hashable(
    value,
):
    """Make nested declaration containers hashable with Python's numeric equality."""
    if isinstance(value, Mapping):
        return frozenset((key, _hashable(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value
