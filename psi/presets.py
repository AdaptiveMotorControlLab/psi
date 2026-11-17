"""Prompt programs for the paper's public datasets."""

from __future__ import annotations

import json
from pathlib import Path
import re

from psi.program import Program, Values, Slots, Gated, Table
from psi.prompts import Prompts


def mcqa(
    variable = 'x_order',
    *,
    dataset = 'mib-bench/copycolors_mcqa',
    subset = '4_answer_choices',
    num_choices = 4,
    counterfactuals = ('answerPosition', 'randomLetter', 'answerPosition_randomLetter'),
    limit = None,
    seed = 11,
):
    """Study the chosen option slot or the answer letter on MIB prompts.

    The preset adds the leading spaces used in the paper's task declaration.
    Slots used directly elsewhere preserve their strings verbatim.
    """
    if variable not in ("x_order", "o_answer"):
        raise ValueError("variable must be x_order or o_answer.")

    def convert(
        row,
    ):
        if len(row["choices"]["label"]) != num_choices:
            return None
        labels = [" " + str(label) for label in row["choices"]["label"]]
        answer = int(row["answerKey"])
        symbol = labels[answer]
        return dict(
            row,
            choices=dict(row["choices"], label=labels),
            _gold=(
                answer
                if variable == "x_order"
                else "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(symbol.strip())
            ),
            _symbol_regex=r"\b" + re.escape(symbol.strip()) + r"\b",
        )

    splits, pairs = _expand(
        _raw(dataset, subset), counterfactuals, convert, limit=limit
    )
    prompts = Prompts(
        [],
        splits=splits,
        pairs=pairs,
        families=counterfactuals,
        gold="_gold",
        seed=seed,
        positions={
            "last_token": "last_token",
            "correct_symbol": {"regex_column": "_symbol_regex"},
            "correct_symbol_period": {
                "regex_column": "_symbol_regex",
                "token_shift": 1,
            },
        },
    )
    return prompts, Program(
        (
            Slots("choices.label")
            if variable == "x_order"
            else Values([" " + letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
        ),
        gold="_gold",
    )


def arc(
    variable = 'x_order',
    *,
    dataset = 'mib-bench/arc_easy',
    subset = None,
    **kwargs,
):
    """Use the MCQA program on the ARC dataset and retain its cut positions."""
    prompts, program = mcqa(variable, dataset=dataset, subset=subset, **kwargs)
    prompts.positions = {
        "correct_symbol": prompts.positions["correct_symbol"],
        "last_token": "last_token",
    }
    return prompts, program


def ravel(
    attribute = 'country',
    *,
    tokenizer,
    dataset = 'mib-bench/ravel',
    entities = None,
    entity_file = None,
    counterfactuals = ('attribute', 'wikipedia'),
    limit = None,
    max_examples = 10000,
    selection_seed = 42,
    seed = 11,
):
    """Reproduce the paper's first-token RAVEL library and separate feature reading.

    The tokenizer is required because the library merges values sharing their
    first token. Generic Values programs score complete continuations.
    """
    import torch

    if attribute not in ("country", "continent", "language"):
        raise ValueError("attribute must be country, continent or language.")
    raw = _raw(dataset, None)
    if entity_file:
        entities = json.loads(Path(entity_file).read_text())
    library_rows = list(entities.values()) if entities else []
    if not entities:
        for split in raw.values():
            for row in split:
                library_rows.append(row)
                library_rows.extend(
                    row[family + "_counterfactual"] for family in counterfactuals
                )

    values = sorted(
        {
            row[key]
            for row in library_rows
            for key in ("Country", "Continent", "Language")
            if row[key] is not None
        }
    )
    first = {
        value: tokenizer.encode(" " + value.strip(), add_special_tokens=False)[0]
        for value in values
    }
    tokens = sorted(set(first.values()))
    classes = {value: tokens.index(token) for value, token in first.items()}
    attr = attribute.title()

    def convert(
        row,
    ):
        if any(row[key] is None for key in ("Country", "Continent", "Language")):
            return None
        queried = row["attribute"]
        value = row[queried] if queried != "wikipedia" else None
        return dict(
            row,
            _gate=queried == attr,
            _feature=f'Q: What is the {attribute} of {row["entity"]}?\nA:',
            _gold=classes[value] if value else classes[row[attr]],
            _state_gold=classes[row[attr]],
            _expected_token=first[value] if value else -1,
            _entity_regex=r"\b" + re.escape(row["entity"]) + r"\b",
        )

    def select(
        rows,
    ):
        if len(rows) > max_examples:
            order = torch.randperm(
                len(rows), generator=torch.Generator().manual_seed(selection_seed)
            )[:max_examples]
            return [rows[i] for i in order.tolist()]
        return rows

    splits, pairs = _expand(raw, counterfactuals, convert, limit=limit, select=select)
    prompts = Prompts(
        [],
        splits=splits,
        pairs=pairs,
        families=counterfactuals,
        gold="_gold",
        seed=seed,
        positions={
            "last_token": "last_token",
            "entity_last_token": {"regex_column": "_entity_regex"},
        },
    )
    program = Program(
        Values([tokenizer.decode([token]) for token in tokens]),
        Gated("_gate"),
        gold="_gold",
        compatibility={
            "token_ids": tokens,
            "feature_column": "_feature",
            "state_gold_column": "_state_gold",
            "greedy_filter": True,
        },
    )
    return prompts, program


def ioi(
    variable = 'output_position',
    *,
    dataset = 'mib-bench/ioi',
    counterfactuals = ('SName', 'IOName', 'S1Name'),
    limit = None,
    seed = 11,
):
    """Study IOI with complete, space-prefixed name continuations.

    This preset reads the last residual token. The legacy IOI task records all
    attention-head positions. This preset uses the fitted table probabilities
    from the paper. Its reported
    score is declared-answer agreement. The legacy task also measures bare-name
    logit margins for the official benchmark. That metric has a separate scale.
    """
    if variable not in ("output_position", "output_token"):
        raise ValueError("variable must be output_position or output_token.")

    def convert(
        row,
    ):
        names = row["choices"]
        first, second, *_ = re.findall(
            r"\b(" + "|".join(map(re.escape, names)) + r")\b", row["prompt"]
        )
        ordered = [" " + first, " " + second]
        if variable == "output_token":
            ordered.sort()
        answer = " " + names[row["answerKey"]]
        gold = ordered.index(answer)
        same = 0.048 + 0.768 + 2.005
        changed = (
            0.048 + 0.768 - 2.005
            if variable == "output_position"
            else 0.048 - 0.768 + 2.005
        )
        import math

        table = []
        for source in range(2):
            p = 1 / (1 + math.exp(-(same if source == gold else changed)))
            table.append([p if target == gold else 1 - p for target in range(2)])
        return dict(
            row,
            _names=ordered,
            _gold=gold,
            _group="|".join(sorted(ordered)),
            _table=table,
        )

    splits, pairs = _expand(_raw(dataset, None), counterfactuals, convert, limit=limit)
    prompts = Prompts(
        [],
        splits=splits,
        pairs=pairs,
        families=counterfactuals,
        gold="_gold",
        seed=seed,
    )
    # A metadata table remains serializable and carries the fitted per-row rule.
    program = Program(
        Slots("_names"),
        Table({0: 0, 1: 1}),
        gold="_gold",
        compatibility={
            "table_column": "_table",
            **({"group_column": "_group"} if variable == "output_token" else {}),
        },
    )
    return prompts, program


def _raw(
    dataset,
    subset,
):
    if isinstance(dataset, str):
        from datasets import load_dataset

        return load_dataset(dataset, subset)
    return dataset


def _expand(
    raw,
    families,
    convert,
    *,
    limit = None,
    select = None,
):
    splits, pairs = {}, {}
    for name, source in (
        ("train", "train"),
        ("validation", "validation" if "validation" in raw else "val"),
        ("test", "test"),
    ):
        rows, links = [], []
        source_rows = list(raw[source])
        if select:
            source_rows = select(source_rows)
        for example_id, row in enumerate(
            source_rows[:limit] if limit is not None else source_rows
        ):
            base = convert(row)
            if base is None:
                continue
            b = len(rows)
            rows.append(base)
            for family_id, family in enumerate(families):
                donor = convert(row[family + "_counterfactual"])
                if donor is not None:
                    links.append(
                        dict(
                            base_index=b,
                            source_index=len(rows),
                            family_id=family_id,
                            example_id=example_id,
                        )
                    )
                    rows.append(donor)
        splits[name], pairs[name] = rows, links
    return splits, pairs
