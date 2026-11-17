"""Prompt collections with fixed splits and metadata columns."""

from __future__ import annotations

import random
from psi.program import column, digest


class Prompts:
    """Prompt text with metadata, as in Prompts(['Paris is in'], columns={'gold': [' France']})."""

    @classmethod
    def from_hf(
        cls,
        dataset,
        subset = None,
        *,
        text = 'prompt',
        options = None,
        gold = None,
        splits = None,
        seed = 11,
        limit = None,
    ):
        """Load prompt records while retaining metadata used by the program.

        Parameters
        ----------
        dataset, subset
            Hugging Face dataset ID and optional configuration. An already
            loaded Dataset or DatasetDict is accepted for offline use.
        text, options, gold
            Text column and optional option/gold metadata column names.
            Dotted text paths can read nested records.
        splits
            Mapping from train/validation/test to source split names. Existing
            splits are kept. A dataset with one split uses a seeded 80/10/10
            assignment; partial split collections require an explicit mapping.
        seed
            Seed stored with the split assignment.
        limit
            Optional row cap per source split, useful for notebook examples.
        """
        from datasets import load_dataset

        raw = load_dataset(dataset, subset) if isinstance(dataset, str) else dataset
        if (
            hasattr(raw, "column_names")
            and not isinstance(raw, dict)
            and isinstance(raw.column_names, list)
        ):
            raw = {"train": raw}
        names = splits or {
            key: key if key != "validation" or "validation" in raw else "val"
            for key in ("train", "validation", "test")
        }

        def read(
            rows,
        ):
            rows = list(rows)
            if limit is not None:
                rows = rows[:limit]
            return [dict(row, prompt=column(row, text)) for row in rows]

        if all(name in raw for name in names.values()):
            return cls(
                [],
                splits={key: read(raw[name]) for key, name in names.items()},
                gold=gold,
                options=options,
                seed=seed,
            )
        if len(raw) != 1:
            raise ValueError(
                "Dataset has partial splits; provide an explicit split mapping."
            )
        rows = read(next(iter(raw.values())))
        keys = set().union(*(row.keys() for row in rows)) - {"prompt"}
        return cls(
            [row["prompt"] for row in rows],
            columns={key: [row.get(key) for row in rows] for key in keys},
            gold=gold,
            options=options,
            seed=seed,
        )

    @classmethod
    def from_pandas(
        cls,
        frame,
        *,
        text = 'prompt',
        options = None,
        gold = None,
        seed = 11,
        split = None,
    ):
        """Build prompt records from a dataframe and retain its metadata.

        ``text`` names the prompt column. ``options`` and ``gold`` store the
        corresponding metadata names. When ``split`` is supplied, its values
        must name train, validation or test. Otherwise ``seed`` fixes a new
        split assignment in the returned Prompts object.
        """
        rows = frame.to_dict("records")
        if split:
            invalid = [
                index
                for index, row in enumerate(rows)
                if row[split] not in {"train", "validation", "test"}
            ]
            if invalid:
                raise ValueError(
                    f"Unknown or missing split labels in dataframe rows: {invalid}."
                )
            return cls(
                [],
                splits={
                    name: [
                        dict(row, prompt=row[text])
                        for row in rows
                        if row[split] == name
                    ]
                    for name in ("train", "validation", "test")
                },
                gold=gold,
                options=options,
                seed=seed,
            )
        return cls(
            [row[text] for row in rows],
            columns={
                key: [row[key] for row in rows] for key in frame.columns if key != text
            },
            gold=gold,
            options=options,
            seed=seed,
        )

    @property
    def all_rows(
        self,
    ):
        """All records ordered by their original prompt index."""
        return sorted(
            [row for rows in self.splits.values() for row in rows],
            key=lambda row: row["_psi_index"],
        )

    @property
    def fingerprint(
        self,
    ):
        """Stable identity of the split assignment, metadata and pair declarations."""
        positions = {}
        for name, position in self.positions.items():
            if callable(position):
                positions[name] = [position(row["prompt"]) for row in self.all_rows]
            else:
                positions[name] = getattr(position, "regex", position)
        return digest(
            {
                "gold": self.gold,
                "options": self.options,
                "splits": self.splits,
                "pairs": self.pairs,
                "families": self.families,
                "positions": positions,
                "seed": self.seed,
            }
        )

    def __init__(
        self,
        texts,
        *,
        columns = None,
        seed = 11,
        gold = None,
        options = None,
        splits = None,
        pairs = None,
        families = None,
        positions = None,
    ):
        """Store text and metadata with reproducible, disjoint splits.

        Parameters
        ----------
        texts
            Prompt strings. With no explicit splits, a seeded shuffle assigns
            approximately 80/10/10 percent to train, validation and test. Each
            split retains at least two rows for donor pairing.
        columns
            Metadata lists with one entry per text. Rules, Slots and gold labels
            can refer to these column names.
        seed
            Seed for the stored split assignment and default evaluation pairs.
        gold, options
            Optional column names identifying gold labels and answer options.
        splits
            Explicit train/validation/test mappings of complete prompt records.
        pairs
            Per-split evaluation links, as TaskPairs, dictionaries, or
            (base_index, source_index) tuples. Training donors are separate.
        families
            Names indexed by each evaluation pair's family_id.
        positions
            Named token selectors. Dataset presets retain the paper's sites.
        """
        self.seed = int(seed)

        self.gold = gold
        self.options = options
        self.pairs = {}
        for name, values in (pairs or {}).items():
            normalized = []
            for index, pair in enumerate(values):
                if hasattr(pair, "base_index"):
                    pair = vars(pair)
                if isinstance(pair, (list, tuple)):
                    pair = dict(base_index=int(pair[0]), source_index=int(pair[1]))
                normalized.append({"family_id": 0, "example_id": index, **pair})
            self.pairs[name] = normalized

        self.families = tuple(families or ("random",))
        self.positions = positions or {"last_token": "last_token"}

        if splits is None:
            texts = list(texts)
            columns = columns or {}
            if any(len(values) != len(texts) for values in columns.values()):
                raise ValueError("Metadata columns must have one entry per prompt.")
            rows = [
                dict(prompt=text, _psi_index=i, **{k: v[i] for k, v in columns.items()})
                for i, text in enumerate(texts)
            ]
            if len(rows) < 6:
                raise ValueError(
                    "At least six prompts are needed for disjoint train, validation and test splits."
                )
            indices = list(range(len(rows)))
            random.Random(self.seed).shuffle(indices)
            held = max(2, len(rows) // 10)
            self.splits = {
                name: [rows[i] for i in selection]
                for name, selection in (
                    ("train", indices[: len(rows) - 2 * held]),
                    ("validation", indices[len(rows) - 2 * held : len(rows) - held]),
                    ("test", indices[len(rows) - held :]),
                )
            }
        else:
            self.splits = {
                name: [dict(row) for row in rows] for name, rows in splits.items()
            }
            if set(self.splits) != {"train", "validation", "test"}:
                raise ValueError("Provide train, validation and test splits.")
            index = 0
            for rows in self.splits.values():
                for row in rows:
                    row.setdefault("_psi_index", index)
                    index += 1

        for name, rows in self.splits.items():
            if len(rows) < 2:
                raise ValueError(f"{name} needs at least two prompts.")
            if any(
                not isinstance(row["prompt"], str) or not row["prompt"].strip()
                for row in rows
            ):
                raise ValueError(f"{name} contains empty prompt text.")

    def head(
        self,
        count,
    ):
        """Select the first count base examples in each split, retaining their donors."""
        splits, pairs = {}, {}
        for name, rows in self.splits.items():
            source_pairs = self.pairs.get(name)
            if source_pairs:
                selected = [p for p in source_pairs if p["example_id"] < count]
                indices = sorted(
                    {p[key] for p in selected for key in ("base_index", "source_index")}
                )
                remap = {value: i for i, value in enumerate(indices)}
                pairs[name] = [
                    dict(
                        p,
                        base_index=remap[p["base_index"]],
                        source_index=remap[p["source_index"]],
                    )
                    for p in selected
                ]
                splits[name] = [rows[i] for i in indices]
            else:
                splits[name] = rows[: max(2, count)]
        return Prompts(
            [],
            splits=splits,
            pairs=pairs,
            families=self.families,
            gold=self.gold,
            options=self.options,
            positions=self.positions,
            seed=self.seed,
        )
