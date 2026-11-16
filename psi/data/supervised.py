"""The arm of the supervision comparison that uses the benchmark's own pairs.

Reached only through `data.donor_pairing: curated`, so the ordinary path stays
provably donor-free. The official source is attached to the base's record as
`source_activation` and `source_state_features`; `psi/psi.py` uses those when
present and otherwise deranges within the batch.
"""
from typing import Mapping

import torch
from torch import Tensor

from psi.data.module import PSIDataModule


class SupervisedPairDataModule(PSIDataModule):
    """One item per official (base, source) pair that survives correctness filtering."""

    def __init__(self, config: Mapping[str, float | int | str], prepared) -> None:
        super().__init__(config, prepared)
        # These three controls reshape the row multiset, and pairing reshapes it
        # again; combining them has no defined meaning and the 2x2 needs none of
        # them. Refuse rather than silently compose.
        for key, default in (("training_pool", "all"),
                             ("shuffle_state_features", False),
                             ("donor_balance", "none")):
            if config.get(key, default) != default:
                raise ValueError(f"donor_pairing: curated cannot be combined with {key}.")
        self.pair_limit = config.get("pair_limit")
        self.control_summary["donor_pairing"] = "curated"
        
        self.pair_coupling = str(config.get("pair_coupling", "curated"))

    def pair(self, split: str, records: dict[str, Tensor], summary: dict) -> dict[str, Tensor]:
        pairs = torch.load(
            self.data.root / split / "pairs.pt", map_location="cpu", weights_only=True
        )
        base, source = pairs["base_index"], pairs["source_index"]

        # `records` is indexed by record position; `pairs.pt` is indexed by raw
        # observation id. `correct_indices` maps position -> raw id, so invert it.
        correct = self.data.outcomes(split).correct_indices
        width = int(max(correct.max(), base.max(), source.max())) + 1
        position = torch.full((width,), -1, dtype=torch.long)
        position[correct] = torch.arange(len(correct))

        base_at, source_at = position[base], position[source]
        # A pair survives only if both of its members passed correctness
        # filtering; the donor-free arm has no such attrition, which is why the
        # experiment has to match pool sizes afterwards.
        keep = (base_at >= 0) & (source_at >= 0)
        base_at, source_at = base_at[keep], source_at[keep]
        if len(base_at) < 2:
            raise ValueError(f"{split}: fewer than two official pairs survive filtering.")

        if self.pair_limit is not None and len(base_at) > int(self.pair_limit):
            chosen = torch.randperm(
                len(base_at), generator=torch.Generator().manual_seed(self.seed)
            )[: int(self.pair_limit)].sort().values
            base_at, source_at = base_at[chosen], source_at[chosen]

        if self.pair_coupling == "shuffled":
            generator = torch.Generator().manual_seed(self.seed)
            for _ in range(100):
                order = torch.randperm(len(source_at), generator=generator)
                if bool((source_at[order] != base_at).all()):
                    break
            else:
                raise ValueError("Could not shuffle the pair list without a self-pair.")
            source_at = source_at[order]
            source_activation = records["activation"][source_at]
            source_state_features = records["state_features"][source_at]
            source_gold_state_features = records["gold_state_features"][source_at] if "gold_state_features" in records else None
        elif self.pair_coupling == "curated":
            source_activation = records["activation"][source_at]
            source_state_features = records["state_features"][source_at]
            source_gold_state_features = records["gold_state_features"][source_at] if "gold_state_features" in records else None
        elif self.pair_coupling == "derangement":
            source_activation = None
            source_state_features = None
            source_gold_state_features = None
        else:
            raise ValueError(f"Unknown pair_coupling: {self.pair_coupling}")

        paired = {key: value[base_at] for key, value in records.items()}
        
        if source_activation is not None and source_state_features is not None:
            paired["source_activation"] = source_activation
            paired["source_state_features"] = source_state_features
        if source_gold_state_features is not None:
            paired["source_gold_state_features"] = source_gold_state_features

        summary.update(
            donor_pairing="curated",
            official_pairs=int(len(base)),
            retained_pairs=int(len(base_at)),
            distinct_bases=int(base_at.unique().numel()),
            distinct_sources=int(source_at.unique().numel()),
        )
        return paired
