"""Donors chosen to differ from the base in the declared variable.

An unconstrained derangement draws a donor at random. On arithmetic that means
roughly half of them already share the base's carry, so the transport rule
returns the base's own reading and the pair asks for nothing. The benchmark's
`ones_carry` family is built from minimal counterfactuals that always differ,
which is the case such a derangement almost never produces, and it is the family
where the carrier is weakest.

This module pairs a base with a source whose *measured* reading of the variable
differs. The reading comes from the frozen subject through the declared library,
not from a label file or a supplied pair list, so it is the same information the
outcome library already uses.

`contrast_fraction` interpolates: at 0 every donor is drawn uniformly, which
reproduces the ordinary pairing through this module's own machinery; at 1 every
donor differs. Intermediate values trace the trade-off between the two
counterfactual families.
"""
from typing import Mapping

import torch
from torch import Tensor

from psi.data.module import PSIDataModule, _SPLIT_OFFSET


class ContrastivePairDataModule(PSIDataModule):
    """One donor per base, drawn to differ in the declared variable."""

    def __init__(self, config: Mapping[str, float | int | str], prepared) -> None:
        super().__init__(config, prepared)
        fraction = float(config.get("contrast_fraction", 1.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"contrast_fraction must lie in [0, 1], got {fraction}.")
        self.contrast_fraction = fraction
        self.control_summary["donor_pairing"] = "contrastive"
        self.control_summary["contrast_fraction"] = fraction

    def pair(self, split: str, records: dict[str, Tensor], summary: dict) -> dict[str, Tensor]:
        records = super().pair(split, records, summary)
        value = records["state_features"].argmax(-1)
        count = len(value)
        if count < 2:
            raise ValueError(f"{split}: contrastive pairing needs at least two records.")

        generator = torch.Generator().manual_seed(
            self.program_seed + _SPLIT_OFFSET[split]
        )
        # Which rows get a differing donor; the rest are drawn uniformly, so the
        # fraction is a property of the pairing rather than of the data.
        contrast = (
            torch.rand(count, generator=generator) < self.contrast_fraction
        )
        source = torch.randint(count, (count,), generator=generator)
        for level in value.unique():
            mine = torch.where((value == level) & contrast)[0]
            if not len(mine):
                continue
            others = torch.where(value != level)[0]
            if not len(others):
                raise ValueError(
                    f"{split}: no record differs from declared value {int(level)}, "
                    "so a contrastive donor does not exist."
                )
            source[mine] = others[
                torch.randint(len(others), (len(mine),), generator=generator)
            ]
        # A base is never its own donor, whichever arm drew it.
        self_paired = torch.where(source == torch.arange(count))[0]
        for row in self_paired.tolist():
            source[row] = (row + 1) % count

        paired = dict(records)
        paired["source_activation"] = records["activation"][source]
        paired["source_state_features"] = records["state_features"][source]
        summary.update(
            donor_pairing="contrastive",
            contrast_fraction=self.contrast_fraction,
            realised_contrast=float((value[source] != value).float().mean()),
            self_pairs_repaired=int(len(self_paired)),
        )
        return paired
