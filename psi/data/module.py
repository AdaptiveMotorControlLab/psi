import logging

import torch
from torch import Tensor
from typing import Mapping
from collections import defaultdict

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from psi.data.types import TaskCache

# Independent split randomizations, including when lengths happen to agree.
_SPLIT_OFFSET = {"train": 0, "validation": 1, "test": 2}

class TensorRecords(Dataset):
    """A dataset that wraps a mapping of string keys to tensors,
    allowing access to the tensors by index. Each tensor in the
    mapping must have the same length, and the dataset will return
    a dictionary containing the corresponding elements from each
    tensor for a given index.
    """

    def __init__(self, records: Mapping[str, Tensor]) -> None:
        # All records must have the same length
        lengths = {len(tensor) for tensor in records.values() if tensor is not None}
        if len(lengths) > 1:
            raise ValueError("All tensors in records must have the same length.")

        self.records = records

    def __len__(self) -> int:
        return len(next(iter(self.records.values())))

    def __getitem__(self, idx: int) -> Mapping[str, Tensor]:
        return {key: tensor[idx]
            for key, tensor in self.records.items()
            if tensor is not None
        }

class CompatibleBatchSampler:
    """A batch sampler that ensures each batch contains at least two samples.
    This sampler ensure we can always construct derangements for each batch.
    """

    def __init__(
        self,
        size: int,
        batch_size: int,
        *,
        groups: Tensor | None = None,
        shuffle: bool = False,
        mix_groups: bool = True,
        generator : torch.Generator | None = None
    ) -> None:
        if size < 2:
            raise ValueError("Dataset size must be at least 2 to form batches.")
        if batch_size < 2:
            raise ValueError("Batch size must be at least 2 to form batches.")
        if groups is None:
            groups = torch.zeros(size, dtype=torch.long)

        groups = torch.as_tensor(groups, dtype=torch.long).cpu()

        if groups.shape != (size,):
            raise ValueError(f"Groups tensor must have shape ({size},), got {groups.shape}.")

        self.shuffle = bool(shuffle)
        self.generator = generator
        self._fixed_seed = (
            generator.initial_seed() if generator is not None else torch.initial_seed()
        )
        self.batch_size = int(batch_size)
        self.mix_groups = bool(mix_groups)

        self.group_to_indices = {
            int(group): torch.where(groups == group)[0]
            for group in groups.unique()
        }

        # Make sure we never got a singleton group
        if any(len(ids) < 2 for ids in self.group_to_indices.values()):
            raise ValueError("All groups must have at least two samples to form batches.")

        # Build blocks: each block is a tuple (group, size) which can be safely
        # assigned to a batch since it guarantees that at least two samples of
        # the same group are present.
        self.blocks: list[tuple[int, int]] = []
        self.groups_to_blocks: dict[int, list[int]] = {}

        for group, ids in self.group_to_indices.items():
            count = len(ids)
            block_sizes = []

            # If the group has an odd number of samples, split
            # into a triplet and the rest
            if count % 2:
                if batch_size < 3:
                    raise ValueError(f"Odd-sized groups require batch_size >= 3.")
                block_sizes.append(3)
                count -= 3

            # Split the remaining samples into pairs
            block_sizes.extend([2] * (count // 2))

            # Populate the blocks and the mapping from groups to blocks
            block_ids = []
            for size in block_sizes:
                block_ids.append(len(self.blocks))
                self.blocks.append((group, size))
            self.groups_to_blocks[group] = block_ids

        # Pack the blocks into ~full batches
        self.batch_templates: list[list[int]] = []
        current_size = 0
        current_pack = []

        for block_id, (group, block_size) in enumerate(self.blocks):
            if current_pack and current_size + block_size > batch_size:
                self.batch_templates.append(current_pack)
                current_pack = []
                current_size = 0

            current_pack.append(block_id)
            current_size += block_size

        # Append what's left to the last template
        if current_pack:
            self.batch_templates.append(current_pack)

        # Build the size to blocks mapping, used later to reshuffle
        # within blocks of the same size to enable within batch group diversity
        self.size_to_blocks: dict[int, list[int]] = defaultdict(list)
        for block_id, (_, block_size) in enumerate(self.blocks):
            self.size_to_blocks[block_size].append(block_id)

    def __iter__(self):
        # Validation repeats one random partition without advancing any RNG.
        generator = (
            self.generator if self.shuffle
            else torch.Generator().manual_seed(self._fixed_seed)
        )
        mix = self.mix_groups and len(self.group_to_indices) > 1
        block_values: dict[int, Tensor] = {}

        for group, ids in self.group_to_indices.items():
            if self.shuffle or mix:
                ids = ids[
                    torch.randperm(len(ids), generator=generator)
                ]

            # Fill the block_values dictionary with the shuffled
            # ids for this group.
            offset = 0
            for block_id in self.groups_to_blocks[group]:
                _, block_size = self.blocks[block_id]
                block_values[block_id] = ids[offset:offset + block_size]
                offset += block_size

        # Shuffle the blocks of the same size to ensure diversity within batches
        remap = list(range(len(self.blocks)))
        if mix:
            for ids in self.size_to_blocks.values():
                order = torch.randperm(len(ids), generator=generator)
                for slot, choice in zip(ids, order.tolist()):
                    remap[slot] = ids[choice]

        template_order = torch.arange(len(self.batch_templates))
        if self.shuffle:
            template_order = template_order[
                torch.randperm(len(template_order), generator=generator)
            ]

        for template_id in template_order:
            batch = torch.cat(
                [
                    block_values[remap[block_id]]
                    for block_id in self.batch_templates[template_id]
                ]
            )

            if self.shuffle:
                batch = batch[
                    torch.randperm(len(batch), generator=generator)
                ]

            yield batch.tolist()

    def __len__(self) -> int:
        return len(self.batch_templates)


class PSIDataModule(LightningDataModule):
    """A PyTorch Lightning data module that wraps a task preparation
    and provides data loaders for training and validation.
    """

    def __init__(
        self,
        config: Mapping[str, float | int | str],
        prepared: TaskCache,
    ) -> None:
        super().__init__()
        self.data = prepared
        self.config = config

        self.seed = int(config.get("seed", 42))
        self.batch_size = int(config.get("batch_size", 32))
        self.num_workers = int(config.get("num_workers", 4))
        self.donor_balance = str(config.get("donor_balance", "none")).lower()
        self.training_pool = str(config.get("training_pool", "all"))
        if self.training_pool not in {"all", "base_only", "matched_all"}:
            raise ValueError(f"Unknown training_pool: {self.training_pool!r}.")

        if self.donor_balance not in {"none", "inverse_frequency"}:
            raise ValueError(f"Unknown donor_balance: {self.donor_balance!r}.")

        if "program" in config:
            raise ValueError("data.program was withdrawn; use data.shuffle_state_features instead.")
        self.shuffle_state_features = config.get("shuffle_state_features", False)
        if not isinstance(self.shuffle_state_features, bool):
            raise ValueError("shuffle_state_features must be a boolean.")
        self.program_seed = int(config.get("program_seed", 11))
        self.control_summary = {
            "shuffle_state_features": self.shuffle_state_features,
            "program_seed": self.program_seed,
            "donor_balance": self.donor_balance,
            "training_pool": self.training_pool,
            "splits": {},
        }
    
    def pair(self, split: str, records: Mapping[str, Tensor], summary: dict) -> Mapping[str, Tensor]:
        """Extension point for the supervision comparison. A donor-free run
        never pairs: this returns the records untouched, and the donor is drawn
        by derangement inside the batch. See `psi/data/supervised.py`."""
        return records
    
    def setup(self, stage: str | None = None) -> None:
        """Load each split once, then apply only in-memory training controls."""
        match stage:
            case "fit" | None:
                splits = ("train", "validation")
            case "validate":
                splits = ("validation",)
            case "test":
                splits = ("test",)
            case _:
                raise ValueError(f"Unknown stage: {stage}")

        for split in splits:
            records = dict(self.data.records(split))
            summary = {}
            if split == "train" and self.training_pool != "all":
                # The cached base IDs identify observation provenance only.
                # Source IDs, pair assignments and intervention targets are unused.
                pairs = torch.load(
                    self.data.root / split / "pairs.pt",
                    map_location="cpu", weights_only=True,
                )
                correct = self.data.outcomes(split).correct_indices
                base = torch.where(torch.isin(correct, pairs["base_index"].unique()))[0]
                if len(base) < 2:
                    raise ValueError("Training pool needs at least two eligible base observations.")
                selected = base
                if self.training_pool == "matched_all":
                    selected = torch.randperm(
                        len(correct), generator=torch.Generator().manual_seed(self.seed)
                    )[:len(base)].sort().values
                records = {key: value[selected] for key, value in records.items()}
                summary.update(
                    eligible_pool_size=len(correct), selected_pool_size=len(selected),
                    selected_observation_indices=correct[selected].tolist(),
                )
            
            records = self.pair(split, records, summary)
            
            if self.shuffle_state_features:
                features = records["state_features"]
                generator = torch.Generator().manual_seed(
                    self.program_seed + _SPLIT_OFFSET[split]
                )
                order = torch.randperm(len(features), generator=generator)
                records["state_features"] = features[order]
                summary["feature_total_variation_shift"] = float(
                    (features - records["state_features"]).abs().sum(-1).mean() / 2
                )

            if self.donor_balance == "inverse_frequency":
                gate = records.get("transport_gate")
                if (
                    gate is None or gate.ndim != 1
                    or not bool(((gate == 0) | (gate == 1)).all())
                ):
                    raise ValueError("Donor balancing requires a binary per-record transport gate.")
                gate = gate.bool()
                rate = float(gate.float().mean())
                if not 0 < rate < 1:
                    raise ValueError(f"{split} needs both classes for donor balancing; gate fraction={rate}.")
                # Each split estimates the same equal-conditional-risk objective,
                # even when train and validation have different gate frequencies.
                records["donor_weight"] = torch.where(
                    gate, 1 / (2 * rate), 1 / (2 * (1 - rate))
                )
                summary["donor_gate_rate"] = rate

            self.control_summary["splits"][split] = summary
            if summary:
                logging.getLogger(__name__).info("data controls split=%s: %s", split, summary)
            name = "val" if split == "validation" else split
            setattr(self, f"{name}_dataset", TensorRecords(records))

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        dataset = self.train_dataset # type: ignore
        groups = dataset.records.get("transport_group")
        mixing = bool(self.config.get("mix_groups", True))

        batch_sampler = CompatibleBatchSampler(
            size=len(dataset),
            groups=groups,
            batch_size=self.batch_size,
            shuffle=True,
            mix_groups=mixing,
            generator=torch.Generator().manual_seed(self.seed)
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""
        dataset = self.val_dataset # type: ignore
        groups = dataset.records.get("transport_group")
        mixing = bool(self.config.get("mix_groups", True))

        batch_sampler = CompatibleBatchSampler(
            size=len(dataset),
            batch_size=self.batch_size,
            groups=groups,
            shuffle=False,
            mix_groups=mixing,
            generator=torch.Generator().manual_seed(self.seed)
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )
