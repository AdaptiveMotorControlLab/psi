"""A batch must contain the groups the free nuisance derangement needs to reach.

`ProgrammableSemanticIntervention.loss` draws the nuisance donor with
`derangement(n)` and no group constraint, so the batch sampler decides what
"a different context" can mean. `CompatibleBatchSampler` emits each group's
blocks consecutively and packs them in that order, which makes most batches
single-group and quietly turns the unconstrained derangement into a
within-group one.

These tests assert statistical support, not just that every record is visited
and has a compatible peer. See docs/spec-02-batch-group-mixing.md.
"""

from __future__ import annotations

import torch

from psi.data.module import CompatibleBatchSampler
from psi.utils.misc import derangement

GROUPS = 64
PER_GROUP = 32
BATCH = 16


def _fixture() -> torch.Tensor:
    """64 groups of 32 records: enough groups that mixing is clearly possible."""
    return torch.arange(GROUPS).repeat_interleave(PER_GROUP)


def _schedule(groups: torch.Tensor, *, seed: int = 11, **kwargs) -> list[list[int]]:
    return list(
        CompatibleBatchSampler(
            len(groups),
            BATCH,
            groups=groups,
            shuffle=kwargs.pop("shuffle", True),
            generator=torch.Generator().manual_seed(seed),
            **kwargs,
        )
    )


def _cross_group_rate(batches: list[list[int]], groups: torch.Tensor) -> float:
    """The share of nuisance donors that come from another group."""
    torch.manual_seed(0)
    crossed = 0
    total = 0
    for batch in batches:
        member_groups = groups[torch.tensor(batch)]
        donor = derangement(len(batch))
        crossed += int((member_groups[donor] != member_groups).sum())
        total += len(batch)
    return crossed / total


def test_mixed_batches_reach_other_groups() -> None:
    """The unconstrained derangement must actually be unconstrained.

    With 64 groups the derangement over the whole dataset would cross groups
    almost always. Anything close to zero means the batch sampler, not the
    objective, is deciding what the nuisance term tests.
    """
    groups = _fixture()
    rate = _cross_group_rate(_schedule(groups), groups)

    assert rate > 0.8, (
        f"Nuisance donors reach another group on only {rate:.1%} of positions."
    )


def test_mixing_preserves_the_batch_schedule() -> None:
    """Mixing must not change how many batches there are, or how big they are.

    Blocks are permuted within their size class precisely so that every
    template slot still receives a block of the size it was packed for. If this
    fails, `__len__` no longer describes the iteration.
    """
    groups = _fixture()
    sampler = CompatibleBatchSampler(
        len(groups),
        BATCH,
        groups=groups,
        shuffle=True,
        generator=torch.Generator().manual_seed(11),
    )

    mixed = list(sampler)
    unmixed = _schedule(groups, mix_groups=False)

    assert len(mixed) == len(sampler)
    assert sorted(len(batch) for batch in mixed) == sorted(
        len(batch) for batch in unmixed
    )


def test_every_record_keeps_a_group_mate() -> None:
    """The repair must not be bought by dropping the derangeability guarantee."""
    groups = _fixture()
    batches = _schedule(groups)

    for batch in batches:
        _, counts = groups[torch.tensor(batch)].unique(return_counts=True)
        assert torch.all(counts >= 2)
        # The constrained donor must exist for every member of every batch.
        donor = derangement(len(batch), groups=groups[torch.tensor(batch)])
        assert torch.all(donor != torch.arange(len(batch)))

    assert sorted(index for batch in batches for index in batch) == list(
        range(len(groups))
    )


def test_mix_groups_false_reproduces_the_current_schedule() -> None:
    """The old schedule stays reachable so the change can be attributed.

    The term factorials were trained under consecutive packing. Comparing them
    against mixed batching is only a controlled comparison if the old schedule
    is still exactly expressible.
    """
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    first = _schedule(groups, seed=4, mix_groups=False)
    second = _schedule(groups, seed=4, mix_groups=False)

    assert first == second
    for batch in first:
        _, counts = groups[torch.tensor(batch)].unique(return_counts=True)
        assert torch.all(counts >= 2)


def test_validation_batches_mix_groups_like_training() -> None:
    """Checkpoint selection must see the distribution training optimizes.

    `val/loss` includes the nuisance term, and the nuisance term is defined by
    which groups share a batch. If validation kept the consecutive packing while
    training mixed, the monitored quantity would be a within-group swap while the
    optimized one was a cross-group swap, and `ModelCheckpoint` would be
    selecting on a different objective than the one being trained.
    """
    groups = _fixture()

    validation = _cross_group_rate(_schedule(groups, shuffle=False), groups)
    training = _cross_group_rate(_schedule(groups, shuffle=True), groups)

    assert validation > 0.8
    assert abs(validation - training) < 0.1

    # Mixing is still something validation can decline, which is what keeps the
    # unmixed arm of the batching comparison expressible on both splits.
    unmixed = _cross_group_rate(
        _schedule(groups, shuffle=False, mix_groups=False), groups
    )
    assert unmixed == 0.0


def test_validation_partition_repeats_without_advancing_generators() -> None:
    groups = _fixture()
    generator = torch.Generator().manual_seed(11)
    before = generator.get_state().clone()
    global_before = torch.random.get_rng_state().clone()
    sampler = CompatibleBatchSampler(len(groups), BATCH, groups=groups, shuffle=False, generator=generator)
    assert list(sampler) == list(sampler)
    torch.testing.assert_close(generator.get_state(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), global_before)


def test_single_group_mixing_is_inert_including_rng_consumption() -> None:
    groups = torch.zeros(35, dtype=torch.long)
    for shuffle in (True, False):
        generators = [torch.Generator().manual_seed(11) for _ in range(2)]
        samplers = [CompatibleBatchSampler(len(groups), BATCH, groups=groups, shuffle=shuffle, mix_groups=mix, generator=g) for mix, g in zip((False, True), generators)]
        for _ in range(2):
            assert list(samplers[0]) == list(samplers[1])
            torch.testing.assert_close(generators[0].get_state(), generators[1].get_state())
