"""The supervision comparison: an official pair is data, not a sampling flag.

Two axes, each reachable only from config, each defaulting to what the method
already did. The point of these tests is that the *ordinary* path is provably
untouched -- if `donor_pairing` is absent, no pair is ever read and the loss is
the same tensor it was before the hooks existed.

See spec/spec-03-supervision-2x2.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from psi.data import get_datamodule
from psi.data.module import PSIDataModule
from psi.data.supervised import SupervisedPairDataModule
from psi.model.abstraction import LinearAbstraction
from psi.model.subject import LMSubject
from psi.model.transport import SemanticTransport
from psi.psi import ProgrammableSemanticIntervention
from psi.utils.cut import ResidualCut

VOCAB = 40
HIDDEN = 16
PROMPT = 6
OUTCOMES = 3
RAW = 12


def _records(count: int, seed: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(1, VOCAB, (count, PROMPT), generator=generator),
        "attention_mask": torch.ones(count, PROMPT, dtype=torch.long),
        "position_ids": torch.arange(PROMPT).expand(count, PROMPT),
        "outcome_ids": torch.randint(1, VOCAB, (count, OUTCOMES, 1), generator=generator),
        "outcome_mask": torch.ones(count, OUTCOMES, 1, dtype=torch.long),
        "activation": torch.randn(count, HIDDEN, generator=generator),
        "cut_positions": torch.full((count, 1), PROMPT - 1),
        "state_features": (torch.rand(count, OUTCOMES, generator=generator) * 8).softmax(-1),
        "base_behaviour": torch.rand(count, OUTCOMES, generator=generator).softmax(-1),
    }


class _Outcomes:
    def __init__(self, correct: torch.Tensor) -> None:
        self.correct_indices = correct


class _Cache:
    """A cache with the two reads the supervised module makes, and nothing else."""

    def __init__(self, root: Path, correct: torch.Tensor, records) -> None:
        self.root = root
        self._correct = correct
        self._records = records

    def records(self, split: str):
        return dict(self._records)

    def outcomes(self, split: str):
        return _Outcomes(self._correct)


# Raw observations 0..11; 0, 5 and 9 fail correctness filtering, so records are
# the other nine and record position k maps to `correct[k]`.
CORRECT = torch.tensor([1, 2, 3, 4, 6, 7, 8, 10, 11])
# Four pairs survive (both members retained); three die on a filtered member.
PAIRS = {
    "base_index": torch.tensor([1, 1, 2, 3, 4, 6, 7]),
    "source_index": torch.tensor([2, 5, 3, 0, 6, 9, 8]),
    "family_id": torch.zeros(7, dtype=torch.long),
    "example_id": torch.arange(7),
}
SURVIVING = [(1, 2), (2, 3), (4, 6), (7, 8)]


def _cache(tmp_path: Path, records) -> _Cache:
    for split in ("train", "validation"):
        (tmp_path / split).mkdir(parents=True, exist_ok=True)
        torch.save(PAIRS, tmp_path / split / "pairs.pt")
    return _Cache(tmp_path, CORRECT, records)


def _supervised(tmp_path, records, **config) -> SupervisedPairDataModule:
    return SupervisedPairDataModule(
        {"seed": 11, "batch_size": 4, "num_workers": 0, "donor_pairing": "curated", **config},
        _cache(tmp_path, records),  # type: ignore[arg-type]
    )


# --- the ordinary path is untouched ----------------------------------------


def test_the_ordinary_module_never_pairs() -> None:
    records = _records(len(CORRECT))
    module = PSIDataModule({"seed": 11, "batch_size": 4, "num_workers": 0},
                           _Cache(Path("/nonexistent"), CORRECT, records))  # type: ignore[arg-type]
    module.setup("fit")

    # Same objects, not merely equal ones: the hook cannot have re-indexed.
    for key, value in records.items():
        assert module.train_dataset.records[key] is value
    assert "source_activation" not in module.train_dataset.records
    # A pairs.pt read would have raised on this root.


def test_get_datamodule_defaults_to_the_donor_free_module(tmp_path) -> None:
    records = _records(len(CORRECT))
    cache = _cache(tmp_path, records)
    plain = get_datamodule({"seed": 11, "batch_size": 4, "num_workers": 0}, cache)  # type: ignore[arg-type]
    assert type(plain) is PSIDataModule

    curated = get_datamodule(
        {"seed": 11, "batch_size": 4, "num_workers": 0, "donor_pairing": "curated"}, cache  # type: ignore[arg-type]
    )
    assert isinstance(curated, SupervisedPairDataModule)

    with pytest.raises(ValueError, match="donor_pairing"):
        get_datamodule({"donor_pairing": "derangment"}, cache)  # type: ignore[arg-type]


# --- the supervised arm -----------------------------------------------------


def test_curated_pairs_reproduce_the_benchmark_indices(tmp_path) -> None:
    records = _records(len(CORRECT))
    module = _supervised(tmp_path, records)
    module.setup("fit")
    paired = module.train_dataset.records

    assert len(paired["activation"]) == len(SURVIVING)
    where = {int(raw): position for position, raw in enumerate(CORRECT.tolist())}
    for row, (base_raw, source_raw) in enumerate(SURVIVING):
        torch.testing.assert_close(
            paired["activation"][row], records["activation"][where[base_raw]]
        )
        torch.testing.assert_close(
            paired["source_activation"][row], records["activation"][where[source_raw]]
        )
        torch.testing.assert_close(
            paired["source_state_features"][row], records["state_features"][where[source_raw]]
        )
        # Every other column still belongs to the base.
        torch.testing.assert_close(
            paired["base_behaviour"][row], records["base_behaviour"][where[base_raw]]
        )


def test_a_pair_dies_with_either_member(tmp_path) -> None:
    """Attrition is the supervised arm's own cost and has to be reported."""
    module = _supervised(tmp_path, _records(len(CORRECT)))
    module.setup("fit")
    summary = module.control_summary["splits"]["train"]

    assert summary["donor_pairing"] == "curated"
    assert summary["official_pairs"] == len(PAIRS["base_index"])
    assert summary["retained_pairs"] == len(SURVIVING)
    assert summary["distinct_bases"] == len({base for base, _ in SURVIVING})
    assert summary["distinct_sources"] == len({source for _, source in SURVIVING})


@pytest.mark.parametrize(
    "control",
    [{"training_pool": "base_only"}, {"shuffle_state_features": True},
     {"donor_balance": "inverse_frequency"}],
)
def test_curated_pairing_refuses_the_row_reshaping_controls(tmp_path, control) -> None:
    with pytest.raises(ValueError):
        _supervised(tmp_path, _records(len(CORRECT)), **control)


def test_pair_limit_subsamples_deterministically(tmp_path) -> None:
    records = _records(len(CORRECT))
    first = _supervised(tmp_path, records, pair_limit=2)
    first.setup("fit")
    second = _supervised(tmp_path, records, pair_limit=2)
    second.setup("fit")

    assert len(first.train_dataset.records["activation"]) == 2
    assert first.control_summary["splits"]["train"]["retained_pairs"] == 2
    torch.testing.assert_close(
        first.train_dataset.records["source_activation"],
        second.train_dataset.records["source_activation"],
    )
    # A different seed must be free to choose differently, or the limit is not
    # a sample.
    other = _supervised(tmp_path, records, pair_limit=2, seed=4)
    other.setup("fit")
    assert len(other.train_dataset.records["activation"]) == 2


# --- the training step ------------------------------------------------------


def _psi(**kwargs) -> ProgrammableSemanticIntervention:
    torch.manual_seed(3)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    model.eval()
    return ProgrammableSemanticIntervention(
        test_dim=OUTCOMES,
        subject=LMSubject(model, ResidualCut(layer=0, position="last")),
        abstraction=LinearAbstraction(activation_dim=HIDDEN, carrier_dim=6),
        transport=SemanticTransport.from_name("full_change"),
        **kwargs,
    )


def test_a_shipped_source_replaces_the_derangement(monkeypatch) -> None:
    """With a pair in the batch the semantic donor is not sampled at all."""
    psi = _psi()
    batch = _records(6)
    paired = dict(batch)
    paired["source_activation"] = _records(6, seed=9)["activation"]
    paired["source_state_features"] = _records(6, seed=9)["state_features"]

    calls: list[object] = []
    import psi.psi as module

    original = module.derangement
    monkeypatch.setattr(
        module, "derangement",
        lambda *args, **kwargs: (calls.append(kwargs.get("groups")), original(*args, **kwargs))[1],
    )

    with torch.no_grad():
        psi.loss(paired)
    supervised_calls = len(calls)
    calls.clear()
    with torch.no_grad():
        psi.loss(batch)

    # Nuisance keeps its derangement in both arms; the semantic donor is drawn
    # only when no official source was shipped.
    assert supervised_calls == 1
    assert len(calls) == 2


def test_the_shipped_source_intervenes_with_its_own_carrier(monkeypatch) -> None:
    psi = _psi()
    batch = _records(6)
    source = _records(6, seed=9)
    paired = dict(batch)
    paired["source_activation"] = source["activation"]
    paired["source_state_features"] = source["state_features"]

    steers: list[torch.Tensor] = []
    original = psi.subject.outcome_log_probability

    def record(batch_, steer=None):
        if steer is not None:
            steers.append(steer[1].detach().clone())
        return original(batch_, steer)

    monkeypatch.setattr(psi.subject, "outcome_log_probability", record)
    with torch.no_grad():
        psi.loss(paired)

    _, nuisance = psi.abstraction.lift(batch["activation"].float())
    _, source_nuisance = psi.abstraction.lift(source["activation"].float())
    expected = (source["activation"].float() - source_nuisance) + nuisance
    torch.testing.assert_close(steers[0], expected)


def test_the_donor_free_loss_is_unchanged_by_the_hooks() -> None:
    """The regression guard: no pair, no behaviour change, to the last bit."""
    batch = _records(6)
    losses = []
    for _ in range(2):
        psi = _psi()
        torch.manual_seed(77)
        with torch.no_grad():
            losses.append(float(psi.loss(batch).loss))
    assert losses[0] == losses[1]


def test_hard_targets_are_one_hot_at_the_soft_argmax() -> None:
    batch = _records(6)
    soft, hard = _psi(), _psi(target_kind="hard")

    with torch.no_grad():
        torch.manual_seed(5)
        soft_loss = float(soft.loss(batch).loss)
        torch.manual_seed(5)
        hard_loss = float(hard.loss(batch).loss)

    # Same donor draw from the same seed, so any difference is the target.
    assert soft_loss != hard_loss

    with torch.no_grad():
        target = soft.transport(
            source_state=batch["state_features"],
            base_behaviour=batch["base_behaviour"],
            transport_gate=None,
        )
    hardened = torch.nn.functional.one_hot(target.argmax(-1), OUTCOMES).to(target.dtype)
    torch.testing.assert_close(hardened.sum(-1), torch.ones(len(target)))
    torch.testing.assert_close(hardened.argmax(-1), target.argmax(-1))
    # A soft target that is already one-hot would make the arms identical, so
    # the fixture has to be genuinely soft.
    assert float(target.max(-1).values.min()) < 0.99


def test_an_unrecognised_target_kind_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError), match="target_kind"):
        _psi(target_kind="hrad")


@pytest.mark.parametrize("kind", ["soft", "hard", "gold"])
@pytest.mark.parametrize("paired", [False, True])
def test_wrong_donor_targets_share_the_activation_donor(monkeypatch, kind, paired):
    import psi.psi as module
    from psi.model.objective import LossWeights

    psi = _psi(target_kind=kind, loss_weights=LossWeights(donor=1, nuisance=0, calibration=0, control=0))
    batch = _records(6)
    batch["gold_state_features"] = torch.nn.functional.one_hot(
        (batch["state_features"].argmax(-1) + 1) % OUTCOMES, OUTCOMES).float()
    donor = torch.tensor([1, 2, 3, 4, 5, 0])
    expected_soft = batch["state_features"][donor]
    expected_gold = batch["gold_state_features"][donor]
    if paired:
        source = _records(6, seed=9)
        batch["source_activation"] = source["activation"]
        batch["source_state_features"] = source["state_features"]
        batch["source_gold_state_features"] = torch.nn.functional.one_hot(
            (source["state_features"].argmax(-1) + 1) % OUTCOMES, OUTCOMES).float()
        expected_soft = batch["source_state_features"]
        expected_gold = batch["source_gold_state_features"]
    calls, targets = [], []
    monkeypatch.setattr(module, "derangement", lambda *a, **kw: (calls.append(1), donor)[1])
    original = module.cross_entropy
    def measured(logp, target, *a, **kw):
        targets.append(target.detach().clone())
        return original(logp, target, *a, **kw)
    monkeypatch.setattr(module, "cross_entropy", measured)
    with torch.no_grad():
        psi.loss(batch)
    assert len(calls) == (1 if paired else 2), "target selection must not draw an independent donor"
    assert len(targets) == 1
    expected = {"soft": expected_soft, "hard": torch.nn.functional.one_hot(
        expected_soft.argmax(-1), OUTCOMES).float(), "gold": expected_gold}[kind]
    torch.testing.assert_close(targets[0], expected)
    assert not torch.equal(expected_soft.argmax(-1), expected_gold.argmax(-1))


@pytest.mark.parametrize("coupling", ["curated", "shuffled"])
def test_supplied_gold_values_follow_the_final_source_rows(tmp_path, coupling):
    records = _records(len(CORRECT))
    records["gold_state_features"] = torch.nn.functional.one_hot(
        torch.arange(len(CORRECT)) % OUTCOMES, OUTCOMES).float()
    module = _supervised(tmp_path, records, pair_coupling=coupling)
    module.setup("fit")
    paired = module.train_dataset.records
    for activation, gold in zip(paired["source_activation"], paired["source_gold_state_features"]):
        original = (records["activation"] == activation).all(-1).nonzero().item()
        torch.testing.assert_close(gold, records["gold_state_features"][original])


@pytest.mark.parametrize("kind", ["soft", "hard"])
def test_model_targets_ignore_perturbed_gold_values(kind):
    psi = _psi(target_kind=kind)
    batch = _records(6)
    batch["gold_state_features"] = torch.nn.functional.one_hot(torch.arange(6) % OUTCOMES, OUTCOMES).float()
    with torch.no_grad():
        before = psi.loss(batch, generator=torch.Generator().manual_seed(9)).loss
        batch["gold_state_features"] = batch["gold_state_features"].roll(1, -1)
        after = psi.loss(batch, generator=torch.Generator().manual_seed(9)).loss
    torch.testing.assert_close(before, after, rtol=0, atol=0)
