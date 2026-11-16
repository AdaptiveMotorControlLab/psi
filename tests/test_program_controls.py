"""Fixed feature corruption isolates correspondence from measured behaviour.

Only the feature column changes; cached audit targets and all other record
columns retain their original alignment. See spec 05.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from psi.data.module import PSIDataModule
from psi.model.abstraction import LinearAbstraction
from psi.model.subject import LMSubject
from psi.model.transport import SemanticTransport
from psi.psi import ProgrammableSemanticIntervention
from psi.utils.cut import ResidualCut

VOCAB = 40
HIDDEN = 16
PROMPT = 6
OUTCOMES = 3
RECORDS = 64


def _records(seed: int = 5, count: int = RECORDS) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(1, VOCAB, (count, PROMPT), generator=generator),
        "attention_mask": torch.ones(count, PROMPT, dtype=torch.long),
        "position_ids": torch.arange(PROMPT).expand(count, PROMPT),
        "outcome_ids": torch.randint(1, VOCAB, (count, OUTCOMES, 1), generator=generator),
        "outcome_mask": torch.ones(count, OUTCOMES, 1, dtype=torch.long),
        "activation": torch.randn(count, HIDDEN, generator=generator),
        "cut_positions": torch.full((count, 1), PROMPT - 1),
        # Deliberately spread out, so a permutation changes the
        # declaration rather than swapping near-identical rows.
        "state_features": (
            torch.rand(count, OUTCOMES, generator=generator) * 8
        ).softmax(-1),
        "base_behaviour": torch.rand(count, OUTCOMES, generator=generator).softmax(-1),
    }


class _Cache:
    """Only the public records interface is needed; no extra cache reads."""

    def __init__(self, records) -> None:
        self._records = records

    def records(self, split: str):
        return dict(self._records)


def _module(records, **config) -> PSIDataModule:
    return PSIDataModule(
        {"seed": 11, "batch_size": 8, "num_workers": 0, **config},
        _Cache(records),  # type: ignore[arg-type]
    )


def test_correct_program_leaves_the_records_alone() -> None:
    """Existing configs must train against exactly what they trained against.

    This is the regression guard: it passes before the change and must keep
    passing after it.
    """
    records = _records()
    module = _module(records)
    module.setup("fit")

    torch.testing.assert_close(
        module.train_dataset.records["state_features"], records["state_features"]
    )
    torch.testing.assert_close(
        module.train_dataset.records["base_behaviour"], records["base_behaviour"]
    )


def test_shuffled_features_preserve_marginal_and_other_columns() -> None:
    """An ordinary permutation changes correspondence, preserving measurements."""
    records = _records()
    module = _module(records, shuffle_state_features=True)
    module.setup("fit")

    shuffled = module.train_dataset.records["state_features"]
    original = records["state_features"]

    expected = torch.randperm(RECORDS, generator=torch.Generator().manual_seed(11))
    torch.testing.assert_close(shuffled, original[expected])
    for key, value in records.items():
        if key != "state_features":
            assert module.train_dataset.records[key] is value
    torch.testing.assert_close(records["state_features"], original)
    # The marginal distribution of declarations is preserved: this is a
    # relabelling, not a corruption of the feature space.
    torch.testing.assert_close(
        shuffled.sort(0).values, original.sort(0).values
    )
    # The nuisance term's target is a preservation constraint on real behaviour
    # and is not part of the declaration.
    torch.testing.assert_close(
        module.train_dataset.records["base_behaviour"], records["base_behaviour"]
    )


def test_shuffled_program_is_fixed_across_epochs() -> None:
    """The program is a property of the data, not of sampling."""
    records = _records()

    first = _module(records, shuffle_state_features=True)
    first.setup("fit")
    second = _module(records, shuffle_state_features=True)
    second.setup("fit")

    torch.testing.assert_close(
        first.train_dataset.records["state_features"],
        second.train_dataset.records["state_features"],
    )
    # Train and validation must not receive the same permutation.
    assert not torch.equal(
        first.train_dataset.records["state_features"],
        first.val_dataset.records["state_features"],
    )


def test_program_shift_is_reported() -> None:
    """A vacuous wrong-program control must be visible as a number."""
    records = _records()
    module = _module(records, shuffle_state_features=True)
    module.setup("fit")

    shift = module.control_summary["splits"]["train"]["feature_total_variation_shift"]

    expected = float(
        (records["state_features"] - module.train_dataset.records["state_features"])
        .abs().sum(-1).mul(0.5).mean()
    )
    assert shift == pytest.approx(expected)
    assert 0.0 < shift <= 1.0



def test_control_seed_is_independent_of_training_seed_and_setup_stage() -> None:
    records = _records()
    first = _module(records, seed=11, shuffle_state_features=True, program_seed=37)
    second = _module(records, seed=13, shuffle_state_features=True, program_seed=37)
    before = torch.random.get_rng_state().clone()
    first.setup("fit")
    second.setup("validate")
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    torch.testing.assert_close(first.val_dataset.records["state_features"], second.val_dataset.records["state_features"])
    previous = first.train_dataset.records["state_features"].clone()
    first.setup("fit")
    torch.testing.assert_close(first.train_dataset.records["state_features"], previous)


@pytest.mark.parametrize("config", [{"program": "shuffled"}, {"program": "variable:o_answer"}, {"shuffle_state_features": "false"}])
def test_withdrawn_or_mistyped_control_is_rejected(config) -> None:
    with pytest.raises(ValueError):
        _module(_records(), **config)


def test_wrong_program_changes_donor_and_calibration_but_not_nuisance() -> None:
    """The substitution must isolate the declaration from the preservation term."""
    torch.manual_seed(3)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    model.eval()
    psi = ProgrammableSemanticIntervention(
        test_dim=OUTCOMES,
        subject=LMSubject(model, ResidualCut(layer=0, position="last")),
        abstraction=LinearAbstraction(
            activation_dim=HIDDEN, carrier_dim=6
        ),
        transport=SemanticTransport.from_name("full_change"),
    )

    records = _records(count=8)
    correct = _module(records)
    correct.setup("fit")
    wrong = _module(records, shuffle_state_features=True)
    wrong.setup("fit")

    def _batch(module):
        return {key: value for key, value in module.train_dataset.records.items()}

    torch.manual_seed(101)
    with torch.no_grad():
        reference = psi.loss(_batch(correct))
    torch.manual_seed(101)
    with torch.no_grad():
        substituted = psi.loss(_batch(wrong))

    torch.testing.assert_close(
        reference.nuisance_loss, substituted.nuisance_loss
    )
    assert not torch.isclose(reference.donor_loss, substituted.donor_loss)
    assert not torch.isclose(
        reference.calibration_loss, substituted.calibration_loss
    )
