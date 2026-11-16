"""The donor loss must be able to weight its two gate classes equally.

On RAVEL the donor term averages two different measurements: efficacy on the
records whose queried attribute is the target, and preservation on the rest.
The target class is about 22% of the training records, so a plain mean lets
efficacy be traded away cheaply.

These tests pin the estimand — the unweighted average of the two conditional
donor risks — rather than the arithmetic that happens to produce it. See
docs/spec-03-donor-gate-balance.md.
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
RECORDS = 500
GATE_RATE = 0.22


def _gate(count: int = RECORDS, rate: float = GATE_RATE) -> torch.Tensor:
    """A deterministic gate column with an exact target fraction."""
    gate = torch.zeros(count, dtype=torch.bool)
    gate[: int(round(count * rate))] = True
    return gate


def _weights(gate: torch.Tensor) -> torch.Tensor:
    rate = float(gate.float().mean())
    return torch.where(
        gate,
        torch.full(gate.shape, 1.0 / (2.0 * rate)),
        torch.full(gate.shape, 1.0 / (2.0 * (1.0 - rate))),
    )


class _Cache:
    """The one method PSIDataModule.setup needs from a TaskCache."""

    def __init__(self, records) -> None:
        self._records = records

    def records(self, split: str):
        return dict(self._records)


def _module(records, **config) -> PSIDataModule:
    return PSIDataModule(
        {"seed": 1, "batch_size": 4, "num_workers": 0, **config}, _Cache(records)
    )


def test_weights_equalize_the_two_conditional_risks() -> None:
    """The weighted mean must be the mean of the two conditional means.

    This is the whole claim. If it holds, the donor term no longer cares that one
    class is four times rarer than the other.
    """
    gate = _gate()
    generator = torch.Generator().manual_seed(5)
    loss = torch.rand(RECORDS, generator=generator) * 3

    weighted = (_weights(gate) * loss).mean()
    conditional = 0.5 * loss[gate].mean() + 0.5 * loss[~gate].mean()

    torch.testing.assert_close(weighted, conditional)


def test_balanced_weights_average_to_one() -> None:
    """The donor term must keep its scale, or the loss weights change meaning."""
    torch.testing.assert_close(
        _weights(_gate()).mean(), torch.tensor(1.0), rtol=1e-6, atol=1e-6
    )


def test_balance_is_off_by_default() -> None:
    """Existing configs must train exactly as before."""
    records = {
        "activation": torch.randn(8, HIDDEN),
        "transport_gate": _gate(8, 0.25),
    }
    module = _module(records)
    module.setup("fit")

    assert "donor_weight" not in module.train_dataset.records
    assert getattr(module, "donor_gate_rate", None) is None


def test_missing_gate_class_is_a_protocol_error() -> None:
    """A two-class objective with one class is a protocol error, not a warning."""
    records = {
        "activation": torch.randn(8, HIDDEN),
        "transport_gate": torch.ones(8, dtype=torch.bool),
    }
    module = _module(records, donor_balance="inverse_frequency")
    with pytest.raises(ValueError, match="both classes"):
        module.setup("fit")


def _psi() -> tuple[ProgrammableSemanticIntervention, dict[str, torch.Tensor]]:
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
    subject = LMSubject(model, ResidualCut(layer=0, position="last"))

    psi = ProgrammableSemanticIntervention(
        test_dim=3,
        subject=subject,
        abstraction=LinearAbstraction(activation_dim=HIDDEN, carrier_dim=6),
        transport=SemanticTransport.from_name("gated_change"),
    )

    generator = torch.Generator().manual_seed(9)
    size = 8
    gate = _gate(size, 0.25)
    batch = {
        "input_ids": torch.randint(1, VOCAB, (size, PROMPT), generator=generator),
        "attention_mask": torch.ones(size, PROMPT, dtype=torch.long),
        "position_ids": torch.arange(PROMPT).expand(size, PROMPT),
        "outcome_ids": torch.randint(1, VOCAB, (size, 3, 1), generator=generator),
        "outcome_mask": torch.ones(size, 3, 1, dtype=torch.long),
        "activation": torch.randn(size, HIDDEN, generator=generator),
        "cut_positions": torch.full((size, 1), PROMPT - 1),
        "state_features": torch.rand(size, 3, generator=generator).softmax(-1),
        "base_behaviour": torch.rand(size, 3, generator=generator).softmax(-1),
        "transport_gate": gate,
    }
    return psi, batch


def test_donor_weight_changes_only_the_donor_term() -> None:
    """Weighting must touch the donor term and leave the other two alone."""
    psi, batch = _psi()

    torch.manual_seed(17)
    plain = psi.loss(batch)
    torch.manual_seed(17)
    weighted = psi.loss({**batch, "donor_weight": _weights(batch["transport_gate"])})

    torch.testing.assert_close(plain.calibration_loss, weighted.calibration_loss)
    torch.testing.assert_close(plain.nuisance_loss, weighted.nuisance_loss)
    assert not torch.allclose(plain.donor_loss, weighted.donor_loss)

    # An all-ones weight must be exactly the unweighted reduction.
    torch.manual_seed(17)
    ones = psi.loss({**batch, "donor_weight": torch.ones(len(batch["activation"]))})
    torch.testing.assert_close(plain.donor_loss, ones.donor_loss)


def test_conditional_donor_losses_are_reported() -> None:
    """The claim is about two conditional means, so both must be visible."""
    psi, batch = _psi()

    torch.manual_seed(17)
    result = psi.loss(batch)

    assert "support/donor_loss_gate_true" in result.metrics
    assert "support/donor_loss_gate_false" in result.metrics
    assert "support/gate_fraction" in result.metrics

    # A batch that happens to hold one class is not an error; it reports what it has.
    torch.manual_seed(17)
    single = psi.loss({**batch, "transport_gate": torch.zeros(8, dtype=torch.bool)})
    assert "support/donor_loss_gate_true" not in single.metrics
    assert "support/donor_loss_gate_false" in single.metrics


def test_each_split_estimates_equal_conditional_risk_even_with_different_rates() -> None:
    class Cache:
        def records(self, split):
            return {"activation": torch.zeros(8, HIDDEN), "transport_gate": _gate(8, .25 if split == "train" else .75)}
    module = PSIDataModule({"donor_balance": "inverse_frequency"}, Cache())
    module.setup("fit")
    losses = torch.arange(8).float().square()
    for dataset in (module.train_dataset, module.val_dataset):
        gate = dataset.records["transport_gate"]
        weight = dataset.records["donor_weight"]
        torch.testing.assert_close((weight * losses).mean(), .5 * (losses[gate].mean() + losses[~gate].mean()))
        torch.testing.assert_close(weight.mean(), torch.tensor(1.))
    expected = module.val_dataset.records["donor_weight"].clone()
    module.setup("validate")
    torch.testing.assert_close(module.val_dataset.records["donor_weight"], expected)


@pytest.mark.parametrize("gate", [None, torch.eye(3).expand(8, 3, 3), torch.full((8,), .5), torch.zeros(8)])
def test_balancing_rejects_missing_nonbinary_or_degenerate_gates(gate) -> None:
    module = _module({"activation": torch.zeros(8, HIDDEN), "transport_gate": gate}, donor_balance="inverse_frequency")
    with pytest.raises(ValueError, match="binary|both classes"):
        module.setup("fit")


def test_conditional_epoch_logging_uses_class_counts() -> None:
    psi, batch = _psi()
    logged = {}
    psi.log_dict = lambda *args, **kwargs: None
    psi.log = lambda name, value, **kwargs: logged.update({name: kwargs["batch_size"]})
    psi.training_step(batch, 0)
    assert logged == {"train/support/donor_loss_gate_true": 2, "train/support/donor_loss_gate_false": 6}
