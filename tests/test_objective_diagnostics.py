"""The donor objective's scale must be readable from the log, not inferred.

For a target that sums to one, the donor cross-entropy decomposes exactly into
`H(t) + KL(t || rho) - log Z`, where `Z` is the total vocabulary mass on the
declared outcomes. `H` is constant in the parameters, so the objective trades
matching the ratio against buying mass. On IOI SToken the mass side apparently
won, producing a carrier worse than a matched random subspace, and nothing in
metrics.csv said so.

These tests pin the decomposition and the sampled gradient probe.
See docs/spec-04-objective-diagnostics.md.
"""

from __future__ import annotations

import torch
from torch.nn.functional import mse_loss
from transformers import LlamaConfig, LlamaForCausalLM

from psi.model.abstraction import LinearAbstraction
from psi.model.objective import LossWeights
from psi.model.subject import LMSubject
from psi.model.transport import SemanticTransport
from psi.psi import ProgrammableSemanticIntervention
from psi.utils.cut import ResidualCut

VOCAB = 40
HIDDEN = 16
PROMPT = 6
OUTCOMES = 3


def _psi(**kwargs) -> tuple[ProgrammableSemanticIntervention, dict[str, torch.Tensor]]:
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
        test_dim=OUTCOMES,
        subject=subject,
        abstraction=LinearAbstraction(
            activation_dim=HIDDEN, carrier_dim=6
        ),
        transport=SemanticTransport.from_name("full_change"),
        **kwargs,
    )

    generator = torch.Generator().manual_seed(9)
    size = 8
    batch = {
        "input_ids": torch.randint(1, VOCAB, (size, PROMPT), generator=generator),
        "attention_mask": torch.ones(size, PROMPT, dtype=torch.long),
        "position_ids": torch.arange(PROMPT).expand(size, PROMPT),
        "outcome_ids": torch.randint(1, VOCAB, (size, OUTCOMES, 1), generator=generator),
        "outcome_mask": torch.ones(size, OUTCOMES, 1, dtype=torch.long),
        "activation": torch.randn(size, HIDDEN, generator=generator),
        "cut_positions": torch.full((size, 1), PROMPT - 1),
        "state_features": torch.rand(size, OUTCOMES, generator=generator).softmax(-1),
        "base_behaviour": torch.rand(size, OUTCOMES, generator=generator).softmax(-1),
    }
    return psi, batch


def test_cross_entropy_decomposition_is_exact() -> None:
    """H(t) + KL(t || rho) - log Z is the cross-entropy, not an approximation.

    Stated as arithmetic with no psi dependency, so the other tests have
    something to be checked against rather than checked by.
    """
    generator = torch.Generator().manual_seed(1)
    log_probability = torch.rand(5, OUTCOMES, generator=generator).log() - 2.0
    target = torch.rand(5, OUTCOMES, generator=generator).softmax(-1)

    cross_entropy = -(target * log_probability).sum(-1)

    log_mass = log_probability.logsumexp(-1)
    conditional = log_probability - log_mass.unsqueeze(-1)
    target_log = target.clamp_min(torch.finfo(target.dtype).tiny).log()
    entropy = -(target * target_log).sum(-1)
    divergence = (target * (target_log - conditional)).sum(-1)

    torch.testing.assert_close(cross_entropy, entropy + divergence - log_mass)


def test_diagnostics_report_the_decomposition() -> None:
    """A real loss() call must report the three terms, and they must add up."""
    psi, batch = _psi()

    with torch.no_grad():
        result = psi.loss(batch)

    metrics = result.metrics
    for name in (
        "support/donor_entropy",
        "support/donor_conditional_kl",
        "support/donor_outcome_mass",
        "support/donor_target_mass",
        "support/nuisance_outcome_mass",
    ):
        assert name in metrics, f"{name} is not reported."

    # Targets come from a softmax, so the decomposition is applicable.
    torch.testing.assert_close(
        metrics["support/donor_target_mass"],
        torch.ones_like(metrics["support/donor_target_mass"]),
        rtol=1e-5,
        atol=1e-5,
    )

    # The identity holds on the batch means because every term is a mean over
    # the same batch.
    reconstructed = (
        metrics["support/donor_entropy"]
        + metrics["support/donor_conditional_kl"]
        - metrics["support/donor_log_outcome_mass"]
    )
    torch.testing.assert_close(reconstructed, result.donor_loss)


def test_behavior_is_the_softmax_of_the_kept_log_probabilities() -> None:
    """The premise the nuisance refactor rests on.

    Spec 04 replaces `subject.behavior(...)` with
    `subject.outcome_log_probability(...).softmax(-1)` so the declared-outcome
    mass survives. That is only a free change if the two are the same thing.
    This passes before the change and must keep passing after it.
    """
    psi, batch = _psi()
    steer = (batch["cut_positions"], torch.randn_like(batch["activation"]))

    with torch.no_grad():
        log_probability = psi.subject.outcome_log_probability(batch, steer)
        behavior = psi.subject.behavior(batch, steer)

    torch.testing.assert_close(log_probability.softmax(-1), behavior)
    # And the mass the refactor exists to expose is a genuine probability.
    mass = log_probability.logsumexp(-1).exp()
    assert bool(((mass > 0) & (mass <= 1.0 + 1e-5)).all())


def test_gradient_probe_reports_three_terms() -> None:
    """Loss weights are not gradient influence; the probe measures the latter."""
    psi, batch = _psi(gradient_probe_every=1)
    logged: dict[str, torch.Tensor] = {}
    psi.log_dict = lambda values, **kwargs: logged.update(values)  # type: ignore[method-assign]

    loss = psi.training_step(batch, 0)
    loss.backward()

    for name in ("grad/donor", "grad/nuisance", "grad/calibration"):
        assert name in logged, f"{name} was not probed."
        assert torch.isfinite(logged[name]), f"{name} is not finite."


def test_zero_weight_term_probes_as_zero() -> None:
    """A term at weight zero must probe as zero rather than raise.

    The term factorial runs cells with calibration or nuisance at 0.0. If the
    probe raised on a term that no longer requires grad, the probe could not be
    switched on for the experiment it was written for.
    """
    psi, batch = _psi(
        gradient_probe_every=1,
        loss_weights=LossWeights(donor=1.0, nuisance=0.0, calibration=0.0),
    )
    logged: dict[str, torch.Tensor] = {}
    psi.log_dict = lambda values, **kwargs: logged.update(values)  # type: ignore[method-assign]

    psi.training_step(batch, 0)

    assert float(logged["grad/nuisance"]) == 0.0
    assert float(logged["grad/calibration"]) == 0.0
    assert float(logged["grad/donor"]) > 0.0


def test_weighted_decomposition_reconstructs_balanced_ce() -> None:
    psi, batch = _psi()
    batch["donor_weight"] = torch.tensor([2., 2., 2/3, 2/3, 2/3, 2/3, 2/3, 2/3])
    result = psi.loss(batch)
    d = result.diagnostics
    torch.testing.assert_close(result.donor_loss, d["weighted_donor_entropy"] + d["weighted_donor_conditional_kl"] - d["weighted_donor_log_outcome_mass"])
    assert all(not value.requires_grad for value in d.values())


def test_table_transport_is_not_mistaken_for_a_binary_gate() -> None:
    psi, batch = _psi()
    psi.transport = SemanticTransport.from_name("table")
    batch["transport_gate"] = torch.eye(OUTCOMES).expand(8, OUTCOMES, OUTCOMES)
    result = psi.loss(batch)
    assert torch.isfinite(result.loss)
    assert "gate_fraction" not in result.diagnostics
    torch.testing.assert_close(result.diagnostics["donor_target_mass"], torch.tensor(1.))


def test_gradient_probe_does_not_change_the_optimizer_update() -> None:
    plain, batch = _psi()
    probed, _ = _psi(gradient_probe_every=1)
    for psi in (plain, probed):
        psi.log_dict = lambda *args, **kwargs: None
        optimizer = psi.configure_optimizers()
        torch.manual_seed(19)
        loss = psi.training_step(batch, 0)
        assert all(p.grad is None for p in psi.parameters())
        loss.backward()
        assert all(p.grad is None for p in psi.subject.model.parameters())
        optimizer.step()
    for first, second in zip(plain.parameters(), probed.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_validation_repeats_donors_without_advancing_training_rng() -> None:
    psi, batch = _psi()
    psi.eval()
    psi.log_dict = lambda *args, **kwargs: None
    before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = psi.validation_step(batch, 0)
        second = psi.validation_step(batch, 0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before)

def test_cpu_training_writes_checkpoint_metrics_and_control_metadata(tmp_path, monkeypatch) -> None:
    """Exercise the real Trainer/logging/checkpoint path without a downloaded LM."""
    import csv
    import json
    from types import SimpleNamespace
    import torch
    import psi.train as training

    psi, records = _psi(gradient_probe_every=1)
    records["transport_gate"] = torch.tensor([True, True, False, False, False, False, False, False])
    cache = SimpleNamespace(records=lambda split: dict(records))
    task = SimpleNamespace(num_outcomes=3, materialize=lambda subject: cache)
    monkeypatch.setattr(training, "get_subject", lambda config: (None, psi.subject))
    monkeypatch.setattr(training, "get_task", lambda config, tokenizer: task)
    monkeypatch.setattr(training.ProgrammableSemanticIntervention, "from_config", lambda config, subject: psi)
    config = {
        "seed": 11,
        "data": {"seed": 11, "batch_size": 4, "num_workers": 0, "shuffle_state_features": True, "program_seed": 37, "donor_balance": "inverse_frequency"},
        "psi": {"test_dim": 3},
        "trainer": {"max_epochs": 2, "accelerator": "cpu", "devices": 1, "enable_progress_bar": False, "enable_model_summary": False},
        "checkpoint": {"monitor": "val/loss", "mode": "min", "save_top_k": 1},
    }
    directory = training.run_training(config, run_dir=tmp_path / "run")
    assert (directory / "resolved.yaml").is_file()
    assert (directory / "checkpoints" / "best.ckpt").is_file()
    metadata = json.loads((directory / "data_controls.json").read_text())
    assert metadata["program_seed"] == 37
    assert metadata["splits"]["validation"]["donor_gate_rate"] == .25
    rows = list(csv.DictReader((directory / "metrics.csv").open()))
    assert any(row.get("grad/donor") for row in rows)
    assert any(row.get("val/support/donor_log_outcome_mass") for row in rows)
    assert any(row.get("val/support/donor_loss_gate_true") for row in rows)
    assert any(row.get("val/support/weighted_donor_conditional_kl") for row in rows)
