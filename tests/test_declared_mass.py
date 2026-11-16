"""Declared-outcome mass as a target rather than a reward.

For k disjoint declared outcomes the soft cross-entropy decomposes exactly as
`H(t) + KL(t‖rho) - log Z`, so the default objective has a descent direction
(raising Z) that is not the requested distribution. These tests pin the
decomposition, the three modes, and the shapes — a batch-by-batch broadcast in
the reference branch produces a plausible scalar after `.mean()` and biases the
loss by `-Cov(1-Z_ref, log(1-Z))` without ever raising.
"""

import math

import pytest
import torch

from psi.model.abstraction import LinearAbstraction
from psi.model.objective import LossWeights, cross_entropy, log1mexp
from psi.psi import ProgrammableSemanticIntervention

OUTCOMES = 4
CARRIER = 6
HIDDEN = 10
BATCH = 8
MODES = ("full", "conditional", "reference")


def _case(batch: int = BATCH, constant_mass: bool = False):
    """Log-probabilities with a known, deliberately non-constant, total mass."""
    torch.manual_seed(7)
    mass = (
        torch.full((batch, 1), 0.7)
        if constant_mass
        else torch.rand(batch, 1) * 0.5 + 0.3
    )
    log_probability = torch.randn(batch, OUTCOMES).log_softmax(-1) + mass.log()
    target = torch.rand(batch, OUTCOMES)
    target = target / target.sum(-1, keepdim=True)
    reference = torch.rand(batch) * 0.4 + 0.4
    return log_probability, target, reference, mass.squeeze(-1)


class _Subject:
    """Declared-outcome log-probabilities that do not sum to one."""

    def outcome_log_probability(self, batch, steer=None):
        activation = batch["activation"].float() if steer is None else steer[1]
        return activation[..., :OUTCOMES].log_softmax(-1) + math.log(0.6)

    def behavior(self, batch, steer=None):
        return self.outcome_log_probability(batch, steer).softmax(-1)

    def behavior_mass(self, batch, steer=None):
        return self.outcome_log_probability(batch, steer).logsumexp(-1).exp()


def _model(**objective) -> ProgrammableSemanticIntervention:
    torch.manual_seed(11)
    return ProgrammableSemanticIntervention(
        test_dim=OUTCOMES,
        subject=_Subject(),  # type: ignore[arg-type]
        abstraction=LinearAbstraction(carrier_dim=CARRIER, activation_dim=HIDDEN),
        transport=lambda source_state, base_behaviour, transport_gate: source_state,
        loss_weights=LossWeights(donor=1.0, nuisance=0.0, calibration=0.0),
        objective=objective or None,
    )


def _batch():
    torch.manual_seed(5)
    return {
        "activation": torch.randn(BATCH, HIDDEN),
        "cut_positions": torch.zeros(BATCH, 1, dtype=torch.long),
        "state_features": torch.rand(BATCH, OUTCOMES).softmax(-1),
        "base_behaviour": torch.rand(BATCH, OUTCOMES).softmax(-1),
    }


# --------------------------------------------------------------------------
# Shapes: the failure mode that does not raise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_one_value_per_record(mode) -> None:
    log_probability, target, reference, _ = _case()

    per_example = cross_entropy(log_probability, target, mode, reference)

    assert per_example.shape == (BATCH,)


def test_reference_mode_matches_its_definition_elementwise() -> None:
    """Assert per-record, not on the mean.

    The mean of a batch-by-batch outer sum equals the true mean whenever Z is
    constant across records, so a mean-only check passes on tidy synthetic data
    and misses the broadcast entirely.
    """
    log_probability, target, reference, _ = _case()

    per_example = cross_entropy(log_probability, target, "reference", reference)
    expected = -(
        reference * (target * log_probability).sum(-1)
        + (1 - reference) * log1mexp(log_probability.logsumexp(-1))
    )

    torch.testing.assert_close(per_example, expected, atol=1e-6, rtol=0)


def test_a_constant_mass_would_hide_a_broadcast() -> None:
    """Document why the fixtures use a varying Z, so nobody 'simplifies' them."""
    varying = _case(constant_mass=False)
    constant = _case(constant_mass=True)

    assert varying[3].std() > 0.05
    torch.testing.assert_close(
        constant[3], torch.full((BATCH,), 0.7), atol=1e-5, rtol=0
    )


# --------------------------------------------------------------------------
# The decomposition the whole spec rests on
# --------------------------------------------------------------------------


def test_full_mode_is_entropy_plus_conditional_kl_minus_log_mass() -> None:
    log_probability, target, _, _ = _case()
    log_mass = log_probability.logsumexp(-1)
    conditional = log_probability - log_mass.unsqueeze(-1)
    tiny = torch.finfo(target.dtype).tiny
    log_target = target.clamp_min(tiny).log()

    entropy = -(target * log_target).sum(-1)
    divergence = (target * (log_target - conditional)).sum(-1)

    torch.testing.assert_close(
        cross_entropy(log_probability, target, "full"),
        entropy + divergence - log_mass,
        atol=1e-5,
        rtol=0,
    )


def test_conditional_mode_ignores_the_total_mass() -> None:
    """A uniform log-offset changes Z but not rho."""
    log_probability, target, _, _ = _case()
    offset = math.log(0.5)
    shifted = log_probability + offset

    torch.testing.assert_close(
        cross_entropy(shifted, target, "conditional"),
        cross_entropy(log_probability, target, "conditional"),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        cross_entropy(shifted, target, "full"),
        cross_entropy(log_probability, target, "full") - offset,
        atol=1e-5,
        rtol=0,
    )


def test_reference_mode_expands_into_two_separable_terms() -> None:
    """Z_r*(H + KL) + H_b(Z_r) + KL_b(Z_r‖Z), the point of the whole change."""
    log_probability, target, reference, _ = _case()
    log_mass = log_probability.logsumexp(-1)
    mass = log_mass.exp()
    conditional = log_probability - log_mass.unsqueeze(-1)
    tiny = torch.finfo(target.dtype).tiny
    log_target = target.clamp_min(tiny).log()

    semantics = -(target * conditional).sum(-1)
    binary_entropy = -(
        reference * reference.log() + (1 - reference) * (1 - reference).log()
    )
    binary_divergence = reference * (reference.log() - log_mass) + (
        1 - reference
    ) * ((1 - reference).log() - log1mexp(log_mass))

    torch.testing.assert_close(
        cross_entropy(log_probability, target, "reference", reference),
        reference * semantics + binary_entropy + binary_divergence,
        atol=1e-5,
        rtol=0,
    )
    del log_target, mass


def test_reference_mode_prefers_the_declared_mass_over_certainty() -> None:
    """`full` is minimised at Z = 1; `reference` is minimised at Z = Z_ref."""
    torch.manual_seed(3)
    target = torch.rand(1, OUTCOMES)
    target = target / target.sum(-1, keepdim=True)
    reference = torch.full((1,), 0.6)
    conditional = target.log()  # rho already equals t, so only Z varies

    grid = torch.linspace(0.05, 0.999, 60)
    reference_loss = torch.stack([
        cross_entropy(conditional + z.log(), target, "reference", reference)[0]
        for z in grid
    ])
    full_loss = torch.stack([
        cross_entropy(conditional + z.log(), target, "full")[0] for z in grid
    ])

    assert abs(float(grid[int(reference_loss.argmin())]) - 0.6) < 0.02
    assert float(grid[int(full_loss.argmin())]) > 0.99
    assert float(reference_loss[-1]) > float(reference_loss.min())


# --------------------------------------------------------------------------
# log1mexp
# --------------------------------------------------------------------------


def test_log1mexp_is_accurate_and_differentiable_across_the_branch() -> None:
    # Accuracy is checked away from the deliberate -1e-7 clamp, which saturates
    # the last decade before zero. In practice Z reaches .997, so log Z is about
    # -3e-3 and the clamp never binds.
    accurate = torch.cat([
        torch.linspace(-40.0, -0.7, 200),
        torch.linspace(-0.69, -1e-6, 200),
    ]).double()
    torch.testing.assert_close(
        log1mexp(accurate),
        torch.log1p(-torch.exp(accurate)),
        atol=1e-6,
        rtol=1e-6,
    )

    # Finiteness and a finite gradient must hold over the whole range, both
    # sides of the -log 2 branch point included: that is what the clamps buy.
    value = torch.cat([
        torch.linspace(-40.0, -0.7, 200),
        torch.linspace(-0.69, -1e-12, 200),
    ]).double().requires_grad_(True)
    result = log1mexp(value)
    result.sum().backward()

    assert torch.isfinite(result).all()
    assert torch.isfinite(value.grad).all()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_default_objective_is_the_previous_loss() -> None:
    batch = _batch()

    torch.testing.assert_close(
        _model().loss(batch).donor_loss,
        _model(mass_mode="full").loss(batch).donor_loss,
        atol=0,
        rtol=0,
    )


def test_the_reference_mass_is_the_unedited_recipient_mass() -> None:
    """Not `base_behaviour`, whose rows sum to one by construction."""
    batch = _batch()
    model = _model(mass_mode="reference")

    metrics = model.loss(batch).metrics
    expected = _Subject().behavior_mass(batch).mean()

    torch.testing.assert_close(
        metrics["support/reference_mass"], expected, atol=1e-5, rtol=0
    )
    assert abs(float(batch["base_behaviour"].sum(-1).mean()) - 1.0) < 1e-5
    assert abs(float(metrics["support/reference_mass"]) - 1.0) > 0.1


def test_a_declared_reference_mass_is_used_verbatim() -> None:
    metrics = _model(mass_mode="reference", reference_mass=0.42).loss(_batch()).metrics

    torch.testing.assert_close(
        metrics["support/reference_mass"], torch.tensor(0.42), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("mode", MODES)
def test_the_nuisance_term_is_unaffected_by_the_mass_mode(mode) -> None:
    torch.manual_seed(11)
    model = ProgrammableSemanticIntervention(
        test_dim=OUTCOMES,
        subject=_Subject(),  # type: ignore[arg-type]
        abstraction=LinearAbstraction(carrier_dim=CARRIER, activation_dim=HIDDEN),
        transport=lambda source_state, base_behaviour, transport_gate: source_state,
        loss_weights=LossWeights(donor=1.0, nuisance=0.25, calibration=0.0),
        objective={"mass_mode": mode},
    )

    baseline = _model(mass_mode="full")
    baseline.loss_weights = LossWeights(donor=1.0, nuisance=0.25, calibration=0.0)
    batch = _batch()
    draw = lambda: torch.Generator().manual_seed(4242)

    torch.testing.assert_close(
        model.loss(batch, generator=draw()).nuisance_loss,
        baseline.loss(batch, generator=draw()).nuisance_loss,
        atol=1e-6,
        rtol=0,
    )


def test_an_unknown_mass_mode_raises() -> None:
    log_probability, target, reference, _ = _case()

    with pytest.raises(ValueError):
        cross_entropy(log_probability, target, "nope", reference)


def test_reference_mode_without_a_reference_raises() -> None:
    log_probability, target, _, _ = _case()

    with pytest.raises(ValueError):
        cross_entropy(log_probability, target, "reference", None)
