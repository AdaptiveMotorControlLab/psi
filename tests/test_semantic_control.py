"""The semantic controller: zero-init, transported targets, capping, actuation."""

import pytest
import torch

from psi.model.abstraction import LinearAbstraction
from psi.model.control import AffineController, GainController
from psi.model.objective import LossWeights
from psi.model.transport import GatedTransport
from psi.psi import ProgrammableSemanticIntervention

OUTCOMES = 4
CARRIER = 6
HIDDEN = 10
BATCH = 8


class _Subject:
    """A stub subject whose declared-outcome logits are the leading coordinates."""

    def outcome_log_probability(self, _batch, steer):
        _, activation = steer
        return activation[..., :OUTCOMES].log_softmax(-1)

    def behavior(self, batch, steer):
        return self.outcome_log_probability(batch, steer).softmax(-1)


def _distribution(rows: int, columns: int = OUTCOMES) -> torch.Tensor:
    return torch.rand(rows, columns).softmax(-1)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "activation": torch.randn(BATCH, HIDDEN),
        "cut_positions": torch.zeros(BATCH, 1, dtype=torch.long),
        "state_features": _distribution(BATCH),
        "base_behaviour": _distribution(BATCH),
    }


def _model(*, control_weight: float = 1.0, transport=None, **control) -> ProgrammableSemanticIntervention:
    torch.manual_seed(11)
    settings = {
        "controller": {"kind": "gain", "test_dim": OUTCOMES, "carrier_dim": CARRIER},
        "readout": "measured",
    }
    settings["controller"].update(control.pop("controller", {}))
    settings.update(control)
    return ProgrammableSemanticIntervention(
        test_dim=OUTCOMES,
        subject=_Subject(),  # type: ignore[arg-type]
        abstraction=LinearAbstraction(carrier_dim=CARRIER, activation_dim=HIDDEN),
        transport=transport
        or (lambda source_state, base_behaviour, transport_gate: source_state),
        loss_weights=LossWeights(
            donor=1.0, nuisance=0.0, calibration=0.0, control=control_weight
        ),
        control=settings,
    )


# --------------------------------------------------------------------------
# Controller construction and initialization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [GainController, AffineController])
def test_zero_initialization_leaves_the_carrier_untouched(kind) -> None:
    """A fresh controller must not perturb the carrier at step zero.

    The control term has to grow in against the donor term rather than
    displacing it before the gain has learned anything.
    """
    controller = kind(OUTCOMES, CARRIER)
    delta = controller(_distribution(BATCH), _distribution(BATCH), torch.randn(BATCH, CARRIER))

    torch.testing.assert_close(delta, torch.zeros(BATCH, CARRIER), atol=0, rtol=0)


def test_affine_controller_depends_on_the_carrier() -> None:
    """The affine kind exists precisely to be recipient-dependent.

    A gain alone writes into a fixed subspace no matter the recipient, so if the
    affine controller ignores the carrier the two kinds are the same hypothesis
    class and there is no escape from the actuator restriction below.
    """
    controller = AffineController(OUTCOMES, CARRIER)
    with torch.no_grad():
        for parameter in controller.parameters():
            parameter.normal_(0, 0.1)

    target, current = _distribution(BATCH), _distribution(BATCH)
    first = controller(target, current, torch.randn(BATCH, CARRIER))
    second = controller(target, current, torch.randn(BATCH, CARRIER) * 5)

    assert (first - second).abs().max() > 1e-6


def test_gain_edits_span_at_most_one_fewer_direction_than_outcomes() -> None:
    """The gain residual is simplex-tangent, so its edits are rank-limited.

    This is a real restriction on what a constant gain can actuate, not a bug:
    it is pinned here so a later control failure is not misread as the carrier
    being uncontrollable.
    """
    controller = GainController(OUTCOMES, CARRIER)
    with torch.no_grad():
        controller.gain.normal_()

    delta = controller(_distribution(BATCH), _distribution(BATCH), torch.randn(BATCH, CARRIER))

    assert int(torch.linalg.matrix_rank(delta)) <= OUTCOMES - 1


def test_unknown_controller_and_write_space_names_raise() -> None:
    with pytest.raises(KeyError):
        GainController(OUTCOMES, CARRIER, write_space="nope")
    with pytest.raises((KeyError, ValueError)):
        _model(controller={"kind": "nope"})


def test_declared_controller_dimensions_match_the_model() -> None:
    """The controller's dims are declared in config, so they can drift.

    `psi.test_dim` and `abstraction.carrier_dim` are the authority; a config
    that disagrees would otherwise fail late with an opaque shape error.
    """
    model = _model()
    assert model.controller.test_dim == model.decoder.out_features
    assert model.controller.carrier_dim == model.abstraction.carrier_dim


def test_zero_control_weight_builds_no_controller() -> None:
    model = _model(control_weight=0.0)

    assert model.controller is None
    assert all(
        parameter is not None
        for group in model.configure_optimizers().param_groups
        for parameter in group["params"]
    )


# --------------------------------------------------------------------------
# The edit
# --------------------------------------------------------------------------


def test_the_edit_holds_the_nuisance_complement_fixed() -> None:
    model = _model()
    with torch.no_grad():
        model.controller.gain.normal_()
    batch = _batch()
    activation = batch["activation"]
    carrier, nuisance = model.abstraction.lift(activation)

    edited = model._write_control(
        carrier=carrier,
        nuisance=nuisance,
        target=_distribution(BATCH),
        current=batch["base_behaviour"],
    )
    edited = edited[0] if isinstance(edited, tuple) else edited
    _, edited_nuisance = model.abstraction.lift(edited)

    torch.testing.assert_close(edited_nuisance, nuisance, atol=1e-5, rtol=0)


@pytest.mark.parametrize("cap", [0, 0.0, 1e-3, 2.5])
def test_the_cap_bounds_the_applied_edit(cap) -> None:
    """A cap of zero must give a zero edit, not an unbounded one.

    `if self.control.get("cap"):` treats 0 as absent, so a zero cap silently
    disables capping instead of freezing the carrier.
    """
    model = _model(cap=cap)
    with torch.no_grad():
        model.controller.gain.normal_(0, 3.0)
    batch = _batch()
    carrier, nuisance = model.abstraction.lift(batch["activation"])

    edited = model._write_control(
        carrier=carrier,
        nuisance=nuisance,
        target=_distribution(BATCH),
        current=batch["base_behaviour"],
    )
    edited = edited[0] if isinstance(edited, tuple) else edited
    applied = (edited - batch["activation"]).norm(dim=-1)

    assert float(applied.max()) <= cap + 1e-5


def test_reported_edit_norm_is_the_applied_edit() -> None:
    """`control_edit_norm` must describe the edit that reached the subject.

    Recomputing the controller for the diagnostic reports the *requested* norm,
    which the cap has not been applied to, so the logged value can be thousands
    of times larger than the edit actually written.
    """
    model = _model(cap=1e-3)
    with torch.no_grad():
        model.controller.gain.normal_(0, 3.0)

    metrics = model.loss(_batch()).metrics

    assert float(metrics["support/control_edit_norm"]) <= 1e-3 + 1e-6


# --------------------------------------------------------------------------
# Targets and requests
# --------------------------------------------------------------------------


def test_the_control_target_is_transported_not_the_raw_request() -> None:
    """Under a false gate the declared target is the base's own behaviour.

    Training against the raw request would teach a gated task to answer every
    question with the requested value.
    """
    model = _model(transport=GatedTransport())
    with torch.no_grad():
        model.controller.gain.normal_()
    batch = _batch() | {"transport_gate": torch.zeros(BATCH, dtype=torch.bool)}
    carrier, nuisance = model.abstraction.lift(batch["activation"])

    request = model._sample_request(features=batch["state_features"])
    target = model.transport(
        source_state=request,
        base_behaviour=batch["base_behaviour"],
        transport_gate=batch["transport_gate"],
    )

    torch.testing.assert_close(target, batch["base_behaviour"], atol=0, rtol=0)


def test_request_families_are_distributions_of_the_declared_shape() -> None:
    """Each family named in the mixture, drawn on its own, is what it claims."""
    features = _distribution(256)
    model = _model()

    donors = model._sample_request(features=features, mixture=[("donor", 1.0)])
    uniform = model._sample_request(features=features, mixture=[("uniform", 1.0)])
    corners = model._sample_request(
        features=features, mixture=[("corner", 1.0)], corner_mass=0.8
    )

    for requests in (donors, uniform, corners):
        torch.testing.assert_close(
            requests.sum(-1), torch.ones(len(features)), atol=1e-5, rtol=0
        )
    # Every donor request is some other record's measured distribution.
    # cdist is float32, so the matched distance is near zero, not exactly zero.
    assert torch.cdist(donors, features).min(dim=-1).values.max() < 1e-3
    # Uniform draws on the simplex have mean 1/k and are not copies of a row.
    torch.testing.assert_close(
        uniform.mean(0), torch.full((OUTCOMES,), 1 / OUTCOMES), atol=0.05, rtol=0
    )
    # A corner puts `corner_mass` on one attested outcome and spreads the rest,
    # so its peak is that mass plus its own share of the spread.
    peak = 0.8 + 0.2 / OUTCOMES
    torch.testing.assert_close(
        corners.max(-1).values, torch.full((len(features),), peak), atol=1e-5, rtol=0
    )
    attested = set(int(v) for v in features.argmax(-1).unique())
    assert set(int(v) for v in corners.argmax(-1).unique()) <= attested


def test_a_mixture_draws_whole_requests_not_coordinates() -> None:
    """A mixed request is one family's row, never a splice of several."""
    features = _distribution(128)
    model = _model()
    mixture = [("donor", 0.5), ("corner", 0.5)]

    requests = model._sample_request(features=features, mixture=mixture)

    torch.testing.assert_close(
        requests.sum(-1), torch.ones(len(features)), atol=1e-5, rtol=0
    )
    # A donor row peaks wherever that record's reading peaks; a corner row peaks
    # at exactly `corner_mass` plus its share. Nothing else is a valid peak.
    peak = requests.max(-1).values
    corner_peak = torch.isclose(peak, torch.tensor(0.8 + 0.2 / OUTCOMES), atol=1e-5)
    donor_peak = torch.tensor([
        bool(torch.cdist(row[None], features).min() < 1e-3) for row in requests
    ])
    assert bool((corner_peak | donor_peak).all())


def test_validation_requests_are_reproducible_under_a_fixed_generator() -> None:
    model = _model()
    features = _distribution(64)

    def sample():
        return model._sample_request(
            features=features, generator=torch.Generator().manual_seed(10011)
        )

    torch.testing.assert_close(sample(), sample(), atol=0, rtol=0)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_inference_control_matches_the_training_edit() -> None:
    """The published control numbers must use the trained path, not a copy."""
    model = _model()
    with torch.no_grad():
        model.controller.gain.normal_()
    batch = _batch()
    request = _distribution(BATCH)

    carrier, nuisance = model.abstraction.lift(batch["activation"])
    target = model.transport(
        source_state=request,
        base_behaviour=batch["base_behaviour"],
        transport_gate=None,
    )
    training = model._write_control(
        carrier=carrier, nuisance=nuisance, target=target,
        current=batch["base_behaviour"],
    )
    training = training[0] if isinstance(training, tuple) else training
    inference = model.get_control(
        request=request,
        activation=batch["activation"],
        base_behaviour=batch["base_behaviour"],
    )

    torch.testing.assert_close(inference, training, atol=1e-6, rtol=0)


def test_control_reaches_the_controller_and_reports_its_diagnostics() -> None:
    model = _model()
    result = model.loss(_batch())
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert model.controller.gain.grad is not None
    assert model.abstraction.raw_basis.grad is not None
    assert {
        "loss/control",
        "support/control_tv",
        "support/control_outcome_mass",
        "support/control_edit_norm",
    } <= set(result.metrics)


def test_gradient_probe_reports_the_control_term() -> None:
    model = _model()
    with torch.no_grad():
        model.controller.gain.normal_()
    result = model.loss(_batch())
    norms: dict[str, torch.Tensor] = {}
    model.log_dict = lambda values, **_: norms.update(values)  # type: ignore[assignment]

    model._probe_gradients(result)

    assert "grad/control" in norms


# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------


def test_the_controller_gets_its_own_learning_rate() -> None:
    """A zero-initialized gain needs a far larger step than the basis.

    At the shared 3e-4 the edit norm reached 1.0 of a 57.8 budget over eight
    epochs; at 1e-1 it reached 36.8 with TV .200. Raising the shared rate
    instead is not equivalent, because it also moves the donor optimization the
    frozen benchmark configs depend on.
    """
    model = _model(learning_rate=0.5)
    groups = model.configure_optimizers().param_groups
    controller = {id(parameter) for parameter in model.controller.parameters()}

    rates = {
        "controller" if any(id(p) in controller for p in group["params"]) else "general":
        group["lr"]
        for group in groups
    }
    assert rates == {"general": model.learning_rate, "controller": 0.5}
    # No parameter may appear in both groups, or Adam would step it twice.
    seen = [id(p) for group in groups for p in group["params"]]
    assert len(seen) == len(set(seen))


def test_controller_learning_rate_defaults_to_the_shared_rate() -> None:
    model = _model()
    groups = model.configure_optimizers().param_groups

    assert {group["lr"] for group in groups} == {model.learning_rate}
