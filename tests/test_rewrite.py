import torch

from psi.data.module import CompatibleBatchSampler
from psi.model.abstraction import LinearAbstraction, SitewiseLinearAbstraction
from psi.model.objective import LossWeights, ObjectiveResult
from psi.model.transport import SemanticTransport
from psi.psi import ProgrammableSemanticIntervention


def test_linear_abstraction_is_an_exact_split() -> None:
    abstraction = LinearAbstraction(activation_dim=8, carrier_dim=4)
    activation = torch.randn(7, 8)

    carrier, nuisance = abstraction.lift(activation)
    reconstructed = abstraction.compose(carrier, nuisance)

    torch.testing.assert_close(reconstructed, activation, atol=1e-6, rtol=0)
    assert carrier.shape == (7, 4)
    # The nuisance is the ambient residual, so it is orthogonal to every basis row.
    torch.testing.assert_close(
        nuisance @ abstraction.basis().T, torch.zeros(7, 4), atol=1e-5, rtol=0
    )


def test_objective_reports_each_explicit_term() -> None:
    result = ObjectiveResult(
        donor_loss=torch.tensor(2.0),
        control_loss=torch.tensor(7.0),
        nuisance_loss=torch.tensor(3.0),
        calibration_loss=torch.tensor(5.0),
        loss_weights=LossWeights(
            donor=1.0, nuisance=0.5, calibration=2.0, control=0.25
        ),
    )

    torch.testing.assert_close(result.loss, torch.tensor(15.25))
    assert set(result.metrics) == {
        "loss/calibration",
        "loss/donor",
        "loss/control",
        "loss/nuisance",
    }


def test_data_loader_never_emits_a_singleton_batch() -> None:
    batches = list(CompatibleBatchSampler(9, 4, shuffle=False))

    assert all(2 <= len(batch) <= 4 for batch in batches)
    assert sorted(index for batch in batches for index in batch) == list(range(9))


def test_grouped_batches_keep_every_sample_with_a_compatible_peer() -> None:
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    batches = list(
        CompatibleBatchSampler(
            len(groups),
            5,
            groups=groups,
            shuffle=True,
            generator=torch.Generator().manual_seed(4),
        )
    )

    for batch in batches:
        batch_groups = groups[batch]
        _, counts = batch_groups.unique(return_counts=True)
        assert torch.all(counts >= 2)
    assert sorted(index for batch in batches for index in batch) == list(range(9))


def test_table_transport_selects_each_donors_semantic_row() -> None:
    transport = SemanticTransport.from_name("table")
    source_state = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]])
    table = torch.tensor(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.6, 0.4], [0.1, 0.9]],
            [[0.7, 0.3], [0.2, 0.8]],
        ]
    )

    actual = transport(source_state, torch.empty(3, 2), table)

    torch.testing.assert_close(
        actual,
        torch.stack((table[0, 0], table[1, 1], table[2, 1])),
    )


def test_sitewise_abstraction_is_an_exact_split_and_interchange() -> None:
    abstraction = SitewiseLinearAbstraction(
        site_dim=8,
        num_sites=3,
        num_positions=4,
        carrier_dim_per_site=5,
    )
    base = torch.randn(7, 3 * 4 * 8)
    source = torch.randn(7, 3 * 4 * 8)

    carrier, nuisance = abstraction.lift(base)
    reconstructed = abstraction.compose(carrier, nuisance)
    interchanged = abstraction.interchange(base, source)
    source_carrier, _ = abstraction.lift(source)
    _, base_nuisance = abstraction.lift(base)

    torch.testing.assert_close(reconstructed, base, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        interchanged,
        abstraction.compose(source_carrier, base_nuisance),
        atol=1e-6,
        rtol=0,
    )
    assert carrier.shape == (7, 3 * 4 * 5)


class _Subject:
    def outcome_log_probability(self, _batch, steer):
        _, activation = steer
        return activation[..., :2].log_softmax(-1)

    def behavior(self, _batch, steer):
        return self.outcome_log_probability(_batch, steer).softmax(-1)


def test_psi_loss_reaches_decoder_and_abstraction() -> None:
    torch.manual_seed(4)
    model = ProgrammableSemanticIntervention(
        test_dim=2,
        subject=_Subject(),  # type: ignore[arg-type]
        abstraction=LinearAbstraction(carrier_dim=3, activation_dim=6),
        transport=lambda source_state, base_behaviour, transport_gate: source_state,
    )
    batch = {
        "activation": torch.randn(5, 6),
        "cut_positions": torch.zeros(5, 1, dtype=torch.long),
        "state_features": torch.rand(5, 2).softmax(-1),
        "base_behaviour": torch.rand(5, 2).softmax(-1),
    }

    result = model.loss(batch)
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert model.decoder.weight.grad is not None
    assert model.abstraction.raw_basis.grad is not None


def test_donor_uses_log_loss_and_nuisance_preserves_the_reading() -> None:
    abstraction = LinearAbstraction(activation_dim=2, carrier_dim=2)
    with torch.no_grad():
        abstraction.raw_basis.copy_(torch.eye(2))
    model = ProgrammableSemanticIntervention(
        test_dim=2,
        subject=_Subject(),  # type: ignore[arg-type]
        abstraction=abstraction,
        transport=lambda source_state, base_behaviour, transport_gate: source_state,
    )
    activation = torch.tensor([[-2.0, -1.0], [-0.5, -3.0]])
    state_features = torch.tensor([[0.25, 0.75], [0.60, 0.40]])
    batch = {
        "activation": activation,
        "cut_positions": torch.zeros(2, 1, dtype=torch.long),
        "state_features": state_features,
        "base_behaviour": state_features,
    }

    result = model.loss(batch)

    donor = torch.tensor([1, 0])
    expected = -(
        state_features[donor] * activation[donor].log_softmax(-1)
    ).sum(-1).mean()
    torch.testing.assert_close(result.donor_loss, expected)
    expected = torch.nn.functional.mse_loss(
        activation.softmax(-1), state_features
    )
    torch.testing.assert_close(result.nuisance_loss, expected)
