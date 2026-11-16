import pytest

from psi.train import _apply_overrides, _override_seed


def test_config_overrides_are_typed_and_nested() -> None:
    config = {"trainer": {"devices": [0], "epochs": 1}, "name": "base"}

    _apply_overrides(
        config,
        ["trainer.devices=[2, 3]", "trainer.epochs=100", "name=experiment"],
    )

    assert config == {
        "trainer": {"devices": [2, 3], "epochs": 100},
        "name": "experiment",
    }


@pytest.mark.parametrize("override", ["missing", "trainer.unknown=1", "x.y=1"])
def test_config_overrides_reject_typos(override: str) -> None:
    with pytest.raises((KeyError, ValueError)):
        _apply_overrides({"trainer": {"epochs": 1}}, [override])


def test_seed_override_reaches_nested_data_seed() -> None:
    config = {"seed": 11, "data": {"seed": 11}, "other": 11}

    assert _override_seed(config, 22) == {
        "seed": 22,
        "data": {"seed": 22},
        "other": 11,
    }


def test_control_seeds_are_not_overridden_with_training_seed() -> None:
    config = {"seed": 11, "data": {"seed": 11, "program_seed": 37}, "psi": {"validation_seed": 10011}}
    resolved = _override_seed(config, 13)
    assert resolved["data"] == {"seed": 13, "program_seed": 37}
    assert resolved["psi"]["validation_seed"] == 10011



@pytest.mark.parametrize("weights", [{"donor": float("nan")}, {"nuisance": float("inf")}, {"calibration": -1}, {"donor": 0, "nuisance": 0, "calibration": 0}])
def test_invalid_loss_weights_fail_before_training(weights) -> None:
    from psi.model.objective import LossWeights
    with pytest.raises(ValueError):
        LossWeights(**weights)
