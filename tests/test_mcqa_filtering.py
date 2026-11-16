"""Retained observation IDs must align inputs, activations, readings and gold."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from psi.data.tasks.mcqa import MCQATask, MCQAObservations
from psi.data.types import TaskCache, TaskInputs, TaskOutcomes, TaskPairs


class Labels:
    is_fast = True

    def __call__(self, labels, **kwargs):
        ids = torch.tensor([["ABCD".index(label.strip()) + 1] for label in labels])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def prepare(tmp_path, monkeypatch, filtering, *, answers=(0, 1, 2, 3), gate=0):
    config = {"cache_root": str(tmp_path), "dataset": "synthetic", "variable": "x_order",
              "max_length": 6, "num_choices": 4, "quality_gate": gate,
              "counterfactuals": [], "correctness_filter": filtering}
    task = MCQATask(config, Labels())
    ids = torch.arange(24).reshape(4, 6)
    inputs = TaskInputs(ids, torch.arange(6).expand_as(ids), torch.ones_like(ids))
    readings = torch.tensor([[.7, .1, .1, .1], [.6, .2, .1, .1],
                             [.1, .1, .7, .1], [.1, .1, .65, .15]])
    observations = MCQAObservations(["prompt"] * 4, list(answers),
                                    [" " + "ABCD"[i] for i in answers],
                                    [(" A", " B", " C", " D")] * 4)
    pairs = TaskPairs(torch.zeros(4, dtype=torch.long), torch.arange(4), torch.arange(4),
                      torch.tensor([1, 2, 3, 0]))
    monkeypatch.setattr("psi.data.tasks.mcqa.record_behavior", lambda *a, **kw: readings.clone())
    outcomes, _, info = task._prepare_outcomes(inputs, SimpleNamespace(), observations, pairs)
    return task, inputs, outcomes, info, readings


@pytest.mark.parametrize("filtering", [True, False])
def test_quality_threshold_and_retention_are_independent(tmp_path, monkeypatch, filtering):
    _, _, outcomes, info, readings = prepare(tmp_path, monkeypatch, filtering)
    expected = torch.tensor([0, 2]) if filtering else torch.arange(4)
    torch.testing.assert_close(outcomes.correct_indices, expected)
    torch.testing.assert_close(outcomes.state_features, readings[expected])
    torch.testing.assert_close(outcomes.base_behaviour, readings[expected])
    torch.testing.assert_close(outcomes.gold_state_features, torch.eye(4)[expected])
    assert info["correct_examples"] == 2
    assert info["num_filtered"] == (2 if filtering else 0)


def test_unfiltered_records_keep_raw_inputs_and_activations_aligned(tmp_path, monkeypatch):
    _, inputs, outcomes, _, _ = prepare(tmp_path, monkeypatch, False)
    (tmp_path / "train").mkdir()
    (tmp_path / "x_order/train").mkdir(parents=True)
    (tmp_path / "activations/residual_layer0").mkdir(parents=True)
    torch.save(asdict(inputs), tmp_path / "train/inputs.pt")
    torch.save(asdict(outcomes), tmp_path / "x_order/train/outcomes.pt")
    values = torch.arange(8).reshape(4, 1, 2).float()
    torch.save({"names": ("last_token",), "positions": torch.full((4, 1, 1), 5), "values": values},
               tmp_path / "activations/residual_layer0/train.pt")
    cache = TaskCache(tmp_path, frozenset({"train"}), {}, "last_token", "x_order", "residual_layer0")
    records = cache.records("train")
    assert {len(value) for value in records.values()} == {4}
    torch.testing.assert_close(records["input_ids"], inputs.input_ids)
    torch.testing.assert_close(records["activation"], values[:, 0])


def test_unfiltered_preparation_is_invariant_to_gold_label_changes(tmp_path, monkeypatch):
    task, _, first, _, _ = prepare(tmp_path, monkeypatch, False)
    _, _, second, _, _ = prepare(tmp_path, monkeypatch, False, answers=(1, 2, 3, 0), gate=1)
    for name in ("correct_indices", "outcome_ids", "outcome_mask", "state_features", "base_behaviour"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
    assert not torch.equal(first.gold_state_features, second.gold_state_features)
    filtered, *_ = prepare(tmp_path, monkeypatch, True)
    subject = SimpleNamespace(model_fingerprint="tiny")
    assert task.identity(subject) != filtered.identity(subject)


def test_old_outcome_payloads_need_no_gold_field():
    payload = {"outcome_ids": torch.ones(2, 4, 1, dtype=torch.long),
               "outcome_mask": torch.ones(2, 4, 1, dtype=torch.long),
               "correct_indices": torch.arange(2), "state_features": torch.ones(2, 4) / 4,
               "base_behaviour": torch.ones(2, 4) / 4}
    assert TaskOutcomes(**payload).gold_state_features is None


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_correctness_filter_requires_a_boolean(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match="correctness_filter"):
        prepare(tmp_path, monkeypatch, value)
