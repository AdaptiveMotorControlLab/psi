import torch

from psi.data.tasks.ravel import RavelTask


def test_incomplete_donor_is_excluded_without_dropping_base_or_valid_family():
    def tokenize(prompts, **kwargs):
        ids = torch.ones(len(prompts), 2, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids, "offset_mapping": [[(0, 1), (1, 2)]] * len(prompts)}

    task = object.__new__(RavelTask)
    task.max_examples = None
    task.max_length = 2
    task.counterfactuals = ("attribute", "wikipedia")
    task.tokenizer = tokenize
    base = {"prompt": "Paris is in", "entity": "Paris", "Country": "France", "Language": "French", "Continent": "Europe", "attribute": "Continent"}
    valid = {**base, "entity": "Rome", "Country": "Italy", "Language": "Italian"}
    missing = {**base, "entity": "Guatemala", "Continent": None, "attribute": "wikipedia"}
    row = {**base, "attribute_counterfactual": valid, "wikipedia_counterfactual": missing}
    inputs, observations, pairs = task._prepare_input([row])
    assert len(inputs) == 2
    assert observations.entities == ["Paris", "Rome"]
    assert pairs.family_id.tolist() == [0]
    assert pairs.base_index.tolist() == [0]
    assert pairs.source_index.tolist() == [1]
