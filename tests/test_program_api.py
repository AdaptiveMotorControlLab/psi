"""Offline CPU checks for program declarations and the generic task."""

from dataclasses import fields
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM

from psi import At, Gated, Program, Prompts, Slots, Table, Values, presets
from psi.data.tasks.program import ProgramTask
from psi.model.subject import LMSubject
from psi.utils.cut import ResidualCut


def tiny():
    torch.set_num_threads(2)
    words = (
        ["<pad>", "<unk>", "<eos>", ".", ":", "?", "Q", "A"]
        + list("BCDEFGHIJKLMNOPQRSTUVWXYZ")
        + [
            "Paris",
            "Tokyo",
            "France",
            "Japan",
            "Europe",
            "Asia",
            "French",
            "Japanese",
            "What",
            "is",
            "the",
            "country",
            "of",
            "Answer",
            "Alice",
            "Bob",
            "gave",
            "to",
        ]
    )
    vocabulary = {word: i for i, word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        padding_side="left",
    )
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
    )
    return tokenizer, model


def mcqa_rows():
    def row(labels, answer):
        return dict(
            prompt=" ".join(label + ". Paris" for label in labels) + " Answer:",
            choices={"label": labels},
            answerKey=answer,
        )

    rows = []
    for i in range(3):
        base = row(list("WXYZ"), i % 4)
        base["answerPosition_counterfactual"] = row(list("WXYZ"), (i + 1) % 4)
        base["randomLetter_counterfactual"] = row(list("ABCD"), i % 4)
        base["answerPosition_randomLetter_counterfactual"] = row(
            list("ABCD"), (i + 1) % 4
        )
        rows.append(base)
    return rows


def assert_fields(left, right, names):
    for name in names:
        a, b = getattr(left, name), getattr(right, name)
        if a is None:
            assert b is None, name
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, check_dtype=True, msg=name)
        else:
            assert a == b, name


@pytest.mark.parametrize("kind", ["mcqa", "arc"])
@pytest.mark.parametrize("variable", ["x_order", "o_answer"])
def test_mcqa_arc_equivalence(kind, variable):
    from psi.data.tasks.mcqa import MCQATask
    from psi.data.tasks.arc import ARCTask

    tokenizer, model = tiny()
    rows = mcqa_rows()
    raw = dict(train=rows, validation=rows, test=rows)
    prompts, program = getattr(presets, kind)(variable, dataset=raw)
    config = dict(
        cache_root="unused",
        dataset="synthetic",
        variable=variable,
        num_choices=4,
        max_length=32,
        quality_gate=0.0,
        correctness_filter=False,
        counterfactuals=[
            "answerPosition",
            "randomLetter",
            "answerPosition_randomLetter",
        ],
    )
    original = (MCQATask if kind == "mcqa" else ARCTask)(config, tokenizer)
    generic = ProgramTask.from_objects(prompts, program, tokenizer, max_length=32)
    subject = LMSubject(model, ResidualCut(0, "last_token"))
    old_input, old_obs, old_pairs = original._prepare_input(rows)
    new_input, new_obs, new_pairs = generic._prepare_input(
        prompts.splits["train"], split="train"
    )
    assert_fields(
        old_input,
        new_input,
        ["input_ids", "position_ids", "attention_mask", "offset_mapping"],
    )
    assert_fields(
        old_pairs, new_pairs, ["base_index", "source_index", "family_id", "example_id"]
    )
    assert_fields(
        original._prepare_activations(old_input, subject, old_obs),
        generic._prepare_activations(new_input, subject, new_obs),
        ["positions", "values", "names"],
    )
    a, ta, _ = original._prepare_outcomes(old_input, subject, old_obs, old_pairs)
    b, tb, _ = generic._prepare_outcomes(new_input, subject, new_obs, new_pairs)
    assert_fields(a, b, [f.name for f in fields(a)])
    assert torch.equal(ta, tb)


def test_ravel_equivalence(tmp_path, monkeypatch):
    from datasets import Dataset
    from psi.data.tasks.ravel import RavelTask

    tokenizer, model = tiny()
    entities = {
        "Paris": dict(Country="France", Continent="Europe", Language="French"),
        "Tokyo": dict(Country="Japan", Continent="Asia", Language="Japanese"),
    }

    def row(entity, attribute):
        return dict(
            prompt=f"What is the country of {entity}? A:",
            entity=entity,
            attribute=attribute,
            **entities[entity],
        )

    rows = []
    for attr in ["Country", "Language", "Continent"]:
        base = row("Paris", attr)
        base["attribute_counterfactual"] = row("Tokyo", "Country")
        base["wikipedia_counterfactual"] = row("Tokyo", "wikipedia")
        rows.append(base)
    entity_path = tmp_path / "entities.json"
    entity_path.write_text(json.dumps(entities))
    config = dict(
        cache_root=str(tmp_path),
        dataset="synthetic",
        variable="a_country",
        entity_file=str(entity_path),
        max_length=32,
        quality_gate=0.0,
        counterfactuals=["attribute", "wikipedia"],
    )
    legacy = RavelTask(config, tokenizer)
    raw = {name: Dataset.from_list(rows) for name in ["train", "val", "test"]}
    prompts, program = presets.ravel(
        dataset=raw, tokenizer=tokenizer, entities=entities
    )
    generic = ProgramTask.from_objects(
        prompts, program, tokenizer, max_length=32, correctness_filter=True
    )
    subject = LMSubject(model, ResidualCut(0, "last_token"))
    a, oa, pa = legacy._prepare_input(raw["train"])
    b, ob, pb = generic._prepare_input(prompts.splits["train"], split="train")
    assert_fields(
        a, b, ["input_ids", "attention_mask", "position_ids", "offset_mapping"]
    )
    assert_fields(pa, pb, ["base_index", "source_index", "family_id", "example_id"])
    assert_fields(
        legacy._prepare_activations(a, subject, oa),
        generic._prepare_activations(b, subject, ob),
        ["positions", "values", "names"],
    )
    # Control the greedy gate while retaining actual tiny-model feature readings.
    expected = torch.tensor(
        [legacy.library.token_of(answer) if answer else -1 for answer in oa.answers]
    )
    monkeypatch.setattr(subject, "greedy", lambda batch, batch_size: expected)
    x, tx, _ = legacy._prepare_outcomes(a, subject, oa, pa)
    y, ty, _ = generic._prepare_outcomes(b, subject, ob, pb)
    assert len(y.correct_indices) == len(oa.prompts)
    assert_fields(
        x,
        y,
        [
            "outcome_ids",
            "outcome_mask",
            "state_features",
            "base_behaviour",
            "transport_gate",
            "correct_indices",
        ],
    )
    assert torch.equal(tx, ty)
    # The generic declaration exposes gold explicitly; the legacy task omits it.
    assert x.gold_state_features is None
    assert y.gold_state_features is not None


def test_slots_values_and_table():
    rows = [
        dict(options=[" W", " X", " Y", " Z"]),
        dict(options=[" A", " B", " C", " D"]),
    ]
    assert Slots("options").resolve(rows)[0][2] == " Y"
    assert Values([" C", " D"]).resolve(rows)[0][0] == " C"
    with pytest.raises(ValueError, match="offending rows"):
        Slots([[" A", " B"], [" C"]]).resolve(rows)
    assert (
        Table(lambda k: (k + 1) % 2).compile(rows, 2) == [[[0.0, 1.0], [1.0, 0.0]]] * 2
    )
    with pytest.raises(ValueError, match="outside"):
        Table({0: 2, 1: 0}).compile(rows, 2)
    assert hash(Program(Values([" A", " B"]))) == hash(Program(Values([" A", " B"])))


def test_splits_positions_and_cache(tmp_path):
    tokenizer, model = tiny()
    prompts = Prompts(["Paris Answer:"] * 20, columns={"gold": [0] * 20}, gold="gold")
    program = Program(Values([" France", " Japan"]))
    again = Prompts(["Paris Answer:"] * 20, columns={"gold": [0] * 20}, gold="gold")
    assert prompts.fingerprint == again.fingerprint
    groups = [{r["_psi_index"] for r in rows} for rows in prompts.splits.values()]
    assert not groups[0] & groups[1] and not groups[1] & groups[2]
    task = ProgramTask.from_objects(
        prompts,
        program,
        tokenizer,
        cache_root=str(tmp_path),
        position=At("Paris"),
        max_length=16,
    )
    subject = LMSubject(model, ResidualCut(0, task.selected_position))
    cache = task.materialize(subject)
    assert len(cache.records("train")["activation"]) == 16
    changed = ProgramTask.from_objects(
        prompts,
        Program(Values([" France", " Japan"]), "keep"),
        tokenizer,
        max_length=16,
    )
    assert task.identity(subject) != changed.identity(subject)


def test_tiny_estimator_roundtrip_and_control(tmp_path):
    from psi import PSI, scan_layers

    tokenizer, subject = tiny()
    prompts = Prompts(["Paris Answer:", "Tokyo Answer:"] * 10)
    program = Program(Values([" France", " Japan"]))
    model = PSI(
        subject,
        tokenizer=tokenizer,
        layer=0,
        rank=4,
        epochs=2,
        batch_size=8,
        device="cpu",
        dtype="float32",
        cache_dir=tmp_path / "cache",
        progress=False,
        steering=True,
        max_length=16,
    )
    checked = program.check(model, prompts)
    assert sum(checked.argmax_counts) == 20
    assert 0 <= checked.lowest_answer_mass <= checked.mean_answer_mass <= 1
    assert model.fit(prompts, program) is model
    assert len(model.history) == 2
    assert all("train/loss" in row for row in model._history)
    report = model.score()
    assert report.prompts == 2 and report.pairs == 2
    assert report.steering_success is not None
    before = model.reading("Paris Answer:")
    swaps = model.swap(base="Paris Answer:", donor="Tokyo Answer:")
    assert swaps.shape == (2, 5)
    assert model.steer("Paris Answer:", answer=" France").shape == (2, 4)
    model.push_to_hub(tmp_path / "saved")
    metadata = json.loads((tmp_path / "saved" / "psi.json").read_text())
    assert metadata["program"]["answers"] == {"values": [" France", " Japan"]}
    from safetensors.torch import load_file

    state = load_file(tmp_path / "saved" / "weights.safetensors")
    assert set(key.split(".")[0] for key in state) == {
        "abstraction",
        "decoder",
        "controller",
    }
    restored = PSI.from_pretrained(
        tmp_path / "saved", model=subject, tokenizer=tokenizer, device="cpu"
    )
    torch.testing.assert_close(
        torch.tensor(before.values),
        torch.tensor(restored.reading("Paris Answer:").values),
        rtol=0,
        atol=0,
    )
    assert restored.score(prompts).interchange_agreement == report.interchange_agreement
    frame = scan_layers(
        subject,
        prompts,
        program,
        tokenizer=tokenizer,
        layers=[0, 1],
        rank=4,
        epochs=1,
        cache_dir=tmp_path / "scan",
        device="cpu",
        max_length=16,
        progress=False,
    )
    assert frame.layer.tolist() == [0, 1]


def test_gold_and_new_slots_errors(tmp_path):
    from psi import PSI

    tokenizer, subject = tiny()
    model = PSI(
        subject,
        tokenizer=tokenizer,
        layer=0,
        rank=4,
        epochs=1,
        batch_size=8,
        cache_dir=tmp_path,
        progress=False,
        max_length=16,
        target="gold",
    )
    prompts = Prompts(["Paris Answer:"] * 20)
    with pytest.raises(ValueError, match="gold column"):
        model.fit(prompts, Program(Values([" France", " Japan"])))
    prompts = Prompts(
        ["Paris Answer:"] * 20,
        columns={"options": [[" France", " Japan"]] * 20, "gold": [0] * 20},
        gold="gold",
    )
    model.fit(prompts, Program(Slots("options")))
    with pytest.raises(ValueError, match="Slots needs options"):
        model.reading("Tokyo Answer:")
    assert len(model.reading("Tokyo Answer:", options=[" France", " Japan"])) == 2
    with pytest.raises(RuntimeError, match="steering=True"):
        model.steer("Paris Answer:", answer=0)


def test_ioi_documents_metric_difference():
    from psi.data.tasks.ioi import IOITask

    # The preset deliberately reports classification on full semantic names.
    # The legacy official metric uses bare names and floating logit margins.
    tokenizer, _ = tiny()
    row = dict(prompt="Alice Bob gave to Alice", choices=["Alice", "Bob"], answerKey=1)
    raw = {
        name: [
            dict(
                row,
                SName_counterfactual=row,
                IOName_counterfactual=row,
                S1Name_counterfactual=row,
            )
        ]
        * 2
        for name in ["train", "validation", "test"]
    }
    prompts, program = presets.ioi(dataset=raw)
    assert program.answers.resolve(prompts.splits["train"])[0] == (" Alice", " Bob")
    legacy = IOITask(
        dict(
            cache_root="unused",
            dataset="synthetic",
            variable="output_position",
            max_length=32,
            quality_gate=0.0,
            counterfactuals=["SName", "IOName", "S1Name"],
        ),
        tokenizer,
    )
    assert float(legacy._ioi_target_model(torch.tensor(1.0), torch.tensor(-1.0))) < 0
    assert program.compatibility["table_column"] == "_table"


def test_multitoken_positions_gates_and_pairs(tmp_path):
    from psi import PSI

    tokenizer, subject = tiny()
    texts = ["Paris Answer:", "Tokyo Answer:"] * 10
    prompts = Prompts(
        texts, columns={"gate": [True, False] * 10}, pairs={"test": [(0, 1), (1, 0)]}
    )
    assert prompts.pairs["test"][0] == dict(
        family_id=0, example_id=0, base_index=0, source_index=1
    )
    program = Program(Values([" France", " Japan Asia"]), Gated("gate"))
    model = PSI(
        subject,
        tokenizer=tokenizer,
        layer=0,
        rank=4,
        epochs=1,
        batch_size=8,
        max_length=16,
        cache_dir=tmp_path,
        progress=False,
    )
    report = program.check(model, prompts)
    assert program.check(model, prompts) is report
    model.fit(prompts, program)
    outcomes = model.cache_.outcomes("test")
    assert outcomes.outcome_ids.shape[-1] == 2
    assert (outcomes.outcome_mask[..., :-1] >= outcomes.outcome_mask[..., 1:]).all()
    assert outcomes.transport_gate.dtype == torch.bool
    task = ProgramTask.from_objects(
        prompts, program, tokenizer, position=At("absent"), max_length=16
    )
    inputs, obs, _ = task._prepare_input(prompts.splits["test"])
    with pytest.raises(ValueError, match="no regex match"):
        task._prepare_activations(inputs, model.subject, obs)


def test_generic_cli_infers_only_missing_dimensions(tmp_path, monkeypatch):
    import yaml
    import psi.train as training
    from psi.data import get_task

    tokenizer, model = tiny()
    prompts = Prompts(["Paris Answer:", "Tokyo Answer:"] * 10)
    monkeypatch.setattr(Prompts, "from_hf", lambda *args, **kwargs: prompts)
    subject = LMSubject(model, ResidualCut(0, "last_token"))
    monkeypatch.setattr(training, "get_subject", lambda config: (tokenizer, subject))
    config = dict(
        seed=11,
        data=dict(
            task="program",
            dataset="synthetic",
            answers={"values": [" France", " Japan"]},
            rule="copy",
            max_length=16,
            cache_root=str(tmp_path / "cache"),
            batch_size=8,
            num_workers=0,
        ),
        psi=dict(
            abstraction=dict(kind="linear", carrier_rank=4), transport="full_change"
        ),
        trainer=dict(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        ),
    )
    training.run_training(config, run_dir=tmp_path / "cli")
    resolved = yaml.safe_load((tmp_path / "cli/resolved.yaml").read_text())
    assert resolved["psi"]["test_dim"] == 2
    assert resolved["psi"]["abstraction"]["activation_dim"] == 16
    assert "test_dim" not in config["psi"]
    config["psi"]["test_dim"] = 3
    with pytest.raises(ValueError, match="disagrees"):
        training.run_training(config, dry_run=True)


def test_cache_identity_tracks_semantics_not_tokenizer_call_state(tmp_path):
    from dataclasses import replace

    tokenizer, model = tiny()
    texts = ["Paris Answer:", "Tokyo Answer:"] * 10
    columns = {"gold_a": [0] * 20, "gold_b": [1] * 20}
    prompts = Prompts(texts, columns=columns, gold="gold_a")
    other = Prompts(texts, columns=columns, gold="gold_b")
    program = Program(Values([" France", " Japan"]))
    task = ProgramTask.from_objects(prompts, program, tokenizer, max_length=16)
    subject = LMSubject(model, ResidualCut(0, "last_token"))

    model.register_buffer("scalar_marker", torch.tensor(1.0))
    initial = task.identity(subject)
    inputs, observations, _ = task._prepare_input(prompts.splits["train"])
    assert task.identity(subject) == initial
    task.encode_batch(inputs, observations)
    assert task.identity(subject) == initial

    other_task = ProgramTask.from_objects(other, program, tokenizer, max_length=16)
    assert prompts.fingerprint != other.fingerprint
    assert other_task.identity(subject) != initial

    tokenizer.truncation_side = "left"
    assert task.identity(subject) != initial
    tokenizer.truncation_side = "right"

    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    assert task.identity(subject) != initial

    different = replace(program, compatibility={"token_ids": [3, 4]})
    assert different != program
    assert hash(different) != hash(program)
    with pytest.raises(ValueError, match="probability vector"):
        Table({0: [float("nan"), 1.0], 1: [0.0, 1.0]}).compile([{}], 2)


def test_failed_refit_is_not_exposed_as_a_fitted_model(tmp_path, monkeypatch):
    from psi import PSI

    tokenizer, model = tiny()
    prompts = Prompts(
        ["Paris Answer:", "Tokyo Answer:"] * 10,
        columns={"gold": [0, 1] * 10},
        gold="gold",
    )
    estimator = PSI(
        model,
        tokenizer=tokenizer,
        layer=0,
        rank=4,
        epochs=1,
        batch_size=8,
        max_length=16,
        cache_dir=tmp_path,
        progress=False,
    )
    program = Program(Values([" France", " Japan"]))
    estimator.fit(prompts, program)
    assert estimator.program_.gold == "gold"
    estimator.save(tmp_path / "saved")
    saved = json.loads((tmp_path / "saved" / "psi.json").read_text())
    assert saved["program"]["gold"] == "gold"

    def fail(*args, **kwargs):
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(ProgramTask, "materialize", fail)
    with pytest.raises(RuntimeError, match="preparation failed"):
        estimator.fit(prompts, Program(Values([" A", " B"])))
    with pytest.raises(RuntimeError, match="Call fit"):
        estimator.reading("Paris Answer:")


def test_explicit_slots_roundtrip_preserves_edited_readings(tmp_path):
    from psi import PSI

    tokenizer, subject = tiny()
    texts = [f"Paris Answer: {index}" for index in range(20)]
    answers = [
        [" France", " Japan"] if index % 2 == 0 else [" Europe", " Asia"]
        for index in range(20)
    ]
    prompts = Prompts(texts)
    program = Program(Slots(answers))
    model = PSI(
        subject,
        tokenizer=tokenizer,
        layer=0,
        rank=4,
        epochs=1,
        batch_size=8,
        max_length=16,
        cache_dir=tmp_path / "cache",
        progress=False,
        steering=True,
    )
    model.fit(prompts, program)
    model.save(tmp_path / "saved")
    restored = PSI.load(
        tmp_path / "saved", model=subject, tokenizer=tokenizer, device="cpu"
    )

    base, donor = prompts.splits["test"]
    base_options = program.answers.resolve([base])[0]
    donor_options = program.answers.resolve([donor])[0]
    with pytest.raises(ValueError, match="original row metadata"):
        restored.reading(base["prompt"])

    before = model.swap(base=base, donor=donor)
    after = restored.swap(
        base=base["prompt"],
        donor=donor["prompt"],
        base_options=base_options,
        donor_options=donor_options,
    )
    torch.testing.assert_close(
        torch.tensor(before.base_after.values),
        torch.tensor(after.base_after.values),
        rtol=0,
        atol=0,
    )
    original_control = model.steer(base, answer=0)
    restored_control = restored.steer(base["prompt"], answer=0, options=base_options)
    torch.testing.assert_close(
        torch.tensor(original_control.after.values),
        torch.tensor(restored_control.after.values),
        rtol=0,
        atol=0,
    )
    for key, value in model.psi_.state_dict().items():
        assert torch.equal(value, restored.psi_.state_dict()[key])


def test_dataframe_rejects_unrecognized_split_rows():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "prompt": ["Paris"] * 7,
            "split": [
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
                "typo",
            ],
        }
    )
    with pytest.raises(ValueError, match="dataframe rows"):
        Prompts.from_pandas(frame, split="split")


def test_ravel_default_library_includes_donor_only_values():
    tokenizer, _ = tiny()

    def record(city, country, continent, language, attribute):
        return dict(
            prompt=f"What is the country of {city}? A:",
            entity=city,
            Country=country,
            Continent=continent,
            Language=language,
            attribute=attribute,
        )

    donor = record("Tokyo", "Japan", "Asia", "Japanese", "Country")
    wikipedia = dict(donor, attribute="wikipedia")
    base = record("Paris", "France", "Europe", "French", "Country")
    base.update(attribute_counterfactual=donor, wikipedia_counterfactual=wikipedia)
    raw = {name: [base, base] for name in ["train", "val", "test"]}
    prompts, program = presets.ravel(dataset=raw, tokenizer=tokenizer)
    expected = tokenizer.encode(" Japan", add_special_tokens=False)[0]
    assert expected in program.compatibility["token_ids"]
    assert len(prompts.splits["train"]) == 6


def test_program_hash_matches_numeric_table_equality():
    integers = Program(Values([" A", " B"]), Table({0: [1, 0], 1: [0, 1]}))
    decimals = Program(Values([" A", " B"]), Table({0: [1.0, 0.0], 1: [0.0, 1.0]}))
    assert integers == decimals
    assert hash(integers) == hash(decimals)
    assert {integers: "found"}[decimals] == "found"
    callable_program = Program(Values([" A", " B"]), Table(lambda slot: 1 - slot))
    assert isinstance(hash(callable_program), int)


def test_original_top_level_estimator_export_is_preserved():
    from psi import ProgrammableSemanticIntervention
    from psi.psi import ProgrammableSemanticIntervention as original

    assert ProgrammableSemanticIntervention is original
