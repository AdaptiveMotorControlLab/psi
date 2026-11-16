"""Continuation padding must not move the position a candidate is scored at.

`LMSubject.measure` reads the final `horizon` positions of the sequence built by
`LMSubject.prepare`. That is only the log-probability of the candidate string
appended to the prompt when the candidate's real tokens are right-aligned to the
start of its slot. A left pad sits between the prompt and the candidate's first
token, and that token's probability is then read off the pad's own prediction.

These tests pin the contract with a randomly initialized tiny model, so they need
no download, no dataset and no GPU. See docs/spec-01-continuation-padding.md.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from psi.model.subject import LMSubject
from psi.utils.cut import ResidualCut

VOCAB = 40
PAD = 0
PROMPT_LENGTH = 6
HIDDEN = 16


def _subject() -> LMSubject:
    """A tiny randomly initialized decoder with the layer layout ResidualCut expects."""
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
    return LMSubject(model, ResidualCut(layer=0, position="last"))


def _prompts(count: int) -> dict[str, torch.Tensor]:
    """Left-padded prompts, which is the project's input convention."""
    generator = torch.Generator().manual_seed(5)
    input_ids = torch.randint(
        1, VOCAB, (count, PROMPT_LENGTH), generator=generator
    )
    attention_mask = torch.ones_like(input_ids)
    # Give the first row one leading pad so the two padding conventions are
    # exercised together rather than in isolation.
    input_ids[0, 0] = PAD
    attention_mask[0, 0] = 0
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def _candidates() -> list[list[list[int]]]:
    """Mixed-length candidates: this is the whole point of the test."""
    return [
        [[11, 12], [21, 22, 23]],
        [[31, 32, 33], [7]],
        [[9], [13, 14]],
    ]


def _pack(
    candidates: list[list[list[int]]], side: str
) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = max(len(tokens) for row in candidates for tokens in row)
    ids = torch.full((len(candidates), len(candidates[0]), horizon), PAD)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(candidates):
        for c, tokens in enumerate(row):
            length = len(tokens)
            slot = slice(0, length) if side == "right" else slice(horizon - length, horizon)
            ids[i, c, slot] = torch.tensor(tokens)
            mask[i, c, slot] = 1
    return ids, mask


def _unpadded_reference(
    subject: LMSubject,
    prompts: dict[str, torch.Tensor],
    candidates: list[list[list[int]]],
    steer: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Score each candidate on its own, with no padding anywhere.

    This is the estimand: the teacher-forced log-probability of the candidate
    string appended directly to its prompt.
    """
    scores = torch.zeros(len(candidates), len(candidates[0]))
    for i, row in enumerate(candidates):
        for c, tokens in enumerate(row):
            length = len(tokens)
            token_tensor = torch.tensor(tokens).unsqueeze(0)
            input_ids = torch.cat(
                [prompts["input_ids"][i : i + 1], token_tensor[:, :-1]], dim=-1
            )
            attention_mask = torch.cat(
                [
                    prompts["attention_mask"][i : i + 1],
                    torch.ones_like(token_tensor[:, :-1]),
                ],
                dim=-1,
            )
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            positions, activation = (None, None) if steer is None else steer
            if steer is not None:
                positions = positions[i : i + 1]
                activation = activation[i : i + 1]

            with subject.cut.patch(subject.model, positions, activation):
                with torch.no_grad():
                    logits = subject.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    ).logits.float()

            window = logits[:, -length:].log_softmax(-1)
            scores[i, c] = (
                window.gather(-1, token_tensor.unsqueeze(-1)).squeeze(-1).sum()
            )
    return scores


def _batch(side: str) -> tuple[LMSubject, dict[str, torch.Tensor], list[list[list[int]]]]:
    subject = _subject()
    prompts = _prompts(len(_candidates()))
    candidates = _candidates()
    ids, mask = _pack(candidates, side)
    batch = {
        **prompts,
        "outcome_ids": ids,
        "outcome_mask": mask,
        "cut_positions": torch.full((len(candidates), 1), PROMPT_LENGTH - 1),
    }
    return subject, batch, candidates


def test_mixed_length_batch_matches_unpadded_scoring() -> None:
    """A padded batch must score every candidate as if it stood alone."""
    subject, batch, candidates = _batch("right")

    with torch.no_grad():
        measured = subject.outcome_log_probability(batch)

    reference = _unpadded_reference(subject, batch, candidates)
    torch.testing.assert_close(measured, reference, rtol=1e-4, atol=1e-4)


def test_intervened_scoring_matches_unpadded() -> None:
    """The same must hold on the patched path the donor loss actually uses."""
    subject, batch, candidates = _batch("right")

    generator = torch.Generator().manual_seed(7)
    activation = torch.randn(len(candidates), HIDDEN, generator=generator)
    steer = (batch["cut_positions"], activation)

    with torch.no_grad():
        measured = subject.outcome_log_probability(batch, steer)

    reference = _unpadded_reference(subject, batch, candidates, steer)
    torch.testing.assert_close(measured, reference, rtol=1e-4, atol=1e-4)


def test_left_padded_outcomes_are_rejected() -> None:
    """A left-padded candidate cache is wrong on disk and must not be scored."""
    subject, batch, _ = _batch("left")

    with pytest.raises(ValueError, match="right-padded"):
        subject.prepare(batch)


def test_arithmetic_declares_its_candidate_padding_side() -> None:
    """The Arithmetic candidate encoding must not inherit the input padding side.

    `psi/utils/hf.py` sets the tokenizer to left padding for prompts, and the
    candidate labels are encoded with that same tokenizer. The task therefore has
    to ask for right padding explicitly. The guard in `LMSubject.prepare` would
    also catch a regression here, but only after a full materialization pass,
    which is an expensive way to learn about a one-line omission.
    """
    import inspect

    from psi.data.tasks.arithmetic import ArithmeticTask

    source = inspect.getsource(ArithmeticTask._prepare_outcomes)
    assert 'padding_side="right"' in source or "padding_side='right'" in source, (
        "ArithmeticTask._prepare_outcomes must encode its candidate labels with "
        "padding_side='right'."
    )
