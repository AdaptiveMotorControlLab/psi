import torch

from psi.model.subject import LMSubject


def test_one_token_measurement_keeps_full_vocabulary_normalization() -> None:
    logits = torch.tensor([[[2.0, 1.0, 4.0]]])
    outcome_ids = torch.tensor([[[0], [1]]])
    outcome_mask = torch.ones_like(outcome_ids)

    measured = LMSubject.measure(
        object.__new__(LMSubject),
        logits,
        slice(-1, None),
        outcome_ids,
        outcome_mask,
    )

    expected = logits[:, -1].log_softmax(-1)[..., :2]
    torch.testing.assert_close(measured, expected)
    assert measured.exp().sum() < 1


def test_multi_token_measurement_returns_sequence_log_probability() -> None:
    logits = torch.tensor(
        [
            [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
            [[1.0, 0.0, 3.0], [2.0, 1.0, 0.0]],
        ]
    )
    outcome_ids = torch.tensor([[[0, 1], [2, 0]]])
    outcome_mask = torch.ones_like(outcome_ids)

    measured = LMSubject.measure(
        object.__new__(LMSubject),
        logits,
        slice(-2, None),
        outcome_ids,
        outcome_mask,
    )

    token_log_probability = logits.log_softmax(-1)
    expected = torch.stack(
        (
            token_log_probability[0, 0, 0] + token_log_probability[0, 1, 1],
            token_log_probability[1, 0, 2] + token_log_probability[1, 1, 0],
        )
    ).unsqueeze(0)
    torch.testing.assert_close(measured, expected)
