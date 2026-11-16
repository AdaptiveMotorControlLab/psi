from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Mapping

from einops import rearrange, repeat
from torch import Tensor
import torch

from psi.utils.cut import ActivationCut

if TYPE_CHECKING:
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

class LMSubject:
    """A subject is a language model that can be used for causal experiments.

    This class is a wrapper around a transformer model that provides additional
    functionality for causal experiments, such as freezing the model parameters
    and providing access to the model's layers.
    """

    def __init__(self, model: _BaseModelWithGenerate, cut : ActivationCut) -> None:
        self.cut = cut
        self.model = model

        self.freeze()

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.model.parameters()).device

    @property
    def model_fingerprint(self) -> str | None:
        """Return the fingerprint of the model."""
        name = self.model.name_or_path or ''
        name = name.replace('/', '_')
        return name

    @property
    def cut_fingerprint(self) -> str:
        """Return the fingerprint of the cut."""
        return self.cut.fingerprint

    def freeze(self) -> None:
        """Freeze the model parameters to prevent training."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def extract(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Extract the model's activation at the cut point."""
        return self.cut.capture(
            self.model,
            batch['input_ids'],
            batch['cut_positions'],
            position_ids=batch['position_ids'],
            attention_mask=batch['attention_mask']
        )

    def outcome_log_probability(
        self,
        batch: Mapping[str, Tensor],
        steer: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Return full-vocabulary log-probabilities for the declared outcomes."""
        inp, out = self.prepare(batch)

        positions, activation = (None, None) if steer is None else steer

        intervention = self.cut.patch(self.model, positions, activation)

        with intervention:
            logits = self.model(
                **inp,
                use_cache=False,
            ).logits.float() # (b, s, vocab)

        return self.measure(
            logits,
            **out,
        ) # (b, c)

    def behavior(
        self,
        batch: Mapping[str, Tensor],
        steer: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Return the subject's distribution conditional on the declared outcomes."""
        return self.outcome_log_probability(batch, steer).softmax(-1)
    
    def behavior_mass(
        self,
        batch: Mapping[str, Tensor],
        steer: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Return the subject's total probability mass on the declared outcomes."""
        return self.outcome_log_probability(batch, steer).logsumexp(-1).exp()

    def greedy(
        self,
        batch: Mapping[str, Tensor],
        batch_size: int = 32,
    ) -> Tensor:
        """Return the subject's greedy next token prediction."""
        greedy_token = []

        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        attention_mask = batch['attention_mask']

        for start in range(0, len(input_ids), batch_size):
            end = start + batch_size
            chunk = {
                "input_ids": input_ids[start:end].to(self.device),
                "attention_mask": attention_mask[start:end].to(self.device),
                "position_ids": position_ids[start:end].to(self.device),
            }
            with torch.inference_mode():
                greedy: torch.Tensor = self.model(
                    **chunk,
                    use_cache=False,
                ).logits[:, -1].argmax(dim=-1) # (b, )
            greedy_token.append(greedy.cpu())

        return torch.cat(greedy_token, dim=0)


    def prepare(self, batch: Mapping[str, Tensor]) -> tuple[Mapping[str, Tensor], dict]:
        """Prepare the input batch for the model based on the requested output behavior shape."""
        outcome_ids = batch['outcome_ids']
        outcome_mask = batch['outcome_mask']

        b, c, n = outcome_ids.shape

        # For n > 1 the candidate histories diverge, i.e. the model
        # sequence for c[0] is different from c[1], so we need to
        # compute the model's logits output as if it were a completely
        # different prompt, i.e. making the batch size effectively b * c.
        if n > 1:
            # Candidates must be right-padded: `measure` reads the last n
            # positions, so a pad between the prompt and a candidate's first
            # token silently replaces that token's log-probability.
            valid = outcome_mask.bool().int()
            
            # Check that outcome_mask is a prefix of ones
            if not bool((valid[..., :-1] >= valid[..., 1:]).all()):
                raise ValueError(
                    "Outcome masks must be right-padded (a prefix of ones). "
                    "A left-padded candidate is measured at the wrong position."
                )
            if not bool(valid[..., 0].all()):
                raise ValueError("Every declared outcome must have at least one token.")
            
            model_ids = repeat(batch['input_ids'], 'b s -> (b c) s', c=c)
            model_mask = repeat(batch['attention_mask'], 'b s -> (b c) s', c=c)

            continuation_ids = rearrange(outcome_ids[..., :-1], 'b c n -> (b c) n')
            continuation_mask = rearrange(outcome_mask[..., :-1], 'b c n -> (b c) n')

            input_ids = torch.cat([model_ids, continuation_ids], dim=-1)
            attention_mask = torch.cat([model_mask, continuation_mask], dim=-1)

            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~attention_mask.bool(), 0)

            inp = {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
            }

        # For one-token outcomes, all c candidates share the same context.
        # So we can get away by computing the model's logits output once for
        # each context prompt in the batch.
        else:
            inp = {
                'input_ids': batch['input_ids'],
                'position_ids': batch['position_ids'],
                'attention_mask': batch['attention_mask'],
            }

        out = {
            'outcome_pos': slice(-n, None),  # the last n position in the sequence
            'outcome_ids': outcome_ids,
            'outcome_mask': outcome_mask,
        }
        return inp, out


    def measure(self, logits: Tensor, outcome_pos: slice, outcome_ids: Tensor, outcome_mask: Tensor) -> Tensor:
        """Measure each outcome's full-vocabulary log-probability."""
        b, c, n = outcome_ids.shape

        if n == 1:
            log_probability = logits[:, -1].log_softmax(-1).gather(
                -1, outcome_ids[..., 0]
            )
            return log_probability.masked_fill(
                ~outcome_mask[..., 0].bool(), float('-inf')
            ) # (b, c)

        outcome_ids = rearrange(outcome_ids, "b c n -> (b c) n 1")
        outcome_mask = rearrange(outcome_mask, "b c n -> (b c) n")

        log_probability = (
            logits.log_softmax(-1)[:, outcome_pos]
            .gather(-1, outcome_ids)
            .squeeze(-1)
        ) # (b * c, n)

        score = rearrange(
            (log_probability * outcome_mask).sum(-1),
            "(b c) -> b c",
            b=b,
            c=c,
        )

        return score
