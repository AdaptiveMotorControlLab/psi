"""Declared-answer interchange scoring for prepared task caches."""

from dataclasses import dataclass, asdict

import torch
from torch.nn.functional import one_hot


@dataclass(frozen=True)
class ScoreReport:
    """Held-out declared-answer metrics and the population they describe.

    ``interchange_agreement`` and ``no_edit_baseline`` use cached pair targets,
    averaged equally across families. ``model_agreement`` uses transported
    frozen model readings. Steering scores request every answer slot on every
    eligible prompt. ``mean_answer_mass`` is measured before edits.
    """

    interchange_agreement: float
    no_edit_baseline: float
    mean_answer_mass: float
    prompts: int
    pairs: int
    steering_success: float | None = None
    steering_no_edit: float | None = None
    model_agreement: float | None = None
    no_edit_model_agreement: float | None = None
    decision_rule: str = "declared-answer argmax"

    def to_frame(
        self,
    ):
        """Return one row suitable for notebook display or tabular export."""
        import pandas as pd

        return pd.DataFrame([asdict(self)])

    def __repr__(
        self,
    ):
        return "\n".join(f"{name:28s} {value}" for name, value in asdict(self).items())

    def _repr_html_(
        self,
    ):
        return self.to_frame().to_html(index=False)


@torch.inference_mode()
def score_cache(
    psi,
    cache,
    split = 'test',
    batch_size = 32,
):
    """Match the audit's macro average of family classification accuracies.

    Pair targets come from the cache. Model agreement additionally compares
    against the rule applied to frozen model readings, including wrong answers.
    """
    if batch_size < 1:
        raise ValueError("Scoring batch size must be positive.")

    records = cache.records(split)
    pairs = cache.interventions(split)
    count = len(pairs.base_index)
    if not count:
        raise ValueError("The score split has no eligible pairs.")
    expected = len(cache.manifest["counterfactuals"])
    if set(pairs.family_id.tolist()) != set(range(expected)):
        raise ValueError(
            "A score family has no eligible pairs or an unknown family ID."
        )
    if pairs.target.dtype != torch.long:
        raise ValueError(
            "This report uses classification targets. IOI margin regression needs its dedicated audit."
        )

    device = psi.subject.device
    psi.to(device).eval()
    predicted, original, semantic, masses = [], [], [], []

    for start in range(0, count, batch_size):
        selection = slice(start, start + batch_size)
        base_index, source_index = (
            pairs.base_index[selection],
            pairs.source_index[selection],
        )
        base = {key: value[base_index].to(device) for key, value in records.items()}
        activation = psi.abstraction.interchange(
            base=base["activation"].float(),
            source=records["activation"][source_index].float().to(device),
        )
        predicted.append(
            psi.subject.behavior(base, (base["cut_positions"], activation))
            .argmax(-1)
            .cpu()
        )
        original.append(base["base_behaviour"].argmax(-1).cpu())
        semantic.append(
            psi.transport(
                source_state=records["state_features"][source_index].to(device),
                base_behaviour=base["base_behaviour"],
                transport_gate=base.get("transport_gate"),
            )
            .argmax(-1)
            .cpu()
        )

    hits = []
    unedited = []
    classes = records["outcome_ids"].shape[1]

    for start in range(0, len(records["input_ids"]), batch_size):
        base = {
            key: value[start : start + batch_size].to(device)
            for key, value in records.items()
        }
        masses.append(psi.subject.behavior_mass(base).cpu())
        if psi.controller is None:
            continue

        for answer in range(classes):
            request = one_hot(
                torch.full((len(base["input_ids"]),), answer, device=device),
                classes,
            ).float()
            target = psi.transport(
                source_state=request,
                base_behaviour=base["base_behaviour"],
                transport_gate=base.get("transport_gate"),
            ).argmax(-1)
            edit = psi.get_control(
                request,
                base["activation"].float(),
                base["base_behaviour"],
                base.get("transport_gate"),
            )
            response = psi.subject.behavior(
                base,
                (base["cut_positions"], edit),
            ).argmax(-1)
            hits.append((response == target).cpu())
            unedited.append((base["base_behaviour"].argmax(-1) == target).cpu())

    predicted = torch.cat(predicted)
    original = torch.cat(original)
    semantic = torch.cat(semantic)

    def macro(
        hit,
    ):
        families = pairs.family_id.unique()
        family_means = [
            hit[pairs.family_id == family].float().mean() for family in families
        ]
        return sum(float(value) for value in family_means) / len(families)

    success = float(torch.cat(hits).float().mean()) if hits else None
    baseline = float(torch.cat(unedited).float().mean()) if unedited else None

    return ScoreReport(
        interchange_agreement=macro(predicted == pairs.target),
        no_edit_baseline=macro(original == pairs.target),
        mean_answer_mass=float(torch.cat(masses).mean()),
        prompts=len(records["input_ids"]),
        pairs=count,
        steering_success=success,
        steering_no_edit=baseline,
        model_agreement=macro(predicted == semantic),
        no_edit_model_agreement=macro(original == semantic),
    )
