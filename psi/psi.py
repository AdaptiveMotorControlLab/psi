"""The single Programmable Semantic Interventions (PSI) model."""

from typing import Literal, Mapping

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.nn.functional import mse_loss, one_hot

from psi.model import get_controller
from psi.model.abstraction import Abstraction
from psi.model.transport import SemanticTransport
from psi.model.objective import LossWeights, ObjectiveResult, cross_entropy
from psi.model.subject import LMSubject

from psi.utils.misc import derangement


def psi_config(config: Mapping) -> dict:
    """Return a config's PSI hyperparameter block, `psi:`."""
    if "psi" not in config:
        raise KeyError("A config needs a `psi:` block of PSI hyperparameters.")
    return dict(config["psi"])


class ProgrammableSemanticIntervention(LightningModule):
    """Calibrate and causally validate a predictive program state."""

    @classmethod
    def from_config(cls, config: dict, subject: LMSubject) -> "ProgrammableSemanticIntervention":
        """Create a ProgrammableSemanticIntervention from a configuration dictionary."""

        abstraction = Abstraction.from_config(config["abstraction"])
        transport = SemanticTransport.from_name(config["transport"])

        return cls(
            test_dim=config["test_dim"],
            subject=subject,
            abstraction=abstraction,
            transport=transport,
            loss_weights=LossWeights(**config.get("loss_weights", {})),
            control=config.get("control"),
            objective=config.get("objective"),
            target_kind=config.get("target_kind", 'soft'),
            learning_rate=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 1e-4),
            grad_clip=config.get("grad_clip", 1.0),
            gradient_probe_every=config.get("gradient_probe_every"),
            validation_seed=config.get("validation_seed", 10011),
        )

    def __init__(
        self,
        test_dim: int,
        subject: LMSubject,
        abstraction: Abstraction,
        transport: SemanticTransport,
        *,
        loss_weights: LossWeights = LossWeights(),
        control: dict | None = None,
        objective: dict | None = None,
        target_kind: str = 'soft',
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float | None = 1.0,
        gradient_probe_every: int | None = None,
        validation_seed: int = 10011,
    ) -> None:
        super().__init__()
        self.subject = subject
        self.transport = transport
        self.objective = objective or {}
        self.abstraction = abstraction

        self.decoder = nn.Linear(abstraction.carrier_dim, test_dim)
        
        self.control = control or {}
        self.controller = (
            get_controller(self.control)
            if loss_weights.control > 0
            else None
        )

        assert target_kind in ('soft', 'hard', 'gold'), "target_kind must be either 'soft', 'hard', or 'gold'."

        self.target_kind = target_kind
        self.loss_weights = loss_weights
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.grad_clip = grad_clip
        if gradient_probe_every is not None and (
            isinstance(gradient_probe_every, bool)
            or not isinstance(gradient_probe_every, int)
            or gradient_probe_every < 1
        ):
            raise ValueError("gradient_probe_every must be a positive integer or null.")
        self.gradient_probe_every = gradient_probe_every
        self.validation_seed = int(validation_seed)

    def loss(
        self, batch: Mapping[str, Tensor], *, generator: torch.Generator | None = None
    ) -> ObjectiveResult:
        '''Compute the loss for a batch of data.
        The batch contrat is the following:
        batch = {
            # model input data
            "input_ids": (batch, seq_len),  # input token ids
            "position_ids": (batch, seq_len),  # position ids for the input
            "attention_mask": (batch, seq_len),  # attention mask for the input
            "outcome_ids": (batch, out_dim, out_len),  # which token in the vocabulary is the outcome of the test
            "outcome_mask": (batch, out_dim, out_len),  # which tokens in the vocabulary are valid outcomes

            # cached model evaluations
            "activation": ...,   # cached model activation at the cut point
            "state_features": ...,   # cached model test outcomes q(x)
            "base_behaviour": ...,   # cached model behaviour r(x, A_x)
            "transport_gate": ...,   # info for the semantic transport
        }

        Returns:
            An ObjectiveResult containing the loss and metrics.
        '''
        diagnostics: dict[str, Tensor] = {}

        activation = batch['activation'].float()
        
        reference = None
        if self.objective.get("mass_mode", 'full') == 'reference':
            with torch.no_grad():
                reference = (
                    self.subject.behavior_mass(batch)
                    if (Z_ref := self.objective.get('reference_mass')) is None
                    else activation.new_full((len(activation),), float(Z_ref))
                )
            eps = torch.finfo(reference.dtype).eps
            reference = reference.clamp(eps, 1.0 - eps)

        # Lift the activation to the abstract state decomposition
        carrier, nuisance = self.abstraction.lift(activation)

        # * ----- CALIBRATION LOSS -----
        # Decode the abstract state to the predictive feature space
        pred_features = self.decoder(carrier)

        # Compute the calibration loss
        calibration_loss = mse_loss(pred_features, batch["state_features"])

        # * ----- CAUSAL INTERVENTION LOSS -----
        # NOTE: donor and nuisance interventions get difference derangement
        # samples, since donor requires the existence of a semantic transport
        # between source and base, while nuisance does not.
        groups = batch.get("transport_group")
        nuisance_donor = derangement(len(carrier), device=carrier.device, generator=generator)
        
        causal = activation - nuisance
        
        # Check whether a supervised path was provided from the data module
        if (source := batch.get("source_activation")) is not None:
            source = source.float()
            _, source_nuisance = self.abstraction.lift(source)
            donor_causal = source - source_nuisance
            
            source_key = "source_gold_state_features" if self.target_kind == 'gold' else "source_state_features"
            source_state = batch[source_key]
        else:
            # This is the self-supervised branch
            semantic_donor = derangement(
                len(carrier), device=carrier.device, groups=groups, generator=generator
            )
            donor_causal = causal[semantic_donor]
            
            source_key = "gold_state_features" if self.target_kind == 'gold' else "state_features"
            source_state = batch[source_key][semantic_donor]

        # Compute the two intervention activations
        donor_activation = donor_causal + nuisance
        nuisance_activation = causal + nuisance[nuisance_donor]

        # § Perform the donor intervention experiment
        donor_steer = (batch['cut_positions'], donor_activation)
        donor_log_probability = self.subject.outcome_log_probability(
            batch, donor_steer
        )
        donor_targ = self.transport(
            source_state=source_state,
            base_behaviour=batch["base_behaviour"],
            transport_gate=batch.get("transport_gate"),
        )
        
        if self.target_kind == 'hard':
            donor_targ = one_hot(
                donor_targ.argmax(-1),
                num_classes=donor_targ.shape[-1],
            ).to(donor_targ.dtype)

        weight = batch.get('donor_weight')
        if weight is not None and weight.shape != (len(carrier),):
            raise ValueError("donor_weight must contain one scalar per record.")
        
        per_example = cross_entropy(
            donor_log_probability,
            donor_targ,
            mode=self.objective.get("mass_mode", "full"),
            reference=reference,
        )

        donor_loss = (
            per_example.mean()
            if weight is None
            else (per_example * weight).mean()
        )

        # $ Perform the nuisance intervention experiment
        nuisance_loss = torch.tensor(0.0, device=self.device)
        if self.loss_weights.nuisance > 0:
            nuisance_steer = (batch['cut_positions'], nuisance_activation)
            nuisance_targ = batch['base_behaviour']
            nuisance_log_probability = self.subject.outcome_log_probability(
                batch, nuisance_steer
            )
            nuisance_pred = nuisance_log_probability.softmax(-1)

            nuisance_loss = mse_loss(nuisance_pred, nuisance_targ)

        # * -------- SEMANTIC CONTROL LOSS --------
        control_loss = torch.tensor(0.0, device=self.device)
        if self.controller is not None:
            # Sample control targets
            request = self._sample_request(
                features=batch["state_features"],
                generator=generator,
                mixture=self.control.get('mixture'),
                corner_mass=self.control.get('corner_mass', 0.8)
            )
            
            # Transport the requested features into the base context
            control_target = self.transport(
                source_state=request,
                base_behaviour=batch["base_behaviour"],
                transport_gate=batch.get("transport_gate"),
            )
            
            # Get the current subject behaviour in the base context
            match self.control.get('readout', 'measured'):
                case 'measured':
                    current = batch["base_behaviour"]
                case 'decoder':
                    current = self.decoder(carrier)
                case _:
                    raise ValueError(f"Unknown control readout: {self.control.get('readout')}.")
            
            # Write the control edits into the causal carrier
            control_activations, control_delta = self._write_control(
                carrier=carrier,
                nuisance=nuisance,
                target=control_target,
                current=current,
            )
            
            # Steer model behaviour based on the control edits
            control_steer = (batch['cut_positions'], control_activations)
            control_log_probability = self.subject.outcome_log_probability(
                batch, control_steer
            )
            
            # Compute the cross-entropy control loss
            control_loss = cross_entropy(
                control_log_probability,
                control_target,
                mode=self.objective.get("mass_mode", "full"),
                reference=reference,
            ).mean()

        # * -------- DIAGNOSTICS BLOCK --------
        if groups is not None:
            diagnostics['nuisance_cross_group_rate'] = (
                groups[nuisance_donor] != groups
            ).float().mean()
            diagnostics['donor_cross_group_rate'] = (
                groups[semantic_donor] != groups
            ).float().mean()
        gate = batch.get("transport_gate")
        # IOI uses a [batch, source, target] table in this field, not a
        # per-record gate. Conditional risks apply only to binary gates.
        if gate is not None and gate.ndim == 1:
            gate = gate.bool()
            if gate.any():
                diagnostics["donor_loss_gate_true"] = per_example[gate].mean()
            if (~gate).any():
                diagnostics["donor_loss_gate_false"] = per_example[~gate].mean()
            diagnostics["gate_fraction"] = gate.float().mean()

        tiny = torch.finfo(donor_targ.dtype).tiny
        donor_log_mass = donor_log_probability.logsumexp(-1)
        conditional_log_probability = (
            donor_log_probability - donor_log_mass.unsqueeze(-1)
        )
        target_log = donor_targ.clamp_min(tiny).log()

        diagnostics["donor_target_mass"] = donor_targ.sum(-1).mean()
        diagnostics["donor_outcome_mass"] = donor_log_mass.exp().mean()
        entropy = -(donor_targ * target_log).sum(-1)
        divergence = (
            donor_targ * (target_log - conditional_log_probability)
        ).sum(-1)
        diagnostics["donor_entropy"] = entropy.mean()
        diagnostics["donor_conditional_kl"] = divergence.mean()
        diagnostics["donor_log_outcome_mass"] = donor_log_mass.mean()
        diagnostics["donor_cross_entropy"] = (
            -(donor_targ * donor_log_probability).sum(-1).mean()
        )
        if weight is not None:
            diagnostics["weighted_donor_entropy"] = (weight * entropy).mean()
            diagnostics["weighted_donor_conditional_kl"] = (weight * divergence).mean()
            diagnostics["weighted_donor_log_outcome_mass"] = (
                weight * donor_log_mass
            ).mean()
        
        if self.loss_weights.nuisance > 0:
            diagnostics["nuisance_outcome_mass"] = (
                nuisance_log_probability.logsumexp(-1).exp().mean()
            )
        
        # § Control diagnostic
        if self.controller is not None:
            realized = control_log_probability.softmax(-1)
            edit_norm = control_delta.norm(dim=-1)
            
            diagnostics['control_tv'] = (
                0.5 * (realized - control_target).abs().sum(-1).mean()
            )
            diagnostics['control_outcome_mass'] = control_log_probability.logsumexp(-1).exp().mean()
            diagnostics['control_edit_norm'] = edit_norm.mean()
            
            if (cap := self.control.get('cap')) is not None:
                diagnostics['control_edit_cap_fraction'] = (
                    (edit_norm > float(cap) - 1e-6).float().mean()
                )
            control_log_mass = control_log_probability.logsumexp(-1)
            control_conditional = control_log_probability - control_log_mass.unsqueeze(-1)
            tiny = torch.finfo(control_target.dtype).tiny
            diagnostics['control_conditional_kl'] = (
                control_target
                * (control_target.clamp_min(tiny).log() - control_conditional)
            ).sum(-1).mean()
        if reference is not None:
            diagnostics['reference_mass'] = reference.mean()
            if self.controller is not None:
                diagnostics['control_mass_error'] = (
                    control_log_mass.exp() - reference
                ).abs().mean()


        return ObjectiveResult(
            donor_loss=donor_loss,
            control_loss=control_loss,
            nuisance_loss=nuisance_loss,
            calibration_loss=calibration_loss,
            loss_weights=self.loss_weights,
            diagnostics={name: value.detach() for name, value in diagnostics.items()},
        )
    
    @torch.no_grad()
    def get_control(
        self,
        request: Tensor,
        activation: Tensor,
        base_behaviour: Tensor,
        transport_gate: Tensor | None = None,
    ) -> Tensor:
        """Return the activation edited to realize the transported request."""
        
        carrier, nuisance = self.abstraction.lift(activation)
        target = self.transport(
            source_state=request,
            base_behaviour=base_behaviour,
            transport_gate=transport_gate,
        )
        
        match self.control.get('readout', 'measured'):
            case 'measured':
                current = base_behaviour
            case 'decoder':
                current = self.decoder(carrier)
            case _:
                raise ValueError(f"Unknown control readout: {self.control.get('readout')}")
        
        edited, _ = self._write_control(
            carrier=carrier,
            nuisance=nuisance,
            target=target,
            current=current,
        )
        
        return edited
    
    def _sample_request(
        self,
        features: Tensor,
        *,
        mixture: list[tuple[Literal['donor', 'uniform', 'corner'], float]] | None = None,
        corner_mass: float = 0.8,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample one requested outcome distribution per record."""
        
        mixture = mixture or [('donor', 0.5), ('uniform', 0.5)]
        
        # This are candidate behavioural features
        fractions = []
        candidates = []
        
        for kind, fraction in mixture:
            if fraction <= 0:
                continue
            
            fractions.append(float(fraction))
            
            if kind == 'donor':
                donor = derangement(len(features), device=features.device, generator=generator)
                
                candidates.append(features[donor])
            elif kind == 'uniform':
                exponential = torch.empty_like(features).exponential_(generator=generator)
                uniform = exponential / exponential.sum(-1, keepdim=True)
                candidates.append(uniform)
            elif kind == 'corner':
                num_dim = features.shape[-1]
                corners = features.argmax(-1).unique()
                samples = torch.randint(0, len(corners), (len(features),), device=features.device, generator=generator)
                targets = one_hot(corners[samples], num_classes=num_dim).to(features.dtype)
                candidates.append(targets * corner_mass + (1 - corner_mass) / num_dim)
            else:
                raise ValueError(f"Unknown request kind: {kind}")
        
        if not candidates:
            raise ValueError("No valid request kinds were provided.")
        
        candidates = torch.stack(candidates, dim=0)
        fractions = torch.tensor(fractions, device=features.device) / sum(fractions)
        
        choice = torch.multinomial(fractions.expand(len(features), -1), 1, generator=generator)
        request_idx = choice.T.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        return torch.gather(candidates, 0, request_idx).squeeze(0)
    
    def _write_control(
        self,
        carrier: Tensor,
        nuisance: Tensor,
        target: Tensor,
        current: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Write the control edits into the causal carrier, preserving nuisance."""
        if self.controller is None:
            raise RuntimeError("Control is not enabled, but _write_control was called.")
        
        delta = self.controller(target, current, carrier)
        
        # Cap the control edits to a specified maximum magnitude, if requested
        if (control_cap := self.control.get("cap")) is not None:
            tiny = torch.finfo(delta.dtype).tiny
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(tiny)
            
            delta = delta * (control_cap / norm).clamp_max(1.0)
        
        return self.abstraction.compose(carrier + delta, nuisance), delta

    def _step(self, batch: Mapping[str, Tensor], batch_idx: int, split: str) -> Tensor:
        generator = None
        if split != "train":
            generator = torch.Generator(device=self.device).manual_seed(
                self.validation_seed + batch_idx
            )
        result = self.loss(batch, generator=generator)

        if (
            split == "train"
            and self.gradient_probe_every
            and batch_idx % self.gradient_probe_every == 0
        ):
            self._probe_gradients(result)

        values = {f"{split}/loss": result.loss.detach()}
        conditional = {"support/donor_loss_gate_true", "support/donor_loss_gate_false"}
        values |= {
            f"{split}/{name}": value.detach()
            for name, value in result.metrics.items() if name not in conditional
        }
        self.log_dict(
            values,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["input_ids"]),
        )
        # A conditional mean must be weighted by its class count, not by the
        # whole batch, when Lightning aggregates it across an epoch.
        for name in conditional & result.metrics.keys():
            gate = batch["transport_gate"].bool()
            count = int((gate if name.endswith("true") else ~gate).sum())
            self.log(
                f"{split}/{name}", result.metrics[name], on_step=False,
                on_epoch=True, sync_dist=True, batch_size=count,
            )
        return result.loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx, "val")

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in self.parameters() if parameter.requires_grad),
                float(self.grad_clip),
                error_if_nonfinite=True,
            )

    def configure_optimizers(self):
        general = [
            parameter for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("controller.")
        ]
        
        groups = [{"params": general, 'lr' : float(self.learning_rate)}]
        
        if self.controller is not None:
            groups.append({
                "params": self.controller.parameters(),
                "lr": float(self.control.get("learning_rate", self.learning_rate)),
            })
        
        return torch.optim.AdamW(
            groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _probe_gradients(self, result: ObjectiveResult) -> None:
        """Log each weighted term's gradient norm on the shared basis.

        The three terms compete for one parameter, the abstraction basis, so
        that is where their influence is comparable. Each call costs one extra
        backward pass per term through the frozen subject, which is why it is
        sampled rather than continuous. retain_graph keeps the graph alive for
        Lightning's own backward.
        """
        parameters = [
            parameter
            for parameter in self.abstraction.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            return

        weights = self.loss_weights
        terms = (
            ("donor", weights.donor, result.donor_loss),
            ("control", weights.control, result.control_loss),
            ("nuisance", weights.nuisance, result.nuisance_loss),
            ("calibration", weights.calibration, result.calibration_loss),
        )
        norms = {}
        for name, weight, term in terms:
            if weight == 0 or not term.requires_grad:
                norms[f"grad/{name}"] = torch.zeros((), device=self.device)
                continue
            gradients = torch.autograd.grad(
                weight * term,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            norms[f"grad/{name}"] = torch.linalg.vector_norm(
                torch.cat([
                    gradient.flatten()
                    for gradient in gradients
                    if gradient is not None
                ])
            ) if any(gradient is not None for gradient in gradients) else torch.zeros((), device=self.device)

        self.log_dict(norms, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
