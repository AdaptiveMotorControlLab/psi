"""Public estimation, inspection and persistence of PSI carriers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Callable

import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback

from psi import __version__
from psi.data.module import PSIDataModule
from psi.data.tasks.program import ProgramTask
from psi.model.subject import LMSubject
from psi.program import At, Gated, Program, Slots, Table, column
from psi.prompts import Prompts
from psi.psi import ProgrammableSemanticIntervention, psi_config
from psi.scoring import ScoreReport, score_cache
from psi.utils.cut import ResidualCut

if TYPE_CHECKING:
    import pandas as pd
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class _History(Callback):
    """Collect epoch metrics without creating a logging directory."""

    def __init__(
        self,
    ) -> None:
        self.rows: list[dict[str, float]] = []

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: ProgrammableSemanticIntervention,
    ) -> None:
        metrics = {
            name: float(value.detach().cpu())
            for name, value in trainer.callback_metrics.items()
            if value.numel() == 1
        }
        self.rows.append({"epoch": trainer.current_epoch, **metrics})


class PSI:
    """Learn and inspect a carrier for a declared prompt program.

    The subject language model stays frozen. ``fit`` trains the carrier and
    decoder through the existing PSI objective. With ``steering=True`` it also
    trains a controller, so the same fit supports both swaps and requests.

    Examples
    --------
    >>> import psi
    >>> prompts, program = psi.presets.mcqa(limit=64)
    >>> model = psi.PSI("Qwen/Qwen2.5-0.5B", layer=15, steering=True)
    >>> model.fit(prompts, program)
    >>> report = model.score(split="test")
    >>> response = model.steer(prompts.splits["test"][0], answer=2)
    """

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        device: str | torch.device = 'auto',
        dtype: str | torch.dtype = 'auto',
        model_cache_dir: str | Path | None = None,
    ) -> PSI:
        """Restore an estimator saved with :meth:`save`.

        Parameters
        ----------
        directory
            Directory containing ``psi.json`` and ``weights.safetensors``.
        model
            An already loaded subject. Supply its tokenizer too. Omit this
            argument to load the model ID and revision recorded in the save.
        tokenizer
            Fast tokenizer associated with an explicitly supplied model.
        device
            ``"auto"`` chooses CUDA when available and CPU otherwise. Use
            ``"cpu"`` to restore without occupying a GPU.
        dtype
            Subject precision. ``"auto"`` selects float16 on CUDA and float32
            on CPU. Learned carrier and controller weights use float32.
        model_cache_dir
            Hugging Face cache used when the recorded subject is loaded.

        Returns
        -------
        PSI
            A fitted estimator. Pass prompts to ``score`` after restoring;
            the save contains the program and learned weights.
        """
        from safetensors.torch import load_file

        directory = Path(directory)
        metadata = json.loads((directory / "psi.json").read_text())
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported PSI save format.")

        position = metadata["position"]
        if isinstance(position, dict):
            position = At(position["regex"])

        subject = model if model is not None else metadata["model"]
        result = cls(
            subject,
            tokenizer=tokenizer,
            revision=metadata["revision"],
            layer=metadata["layer"],
            rank=metadata["rank"],
            position=position,
            device=device,
            dtype=dtype,
            model_cache_dir=model_cache_dir,
            **metadata["parameters"],
        )

        result.program_ = Program.from_dict(metadata["program"])
        result.config_ = metadata["hyperparameters"]
        result.psi_ = ProgrammableSemanticIntervention.from_config(
            result.config_,
            result.subject,
        ).to(result.subject.device)

        state = load_file(
            str(directory / "weights.safetensors"),
            device=str(result.subject.device),
        )
        result.psi_.load_state_dict(state)
        result.psi_.eval()
        result._history = metadata["history"]
        result._fitted = True

        return result

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path,
        **kwargs: Any,
    ) -> PSI:
        """Load a local save directory or a PSI repository on Hugging Face.

        Parameters
        ----------
        repo_id
            A local directory containing a PSI save, or a Hub repository ID
            such as ``"organization/carrier-name"``.
        **kwargs
            Arguments forwarded to :meth:`load`, including ``model``,
            ``tokenizer``, ``device``, ``dtype`` and ``model_cache_dir``.

        Returns
        -------
        PSI
            The restored estimator. Hub downloads include only the metadata
            and learned weights; the subject is loaded by its recorded ID.
        """
        directory = Path(repo_id)
        if not directory.is_dir():
            from huggingface_hub import snapshot_download

            directory = Path(
                snapshot_download(
                    str(repo_id),
                    allow_patterns=["psi.json", "weights.safetensors"],
                )
            )

        return cls.load(directory, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        config: dict[str, Any],
        checkpoint: str | Path,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        subject: LMSubject | None = None,
    ) -> PSI:
        """Wrap a CLI checkpoint for scoring on its prepared task.

        Parameters
        ----------
        config
            The run's resolved YAML configuration, loaded as a dictionary.
        checkpoint
            Lightning checkpoint containing the learned ``state_dict``.
        tokenizer, subject
            Optional preloaded pair, useful for offline comparisons. Supply
            both or let the configuration load them.

        Returns
        -------
        PSI
            An estimator whose ``score()`` uses the CLI task cache and the
            same family-macro classification definition as the native audit.
            Generic program checkpoints also expose reading and persistence.
            Legacy task checkpoints retain their task-specific score targets.
        """
        from psi.data import get_task

        if subject is None:
            from psi.model import get_subject

            tokenizer, subject = get_subject(config)
        if tokenizer is None:
            raise ValueError("A supplied subject also requires its tokenizer.")

        hyperparameters = psi_config(config)
        data = config["data"]
        result = cls(
            subject.model,
            tokenizer=tokenizer,
            layer=config["cut"]["layer"],
            rank=hyperparameters["abstraction"]["carrier_rank"],
            position=config["cut"]["position"],
            device=subject.device,
            dtype=next(subject.model.parameters()).dtype,
            target=hyperparameters.get("target_kind", "soft"),
            epochs=config["trainer"].get("max_epochs", 8),
            learning_rate=hyperparameters.get("learning_rate", 3e-4),
            seed=config.get("seed", 11),
            steering=hyperparameters.get("loss_weights", {}).get("control", 0) > 0,
            cache_dir=data["cache_root"],
            correctness_filter=data.get("correctness_filter", False),
            quality_gate=data.get("quality_gate", 0),
            max_length=data["max_length"],
            batch_size=data.get("record_batch_size", 32),
            training_pool=data.get("training_pool", "all"),
        )

        result.subject = subject
        result.task_ = get_task(data, tokenizer)
        result.cache_ = result.task_.materialize(subject)
        result.config_ = hyperparameters
        result.psi_ = ProgrammableSemanticIntervention.from_config(
            hyperparameters,
            subject,
        ).to(subject.device)

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )["state_dict"]
        result.psi_.load_state_dict(state)
        result.psi_.eval()

        result._fitted = True

        if isinstance(result.task_, ProgramTask):
            result.program_ = result.task_.program
            result.prompts_ = result.task_.prompts

        return result

    @property
    def history(
        self,
    ) -> pd.DataFrame:
        """Epoch metrics as a DataFrame, including training and validation loss.

        The frame is empty before fitting. ``epoch`` is zero-based. Other
        column names come directly from the PSI training objective.
        """
        import pandas as pd

        return pd.DataFrame(self._history)

    def __init__(
        self,
        model: str | PreTrainedModel,
        *,
        layer: int,
        rank: int | None = 16,
        position: str | At | Callable[[str], int] = 'last_token',
        tokenizer: PreTrainedTokenizerBase | None = None,
        device: str | torch.device = 'auto',
        dtype: str | torch.dtype = 'auto',
        target: str = 'soft',
        epochs: int = 8,
        batch_size: int = 32,
        learning_rate: float = 0.0003,
        seed: int = 11,
        steering: bool = False,
        cache_dir: str | Path | None = None,
        correctness_filter: bool = False,
        quality_gate: float = 0.0,
        max_length: int = 128,
        training_pool: str = 'all',
        revision: str | None = None,
        model_cache_dir: str | Path | None = None,
        progress: bool = True,
    ) -> None:
        """Configure the subject, site and training budget.

        Parameters
        ----------
        model
            Hugging Face causal-model ID or an already loaded model. Residual
            cuts require a model exposing ``model.layers``. The subject's
            parameters are frozen when the estimator is created.
        layer
            Zero-based residual layer to read and patch.
        rank
            Carrier dimension. It must fit within the model's hidden size.
            ``None`` selects the smaller of 16 and that hidden size.
        position
            ``"last_token"``, ``At(regex)``, or a callable receiving prompt
            text and returning a zero-based character index. ``At`` selects
            the last token of the first match. Callable positions cannot be
            saved; use ``At`` for a portable estimator.
        tokenizer
            Required with a loaded model. It must provide fast offset
            mappings. Prompt padding is set to the left; answer sequences
            are padded on the right. An existing padding token is retained.
        device
            ``"auto"``, ``"cpu"``, ``"cuda"`` or an explicit CUDA device.
            Automatic selection uses CUDA when it is available.
        dtype
            Subject precision, as a torch dtype or name. Automatic selection
            uses float32 on CPU and float16 on CUDA.
        target
            ``"soft"`` uses the frozen model's distribution, ``"hard"`` its
            argmax, and ``"gold"`` a declared gold column.
        epochs
            Number of training epochs for the existing Lightning trainer.
        batch_size
            Batch size for recording, fitting and scoring. Training batches
            need at least two compatible observations.
        learning_rate
            AdamW learning rate for the learned carrier and decoder.
        seed
            Seed for optimization and training-batch randomization. Prompt
            splitting uses the seed stored in the Prompts object.
        steering
            Train the gain controller jointly with the carrier. Enable this
            before fitting to reuse the fit for donor-free ``steer`` calls.
        cache_dir
            Task-cache root. ``None`` uses ``data/psi``. For a notebook that
            should write only to temporary storage, pass a temporary folder.
        correctness_filter
            Keep only prompts whose declared argmax agrees with gold.
            Requires a gold column and defaults to false.
        quality_gate
            Minimum fraction agreeing with gold when filtering is enabled.
        max_length
            Maximum number of prompt tokens. Longer prompts are truncated.
            Choose a value that retains the intended answer position.
        training_pool
            ``"all"`` uses all eligible observations. ``"base_only"`` uses
            original recipients. ``"matched_all"`` matches their count with
            a seeded sample from the full pool. Evaluation pairs are separate.
        revision
            Subject model revision. Saved estimators retain the resolved
            revision when Hugging Face supplies one.
        model_cache_dir
            Hugging Face cache for subject weights and tokenizer files.
            This directory is separate from the prepared task cache.
        progress
            Show Lightning's progress display. Set false for a quiet cell.
            Fit creates no logger or checkpoint directory.
        """
        if target not in ("soft", "hard", "gold"):
            raise ValueError("target must be soft, hard or gold.")

        self.model_id = model if isinstance(model, str) else model.name_or_path
        self.revision = revision
        self.layer = int(layer)
        self.position = position

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        if dtype == "auto":
            dtype = "float16" if self.device.startswith("cuda") else "float32"
        self.dtype = str(dtype).removeprefix("torch.")

        self.target = target
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.steering = bool(steering)
        self.cache_dir = cache_dir
        self.correctness_filter = correctness_filter
        self.quality_gate = quality_gate
        self.max_length = int(max_length)
        self.training_pool = training_pool
        self.progress = bool(progress)
        self._history: list[dict[str, float]] = []
        self._fitted = False

        if isinstance(model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                model,
                revision=revision,
                cache_dir=model_cache_dir,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                cache_dir=model_cache_dir,
                dtype=getattr(torch, self.dtype),
            )

        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError(
                "A loaded model needs a fast tokenizer with offset mappings."
            )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a padding or EOS token.")

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "PSI residual cuts require a causal model exposing model.layers."
            )
        if not 0 <= self.layer < len(model.model.layers):
            raise ValueError(f"layer must be in [0, {len(model.model.layers)}).")

        self.rank = min(16, model.config.hidden_size) if rank is None else int(rank)
        if not 0 < self.rank <= model.config.hidden_size:
            raise ValueError("rank must be between one and the model hidden size.")

        model.to(device=self.device, dtype=getattr(torch, self.dtype))
        position_name = "last_token" if position == "last_token" else "selected"
        self.subject = LMSubject(model, ResidualCut(self.layer, position_name))

    def fit(
        self,
        prompts: Prompts,
        program: Program,
    ) -> PSI:
        """Learn a carrier and, when enabled, its controller.

        Parameters
        ----------
        prompts
            Text, metadata and reproducible train/validation/test splits.
            Pair annotations are used for evaluation; the shared data module
            draws training donors by derangement inside each batch.
        program
            Answers and their transport rule. A leading space is part of an
            answer. Gold targets and correctness filtering need a gold column.

        Returns
        -------
        PSI
            This estimator, with ``program_``, ``check_``, ``cache_`` and
            ``history`` populated. A repeated fit starts a new carrier.

        Notes
        -----
        Pre-flight diagnostics run before training. A matching explicit
        ``program.check(model, prompts)`` is reused because the subject stays
        frozen. The fit uses the existing PSI objective and data module.
        """
        needs_gold = self.target == "gold" or self.correctness_filter
        if needs_gold and not (program.gold or prompts.gold):
            raise ValueError(
                "Gold targets and correctness filtering require a gold column."
            )

        self._fitted = False
        self._history = []
        seed_everything(self.seed, workers=True, verbose=False)
        self.program_ = program
        self.prompts_ = prompts
        self.check_ = program.check(self, prompts)

        self.task_ = ProgramTask.from_objects(
            prompts,
            program,
            self.tokenizer,
            position=self.position,
            cache_root=str(self.cache_dir or "data/psi"),
            max_length=self.max_length,
            quality_gate=self.quality_gate,
            correctness_filter=self.correctness_filter,
            record_batch_size=self.batch_size,
        )
        self.program_ = self.task_.program
        self.cache_ = self.task_.materialize(self.subject)

        self.config_ = {
            "test_dim": self.task_.num_outcomes,
            "abstraction": {
                "kind": "linear",
                "carrier_rank": self.rank,
                "activation_dim": self.subject.model.config.hidden_size,
            },
            "transport": program.transport,
            "target_kind": self.target,
            "objective": {"mass_mode": "conditional"},
            "learning_rate": self.learning_rate,
            "loss_weights": {"donor": 1.0, "nuisance": 0.25, "calibration": 1.0},
        }

        if self.steering:
            # Joint conditional MCQA control defaults, including its measured cap.
            self.config_["loss_weights"] = {
                "donor": 1.0,
                "nuisance": 0.0,
                "calibration": 0.0,
                "control": 1.0,
            }
            self.config_["control"] = {
                "controller": {
                    "kind": "gain",
                    "test_dim": self.task_.num_outcomes,
                    "carrier_dim": self.rank,
                    "write_space": "log_odds",
                },
                "readout": "measured",
                "donor_fraction": 0.5,
                "cap": 104.4299,
                "learning_rate": 0.1,
            }

        self.psi_ = ProgrammableSemanticIntervention.from_config(
            self.config_,
            self.subject,
        ).to(self.subject.device)
        data_module = PSIDataModule(
            {
                "batch_size": self.batch_size,
                "num_workers": 0,
                "seed": self.seed,
                "training_pool": self.training_pool,
            },
            self.cache_,
        )

        history = _History()
        device = self.subject.device
        devices = [device.index or 0] if device.type == "cuda" else 1
        self.trainer_ = Trainer(
            max_epochs=self.epochs,
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=devices,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=self.progress,
            callbacks=[history],
            num_sanity_val_steps=0,
        )
        self.trainer_.fit(self.psi_, datamodule=data_module)
        self._history = history.rows
        self.psi_.eval()
        self._fitted = True

        return self

    def score(
        self,
        prompts: Prompts | None = None,
        *,
        split: str = 'test',
    ) -> ScoreReport:
        """Compare held-out interchanges with no edit and the declared rule.

        Parameters
        ----------
        prompts
            Prompt collection to evaluate. Omit it to use the fitted cache.
            A restored estimator needs this argument on its first score call.
        split
            ``"test"`` by default. Use ``"validation"`` while selecting a
            layer; reserve test for the selected estimator.

        Returns
        -------
        ScoreReport
            Family-macro agreement with cached pair targets, its no-edit
            baseline, mean answer mass, counts and frozen-model agreement.
            Programs with gold use gold-derived pair targets; model agreement
            separately follows the frozen model's own readings.
            Jointly trained models also report requested-answer success.
            Every success metric uses argmax within the declared answer list.
        """
        self._require_fitted()

        if prompts is None:
            if not hasattr(self, "cache_"):
                raise ValueError("Provide prompts when scoring a loaded estimator.")
            return score_cache(self.psi_, self.cache_, split, self.batch_size)

        task = ProgramTask.from_objects(
            prompts,
            self.program_,
            self.tokenizer,
            position=self.position,
            cache_root=str(self.cache_dir or "data/psi"),
            max_length=self.max_length,
            quality_gate=self.quality_gate,
            correctness_filter=self.correctness_filter,
            record_batch_size=self.batch_size,
        )
        cache = task.materialize(self.subject)

        return score_cache(self.psi_, cache, split, self.batch_size)

    @torch.inference_mode()
    def reading(
        self,
        prompt: str | dict[str, Any],
        *,
        options: list[str] | None = None,
    ) -> pd.Series:
        """Read the frozen model's distribution over the program's answers.

        Parameters
        ----------
        prompt
            Prompt text or a metadata record containing ``prompt``. Known
            fitted text reuses its metadata. New text with Slots needs an
            options column in its record, or an explicit ``options`` list.
        options
            Exact per-prompt answer strings for a Slots program.
            Include any leading spaces required by the continuation spelling.

        Returns
        -------
        pandas.Series
            Conditional probabilities indexed by this prompt's answer strings.
            The entries sum to one; ``check`` measures their full answer mass.
        """
        import pandas as pd

        self._require_fitted()
        row = self._resolve_prompt(prompt, options)
        batch = self._prepare_query(row)
        probabilities = self.subject.behavior(batch)[0].cpu().numpy()

        return pd.Series(
            probabilities,
            index=self.program_.answers.resolve([row])[0],
            name="probability",
        )

    @torch.inference_mode()
    def swap(
        self,
        *,
        base: str | dict[str, Any],
        donor: str | dict[str, Any],
        base_options: list[str] | None = None,
        donor_options: list[str] | None = None,
    ) -> pd.DataFrame:
        """Replace the base's carrier coordinates with the donor's.

        Parameters
        ----------
        base, donor
            Prompt strings or metadata records. The base receives the edit;
            its nuisance activation stays fixed. The donor supplies carrier
            coordinates. Records are needed for unfamiliar Slots prompts.
        base_options, donor_options
            Optional answer lists for new column-based Slots prompts. Values
            programs share one list and do not require these arguments.

        Returns
        -------
        pandas.DataFrame
            One row per slot, with each prompt's answer spelling, both frozen
            readings and the base reading after interchange. With Slots, a
            donor's chosen slot refers to the base's own option in that slot.
        """
        import pandas as pd

        self._require_fitted()
        base_row = self._resolve_prompt(base, base_options)
        donor_row = self._resolve_prompt(donor, donor_options)
        base_batch = self._prepare_query(base_row, capture=True)
        donor_batch = self._prepare_query(donor_row, capture=True)

        edited = self.psi_.abstraction.interchange(
            base_batch["activation"].float(),
            donor_batch["activation"].float(),
        )
        base_before = self.subject.behavior(base_batch)[0]
        donor_before = self.subject.behavior(donor_batch)[0]
        base_after = self.subject.behavior(
            base_batch,
            (base_batch["cut_positions"], edited),
        )[0]

        return pd.DataFrame(
            {
                "base_answer": self.program_.answers.resolve([base_row])[0],
                "donor_answer": self.program_.answers.resolve([donor_row])[0],
                "base_before": base_before.cpu().numpy(),
                "donor_before": donor_before.cpu().numpy(),
                "base_after": base_after.cpu().numpy(),
            }
        )

    @torch.inference_mode()
    def steer(
        self,
        prompt: str | dict[str, Any],
        *,
        answer: str | int,
        options: list[str] | None = None,
    ) -> pd.DataFrame:
        """Request an answer through the jointly trained controller.

        Parameters
        ----------
        prompt
            Prompt text or a metadata record, as accepted by ``reading``.
        answer
            Exact declared answer string or zero-based slot index. Gated and
            table programs transport this request through their declared rule.
        options
            Answer strings for a new Slots prompt.

        Returns
        -------
        pandas.DataFrame
            The declared answers, before and after probabilities, and the
            requested one-hot vector. Use the same fitted estimator for swaps
            and steering; ``steering=True`` is required at fit time.
        """
        import pandas as pd

        self._require_fitted()
        if self.psi_.controller is None:
            raise RuntimeError("steer requires an estimator fitted with steering=True.")

        row = self._resolve_prompt(prompt, options)
        answers = self.program_.answers.resolve([row])[0]
        index = answers.index(answer) if isinstance(answer, str) else int(answer)
        if not 0 <= index < len(answers):
            raise ValueError("Requested slot is outside the answer list.")

        batch = self._prepare_query(row, capture=True)
        before = self.subject.behavior(batch)
        request = torch.zeros_like(before)
        request[:, index] = 1.0

        gate = None
        if isinstance(self.program_.rule, Gated):
            gate = torch.tensor(
                [column(row, self.program_.rule.column)],
                device=self.subject.device,
            )
        if isinstance(self.program_.rule, Table):
            tables = self.program_.rule.compile([row], len(answers))
            table_column = self.program_.compatibility.get("table_column")
            if table_column:
                tables = [column(row, table_column)]
            gate = torch.tensor(tables, device=self.subject.device, dtype=before.dtype)

        edited = self.psi_.get_control(
            request,
            batch["activation"].float(),
            before,
            gate,
        )
        after = self.subject.behavior(
            batch,
            (batch["cut_positions"], edited),
        )

        return pd.DataFrame(
            {
                "answer": answers,
                "before": before[0].cpu().numpy(),
                "after": after[0].cpu().numpy(),
                "requested": request[0].cpu().numpy(),
            }
        )

    def save(
        self,
        directory: str | Path,
    ) -> Path:
        """Save the program, configuration and learned tensors.

        Parameters
        ----------
        directory
            Destination for ``psi.json`` and ``weights.safetensors``. Subject
            weights and task caches are referenced by the metadata.

        Returns
        -------
        pathlib.Path
            The save directory. ``PSI.load(directory)`` restores the learned
            carrier, decoder and optional controller. Callable positions and
            row-dependent callable tables require portable declarations before
            saving; a regex position and a metadata table provide those forms.
        """
        from safetensors.torch import save_file

        self._require_fitted()
        if not hasattr(self, "program_"):
            raise ValueError(
                "Saving requires a declared Program, as in a generic CLI run."
            )
        if callable(self.position):
            raise ValueError("A callable position cannot be serialized. Use At(regex).")

        rows = self.prompts_.all_rows if hasattr(self, "prompts_") else None
        position = self.position
        if isinstance(position, At):
            position = {"regex": position.regex}

        parameter_names = (
            "target",
            "epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "steering",
            "correctness_filter",
            "quality_gate",
            "max_length",
            "training_pool",
        )
        metadata = {
            "format_version": 1,
            "package_version": __version__,
            "model": self.model_id,
            "revision": self.revision
            or getattr(self.subject.model.config, "_commit_hash", None),
            "layer": self.layer,
            "rank": self.rank,
            "position": position,
            "program": self.program_.to_dict(rows),
            "hyperparameters": self.config_,
            "parameters": {name: getattr(self, name) for name in parameter_names},
            "history": self._history,
        }
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.psi_.state_dict().items()
        }

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(state, str(directory / "weights.safetensors"))
        (directory / "psi.json").write_text(json.dumps(metadata, indent=2) + "\n")

        return directory

    def push_to_hub(
        self,
        repo_id: str | Path,
    ) -> Any:
        """Save locally or upload a saved carrier to Hugging Face.

        Parameters
        ----------
        repo_id
            An existing local directory or a path beginning with ``.`` or
            ``/`` saves locally. A Hub ID such as ``"organization/name"``
            creates that repository if needed and uploads the save files.

        Returns
        -------
        pathlib.Path or huggingface_hub.CommitInfo
            The local save path or the upload result. Hub uploads use the
            credentials configured by the caller.
        """
        path = Path(repo_id)
        if path.exists() or str(repo_id).startswith((".", "/")):
            return self.save(path)

        from huggingface_hub import HfApi

        directory = Path(tempfile.mkdtemp(prefix="psi-upload-"))
        self.save(directory)

        hub = HfApi()
        hub.create_repo(str(repo_id), exist_ok=True)
        return hub.upload_folder(repo_id=str(repo_id), folder_path=str(directory))

    def _require_fitted(
        self,
    ) -> None:
        """Keep the fitted-state error consistent across public operations."""
        if not self._fitted:
            raise RuntimeError("Call fit or load a saved estimator first.")

    def _resolve_prompt(
        self,
        prompt: str | dict[str, Any],
        options: list[str] | None,
    ) -> dict[str, Any]:
        """Recover known metadata and validate options for all query methods."""
        if not isinstance(prompt, (str, dict)):
            raise TypeError("Provide a prompt string or a metadata record.")

        row = dict(prompt) if isinstance(prompt, dict) else {"prompt": prompt}
        if "prompt" not in row:
            raise ValueError("A prompt record needs a prompt field.")

        if isinstance(prompt, str) and hasattr(self, "prompts_"):
            matches = [
                record
                for record in self.prompts_.all_rows
                if record["prompt"] == prompt
            ]
            if matches:
                row = dict(matches[0])
                answer_rows = [
                    self.program_.answers.resolve([record]) for record in matches
                ]
                if any(answers != answer_rows[0] for answers in answer_rows[1:]):
                    raise ValueError(
                        "This text has several option lists; pass a prompt record."
                    )

        if not isinstance(self.program_.answers, Slots):
            return row

        if options is not None:
            name = self.program_.answers.column
            if isinstance(name, str):
                row[name] = list(options)
            else:
                row["_psi_options"] = list(options)

        try:
            self.program_.answers.resolve([row])
        except (KeyError, IndexError) as error:
            raise ValueError(
                "Slots needs options for a new prompt. Pass a record containing "
                "its options column, or options=."
            ) from error

        return row

    def _prepare_query(
        self,
        row: dict[str, Any],
        *,
        capture: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Use the task's one-row encoder for all public query methods."""
        return ProgramTask.prepare_query(
            row,
            self.program_,
            self.tokenizer,
            self.subject,
            position=self.position,
            max_length=self.max_length,
            capture=capture,
        )


def scan_layers(
    model_id: str | PreTrainedModel,
    prompts: Prompts,
    program: Program,
    layers: list[int] | None = None,
    epochs: int = 1,
    rank: int = 16,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compare short fits at candidate layers using validation pairs.

    Parameters
    ----------
    model_id
        A Hub ID or loaded subject. Supply ``tokenizer`` for a loaded model.
        One frozen subject is shared across the independent carrier fits.
    prompts, program
        The same prompt collection and semantic declaration used by ``fit``.
    layers
        Zero-based residual layers. ``None`` scans every layer. A short list
        reduces the time needed for a first comparison.
    epochs, rank
        Training budget and carrier dimension for every candidate.
    **kwargs
        PSI constructor options such as ``device``, ``dtype``, ``batch_size``,
        ``max_length``, ``seed`` and ``cache_dir``. Each fit uses the same seed.

    Returns
    -------
    pandas.DataFrame
        One row per layer with validation agreement, no-edit baselines, mass
        and counts. Select a layer on validation, then fit it and evaluate on
        test separately. The scan leaves the subject weights frozen.
    """
    import pandas as pd
    from dataclasses import asdict

    options = dict(kwargs)
    options.pop("layer", None)
    estimator = PSI(model_id, layer=0, epochs=epochs, rank=rank, **options)
    depth = len(estimator.subject.model.model.layers)
    if layers is None:
        layers = list(range(depth))
    if any(not 0 <= layer < depth for layer in layers):
        raise ValueError(f"Every layer must be in [0, {depth}).")

    rows = []
    for layer in layers:
        estimator.layer = int(layer)
        estimator.subject.cut = ResidualCut(
            estimator.layer,
            estimator.subject.cut.position,
        )
        estimator.fit(prompts, program)
        report = estimator.score(split="validation")
        rows.append({"layer": layer, **asdict(report)})

    return pd.DataFrame(rows)
