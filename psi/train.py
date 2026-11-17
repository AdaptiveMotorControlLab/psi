"""Train one resolved Programmable Semantic Interventions (PSI) configuration."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from psi.data import get_task, get_datamodule
from psi.model import get_subject
from psi.psi import ProgrammableSemanticIntervention, psi_config

torch.set_float32_matmul_precision("high")


def _override_seed(config: Any, seed: int) -> Any:
    if isinstance(config, dict):
        return {
            key: seed if key == "seed" else _override_seed(value, seed)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [_override_seed(item, seed) for item in config]
    return config


def _apply_overrides(config: dict[str, Any], assignments: list[str]) -> None:
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Config override must have form path=value: {assignment!r}")
        path, raw = assignment.split("=", 1)
        keys = path.split(".")
        target: Any = config
        for key in keys[:-1]:
            if not isinstance(target, dict) or key not in target:
                raise KeyError(f"Unknown config path {path!r}.")
            target = target[key]
        if not isinstance(target, dict) or keys[-1] not in target:
            raise KeyError(f"Unknown config path {path!r}.")
        target[keys[-1]] = yaml.safe_load(raw)


def _manual_run_dir(seed: int | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version = f"{timestamp}-s{seed}" if seed is not None else timestamp
    return Path("runs/manual") / version


def run_training(
    config: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    seed_everything(config.get("seed"), workers=True)
    log_config = dict(config.get("logging", {}))

    if not dry_run:
        run_dir = Path(run_dir or _manual_run_dir(config.get("seed"))).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "resolved.yaml").open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        level = log_config.get("level", logging.INFO)
        if isinstance(level, str):
            level = logging.getLevelNamesMapping()[level.upper()]
        logging.getLogger().setLevel(int(level))

    tokenizer, subject = get_subject(config)
    task = get_task(config["data"], tokenizer)
    hyperparameters = psi_config(config)
    abstraction = hyperparameters.get("abstraction")
    infer_activation = (
        abstraction is not None
        and abstraction.get("kind", "linear") == "linear"
        and "activation_dim" not in abstraction
    )
    inferred = "test_dim" not in hyperparameters or infer_activation
    hyperparameters.setdefault("test_dim", task.num_outcomes)
    if infer_activation:
        hyperparameters["abstraction"] = dict(
            abstraction,
            activation_dim=subject.model.config.hidden_size,
        )
    if hyperparameters["test_dim"] != task.num_outcomes:
        raise ValueError(
            f"test_dim={hyperparameters['test_dim']} disagrees with "
            f"the task's {task.num_outcomes} outcomes."
        )
    if inferred and not dry_run:
        resolved = dict(config)
        resolved["psi"] = hyperparameters
        with (run_dir / "resolved.yaml").open("w") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)
    data = task.materialize(subject)
    dm = get_datamodule(config["data"], data)
    psi = ProgrammableSemanticIntervention.from_config(hyperparameters, subject)

    logger = (
        False
        if dry_run
        else CSVLogger(
            save_dir=run_dir, # type: ignore
            name="",
            version="",
            flush_logs_every_n_steps=int(log_config.get("flush_every_n_steps", 1)),
        )
    )
    callbacks: list[Callback] = []
    if not dry_run:
        checkpoint = config.get("checkpoint")
        if checkpoint:
            checkpoint = dict(checkpoint)
            checkpoint.pop("dirpath", None)
            checkpoint["dirpath"] = str(run_dir / "checkpoints") # type: ignore
            checkpoint["filename"] = "best"
            checkpoint["auto_insert_metric_name"] = False
            checkpoint["enable_version_counter"] = False
            callbacks.append(ModelCheckpoint(**checkpoint))

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=bool(config.get("checkpoint")) and not dry_run,
        fast_dev_run=dry_run,
        **config["trainer"],
    )
    trainer.fit(
        model=psi,
        datamodule=dm,
    )
    if not dry_run:
        (run_dir / "data_controls.json").write_text( # type: ignore
            json.dumps(dm.control_summary, indent=2, sort_keys=True) + "\n"
        )
    return None if dry_run else run_dir # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args()

    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    _apply_overrides(config, args.set)
    if args.seed is not None:
        config["seed"] = args.seed
        config = _override_seed(config, args.seed)
    run_training(config, run_dir=args.run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
