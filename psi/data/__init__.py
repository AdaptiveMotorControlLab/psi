"""Task definitions and dataset configurations"""

from typing import Any
from transformers import TokenizersBackend

from psi.data.task import TaskPreparation
from psi.data.module import PSIDataModule
from psi.data.types import TaskCache

__all__ = ["get_task", "get_datamodule"]


def get_task(config: dict[str, Any], tokenizer: TokenizersBackend) -> "TaskPreparation":
    """Create a task from the given configuration."""

    task_name = config["task"]
    match task_name:
        case "program":
            from psi.data.tasks.program import ProgramTask
            return ProgramTask(config, tokenizer)
        case "mcqa":
            from psi.data.tasks.mcqa import MCQATask
            return MCQATask(config, tokenizer)
        case "arithmetic":
            from psi.data.tasks.arithmetic import ArithmeticTask
            return ArithmeticTask(config, tokenizer)
        case "ravel":
            from psi.data.tasks.ravel import RavelTask
            return RavelTask(config, tokenizer)
        case "arc":
            from psi.data.tasks.arc import ARCTask
            return ARCTask(config, tokenizer)
        case "ioi":
            from psi.data.tasks.ioi import IOITask
            return IOITask(config, tokenizer)

        case _:
            raise ValueError(f"Unknown task: {task_name}")

def get_datamodule(config: dict[str, Any], prepared : TaskCache) -> "PSIDataModule":
    """Create a data module from the given configuration."""

    match str(config.get("donor_pairing", "derangement")):
        case "derangement":
            return PSIDataModule(config, prepared)
        case "curated":
            from psi.data.supervised import SupervisedPairDataModule
            return SupervisedPairDataModule(config, prepared)
        case "contrastive":
            from psi.data.contrastive import ContrastivePairDataModule
            return ContrastivePairDataModule(config, prepared)
        case unknown:
            raise ValueError(f"Unknown donor_pairing: {unknown!r}")