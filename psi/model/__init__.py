from transformers import TokenizersBackend

from psi.model.control import SemanticController
from psi.model.subject import LMSubject
from psi.utils.cut import GPT2AttentionHeadCut, ResidualCut
from psi.utils.hf import get_hf_model

__all__ = ["LMSubject", "get_subject"]


def get_subject(config: dict) -> tuple[TokenizersBackend, LMSubject]:
    """Create a subject (language model) from the given configuration."""

    subject = config['subject']
    tokenizer, model = get_hf_model(
        str(subject["model"]),
        device=subject.get("device", "cpu"),  # type: ignore
        dtype=subject.get("dtype", "float32"),  # type: ignore
    )
    cut_config = config['cut']
    match kind := cut_config.get('kind', 'residual'):
        case 'residual':
            cut = ResidualCut.from_config(cut_config)
        case 'gpt2_attention_head':
            cut = GPT2AttentionHeadCut.from_config(cut_config)
        case _:
            raise ValueError(f"Unknown cut kind: {kind}")
    return tokenizer, LMSubject(model, cut)

def get_controller(config: dict) -> "SemanticController":
    """Create a controller from the given configuration."""
    from psi.model.control import GainController, AffineController

    controller_config = config['controller']
    match kind := controller_config.get('kind', 'gain'):
        case 'gain':
            return GainController.from_config(controller_config)
        case 'affine':
            return AffineController.from_config(controller_config)
        case _:
            raise ValueError(f"Unknown controller kind: {kind}")