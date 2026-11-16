from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TokenizersBackend
from datasets import DatasetDict, load_dataset

if TYPE_CHECKING:
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

MODEL_IDS = {
    "gpt2": "openai-community/gpt2",
    "qwen": "Qwen/Qwen2.5-0.5B",
    "gemma": "google/gemma-2-2b",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

DATASET_IDS = {
    "arithmetic": "mib-bench/arithmetic_addition",
    "ravel": "mib-bench/ravel",
    "ioi": "mib-bench/ioi",
}

HF_HUB_HOME = 'data/huggingface/hub'
HF_DATA_HOME = 'data/huggingface/datasets'

def get_hf_model(
    model_id: str,
    device: torch.device | str | Literal['auto'] = "cpu",
    dtype: torch.dtype | str | None = torch.float16,
    cache_dir: str | None = HF_HUB_HOME,
) -> tuple[TokenizersBackend, _BaseModelWithGenerate]:
    model_name_or_path = MODEL_IDS.get(model_id, model_id)
    
    token_path = os.environ.get("HF_TOKEN_PATH")
    token = Path(token_path).read_text().strip() if token_path else None
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=token,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    dtype = dtype or torch.float32
    
    subject = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        token=token,
        dtype=dtype,
        device_map=device,
        cache_dir=cache_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    subject.config.use_cache = False
    return tokenizer, subject

def get_hf_dataset(
    name: str,
    subset: str | None = None,
) -> DatasetDict:
    data_id = DATASET_IDS.get(name, name)
    
    dataset = load_dataset(
        data_id,
        subset,
        cache_dir=HF_DATA_HOME,
    )
    return dataset
