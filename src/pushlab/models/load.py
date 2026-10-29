"""Load a registered model and its tokenizer from its downloaded folder."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from pushlab.models.discovery import ModelInfo, model_info
from pushlab.models.download import MARKER, is_downloaded
from pushlab.models.registry import REGISTRY, ModelSpec

# scripts/setup_pod.sh downloads each model into a folder of its name here.
POD_MODELS_DIR = Path("/workspace/models")


@dataclass(frozen=True)
class LoadedModel:
    """A model ready to generate, with its tokenizer, shape and origin.

    revision is the Hugging Face commit the folder was downloaded at.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec
    info: ModelInfo
    folder: Path
    revision: str


def default_models_dir() -> Path:
    """The folder that holds one folder per downloaded model.

    It is the pod's /workspace/models unless PUSHLAB_MODELS_DIR names another.

    Example:
        >>> default_models_dir()  # doctest: +SKIP
        PosixPath('/workspace/models')
    """
    return Path(os.environ.get("PUSHLAB_MODELS_DIR") or POD_MODELS_DIR)


def default_device() -> str:
    """'cuda' if torch can use a GPU, otherwise 'cpu'.

    Example:
        >>> default_device() in ("cuda", "cpu")
        True
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype(device: torch.device | str) -> torch.dtype:
    """bfloat16 on a GPU, like the plan's main runs, and float32 on a CPU.

    Example:
        >>> default_dtype("cuda"), default_dtype("cpu")
        (torch.bfloat16, torch.float32)
    """
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


def loading_class(spec: ModelSpec) -> type:
    """The transformers class that loads spec's kind of model.

    Qwen3.5 ships as a vision-language model; its model card loads it with
    AutoModelForMultimodalLM. pushlab only ever gives it text.

    Example:
        >>> loading_class(REGISTRY["qwen3.5-4b"]).__name__
        'AutoModelForMultimodalLM'
    """
    if spec.kind == "vision_language":
        return AutoModelForMultimodalLM
    if spec.kind == "causal_lm":
        return AutoModelForCausalLM
    raise ValueError(f"{spec.name!r} is built by the tests, not loaded from disk")


def load_model(
    name: str,
    dtype: torch.dtype,
    *,
    device: torch.device | str | None = None,
    models_dir: Path | None = None,
) -> LoadedModel:
    """Load the registered model name in dtype, with its tokenizer.

    It loads from models_dir/<name> (default: default_models_dir()), which
    must hold a finished download of the registry's ID and revision, and
    never downloads anything. device defaults to the GPU if there is one.
    The loaded model's shape must match the registry's.

    Example:
        >>> loaded = load_model("qwen3.5-4b", torch.bfloat16)  # doctest: +SKIP
        >>> loaded.info  # doctest: +SKIP
        ModelInfo(layers=32, hidden_size=2560, mlp_size=9216)
    """
    spec = REGISTRY.get(name)
    if spec is None:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"{name!r} is not in the model registry ({known})")
    model_class = loading_class(spec)
    folder = (models_dir or default_models_dir()) / name
    if not is_downloaded(spec, folder):
        raise FileNotFoundError(
            f"{name} ({spec.hf_id} at {spec.revision}) is not downloaded into "
            f"{folder}. On the GPU pod, add {name} to configs/models.yaml if it "
            "is not listed there, then run `bash scripts/setup_pod.sh` "
            "(docs/runbooks/pod.md). Elsewhere, set PUSHLAB_MODELS_DIR to the "
            "folder that holds the models."
        )
    revision = json.loads((folder / MARKER).read_text()).get("commit", spec.revision)
    model = model_class.from_pretrained(
        folder,
        dtype=dtype,
        device_map=str(device or default_device()),
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    info = model_info(model)
    expected = ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size)
    if info != expected:
        raise ValueError(
            f"{name} loaded with {info}, but the registry expects {expected}; "
            "check src/pushlab/models/registry.py"
        )
    return LoadedModel(model, tokenizer, spec, info, folder, revision)
