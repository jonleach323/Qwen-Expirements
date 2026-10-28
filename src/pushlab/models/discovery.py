"""Find every MLP block in a model, wherever it sits in the module tree."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelInfo:
    """The MLP shape of a model: layer count, hidden size and MLP size."""

    layers: int
    hidden_size: int
    mlp_size: int


def find_mlp_modules(model: nn.Module) -> list[tuple[int, nn.Module]]:
    """Return (layer index, MLP module) for every module with a down_proj.

    The whole module tree is searched, so a language model nested inside a
    wrapper is found the same way as a plain one, as long as nothing else in
    the wrapper has a down_proj. That holds for Qwen3.5, whose vision tower
    uses other names. Each MLP's layer index is the last number in its
    module name, for example 14 in "model.layers.14.mlp", and the result is
    sorted by it.

    Anything that is not exactly one MLP per layer, numbered from 0, raises
    ValueError instead of returning a wrong answer. That covers two models
    in one container, a vision tower that also has down_proj modules (as in
    Gemma 4 or Qwen2.5-VL), and mixture-of-experts models, whose experts
    keep their down_proj weights in a single tensor. Dense models only.

    Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> config = Qwen3Config(
        ...     vocab_size=512,
        ...     hidden_size=64,
        ...     intermediate_size=128,
        ...     num_hidden_layers=2,
        ...     num_attention_heads=4,
        ...     num_key_value_heads=2,
        ...     head_dim=16,
        ... )
        >>> [index for index, _ in find_mlp_modules(Qwen3ForCausalLM(config))]
        [0, 1]
    """
    found = []
    for name, module in model.named_modules():
        down_proj = getattr(module, "down_proj", None)
        if isinstance(down_proj, torch.Tensor):
            raise ValueError(
                f"{name!r} holds fused expert weights; "
                "mixture-of-experts models are not supported"
            )
        if not isinstance(down_proj, nn.Module):
            continue
        numbers = [int(part) for part in name.split(".") if part.isdigit()]
        if not numbers:
            raise ValueError(f"MLP module {name!r} has no layer number in its name")
        found.append((numbers[-1], name, module))
    found.sort(key=lambda item: item[0])
    indices = [index for index, _, _ in found]
    if indices != list(range(len(found))):
        names = [name for _, name, _ in found]
        raise ValueError(
            "expected one MLP per layer, numbered from 0; "
            f"found layer numbers {indices} in {names}"
        )
    return [(index, module) for index, _, module in found]


def model_info(model: nn.Module) -> ModelInfo:
    """Read the layer count, hidden size and MLP size from a model's MLPs.

    Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> config = Qwen3Config(
        ...     vocab_size=512,
        ...     hidden_size=64,
        ...     intermediate_size=128,
        ...     num_hidden_layers=2,
        ...     num_attention_heads=4,
        ...     num_key_value_heads=2,
        ...     head_dim=16,
        ... )
        >>> model_info(Qwen3ForCausalLM(config))
        ModelInfo(layers=2, hidden_size=64, mlp_size=128)
    """
    mlps = find_mlp_modules(model)
    if not mlps:
        raise ValueError("no module with a down_proj found; not a language model?")
    shapes = {(m.down_proj.in_features, m.down_proj.out_features) for _, m in mlps}
    if len(shapes) != 1:
        raise ValueError(f"MLP sizes differ between layers: {sorted(shapes)}")
    ((mlp_size, hidden_size),) = shapes
    return ModelInfo(layers=len(mlps), hidden_size=hidden_size, mlp_size=mlp_size)
