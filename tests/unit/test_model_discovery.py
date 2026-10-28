"""MLP discovery on the tiny test model, wrapped models and real architectures."""

import copy
import doctest

import pytest
import torch
from torch import nn
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from pushlab.models import discovery
from pushlab.models.discovery import ModelInfo, find_mlp_modules, model_info
from pushlab.models.registry import REGISTRY


def test_tiny_model_has_two_mlps_in_layer_order(tiny_model: Qwen3ForCausalLM) -> None:
    pairs = find_mlp_modules(tiny_model)
    assert [index for index, _ in pairs] == [0, 1]
    for index, module in pairs:
        assert module is tiny_model.model.layers[index].mlp
        assert module.down_proj.in_features == 128


def test_wrapping_in_a_container_gives_the_same_result(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    # Like Qwen3.5's packaging: the language model sits one level deeper,
    # next to a vision tower whose numbered blocks have no down_proj.
    wrapper = nn.ModuleDict(
        {
            "visual": nn.ModuleList([nn.Linear(8, 8) for _ in range(3)]),
            "language_model": tiny_model,
        }
    )
    assert find_mlp_modules(wrapper) == find_mlp_modules(tiny_model)


def test_model_info_matches_the_tiny_configuration(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    config = tiny_model.config
    assert model_info(tiny_model) == ModelInfo(
        layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        mlp_size=config.intermediate_size,
    )
    assert model_info(tiny_model) == ModelInfo(layers=2, hidden_size=64, mlp_size=128)


def _real_architecture(name: str) -> nn.Module:
    """Build a registered model's real class tree on the meta device (no weights)."""
    spec = REGISTRY[name]
    sizes = {
        "num_hidden_layers": spec.layers,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.mlp_size,
    }
    with torch.device("meta"):
        if spec.kind == "vision_language":
            return Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=sizes))
        return Qwen3ForCausalLM(
            Qwen3Config(
                **sizes, num_attention_heads=32, num_key_value_heads=8, head_dim=128
            )
        )


@pytest.mark.parametrize("name", ["qwen3.5-4b", "qwen3-8b"])
def test_real_architectures_match_the_registry(name: str) -> None:
    spec = REGISTRY[name]
    model = _real_architecture(name)
    assert model_info(model) == ModelInfo(
        layers=spec.layers, hidden_size=spec.hidden_size, mlp_size=spec.mlp_size
    )
    assert [index for index, _ in find_mlp_modules(model)] == list(range(spec.layers))


def test_two_models_in_one_container_are_refused(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    both = nn.ModuleList([tiny_model, copy.deepcopy(tiny_model)])
    with pytest.raises(ValueError, match="layer"):
        find_mlp_modules(both)


def test_model_info_refuses_a_model_without_mlps() -> None:
    with pytest.raises(ValueError, match="down_proj"):
        model_info(nn.Linear(4, 4))


def test_docstring_examples_run() -> None:
    result = doctest.testmod(discovery)
    assert result.attempted > 0
    assert result.failed == 0
