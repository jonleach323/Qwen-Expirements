"""MLP discovery on the tiny test model, wrapped models and real architectures."""

import copy
import doctest
from collections.abc import Callable

import pytest
import torch
from torch import nn
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from pushlab.models import discovery
from pushlab.models.discovery import (
    ModelInfo,
    find_decoder_layers,
    find_mlp_modules,
    model_info,
)
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


def _qwen3_5(sizes: dict[str, int]) -> nn.Module:
    return Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=sizes))


def _qwen3(sizes: dict[str, int]) -> nn.Module:
    return Qwen3ForCausalLM(
        Qwen3Config(
            **sizes, num_attention_heads=32, num_key_value_heads=8, head_dim=128
        )
    )


@pytest.mark.parametrize(
    ("name", "build"), [("qwen3.5-4b", _qwen3_5), ("qwen3-8b", _qwen3)]
)
def test_discovery_on_real_class_trees(
    name: str, build: Callable[[dict[str, int]], nn.Module]
) -> None:
    # The real classes at full size on the meta device (no weights). Qwen3.5's
    # tree is the vision-language wrapper, with a 27-block vision tower.
    spec = REGISTRY[name]
    sizes = {
        "num_hidden_layers": spec.layers,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.mlp_size,
    }
    with torch.device("meta"):
        model = build(sizes)
    assert model_info(model) == ModelInfo(
        layers=spec.layers, hidden_size=spec.hidden_size, mlp_size=spec.mlp_size
    )
    assert [index for index, _ in find_mlp_modules(model)] == list(range(spec.layers))


def test_two_models_in_one_container_are_refused(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    both = nn.ModuleList([tiny_model, copy.deepcopy(tiny_model)])
    with pytest.raises(ValueError, match="numbered from 0"):
        find_mlp_modules(both)


def test_layer_numbers_must_start_at_zero_without_gaps(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    first, second = (layer.mlp for layer in tiny_model.model.layers)
    with pytest.raises(ValueError, match="numbered from 0"):
        find_mlp_modules(nn.ModuleDict({"1": first, "2": second}))


def test_an_mlp_without_a_layer_number_is_refused(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    with pytest.raises(ValueError, match="no layer number"):
        find_mlp_modules(tiny_model.model.layers[0].mlp)


def test_mixture_of_experts_is_refused() -> None:
    # Its experts' down_proj weights are one fused tensor, not a module.
    config = Qwen3MoeConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
    )
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    with pytest.raises(ValueError, match="mixture-of-experts"):
        find_mlp_modules(model)


def test_model_info_refuses_a_model_without_mlps() -> None:
    with pytest.raises(ValueError, match="down_proj"):
        model_info(nn.Linear(4, 4))


def test_model_info_refuses_mlp_sizes_that_differ_by_layer(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    tiny_model.model.layers[1].mlp.down_proj = nn.Linear(96, 64, bias=False)
    with pytest.raises(ValueError, match="differ"):
        model_info(tiny_model)


def test_decoder_layers_of_the_tiny_model_are_its_blocks(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    assert find_decoder_layers(tiny_model) == list(tiny_model.model.layers)


def test_decoder_layers_inside_the_vision_language_wrapper(
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
) -> None:
    layers = tiny_vl_model.model.language_model.layers
    assert find_decoder_layers(tiny_vl_model) == list(layers)


@pytest.mark.parametrize(
    ("name", "build"), [("qwen3.5-4b", _qwen3_5), ("qwen3-8b", _qwen3)]
)
def test_decoder_layers_on_real_class_trees(
    name: str, build: Callable[[dict[str, int]], nn.Module]
) -> None:
    spec = REGISTRY[name]
    sizes = {
        "num_hidden_layers": spec.layers,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.mlp_size,
    }
    with torch.device("meta"):
        model = build(sizes)
    blocks = find_decoder_layers(model)
    assert len(blocks) == spec.layers
    for block, (_, mlp) in zip(blocks, find_mlp_modules(model), strict=True):
        assert block.mlp is mlp


def test_decoder_layers_inside_a_numbered_container(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    # In "0.model.layers.1.mlp" the layer is the last number, not the first.
    layers = list(tiny_model.model.layers)
    assert find_decoder_layers(nn.ModuleList([tiny_model])) == layers


def test_an_mlp_outside_any_decoder_layer_is_refused(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    # Numbered MLPs straight in a list: no block around them to read from.
    first, second = (layer.mlp for layer in tiny_model.model.layers)
    with pytest.raises(ValueError, match="not inside a decoder layer"):
        find_decoder_layers(nn.ModuleList([first, second]))


def test_docstring_examples_run() -> None:
    result = doctest.testmod(discovery)
    assert result.attempted > 0
    assert result.failed == 0
