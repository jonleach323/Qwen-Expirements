"""Neuron units: IDs, range checks and write directions."""

import dataclasses
import doctest

import numpy as np
import pytest
import torch
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration, Qwen3ForCausalLM

from pushlab.models.discovery import ModelInfo
from pushlab.models.load import load_model
from pushlab.models.registry import REGISTRY, ModelSpec
from pushlab.units import neuron
from pushlab.units.neuron import NeuronUnit

# Qwen3.5-4B's published shape, written out so a registry typo cannot hide here.
QWEN3_5_4B = ModelInfo(layers=32, hidden_size=2560, mlp_size=9216)


@pytest.mark.parametrize(
    "text",
    [
        "qwen3.5-4b:L14:N8949",
        "qwen3.5-4b:L13:N7956",
        "qwen3.5-4b:L0:N0",
        "qwen3-8b:L35:N12287",
        "tiny:L1:N127",
    ],
)
def test_ids_round_trip(text: str) -> None:
    unit = NeuronUnit.parse(text)
    assert str(unit) == text
    assert NeuronUnit.parse(str(unit)) == unit


def test_parse_reads_the_model_layer_and_index() -> None:
    unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
    assert unit == NeuronUnit(model_name="qwen3.5-4b", layer=14, index=8949)
    assert type(unit.layer) is int and type(unit.index) is int


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_model_can_name_its_neurons(name: str) -> None:
    # A future registry name that the ID format cannot carry fails here.
    unit = NeuronUnit(name, layer=0, index=0)
    assert NeuronUnit.parse(str(unit)) == unit


@pytest.mark.parametrize(
    "text",
    [
        "",
        "qwen3.5-4b",
        ":L14:N8949",  # no model
        "qwen3.5-4b:14:8949",  # no L and no N
        "qwen3.5-4b:L14:8949",  # no N
        "qwen3.5-4b:14:N8949",  # no L
        "qwen3.5-4b:N8949:L14",  # N before L
        "qwen3.5-4b:l14:n8949",  # lowercase
        "qwen3.5-4b:L-1:N8949",  # negative
        "qwen3.5-4b:L14:N-1",
        "qwen3.5-4b:L+14:N8949",  # a sign
        "qwen3.5-4b:Lx:N8949",  # not a number
        "qwen3.5-4b:L14.0:N8949",
        "qwen3.5-4b:L1_4:N8949",  # int() would read this as 14
        "qwen3.5-4b:L014:N8949",  # leading zeros: a second spelling
        "qwen3.5-4b:L14:N08949",
        "qwen3.5-4b:L١٤:N8949",  # Arabic-Indic digits for 14
        "qwen3.5-4b:L1٤:N8949",  # an ASCII 1, then an Arabic-Indic 4
        "qwen3.5-4b:L14:N8٩٤٩",
        "qwen3.5-4b:L１４:N8949",  # full-width digits for 14
        " qwen3.5-4b:L14:N8949",
        "qwen3.5-4b:L14:N8949 ",
        "qwen3.5-4b:L14:N8949\n",
        "qwen3.5-4b::L14:N8949",
        "qwen3.5-4b:L14:N8949:N1",
    ],
)
def test_malformed_ids_are_refused_with_the_expected_form(text: str) -> None:
    with pytest.raises(ValueError, match="is not a neuron ID such as") as error:
        NeuronUnit.parse(text)
    assert repr(text) in str(error.value)


@pytest.mark.parametrize("value", [None, 14, b"qwen3.5-4b:L14:N8949"])
def test_only_text_is_parsed(value: object) -> None:
    with pytest.raises(ValueError, match="is not a neuron ID"):
        NeuronUnit.parse(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["Qwen3.5-4B:L14:N8949", "qwen3.5-9b:L14:N8949"])
def test_models_outside_the_registry_are_refused(text: str) -> None:
    with pytest.raises(ValueError, match="not in the model registry") as error:
        NeuronUnit.parse(text)
    assert "qwen3.5-4b" in str(error.value)  # it lists the registered names


@pytest.mark.parametrize(
    ("layer", "index"),
    [(-1, 0), (0, -1), (True, 0), (0, False), (14.0, 0), ("14", 0), (np.int64(14), 0)],
)
def test_layer_and_index_must_be_whole_numbers_from_0(
    layer: object, index: object
) -> None:
    with pytest.raises(ValueError, match="whole number from 0"):
        NeuronUnit("qwen3.5-4b", layer, index)  # type: ignore[arg-type]


def test_units_are_values() -> None:
    unit = NeuronUnit("qwen3.5-4b", 14, 8949)
    assert {unit, NeuronUnit.parse("qwen3.5-4b:L14:N8949")} == {unit}
    assert NeuronUnit("qwen3-8b", 14, 8949) != unit
    with pytest.raises(dataclasses.FrozenInstanceError):
        unit.layer = 13  # type: ignore[misc]


@pytest.mark.parametrize(("layer", "index"), [(0, 0), (31, 9215)])
def test_validate_accepts_the_first_and_last_neuron(layer: int, index: int) -> None:
    NeuronUnit("qwen3.5-4b", layer, index).validate(QWEN3_5_4B)


def test_validate_refuses_layer_32_of_qwen3_5_4b() -> None:
    with pytest.raises(ValueError, match="layer 32 is out of range.* 0 to 31"):
        NeuronUnit("qwen3.5-4b", 32, 8949).validate(QWEN3_5_4B)


def test_validate_refuses_neuron_9216_of_qwen3_5_4b() -> None:
    with pytest.raises(ValueError, match="neuron 9216 is out of range.* 0 to 9215"):
        NeuronUnit("qwen3.5-4b", 14, 9216).validate(QWEN3_5_4B)


@pytest.mark.parametrize(("layer", "index"), [(0, 0), (0, 127), (1, 5), (1, 64)])
def test_write_direction_is_exactly_the_down_proj_column(
    tiny_model: Qwen3ForCausalLM, layer: int, index: int
) -> None:
    # Index 127 is past the hidden size (64), so a row would not even exist.
    direction = NeuronUnit("tiny", layer, index).write_direction(tiny_model)
    weight = tiny_model.model.layers[layer].mlp.down_proj.weight
    assert direction.dtype == torch.float32
    assert direction.shape == (64,)
    assert torch.equal(direction, weight[:, index])


@pytest.mark.parametrize("layer", [0, 1, 2, 3])
def test_write_direction_inside_the_vision_language_wrapper(
    tiny_downloads: dict[str, ModelSpec],
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
    layer: int,
) -> None:
    direction = NeuronUnit("tiny-vl", layer, 100).write_direction(tiny_vl_model)
    mlp = tiny_vl_model.model.language_model.layers[layer].mlp
    assert torch.equal(direction, mlp.down_proj.weight[:, 100])


def test_write_direction_is_a_copy_that_never_changes_the_model(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    weight = tiny_model.model.layers[1].mlp.down_proj.weight
    before = weight.detach().clone()
    direction = NeuronUnit("tiny", 1, 5).write_direction(tiny_model)
    assert direction.is_contiguous()
    assert not direction.requires_grad
    direction += 1.0
    assert torch.equal(weight.detach(), before)


def test_write_direction_of_a_bfloat16_model_is_exact(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    model = tiny_model.to(torch.bfloat16)
    column = model.model.layers[0].mlp.down_proj.weight[:, 5]
    direction = NeuronUnit("tiny", 0, 5).write_direction(model)
    assert direction.dtype == torch.float32
    assert torch.equal(direction, column.float())
    assert torch.equal(direction.to(torch.bfloat16), column)  # nothing rounded


@pytest.mark.parametrize(
    ("text", "message"),
    [("tiny:L2:N0", "layer 2 is out of range"), ("tiny:L0:N128", "neuron 128 is")],
)
def test_write_direction_refuses_a_neuron_the_model_lacks(
    tiny_model: Qwen3ForCausalLM, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NeuronUnit.parse(text).write_direction(tiny_model)


def test_write_direction_refuses_another_models_shape(
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
) -> None:
    # tiny:L1:N5 is in range for the 4-layer model too; only its shape differs,
    # as Qwen3.5-4B's L14 N8949 is in range for Qwen3-8B.
    with pytest.raises(ValueError, match="another model"):
        NeuronUnit("tiny", 1, 5).write_direction(tiny_vl_model)


def test_write_direction_takes_the_model_not_what_load_model_returns(
    tiny_downloads: dict[str, ModelSpec],
) -> None:
    loaded = load_model("tiny-lm", torch.float32, device="cpu")
    unit = NeuronUnit("tiny-lm", 1, 5)
    with pytest.raises(ValueError, match=r"load_model\(\.\.\.\)\.model, not a Loaded"):
        unit.write_direction(loaded)  # type: ignore[arg-type]
    weight = loaded.model.model.layers[1].mlp.down_proj.weight
    assert torch.equal(unit.write_direction(loaded.model), weight[:, 5])


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_write_direction_refuses_quantized_weights(
    tiny_model: Qwen3ForCausalLM, dtype: torch.dtype
) -> None:
    down_proj = tiny_model.model.layers[0].mlp.down_proj
    down_proj.weight = nn.Parameter(torch.zeros(64, 128).to(dtype), requires_grad=False)
    with pytest.raises(ValueError, match="quantized"):
        NeuronUnit("tiny", 0, 5).write_direction(tiny_model)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(neuron)
    assert result.attempted > 0
    assert result.failed == 0
