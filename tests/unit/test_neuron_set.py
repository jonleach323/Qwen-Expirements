"""Neuron sets: what a set holds, its write direction and its dose (decision 0019)."""

import doctest
import math

import pytest
import torch
from transformers import Qwen3_5ForConditionalGeneration, Qwen3ForCausalLM

from pushlab.models.discovery import ModelInfo
from pushlab.units import dose, neuron_set
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import (
    NeuronSet,
    SetEntry,
    is_unit,
    unit_layers,
    unit_neurons,
)

NORMS = [7.0, 10.0]  # tiny's two layers; layer 0's differs, so a mix-up shows
TINY = ModelInfo(layers=2, hidden_size=64, mlp_size=128)
MIX = [(1, 5, 0.8), (0, 3, -1.2), (1, 9, 0.3)]  # two layers, mixed signs


def _set(entries=None, size: float | None = 4.0, **fields) -> NeuronSet:
    values = {"model_name": "tiny", "name": "mix", "source": "made up"} | fields
    return NeuronSet(
        values["model_name"],
        values["name"],
        MIX if entries is None else entries,
        values["source"],
        size,
    )


def _column(model: Qwen3ForCausalLM, layer: int, index: int) -> torch.Tensor:
    return model.model.layers[layer].mlp.down_proj.weight.detach()[:, index]


# --- What a set holds


def test_a_set_keeps_its_entries_in_order_as_set_entries() -> None:
    mix = _set(entries=[[1, 5, 1], (0, 3, -1.2)])  # lists and an int coefficient
    assert mix.entries == (SetEntry(1, 5, 1.0), SetEntry(0, 3, -1.2))
    assert type(mix.entries[0].coefficient) is float
    assert mix.layers == (0, 1)
    assert mix.neurons() == (NeuronUnit("tiny", 1, 5), NeuronUnit("tiny", 0, 3))


def test_its_id_is_model_set_name_and_never_a_neuron_id() -> None:
    assert str(_set(name="pirate-top50")) == "tiny:set:pirate-top50"
    with pytest.raises(ValueError, match="not a neuron ID"):
        NeuronUnit.parse(str(_set()))


def test_equal_fields_make_equal_sets_that_can_be_keys() -> None:
    a = _set(entries=[(1, 5, 1.0), (0, 3, 2.0)], size=3)
    b = _set(entries=([1, 5, 1], [0, 3, 2]), size=3.0)
    assert a == b and hash(a) == hash(b) and len({a: 1, b: 2}) == 1
    assert a != _set(entries=[(0, 3, 2.0), (1, 5, 1.0)], size=3)  # order counts
    assert a != _set(entries=[(1, 5, 1.0), (0, 3, 2.0)], size=3, source="other")
    with pytest.raises(AttributeError):
        a.size = 2.0  # type: ignore[misc]


def test_a_set_without_a_size_must_sit_in_one_layer() -> None:
    assert _set(entries=[(1, 5, 1.0), (1, 9, 2.0)], size=None).size is None
    with pytest.raises(ValueError, match="layers 0, 1, so it needs a size"):
        _set(size=None)


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"model_name": "Qwen3-8B"}, "not in the model registry"),
        ({"name": "Pirate"}, "lowercase letters and digits"),
        ({"name": "pirate_top"}, "lowercase letters and digits"),
        ({"name": ""}, "lowercase letters and digits"),
        ({"source": " "}, "source must say where"),
        ({"source": None}, "source must say where"),
        ({"size": 0.0}, "size must be None or a finite number above 0"),
        ({"size": -2.0}, "size must be None or a finite number above 0"),
        ({"size": math.inf}, "size must be None or a finite number above 0"),
        ({"size": True}, "size must be None or a finite number above 0"),
        ({"size": 10**400}, "size must be None or a finite number above 0"),
    ],
)
def test_bad_fields_are_refused(fields: dict, problem: str) -> None:
    size = fields.pop("size", 4.0)
    with pytest.raises(ValueError, match=problem):
        _set(size=size, **fields)


@pytest.mark.parametrize(
    ("entries", "problem"),
    [
        ([], "non-empty list"),
        ((1, 5, 1.0), "must be \\(layer, neuron, coefficient\\)"),
        ([(1, 5)], "must be \\(layer, neuron, coefficient\\)"),
        ([(1.0, 5, 1.0)], "layer must be a Python int from 0"),
        ([(True, 5, 1.0)], "layer must be a Python int from 0"),
        ([(-1, 5, 1.0)], "layer must be a Python int from 0"),
        ([(1, -5, 1.0)], "neuron must be a Python int from 0"),
        ([(1, 5, 0.0)], "coefficient must be a finite number other than 0"),
        ([(1, 5, math.nan)], "coefficient must be a finite number other than 0"),
        ([(1, 5, "1")], "coefficient must be a finite number other than 0"),
        ([(1, 5, 10**400)], "coefficient must be a finite number other than 0"),
        ([(1, 5, 1.0), (0, 3, 1.0), (1, 5, -1.0)], "entries\\[2\\] repeats layer 1"),
        ("tiny:L1:N5", "non-empty list"),
    ],
)
def test_bad_entries_are_refused(entries: object, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        _set(entries=entries)


def test_validate_checks_every_entry_against_the_shape() -> None:
    _set().validate(TINY)
    with pytest.raises(ValueError, match=r"tiny:set:x: tiny:L2:N0: layer 2 is out"):
        _set(entries=[(1, 5, 1.0), (2, 0, 1.0)], name="x").validate(TINY)
    with pytest.raises(ValueError, match="neuron 128 is out of range"):
        _set(entries=[(1, 128, 1.0)]).validate(TINY)


def test_the_unit_helpers_treat_a_neuron_as_a_set_of_one() -> None:
    unit = NeuronUnit("tiny", 1, 5)
    mix = _set()
    assert is_unit(unit) and is_unit(mix) and not is_unit("tiny:L1:N5")
    assert unit_layers(unit) == (1,) and unit_layers(mix) == (0, 1)
    assert unit_neurons(unit) == ((unit, 1.0),)
    assert unit_neurons(mix) == tuple(
        (NeuronUnit("tiny", layer, index), c) for layer, index, c in MIX
    )
    for helper in (unit_layers, unit_neurons):
        with pytest.raises(ValueError, match="a NeuronUnit or a NeuronSet"):
            helper("tiny:L1:N5")  # type: ignore[arg-type]


# --- The write direction


def test_a_sets_write_direction_is_the_hand_computed_weighted_sum(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    direction = _set().write_direction(tiny_model)
    assert direction.dtype == torch.float32 and direction.shape == (64,)
    column = {(layer, n): _column(tiny_model, layer, n).double() for layer, n, _ in MIX}
    # In float64, layer by layer as the set sums it, then rounded once: exact.
    layer_0 = -1.2 * column[0, 3]
    layer_1 = 0.8 * column[1, 5] + 0.3 * column[1, 9]
    assert torch.equal(direction, (layer_0 + layer_1).float())
    # Summed in the entries' order instead: the same up to float32 rounding.
    in_order = sum(c * column[layer, n] for layer, n, c in MIX)
    torch.testing.assert_close(direction.double(), in_order, rtol=1e-6, atol=1e-7)


def test_one_neuron_with_coefficient_1_gives_exactly_its_column(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    one = NeuronSet("tiny", "one", [(1, 5, 1.0)], "test")
    unit = NeuronUnit("tiny", 1, 5)
    assert torch.equal(
        one.write_direction(tiny_model), unit.write_direction(tiny_model)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_one_neuron_keeps_every_bit_of_its_column_signed_zeros_too(
    tiny_model: Qwen3ForCausalLM, dtype: torch.dtype
) -> None:
    model = tiny_model.to(dtype)
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[[3, 7], 5] = -0.0
    one = NeuronSet("tiny", "one", [(1, 5, 1.0)], "test")
    ours = one.write_direction(model)
    theirs = NeuronUnit("tiny", 1, 5).write_direction(model)
    assert torch.equal(ours, theirs)
    assert torch.equal(torch.signbit(ours), torch.signbit(theirs))
    assert bool(torch.signbit(ours[3])) and bool(torch.signbit(ours[7]))


def test_the_write_direction_is_a_copy_in_float32(tiny_model: Qwen3ForCausalLM) -> None:
    one = NeuronSet("tiny", "one", [(1, 5, 1.0)], "test")
    direction = one.write_direction(tiny_model)
    direction += 1.0
    assert not torch.equal(direction, _column(tiny_model, 1, 5))


def test_a_bfloat16_model_gives_its_columns_exactly(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    model = tiny_model.to(torch.bfloat16)
    mix = _set(entries=[(1, 5, 2.0), (1, 9, -0.5)], size=None)
    by_hand = 2.0 * _column(model, 1, 5).double() - 0.5 * _column(model, 1, 9).double()
    assert torch.equal(mix.write_direction(model), by_hand.float())


def test_another_models_shape_is_refused(
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
) -> None:
    with pytest.raises(ValueError, match="is it another model"):
        _set().write_direction(tiny_vl_model)  # 4 layers, not tiny's 2


def test_write_direction_needs_the_model_itself() -> None:
    with pytest.raises(ValueError, match="model must be the model itself"):
        _set().write_direction("tiny")  # type: ignore[arg-type]


def test_quantized_weights_are_refused(tiny_model: Qwen3ForCausalLM) -> None:
    layer = tiny_model.model.layers[1].mlp.down_proj
    layer.weight = torch.nn.Parameter(
        layer.weight.detach().to(torch.int8), requires_grad=False
    )
    with pytest.raises(ValueError, match="quantized models are not supported"):
        _set().write_direction(tiny_model)


# --- The dose


@pytest.mark.parametrize("alpha", [1.0, -1.0, 0.37, 2.5, 0.0, -0.0, 1e-3])
def test_one_neuron_with_coefficient_1_adds_exactly_the_neurons_amount(
    tiny_model: Qwen3ForCausalLM, alpha: float
) -> None:
    one = NeuronSet("tiny", "one", [(1, 5, 1.0)], "test")
    unit = NeuronUnit("tiny", 1, 5)
    expected = amount(unit, alpha, tiny_model, NORMS)
    got = amount(one, alpha, tiny_model, NORMS)
    assert got == expected and math.copysign(1, got) == math.copysign(1, expected)
    ((neuron, coefficient),) = unit_neurons(one)
    assert (neuron, got * coefficient) == (unit, expected)


def test_a_sized_sets_push_is_alpha_times_its_size_along_its_direction(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    mix = _set(size=4.0)
    direction = mix.write_direction(tiny_model).double()
    for alpha in (1.0, -0.5, 2.0):
        per_coefficient = amount(mix, alpha, tiny_model, NORMS)
        push = sum(
            per_coefficient * c * _column(tiny_model, layer, n).double()
            for layer, n, c in MIX
        )
        assert push.norm().item() == pytest.approx(abs(alpha) * 4.0, rel=1e-6)
        cosine = torch.nn.functional.cosine_similarity(push, direction, dim=0)
        assert cosine.item() == pytest.approx(math.copysign(1.0, alpha), abs=1e-9)


def test_a_sized_set_ignores_the_residual_sizes(tiny_model: Qwen3ForCausalLM) -> None:
    mix = _set(size=4.0)
    assert amount(mix, 1.0, tiny_model, NORMS) == amount(
        mix, 1.0, tiny_model, [1.0, 99.0]
    )
    length = float(torch.linalg.vector_norm(mix.write_direction(tiny_model)))
    assert amount(mix, 1.0, tiny_model, NORMS) == 4.0 / length


def test_a_set_without_a_size_takes_its_layers_residual_size(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    for layer, size in ((0, 7.0), (1, 10.0)):
        one_layer = _set(entries=[(layer, 5, 1.5), (layer, 9, -0.5)], size=None)
        length = float(torch.linalg.vector_norm(one_layer.write_direction(tiny_model)))
        assert amount(one_layer, 1.0, tiny_model, NORMS) == size / length


@pytest.mark.parametrize("alpha", [0.25, 1.65, -0.75])
def test_a_sets_amount_scales_linearly_with_alpha(
    tiny_model: Qwen3ForCausalLM, alpha: float
) -> None:
    mix = _set()
    assert amount(mix, alpha, tiny_model, NORMS) == alpha * amount(
        mix, 1.0, tiny_model, NORMS
    )


def test_a_set_still_needs_good_norms(tiny_model: Qwen3ForCausalLM) -> None:
    with pytest.raises(ValueError, match="the model has 2 layers"):
        amount(_set(), 1.0, tiny_model, [7.0])
    with pytest.raises(ValueError, match=r"norms\[1\] must be a finite"):
        amount(_set(), 1.0, tiny_model, [7.0, math.nan])


def test_a_dose_too_large_for_a_float_is_refused(tiny_model: Qwen3ForCausalLM) -> None:
    for unit in (_set(), NeuronUnit("tiny", 1, 5)):
        with pytest.raises(ValueError, match="alpha must be a finite number"):
            amount(unit, 10**400, tiny_model, NORMS)


def test_amount_refuses_what_is_not_a_unit(tiny_model: Qwen3ForCausalLM) -> None:
    with pytest.raises(ValueError, match="a NeuronUnit or a NeuronSet"):
        amount("tiny:L1:N5", 1.0, tiny_model, NORMS)  # type: ignore[arg-type]


def test_a_set_whose_columns_cancel_is_refused(tiny_model: Qwen3ForCausalLM) -> None:
    with torch.no_grad():
        weight = tiny_model.model.layers[1].mlp.down_proj.weight
        weight[:, 9] = weight[:, 5]
    cancels = _set(entries=[(1, 5, 1.0), (1, 9, -1.0)], size=None)
    with pytest.raises(ValueError, match="length 0.0, so no dose can scale it"):
        amount(cancels, 1.0, tiny_model, NORMS)


def test_docstring_examples_run() -> None:
    for module in (neuron_set, dose):
        result = doctest.testmod(module)
        assert result.attempted > 0
        assert result.failed == 0
