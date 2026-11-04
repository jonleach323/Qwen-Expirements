"""The dose rule: amount = alpha * r_L / length of the write direction."""

import doctest

import numpy as np
import pytest
import torch
from transformers import Qwen3_5ForConditionalGeneration, Qwen3ForCausalLM

from pushlab.models.registry import ModelSpec
from pushlab.units import dose
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

UNIT = NeuronUnit("tiny", layer=1, index=5)
# Layer 0's value differs, so reading the wrong layer's norm shows.
NORMS = [7.0, 10.0]


@pytest.fixture
def column_of_length_2(tiny_model: Qwen3ForCausalLM) -> Qwen3ForCausalLM:
    """tiny_model with UNIT's column set to 64 entries of 0.25: length exactly 2."""
    with torch.no_grad():
        tiny_model.model.layers[1].mlp.down_proj.weight[:, 5] = 0.25
    return tiny_model


@pytest.mark.parametrize(
    ("alpha", "expected"), [(1.0, 5.0), (0.0, 0.0), (-0.5, -2.5), (2, 10.0)]
)
def test_amount_matches_a_hand_calculation(
    column_of_length_2: Qwen3ForCausalLM, alpha: float, expected: float
) -> None:
    result = amount(UNIT, alpha, column_of_length_2, NORMS)  # alpha * 10 / 2
    assert type(result) is float
    assert result == expected


def test_amount_uses_the_norm_of_the_neurons_own_layer(
    column_of_length_2: Qwen3ForCausalLM,
) -> None:
    with torch.no_grad():
        column_of_length_2.model.layers[0].mlp.down_proj.weight[:, 5] = 0.25
    in_layer_0 = NeuronUnit("tiny", layer=0, index=5)
    assert amount(in_layer_0, 1.0, column_of_length_2, NORMS) == 3.5  # 7 / 2


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.65, 3.0, -0.75, 1e-3])
def test_amount_scales_linearly_with_alpha(
    tiny_model: Qwen3ForCausalLM, alpha: float
) -> None:
    unit = NeuronUnit("tiny", layer=1, index=77)  # its real random column
    assert amount(unit, alpha, tiny_model, NORMS) == alpha * amount(
        unit, 1.0, tiny_model, NORMS
    )


@pytest.mark.parametrize("alpha", [0.3, 1.0, 1.65])
def test_a_negative_alpha_flips_the_sign(
    tiny_model: Qwen3ForCausalLM, alpha: float
) -> None:
    pushed_up = amount(UNIT, alpha, tiny_model, NORMS)
    assert pushed_up > 0
    assert amount(UNIT, -alpha, tiny_model, NORMS) == -pushed_up


@pytest.mark.parametrize("alpha", [1.0, -0.5])
def test_the_push_is_alpha_times_the_layers_residual_size_long(
    tiny_downloads: dict[str, ModelSpec],
    tiny_model: Qwen3ForCausalLM,
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
    alpha: float,
) -> None:
    # Adding c to the neuron adds c * write_direction to the residual stream.
    for name, model, norms in [
        ("tiny", tiny_model, [7.0, 10.0]),
        ("tiny-vl", tiny_vl_model, [3.0, 5.0, 8.0, 13.0]),
    ]:
        for layer, norm in enumerate(norms):
            unit = NeuronUnit(name, layer, 100)
            push = amount(unit, alpha, model, norms) * unit.write_direction(model)
            length = float(torch.linalg.vector_norm(push))
            assert length == pytest.approx(abs(alpha) * norm, rel=1e-6)


def test_the_length_is_taken_in_float32_on_a_bfloat16_model(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    model = tiny_model.to(torch.bfloat16)
    column = model.model.layers[1].mlp.down_proj.weight[:, 5].detach()
    exact = 10.0 / float(torch.linalg.vector_norm(column.double()))
    # A length taken in bfloat16 is off by about 3e-4 here; float32 by 4e-8.
    assert amount(UNIT, 1.0, model, NORMS) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize(
    "alpha", [float("nan"), float("inf"), float("-inf"), True, "1", None]
)
def test_alpha_must_be_a_finite_number(
    tiny_model: Qwen3ForCausalLM, alpha: object
) -> None:
    with pytest.raises(ValueError, match="alpha must be a finite number"):
        amount(UNIT, alpha, tiny_model, NORMS)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "norms",
    [np.array(NORMS), torch.tensor(NORMS), {0: 7.0, 1: 10.0}, "ab", 10.0],
    ids=["ndarray", "tensor", "dict", "str", "float"],
)
def test_norms_must_be_a_list_or_tuple(
    tiny_model: Qwen3ForCausalLM, norms: object
) -> None:
    with pytest.raises(ValueError, match="norms must be a list"):
        amount(UNIT, 1.0, tiny_model, norms)  # type: ignore[arg-type]


def test_norms_may_be_a_tuple(column_of_length_2: Qwen3ForCausalLM) -> None:
    assert amount(UNIT, 1.0, column_of_length_2, tuple(NORMS)) == 5.0


@pytest.mark.parametrize("norms", [[], [10.0], [7.0, 10.0, 12.0]])
def test_norms_need_one_value_per_layer(
    tiny_model: Qwen3ForCausalLM, norms: list[float]
) -> None:
    # Three values is what output_hidden_states gives a 2-layer model.
    with pytest.raises(ValueError, match="the model has 2 layers"):
        amount(UNIT, 1.0, tiny_model, norms)


@pytest.mark.parametrize("bad", [0.0, -7.0, float("nan"), float("inf"), True, "7"])
def test_every_norm_must_be_a_finite_number_above_0(
    tiny_model: Qwen3ForCausalLM, bad: object
) -> None:
    # The bad value is in layer 0, not the pushed layer: it is still refused.
    with pytest.raises(ValueError, match=r"norms\[0\] must be a finite number"):
        amount(UNIT, 1.0, tiny_model, [bad, 10.0])  # type: ignore[list-item]


def test_a_column_of_zeros_is_refused(tiny_model: Qwen3ForCausalLM) -> None:
    with torch.no_grad():
        tiny_model.model.layers[1].mlp.down_proj.weight[:, 5] = 0.0
    with pytest.raises(ValueError, match="length 0.0"):
        amount(UNIT, 1.0, tiny_model, NORMS)


def test_amount_refuses_another_model(
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
) -> None:
    with pytest.raises(ValueError, match="another model"):
        amount(UNIT, 1.0, tiny_vl_model, [3.0, 5.0, 8.0, 13.0])


def test_amount_refuses_a_neuron_the_model_lacks(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    with pytest.raises(ValueError, match="layer 2 is out of range"):
        amount(NeuronUnit("tiny", 2, 5), 1.0, tiny_model, NORMS)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(dose)
    assert result.attempted > 0
    assert result.failed == 0
