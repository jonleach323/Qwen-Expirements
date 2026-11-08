"""add_amounts, the add every push hook shares: each row's own amounts, and
nothing else, added in float32 and rounded once (decision 0008)."""

import pytest
import torch
from torch import nn

from pushlab.engine.hooks import _table_hook
from pushlab.engine.push import StepState, add_amounts
from pushlab.schedules.compile import LayerTable
from pushlab.units.neuron import NeuronUnit

VALUES = torch.linspace(-1, 1, 2 * 5 * 8).bfloat16().reshape(2, 5, 8)
NEURONS = torch.tensor([1, 6])


def _exactly(
    values: torch.Tensor, pushes: list[tuple[int, int, float]]
) -> torch.Tensor:
    """values with each (row, neuron, amount) added in float32, rounded once."""
    expected = values.clone()
    for row, neuron, value in pushes:
        exact = values[row, :, neuron].float() + torch.tensor(
            value, dtype=torch.float32
        )
        expected[row, :, neuron] = exact.to(values.dtype)
    return expected


def test_each_row_adds_its_own_amounts_never_rounded_first() -> None:
    # 22.21 is no bfloat16 number: rounded first, it would be 22.25.
    amounts = torch.tensor([[22.21, -3.3], [0.5, 7.77]], dtype=torch.float32)
    active = torch.tensor([[True, True], [True, False]])
    pushed = add_amounts(VALUES, NEURONS, amounts, active, None)
    assert torch.equal(
        pushed, _exactly(VALUES, [(0, 1, 22.21), (0, 6, -3.3), (1, 1, 0.5)])
    )
    assert torch.equal(VALUES, torch.linspace(-1, 1, 80).bfloat16().reshape(2, 5, 8))


def test_the_table_hook_adds_in_float32_and_rounds_once() -> None:
    # The path generate takes: a LayerTable's float64 amounts, never rounded
    # to bfloat16 before the add, though the activations are bfloat16.
    values = torch.linspace(-1, 1, 2001).bfloat16().reshape(1, -1, 1).repeat(2, 1, 8)
    units = (NeuronUnit("tiny", 1, 1), NeuronUnit("tiny", 1, 6))
    amounts = torch.tensor([[[22.21, -3.3], [0.5, 7.77]]], dtype=torch.float64)
    active = torch.tensor([[[True, True], [True, False]]])
    table = LayerTable(1, units, amounts, active, (True,))
    (pushed,) = _table_hook(table, StepState())(nn.Identity(), (values,))
    assert torch.equal(
        pushed, _exactly(values, [(0, 1, 22.21), (0, 6, -3.3), (1, 1, 0.5)])
    )
    rounded_first = values[0, :, 1] + torch.tensor(22.21).bfloat16()
    assert not torch.equal(pushed[0, :, 1], rounded_first)


def test_what_is_not_pushed_keeps_every_bit_negative_zeros_too() -> None:
    values = torch.full((2, 3, 4), -0.0)
    amounts = torch.zeros(2, 1)
    padding = torch.tensor([[True, False, False], [False, False, False]])
    active = torch.tensor([[True], [False]])
    pushed = add_amounts(values, torch.tensor([2]), amounts, active, padding)
    signs = torch.signbit(pushed)
    # Row 0's real positions went through the add, at 0: -0.0 + 0.0 is +0.0.
    assert not signs[0, 1:, 2].any()
    signs[0, 1:, 2] = True
    assert signs.all()  # padding, row 1 and every other neuron kept -0.0


def test_one_row_of_amounts_is_shared_by_every_row() -> None:
    amounts = torch.tensor([[0.5, -0.25]])
    pushed = add_amounts(
        VALUES, NEURONS, amounts, torch.ones(1, 2, dtype=torch.bool), None
    )
    pushes = [
        (row, neuron, value)
        for row in (0, 1)
        for neuron, value in ((1, 0.5), (6, -0.25))
    ]
    assert torch.equal(pushed, _exactly(VALUES, pushes))


ONES = torch.ones(1, 2, dtype=torch.bool)
SHAPES = {
    "values-2d": (VALUES[0], NEURONS, torch.zeros(2, 2), None, "expected"),
    "neurons-2d": (VALUES, NEURONS.reshape(2, 1), torch.zeros(2, 2), None, "one row"),
    "columns": (VALUES, NEURONS, torch.zeros(2, 1), None, r"\(rows, 2\), one per"),
    "amounts-1d": (VALUES, NEURONS, torch.zeros(2), None, r"\(rows, 2\)"),
    "rows": (VALUES, NEURONS, torch.zeros(3, 2), None, "amounts for 3 rows"),
    "mask": (VALUES, NEURONS, torch.zeros(2, 2), ONES, r"the active mask is \(1, 2\)"),
}


@pytest.mark.parametrize(
    "values, neurons, amounts, active, problem", SHAPES.values(), ids=SHAPES.keys()
)
def test_add_amounts_checks_its_shapes(
    values: torch.Tensor,
    neurons: torch.Tensor,
    amounts: torch.Tensor,
    active: torch.Tensor | None,
    problem: str,
) -> None:
    if active is None:
        active = torch.ones_like(amounts, dtype=torch.bool)
    with pytest.raises(ValueError, match=problem):
        add_amounts(values, neurons, amounts, active, None)
