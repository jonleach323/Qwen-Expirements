"""layer_tables: units' amounts spread over the neurons they push, layer by layer.

Each test hands layer_tables the per-unit amounts compile would, by hand, so
the numbers are exact: a neuron's amount is its unit's amount times its
coefficient, and parts on one neuron add in each row's own order (decision
0019).
"""

import doctest
import math

import torch

from pushlab.schedules import layer_tables as layer_tables_module
from pushlab.schedules.layer_tables import LayerTable, layer_tables
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

A = NeuronUnit("tiny", 1, 5)
B = NeuronUnit("tiny", 1, 9)
C = NeuronUnit("tiny", 0, 3)
FIRST = NeuronSet("tiny", "first", [(1, 5, 1.0)], "test")
SECOND = NeuronSet("tiny", "second", [(1, 5, 1.0), (1, 9, -0.5)], "test")
WIDE = NeuronSet("tiny", "wide", [(1, 9, 2.0), (0, 7, 0.5), (0, 3, -1.5)], "x", 3.0)


def _column(table: LayerTable, neuron: NeuronUnit, row: int) -> list[float]:
    """The neuron's amounts on every pass, in row."""
    return table.amounts[:, row, table.units.index(neuron)].tolist()


def test_each_neuron_of_a_set_gets_the_amount_times_its_coefficient() -> None:
    by_pass = [[((WIDE, 4.0),), (), ((WIDE, -1.0),)]]  # one row, three passes
    zero, one = layer_tables(by_pass, [(WIDE,)])
    assert (zero.layer, one.layer) == (0, 1)
    assert zero.units == (C, NeuronUnit("tiny", 0, 7))  # by index
    assert one.units == (B,)
    assert _column(zero, C, 0) == [-6.0, 0.0, 1.5]
    assert _column(zero, NeuronUnit("tiny", 0, 7), 0) == [2.0, 0.0, -0.5]
    assert _column(one, B, 0) == [8.0, 0.0, -2.0]
    for table in (zero, one):
        assert table.active[:, 0].all(dim=1).tolist() == [True, False, True]
        assert not table.active[1].any() and table.busy == (True, False, True)
        assert table.amounts.dtype == torch.float64


def test_a_neuron_one_unit_pushes_keeps_its_part_bit_for_bit() -> None:
    # Set, not added to 0.0: -0.0 stays -0.0, as 0.0 + -0.0 would not.
    by_pass = [[((A, -0.0),), ((SECOND, 0.0),), ((A, 2.5),)]]
    (one,) = layer_tables(by_pass, [(A, SECOND)])
    assert one.units == (A, B)
    signs = [math.copysign(1.0, x) for x in _column(one, A, 0)]
    assert _column(one, A, 0) == [0.0, 0.0, 2.5] and signs == [-1.0, 1.0, 1.0]
    # B's coefficient is -0.5, so dose 0 gives -0.0; pushed at 0 is still pushed.
    assert math.copysign(1.0, _column(one, B, 0)[1]) == -1.0
    assert one.active[:, 0].tolist() == [[True, False], [True, True], [True, False]]


def test_parts_on_one_neuron_add_in_each_rows_own_order() -> None:
    # 1 + 1e-16 rounds to 1, but -1 + 1e-16 does not round to -1.
    parts = {FIRST: 1.0, SECOND: 1e-16, A: -1.0}
    forward, backward = (FIRST, SECOND, A), (A, SECOND, FIRST)
    by_pass = [
        [tuple((unit, parts[unit]) for unit in row)] for row in (forward, backward)
    ]
    (one,) = layer_tables(by_pass, [forward, backward])
    assert _column(one, A, 0) == [(1.0 + 1e-16) + -1.0] == [0.0]
    assert _column(one, A, 1) == [(-1.0 + 1e-16) + 1.0] == [2.0**-53]
    # Each row alone gives the same numbers: no row depends on another.
    for row, units in enumerate((forward, backward)):
        (alone,) = layer_tables([by_pass[row]], [units])
        assert torch.equal(alone.amounts[:, 0], one.amounts[:, row])
        assert torch.equal(alone.active[:, 0], one.active[:, row])


def test_the_order_is_the_schedules_not_the_order_units_first_push() -> None:
    # The schedule names A first, though FIRST pushes on an earlier pass:
    # FIRST first would give (1.0 + -1.0) + 1e-16, which is 1e-16.
    by_pass = [[((FIRST, 1.0),), ((A, -1.0), (SECOND, 1e-16), (FIRST, 1.0))]]
    (one,) = layer_tables(by_pass, [(A, SECOND, FIRST)])
    assert _column(one, A, 0) == [1.0, (-1.0 + 1e-16) + 1.0] == [1.0, 2.0**-53]


def test_every_neuron_of_every_unit_has_a_column_pushed_or_not() -> None:
    # Row 1's units push nothing on these passes, yet their neurons are listed.
    by_pass = [[((A, 1.0),), ()], [(), ()]]
    tables = layer_tables(by_pass, [(A,), (WIDE, B)])
    assert [table.layer for table in tables] == [0, 1]
    zero, one = tables
    assert zero.units == (C, NeuronUnit("tiny", 0, 7)) and one.units == (A, B)
    assert not zero.active.any() and zero.busy == (False, False)
    assert one.active[:, :, 0].tolist() == [[True, False], [False, False]]
    assert one.busy == (True, False)
    assert layer_tables([[(), ()]], [()]) == ()  # no units, no tables


def test_select_keeps_the_neurons_and_takes_the_rows_given() -> None:
    by_pass = [[((A, 1.0),)], [((SECOND, 2.0),)], [()]]
    (one,) = layer_tables(by_pass, [(A,), (SECOND,), ()])
    picked = one.select([2, 1])
    assert picked.units == (A, B) and picked.busy == (True,)
    assert picked.amounts[0].tolist() == [[0.0, 0.0], [2.0, -1.0]]
    assert picked.active[0].tolist() == [[False, False], [True, True]]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(layer_tables_module)
    assert result.attempted > 0
    assert result.failed == 0
