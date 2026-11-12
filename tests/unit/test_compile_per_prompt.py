"""compile with PerPromptNorms: each row's doses take its own residual sizes,
the per_prompt calibration mode (decision 0019)."""

import pytest
import torch
from transformers import PreTrainedModel

from pushlab.schedules import compile as compile_module
from pushlab.schedules.compile import DoseTable, UnitAmount, compile
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import constant, handoff, pulse, ramp
from pushlab.units.dose import PerPromptNorms, amount
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit

NORMS = [7.0, 10.0]  # tiny's two layers
A = NeuronUnit("tiny", 1, 5)
C = NeuronUnit("tiny", 0, 3)
PAIR = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "test")  # r_L's size
WIDE = NeuronSet("tiny", "wide", [(1, 9, 2.0), (0, 3, -1.5)], "test", 12.0)  # its own
SCHEDULES = [
    constant(A, 1.0),
    pulse(PAIR, -1.0, 1, 3),
    handoff(WIDE, C, 1.0, -1.0, 2, overlap=1),
    Schedule("none"),
    ramp(PAIR, 0.0, 2.0, 0, 4),
]
SIZES = PerPromptNorms([[7.0, 10.0], [3.5, 20.0], [2.0, 4.0], [9.0, 9.0], [7.0, 10.0]])
PASSES = [None, *range(10)]  # past a 6-pass table too


def _by_s1s_rule(
    schedule: Schedule, reply_token: int | None, model: PreTrainedModel, norms: list
) -> tuple[UnitAmount, ...]:
    """Each unit's amount on the pass by S1's dose rule, segment by segment."""
    totals: dict[Unit, float] = {}
    for segment in schedule.segments:
        if segment.active_at(reply_token):
            value = amount(segment.unit, segment.alpha_at(reply_token), model, norms)
            totals[segment.unit] = totals.get(segment.unit, 0.0) + value
    return tuple(UnitAmount(u, totals[u]) for u in schedule.units() if u in totals)


def _same(one: DoseTable, two: DoseTable) -> None:
    """The two tables hold the same numbers, bit for bit."""
    assert one.scales == two.scales and one.rows == two.rows
    for row in range(one.rows):
        for reply_token in PASSES:
            x, y = one.amounts(row, reply_token), two.amounts(row, reply_token)
            assert x == y and [v.hex() for _, v in x] == [v.hex() for _, v in y]
    assert len(one.layers) == len(two.layers)
    for x, y in zip(one.layers, two.layers, strict=True):
        assert (x.layer, x.units, x.busy) == (y.layer, y.units, y.busy)
        assert torch.equal(x.amounts.view(torch.int64), y.amounts.view(torch.int64))
        assert torch.equal(x.active, y.active)


def test_each_row_takes_its_own_sizes(tiny_model: PreTrainedModel) -> None:
    table = compile(SCHEDULES, tiny_model, SIZES, 6)
    for row, (schedule, sizes) in enumerate(zip(SCHEDULES, SIZES.rows, strict=True)):
        alone = compile([schedule], tiny_model, list(sizes), 6)
        for reply_token in PASSES:
            expected = _by_s1s_rule(schedule, reply_token, tiny_model, list(sizes))
            assert table.amounts(row, reply_token) == expected, (row, reply_token)
        for part in alone.layers:  # each neuron's column, as the row alone has it
            (whole,) = [layer for layer in table.layers if layer.layer == part.layer]
            for i, neuron in enumerate(part.units):
                j = whole.units.index(neuron)
                assert torch.equal(whole.amounts[:, row, j], part.amounts[:, 0, i])
                assert torch.equal(whole.active[:, row, j], part.active[:, 0, i])


def test_a_neurons_and_an_unsized_sets_amounts_scale_with_the_rows_size(
    tiny_model: PreTrainedModel,
) -> None:
    sizes = PerPromptNorms([[7.0, 10.0], [3.5, 20.0]])
    for unit in (A, PAIR):
        table = compile([constant(unit, 1.0)] * 2, tiny_model, sizes, 4)
        ((_, low),), ((_, high),) = table.amounts(0, 0), table.amounts(1, 0)
        assert high == 2 * low  # layer 1's size doubles
    # A set with a size of its own doses the same whatever the rows' sizes.
    table = compile([constant(WIDE, 1.0)] * 2, tiny_model, sizes, 4)
    assert table.amounts(0, 0) == table.amounts(1, 0)
    assert table.amounts(0, 0) == compile(
        [constant(WIDE, 1.0)], tiny_model, NORMS, 4
    ).amounts(0, 0)


def test_rows_all_equal_to_the_fixed_sizes_give_the_fixed_table(
    tiny_model: PreTrainedModel,
) -> None:
    fixed = compile(SCHEDULES, tiny_model, NORMS, 6)
    _same(fixed, compile(SCHEDULES, tiny_model, PerPromptNorms([NORMS] * 5), 6))
    _same(fixed, compile(SCHEDULES, tiny_model, tuple(NORMS), 6))  # as norms_for


@pytest.mark.parametrize("count", [1, 3])
def test_per_prompt_sizes_need_one_row_per_schedule(
    tiny_model: PreTrainedModel, count: int
) -> None:
    sizes = PerPromptNorms([NORMS] * count)
    problem = f"norms has sizes for {count} prompts, but there are 2 schedules"
    with pytest.raises(ValueError, match=problem):
        compile([constant(A, 1.0), Schedule("none")], tiny_model, sizes, 4)


def test_per_prompt_sizes_need_one_size_per_layer(tiny_model: PreTrainedModel) -> None:
    sizes = PerPromptNorms([[7.0, 10.0, 3.0]] * 2)
    with pytest.raises(ValueError, match="norms has 3 values, but the model has 2"):
        compile([constant(A, 1.0), Schedule("none")], tiny_model, sizes, 4)


def test_each_unit_is_worked_once_per_distinct_row_of_sizes(
    tiny_model: PreTrainedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    worked: list[tuple[Unit, tuple[float, ...]]] = []

    def counted(unit: Unit, alpha: float, model: object, norms: list) -> float:
        worked.append((unit, tuple(norms)))
        return amount(unit, alpha, model, norms)  # type: ignore[arg-type]

    monkeypatch.setattr(compile_module, "amount", counted)
    both = Schedule("both", constant(A, 1.0).segments + constant(PAIR, 1.0).segments)
    sizes = PerPromptNorms([[7.0, 10.0], [3.5, 20.0], [7.0, 10.0], [3.5, 20.0]])
    compile([both] * 4, tiny_model, sizes, 4)
    assert len(worked) == len(set(worked)) == 4  # two units, two distinct rows
    worked.clear()
    compile([both] * 4, tiny_model, NORMS, 4)
    assert len(worked) == 2  # shared sizes: once per unit


def test_select_keeps_each_rows_own_sizes(tiny_model: PreTrainedModel) -> None:
    sizes = PerPromptNorms([[7.0, 10.0], [3.5, 20.0]])
    table = compile([constant(A, 1.0), constant(PAIR, 1.0)], tiny_model, sizes, 4)
    picked = table.select([1, 0])
    assert picked.scales == (table.scales[1], table.scales[0])
    for reply_token in (None, 0, 2, 50):  # past the table, from the scales
        assert picked.amounts(0, reply_token) == table.amounts(1, reply_token)
        assert picked.amounts(1, reply_token) == table.amounts(0, reply_token)
    assert table.scales[0] == {A: amount(A, 1.0, tiny_model, [7.0, 10.0])}
    assert table.scales[1] == {PAIR: amount(PAIR, 1.0, tiny_model, [3.5, 20.0])}
