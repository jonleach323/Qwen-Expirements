"""compile with NeuronSets: each set's neurons in every layer it touches, and
units that share a neuron added in each row's own order (decision 0019)."""

import math
import warnings
from collections.abc import Callable

import pytest
import torch
from transformers import PreTrainedModel

from pushlab.models.discovery import find_mlp_modules
from pushlab.schedules.compile import DoseTable, UnitAmount, compile, pass_index
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import alternate, constant, handoff, onset, pulse, ramp
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit, unit_neurons

NORMS = [7.0, 10.0]  # tiny's two layers
A = NeuronUnit("tiny", 1, 5)
B = NeuronUnit("tiny", 1, 9)  # A's layer
C = NeuronUnit("tiny", 0, 3)
ONE = NeuronSet("tiny", "one", [(1, 5, 1.0)], "test")  # A, as a set
PAIR = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "test")
WIDE = NeuronSet(  # across both layers, so it has a size
    "tiny", "wide", [(1, 9, 2.0), (0, 3, -1.5), (1, 5, 0.75), (0, 7, 0.5)], "x", 12.0
)
PASSES = [None, *range(12)]  # past an 8-pass table too

# Shapes that push the unit given, some with other units beside it.
SHAPES: list[Callable[[Unit], Schedule]] = [
    lambda unit: constant(unit, 1.0),
    lambda unit: pulse(unit, -1.5, 1, 3),
    lambda unit: ramp(unit, 0.0, 1.0, 1, 5),
    lambda unit: alternate(unit, C, 1.0, -0.5, 2, include_prompt=True),
    lambda unit: handoff(C, unit, 1.0, -1.0, 2, overlap=1),
    lambda unit: Schedule(
        "both",
        [
            Segment(unit, 1.0, 1.0, 0, 4),
            Segment(B, 0.5, 0.5, 1),
            Segment(unit, 0.25, 0.25, 2),
        ],
    ),
]


def _hex(amounts: tuple[UnitAmount, ...]) -> list[str]:
    """Each amount's exact bits, so that -0.0 differs from 0.0."""
    return [value.hex() for _, value in amounts]


def _by_s1s_rule(
    schedule: Schedule, reply_token: int | None, model: PreTrainedModel, norms: list
) -> tuple[UnitAmount, ...]:
    """Each unit's amount on the pass by S1's dose rule, segment by segment,
    units in the schedule's order; a set's per unit of coefficient."""
    totals: dict[Unit, float] = {}
    for segment in schedule.segments:
        if segment.active_at(reply_token):
            value = amount(segment.unit, segment.alpha_at(reply_token), model, norms)
            totals[segment.unit] = totals.get(segment.unit, 0.0) + value
    return tuple(UnitAmount(u, totals[u]) for u in schedule.units() if u in totals)


def _by_neuron(pushed: tuple[UnitAmount, ...]) -> dict[NeuronUnit, float]:
    """Each neuron's amount: its units' amounts times its coefficients, the
    first set and the rest added, in the order the units come."""
    totals: dict[NeuronUnit, float] = {}
    for unit, value in pushed:
        for neuron, coefficient in unit_neurons(unit):
            part = value * coefficient
            totals[neuron] = totals[neuron] + part if neuron in totals else part
    return totals


def _check_layers(table: DoseTable) -> None:
    """Every layer table holds, bit for bit, each row's neuron amounts."""
    for layer in table.layers:
        for p, reply_token in enumerate(PASSES[: table.passes]):
            for row in range(table.rows):
                expected = _by_neuron(table.amounts(row, reply_token))
                for i, neuron in enumerate(layer.units):
                    assert bool(layer.active[p, row, i]) == (neuron in expected)
                    got = float(layer.amounts[p, row, i])
                    assert got.hex() == expected.get(neuron, 0.0).hex(), (p, row, i)
            assert layer.busy[p] == bool(layer.active[p].any())


@pytest.mark.parametrize("shape", range(len(SHAPES)))
def test_a_one_neuron_set_compiles_exactly_as_its_neuron(
    tiny_model: PreTrainedModel, shape: int
) -> None:
    build = SHAPES[shape]
    as_neuron = compile([build(A), Schedule("none")], tiny_model, NORMS, 8)
    as_set = compile([build(ONE), Schedule("none")], tiny_model, NORMS, 8)
    assert len(as_neuron.layers) == len(as_set.layers)
    for x, y in zip(as_neuron.layers, as_set.layers, strict=True):
        assert (x.layer, x.units, x.busy) == (y.layer, y.units, y.busy)
        assert torch.equal(x.amounts.view(torch.int64), y.amounts.view(torch.int64))
        assert torch.equal(x.active, y.active)
    for row in range(2):
        for reply_token in PASSES:
            neuron = as_neuron.amounts(row, reply_token)
            whole = as_set.amounts(row, reply_token)
            assert [ONE if u == A else u for u, _ in neuron] == [u for u, _ in whole]
            assert _hex(neuron) == _hex(whole)


def test_a_sets_amounts_follow_s1s_rule_and_its_neurons_share_them(
    tiny_model: PreTrainedModel,
) -> None:
    schedules = [
        *(build(unit) for build in SHAPES for unit in (PAIR, WIDE)),
        Schedule("none"),
        Schedule("mix", [*constant(A, 0.5).segments, *pulse(WIDE, 1.0, 2, 3).segments]),
    ]
    table = compile(schedules, tiny_model, NORMS, max_new_tokens=8)
    for row, schedule in enumerate(schedules):
        for reply_token in PASSES:
            expected = _by_s1s_rule(schedule, reply_token, tiny_model, NORMS)
            assert table.amounts(row, reply_token) == expected, (row, reply_token)
    _check_layers(table)


def test_a_set_across_layers_pushes_alpha_times_its_size_along_its_direction(
    tiny_model: PreTrainedModel,
) -> None:
    table = compile([constant(WIDE, -0.5), Schedule("none")], tiny_model, NORMS, 4)
    per = amount(WIDE, -0.5, tiny_model, NORMS)
    assert table.amounts(0, None) == (UnitAmount(WIDE, per),)
    zero, one = table.layers
    assert zero.units == (C, NeuronUnit("tiny", 0, 7)) and one.units == (A, B)
    coefficients = dict(unit_neurons(WIDE))
    mlps = dict(find_mlp_modules(tiny_model))
    push = torch.zeros(64, dtype=torch.float64)
    for layer in (zero, one):
        assert layer.active[:, 0].all() and not layer.active[:, 1].any()
        assert not layer.amounts[:, 1].any()
        for i, neuron in enumerate(layer.units):
            assert layer.amounts[:, 0, i].tolist() == [per * coefficients[neuron]] * 4
            column = (
                mlps[neuron.layer].down_proj.weight.detach()[:, neuron.index].double()
            )
            push += float(layer.amounts[0, 0, i]) * column
    # The parts make one push along w, of length |alpha| * size = 0.5 * 12.
    direction = WIDE.write_direction(tiny_model).double()
    assert math.isclose(float(push.norm()), 6.0, rel_tol=1e-6)
    assert math.isclose(
        float(push @ direction), -6.0 * float(direction.norm()), rel_tol=1e-6
    )


def test_a_set_and_a_neuron_on_one_neuron_add(tiny_model: PreTrainedModel) -> None:
    both = Schedule(
        "x", [*pulse(PAIR, 1.0, 0, 4).segments, *pulse(A, -0.5, 2, 4).segments]
    )
    table = compile([both], tiny_model, NORMS, 8)
    pair, a = amount(PAIR, 1.0, tiny_model, NORMS), amount(A, -0.5, tiny_model, NORMS)
    (one,) = table.layers
    assert one.units == (A, B)
    for p, reply_token in enumerate(PASSES[:8]):
        on_pair = reply_token is not None and reply_token < 4
        on_a = reply_token is not None and 2 <= reply_token < 6
        fifth = {(True, True): pair + a, (True, False): pair, (False, True): a}
        assert float(one.amounts[p, 0, 0]) == fifth.get((on_pair, on_a), 0.0)
        assert bool(one.active[p, 0, 0]) == (on_pair or on_a)
        assert float(one.amounts[p, 0, 1]) == (pair * -0.5 if on_pair else 0.0)


def test_two_sets_on_one_neuron_add_in_each_rows_own_order(
    tiny_model: PreTrainedModel,
) -> None:
    first = NeuronSet("tiny", "first", [(1, 5, 1.0), (1, 9, 1.0)], "test")
    second = NeuronSet("tiny", "second", [(0, 3, 1.0), (1, 5, -2.0)], "test", 4.0)
    forward = Schedule(
        "forward",
        [*constant(first, 1.0).segments, *constant(second, 1.0).segments]
        + list(constant(A, -0.5).segments),
    )
    backward = Schedule("backward", list(reversed(forward.segments)))
    table = compile([forward, backward], tiny_model, NORMS, 4)
    f = amount(first, 1.0, tiny_model, NORMS)
    s = amount(second, 1.0, tiny_model, NORMS) * -2.0
    a = amount(A, -0.5, tiny_model, NORMS)
    one = table.layers[1]
    assert one.units == (A, B)
    assert float(one.amounts[0, 0, 0]) == (f + s) + a  # first, second, then A
    assert float(one.amounts[0, 1, 0]) == (a + s) + f  # A, second, then first
    assert (f + s) + a != (a + s) + f  # so the order shows
    assert float(one.amounts[0, 0, 1]) == float(one.amounts[0, 1, 1]) == f
    for row, schedule in enumerate((forward, backward)):  # alone, the same
        alone = compile([schedule], tiny_model, NORMS, 4)
        for whole, part in zip(table.layers, alone.layers, strict=True):
            assert torch.equal(whole.amounts[:, row], part.amounts[:, 0])
    _check_layers(table)


def test_a_set_at_a_negative_dose_and_along_a_ramp(tiny_model: PreTrainedModel) -> None:
    schedules = [constant(PAIR, -1.0), ramp(PAIR, 0.0, 1.0, 1, 5)]
    table = compile(schedules, tiny_model, NORMS, 8)
    scale = amount(PAIR, 1.0, tiny_model, NORMS)
    (one,) = table.layers
    up = schedules[1].segments[0]
    for p, reply_token in enumerate(PASSES[:8]):
        assert one.amounts[p, 0].tolist() == [-scale, 0.5 * scale]
        if reply_token is not None and 1 <= reply_token < 5:
            per = up.alpha_at(reply_token) * scale
            assert one.amounts[p, 1].tolist() == [per, per * -0.5]
            assert one.active[p, 1].all()
        else:
            assert not one.active[p, 1].any()
    # At the ramp's dose 0 both neurons are pushed, B by -0.0 (-0.5 times 0).
    start = one.amounts[pass_index(1), 1].tolist()
    assert [math.copysign(1.0, x) for x in start] == [1.0, -1.0]
    _check_layers(table)


def test_a_set_at_dose_0_is_listed_and_its_neurons_are_active(
    tiny_model: PreTrainedModel,
) -> None:
    table = compile([onset(WIDE, 0.0, 1)], tiny_model, NORMS, 4)
    assert table.amounts(0, 0) == () and table.amounts(0, 1) == (UnitAmount(WIDE, 0.0),)
    for layer in table.layers:
        assert layer.active[:, 0].tolist() == [
            [False] * 2,
            [False] * 2,
            [True] * 2,
            [True] * 2,
        ]
        assert layer.busy == (False, False, True, True)


def test_compile_leaves_the_ceiling_to_validate_for_sets_too(
    tiny_model: PreTrainedModel,
) -> None:
    over = Schedule("over", constant(PAIR, 1.0).segments + constant(A, 1.0).segments)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # and it never warns
        table = compile([over, pulse(WIDE, 2.0, 10, 50)], tiny_model, NORMS, 4)
    assert dict(table.amounts(0, 0)) == {
        PAIR: amount(PAIR, 1.0, tiny_model, NORMS),
        A: amount(A, 1.0, tiny_model, NORMS),
    }


@pytest.mark.parametrize(
    "schedules, problem",
    [
        (
            [constant(NeuronSet("tiny", "deep", [(2, 0, 1.0)], "x", 1.0), 1.0)],
            "(?s)row 0: .*layer 2 is out of range",
        ),
        (
            [
                Schedule("x"),
                constant(NeuronSet("tiny", "w", [(1, 128, 1.0)], "x"), 1.0),
            ],
            "(?s)row 1: .*neuron 128 is out of range",
        ),
        (
            [Schedule("x"), Schedule("bad", [Segment(PAIR, 1.0, 1.0, 5, 2)])],
            "(?s)row 1: schedule 'bad' is refused:.*ends before it starts",
        ),
    ],
)
def test_compile_refuses_sets_the_model_lacks(
    tiny_model: PreTrainedModel, schedules: list[Schedule], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        compile(schedules, tiny_model, NORMS, 4)


def test_compile_refuses_a_set_of_another_model(
    tiny_downloads: dict[str, object], tiny_model: PreTrainedModel
) -> None:
    other = constant(NeuronSet("tiny-lm", "pair", [(1, 5, 1.0)], "test"), 1.0)
    with pytest.raises(ValueError, match="more than one model: tiny, tiny-lm"):
        compile([constant(PAIR, 1.0), other], tiny_model, NORMS, 4)


def test_select_keeps_each_rows_set_columns(tiny_model: PreTrainedModel) -> None:
    schedules = [constant(PAIR, 1.0), Schedule("none"), handoff(WIDE, A, 1.0, -1.0, 2)]
    table = compile(schedules, tiny_model, NORMS, 6)
    picked = table.select([2, 0])
    for reply_token in PASSES:
        assert picked.amounts(0, reply_token) == table.amounts(2, reply_token)
        assert picked.amounts(1, reply_token) == table.amounts(0, reply_token)
    for whole, part in zip(table.layers, picked.layers, strict=True):
        assert part.units == whole.units
        assert torch.equal(part.amounts, whole.amounts[:, [2, 0]])
        assert torch.equal(part.active, whole.active[:, [2, 0]])
