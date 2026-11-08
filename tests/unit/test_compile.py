"""compile: schedules into the exact amount every row adds on every pass."""

import doctest
import math
import warnings

import pytest
import torch
from transformers import PreTrainedModel

from pushlab.schedules import compile as compile_module
from pushlab.schedules.compile import UnitAmount, compile, pass_index
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import alternate, constant, handoff, onset, pulse, ramp
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

NORMS = [7.0, 10.0]  # tiny's two layers
A = NeuronUnit("tiny", 1, 5)
B = NeuronUnit("tiny", 1, 9)  # A's layer
C = NeuronUnit("tiny", 0, 3)
SCHEDULES = [
    Schedule("none"),
    pulse(A, 1.0, 1, 3),
    handoff(A, C, 1.0, -1.0, 2, overlap=1),
    alternate(A, B, 1.0, -0.5, 2, include_prompt=True),
    ramp(B, 0.0, 1.0, 1, 5),
    Schedule("both", [Segment(A, 1.0, 1.0, 0, 4), Segment(A, 0.25, 0.25, 2)]),
]
PASSES = [None, *range(12)]


def _by_s1s_rule(
    schedule: Schedule, reply_token: int | None, model: PreTrainedModel
) -> tuple[UnitAmount, ...]:
    """The amounts S1's dose rule gives: each segment's amount at its own dose,
    added up per unit in the schedule's order, units in order of appearance."""
    totals: dict[NeuronUnit, float] = {}
    for segment in schedule.segments:
        if segment.active_at(reply_token):
            dose = segment.alpha_at(reply_token)
            totals[segment.unit] = totals.get(segment.unit, 0.0) + amount(
                segment.unit, dose, model, NORMS
            )
    return tuple(UnitAmount(u, totals[u]) for u in schedule.units() if u in totals)


def test_every_row_gets_its_own_amounts_on_every_pass(
    tiny_model: PreTrainedModel,
) -> None:
    table = compile(SCHEDULES, tiny_model, NORMS, max_new_tokens=8)
    assert table.rows == len(SCHEDULES)
    for row, schedule in enumerate(SCHEDULES):
        for reply_token in PASSES:  # past the table's passes too
            expected = _by_s1s_rule(schedule, reply_token, tiny_model)
            assert table.amounts(row, reply_token) == expected, (row, reply_token)


def test_alpha_1_adds_what_s1_adds(tiny_model: PreTrainedModel) -> None:
    for alpha in (1.0, -1.0, 0.37, 2.5):
        table = compile([constant(A, alpha)], tiny_model, NORMS, 4)
        ((unit, added),) = table.amounts(0, None)
        assert unit == A and added == amount(A, alpha, tiny_model, NORMS)  # exactly


def test_a_unit_on_at_dose_0_is_listed(tiny_model: PreTrainedModel) -> None:
    table = compile([onset(A, 0.0, 1)], tiny_model, NORMS, 4)
    assert table.amounts(0, 0) == ()
    assert table.amounts(0, 1) == (UnitAmount(A, 0.0),)


def test_overlapping_segments_of_one_unit_add_their_amounts(
    tiny_model: PreTrainedModel,
) -> None:
    table = compile([SCHEDULES[-1]], tiny_model, NORMS, 8)
    one, quarter = (amount(A, alpha, tiny_model, NORMS) for alpha in (1.0, 0.25))
    assert [table.amounts(0, k) for k in (1, 2, 4)] == [
        (UnitAmount(A, one),),
        (UnitAmount(A, one + quarter),),
        (UnitAmount(A, quarter),),
    ]


def test_amounts_are_summed_segment_by_segment_as_s1_sums_pushes(
    tiny_model: PreTrainedModel,
) -> None:
    # Summing the doses first would round differently somewhere.
    pairs = [(0.1, 0.2), (0.3, 0.6), (0.7, 0.1), (1.1, 0.2), (0.35, 0.95)]
    one = amount(A, 1.0, tiny_model, NORMS)
    differ = False
    for x, y in pairs:
        both = Schedule("x", [Segment(A, x, x), Segment(A, y, y)])
        ((_, added),) = compile([both], tiny_model, NORMS, 4).amounts(0, 0)
        assert added == amount(A, x, tiny_model, NORMS) + amount(
            A, y, tiny_model, NORMS
        )
        differ = differ or added != (x + y) * one
    assert differ


def test_passes_are_the_prompt_pass_then_one_per_reply_token_read() -> None:
    assert [pass_index(None), pass_index(0), pass_index(7)] == [0, 1, 8]
    for wrong in (-1, True, 1.0, "0"):
        with pytest.raises(ValueError, match="None for the prompt pass"):
            pass_index(wrong)  # type: ignore[arg-type]


@pytest.mark.parametrize("row", [-1, 6, True, 1.0])
def test_amounts_names_a_row_of_the_table(
    tiny_model: PreTrainedModel, row: object
) -> None:
    table = compile(SCHEDULES, tiny_model, NORMS, max_new_tokens=8)
    with pytest.raises(ValueError, match="row must be 0 to 5"):
        table.amounts(row, 0)  # type: ignore[arg-type]


def test_the_table_holds_only_the_pushed_layers_and_neurons(
    tiny_model: PreTrainedModel,
) -> None:
    table = compile(SCHEDULES, tiny_model, NORMS, max_new_tokens=8)
    # A table of 8 passes: the prompt pass, then reply tokens 0 to 6.
    assert table.passes == 8
    assert [layer.layer for layer in table.layers] == [0, 1]
    zero, one = table.layers
    assert zero.units == (C,) and one.units == (A, B)  # by neuron index
    for layer in table.layers:
        shape = (table.passes, table.rows, len(layer.units))
        assert layer.amounts.shape == layer.active.shape == shape
        assert layer.amounts.dtype == torch.float64
        assert layer.active.dtype == torch.bool


def test_the_layer_tables_hold_the_rows_amounts(tiny_model: PreTrainedModel) -> None:
    table = compile(SCHEDULES, tiny_model, NORMS, max_new_tokens=8)
    for layer in table.layers:
        for p, reply_token in enumerate(PASSES[: table.passes]):
            for row in range(table.rows):
                pushed = dict(table.amounts(row, reply_token))
                for i, unit in enumerate(layer.units):
                    assert bool(layer.active[p, row, i]) == (unit in pushed)
                    expected = pushed.get(unit, 0.0)
                    assert float(layer.amounts[p, row, i]) == expected
            assert layer.busy[p] == bool(layer.active[p].any())


def test_a_table_of_no_pushes_has_no_layers(tiny_model: PreTrainedModel) -> None:
    table = compile([Schedule("none"), Schedule("none")], tiny_model, NORMS, 4)
    assert table.layers == () and table.rows == 2
    assert table.amounts(1, None) == ()


def test_select_takes_rows_in_the_order_given(tiny_model: PreTrainedModel) -> None:
    table = compile(SCHEDULES, tiny_model, NORMS, max_new_tokens=8)
    picked = table.select([4, 1])
    assert picked.rows == 2 and picked.schedules == (SCHEDULES[4], SCHEDULES[1])
    for reply_token in PASSES:
        assert picked.amounts(0, reply_token) == table.amounts(4, reply_token)
        assert picked.amounts(1, reply_token) == table.amounts(1, reply_token)
    for whole, part in zip(table.layers, picked.layers, strict=True):
        assert torch.equal(part.amounts, whole.amounts[:, [4, 1]])
        assert torch.equal(part.active, whole.active[:, [4, 1]])
    with pytest.raises(ValueError, match="row"):
        table.select([6])


def test_compile_leaves_the_ceiling_to_validate(tiny_model: PreTrainedModel) -> None:
    # S1's pushes had no ceiling; validate() holds schedule files to one.
    over = Schedule("over", constant(A, 2.0).segments + constant(B, 2.0).segments)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # and it never warns
        table = compile([over, pulse(A, 1.0, 10, 50)], tiny_model, NORMS, 4)
    assert dict(table.amounts(0, 0)) == {
        A: amount(A, 2.0, tiny_model, NORMS),
        B: amount(B, 2.0, tiny_model, NORMS),
    }


@pytest.mark.parametrize(
    "schedules, problem",
    [
        ([], "at least one schedule"),
        (Schedule("none"), "a list of schedules"),
        ([pulse(A, 1.0, 0, 4), "pulse"], "a list of schedules"),
        (
            [Schedule("x"), Schedule("bad", [Segment(A, 1.0, 1.0, 5, 2)])],
            "(?s)row 1: schedule 'bad' is refused:.*ends before it starts",
        ),
        ([onset(NeuronUnit("tiny", 2, 0), 1.0, 0)], "layer 2 is out of range"),
    ],
)
def test_compile_refuses_schedules_that_cannot_be(
    tiny_model: PreTrainedModel, schedules: object, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        compile(schedules, tiny_model, NORMS, 4)  # type: ignore[arg-type]


def test_compile_refuses_units_of_two_models(
    tiny_downloads: dict[str, object], tiny_model: PreTrainedModel
) -> None:
    # tiny and tiny-lm have the same shape, so only their names tell them apart.
    other = onset(NeuronUnit("tiny-lm", 1, 5), 1.0, 0)
    with pytest.raises(ValueError, match="more than one model: tiny, tiny-lm"):
        compile([pulse(A, 1.0, 0, 2), other], tiny_model, NORMS, 4)


def test_compile_needs_the_model_itself() -> None:
    with pytest.raises(ValueError, match="model must be the model itself"):
        compile([pulse(A, 1.0, 0, 2)], "tiny", NORMS, 4)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "norms, max_new_tokens, problem",
    [
        ([7.0], 4, "norms has 1 values"),
        ([7.0, math.nan], 4, r"norms\[1\] must be a finite number"),
        (NORMS, 0, "max_new_tokens must be a whole number from 1"),
        (NORMS, 2.0, "max_new_tokens must be a whole number from 1"),
    ],
)
def test_compile_checks_norms_and_the_reply_length(
    tiny_model: PreTrainedModel, norms: list[float], max_new_tokens: int, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        compile([pulse(A, 1.0, 0, 2)], tiny_model, norms, max_new_tokens)


def test_doctests() -> None:
    result = doctest.testmod(compile_module)
    assert result.attempted > 0
    assert result.failed == 0
