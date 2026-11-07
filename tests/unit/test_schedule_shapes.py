"""Shapes: each helper builds the segments you would write by hand."""

import doctest
import inspect
import math
import random

import pytest

from pushlab.schedules import shapes
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import (
    SHAPES,
    alternate,
    constant,
    handoff,
    onset,
    pulse,
    ramp,
)
from pushlab.units.neuron import NeuronUnit

A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
B = NeuronUnit.parse("qwen3.5-4b:L13:N7956")


def _prompt(unit: NeuronUnit, alpha: float) -> Segment:
    return Segment(unit, alpha, alpha, 0, None, "prompt")


def _reply(unit: NeuronUnit, alpha: float, start: int, end: int | None) -> Segment:
    return Segment(unit, alpha, alpha, start, end, "reply")


def test_constant_runs_from_the_prompt_pass_to_the_end() -> None:
    assert constant(A, 1.0) == Schedule(
        "constant", [_prompt(A, 1.0), _reply(A, 1.0, 0, None)]
    )


def test_constant_can_leave_the_prompt_pass_out() -> None:
    assert constant(B, -1.0, include_prompt=False) == Schedule(
        "constant", [_reply(B, -1.0, 0, None)]
    )


def test_pulse_on_reply_tokens_0_to_15() -> None:
    # The research plan's pulse "on for tokens 0 to 15": start 0, end 16.
    assert pulse(A, 1.0, 0, 16) == Schedule("pulse", [_reply(A, 1.0, 0, 16)])


def test_pulse_later_in_the_reply() -> None:
    assert pulse(A, 0.5, 30, 4) == Schedule("pulse", [_reply(A, 0.5, 30, 34)])


@pytest.mark.parametrize("length", [0, -2, 1.5, True])
def test_a_pulse_lasts_at_least_one_reply_token(length: object) -> None:
    with pytest.raises(ValueError, match="length must be a whole number from 1"):
        pulse(A, 1.0, 0, length)  # type: ignore[arg-type]


def test_onset_runs_from_a_reply_token_to_the_end() -> None:
    assert onset(A, 1.0, 50) == Schedule("onset", [_reply(A, 1.0, 50, None)])


def test_handoff_at_40() -> None:
    # A on the prompt pass and reply tokens 0 to 39, then B from 40 to the end.
    assert handoff(A, B, 1.0, -1.0, 40) == Schedule(
        "handoff",
        [_prompt(A, 1.0), _reply(A, 1.0, 0, 40), _reply(B, -1.0, 40, None)],
    )


def test_handoff_with_overlap_2() -> None:
    # A stays on 2 tokens past the switch, so both are on for tokens 40 and 41.
    assert handoff(A, B, 1.0, -1.0, 40, overlap=2) == Schedule(
        "handoff",
        [_prompt(A, 1.0), _reply(A, 1.0, 0, 42), _reply(B, -1.0, 40, None)],
    )


def test_handoff_with_gap_3() -> None:
    # B comes on 3 tokens after the switch, so neither is on for tokens 40 to 42.
    assert handoff(A, B, 1.0, -1.0, 40, gap=3) == Schedule(
        "handoff",
        [_prompt(A, 1.0), _reply(A, 1.0, 0, 40), _reply(B, -1.0, 43, None)],
    )


def test_handoff_without_the_prompt_pass() -> None:
    assert handoff(A, B, 1.0, -1.0, 40, include_prompt=False) == Schedule(
        "handoff", [_reply(A, 1.0, 0, 40), _reply(B, -1.0, 40, None)]
    )


def test_handoff_at_0_pushes_a_on_the_prompt_pass_only() -> None:
    # The research plan's A6: A while reading the prompt, B while writing.
    assert handoff(A, B, 1.0, -1.0, 0) == Schedule(
        "handoff", [_prompt(A, 1.0), _reply(B, -1.0, 0, None)]
    )


def test_handoff_is_constant_a_up_to_the_switch() -> None:
    schedule, alone = handoff(A, B, 1.0, -1.0, 40), constant(A, 1.0)
    for reply_token in [None, *range(40)]:
        assert schedule.doses_at(reply_token) == alone.doses_at(reply_token)
    assert schedule.doses_at(40) == {B: -1.0}


def test_a_handoff_overlaps_or_leaves_a_gap_not_both() -> None:
    with pytest.raises(ValueError, match="overlap or a gap, not both"):
        handoff(A, B, 1.0, -1.0, 40, overlap=2, gap=3)


@pytest.mark.parametrize(
    "numbers, name",
    [
        ({"switch_at": -1}, "switch_at"),
        ({"switch_at": 4.0}, "switch_at"),
        ({"switch_at": 40, "overlap": -1}, "overlap"),
        ({"switch_at": 40, "gap": 1.5}, "gap"),
        ({"switch_at": 40, "gap": True}, "gap"),
    ],
)
def test_handoff_refuses_numbers_that_are_not_tokens(
    numbers: dict[str, object], name: str
) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a whole number from 0"):
        handoff(A, B, 1.0, -1.0, **numbers)  # type: ignore[arg-type]


@pytest.mark.parametrize("extra", [{}, {"gap": 3}])
def test_a_handoff_where_a_never_pushes_is_refused(extra: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="A would never be pushed"):
        handoff(A, B, 1.0, -1.0, 0, include_prompt=False, **extra)


def test_alternate_every_4_reply_tokens_from_token_10() -> None:
    assert alternate(A, B, 1.0, -1.0, period=4, start=10, end=26) == Schedule(
        "alternate",
        [
            _reply(A, 1.0, 10, 14),
            _reply(B, -1.0, 14, 18),
            _reply(A, 1.0, 18, 22),
            _reply(B, -1.0, 22, 26),
        ],
    )


def test_alternate_cuts_the_last_turn_short_at_its_end() -> None:
    assert alternate(A, B, 1.0, -1.0, 4, 10, 24) == Schedule(
        "alternate",
        [
            _reply(A, 1.0, 10, 14),
            _reply(B, -1.0, 14, 18),
            _reply(A, 1.0, 18, 22),
            _reply(B, -1.0, 22, 24),
        ],
    )


def test_alternate_every_reply_token() -> None:
    assert alternate(A, B, 1.0, -1.0, 1, end=4) == Schedule(
        "alternate",
        [
            _reply(A, 1.0, 0, 1),
            _reply(B, -1.0, 1, 2),
            _reply(A, 1.0, 2, 3),
            _reply(B, -1.0, 3, 4),
        ],
    )


def test_alternate_can_give_the_prompt_pass_to_a() -> None:
    assert alternate(A, B, 1.0, -1.0, 4, end=8, include_prompt=True) == Schedule(
        "alternate",
        [_prompt(A, 1.0), _reply(A, 1.0, 0, 4), _reply(B, -1.0, 4, 8)],
    )


def test_alternate_needs_an_end() -> None:
    with pytest.raises(ValueError, match="alternate needs an end"):
        alternate(A, B, 1.0, -1.0, 4)


@pytest.mark.parametrize(
    "numbers, problem",
    [
        ({"period": 0, "end": 8}, "period must be a whole number from 1"),
        ({"period": 1.5, "end": 8}, "period must be a whole number from 1"),
        ({"period": 2, "start": -1, "end": 8}, "start must be a whole number from 0"),
        ({"period": 2, "start": 8, "end": 8}, "end must be a reply token after start"),
        ({"period": 2, "start": 8, "end": 4}, "end must be a reply token after start"),
        ({"period": 2, "end": 8.0}, "end must be a reply token after start"),
    ],
)
def test_alternate_refuses_numbers_that_make_no_turns(
    numbers: dict[str, object], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        alternate(A, B, 1.0, -1.0, **numbers)  # type: ignore[arg-type]


def test_alternation_puts_exactly_one_unit_on_each_token_of_its_window() -> None:
    rng = random.Random(20261107)
    for _ in range(300):
        period, start = rng.randrange(1, 9), rng.randrange(0, 40)
        end = start + rng.randrange(1, 80)
        schedule = alternate(A, B, 1.0, -1.0, period, start, end)
        for k in range(start + end + 5):
            doses = schedule.doses_at(k)
            if not start <= k < end:
                assert doses == {}
            elif (k - start) // period % 2 == 0:
                assert doses == {A: 1.0}
            else:
                assert doses == {B: -1.0}
        assert schedule.doses_at(None) == {}


def test_ramp_is_one_segment_from_one_dose_to_another() -> None:
    assert ramp(A, 0.0, 1.0, 10, 30) == Schedule(
        "ramp", [Segment(A, 0.0, 1.0, 10, 30, "reply")]
    )


def test_shapes_take_a_name() -> None:
    assert pulse(A, 1.0, 0, 16, name="pulse16").name == "pulse16"
    with pytest.raises(ValueError, match="name must be lowercase"):
        constant(A, 1.0, name="Constant Hostile")


def test_shapes_check_their_units_and_doses() -> None:
    with pytest.raises(ValueError, match="unit must be a NeuronUnit"):
        constant("qwen3.5-4b:L14:N8949", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha_start must be a finite number"):
        onset(A, math.nan, 5)
    with pytest.raises(ValueError, match="include_prompt must be True or False"):
        constant(A, 1.0, include_prompt=1)  # type: ignore[arg-type]


def test_every_shape_is_listed_by_its_name() -> None:
    assert sorted(SHAPES) == [
        "alternate",
        "constant",
        "handoff",
        "onset",
        "pulse",
        "ramp",
    ]
    for name, build in SHAPES.items():
        assert build.__name__ == name
        assert inspect.signature(build).parameters["name"].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )


def test_doctests() -> None:
    result = doctest.testmod(shapes)
    assert result.attempted > 0
    assert result.failed == 0
