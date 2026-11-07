"""Schedules as data: segments, their doses pass by pass, and schedules."""

import doctest
import math
import random

import pytest

from pushlab.schedules import model
from pushlab.schedules.model import (
    REGIONS,
    SCHEDULE_NAME,
    Schedule,
    Segment,
    describe_pass,
    dose_text,
)
from pushlab.store.records import EXPERIMENT_NAME
from pushlab.units.neuron import NeuronUnit

A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
B = NeuronUnit.parse("qwen3.5-4b:L13:N7956")


def test_fields_come_in_the_order_unit_alphas_window_region() -> None:
    segment = Segment(A, 0.0, 1.0, 10, 30, "reply")
    assert segment == Segment(
        unit=A, alpha_start=0.0, alpha_end=1.0, start=10, end=30, region="reply"
    )


def test_by_default_a_segment_covers_the_whole_reply() -> None:
    segment = Segment(A, 1.0, 1.0)
    assert (segment.start, segment.end, segment.region) == (0, None, "reply")
    assert (segment.on_for, segment.off_for, segment.repeats) == (None, None, False)


def test_regions_are_the_prompt_pass_and_the_reply() -> None:
    assert REGIONS == ("prompt", "reply")


def test_doses_become_floats() -> None:
    segment = Segment(A, 1, -1, 0, 4)
    assert (segment.alpha_start, segment.alpha_end) == (1.0, -1.0)
    assert type(segment.alpha_start) is float and type(segment.alpha_end) is float


@pytest.mark.parametrize("field", ["alpha_start", "alpha_end"])
@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, 10**400, True, "1.0", None]
)
def test_doses_must_be_finite_numbers(field: str, value: object) -> None:
    fields = {"alpha_start": 1.0, "alpha_end": 1.0} | {field: value}
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        Segment(A, **fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, 1.0, True, "3", None])
def test_start_must_be_a_reply_token_from_0(value: object) -> None:
    with pytest.raises(ValueError, match="start must be a reply token from 0"):
        Segment(A, 1.0, 1.0, start=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, 2.0, False, "3"])
def test_end_must_be_none_or_a_reply_token_from_0(value: object) -> None:
    with pytest.raises(ValueError, match="end must be None or a reply token from 0"):
        Segment(A, 1.0, 1.0, end=value)  # type: ignore[arg-type]


def test_how_the_fields_fit_together_is_left_to_validate() -> None:
    # validate() refuses all three; each field is fine on its own.
    Segment(A, 1.0, 1.0, start=10, end=5)
    Segment(A, 0.0, 1.0, start=10, end=None)
    Segment(A, 1.0, 1.0, start=3, end=8, region="prompt")


@pytest.mark.parametrize(
    "on_for, off_for",
    [(4, None), (None, 4), (0, 4), (4, 0), (2.0, 2), (True, 1), ("4", "4")],
)
def test_on_for_and_off_for_are_both_none_or_both_from_1(
    on_for: object, off_for: object
) -> None:
    with pytest.raises(ValueError, match="on_for and off_for must both be None"):
        Segment(A, 1.0, 1.0, on_for=on_for, off_for=off_for)  # type: ignore[arg-type]


@pytest.mark.parametrize("region", ["thinking", "answer", "Reply", "", None])
def test_other_regions_are_refused(region: object) -> None:
    with pytest.raises(ValueError, match="region must be 'prompt' or 'reply'"):
        Segment(A, 1.0, 1.0, region=region)  # type: ignore[arg-type]


@pytest.mark.parametrize("unit", ["qwen3.5-4b:L14:N8949", None, (14, 8949)])
def test_the_unit_must_be_a_neuron_unit(unit: object) -> None:
    with pytest.raises(ValueError, match="unit must be a NeuronUnit"):
        Segment(unit, 1.0, 1.0)  # type: ignore[arg-type]


def test_segments_are_frozen_values() -> None:
    segment = Segment(A, 1.0, 1.0, 0, 16)
    assert segment == Segment(A, 1, 1, 0, 16) and hash(segment) == hash(
        Segment(A, 1, 1, 0, 16)
    )
    with pytest.raises(AttributeError):
        segment.start = 3  # type: ignore[misc]


def test_a_reply_segment_is_active_from_start_to_before_end() -> None:
    segment = Segment(A, 1.0, 1.0, 3, 6)
    assert [k for k in range(10) if segment.active_at(k)] == [3, 4, 5]
    assert not segment.active_at(None)  # the prompt pass


def test_an_open_reply_segment_runs_to_the_end_of_the_reply() -> None:
    segment = Segment(A, 1.0, 1.0, 50)
    assert not segment.active_at(49)
    assert segment.active_at(50) and segment.active_at(10**6)


def test_a_prompt_segment_is_active_only_on_the_prompt_pass() -> None:
    segment = Segment(A, 1.0, 1.0, region="prompt")
    assert segment.active_at(None)
    assert not any(segment.active_at(k) for k in range(5))


def test_a_repeating_segment_is_on_for_on_for_then_off_for_off_for() -> None:
    turns = Segment(A, 1.0, 1.0, 10, 30, on_for=3, off_for=2)
    assert turns.repeats
    on = [k for k in range(40) if turns.active_at(k)]
    assert on == [10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27]
    assert not turns.active_at(None)
    open_ended = Segment(A, 1.0, 1.0, 4, on_for=1, off_for=1)
    assert [k for k in range(10) if open_ended.active_at(k)] == [4, 6, 8]
    assert open_ended.active_at(10**6) and not open_ended.active_at(10**6 + 1)


def test_a_segment_that_ends_before_it_starts_is_never_active() -> None:
    segment = Segment(A, 1.0, 1.0, 10, 5)
    assert not any(segment.active_at(k) for k in range(20))


@pytest.mark.parametrize("reply_token", [-1, 1.0, True, "0"])
def test_a_pass_is_none_or_a_reply_token_from_0(reply_token: object) -> None:
    segment = Segment(A, 1.0, 1.0)
    for method in (segment.active_at, segment.alpha_at):
        with pytest.raises(ValueError, match="None for the prompt pass"):
            method(reply_token)  # type: ignore[arg-type]


def test_a_constant_segment_gives_its_dose_on_every_pass() -> None:
    reply = Segment(A, -0.5, -0.5, 2)
    assert [reply.alpha_at(k) for k in (2, 3, 1000)] == [-0.5, -0.5, -0.5]
    assert Segment(A, 1.0, 1.0, region="prompt").alpha_at(None) == 1.0


@pytest.mark.parametrize(
    "segment, words",
    [
        (Segment(A, 1.0, 1.0, region="prompt"), "on the prompt pass"),
        (Segment(A, 1.0, 1.0, 0, 16), "on reply tokens 0 to 15"),
        (Segment(A, 1.0, 1.0, 7, 8), "on reply token 7"),
        (Segment(A, 1.0, 1.0, 40), "from reply token 40 to the end"),
        (Segment(A, 1.0, 1.0, 10, 5), "with start 10 and end 5"),
    ],
)
def test_describe_says_where_the_push_is(segment: Segment, words: str) -> None:
    assert segment.describe() == f"qwen3.5-4b:L14:N8949 at +1 {words}"


def test_describe_says_how_a_segment_repeats() -> None:
    segment = Segment(A, 1.0, 1.0, 10, 26, on_for=4, off_for=4)
    assert segment.describe() == (
        "qwen3.5-4b:L14:N8949 at +1 on reply tokens 10 to 25, 4 on then 4 off"
    )


def test_describe_shows_every_digit_a_dose_needs() -> None:
    segment = Segment(A, math.nextafter(1.4, 2.0), 1 / 3, 0, 9)
    assert segment.describe() == (
        "qwen3.5-4b:L14:N8949 from +1.4000000000000001 to +0.3333333333333333 "
        "on reply tokens 0 to 8"
    )


@pytest.mark.parametrize(
    "alpha, text",
    [(1.0, "+1"), (-1.0, "-1"), (0.0, "+0"), (-0.5, "-0.5"), (23.5, "+23.5")],
)
def test_dose_text_is_short_when_that_is_exact(alpha: float, text: str) -> None:
    assert dose_text(alpha) == text


def test_dose_text_is_never_rounded_to_another_dose() -> None:
    rng = random.Random(20261107)
    for _ in range(1000):
        alpha = rng.uniform(-2, 2)
        assert float(dose_text(alpha)) == alpha


def test_describe_pass() -> None:
    assert describe_pass(None) == "the prompt pass"
    assert describe_pass(0) == "reply token 0"


def test_describe_gives_a_ramp_both_doses() -> None:
    segment = Segment(B, 0.0, -0.5, 10, 30)
    assert segment.describe() == (
        "qwen3.5-4b:L13:N7956 from +0 to -0.5 on reply tokens 10 to 29"
    )


def test_a_schedule_is_a_name_and_a_tuple_of_segments() -> None:
    segments = [Segment(A, 1.0, 1.0, region="prompt"), Segment(A, 1.0, 1.0)]
    schedule = Schedule("constant-hostile", segments)
    assert schedule.name == "constant-hostile"
    assert schedule.segments == tuple(segments)
    assert schedule == Schedule("constant-hostile", tuple(segments))
    assert schedule.allow_over_ceiling is False
    assert hash(schedule) == hash(Schedule("constant-hostile", tuple(segments)))


def test_a_schedule_with_no_segments_pushes_nothing() -> None:
    schedule = Schedule("no-push")
    assert schedule.segments == () and schedule.units() == ()
    assert schedule.doses_at(None) == {} and schedule.doses_at(0) == {}


@pytest.mark.parametrize(
    "name",
    ["", "Pulse", "pulse_16", "-pulse", "pulse-", "pulse--16", " pulse", "pulse\n"],
)
def test_schedule_names_are_lowercase_words_joined_by_hyphens(name: str) -> None:
    with pytest.raises(ValueError, match="name must be lowercase letters and digits"):
        Schedule(name)


def test_schedule_names_take_the_form_of_experiment_names() -> None:
    # Copied rather than imported, so the schedules need not load pyarrow.
    assert SCHEDULE_NAME.pattern == EXPERIMENT_NAME.pattern


@pytest.mark.parametrize("name", [None, 3, b"pulse"])
def test_a_schedule_name_is_text(name: object) -> None:
    with pytest.raises(ValueError, match="name must be"):
        Schedule(name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "segments", [Segment(A, 1.0, 1.0), [Segment(A, 1.0, 1.0), "x"], "ab", None]
)
def test_a_schedules_segments_are_a_list_of_segments(segments: object) -> None:
    with pytest.raises(ValueError, match="segments must be a list of Segment"):
        Schedule("x", segments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, "true", None])
def test_allow_over_ceiling_is_true_or_false(value: object) -> None:
    with pytest.raises(ValueError, match="allow_over_ceiling must be True or False"):
        Schedule("x", [], value)  # type: ignore[arg-type]


def test_units_come_in_order_of_first_appearance() -> None:
    schedule = Schedule(
        "x",
        [Segment(B, -1.0, -1.0), Segment(A, 1.0, 1.0, 0, 4), Segment(B, 1.0, 1.0, 9)],
    )
    assert schedule.units() == (B, A)


def test_doses_at_adds_up_overlapping_segments_of_one_unit() -> None:
    schedule = Schedule(
        "x",
        [
            Segment(A, 1.0, 1.0, 0, 10),
            Segment(A, 0.5, 0.5, 5, 15),
            Segment(B, -1.0, -1.0, 8),
        ],
    )
    assert schedule.doses_at(None) == {}
    assert schedule.doses_at(4) == {A: 1.0}
    assert schedule.doses_at(5) == {A: 1.5}
    assert schedule.doses_at(9) == {A: 1.5, B: -1.0}
    assert schedule.doses_at(12) == {A: 0.5, B: -1.0}
    assert schedule.doses_at(20) == {B: -1.0}


def test_doses_at_lists_a_unit_on_at_dose_0() -> None:
    # A segment on at dose 0 is still on; the engine will push it with 0.
    schedule = Schedule("x", [Segment(A, 0.0, 0.0, region="prompt")])
    assert schedule.doses_at(None) == {A: 0.0}


def test_doses_at_follows_a_repeating_segment() -> None:
    schedule = Schedule(
        "x",
        [Segment(A, 1.0, 1.0, 0, on_for=2, off_for=2), Segment(B, -1.0, -1.0, 3, 6)],
    )
    assert [schedule.doses_at(k) for k in range(7)] == [
        {A: 1.0},
        {A: 1.0},
        {},
        {B: -1.0},
        {A: 1.0, B: -1.0},
        {A: 1.0, B: -1.0},
        {},
    ]


def test_doses_at_the_prompt_pass_counts_only_prompt_segments() -> None:
    schedule = Schedule(
        "x",
        [
            Segment(A, 1.0, 1.0, region="prompt"),
            Segment(A, 1.0, 1.0),
            Segment(B, 0.0, 1.0, 0, 5),
        ],
    )
    assert schedule.doses_at(None) == {A: 1.0}
    assert schedule.doses_at(0) == {A: 1.0, B: 0.0}


def test_doctests() -> None:
    result = doctest.testmod(model)
    assert result.attempted > 0
    assert result.failed == 0
