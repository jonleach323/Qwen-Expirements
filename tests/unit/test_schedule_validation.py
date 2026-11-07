"""validate(): unknown units, windows that cannot be, warnings and its arguments.

The dose ceiling has its own tests, in test_schedule_ceiling.py.
"""

import doctest
import math
import warnings

import pytest

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY, ModelSpec
from pushlab.schedules import validation
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import (
    alternate,
    constant,
    handoff,
    onset,
    pulse,
    ramp,
)
from pushlab.schedules.validation import DEFAULT_CEILING, ScheduleWarning, validate
from pushlab.units.neuron import NeuronUnit

QWEN = REGISTRY["qwen3.5-4b"]
QWEN_SHAPE = ModelInfo(layers=32, hidden_size=2560, mlp_size=9216)
A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
B = NeuronUnit.parse("qwen3.5-4b:L13:N7956")


def _quiet(schedule: Schedule, max_new_tokens: int = 200, **options: object) -> None:
    """Validate, and fail on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate(schedule, QWEN, max_new_tokens, **options) == []  # type: ignore[arg-type]


def _refused(schedule: Schedule, problem: str, **options: object) -> str:
    with pytest.raises(ValueError, match=problem) as error:
        validate(schedule, QWEN, 200, **options)  # type: ignore[arg-type]
    return str(error.value)


@pytest.mark.parametrize(
    "schedule",
    [
        constant(A, 1.0),
        constant(B, -1.0, include_prompt=False),
        pulse(A, 1.0, 0, 16),
        onset(A, 1.0, 50),
        handoff(A, B, 1.0, -1.0, 40, overlap=2),
        alternate(A, B, 1.0, -1.0, 4, 10, 200),
        ramp(A, 0.0, 1.0, 10, 30),
        Schedule("none"),
    ],
)
def test_the_shapes_pass(schedule: Schedule) -> None:
    _quiet(schedule)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate(schedule, QWEN_SHAPE, 200) == []


def test_the_default_ceiling_is_the_low_end_of_liu_collapse_range() -> None:
    assert DEFAULT_CEILING == 1.4


def test_a_unit_of_another_model_is_unknown() -> None:
    other = NeuronUnit.parse("qwen3-8b:L14:N8949")
    message = _refused(
        constant(other, 1.0), "unknown unit qwen3-8b:L14:N8949: it is a neuron"
    )
    assert "not of qwen3.5-4b" in message


def test_with_a_shape_alone_a_unit_of_another_shape_is_unknown() -> None:
    other = NeuronUnit.parse("qwen3-8b:L14:N8949")  # in range for Qwen3.5-4B too
    with pytest.raises(ValueError, match="unknown unit qwen3-8b:L14:N8949: qwen3-8b"):
        validate(constant(other, 1.0), QWEN_SHAPE, 200)


def test_with_a_shape_alone_units_of_two_models_are_refused() -> None:
    tiny = NeuronUnit("tiny", 1, 5)
    schedule = Schedule("x", [Segment(tiny, 1.0, 1.0), Segment(A, 1.0, 1.0)])
    with pytest.raises(ValueError, match="more than one model: qwen3.5-4b, tiny"):
        validate(schedule, ModelInfo(2, 64, 128), 200)


@pytest.mark.parametrize(
    "unit, problem",
    [
        (
            "qwen3.5-4b:L32:N0",
            "layer 32 is out of range; the model's layers are 0 to 31",
        ),
        ("qwen3.5-4b:L0:N9216", "neuron 9216 is out of range"),
    ],
)
def test_layers_and_neurons_out_of_range(unit: str, problem: str) -> None:
    _refused(onset(NeuronUnit.parse(unit), 1.0, 5), problem)
    with pytest.raises(ValueError, match=problem):
        validate(onset(NeuronUnit.parse(unit), 1.0, 5), QWEN_SHAPE, 200)


def test_the_last_layer_and_neuron_are_in_range() -> None:
    _quiet(onset(NeuronUnit.parse("qwen3.5-4b:L31:N9215"), 1.0, 5))


@pytest.mark.parametrize(
    "start, end, problem",
    [
        (10, 5, "it ends before it starts: end 5 is before start 10"),
        (1, 0, "it ends before it starts: end 0 is before start 1"),
        (10, 10, "it covers no reply token: end 10 is its start"),
    ],
)
def test_a_segment_that_ends_before_it_starts(
    start: int, end: int, problem: str
) -> None:
    _refused(Schedule("x", [Segment(A, 1.0, 1.0, start, end)]), problem)


@pytest.mark.parametrize("start, end", [(3, None), (0, 5), (0, 0)])
def test_a_prompt_segment_takes_no_start_or_end(start: int, end: int | None) -> None:
    segment = Segment(A, 1.0, 1.0, start, end, "prompt")
    _refused(Schedule("x", [segment]), "covers the whole prompt pass")


def _refused_alone(segment: Segment, problem: str) -> None:
    """validate itself refuses the segment, in its own words, with its index."""
    message = _refused(Schedule("x", [segment]), problem)
    assert message.startswith("schedule 'x' is refused:\n- segments[0] (")


@pytest.mark.parametrize(
    "segment",
    [
        Segment(A, 0.0, 1.0, region="prompt"),
        Segment(A, 1.0, 1.0, 0, None, "prompt", 2, 2),
    ],
)
def test_a_prompt_segment_cannot_ramp_or_repeat(segment: Segment) -> None:
    _refused_alone(segment, "a prompt segment cannot ramp or repeat")


def test_a_ramp_needs_an_end() -> None:
    _refused_alone(Segment(A, 0.0, 1.0, 10), "a ramp needs an end")


def test_a_ramp_needs_two_reply_tokens() -> None:
    _refused_alone(Segment(A, 0.0, 1.0, 10, 11), "at least two reply tokens")
    _quiet(ramp(A, 1.0, 1.0, 10, 11))  # equal doses: a one-token pulse


def test_a_segment_that_repeats_cannot_ramp() -> None:
    _refused_alone(
        Segment(A, 0.0, 1.0, 0, 20, "reply", 2, 2), "a segment that repeats takes one"
    )


def test_allow_over_ceiling_lifts_the_ceiling_and_nothing_else() -> None:
    schedule = Schedule(
        "bad",
        [
            Segment(NeuronUnit.parse("qwen3.5-4b:L40:N0"), 1.0, 1.0),
            Segment(A, 1.0, 1.0, 10, 5),
        ],
        allow_over_ceiling=True,
    )
    message = _refused(schedule, "schedule 'bad' is refused")
    assert "layer 40 is out of range" in message
    assert "it ends before it starts" in message
    over = Schedule("over", constant(A, 2.0).segments + pulse(B, 1.0, 190, 20).segments)
    with pytest.warns(ScheduleWarning, match="runs past max_new_tokens 200"):
        validate(Schedule("over", over.segments, allow_over_ceiling=True), QWEN, 200)


def test_a_unit_whose_model_left_the_registry_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ModelSpec("passing", "local:passing", "main", 2, 64, 128, "test")
    monkeypatch.setitem(REGISTRY, "passing", spec)
    unit = NeuronUnit("passing", 1, 5)
    monkeypatch.delitem(REGISTRY, "passing")
    with pytest.raises(ValueError, match="passing is not in the model registry"):
        validate(onset(unit, 1.0, 0), ModelInfo(2, 64, 128), 200)


def test_a_segment_past_max_new_tokens_warns() -> None:
    with pytest.warns(ScheduleWarning, match="runs past max_new_tokens 200") as caught:
        found = validate(pulse(A, 1.0, 190, 20), QWEN, 200)
    assert found == [str(warning.message) for warning in caught]
    assert "reply tokens 200 to 209 never come" in found[0]


def test_a_segment_one_token_past_max_new_tokens_warns() -> None:
    with pytest.warns(ScheduleWarning, match="so its reply token 200 never comes"):
        validate(pulse(A, 1.0, 190, 11), QWEN, 200)


def test_a_ramp_that_ends_on_the_last_reply_token_warns() -> None:
    with pytest.warns(ScheduleWarning, match="never reaches alpha_end"):
        validate(ramp(A, 0.0, 1.0, 180, 200), QWEN, 200)
    _quiet(ramp(A, 0.0, 1.0, 180, 199))


@pytest.mark.parametrize("period", [1, 2, 3, 4, 8, 16, 32])
@pytest.mark.parametrize("max_new_tokens", [200, 201, 600])
def test_an_alternation_to_the_end_of_the_reply_does_not_warn(
    period: int, max_new_tokens: int
) -> None:
    schedule = alternate(A, B, 1.0, -1.0, period, include_prompt=True)
    _quiet(schedule, max_new_tokens)


@pytest.mark.parametrize("start", [199, 200, 250])
def test_a_segment_that_no_pass_reaches_warns(start: int) -> None:
    with pytest.warns(ScheduleWarning, match="never pushes"):
        validate(onset(A, 1.0, start), QWEN, 200)


def test_a_segment_up_to_the_last_reply_token_does_not_warn() -> None:
    _quiet(pulse(A, 1.0, 0, 200))
    _quiet(onset(A, 1.0, 198))


def test_with_one_new_token_only_the_prompt_pass_pushes() -> None:
    _quiet(Schedule("x", [Segment(A, 1.0, 1.0, region="prompt")]), max_new_tokens=1)
    with pytest.warns(ScheduleWarning, match="only the prompt pass runs"):
        validate(constant(A, 1.0), QWEN, 1)


def test_every_problem_is_listed_and_no_warning_given() -> None:
    schedule = Schedule(
        "bad",
        [
            Segment(NeuronUnit.parse("qwen3.5-4b:L40:N0"), 1.0, 1.0),
            Segment(A, 1.0, 1.0, 10, 5),
            Segment(A, 0.0, 1.0, 400),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        message = _refused(schedule, "schedule 'bad' is refused")
    assert "segments[0]" in message and "layer 40 is out of range" in message
    assert "segments[1]" in message and "ends before it starts" in message
    assert "segments[2]" in message and "a ramp needs an end" in message


@pytest.mark.parametrize(
    "arguments, problem",
    [
        ({"schedule": constant(A, 1.0).segments}, "schedule must be a Schedule"),
        ({"model_info": "qwen3.5-4b"}, "model_info must be a ModelSpec"),
        ({"max_new_tokens": 0}, "max_new_tokens must be a whole number from 1"),
        ({"max_new_tokens": 1.5}, "max_new_tokens must be a whole number from 1"),
        ({"max_new_tokens": True}, "max_new_tokens must be a whole number from 1"),
        ({"ceiling": 0}, "ceiling must be a finite number above 0"),
        ({"ceiling": math.nan}, "ceiling must be a finite number above 0"),
        ({"ceiling": "1.4"}, "ceiling must be a finite number above 0"),
    ],
)
def test_validate_checks_its_arguments(
    arguments: dict[str, object], problem: str
) -> None:
    given = {
        "schedule": constant(A, 1.0),
        "model_info": QWEN,
        "max_new_tokens": 200,
    } | arguments
    with pytest.raises(ValueError, match=problem):
        validate(**given)  # type: ignore[arg-type]


def test_doctests() -> None:
    result = doctest.testmod(validation)
    assert result.attempted > 0
    assert result.failed == 0
