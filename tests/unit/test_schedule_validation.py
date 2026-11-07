"""validate(): unknown units, windows that cannot be, the dose ceiling, and warnings."""

import doctest
import math
import random
import warnings

import pytest

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY
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
from pushlab.schedules.validation import (
    DEFAULT_CEILING,
    ScheduleWarning,
    layer_peaks,
    validate,
)
from pushlab.units.neuron import NeuronUnit

QWEN = REGISTRY["qwen3.5-4b"]
QWEN_SHAPE = ModelInfo(layers=32, hidden_size=2560, mlp_size=9216)
A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
A2 = NeuronUnit.parse("qwen3.5-4b:L14:N100")  # A's layer
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


def test_a_prompt_segment_cannot_ramp() -> None:
    segment = Segment(A, 0.0, 1.0, region="prompt")
    _refused(Schedule("x", [segment]), "a prompt segment cannot ramp")


def test_a_ramp_needs_an_end() -> None:
    _refused(Schedule("x", [Segment(A, 0.0, 1.0, 10)]), "a ramp needs an end")


def test_a_ramp_needs_two_reply_tokens() -> None:
    _refused(ramp(A, 0.0, 1.0, 10, 11), "at least two reply tokens")
    _quiet(ramp(A, 1.0, 1.0, 10, 11))  # equal doses: a one-token pulse


def test_two_pushes_of_1_in_different_layers_pass() -> None:
    _quiet(Schedule("x", constant(A, 1.0).segments + constant(B, 1.0).segments))


def test_two_pushes_of_1_in_one_layer_are_refused() -> None:
    schedule = Schedule("x", constant(A, 1.0).segments + constant(A2, 1.0).segments)
    message = _refused(schedule, "layer 14: the combined push reaches 1.414")
    assert "on the prompt pass" in message
    assert "qwen3.5-4b:L14:N8949 at +1, qwen3.5-4b:L14:N100 at +1" in message
    assert "above the ceiling of 1.4" in message
    assert "allow_over_ceiling: true" in message


def test_allow_over_ceiling_lets_it_through() -> None:
    segments = constant(A, 1.0).segments + constant(A2, 1.0).segments
    _quiet(Schedule("x", segments, allow_over_ceiling=True))


def test_the_ceiling_can_be_set() -> None:
    segments = constant(A, 1.0).segments + constant(A2, 1.0).segments
    _quiet(Schedule("x", segments), ceiling=1.5)
    _refused(constant(A, 1.0), "reaches 1 on the prompt pass", ceiling=0.9)


def test_a_push_exactly_at_the_ceiling_passes() -> None:
    _quiet(constant(A, 1.4))
    _quiet(constant(B, -1.4))
    _refused(constant(A, math.nextafter(1.4, 2.0)), "reaches 1.4000000000000001 on")


def test_overlapping_segments_of_one_unit_add_up() -> None:
    schedule = Schedule("x", [Segment(A, 1.0, 1.0, 0, 10), Segment(A, 1.0, 1.0, 5)])
    message = _refused(
        schedule, "layer 14: the combined push reaches 2 on reply token 5"
    )
    assert "qwen3.5-4b:L14:N8949 at +2" in message


def test_opposite_segments_of_one_unit_cancel() -> None:
    _quiet(Schedule("x", [Segment(A, 1.0, 1.0), Segment(A, -1.0, -1.0)]))


def test_the_prompt_pass_counts_on_its_own() -> None:
    prompt_only = [Segment(A, 1.0, 1.0, region="prompt"), Segment(A2, 1.0, 1.0, 0, 9)]
    _quiet(Schedule("x", prompt_only))
    both = Schedule("x", prompt_only + [Segment(A2, 1.0, 1.0, region="prompt")])
    _refused(both, "reaches 1.414 on the prompt pass")


def test_only_passes_where_pushes_meet_count() -> None:
    _quiet(Schedule("x", [Segment(A, 1.0, 1.0, 0, 10), Segment(A2, 1.0, 1.0, 10)]))
    meeting = Schedule("x", [Segment(A, 1.0, 1.0, 0, 11), Segment(A2, 1.0, 1.0, 10)])
    _refused(meeting, "reaches 1.414 on reply token 10")


def test_a_ramp_is_checked_on_every_token() -> None:
    # 1.2 on token 9 with 0.8 beside it: sqrt(1.44 + 0.64) = 1.442.
    late = Schedule("x", [Segment(A, 0.0, 1.2, 0, 10), Segment(A2, 0.8, 0.8, 5, 10)])
    _refused(late, "reaches 1.442 on reply token 9")
    # On tokens 0 to 4 the ramp is at most 1.2 * 4 / 9 = 0.533: 0.96 together.
    early = Schedule("x", [Segment(A, 0.0, 1.2, 0, 10), Segment(A2, 0.8, 0.8, 0, 5)])
    _quiet(early)


def test_the_verdict_does_not_depend_on_max_new_tokens() -> None:
    schedule = Schedule("x", [Segment(A, 1.0, 1.0, 300), Segment(A2, 1.0, 1.0, 300)])
    with pytest.raises(ValueError, match="reaches 1.414 on reply token 300"):
        validate(schedule, QWEN, 50)


def _random_schedule(rng: random.Random) -> Schedule:
    units = [
        NeuronUnit("qwen3.5-4b", layer, rng.randrange(9216)) for layer in (3, 3, 4)
    ]
    segments = []
    for _ in range(rng.randrange(1, 7)):
        unit = rng.choice(units)
        alpha = round(rng.uniform(-1.5, 1.5), 2)
        kind = rng.randrange(4)
        start = rng.randrange(0, 30)
        if kind == 0:
            segments.append(Segment(unit, alpha, alpha, region="prompt"))
        elif kind == 1:
            segments.append(Segment(unit, alpha, alpha, start))
        elif kind == 2:
            segments.append(
                Segment(unit, alpha, alpha, start, start + rng.randrange(1, 9))
            )
        else:
            other = round(rng.uniform(-1.5, 1.5), 2)
            end = start + rng.randrange(2, 12)
            segments.append(Segment(unit, alpha, other, start, end))
    return Schedule("x", segments)


def test_layer_peaks_match_a_scan_of_every_pass() -> None:
    rng = random.Random(20261107)
    for _ in range(400):
        schedule = _random_schedule(rng)
        scanned: dict[int, list[tuple[float, int | None]]] = {}
        for reply_token in [None, *range(50)]:
            doses = schedule.doses_at(reply_token)
            for layer in {unit.layer for unit in doses}:
                push = math.hypot(*(a for u, a in doses.items() if u.layer == layer))
                scanned.setdefault(layer, []).append((push, reply_token))
        peaks = layer_peaks(schedule)
        assert sorted(peaks) == sorted(scanned)
        for layer, passes in scanned.items():
            peak, where = peaks[layer]
            assert peak == pytest.approx(max(push for push, _ in passes), abs=1e-12)
            # The pass named reaches the peak, and no pass before it does.
            order = [p for _, p in passes]
            assert passes[order.index(where)][0] == peak
            assert all(push < peak for push, _ in passes[: order.index(where)])


def test_layer_peaks_by_hand() -> None:
    schedule = Schedule(
        "x",
        [
            Segment(A, 0.5, 0.5, region="prompt"),
            Segment(A, 0.0, 1.0, 0, 5),
            Segment(A2, -1.0, -1.0, 2, 4),
            Segment(B, -1.0, -1.0, 7),
        ],
    )
    peaks = layer_peaks(schedule)
    assert peaks[13] == (1.0, 7)
    # Token 3: A at 0.75, A2 at -1: sqrt(0.5625 + 1) = 1.25.
    assert peaks[14] == (pytest.approx(1.25), 3)
    assert layer_peaks(Schedule("none")) == {}


def test_a_segment_past_max_new_tokens_warns() -> None:
    with pytest.warns(ScheduleWarning, match="runs past max_new_tokens 200") as caught:
        found = validate(pulse(A, 1.0, 190, 20), QWEN, 200)
    assert found == [str(warning.message) for warning in caught]
    assert "reply tokens 200 to 209 never come" in found[0]


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
