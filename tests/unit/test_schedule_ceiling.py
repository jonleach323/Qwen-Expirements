"""The dose ceiling: the combined push in one layer, on every pass, and its peaks."""

import math
import random
import time
import warnings
from fractions import Fraction

import pytest

from pushlab.models.registry import REGISTRY
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import alternate, constant
from pushlab.schedules.validation import TooManyPasses, layer_peaks, validate
from pushlab.units.neuron import NeuronUnit

QWEN = REGISTRY["qwen3.5-4b"]
A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
A2 = NeuronUnit.parse("qwen3.5-4b:L14:N100")  # A's layer
A3 = NeuronUnit.parse("qwen3.5-4b:L14:N200")  # A's layer too
B = NeuronUnit.parse("qwen3.5-4b:L13:N7956")


def _quiet(schedule: Schedule, **options: object) -> None:
    """Validate, and fail on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate(schedule, QWEN, 200, **options) == []  # type: ignore[arg-type]


def _refused(schedule: Schedule, problem: str, **options: object) -> str:
    with pytest.raises(ValueError, match=problem) as error:
        validate(schedule, QWEN, 200, **options)  # type: ignore[arg-type]
    return str(error.value)


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
    _refused(constant(A, 1.5), "above the ceiling of 1.4", ceiling=Fraction(7, 5))


def test_a_push_exactly_at_the_ceiling_passes() -> None:
    _quiet(constant(A, 1.4))
    _quiet(constant(B, -1.4))
    over = math.nextafter(1.4, 2.0)
    message = _refused(constant(A, over), "reaches 1.4000000000000001 on")
    assert "qwen3.5-4b:L14:N8949 at +1.4000000000000001" in message


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


def test_units_that_take_turns_in_one_layer_never_add_up() -> None:
    _quiet(alternate(A, A2, 1.4, -1.4, 4, include_prompt=True))


def test_a_unit_that_meets_an_alternation_is_caught_on_its_turn() -> None:
    # A3 on from token 30 meets A2's turns, 4 on then 4 off from token 4.
    turns = alternate(A, A2, 1.0, 1.0, 4).segments
    schedule = Schedule("x", [*turns, Segment(A3, 1.0, 1.0, 30)])
    message = _refused(schedule, "reaches 1.414 on reply token 30")
    assert "qwen3.5-4b:L14:N100 at +1, qwen3.5-4b:L14:N200 at +1" in message


def test_a_ramp_that_ends_at_the_ceiling_stays_under_it() -> None:
    # Two ends a hair apart: rounding inside the ramp must not pass 1.4.
    _quiet(Schedule("x", [Segment(A, math.nextafter(1.4, 0.0), 1.4, 0, 30)]))


def test_alternations_in_different_layers_are_checked_apart() -> None:
    # Periods 23 to 31 line up only every 27,913,950 tokens across layers,
    # but each layer repeats every 2 * its own period.
    segments = []
    for layer, period in zip(range(2, 7), (23, 25, 27, 29, 31), strict=True):
        a, b = NeuronUnit("qwen3.5-4b", layer, 1), NeuronUnit("qwen3.5-4b", layer, 2)
        segments += alternate(a, b, 1.0, -1.0, period).segments
    began = time.perf_counter()
    _quiet(Schedule("x", segments))
    assert time.perf_counter() - began < 2


def test_a_far_end_costs_no_more_than_an_open_one() -> None:
    began = time.perf_counter()
    _quiet(alternate(A, B, 1.0, -1.0, 1, 0, 200))
    assert layer_peaks(alternate(A, A2, 1.0, -1.0, 1, 0, 10**9)) == {14: (1.0, 0)}
    assert time.perf_counter() - began < 2


def test_repeats_in_one_layer_that_line_up_too_rarely_are_refused() -> None:
    # Cycles 202, 206 and 214 line up every 2,226,242 tokens.
    units = [NeuronUnit("qwen3.5-4b", 14, index) for index in range(6)]
    segments = []
    for (a, b), period in zip(
        [units[0:2], units[2:4], units[4:6]], (101, 103, 107), strict=True
    ):
        segments += alternate(a, b, 0.1, 0.1, period).segments
    schedule = Schedule("x", segments)
    with pytest.raises(TooManyPasses, match="every 2,226,242 reply tokens"):
        layer_peaks(schedule)
    _refused(schedule, "layer 14: its repeating segments line up again only every")


def test_a_push_too_large_for_a_float_is_refused() -> None:
    huge = [Segment(A, 1e308, 1e308), Segment(A, 1e308, 1e308)]
    _refused(Schedule("x", huge), "reaches inf on reply token 0")


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
        start = rng.randrange(0, 30)
        end = rng.choice([None, start + rng.randrange(1, 12)])
        kind = rng.randrange(4)
        if kind == 0:
            segments.append(Segment(unit, alpha, alpha, region="prompt"))
        elif kind == 1:
            segments.append(Segment(unit, alpha, alpha, start, end))
        elif kind == 2:
            on_for, off_for = rng.randrange(1, 4), rng.randrange(1, 4)
            segments.append(
                Segment(unit, alpha, alpha, start, end, "reply", on_for, off_for)
            )
        else:
            other = round(rng.uniform(-1.5, 1.5), 2)
            segments.append(
                Segment(unit, alpha, other, start, start + rng.randrange(2, 12))
            )
    return Schedule("x", segments)


def test_layer_peaks_match_a_scan_of_every_pass() -> None:
    # Every start and end is before token 42, and every cycle is 2 to 6 tokens
    # long, so the pushes repeat every lcm(2, ..., 6) = 60 tokens after it.
    rng = random.Random(20261107)
    for _ in range(400):
        schedule = _random_schedule(rng)
        scanned: dict[int, list[tuple[float, int | None]]] = {}
        for reply_token in [None, *range(120)]:
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


def test_no_ceiling_checks_none() -> None:
    # As compile does: the engine pushes what it is given (S1's pushes had no
    # ceiling); validate holds schedule files to one.
    over = Schedule("x", constant(A, 2.0).segments + constant(A2, 2.0).segments)
    _quiet(over, ceiling=None)
    _refused(Schedule("x", [Segment(A, 1.0, 1.0, 5, 2)]), "ends", ceiling=None)
