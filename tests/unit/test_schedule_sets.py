"""Schedules that push NeuronSets: validate and the per-layer ceiling
(decision 0019). The YAML layer's refusals of sets are tested in
test_schedule_yaml_sets.py.

A set pushes every layer it has a neuron in, each with its whole dose, so it
counts in each of those layers' combined push.
"""

import math
import random
import time
import warnings
from dataclasses import replace

import pytest

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY, ModelSpec
from pushlab.schedules import validation
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import alternate, constant, handoff, onset, pulse, ramp
from pushlab.schedules.validation import ScheduleWarning, layer_peaks, validate
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit, unit_layers

TINY = REGISTRY["tiny"]  # 2 layers, hidden size 64, 128 MLP neurons
TINY_SHAPE = ModelInfo(layers=2, hidden_size=64, mlp_size=128)
QWEN3_8B = REGISTRY["qwen3-8b"]  # 36 layers, 12,288 MLP neurons
A = NeuronUnit("tiny", 1, 5)
B = NeuronUnit("tiny", 1, 9)  # A's layer
C = NeuronUnit("tiny", 0, 3)
ONE = NeuronSet("tiny", "one", [(1, 5, 1.0)], "test")  # A, as a set
PAIR = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "test")  # layer 1
LOW = NeuronSet("tiny", "low", [(0, 3, 1.0), (0, 7, 0.5)], "test")  # layer 0
WIDE = NeuronSet(  # layers 0 and 1, so it has a size
    "tiny", "wide", [(1, 9, 2.0), (0, 3, -1.5), (1, 5, 0.75)], "test", 12.0
)
PIRATE = NeuronSet("qwen3-8b", "pirate", [(14, 100, 1.0)], "test", 30.0)


def _quiet(schedule: Schedule, model_info: object = TINY, **options: object) -> None:
    """Validate, and fail on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate(schedule, model_info, 200, **options) == []  # type: ignore[arg-type]


def _refused(
    schedule: Schedule, model_info: object = TINY, **options: object
) -> list[str]:
    """validate's problems, as it lists them below its first line."""
    with pytest.raises(ValueError) as error:
        validate(schedule, model_info, 200, **options)  # type: ignore[arg-type]
    first, *problems = str(error.value).split("\n")
    assert first == f"schedule {schedule.name!r} is refused:"
    return [problem.removeprefix("- ") for problem in problems]


def _over(layer: int, reached: str, where: str, units: str) -> str:
    """The ceiling's refusal of one layer, word for word."""
    return (
        f"layer {layer}: the combined push reaches {reached} on {where} ({units}), "
        "above the ceiling of 1.4; set allow_over_ceiling: true to allow it"
    )


# --- validate takes sets


@pytest.mark.parametrize(
    "schedule",
    [
        constant(WIDE, 1.0),
        constant(PAIR, -1.0, include_prompt=False),
        pulse(WIDE, 1.4, 0, 16),
        onset(LOW, 1.0, 50),
        ramp(WIDE, 0.0, 1.0, 10, 30),
        alternate(WIDE, PAIR, 1.0, -1.0, 4, 10, 200),
        alternate(PAIR, A, 1.4, -1.4, 4, include_prompt=True),  # turns: never both
        handoff(WIDE, LOW, 1.0, -1.0, 40),
        handoff(PAIR, C, 1.0, 1.0, 40, overlap=2),  # in different layers
        handoff(LOW, WIDE, 0.8, 1.0, 40, overlap=2),  # 1.28 in layer 0 at the switch
    ],
)
def test_every_shape_takes_sets(schedule: Schedule) -> None:
    _quiet(schedule)
    _quiet(schedule, TINY_SHAPE)


def test_a_qwen3_8b_set_is_checked_against_its_36_layers_and_12288_neurons() -> None:
    edges = NeuronSet("qwen3-8b", "edges", [(0, 0, 1.0), (35, 12287, -1.0)], "x", 9.0)
    _quiet(constant(edges, 1.0), QWEN3_8B)
    past = NeuronSet("qwen3-8b", "past", [(36, 0, 1.0), (35, 12288, 1.0)], "x", 9.0)
    problems = _refused(onset(past, 1.0, 5), QWEN3_8B)
    assert len(problems) == 2
    assert problems[0].endswith(
        "qwen3-8b:set:past: qwen3-8b:L36:N0: layer 36 is out of range; the model's "
        "layers are 0 to 35"
    )
    assert problems[1].endswith(
        "qwen3-8b:set:past: qwen3-8b:L35:N12288: neuron 12288 is out of range; each "
        "layer's neurons are 0 to 12287"
    )


def test_every_entry_out_of_range_is_listed() -> None:
    deep = NeuronSet(
        "tiny", "deep", [(1, 5, 1.0), (2, 0, 1.0), (0, 128, 0.5)], "x", 3.0
    )
    where = "segments[0] (tiny:set:deep at +1 from reply token 5 to the end)"
    for shape in (TINY, TINY_SHAPE):
        assert _refused(onset(deep, 1.0, 5), shape) == [
            f"{where}: tiny:set:deep: tiny:L2:N0: layer 2 is out of range; the "
            "model's layers are 0 to 1",
            f"{where}: tiny:set:deep: tiny:L0:N128: neuron 128 is out of range; "
            "each layer's neurons are 0 to 127",
        ]
    with pytest.raises(ValueError) as error:  # the first, in the set's own words
        deep.validate(TINY_SHAPE)
    assert _refused(onset(deep, 1.0, 5))[0] == f"{where}: {error.value}"


def test_a_set_of_another_model_is_unknown() -> None:
    assert _refused(Schedule("x", [Segment(PIRATE, 1.0, 1.0)])) == [
        "segments[0] (qwen3-8b:set:pirate at +1 from reply token 0 to the end): "
        "unknown unit qwen3-8b:set:pirate: it is a set of neurons of qwen3-8b, "
        "not of tiny"
    ]


def test_with_a_shape_alone_a_set_of_a_model_of_another_shape_is_unknown() -> None:
    problems = _refused(Schedule("x", [Segment(PIRATE, 1.0, 1.0)]), TINY_SHAPE)
    assert problems == [
        "segments[0] (qwen3-8b:set:pirate at +1 from reply token 0 to the end): "
        "unknown unit qwen3-8b:set:pirate: qwen3-8b is shaped ModelInfo(layers=36, "
        "hidden_size=4096, mlp_size=12288), but the model is shaped "
        "ModelInfo(layers=2, hidden_size=64, mlp_size=128)"
    ]
    mixed = Schedule("x", [Segment(WIDE, 1.0, 1.0), Segment(PIRATE, 1.0, 1.0)])
    problems = _refused(mixed, TINY_SHAPE)
    assert problems[0] == "the units name more than one model: qwen3-8b, tiny"
    assert len(problems) == 2 and "unknown unit qwen3-8b:set:pirate" in problems[1]


def test_a_set_whose_model_left_the_registry_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ModelSpec("passing", "local:passing", "main", 2, 64, 128, "test")
    monkeypatch.setitem(REGISTRY, "passing", spec)
    gone = NeuronSet("passing", "gone", [(1, 5, 1.0)], "test")
    monkeypatch.delitem(REGISTRY, "passing")
    with pytest.raises(ValueError, match="unknown unit passing:set:gone: passing is"):
        validate(onset(gone, 1.0, 0), TINY_SHAPE, 200)
    with pytest.raises(ValueError, match="of neurons of passing, not of tiny"):
        validate(onset(gone, 1.0, 0), TINY, 200)


def test_every_problem_with_sets_is_listed_at_once() -> None:
    deep = NeuronSet("tiny", "deep", [(2, 0, 1.0)], "test")
    schedule = Schedule(
        "bad",
        [
            Segment(deep, 1.0, 1.0),
            Segment(PAIR, 1.0, 1.0, 10, 5),
            *constant(WIDE, 2.0).segments,
        ],
    )
    problems = _refused(schedule)
    assert len(problems) == 4
    assert "tiny:set:deep: tiny:L2:N0: layer 2 is out of range" in problems[0]
    assert "segments[1] (tiny:set:pair" in problems[1] and "ends before" in problems[1]
    for layer, problem in enumerate(problems[2:]):
        assert problem == _over(layer, "2", "the prompt pass", "tiny:set:wide at +2")


def test_a_sets_segment_past_max_new_tokens_warns() -> None:
    with pytest.warns(ScheduleWarning, match=r"\(tiny:set:wide at \+1 on reply tok"):
        found = validate(pulse(WIDE, 1.0, 190, 20), TINY, 200)
    assert "runs past max_new_tokens 200" in found[0]


# --- The ceiling, layer by layer


def test_a_set_over_the_ceiling_is_refused_in_each_of_its_layers() -> None:
    assert _refused(constant(WIDE, 1.5)) == [
        _over(layer, "1.5", "the prompt pass", "tiny:set:wide at +1.5")
        for layer in (0, 1)
    ]


def test_a_set_and_a_neuron_add_up_only_in_the_layer_they_share() -> None:
    schedule = Schedule("x", constant(WIDE, 1.0).segments + constant(A, 1.0).segments)
    assert _refused(schedule) == [
        _over(1, "1.414", "the prompt pass", "tiny:set:wide at +1, tiny:L1:N5 at +1")
    ]
    assert layer_peaks(schedule) == {0: (1.0, None), 1: (math.hypot(1, 1), None)}


def test_a_neuron_in_a_layer_the_set_does_not_touch_adds_nothing() -> None:
    _quiet(Schedule("x", constant(PAIR, 1.4).segments + constant(C, 1.4).segments))
    beside = Schedule("x", constant(LOW, 1.4).segments + constant(A, -1.4).segments)
    _quiet(beside)
    assert layer_peaks(beside) == {0: (1.4, None), 1: (1.4, None)}
    # A set in layers 14 and 18 of Qwen3-8B, and a neuron in layer 15, between.
    gap = constant(
        NeuronSet("qwen3-8b", "gap", [(14, 1, 1.0), (18, 2, 1.0)], "x", 9.0), 1.0
    )
    between = constant(NeuronUnit("qwen3-8b", 15, 3), 1.0)
    _quiet(Schedule("x", gap.segments + between.segments), QWEN3_8B)
    on_18 = constant(NeuronUnit("qwen3-8b", 18, 3), 1.0)
    all_three = Schedule("x", gap.segments + between.segments + on_18.segments)
    assert _refused(all_three, QWEN3_8B) == [  # layer 15's neuron is not named
        _over(
            18,
            "1.414",
            "the prompt pass",
            "qwen3-8b:set:gap at +1, qwen3-8b:L18:N3 at +1",
        )
    ]


def test_allow_over_ceiling_lets_sets_through() -> None:
    over = Schedule("x", constant(WIDE, 1.5).segments + constant(A, 1.0).segments)
    _quiet(Schedule("x", over.segments, allow_over_ceiling=True))
    _quiet(over, ceiling=1.9)  # sqrt(1.5 ** 2 + 1) = 1.803 in layer 1
    _quiet(over, ceiling=None)
    (problem,) = _refused(over, ceiling=1.6)  # layer 0's 1.5 is under it
    assert problem.startswith("layer 1: the combined push reaches 1.803 on the")


def test_a_sets_ramp_peaks_on_its_last_token_in_each_layer() -> None:
    # WIDE climbs to 1.2 on token 9, with C at 0.8 beside it in layer 0.
    schedule = Schedule(
        "x", [Segment(WIDE, 0.0, 1.2, 0, 10), Segment(C, 0.8, 0.8, 5, 10)]
    )
    assert layer_peaks(schedule) == {0: (math.hypot(1.2, 0.8), 9), 1: (1.2, 9)}
    assert _refused(schedule) == [
        _over(0, "1.442", "reply token 9", "tiny:set:wide at +1.2, tiny:L0:N3 at +0.8")
    ]


def test_a_neuron_that_meets_a_sets_turns_peaks_where_they_first_meet() -> None:
    # WIDE is on for tokens 10 to 13, 18 to 21, ... and PAIR for 14 to 17,
    # 22 to 25, 30 to 33, ...; B, in layer 1 from token 30, meets PAIR on
    # token 30 first, then WIDE on 34, as hard.
    turns = alternate(WIDE, PAIR, 1.0, -1.0, 4, start=10).segments
    schedule = Schedule("x", [*turns, Segment(B, 1.0, 1.0, 30)])
    assert layer_peaks(schedule) == {0: (1.0, 10), 1: (math.hypot(1, 1), 30)}
    assert _refused(schedule) == [
        _over(1, "1.414", "reply token 30", "tiny:set:pair at -1, tiny:L1:N9 at +1")
    ]


def test_a_sets_own_segments_add_up_from_the_first_pass_they_meet() -> None:
    schedule = Schedule(
        "x", [*constant(WIDE, 1.0).segments, Segment(WIDE, 0.25, 0.25, 5)]
    )
    assert layer_peaks(schedule) == {0: (1.25, 5), 1: (1.25, 5)}


def test_layer_peaks_has_every_layer_a_set_touches() -> None:
    spread = NeuronSet(
        "qwen3-8b", "spread", [(layer, 7 * layer, 1.0) for layer in range(19)], "x", 9.0
    )
    assert layer_peaks(pulse(spread, -0.5, 3, 4)) == {
        layer: (0.5, 3) for layer in range(19)
    }


def test_layers_that_the_same_segments_push_are_worked_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two pairs of sets in all 36 layers take turns every 101 and 103 tokens,
    # lining up again only every 20,806: 20,910 passes to check per layer.
    def spread(name: str, index: int) -> NeuronSet:
        entries = [(layer, index, 1.0) for layer in range(36)]
        return NeuronSet("qwen3-8b", name, entries, "test", 9.0)

    first = alternate(spread("a", 1), spread("b", 2), 0.5, 0.5, 101).segments
    second = alternate(spread("c", 3), spread("d", 4), 0.5, 0.5, 103).segments
    seven = Segment(NeuronUnit("qwen3-8b", 7, 0), 0.5, 0.5)  # layer 7 differs
    calls: list[int] = []
    work = validation._peak

    def counted(segments: list[Segment], layer: int) -> tuple[float, int | None] | None:
        calls.append(layer)
        return work(segments, layer)

    monkeypatch.setattr(validation, "_peak", counted)
    began = time.perf_counter()
    peaks = layer_peaks(Schedule("x", [*first, *second, seven]))
    assert time.perf_counter() - began < 2
    assert calls == [0, 7]  # 35 layers alike, worked once, and layer 7 alone
    assert peaks == {layer: (math.hypot(0.5, 0.5), 0) for layer in range(36)} | {
        7: (math.hypot(0.5, 0.5, 0.5), 0)
    }


def _random_schedule(rng: random.Random, units: list[Unit]) -> Schedule:
    """Up to six segments of the units, as test_schedule_ceiling draws them:
    every start and end before token 42, every cycle 2 to 6 tokens long."""
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


def test_layer_peaks_with_sets_match_a_scan_of_every_pass() -> None:
    # The pushes repeat every lcm(2, ..., 6) = 60 tokens after token 42.
    rng = random.Random(20261112)
    for _ in range(300):
        schedule = _random_schedule(rng, [A, C, PAIR, LOW, WIDE])
        scanned: dict[int, list[tuple[float, int | None]]] = {}
        for reply_token in [None, *range(120)]:
            doses = schedule.doses_at(reply_token)
            for layer in {layer for unit in doses for layer in unit_layers(unit)}:
                push = math.hypot(
                    *(dose for u, dose in doses.items() if layer in unit_layers(u))
                )
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


def _verdict(schedule: Schedule) -> str:
    """validate's refusal, or "" for a schedule it accepts."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ScheduleWarning)
            validate(schedule, TINY, 200)
    except ValueError as error:
        return str(error)
    return ""


def test_a_set_of_one_neuron_counts_exactly_as_the_neuron() -> None:
    rng = random.Random(20261113)
    refused = 0
    for _ in range(300):
        schedule = _random_schedule(rng, [A, B, C])
        segments = [
            replace(s, unit=ONE) if s.unit == A else s for s in schedule.segments
        ]
        as_set = Schedule("x", segments)
        assert layer_peaks(as_set) == layer_peaks(schedule)  # bit for bit
        verdict = _verdict(schedule)
        assert _verdict(as_set) == verdict.replace(str(A), str(ONE))
        refused += bool(verdict)
    assert 30 < refused < 270  # both verdicts were tried, many times
