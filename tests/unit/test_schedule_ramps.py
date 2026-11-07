"""Ramps: a segment's dose on every reply token, by the linear formula."""

import random
from fractions import Fraction

import pytest

from pushlab.schedules.model import Segment
from pushlab.units.neuron import NeuronUnit

A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")


def _linear(segment: Segment, k: int) -> Fraction:
    """The linear formula, worked exactly: alpha_start on the first token, alpha_end
    on the last (end - 1), and equal steps between."""
    assert segment.end is not None
    step = Fraction(k - segment.start, segment.end - 1 - segment.start)
    start, end = Fraction(segment.alpha_start), Fraction(segment.alpha_end)
    return start + (end - start) * step


def _ramps() -> list[Segment]:
    rng = random.Random(20261107)
    ramps = [
        Segment(A, 0.0, 1.0, 10, 30),
        Segment(A, 1.0, 0.0, 40, 45),
        Segment(A, -1.0, 1.0, 0, 2),
        Segment(A, 0.1, 1.3, 7, 100),
    ]
    for _ in range(200):
        start = rng.randrange(0, 300)
        ramps.append(
            Segment(
                A,
                round(rng.uniform(-1.5, 1.5), rng.randrange(1, 17)),
                round(rng.uniform(-1.5, 1.5), rng.randrange(1, 17)),
                start,
                start + rng.randrange(2, 70),
            )
        )
    return ramps


@pytest.mark.parametrize("segment", _ramps())
def test_ramp_doses_follow_the_linear_formula(segment: Segment) -> None:
    assert segment.end is not None
    doses = [segment.alpha_at(k) for k in range(segment.start, segment.end)]
    for k, dose in zip(range(segment.start, segment.end), doses, strict=True):
        assert dose == pytest.approx(float(_linear(segment, k)), rel=1e-12, abs=1e-15)
    # Both ends exactly, and every dose between them.
    assert doses[0] == segment.alpha_start and doses[-1] == segment.alpha_end
    low, high = sorted((segment.alpha_start, segment.alpha_end))
    assert all(low <= dose <= high for dose in doses)
    steps = [b - a for a, b in zip(doses, doses[1:], strict=False)]
    assert steps == pytest.approx([steps[0]] * len(steps), rel=1e-9, abs=1e-12)


def test_hand_worked_ramps() -> None:
    up = Segment(A, 0.0, 1.0, 10, 14)
    assert [up.alpha_at(k) for k in range(10, 14)] == pytest.approx(
        [0.0, 1 / 3, 2 / 3, 1.0]
    )
    down = Segment(A, 1.0, 0.0, 40, 45)
    assert [down.alpha_at(k) for k in range(40, 45)] == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_a_ramp_between_equal_doses_is_a_constant_push() -> None:
    ramp = Segment(A, 0.7, 0.7, 5, 25)
    assert all(ramp.alpha_at(k) == 0.7 for k in range(5, 25))


def test_alpha_at_refuses_a_pass_the_segment_is_not_on() -> None:
    segment = Segment(A, 1.0, 1.0, 3, 6)
    for reply_token in (None, 2, 6):
        with pytest.raises(ValueError, match="is not active on"):
            segment.alpha_at(reply_token)


@pytest.mark.parametrize(
    "segment, reply_token, problem",
    [
        (Segment(A, 0.0, 1.0, 10), 12, "no end for alpha_end to fall on"),
        (Segment(A, 0.0, 1.0, 5, 6), 5, "one reply token for alpha_start and"),
        (Segment(A, 0.0, 1.0, region="prompt"), None, "a prompt segment cannot"),
        (Segment(A, 0.0, 1.0, 0, 20, "reply", 2, 2), 1, "repeats takes one dose"),
    ],
)
def test_alpha_at_refuses_a_segment_that_cannot_ramp(
    segment: Segment, reply_token: int | None, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        segment.alpha_at(reply_token)
