"""S1's pushes as schedules: Push.segments and schedule_of."""

import pytest

from pushlab.engine.push import Push, schedule_of
from pushlab.schedules.model import Schedule, Segment
from pushlab.units.neuron import NeuronUnit

UNIT = NeuronUnit("tiny", 1, 5)
OTHER = NeuronUnit("tiny", 0, 3)


def test_a_push_on_the_prompt_pass_is_a_prompt_segment_and_its_window() -> None:
    push = Push(unit=UNIT, alpha=1, start=2, end=6, include_prompt=True)
    assert push.segments() == (
        Segment(UNIT, 1.0, 1.0, region="prompt"),
        Segment(UNIT, 1.0, 1.0, 2, 6, "reply"),
    )


def test_a_push_off_the_prompt_pass_is_its_window_alone() -> None:
    push = Push(unit=UNIT, alpha=-0.5, start=3, include_prompt=False)
    assert push.segments() == (Segment(UNIT, -0.5, -0.5, 3, None, "reply"),)


@pytest.mark.parametrize("include_prompt", [True, False])
@pytest.mark.parametrize("start, end", [(0, None), (2, 6), (5, 6)])
def test_the_segments_push_exactly_the_passes_the_push_does(
    include_prompt: bool, start: int, end: int | None
) -> None:
    push = Push(
        unit=UNIT, alpha=1.0, start=start, end=end, include_prompt=include_prompt
    )
    schedule = schedule_of([push])
    assert schedule.doses_at(None) == ({UNIT: 1.0} if include_prompt else {})
    for k in range(12):
        on = start <= k and (end is None or k < end)
        assert schedule.doses_at(k) == ({UNIT: 1.0} if on else {}), k


def test_schedule_of_keeps_the_pushes_order() -> None:
    first = Push(unit=UNIT, alpha=1.0, include_prompt=True)
    second = Push(unit=OTHER, alpha=-1.0, start=4, include_prompt=False)
    assert schedule_of([first, second]) == Schedule(
        "pushes", [*first.segments(), *second.segments()]
    )
    assert schedule_of([]) == Schedule("pushes")
    assert schedule_of([second], name="late").name == "late"


def test_schedule_of_takes_only_pushes() -> None:
    with pytest.raises(ValueError, match="pushes must be Push records"):
        schedule_of([UNIT])  # type: ignore[list-item]
