"""Push and StepState records: their checks, and which passes a push is on."""

import pytest
import torch

from pushlab.engine.push import Push, StepState
from pushlab.units.neuron import NeuronUnit

UNIT = NeuronUnit("tiny", layer=1, index=5)


def _push(start: int = 0, end: int | None = None, include_prompt: bool = True) -> Push:
    return Push(
        unit=UNIT, alpha=1.0, start=start, end=end, include_prompt=include_prompt
    )


def test_active_at_follows_start_end_and_include_prompt() -> None:
    def active(push: Push, step: int | None) -> bool:
        state = StepState()
        if step is not None:
            state.at_reply_token(step)
        return push.active_at(state)

    window = _push(start=3, end=6, include_prompt=False)
    assert [active(window, k) for k in [None, 0, 1, 2, 3, 4, 5, 6, 7]] == [
        False, False, False, False, True, True, True, False, False,
    ]  # fmt: skip
    to_the_end = _push(start=2, end=None, include_prompt=True)
    assert active(to_the_end, None) and not active(to_the_end, 1)
    assert active(to_the_end, 2) and active(to_the_end, 10_000)


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"unit": "tiny:L1:N5"}, "NeuronUnit"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"start": -1}, "start"),
        ({"start": 1.0}, "start"),
        ({"end": 0}, "end"),
        ({"start": 4, "end": 4}, "end"),
        ({"end": 2.5}, "end"),
        ({"include_prompt": 1}, "include_prompt"),
    ],
)
def test_a_bad_push_is_refused(changes: dict[str, object], problem: str) -> None:
    values: dict[str, object] = {"unit": UNIT, "alpha": 1.0, "start": 0, "end": None}
    values["include_prompt"] = True
    with pytest.raises(ValueError, match=problem):
        Push(**(values | changes))  # type: ignore[arg-type]


def test_include_prompt_must_be_chosen() -> None:
    with pytest.raises(TypeError, match="include_prompt"):
        Push(unit=UNIT, alpha=1.0)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("call", "problem"),
    [
        (lambda state: state.at_reply_token(-1), "reply token"),
        (lambda state: state.at_reply_token(True), "reply token"),
        (lambda state: state.at_prompt(padding=torch.zeros(5)), "padding"),
        (lambda state: state.at_prompt(padding=torch.zeros(1, 5)), "padding"),
    ],
)
def test_a_bad_step_is_refused(call: object, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        call(StepState())  # type: ignore[operator]


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"reply_token": -1}, "reply token"),
        ({"padding": torch.zeros(1, 5, dtype=torch.int64)}, "bool"),
    ],
)
def test_a_step_state_checks_what_it_is_made_with(
    fields: dict[str, object], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        StepState(**fields)  # type: ignore[arg-type]
