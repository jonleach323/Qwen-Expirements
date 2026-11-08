"""verify's checks of each step's pushes, on runs made up here, so no model
runs (pushlab.runs.verify_steps).

A step's units must be the ones its run's schedule pushes on the pass that
chose its token, in the order the schedule names them, with one amount
each: the dose there times the unit's dose-1 amount, so of the dose's sign,
and 0 at dose 0. Step 0 is the prompt pass, which these schedules leave
alone; step j is the pass that read reply token j - 1.
"""

import pytest

from pushlab.runs.verify_steps import check_steps, first_of
from pushlab.store.records import RunRecord
from pushlab.store.steps import StepRecord

A, B = "tiny:L1:N5", "tiny:L0:N3"  # two neurons of the tiny test model
BOTH = [{"unit": A, "alpha": 1.0}, {"unit": B, "alpha": -1.0}]  # A +1, B -1


def _record(schedule: list[dict], tokens: list[int]) -> RunRecord:
    """An ok run of x with the schedule's segments, whose reply is tokens."""
    return RunRecord(
        "x-000001", "x", "tiny@main", "p", schedule, {}, 0, "", "", "", tokens, "ok"
    )


def _step(
    j: int, units: tuple[str, ...] = (), amounts: tuple[float, ...] = ()
) -> StepRecord:
    """Step j of x-000001, whose token is 5 + j."""
    return StepRecord("x-000001", j, "reply", 5 + j, -1.0, units, amounts)


def _problems(record: RunRecord, steps: list[StepRecord]) -> list[str]:
    found: list[str] = []
    check_steps(record, steps, found.append, {})
    return found


def test_a_steps_units_must_be_in_the_order_the_schedule_names_them() -> None:
    record = _record(BOTH, [5, 6, 7])
    steps = [_step(0), _step(1, (A, B), (1.5, -0.75)), _step(2, (A, B), (1.5, -0.75))]
    assert _problems(record, steps) == []
    steps[2] = _step(2, (B, A), (-0.75, 1.5))  # the same pushes, the other way
    assert _problems(record, steps) == [
        f"x-000001: step 2 pushes {B}, {A}, but its schedule pushes {A}, {B} on "
        "the pass that read reply token 1"
    ]


@pytest.mark.parametrize("amount", [-0.125, 0.125])
def test_an_amount_at_dose_0_is_caught_whatever_its_sign(amount: float) -> None:
    record = _record([{"unit": A, "alpha": 0.0}], [5, 6, 7])
    steps = [_step(0), _step(1, (A,), (0.0,)), _step(2, (A,), (0.0,))]
    assert _problems(record, steps) == []  # pushed at dose 0: listed, with 0
    steps[1] = _step(1, (A,), (amount,))
    assert _problems(record, steps) == [
        f"x-000001: step 1 adds {amount:+.4g} to {A}, whose dose there is +0"
    ]


def test_a_unit_listed_with_nothing_added_is_caught() -> None:
    # A is pushed at dose +1, yet nothing was added to it on any step.
    record = _record([{"unit": A, "alpha": 1.0}], [5, 6, 7])
    steps = [_step(0), _step(1, (A,), (0.0,)), _step(2, (A,), (0.0,))]
    assert _problems(record, steps) == [
        f"x-000001: step 1 adds +0 to {A}, whose dose there is +1; 1 more step "
        "disagrees"
    ]


def test_a_step_with_an_amount_too_few_is_caught() -> None:
    record = _record(BOTH, [5, 6])
    steps = [_step(0), _step(1, (A, B), (1.5,))]
    assert _problems(record, steps) == ["x-000001: step 1 has 1 amount for 2 units"]


def test_a_problem_of_many_steps_is_one_line_that_counts_the_rest() -> None:
    assert first_of(["x: step 1 pushes B"]) == "x: step 1 pushes B"
    assert first_of(["x: step 1 pushes B", "x: step 4 pushes B"]) == (
        "x: step 1 pushes B; 1 more step disagrees"
    )
    assert first_of(["x: step 1 pushes B", "x: step 4", "x: step 6"]) == (
        "x: step 1 pushes B; 2 more steps disagree"
    )
