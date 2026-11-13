"""verify's checks of a run's steps: the pushes each records, against its
schedule (pushlab.runs.verify, checks 5 and 6).

Step j is the pass that chose reply token j (decision 0008): the prompt
pass for step 0, the pass that read reply token j - 1 for step j. Its
units must be those the run's schedule pushes on that pass, in the order
the schedule names them, and its amounts the doses there times one
number per unit, the unit's dose-1 amount, on every step of every run, up
to the 16-bit rounding they are saved with. Under the per_prompt
calibration each prompt has its own residual sizes, so the dose-1 amount
is one number per unit and prompt; a set with its own size takes no
sizes, so it is still one number per unit.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pushlab.runs.batch import record_schedule
from pushlab.schedules.model import Schedule, dose_text
from pushlab.store.records import RunRecord
from pushlab.store.steps import REGIONS, StepRecord
from pushlab.units.neuron_set import NeuronSet

# A unit's ID, and the prompt whose sizes its dose-1 amount takes, or None
# when it is one number whatever the prompt: what verify gathers a dose-1
# amount for, over every run.
ScaleKey = tuple[str, str | None]

FEW = 3  # run IDs or steps a problem names before "and N more"
# How far an amount may be from its dose times the dose-1 amount: twice its
# 16-bit rounding, at most 2**-11 of it, or 2**-25 below 2**-14; and a dose
# small enough to give an amount of 0, as what is left when doses cancel is.
RELATIVE, ABSOLUTE, TINY = 2.0**-10, 2.0**-24, 1e-9


def count(number: int, thing: str) -> str:
    """A number of things in words.

    Example:
        >>> count(1, "run"), count(1200, "step")
        ('1 run', '1,200 steps')
    """
    return f"{number:,} {thing}" + ("" if number == 1 else "s")


def listed(items: Sequence[object], noun: str = "") -> str:
    """The first few items, after a noun if given.

    Example:
        >>> listed([3, 4, 5, 6, 7], "step"), listed(["a", "b"])
        ('steps 3, 4, 5 and 2 more', 'a and b')
    """
    shown = [str(item) for item in items[:FEW]]
    if len(items) > FEW:
        text = f"{', '.join(shown)} and {len(items) - FEW} more"
    else:
        text = " and ".join(filter(None, (", ".join(shown[:-1]), *shown[-1:])))
    return f"{noun}{'s' * (len(items) > 1)} {text}" if noun else text


def first_of(problems: list[str]) -> str:
    """The first of a run's problems of one kind, and how many steps more.

    Example:
        >>> first_of(["x: step 2 pushes B", "x: step 5 pushes B"])
        'x: step 2 pushes B; 1 more step disagrees'
    """
    rest = len(problems) - 1
    if not rest:
        return problems[0]
    verb = "disagrees" if rest == 1 else "disagree"
    return f"{problems[0]}; {count(rest, 'more step')} {verb}"


def scheduled_pushes(schedule: Schedule, step: int) -> dict[str, float]:
    """Each unit a schedule pushes on the pass that chose reply token step, in
    the order it names them, with its summed dose (Schedule.doses_at).

    Example:
        >>> from pushlab.schedules.shapes import handoff
        >>> from pushlab.units.neuron import NeuronUnit
        >>> a = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> b = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
        >>> for step in (0, 2, 3):  # the switch is at reply token 2
        ...     print(step, scheduled_pushes(handoff(a, b, 1.0, -1.0, 2), step))
        0 {'qwen3.5-4b:L14:N8949': 1.0}
        2 {'qwen3.5-4b:L14:N8949': 1.0}
        3 {'qwen3.5-4b:L13:N7956': -1.0}
    """
    doses = schedule.doses_at(None if step == 0 else step - 1)
    return {str(unit): doses[unit] for unit in schedule.units() if unit in doses}


@dataclass
class Bound:
    """The dose-1 amounts that fit a unit's amounts so far: those from low
    to high, with where each end comes from and the amount per dose there.

    An amount saved in 16 bits, divided by its dose, gives the unit's dose-1
    amount only up to its rounding: a range of them. The bound is where the
    ranges of all the amounts added overlap. Once low is above high, no one
    dose-1 amount fits them all, and disagreement names two that clash.

    Example:
        >>> bound = Bound()
        >>> bound.add_amount(1.5, 1.0, "step 0")
        >>> bound.add_amount(-3.0, -2.0, "step 1")  # 1.5 per dose again
        >>> bound.disagreement("at") is None
        True
        >>> bound.add_amount(3.1, 2.0, "step 2")  # 1.55 per dose
        >>> bound.disagreement("at")
        'is 1.5 at step 0 but 1.55 at step 2, not one number'
    """

    low: float = -math.inf
    high: float = math.inf
    low_at: tuple[str, float] = ("", 0.0)
    high_at: tuple[str, float] = ("", 0.0)

    def add(self, low: float, high: float, where: str, ratio: float) -> None:
        """Narrow the bound to the dose-1 amounts from low to high, which
        where allows; ratio is where's amount per dose, for the message.

        Example:
            >>> bound = Bound()
            >>> bound.add(1.0, 2.0, "run 1", 1.5)
            >>> bound.add(1.5, 3.0, "run 2", 2.0)
            >>> bound.low, bound.high, bound.low_at
            (1.5, 2.0, ('run 2', 2.0))
        """
        if low > self.low:
            self.low, self.low_at = low, (where, ratio)
        if high < self.high:
            self.high, self.high_at = high, (where, ratio)

    def add_amount(self, amount: float, dose: float, where: str) -> None:
        """Narrow the bound to the dose-1 amounts that give amount at dose,
        a dose other than 0, up to twice the amount's 16-bit rounding.

        Example:
            >>> bound = Bound()
            >>> bound.add_amount(-2.0, -1.0, "step 3")  # 2 per dose, give or take
            >>> bound.low, bound.high
            (1.998046875, 2.001953125)
        """
        slack = max(abs(amount) * RELATIVE, ABSOLUTE)
        ends = sorted(((amount - slack) / dose, (amount + slack) / dose))
        self.add(*ends, where, amount / dose)

    def disagreement(self, at: str) -> str | None:
        """Two amounts per dose that no one dose-1 amount fits, in words,
        each followed by at and where it comes from; None if one fits all.

        Example:
            >>> bound = Bound()
            >>> bound.add_amount(1.0, 1.0, "step 0")
            >>> bound.add_amount(2.0, 1.0, "step 1")
            >>> bound.disagreement("at")
            'is 1 at step 0 but 2 at step 1, not one number'
        """
        if self.low <= self.high:
            return None
        (less, low), (more, high) = self.high_at, self.low_at
        return f"is {low:.4g} {at} {less} but {high:.4g} {at} {more}, not one number"


def _gaps(numbers: list[int], n: int) -> str:
    """How step numbers differ from 0 to n - 1, one each, in order."""
    have = set(numbers)
    lacking = [j for j in range(n) if j not in have]
    extra = sorted(j for j in have if not 0 <= j < n)
    parts = [f"lacks {listed(lacking, 'step')}"] if lacking else []
    parts += [f"also has {listed(extra, 'step')}"] if extra else []
    return " and ".join(parts) or "has them out of order"


def _mismatch(step: int, units: Sequence[str], want: Sequence[str]) -> str:
    at = f"the pass that read reply token {step - 1}" if step else "the prompt pass"
    pushes = f"pushes {', '.join(units) or 'nothing'}, but its schedule pushes"
    return f"step {step} {pushes} {', '.join(want) or 'nothing'} on {at}"


def _agrees(amount: float, dose: float) -> bool:
    """Whether amount can be dose times a dose-1 amount, which is above 0."""
    if not math.isfinite(amount) or amount == 0:
        return amount == 0 and abs(dose) <= TINY
    return dose != 0 and (amount > 0) == (dose > 0)


def scale_key(
    unit: str, record: RunRecord, schedule: Schedule, by_prompt: bool
) -> ScaleKey:
    """Which dose-1 amount a unit's amounts in a run must share with other
    runs': the unit's alone, or with by_prompt (the per_prompt calibration)
    the unit's on the run's prompt, unless the unit is a set with its own
    size, whose dose takes no residual sizes.

    Example:
        >>> from pushlab.schedules.model import Segment
        >>> from pushlab.units.neuron import NeuronUnit
        >>> pair = NeuronSet("tiny", "pair", [(1, 5, 1.0), (0, 3, 0.5)], "-", 2.0)
        >>> neuron = NeuronUnit("tiny", 1, 5)
        >>> both = [Segment(neuron, 1.0, 1.0), Segment(pair, 1.0, 1.0)]
        >>> run = RunRecord(
        ...     "x-000001", "x", "tiny@main", "p-2", [], {}, 0, "", "", "", [], "ok"
        ... )
        >>> scale_key("tiny:L1:N5", run, Schedule("both", both), False)
        ('tiny:L1:N5', None)
        >>> scale_key("tiny:L1:N5", run, Schedule("both", both), True)
        ('tiny:L1:N5', 'p-2')
        >>> scale_key("tiny:set:pair", run, Schedule("both", both), True)
        ('tiny:set:pair', None)
    """
    if by_prompt:
        sized = {
            str(u)
            for u in schedule.units()
            if isinstance(u, NeuronSet) and u.size is not None
        }
        if unit not in sized:
            return unit, record.prompt_id
    return unit, None


def scale_problem(key: ScaleKey, bound: Bound) -> str | None:
    """What is wrong with a dose-1 amount gathered over every run (key and
    its bound), in words, or None when one number fits them all.

    Example:
        >>> bound = Bound()
        >>> bound.add_amount(1.0, 1.0, "run a")
        >>> bound.add_amount(2.0, 1.0, "run b")
        >>> scale_problem(("u", "p-2"), bound)
        "u's amount per dose on p-2 is 1 in run a but 2 in run b, not one number"
        >>> scale_problem(("u", None), Bound()) is None
        True
    """
    if (where := bound.disagreement("in")) is None:
        return None
    unit, prompt = key
    on = "" if prompt is None else f" on {prompt}"
    return f"{unit}'s amount per dose{on} {where}"


def check_steps(
    record: RunRecord,
    steps: list[StepRecord],
    say: Callable[[str], None],
    scales: dict[ScaleKey, Bound],
    by_prompt: bool = False,
) -> None:
    """Checks 5 and 6 of verify on the steps of a run with a reply: say is
    told each problem, and scales gathers each unit's dose-1 amounts over
    every run, for the check across runs (scale_problem): by unit, or with
    by_prompt, as the per_prompt calibration needs, by unit and prompt
    (scale_key).

    Example:
        >>> check_steps(record, steps, print, {})  # doctest: +SKIP
    """
    run_id, tokens = record.run_id, record.output_tokens
    first: dict[int, StepRecord] = {}  # each step number's first step
    for step in steps:
        first.setdefault(step.step, step)
    if list(first) != list(range(len(tokens))):
        how = _gaps(list(first), len(tokens))
        say(f"{run_id}'s steps are not 0 to {len(tokens) - 1}: it {how}")
    schedule: Schedule | None = None
    try:
        schedule = record_schedule(record)
    except (ValueError, TypeError, AttributeError) as error:  # not segments
        say(f"{run_id}'s schedule cannot be read: {error}")
    found: dict[str, list[str]] = {}  # each kind of problem, step by step
    bounds: dict[str, Bound] = {}

    def note(kind: str, why: str) -> None:
        found.setdefault(kind, []).append(f"{run_id}: {why}")

    for j, step in first.items():
        units, amounts = step.active_units, step.active_amounts
        if 0 <= j < len(tokens) and step.token_id != tokens[j]:
            reply = f"its reply's token {j} is {tokens[j]}"
            note("token", f"step {j}'s token is {step.token_id}, but {reply}")
        if step.region not in REGIONS:
            known = " or ".join(REGIONS)
            note("region", f"step {j} is in region {step.region!r}, not {known}")
        if schedule is None or j < 0:
            continue
        if tuple(want := scheduled_pushes(schedule, j)) != units:
            note("units", _mismatch(j, units, list(want)))
        elif len(amounts) != len(units):
            given = f"{count(len(amounts), 'amount')} for {count(len(units), 'unit')}"
            note("amounts", f"step {j} has {given}")
        else:
            for unit, amount in zip(units, amounts, strict=True):
                if not _agrees(amount, dose := want[unit]):
                    adds = f"adds {amount:+.4g} to {unit}, whose dose there is"
                    note("signs", f"step {j} {adds} {dose_text(dose)}")
                    break  # one problem a step, so that steps are counted
                if abs(dose) > TINY:
                    bound = bounds.setdefault(unit, Bound())
                    bound.add_amount(amount, dose, f"step {j}")
    for problems in found.values():
        say(first_of(problems))
    for unit, bound in bounds.items():
        if where := bound.disagreement("at"):
            say(f"{run_id}: {unit}'s amount per dose {where}")
        elif schedule is not None:  # one fits the run's steps: it must fit others'
            key = scale_key(unit, record, schedule, by_prompt)
            scale = scales.setdefault(key, Bound())
            scale.add(bound.low, bound.high, run_id, (bound.low + bound.high) / 2)
