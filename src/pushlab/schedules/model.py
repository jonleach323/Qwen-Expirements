"""Schedules as data: segments that push one unit each, and a named list of them.

A decoding loop runs the prompt pass, then one pass per reply token: the pass
that reads reply token k, counting from 0 (decision 0008). A Segment pushes one
unit, a neuron or a set of neurons (decision 0019), on some of those passes.
In region "prompt" it pushes the prompt pass, which is a region of its own,
not reply token -1. In region "reply" it pushes reply tokens start to end,
end not included (None: to the end of the reply).
Its dose moves in a straight line from alpha_start on its first reply token to
alpha_end on its last; equal values make a constant push. A reply segment can
also repeat: on for on_for tokens, then off for off_for, and again, which is
how an alternation runs to the end of a reply of any length. A Schedule is a
name and a list of segments.

These classes check each field on its own. How the fields fit together (an end
after its start, a ramp with an end) and whether a schedule suits a model and a
reply length is for pushlab.schedules.validation.validate.
"""

import math
import numbers
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

from pushlab.units.neuron import NeuronUnit  # noqa: F401 (the examples use it)
from pushlab.units.neuron_set import Unit, is_unit

# The prompt pass, and the reply's tokens. Regions inside the reply, such as a
# thinking block, come with thinking detection in a later sprint.
REGIONS = ("prompt", "reply")

# The same form as experiment names (pushlab.store.records.EXPERIMENT_NAME):
# lowercase letters and digits joined by single hyphens, such as
# handoff-hostile-to-warm, so a name can go into run IDs and folder names.
SCHEDULE_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _check_pass(reply_token: object) -> None:
    if reply_token is not None and not (_is_whole(reply_token) and reply_token >= 0):
        raise ValueError(
            "a pass is a reply token from 0, or None for the prompt pass, "
            f"not {reply_token!r}"
        )


def describe_pass(reply_token: int | None) -> str:
    """A pass in words: the prompt pass (None), or the pass of a reply token.

    Example:
        >>> describe_pass(None), describe_pass(40)
        ('the prompt pass', 'reply token 40')
    """
    return "the prompt pass" if reply_token is None else f"reply token {reply_token}"


def dose_text(alpha: float) -> str:
    """A dose with its sign: short when that is exact, every digit when not.

    So a dose just over a ceiling is never shown as the ceiling itself.

    Example:
        >>> dose_text(1.0), dose_text(-0.5), dose_text(1.4000000000000001)
        ('+1', '-0.5', '+1.4000000000000001')
    """
    short = format(alpha, "+g")
    return short if float(short) == alpha else format(float(alpha), "+")


@dataclass(frozen=True)
class Segment:
    """One unit pushed on the prompt pass, or on reply tokens start to end.

    The unit is a NeuronUnit or a NeuronSet, which every dose pushes as a
    whole (pushlab.units.dose).

    alpha_start is the dose on the segment's first reply token and alpha_end
    the dose on its last (end - 1), with equal steps between; equal values
    make a constant push. A push on reply token k is on the pass that reads
    token k, so it first changes token k + 1, and only a "prompt" segment
    reaches token 0 (decision 0008). end is not included, and None runs to
    the end of the reply. A "prompt" segment takes start 0 and end None.

    With on_for and off_for, a reply segment repeats from start: on for
    on_for reply tokens, off for off_for, on again, until end. Both are None
    for a segment that does not repeat, or both whole numbers from 1.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> pulse = Segment(unit, 1.0, 1.0, start=0, end=16)  # tokens 0 to 15
        >>> pulse.describe()
        'qwen3.5-4b:L14:N8949 at +1 on reply tokens 0 to 15'
        >>> ramp = Segment(unit, 0.0, 1.0, start=10, end=30)
        >>> ramp.alpha_at(10), ramp.alpha_at(29)
        (0.0, 1.0)
        >>> turns = Segment(unit, 1.0, 1.0, start=10, on_for=4, off_for=4)
        >>> [k for k in range(10, 30) if turns.active_at(k)]
        [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]
    """

    unit: Unit
    alpha_start: float
    alpha_end: float
    start: int = 0
    end: int | None = None
    region: str = "reply"
    on_for: int | None = None
    off_for: int | None = None

    def __post_init__(self) -> None:
        """Check each field on its own; validate() checks how they fit together."""
        if not is_unit(self.unit):
            raise ValueError(
                f"unit must be a NeuronUnit or a NeuronSet, not {self.unit!r}"
            )
        for name in ("alpha_start", "alpha_end"):
            value = getattr(self, name)
            if not _is_finite(value):
                raise ValueError(f"{name} must be a finite number, not {value!r}")
            # A frozen dataclass sets its fields through object.__setattr__.
            object.__setattr__(self, name, float(value))
        if not (_is_whole(self.start) and self.start >= 0):
            raise ValueError(f"start must be a reply token from 0, not {self.start!r}")
        if self.end is not None and not (_is_whole(self.end) and self.end >= 0):
            raise ValueError(
                f"end must be None or a reply token from 0, not {self.end!r}"
            )
        if not isinstance(self.region, str) or self.region not in REGIONS:
            raise ValueError(
                f"region must be 'prompt' or 'reply', not {self.region!r}; "
                "regions inside the reply, such as thinking, come later"
            )
        lengths = (self.on_for, self.off_for)
        if lengths != (None, None) and not all(
            _is_whole(length) and length >= 1 for length in lengths
        ):
            raise ValueError(
                "on_for and off_for must both be None, or both whole numbers "
                f"from 1, not {self.on_for!r} and {self.off_for!r}"
            )

    @property
    def repeats(self) -> bool:
        """Whether the segment turns on and off by on_for and off_for."""
        return self.on_for is not None

    def active_at(self, reply_token: int | None) -> bool:
        """Whether the segment pushes the pass of reply_token (None: the prompt pass).

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
            >>> segment = Segment(unit, 1.0, 1.0, start=3, end=6)
            >>> [k for k in range(8) if segment.active_at(k)]
            [3, 4, 5]
            >>> segment.active_at(None)
            False
        """
        _check_pass(reply_token)
        if reply_token is None or self.region == "prompt":
            return reply_token is None and self.region == "prompt"
        if reply_token < self.start or (
            self.end is not None and reply_token >= self.end
        ):
            return False
        if self.on_for is None or self.off_for is None:
            return True
        return (reply_token - self.start) % (self.on_for + self.off_for) < self.on_for

    def alpha_at(self, reply_token: int | None) -> float:
        """The dose on the pass of reply_token (None: the prompt pass).

        On reply token k it is alpha_start + (alpha_end - alpha_start) * t,
        with t = (k - start) / (end - 1 - start), worked as
        alpha_start * (1 - t) + alpha_end * t so that alpha_start falls on
        the first token and alpha_end on the last exactly. Raises ValueError
        on a pass the segment does not push, and for a ramp that cannot be:
        one with no end, of a single token, on the prompt pass, or repeating.

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
            >>> down = Segment(unit, 1.0, 0.0, start=40, end=45)
            >>> [down.alpha_at(k) for k in range(40, 45)]
            [1.0, 0.75, 0.5, 0.25, 0.0]
        """
        if not self.active_at(reply_token):
            raise ValueError(
                f"{self.describe()} is not active on {describe_pass(reply_token)}"
            )
        if self.alpha_start == self.alpha_end:
            return self.alpha_start
        if reply_token is None:
            self._cannot_ramp("the prompt pass is one pass, so a prompt segment")
        if self.repeats:
            self._cannot_ramp("a segment that repeats takes one dose, so it")
        if self.end is None:
            self._cannot_ramp("with no end for alpha_end to fall on, it")
        if self.end - 1 == self.start:
            self._cannot_ramp("with one reply token for alpha_start and alpha_end, it")
        step = (reply_token - self.start) / (self.end - 1 - self.start)
        dose = self.alpha_start * (1 - step) + self.alpha_end * step
        low, high = sorted((self.alpha_start, self.alpha_end))
        return min(max(dose, low), high)  # never past an end, even by rounding

    def _cannot_ramp(self, why: str) -> NoReturn:
        raise ValueError(f"{self.describe()}: {why} cannot ramp")

    def describe(self) -> str:
        """The segment in words: its unit, its dose and the passes it pushes.

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
            >>> Segment(unit, -1.0, -1.0, start=40).describe()
            'qwen3.5-4b:L13:N7956 at -1 from reply token 40 to the end'
            >>> Segment(unit, 0.0, -1.0, start=0, end=20).describe()
            'qwen3.5-4b:L13:N7956 from +0 to -1 on reply tokens 0 to 19'
            >>> Segment(unit, 1.0, 1.0, start=4, on_for=4, off_for=4).describe()
            'qwen3.5-4b:L13:N7956 at +1 from reply token 4 to the end, 4 on then 4 off'
        """
        if self.alpha_start == self.alpha_end:
            dose = f"at {dose_text(self.alpha_start)}"
        else:
            dose = f"from {dose_text(self.alpha_start)} to {dose_text(self.alpha_end)}"
        if self.region == "prompt":
            where = "on the prompt pass"
        elif self.end is None:
            where = f"from reply token {self.start} to the end"
        elif self.end <= self.start:
            where = f"with start {self.start} and end {self.end}"
        elif self.end - 1 == self.start:
            where = f"on reply token {self.start}"
        else:
            where = f"on reply tokens {self.start} to {self.end - 1}"
        if self.repeats:
            where += f", {self.on_for} on then {self.off_for} off"
        return f"{self.unit} {dose} {where}"


@dataclass(frozen=True)
class Schedule:
    """A named list of segments: every push of one run, pass by pass.

    segments may be given as a list or a tuple; they are kept as a tuple, in
    order. Segments of one unit that cover the same pass add up, as the
    engine adds pushes on one neuron. allow_over_ceiling lets validate()
    accept a combined push in one layer above the dose ceiling. No segments
    means no push.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> schedule = Schedule("pulse16", [Segment(unit, 1.0, 1.0, start=0, end=16)])
        >>> schedule.doses_at(15)[unit], schedule.doses_at(16)
        (1.0, {})
    """

    name: str
    segments: Sequence[Segment] = ()
    allow_over_ceiling: bool = False

    def __post_init__(self) -> None:
        """Refuse a name that is not plain, segments that are not Segments."""
        if not isinstance(self.name, str) or not SCHEDULE_NAME.fullmatch(self.name):
            raise ValueError(
                "name must be lowercase letters and digits joined by single "
                f"hyphens, such as handoff-hostile-to-warm, not {self.name!r}"
            )
        segments = self.segments
        if not isinstance(segments, list | tuple) or not all(
            isinstance(segment, Segment) for segment in segments
        ):
            raise ValueError(
                f"segments must be a list of Segment records, not {segments!r}"
            )
        object.__setattr__(self, "segments", tuple(segments))
        if not isinstance(self.allow_over_ceiling, bool):
            raise ValueError(
                "allow_over_ceiling must be True or False, "
                f"not {self.allow_over_ceiling!r}"
            )

    def units(self) -> tuple[Unit, ...]:
        """Each unit the schedule pushes, once, in order of first appearance.

        Example:
            >>> a = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
            >>> b = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
            >>> schedule = Schedule(
            ...     "x",
            ...     [
            ...         Segment(a, 1.0, 1.0, 0, 9),
            ...         Segment(b, -1.0, -1.0, 9),
            ...         Segment(a, 1, 1, 20),
            ...     ],
            ... )
            >>> [str(unit) for unit in schedule.units()]
            ['qwen3.5-4b:L14:N8949', 'qwen3.5-4b:L13:N7956']
        """
        return tuple(dict.fromkeys(segment.unit for segment in self.segments))

    def doses_at(self, reply_token: int | None) -> dict[Unit, float]:
        """Each unit pushed on the pass of reply_token, with its summed dose.

        None is the prompt pass. Segments of one unit that both push the pass
        add up. A unit whose segments push it at dose 0 is listed with 0.0, so
        "pushed at 0" stays apart from "not pushed".

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
            >>> schedule = Schedule(
            ...     "x", [Segment(unit, 1.0, 1.0, 0, 10), Segment(unit, 0.5, 0.5, 5)]
            ... )
            >>> [schedule.doses_at(k).get(unit) for k in (None, 4, 5, 10)]
            [None, 1.0, 1.5, 0.5]
        """
        _check_pass(reply_token)
        doses: dict[Unit, float] = {}
        for segment in self.segments:
            if segment.active_at(reply_token):
                dose = segment.alpha_at(reply_token)
                doses[segment.unit] = doses.get(segment.unit, 0.0) + dose
        return doses
