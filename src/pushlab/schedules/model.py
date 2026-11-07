"""Schedules as data: segments that push one neuron each, and a named list of them.

A decoding loop runs the prompt pass, then one pass per reply token: the pass
that reads reply token k, counting from 0 (decision 0008). A Segment pushes one
neuron on some of those passes. In region "prompt" it pushes the prompt pass,
which is a region of its own, not reply token -1. In region "reply" it pushes
reply tokens start to end, end not included (None: to the end of the reply).
Its dose moves in a straight line from alpha_start on its first reply token to
alpha_end on its last; equal values make a constant push. A Schedule is a name
and a list of segments.

These classes check each field on its own. How the fields fit together (an end
after its start, a ramp with an end) and whether a schedule suits a model and a
reply length is for pushlab.schedules.validation.validate.
"""

import math
import numbers
import re
from dataclasses import dataclass

from pushlab.units.neuron import NeuronUnit

# The prompt pass, and the reply's tokens. Regions inside the reply, such as a
# thinking block, come with thinking detection in a later sprint.
REGIONS = ("prompt", "reply")

# The same form as experiment names: lowercase letters and digits joined by
# single hyphens, such as handoff-hostile-to-warm, so a name can go into run
# IDs and folder names as it is.
SCHEDULE_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_pass(reply_token: object) -> None:
    if reply_token is not None and not (_is_whole(reply_token) and reply_token >= 0):
        raise ValueError(
            "a pass is a reply token from 0, or None for the prompt pass, "
            f"not {reply_token!r}"
        )


def _pass_name(reply_token: int | None) -> str:
    return "the prompt pass" if reply_token is None else f"reply token {reply_token}"


@dataclass(frozen=True)
class Segment:
    """One neuron pushed on the prompt pass, or on reply tokens start to end.

    alpha_start is the dose on the segment's first reply token and alpha_end
    the dose on its last (end - 1), with equal steps between; equal values
    make a constant push. A push on reply token k is on the pass that reads
    token k, so it first changes token k + 1, and only a "prompt" segment
    reaches token 0 (decision 0008). end is not included, and None runs to
    the end of the reply. A "prompt" segment takes start 0 and end None.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> pulse = Segment(unit, 1.0, 1.0, start=0, end=16)  # tokens 0 to 15
        >>> pulse.describe()
        'qwen3.5-4b:L14:N8949 at +1 on reply tokens 0 to 15'
        >>> ramp = Segment(unit, 0.0, 1.0, start=10, end=30)
        >>> ramp.alpha_at(10), ramp.alpha_at(29)
        (0.0, 1.0)
    """

    unit: NeuronUnit
    alpha_start: float
    alpha_end: float
    start: int = 0
    end: int | None = None
    region: str = "reply"

    def __post_init__(self) -> None:
        """Check each field on its own; validate() checks how they fit together."""
        if not isinstance(self.unit, NeuronUnit):
            raise ValueError(f"unit must be a NeuronUnit, not {self.unit!r}")
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
        return self.start <= reply_token and (
            self.end is None or reply_token < self.end
        )

    def alpha_at(self, reply_token: int | None) -> float:
        """The dose on the pass of reply_token (None: the prompt pass).

        alpha_start + (alpha_end - alpha_start) * (k - start) / (end - 1 - start)
        on reply token k: alpha_start on the first token and alpha_end, exactly,
        on the last. Raises ValueError on a pass the segment does not push, and
        for a ramp that cannot be: one with no end, one of a single token, or
        one on the prompt pass.

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
            >>> down = Segment(unit, 1.0, 0.0, start=40, end=45)
            >>> [down.alpha_at(k) for k in range(40, 45)]
            [1.0, 0.75, 0.5, 0.25, 0.0]
        """
        if not self.active_at(reply_token):
            raise ValueError(
                f"{self.describe()} is not active on {_pass_name(reply_token)}"
            )
        if self.alpha_start == self.alpha_end:
            return self.alpha_start
        if reply_token is None:
            raise ValueError(
                f"{self.describe()}: the prompt pass is one pass, so a prompt "
                "segment cannot ramp"
            )
        if self.end is None:
            raise ValueError(
                f"{self.describe()}: a ramp needs an end, for alpha_end to fall on"
            )
        last = self.end - 1
        if last == self.start:
            raise ValueError(
                f"{self.describe()}: a ramp needs at least two reply tokens, "
                "alpha_start on the first and alpha_end on the last"
            )
        if reply_token == last:
            return self.alpha_end  # exactly, not up to rounding
        step = (reply_token - self.start) / (last - self.start)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * step

    def describe(self) -> str:
        """The segment in words: its unit, its dose and the passes it pushes.

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
            >>> Segment(unit, -1.0, -1.0, start=40).describe()
            'qwen3.5-4b:L13:N7956 at -1 from reply token 40 to the end'
            >>> Segment(unit, 0.0, -1.0, start=0, end=20).describe()
            'qwen3.5-4b:L13:N7956 from +0 to -1 on reply tokens 0 to 19'
        """
        if self.alpha_start == self.alpha_end:
            dose = f"at {self.alpha_start:+g}"
        else:
            dose = f"from {self.alpha_start:+g} to {self.alpha_end:+g}"
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
        return f"{self.unit} {dose} {where}"


@dataclass(frozen=True)
class Schedule:
    """A named list of segments: every push of one run, pass by pass.

    The segments are kept as a tuple, in order. Segments of one unit that
    cover the same pass add up, as the engine adds pushes on one neuron.
    allow_over_ceiling lets validate() accept a combined push in one layer
    above the dose ceiling. No segments means no push.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> schedule = Schedule("pulse16", [Segment(unit, 1.0, 1.0, start=0, end=16)])
        >>> schedule.doses_at(15), schedule.doses_at(16)
        ({NeuronUnit(model_name='qwen3.5-4b', layer=14, index=8949): 1.0}, {})
    """

    name: str
    segments: tuple[Segment, ...] = ()
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

    def units(self) -> tuple[NeuronUnit, ...]:
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

    def doses_at(self, reply_token: int | None) -> dict[NeuronUnit, float]:
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
        doses: dict[NeuronUnit, float] = {}
        for segment in self.segments:
            if segment.active_at(reply_token):
                dose = segment.alpha_at(reply_token)
                doses[segment.unit] = doses.get(segment.unit, 0.0) + dose
        return doses
