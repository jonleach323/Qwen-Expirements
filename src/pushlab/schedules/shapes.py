"""Shapes: the common schedules of the research plan, built as segments.

Each helper returns a Schedule whose segments are the ones you would write by
hand (research plan, "Schedules"). Reply tokens count from 0, and a push on
reply token k is on the pass that reads token k, so it first changes token
k + 1; only the prompt pass reaches token 0 (decision 0008). constant and
handoff start with the reply, so by default they also push the prompt pass,
as the evil-Qwen recipe does; the shapes that start at a reply token you
choose (pulse, onset, alternate, ramp) push only reply tokens unless told
otherwise. Every helper takes a keyword-only name, which defaults to the
shape's own.

Parameters that make no segment (a period of 0, an alternation with no end)
are refused here. How the segments fit together, and whether they suit a
model, is for pushlab.schedules.validation.validate.
"""

from collections.abc import Callable

from pushlab.schedules.model import Schedule, Segment
from pushlab.units.neuron import NeuronUnit


def _whole(name: str, value: object, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be a whole number from {minimum}, not {value!r}")
    return value


def _named(name: str | None, default: str) -> str:
    return default if name is None else name


def _flag(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False, not {value!r}")
    return value


def _on_prompt(unit: NeuronUnit, alpha: float) -> Segment:
    return Segment(unit, alpha, alpha, region="prompt")


def _on_reply(unit: NeuronUnit, alpha: float, start: int, end: int | None) -> Segment:
    return Segment(unit, alpha, alpha, start, end, "reply")


def constant(
    unit: NeuronUnit,
    alpha: float,
    include_prompt: bool = True,
    *,
    name: str | None = None,
) -> Schedule:
    """One neuron at one dose from the prompt pass to the end of the reply.

    This is the evil-Qwen recipe's setup, the reference for every other
    shape. With include_prompt=False the reply's first token is left alone.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> for segment in constant(unit, 1.0).segments:
        ...     print(segment.describe())
        qwen3.5-4b:L14:N8949 at +1 on the prompt pass
        qwen3.5-4b:L14:N8949 at +1 from reply token 0 to the end
    """
    segments = [_on_reply(unit, alpha, 0, None)]
    if _flag("include_prompt", include_prompt):
        segments.insert(0, _on_prompt(unit, alpha))
    return Schedule(_named(name, "constant"), segments)


def pulse(
    unit: NeuronUnit,
    alpha: float,
    start: int,
    length: int,
    *,
    name: str | None = None,
) -> Schedule:
    """One neuron on for length reply tokens from start, then off.

    The research plan's pulse "on for tokens 0 to 15" is pulse(unit, alpha,
    0, 16). The prompt pass is not pushed, so token 0 is never changed.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> pulse(unit, 1.0, 0, 16).segments[0].describe()
        'qwen3.5-4b:L14:N8949 at +1 on reply tokens 0 to 15'
    """
    _whole("length", length, 1)
    return Schedule(
        _named(name, "pulse"), [_on_reply(unit, alpha, start, start + length)]
    )


def onset(
    unit: NeuronUnit,
    alpha: float,
    start: int,
    *,
    name: str | None = None,
) -> Schedule:
    """One neuron on from reply token start to the end of the reply: a late onset.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> onset(unit, 1.0, 50).segments[0].describe()
        'qwen3.5-4b:L14:N8949 at +1 from reply token 50 to the end'
    """
    return Schedule(_named(name, "onset"), [_on_reply(unit, alpha, start, None)])


def handoff(
    a: NeuronUnit,
    b: NeuronUnit,
    alpha_a: float,
    alpha_b: float,
    switch_at: int,
    overlap: int = 0,
    gap: int = 0,
    *,
    include_prompt: bool = True,
    name: str | None = None,
) -> Schedule:
    """A first, then B from reply token switch_at to the end of the reply.

    A pushes reply tokens 0 to switch_at - 1, and the prompt pass too unless
    include_prompt=False, so before the switch the schedule is constant(a,
    alpha_a). With overlap k, A stays on k tokens past the switch, so both
    push tokens switch_at to switch_at + k - 1. With gap g, B comes on g
    tokens after the switch, so neither pushes tokens switch_at to
    switch_at + g - 1. A handoff overlaps or leaves a gap, not both.

    Example:
        >>> a = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> b = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
        >>> for segment in handoff(a, b, 1.0, -1.0, 40, overlap=2).segments:
        ...     print(segment.describe())
        qwen3.5-4b:L14:N8949 at +1 on the prompt pass
        qwen3.5-4b:L14:N8949 at +1 on reply tokens 0 to 41
        qwen3.5-4b:L13:N7956 at -1 from reply token 40 to the end
    """
    _whole("switch_at", switch_at, 0)
    _whole("overlap", overlap, 0)
    _whole("gap", gap, 0)
    if overlap and gap:
        raise ValueError(
            "a handoff has an overlap or a gap, not both: "
            f"overlap {overlap} and gap {gap}"
        )
    a_ends = switch_at + overlap
    if not _flag("include_prompt", include_prompt) and a_ends == 0:
        raise ValueError(
            "A would never be pushed: it ends at reply token 0 and the prompt "
            "pass is off; use onset() for B alone"
        )
    segments = [_on_prompt(a, alpha_a)] if include_prompt else []
    if a_ends > 0:
        segments.append(_on_reply(a, alpha_a, 0, a_ends))
    segments.append(_on_reply(b, alpha_b, switch_at + gap, None))
    return Schedule(_named(name, "handoff"), segments)


def alternate(
    a: NeuronUnit,
    b: NeuronUnit,
    alpha_a: float,
    alpha_b: float,
    period: int,
    start: int = 0,
    end: int | None = None,
    *,
    include_prompt: bool = False,
    name: str | None = None,
) -> Schedule:
    """A and B take turns every period reply tokens, A first, from start to end.

    Each turn is a segment of its own: A on start to start + period - 1, B on
    the next period tokens, and so on; the last turn stops at end (not
    included). period counts reply tokens, so B's first pass is the one that
    reads token start + period and first changes the token after it. end is
    required: a schedule lists every turn, so to alternate to the end of the
    reply give the reply's max_new_tokens. With include_prompt=True, A also
    pushes the prompt pass.

    Example:
        >>> a = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> b = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
        >>> for segment in alternate(a, b, 1.0, -1.0, 4, start=10, end=24).segments:
        ...     print(segment.describe())
        qwen3.5-4b:L14:N8949 at +1 on reply tokens 10 to 13
        qwen3.5-4b:L13:N7956 at -1 on reply tokens 14 to 17
        qwen3.5-4b:L14:N8949 at +1 on reply tokens 18 to 21
        qwen3.5-4b:L13:N7956 at -1 on reply tokens 22 to 23
    """
    _whole("period", period, 1)
    _whole("start", start, 0)
    if end is None:
        raise ValueError(
            "alternate needs an end: a schedule lists every turn as a segment, "
            "so to alternate to the end of the reply, give end=max_new_tokens"
        )
    if not isinstance(end, int) or isinstance(end, bool) or end <= start:
        raise ValueError(
            f"end must be a reply token after start ({start}), not {end!r}"
        )
    segments = (
        [_on_prompt(a, alpha_a)] if _flag("include_prompt", include_prompt) else []
    )
    for turn, first in enumerate(range(start, end, period)):
        unit, alpha = (a, alpha_a) if turn % 2 == 0 else (b, alpha_b)
        segments.append(_on_reply(unit, alpha, first, min(first + period, end)))
    return Schedule(_named(name, "alternate"), segments)


def ramp(
    unit: NeuronUnit,
    alpha_from: float,
    alpha_to: float,
    start: int,
    end: int,
    *,
    name: str | None = None,
) -> Schedule:
    """One neuron whose dose moves in a straight line over reply tokens start to end.

    alpha_from goes on reply token start and alpha_to on end - 1, with equal
    steps between (Segment.alpha_at). A dose held after the ramp is a segment
    of its own, such as onset(unit, alpha_to, end).

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> up = ramp(unit, 0.0, 1.0, 10, 14).segments[0]
        >>> [round(up.alpha_at(k), 3) for k in range(10, 14)]
        [0.0, 0.333, 0.667, 1.0]
    """
    return Schedule(
        _named(name, "ramp"), [Segment(unit, alpha_from, alpha_to, start, end)]
    )


# Every shape by name, as schedule files name them.
SHAPES: dict[str, Callable[..., Schedule]] = {
    build.__name__: build
    for build in (constant, pulse, onset, handoff, alternate, ramp)
}
