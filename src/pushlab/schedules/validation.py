"""validate(): whether a schedule suits a model and a reply length.

It refuses, listing every problem at once:
- units unknown to the model, and layers or neurons out of its range;
- a reply segment that ends before it starts, or covers no reply token;
- a prompt segment with a start or end, or that ramps or repeats (the
  prompt pass is one pass); a ramp with no end, of a single reply token, or
  that repeats;
- any pass where the combined push in one layer is above the ceiling,
  unless the schedule sets allow_over_ceiling, which lifts that rule alone.

The combined push in a layer is the square root of the summed squares of its
units' doses on that pass (research plan, "Dosing rules"), each unit's dose
being the sum of its segments there, as the engine adds them. Every pass is
checked: the prompt pass and every reply token, however long the reply, so
the verdict does not depend on max_new_tokens. Only the warnings do: a
segment that no pass reaches, that runs past max_new_tokens, or whose ramp
ends on a token no pass reads, is warned about but accepted. Nothing here
needs model weights, so a dry run in a cloud session can use it.
"""

import math
import numbers
import warnings
from collections.abc import Mapping

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY, ModelSpec
from pushlab.schedules.model import Schedule, Segment, describe_pass, dose_text
from pushlab.units.neuron import NeuronUnit

# The low end of the collapse range Liu (2026) measured in Qwen and Llama
# models, 1.4 to 1.7 times the residual size (1.65 for Qwen3.5-4B): research
# plan, "A dose law for single neurons". A guard rather than a dose, until
# the plan's dose ladder measures each model's own ceiling.
DEFAULT_CEILING = 1.4


class ScheduleWarning(UserWarning):
    """Part of a schedule will not take effect in a reply this long."""


def _shape(model_info: object) -> tuple[str | None, ModelInfo]:
    if isinstance(model_info, ModelSpec):
        spec = model_info
        return spec.name, ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size)
    if isinstance(model_info, ModelInfo):
        return None, model_info
    raise ValueError(
        "model_info must be a ModelSpec, such as REGISTRY['qwen3.5-4b'], or a "
        f"ModelInfo, not {type(model_info).__name__}"
    )


def _unit_problem(unit: NeuronUnit, name: str | None, shape: ModelInfo) -> str | None:
    """Why the unit is unknown to the model or out of its range, or None."""
    if name is not None and unit.model_name != name:
        return (
            f"unknown unit {unit}: it is a neuron of {unit.model_name}, not of {name}"
        )
    if name is None:
        spec = REGISTRY.get(unit.model_name)  # the registry may have changed since
        if spec is None:
            return (
                f"unknown unit {unit}: {unit.model_name} is not in the model registry"
            )
        registered = ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size)
        if registered != shape:
            return (
                f"unknown unit {unit}: {unit.model_name} is shaped {registered}, "
                f"but the model is shaped {shape}"
            )
    try:
        unit.validate(shape)
    except ValueError as error:
        return str(error)
    return None


def _window_problems(segment: Segment) -> list[str]:
    """Why the segment's window and doses cannot be, as validate words it."""
    ramps = segment.alpha_start != segment.alpha_end
    if segment.region == "prompt":
        problems = []
        if segment.start != 0 or segment.end is not None:
            problems.append(
                "a prompt segment covers the whole prompt pass, so it takes start "
                f"0 and end None, not start {segment.start} and end {segment.end}"
            )
        if ramps or segment.repeats:
            problems.append(
                "the prompt pass is one pass, so a prompt segment cannot ramp or "
                "repeat: give it one dose, and no on_for or off_for"
            )
        return problems
    if segment.end is not None and segment.end < segment.start:
        return [
            f"it ends before it starts: end {segment.end} is before start "
            f"{segment.start}, and end is the first reply token it no longer covers"
        ]
    if segment.end is not None and segment.end == segment.start:
        return [
            f"it covers no reply token: end {segment.end} is its start, and end "
            "is not included"
        ]
    if ramps and segment.repeats:
        return ["a segment that repeats takes one dose: it cannot ramp"]
    if ramps and segment.end is None:
        return [
            "a ramp needs an end, for alpha_end to fall on; a segment with no "
            "end takes one dose"
        ]
    if ramps and segment.end == segment.start + 1:
        return [
            "a ramp needs at least two reply tokens, alpha_start on the first and "
            f"alpha_end on the last; this one covers only reply token {segment.start}"
        ]
    return []


def _passes(schedule: Schedule) -> list[int | None]:
    """The prompt pass, and the reply tokens where a layer's push can peak.

    Between two reply tokens where some segment starts, ends, turns on or
    turns off, the same segments are on, and each dose is constant or moves
    in a straight line. A sum of squares of straight lines is highest at one
    end of the stretch, so the first and last token of every stretch are
    enough. Past the last start or end, only segments with no end are on:
    constant ones, and repeating ones, which repeat together every
    lcm(cycles) tokens, so one such stretch of tokens is enough.
    """
    reply = [s for s in schedule.segments if s.region == "reply"]
    boundaries = {s.start for s in reply} | {s.end for s in reply if s.end is not None}
    cycles = [s.on_for + s.off_for for s in reply if s.on_for and s.off_for]
    if cycles:
        horizon = max(boundaries) + math.lcm(*cycles)
        for s in reply:
            if s.on_for and s.off_for:
                stop = horizon if s.end is None else min(s.end, horizon)
                for first in range(s.start, stop, s.on_for + s.off_for):
                    boundaries |= {first, first + s.on_for}
    tokens = boundaries | {token - 1 for token in boundaries if token > 0}
    return [None, *sorted(tokens)]


def _combined(doses: Mapping[NeuronUnit, float], layer: int) -> float:
    return math.hypot(*(dose for unit, dose in doses.items() if unit.layer == layer))


def layer_peaks(schedule: Schedule) -> dict[int, tuple[float, int | None]]:
    """For each pushed layer, its largest combined push and the first pass with it.

    The combined push on a pass is the square root of the summed squares of
    the layer's units' doses there; a pass is a reply token, or None for the
    prompt pass, which comes first. Each segment must be valid on its own
    terms (validate checks them first). The peak is exact up to rounding:
    a dose inside a ramp is worked by a formula, and can differ from its
    straight line in the last of its 16 digits.

    Example:
        >>> from pushlab.schedules.shapes import constant
        >>> a = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> b = NeuronUnit.parse("qwen3.5-4b:L14:N100")
        >>> both = Schedule("x", constant(a, 1.0).segments + constant(b, 1.0).segments)
        >>> peak, where = layer_peaks(both)[14]
        >>> round(peak, 3), where
        (1.414, None)
    """
    peaks: dict[int, tuple[float, int | None]] = {}
    for reply_token in _passes(schedule):
        doses = schedule.doses_at(reply_token)
        for layer in sorted({unit.layer for unit in doses}):
            push = _combined(doses, layer)
            if layer not in peaks or push > peaks[layer][0]:
                peaks[layer] = (push, reply_token)
    return peaks


def _reach_warning(segment: Segment, max_new_tokens: int) -> str | None:
    """Why part of a reply segment never takes effect, or None.

    A reply of max_new_tokens tokens has tokens 0 to max_new_tokens - 1, and
    no pass reads its last token, so the passes read reply tokens 0 to
    max_new_tokens - 2.
    """
    if segment.region != "reply":
        return None
    last_pass = max_new_tokens - 2
    if segment.start > last_pass:
        if last_pass < 0:
            why = "with max_new_tokens 1 only the prompt pass runs"
        else:
            why = (
                f"with max_new_tokens {max_new_tokens} the passes read reply tokens "
                f"0 to {last_pass}"
            )
        return f"never pushes: {why}, and it starts at reply token {segment.start}"
    if segment.end is not None and segment.end > max_new_tokens:
        lost = (
            f"reply token {max_new_tokens} never comes"
            if segment.end - 1 == max_new_tokens
            else f"reply tokens {max_new_tokens} to {segment.end - 1} never come"
        )
        return (
            f"runs past max_new_tokens {max_new_tokens}: a reply has tokens 0 to "
            f"{max_new_tokens - 1}, so its {lost}"
        )
    if segment.alpha_start != segment.alpha_end and segment.end == max_new_tokens:
        return (
            f"never reaches alpha_end: it falls on reply token {segment.end - 1}, "
            f"the reply's last, and the last pass reads reply token {last_pass}"
        )
    return None


def _is_finite(value: object) -> bool:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _exact(value: float) -> str:
    """A number, short when that is exact (a Fraction ceiling becomes a float)."""
    value = float(value)
    short = format(value, "g")
    return short if float(short) == value else repr(value)


def validate(
    schedule: Schedule,
    model_info: ModelSpec | ModelInfo,
    max_new_tokens: int,
    ceiling: float = DEFAULT_CEILING,
) -> list[str]:
    """Refuse a schedule that cannot run on the model; return its warnings.

    model_info is the model's registry entry, such as REGISTRY["qwen3.5-4b"]:
    then every unit must be one of its neurons. A ModelInfo, which has no
    name, checks only that each unit's model has this shape. ceiling bounds
    the combined push in one layer on any pass (see the module's docstring).
    Every problem found is listed in one ValueError. Otherwise each warning
    is issued as a ScheduleWarning and returned, so a command can print them
    in its own words and ignore ScheduleWarning.

    Example:
        >>> from pushlab.schedules.shapes import constant, pulse
        >>> qwen = REGISTRY["qwen3.5-4b"]
        >>> a = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> validate(constant(a, 1.0), qwen, max_new_tokens=200)
        []
        >>> validate(constant(a, 1.5), qwen, 200)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: schedule 'constant' is refused:
        - layer 14: the combined push reaches 1.5 on the prompt pass ...
        >>> import warnings
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     found = validate(pulse(a, 1.0, 190, 20), qwen, 200)
        >>> print(found[0])  # doctest: +ELLIPSIS
        segments[0] (... on reply tokens 190 to 209) runs past max_new_tokens 200: ...
    """
    if not isinstance(schedule, Schedule):
        raise ValueError(f"schedule must be a Schedule, not {type(schedule).__name__}")
    name, shape = _shape(model_info)
    if not (
        isinstance(max_new_tokens, int)
        and not isinstance(max_new_tokens, bool)
        and max_new_tokens >= 1
    ):
        raise ValueError(
            f"max_new_tokens must be a whole number from 1, not {max_new_tokens!r}"
        )
    if not (_is_finite(ceiling) and ceiling > 0):
        raise ValueError(f"ceiling must be a finite number above 0, not {ceiling!r}")

    problems: list[str] = []
    if name is None and len(models := {u.model_name for u in schedule.units()}) > 1:
        problems.append(
            f"the units name more than one model: {', '.join(sorted(models))}"
        )
    good = []
    for i, segment in enumerate(schedule.segments):
        where = f"segments[{i}] ({segment.describe()})"
        found = _window_problems(segment)
        if problem := _unit_problem(segment.unit, name, shape):
            found.insert(0, problem)
        problems.extend(f"{where}: {problem}" for problem in found)
        if not found:
            good.append(segment)
    if not schedule.allow_over_ceiling:
        checked = Schedule(schedule.name, good)
        for layer, (peak, reply_token) in sorted(layer_peaks(checked).items()):
            # A push too large for a float is inf, and is refused as well.
            if not math.isfinite(peak) or peak > ceiling:
                doses = checked.doses_at(reply_token)
                units = ", ".join(
                    f"{unit} at {dose_text(dose)}"
                    for unit, dose in doses.items()
                    if unit.layer == layer
                )
                shown = f"{peak:.4g}"
                if float(shown) <= ceiling:  # just over: show every digit
                    shown = repr(peak)
                problems.append(
                    f"layer {layer}: the combined push reaches {shown} on "
                    f"{describe_pass(reply_token)} ({units}), above the ceiling of "
                    f"{_exact(ceiling)}; set allow_over_ceiling: true to allow it"
                )
    if problems:
        listed = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"schedule {schedule.name!r} is refused:\n{listed}")

    found_warnings = []
    for i, segment in enumerate(schedule.segments):
        if warning := _reach_warning(segment, max_new_tokens):
            message = f"segments[{i}] ({segment.describe()}) {warning}"
            warnings.warn(message, ScheduleWarning, stacklevel=2)
            found_warnings.append(message)
    return found_warnings
