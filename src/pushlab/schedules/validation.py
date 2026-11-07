"""validate(): whether a schedule suits a model and a reply length.

It refuses, listing every problem at once:
- units unknown to the model, and layers or neurons out of its range;
- a reply segment that ends before it starts, or covers no reply token;
- a prompt segment with a start or end, or a dose that ramps (the prompt
  pass is one pass), and a ramp with no end or of a single reply token;
- any pass where the combined push in one layer is above the ceiling,
  unless the schedule sets allow_over_ceiling.

The combined push in a layer is the square root of the summed squares of its
units' doses on that pass (research plan, "Dosing rules"), each unit's dose
being the sum of its segments there, as the engine adds them. Every pass is
checked: the prompt pass and every reply token, however long the reply, so
the verdict does not depend on max_new_tokens. Only the warnings do: a
segment that runs past max_new_tokens, or that no pass reaches, is warned
about but accepted. Nothing here needs model weights, so a dry run in a
cloud session can use it.
"""

import math
import numbers
import warnings
from collections.abc import Mapping

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY, ModelSpec
from pushlab.schedules.model import Schedule, Segment
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
    spec = REGISTRY.get(unit.model_name)
    if name is None and spec is None:
        return f"unknown unit {unit}: {unit.model_name} is not in the model registry"
    if name is None:
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
    problems = []
    ramps = segment.alpha_start != segment.alpha_end
    if segment.region == "prompt":
        if segment.start != 0 or segment.end is not None:
            problems.append(
                "a prompt segment covers the whole prompt pass, so it takes "
                f"start 0 and end None, not start {segment.start} and end {segment.end}"
            )
        if ramps:
            problems.append(
                "the prompt pass is one pass, so a prompt segment cannot ramp: "
                "give alpha_start and alpha_end the same dose"
            )
        return problems
    if segment.end is not None and segment.end < segment.start:
        problems.append(
            f"it ends before it starts: end {segment.end} is before start "
            f"{segment.start}, and end is the first reply token it no longer covers"
        )
    elif segment.end is not None and segment.end == segment.start:
        problems.append(
            f"it covers no reply token: end {segment.end} is its start, and end "
            "is not included"
        )
    elif ramps and segment.end is None:
        problems.append(
            "a ramp needs an end, for alpha_end to fall on; a segment with no "
            "end takes one dose"
        )
    elif ramps and segment.end == segment.start + 1:
        problems.append(
            "a ramp needs at least two reply tokens, alpha_start on the first and "
            f"alpha_end on the last; this one covers only reply token {segment.start}"
        )
    return problems


def _passes(schedule: Schedule) -> list[int | None]:
    """The prompt pass, and the reply tokens where a layer's push can peak.

    Between two reply tokens where a segment starts or ends, the same
    segments are on, and each dose moves in a straight line. A sum of
    squares of straight lines is highest at one end of the stretch, so the
    first and last token of every stretch are enough. Past the last
    boundary, every segment still on has no end and takes one dose.
    """
    boundaries = set()
    for segment in schedule.segments:
        if segment.region == "reply":
            boundaries.add(segment.start)
            if segment.end is not None:
                boundaries.add(segment.end)
    tokens = boundaries | {token - 1 for token in boundaries if token > 0}
    return [None, *sorted(tokens)]


def _combined(doses: Mapping[NeuronUnit, float], layer: int) -> float:
    return math.hypot(*(dose for unit, dose in doses.items() if unit.layer == layer))


def layer_peaks(schedule: Schedule) -> dict[int, tuple[float, int | None]]:
    """For each pushed layer, its largest combined push and the first pass with it.

    The combined push on a pass is the square root of the summed squares of
    the layer's units' doses there; a pass is a reply token, or None for the
    prompt pass, which comes first. Each segment must be valid on its own
    terms (validate checks them first).

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


def _pass_name(reply_token: int | None) -> str:
    return "the prompt pass" if reply_token is None else f"reply token {reply_token}"


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
        return (
            f"runs past max_new_tokens {max_new_tokens}: a reply has tokens 0 to "
            f"{max_new_tokens - 1}, so its reply tokens {max_new_tokens} to "
            f"{segment.end - 1} never come"
        )
    return None


def validate(
    schedule: Schedule,
    model_info: ModelSpec | ModelInfo,
    max_new_tokens: int,
    ceiling: float = DEFAULT_CEILING,
) -> list[str]:
    """Refuse a schedule that cannot run on the model; return its warnings.

    model_info is the model's registry entry, such as REGISTRY["qwen3.5-4b"]:
    then every unit must be one of its neurons. A ModelInfo, which has no
    name, checks the shape alone. ceiling bounds the combined push in one
    layer on any pass (see the module's docstring). Every problem found is
    listed in one ValueError. Otherwise each warning is issued as a
    ScheduleWarning and returned, so a command can print them itself.

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
    if not (
        isinstance(ceiling, numbers.Real)
        and not isinstance(ceiling, bool)
        and math.isfinite(ceiling)
        and ceiling > 0
    ):
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
            if peak > ceiling:
                doses = checked.doses_at(reply_token)
                units = ", ".join(
                    f"{unit} at {dose:+g}"
                    for unit, dose in doses.items()
                    if unit.layer == layer
                )
                shown = f"{peak:.4g}"
                if float(shown) <= ceiling:  # just over: show every digit
                    shown = repr(peak)
                problems.append(
                    f"layer {layer}: the combined push reaches {shown} on "
                    f"{_pass_name(reply_token)} ({units}), above the ceiling of "
                    f"{ceiling:g}; set allow_over_ceiling: true to allow it"
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
