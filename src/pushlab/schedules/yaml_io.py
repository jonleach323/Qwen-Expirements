"""Schedules as YAML: a shape name with its params, or the segments written out.

A schedule file, or a schedule inside an experiment file, is a mapping with a
name and either a shape and its params:

    name: handoff-hostile-to-warm
    shape: handoff
    params: {a: qwen3.5-4b:L14:N8949, b: qwen3.5-4b:L13:N7956,
             alpha_a: 1.0, alpha_b: -1.0, switch_at: 40}

or its segments, one mapping each:

    name: handoff-hostile-to-warm
    segments:
      - {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, region: prompt}
      - {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0, end: 40}
      - {unit: qwen3.5-4b:L13:N7956, alpha: -1.0, start: 40}

A segment takes alpha for a constant push, or alpha_start and alpha_end for a
ramp; start defaults to 0, end to null (the end of the reply) and region to
reply. A segment that repeats also takes on_for and off_for.
allow_over_ceiling: true lets validate() accept a combined push above the dose
ceiling. Unknown keys, params, shapes and units are refused with a message that
names them. Schedules are always written out as segments, so reading one back
gives an equal schedule, every dose bit for bit.

A segment may push a set of neurons (decision 0019): its unit is the set's ID
and the set's definition goes beside it (pushlab.schedules.yaml_sets):

    name: pirate
    segments:
      - unit: qwen3-8b:set:pirate-top50
        set: {entries: [[17, 1234, 0.52], [18, 99, -0.31]], size: 96.04,
              source: mayner role pirate}
        alpha: 1.0

A set's ID without its definition, a definition beside a neuron, and one ID
given two definitions are refused. to_mapping writes every segment of a set
with its definition, so a schedule with sets reads back equal too, every
coefficient and size bit for bit, and so does a run's record of it. A shape's
params name units by ID alone and cannot hold a definition, so a shape whose
params name a set is refused: a schedule that pushes a set is written as
segments.

The text is read with pushlab.schedules.yaml_text.load_yaml, which refuses a
key given twice and reads numbers and true/false as YAML 1.2 does, and written
with its dump_yaml, so writing then reading changes nothing. An experiment file
that holds a schedule should be read with load_yaml too.
"""

import inspect
import math
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import SHAPES
from pushlab.schedules.yaml_sets import (
    FORM,
    definition,
    is_set_id,
    refuse_two_definitions,
    set_from_definition,
)
from pushlab.schedules.yaml_text import dump_yaml, load_yaml
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit

SCHEDULE_KEYS = ("name", "allow_over_ceiling", "segments", "shape", "params")
SEGMENT_KEYS = (
    "unit",
    "set",
    "alpha",
    "alpha_start",
    "alpha_end",
    "start",
    "end",
    "region",
    "on_for",
    "off_for",
)
# The shapes' params that name a unit, as text such as qwen3.5-4b:L14:N8949.
UNIT_PARAMS = ("unit", "a", "b")


def _keys(given: Mapping, allowed: tuple[str, ...], where: str) -> None:
    unknown = [key for key in given if key not in allowed]
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{where}: unknown key {names}; it takes {', '.join(allowed)}")


def _number_as_text(where: str, key: str, value: object) -> None:
    """Refuse a number that YAML read as text, such as '16' or 010, by name."""
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return
        raise ValueError(
            f"{where}: {key} is the text {value!r}, not a number; write a plain "
            "decimal number, without quotes or leading zeros, such as 16 or 0.5"
        )


def _dose(where: str, key: str, value: object) -> None:
    """Refuse a dose that is not a finite number, by the key the file gave."""
    finite = isinstance(value, numbers.Real) and not isinstance(value, bool)
    try:
        finite = finite and math.isfinite(value)  # type: ignore[arg-type]
    except OverflowError:  # an int too large for a float
        finite = False
    if not finite:
        raise ValueError(f"{where}: {key} must be a finite number, not {value!r}")


def _neuron(where: str, text: object) -> NeuronUnit:
    try:
        return NeuronUnit.parse(text)  # type: ignore[arg-type]
    except ValueError as error:
        raise ValueError(f"{where}: unknown unit {text!r}: {error}") from None


def _param_unit(where: str, text: object) -> NeuronUnit:
    """A shape's unit param: a neuron, as params cannot hold a set's definition."""
    if is_set_id(text):
        raise ValueError(
            f"{where}: {text!r} names a set of neurons, and a shape's params cannot "
            "hold its definition; write the schedule as segments, each with the "
            f"set's definition beside its unit ({FORM})"
        )
    return _neuron(where, text)


def _segment_unit(data: Mapping, where: str) -> Unit:
    """A segment's unit: a neuron, or a set made from the definition beside it."""
    text = data["unit"]
    if is_set_id(text):
        if "set" not in data:
            raise ValueError(
                f"{where}: {text!r} names a set of neurons, so the segment needs the "
                f"set's definition beside it: {FORM}"
            )
        return set_from_definition(text, data["set"], where)
    if "set" in data:
        raise ValueError(
            f"{where}: set: gives a set's definition, but the unit {text!r} is no "
            "set's ID; leave set out for a neuron, or give the set's ID, such as "
            "qwen3-8b:set:pirate-top50"
        )
    return _neuron(where, text)


def _segment(data: object, where: str) -> Segment:
    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be a mapping such as {{unit: ..., alpha: 1.0}}")
    _keys(data, SEGMENT_KEYS, where)
    if "unit" not in data:
        raise ValueError(f"{where} needs a unit, such as qwen3.5-4b:L14:N8949")
    for key in SEGMENT_KEYS:
        if key not in ("unit", "set", "region"):
            _number_as_text(where, key, data.get(key))
    if "alpha" in data:
        if "alpha_start" in data or "alpha_end" in data:
            raise ValueError(
                f"{where}: give alpha or alpha_start and alpha_end, not both"
            )
        _dose(where, "alpha", data["alpha"])
        alpha_start = alpha_end = data["alpha"]
    elif "alpha_start" in data and "alpha_end" in data:
        alpha_start, alpha_end = data["alpha_start"], data["alpha_end"]
    elif "alpha_start" in data or "alpha_end" in data:
        raise ValueError(f"{where}: give alpha_start and alpha_end together")
    else:
        raise ValueError(f"{where} needs alpha, or alpha_start and alpha_end")
    unit = _segment_unit(data, where)
    try:
        return Segment(
            unit,
            alpha_start,
            alpha_end,
            data.get("start", 0),
            data.get("end"),
            data.get("region", "reply"),
            data.get("on_for"),
            data.get("off_for"),
        )
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def _from_shape(shape: object, params: object) -> list[Segment]:
    if not isinstance(shape, str) or shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; the shapes are {', '.join(SHAPES)}")
    build = SHAPES[shape]
    parameters = inspect.signature(build).parameters
    accepted = [n for n in parameters if n != "name"]
    if params is None:
        raise ValueError(f"shape {shape!r} needs params: {', '.join(accepted)}")
    if not isinstance(params, Mapping):
        raise ValueError(f"shape {shape}: params must be a mapping, not {params!r}")
    where = f"shape {shape}"
    unknown = [key for key in params if key not in accepted]
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(
            f"{where}: unknown param {names}; it takes {', '.join(accepted)}"
        )
    required = [n for n in accepted if parameters[n].default is inspect.Parameter.empty]
    missing = [n for n in required if n not in params]
    if missing:
        raise ValueError(
            f"{where}: missing params {', '.join(repr(n) for n in missing)}"
        )
    given: dict[str, Any] = {}
    for key, value in params.items():
        if key in UNIT_PARAMS:
            given[key] = _param_unit(where, value)
            continue
        if not isinstance(parameters[key].default, bool):  # not include_prompt
            _number_as_text(where, key, value)
        if key.startswith("alpha"):
            _dose(where, key, value)
        given[key] = value
    try:
        return list(build(**given).segments)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def from_mapping(data: Mapping) -> Schedule:
    """Read a schedule from a mapping, such as one inside an experiment file.

    Parse the file with load_yaml: other loaders keep the last of two equal
    keys and read 010 as 8, and from_mapping cannot see either.

    Example:
        >>> schedule = from_mapping(
        ...     {
        ...         "name": "pulse16",
        ...         "shape": "pulse",
        ...         "params": {
        ...             "unit": "qwen3.5-4b:L14:N8949",
        ...             "alpha": 1.0,
        ...             "start": 0,
        ...             "length": 16,
        ...         },
        ...     }
        ... )
        >>> schedule.segments[0].describe()
        'qwen3.5-4b:L14:N8949 at +1 on reply tokens 0 to 15'
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            "a schedule must be a mapping with a name and its segments, or a "
            f"shape with params, not {data!r}"
        )
    _keys(data, SCHEDULE_KEYS, "the schedule")
    name = data.get("name")
    if name is None:
        raise ValueError("the schedule needs a name, such as pulse16")
    if not isinstance(name, str):
        raise ValueError(
            f"the schedule's name must be text, but YAML read {name!r}; "
            "put it in quotes"
        )
    if "segments" in data and ("shape" in data or "params" in data):
        raise ValueError("give the schedule's segments or a shape, not both")
    if "segments" in data:
        listed = data["segments"]
        if not isinstance(listed, list):
            raise ValueError(f"segments must be a list of segments, not {listed!r}")
        segments = [_segment(item, f"segments[{i}]") for i, item in enumerate(listed)]
        refuse_two_definitions(
            (f"segments[{i}]", segment.unit) for i, segment in enumerate(segments)
        )
    elif "shape" in data:
        segments = _from_shape(data["shape"], data.get("params"))
    else:
        raise ValueError("give segments, or a shape with params")
    return Schedule(name, segments, data.get("allow_over_ceiling", False))


def _check_schedule(schedule: object) -> Schedule:
    if not isinstance(schedule, Schedule):
        raise ValueError(f"a {type(schedule).__name__} is not a Schedule")
    return schedule


def to_mapping(schedule: Schedule) -> dict[str, Any]:
    """The schedule as a mapping with every segment written out, ready for YAML.

    A constant segment gets alpha, a ramp alpha_start and alpha_end. Every
    segment gets its start, end and region, and one that repeats its on_for
    and off_for. A segment that pushes a set of neurons also gets the set's
    definition beside its ID (pushlab.schedules.yaml_sets), so from_mapping
    makes the same set again.

    Example:
        >>> from pushlab.schedules.shapes import onset
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> mapping = to_mapping(onset(unit, 1.0, 50))
        >>> mapping["name"], mapping["allow_over_ceiling"]
        ('onset', False)
        >>> mapping["segments"]  # doctest: +NORMALIZE_WHITESPACE
        [{'unit': 'qwen3.5-4b:L14:N8949', 'alpha': 1.0, 'start': 50, 'end': None,
          'region': 'reply'}]
        >>> pair = NeuronSet("qwen3-8b", "pair", [(14, 1, 1.0), (14, 2, -0.5)], "x")
        >>> (segment,) = to_mapping(onset(pair, 1.0, 50))["segments"]
        >>> list(segment)
        ['unit', 'set', 'alpha', 'start', 'end', 'region']
        >>> segment["unit"], segment["set"]  # doctest: +NORMALIZE_WHITESPACE
        ('qwen3-8b:set:pair',
         {'entries': [[14, 1, 1.0], [14, 2, -0.5]], 'size': None, 'source': 'x'})
    """
    segments = []
    for segment in _check_schedule(schedule).segments:
        written: dict[str, Any] = {"unit": str(segment.unit)}
        if isinstance(segment.unit, NeuronSet):
            written["set"] = definition(segment.unit)
        start, end = segment.alpha_start, segment.alpha_end
        # The same bits, so a dose of 0.0 next to -0.0 keeps both signs.
        doses = (
            {"alpha": start}
            if start.hex() == end.hex()
            else {"alpha_start": start, "alpha_end": end}
        )
        window = {"start": segment.start, "end": segment.end, "region": segment.region}
        if segment.repeats:
            window |= {"on_for": segment.on_for, "off_for": segment.off_for}
        segments.append(written | doses | window)
    return {
        "name": schedule.name,
        "allow_over_ceiling": schedule.allow_over_ceiling,
        "segments": segments,
    }


def from_yaml(text: str) -> Schedule:
    """Read a schedule from YAML text.

    Example:
        >>> schedule = from_yaml(
        ...     "name: late\\n"
        ...     "segments:\\n"
        ...     "  - {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 50}\\n"
        ... )
        >>> schedule.name, schedule.segments[0].start, schedule.segments[0].end
        ('late', 50, None)
    """
    return from_mapping(load_yaml(text))


def to_yaml(schedule: Schedule) -> str:
    """The schedule as YAML text, which from_yaml reads back to an equal schedule.

    Example:
        >>> from pushlab.schedules.shapes import pulse
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> print(to_yaml(pulse(unit, 1.0, 0, 16)), end="")
        name: pulse
        allow_over_ceiling: false
        segments:
        - unit: qwen3.5-4b:L14:N8949
          alpha: 1.0
          start: 0
          end: 16
          region: reply
    """
    return dump_yaml(to_mapping(schedule))


def read_schedule(path: str | Path) -> Schedule:
    """Read a schedule file. Its problems are refused with the file's name.

    Example:
        >>> read_schedule("experiments/p0/schedules/pulse16.yaml")  # doctest: +SKIP
        Schedule(name='pulse16', segments=(...), allow_over_ceiling=False)
    """
    path = Path(path)
    try:
        return from_yaml(path.read_text(encoding="utf-8"))
    except ValueError as error:  # UnicodeDecodeError, for a file not in UTF-8, too
        raise ValueError(f"{path}: {error}") from None
