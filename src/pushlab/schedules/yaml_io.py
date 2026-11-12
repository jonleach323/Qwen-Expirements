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

Units in files are neurons. Sets of neurons (decision 0019) come to files in a
later step: until then a set's ID, such as qwen3-8b:set:pirate-top50, is
refused, and so is writing a schedule that pushes a set, which could not be
read back.

The text is read with pushlab.schedules.yaml_text.load_yaml, which refuses a
key given twice and reads numbers and true/false as YAML 1.2 does, and written
with its dump_yaml, so writing then reading changes nothing. An experiment file
that holds a schedule should be read with load_yaml too.
"""

import inspect
import math
import numbers
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import SHAPES
from pushlab.schedules.yaml_text import dump_yaml, load_yaml
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

SCHEDULE_KEYS = ("name", "allow_over_ceiling", "segments", "shape", "params")
SEGMENT_KEYS = (
    "unit",
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
# How a set's ID starts (model:set:), which no neuron's ID can.
_SET_ID = re.compile(r"[^:\s]+:set:")


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


def _unit(where: str, text: object) -> NeuronUnit:
    if isinstance(text, str) and _SET_ID.match(text):
        raise ValueError(
            f"{where}: {text!r} names a set of neurons, and sets cannot be read "
            "from schedule files yet; give a neuron's ID, such as "
            "qwen3.5-4b:L14:N8949, or build a schedule with a set in Python"
        )
    try:
        return NeuronUnit.parse(text)  # type: ignore[arg-type]
    except ValueError as error:
        raise ValueError(f"{where}: unknown unit {text!r}: {error}") from None


def _segment(data: object, where: str) -> Segment:
    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be a mapping such as {{unit: ..., alpha: 1.0}}")
    _keys(data, SEGMENT_KEYS, where)
    if "unit" not in data:
        raise ValueError(f"{where} needs a unit, such as qwen3.5-4b:L14:N8949")
    for key in SEGMENT_KEYS:
        if key not in ("unit", "region"):
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
    unit = _unit(where, data["unit"])
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
            given[key] = _unit(where, value)
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
    and off_for. A schedule that pushes a set of neurons is refused, since
    files name neurons only so far (see the module's docstring).

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
        >>> to_mapping(onset(pair, 1.0, 50))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: segments[0] (qwen3-8b:set:pair at +1 from ...) pushes the set ...
    """
    segments = []
    for i, segment in enumerate(_check_schedule(schedule).segments):
        if isinstance(segment.unit, NeuronSet):
            raise ValueError(
                f"segments[{i}] ({segment.describe()}) pushes the set of neurons "
                f"{segment.unit}, and sets cannot be written to schedule files "
                "yet, since from_yaml could not read them back; keep a schedule "
                "with a set in Python"
            )
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
        segments.append({"unit": str(segment.unit)} | doses | window)
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
