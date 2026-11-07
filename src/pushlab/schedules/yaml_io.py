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
reply. allow_over_ceiling: true lets validate() accept a combined push above
the dose ceiling. Unknown keys, params, shapes and units are refused with a
message that names them, and so is a key given twice. Schedules are always
written out as segments, every field given, so reading one back gives an
equal schedule, every dose bit for bit.
"""

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import SHAPES
from pushlab.units.neuron import NeuronUnit

SCHEDULE_KEYS = ("name", "allow_over_ceiling", "segments", "shape", "params")
SEGMENT_KEYS = ("unit", "alpha", "alpha_start", "alpha_end", "start", "end", "region")
# The shapes' params that name a unit, as text such as qwen3.5-4b:L14:N8949.
UNIT_PARAMS = ("unit", "a", "b")


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML's safe loader, except that a key given twice is refused.

    PyYAML keeps the last of two equal keys without a word, so a file that
    gives alpha twice would push the second dose.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)  # resolves merge keys (<<) first, as PyYAML does
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                repeated = key in seen
            except TypeError:  # an unhashable key, which PyYAML refuses itself
                continue
            if repeated:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _keys(given: Mapping, allowed: tuple[str, ...], where: str) -> None:
    unknown = [key for key in given if key not in allowed]
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{where}: unknown key {names}; it takes {', '.join(allowed)}")


def _number_as_text(where: str, key: str, value: object) -> None:
    """Refuse a number YAML read as text, such as 1e-1, with the reason."""
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return
        raise ValueError(
            f"{where}: {key} is the text {value!r}, not a number; YAML reads a "
            "quoted number, or one such as 1e-1 with no decimal point, as text: "
            "write 0.1 or 1.0e-1"
        )


def _unit(where: str, text: object) -> NeuronUnit:
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
    for key in ("alpha", "alpha_start", "alpha_end", "start", "end"):
        _number_as_text(where, key, data.get(key))
    if "alpha" in data:
        if "alpha_start" in data or "alpha_end" in data:
            raise ValueError(
                f"{where}: give alpha or alpha_start and alpha_end, not both"
            )
        alpha_start = alpha_end = data["alpha"]
    elif "alpha_start" in data and "alpha_end" in data:
        alpha_start, alpha_end = data["alpha_start"], data["alpha_end"]
    elif "alpha_start" in data or "alpha_end" in data:
        raise ValueError(f"{where}: give alpha_start and alpha_end together")
    else:
        raise ValueError(f"{where} needs alpha, or alpha_start and alpha_end")
    try:
        return Segment(
            _unit(where, data["unit"]),
            alpha_start,
            alpha_end,
            data.get("start", 0),
            data.get("end"),
            data.get("region", "reply"),
        )
    except ValueError as error:
        if str(error).startswith(where):
            raise
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
        else:
            _number_as_text(where, key, value)
            given[key] = value
    try:
        return list(build(**given).segments)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def from_mapping(data: Mapping) -> Schedule:
    """Read a schedule from a mapping, such as one inside an experiment file.

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
    if "name" not in data:
        raise ValueError("the schedule needs a name, such as pulse16")
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
    return Schedule(data["name"], segments, data.get("allow_over_ceiling", False))


def _check_schedule(schedule: object) -> Schedule:
    if not isinstance(schedule, Schedule):
        raise ValueError(f"{schedule!r} is not a Schedule")
    return schedule


def to_mapping(schedule: Schedule) -> dict[str, Any]:
    """The schedule as a mapping with every segment written out, ready for YAML.

    A constant segment gets alpha, a ramp alpha_start and alpha_end; every
    segment gets its start, end and region.

    Example:
        >>> from pushlab.schedules.shapes import onset
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> mapping = to_mapping(onset(unit, 1.0, 50))
        >>> mapping["name"], mapping["allow_over_ceiling"]
        ('onset', False)
        >>> mapping["segments"]  # doctest: +NORMALIZE_WHITESPACE
        [{'unit': 'qwen3.5-4b:L14:N8949', 'alpha': 1.0, 'start': 50, 'end': None,
          'region': 'reply'}]
    """
    segments = []
    for segment in _check_schedule(schedule).segments:
        if segment.alpha_start == segment.alpha_end:
            doses = {"alpha": segment.alpha_start}
        else:
            doses = {"alpha_start": segment.alpha_start, "alpha_end": segment.alpha_end}
        segments.append(
            {"unit": str(segment.unit)}
            | doses
            | {"start": segment.start, "end": segment.end, "region": segment.region}
        )
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
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # safe: a SafeLoader
    except yaml.YAMLError as error:
        raise ValueError(f"the schedule is not valid YAML: {error}") from None
    return from_mapping(data)


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
    return yaml.safe_dump(to_mapping(schedule), sort_keys=False)


def read_schedule(path: str | Path) -> Schedule:
    """Read a schedule file. Its problems are refused with the file's name.

    Example:
        >>> read_schedule("experiments/p0/schedules/pulse16.yaml")  # doctest: +SKIP
        Schedule(name='pulse16', segments=(...), allow_over_ceiling=False)
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return from_yaml(text)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
