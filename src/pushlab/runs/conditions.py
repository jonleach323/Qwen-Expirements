"""Conditions of an experiment file: a label, its schedules, and its controls.

A condition's schedule is written as a schedule file writes one
(pushlab.schedules.yaml_io): its segments, or a shape and its params. A grid
gives some of a shape's params as lists of values, and the condition then
holds one schedule per combination of values, each labelled with its
values, such as `pulse length=16`; the first param changes slowest.
controls asks for that many random controls of each run, drawn from
control_seed by pushlab.runs.expand. experiments/README.md has the format.
"""

import itertools
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pushlab.schedules.model import SCHEDULE_NAME, Schedule
from pushlab.schedules.yaml_io import from_mapping
from pushlab.store.records import MAX_SEED

CONDITION_KEYS = (
    "label",
    "segments",
    "shape",
    "params",
    "grid",
    "allow_over_ceiling",
    "controls",
    "control_seed",
)
# The most runs an experiment can have: run IDs count with six digits.
MOST_RUNS = 999_999


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def not_text(what: str, value: object, example: str) -> str:
    """Why value is not the text it should be, with a hint to quote a number
    or true or false that YAML read as one.

    Example:
        >>> not_text("label", 16, "pulse-16")
        'label must be text, such as pulse-16, but YAML read 16; put it in quotes'
    """
    if isinstance(value, numbers.Real):  # true and false too
        return (
            f"{what} must be text, such as {example}, but YAML read {value!r}; "
            "put it in quotes"
        )
    return f"{what} must be text, such as {example}, not {value!r}"


def refuse_unknown_keys(given: Mapping, allowed: tuple[str, ...], where: str) -> None:
    """Raise ValueError naming every key of given that is not allowed.

    where starts the message, such as "conditions[1] (pulse): ".

    Example:
        >>> refuse_unknown_keys({"colour": "red"}, ("label",), "conditions[0] (x): ")
        Traceback (most recent call last):
        ...
        ValueError: conditions[0] (x): unknown key 'colour'; it takes label
    """
    if unknown := [key for key in given if key not in allowed]:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{where}unknown key {names}; it takes {', '.join(allowed)}")


@dataclass(frozen=True)
class Condition:
    """One condition: its label, its schedules, and how many random controls.

    points holds a (label, schedule) pair for each grid value, in order, or
    one pair, the condition's own label, when it has no grid. Each schedule
    is named after the condition. controls is how many random controls each
    of its runs gets, drawn from control_seed (None when controls is 0).

    Example:
        >>> condition = Condition("none", (("none", Schedule("none")),))
        >>> condition.controls, condition.control_seed
        (0, None)
    """

    label: str
    points: tuple[tuple[str, Schedule], ...]
    controls: int = 0
    control_seed: int | None = None


def _value_text(param: object, value: object) -> str:
    """A grid value as its point's label shows it: true, false and null as
    YAML writes them, and a dose as the float its schedule holds, so that
    1 and 1.0 give one label."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    dose = isinstance(param, str) and param.startswith("alpha")
    if dose and isinstance(value, numbers.Real):  # checked finite by now
        value = float(value)
    return repr(value) if isinstance(value, float) else str(value)


def _grid_points(
    label: str, schedule: dict[str, Any], grid: object, runs_per_point: int
) -> tuple[tuple[str, Schedule], ...]:
    """One (label, schedule) pair per combination of grid values, the first
    param varying slowest. Each point makes runs_per_point runs for every
    prompt and seed: a grid too large to number them is refused unbuilt."""
    if "shape" not in schedule:
        segments = ", not with segments" if "segments" in schedule else ""
        raise ValueError(f"a grid goes with a shape and its params{segments}")
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError(
            "grid must be a mapping of params to lists of values, such as "
            f"{{length: [4, 16, 64]}}, not {grid!r}"
        )
    params = schedule.get("params")
    params = {} if params is None else params
    if not isinstance(params, Mapping):
        raise ValueError(
            f"shape {schedule['shape']}: params must be a mapping, not {params!r}"
        )
    for key, values in grid.items():
        if key in params:
            raise ValueError(f"{key} is in both params and grid; give it in one")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"grid {key} must be a list of values, such as [4, 16, 64], "
                f"not {values!r}"
            )
        for i, value in enumerate(values):
            if any(value == earlier for earlier in values[:i]):
                raise ValueError(f"grid {key} repeats {value!r}")
    count = math.prod(len(values) for values in grid.values())
    if count * runs_per_point > MOST_RUNS:
        raise ValueError(
            f"its grid makes {count:,} schedules, so {count * runs_per_point:,} "
            f"runs for each prompt and seed, more than the {MOST_RUNS:,} that "
            "six-digit run IDs can number"
        )
    points: dict[str, Schedule] = {}
    for values in itertools.product(*grid.values()):
        point = dict(zip(grid, values, strict=True))
        mapping = schedule | {"name": label, "params": {**params, **point}}
        built = from_mapping(mapping)  # before the label, which needs good values
        text = "".join(f" {key}={_value_text(key, v)}" for key, v in point.items())
        if label + text in points:  # two doses that are one float
            raise ValueError(f"two grid points are both {label + text!r}")
        points[label + text] = built
    return tuple(points.items())


def condition_from_mapping(data: object, i: int) -> Condition:
    """Check conditions[i] of an experiment file and return the condition.

    Every problem is a ValueError that names the condition by its place and
    label. Whether its schedules suit a model is for pushlab.runs.expand.

    Example:
        >>> condition = condition_from_mapping(
        ...     {
        ...         "label": "pulse",
        ...         "shape": "pulse",
        ...         "params": {
        ...             "unit": "qwen3.5-4b:L14:N8949",
        ...             "alpha": 1.0,
        ...             "start": 0,
        ...         },
        ...         "grid": {"length": [4, 16]},
        ...     },
        ...     1,
        ... )
        >>> [label for label, _ in condition.points]
        ['pulse length=4', 'pulse length=16']
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"conditions[{i}] must be a mapping with a label and a schedule, "
            f"not {data!r}"
        )
    label = data.get("label")
    if label is None:
        raise ValueError(f"conditions[{i}] needs a label, such as pulse-16")
    if not isinstance(label, str):
        raise ValueError(f"conditions[{i}]: {not_text('label', label, 'pulse-16')}")
    if not SCHEDULE_NAME.fullmatch(label):
        raise ValueError(
            f"conditions[{i}]: label must be lowercase letters and digits joined "
            f"by single hyphens, such as pulse-16, not {label!r}"
        )
    where = f"conditions[{i}] ({label}): "
    refuse_unknown_keys(data, CONDITION_KEYS, where)
    controls = data.get("controls", 0)
    if not _is_whole(controls) or controls < 0:
        raise ValueError(
            f"{where}controls must be a whole number from 0, not {controls!r}"
        )
    seed = data.get("control_seed")
    if controls and seed is None:
        need = "1 control needs" if controls == 1 else f"{controls} controls need"
        raise ValueError(
            f"{where}{need} a control_seed, the seed their neurons are drawn from"
        )
    if not controls and "control_seed" in data:
        raise ValueError(f"{where}control_seed goes with controls of 1 or more")
    if seed is not None and not (_is_whole(seed) and 0 <= seed <= MAX_SEED):
        raise ValueError(
            f"{where}control_seed must be a whole number from 0 to {MAX_SEED}, "
            f"not {seed!r}"
        )
    schedule = {
        key: data[key] for key in ("segments", "shape", "params") if key in data
    }
    if "allow_over_ceiling" in data:
        schedule["allow_over_ceiling"] = data["allow_over_ceiling"]
    try:
        if "grid" in data:
            points = _grid_points(label, schedule, data["grid"], 1 + controls)
        else:
            points = ((label, from_mapping(schedule | {"name": label})),)
    except ValueError as error:
        raise ValueError(f"{where}{error}") from None
    if controls and not all(built.units() for _, built in points):
        raise ValueError(
            f"{where}it pushes no unit, so it cannot have controls; remove "
            "controls and control_seed"
        )
    return Condition(label, points, controls, seed)
