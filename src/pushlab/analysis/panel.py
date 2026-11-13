"""The persona panel's thresholds, from configs/panel.yaml (decision 0022).

A unit's working dose is the lowest dose of its ladder at which at least
working_rate of its replies speak as its persona; its collapse dose, the
lowest at which its replies' mean coherence falls below coherence_below.
Intervals are at confidence, a dose's from resamples of a bootstrap over
prompts, drawn from seed. ngram, window, repetitive_above and varied_below
are the coherence measures that need no judge
(pushlab.scoring.text_measures). Thresholds live in the file, never in
code: read_panel checks every one, refusing a key it does not know.
"""

import math
from dataclasses import dataclass, fields
from pathlib import Path

from pushlab import PROJECT_ROOT
from pushlab.schedules.yaml_text import load_yaml

PANEL_FILE = PROJECT_ROOT / "configs" / "panel.yaml"


@dataclass(frozen=True)
class Panel:
    """configs/panel.yaml's numbers, checked.

    Example:
        >>> panel = read_panel()
        >>> panel.working_rate, panel.coherence_below
        (0.8, 6.0)
    """

    version: int
    working_rate: float
    coherence_below: float
    confidence: float
    resamples: int
    seed: int
    ngram: int
    window: int
    repetitive_above: float
    varied_below: float
    path: Path | None = None


# Each key's kind and allowed range, low and high, both included unless
# said otherwise: (kind, low, high, low included, high included).
_RULES: dict[str, tuple[type, float, float, bool, bool]] = {
    "version": (int, 1, math.inf, True, False),
    "working_rate": (float, 0, 1, False, True),
    "coherence_below": (float, 0, 10, False, True),
    "confidence": (float, 0, 1, False, False),
    "resamples": (int, 100, math.inf, True, False),
    "seed": (int, 0, math.inf, True, False),
    "ngram": (int, 2, math.inf, True, False),
    "window": (int, 10, math.inf, True, False),
    "repetitive_above": (float, 0, 1, False, False),
    "varied_below": (float, 0, 1, False, False),
}


def _checked(name: str, value: object, where: Path) -> float | int:
    kind, low, high, low_in, high_in = _RULES[name]
    is_int = type(value) is int
    if kind is int and not is_int:
        raise ValueError(f"{where}: {name} must be a whole number, not {value!r}")
    if kind is float and not (is_int or type(value) is float):
        raise ValueError(f"{where}: {name} must be a number, not {value!r}")
    number = float(value) if kind is float else value  # type: ignore[arg-type]
    above = number > low or (low_in and number == low)
    below = number < high or (high_in and number == high)
    if not (math.isfinite(number) and above and below):
        span = f"{'from' if low_in else 'above'} {low:g}"
        if math.isfinite(high):
            span += f" {'to' if high_in else 'and below'} {high:g}"
        raise ValueError(f"{where}: {name} must be {span}, not {value!r}")
    return number


def read_panel(path: Path | None = None) -> Panel:
    """configs/panel.yaml, or the file path names, checked.

    ValueError names the problem: a file that is not YAML, a key missing or
    unknown, a value of the wrong kind or out of its range.

    Example:
        >>> read_panel().resamples
        2000
    """
    path = PANEL_FILE if path is None else Path(path)
    try:
        data = load_yaml(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {path}: {error}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping of the panel's numbers")
    names = [field.name for field in fields(Panel) if field.name != "path"]
    if missing := [name for name in names if name not in data]:
        raise ValueError(f"{path} lacks {', '.join(missing)}")
    if unknown := sorted(str(key) for key in data if key not in names):
        raise ValueError(f"{path} has unknown keys: {', '.join(unknown)}")
    values = {name: _checked(name, data[name], path) for name in names}
    return Panel(**values, path=path)  # type: ignore[arg-type]
