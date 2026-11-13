"""How pushlab panel narrows the candidates down, from configs/panel.yaml's
selection section (decision 0023).

The stages run on one held-out prompt set (prompts), greedy, max_new_tokens
tokens a reply, batch_size runs a batch. screen_a runs every candidate at its
reference dose on the set's first screen_prompts prompts; stage b the
stage_b_size best on every prompt; stage ladder stage b's ladder_size best at
ladder_doses, multiples of each candidate's reference dose, on the first
ladder_prompts prompts; stage confirm each that works at its working dose on
every prompt. select keeps at most panel_size personas, and writes none when
fewer than panel_minimum pass; each must be named by the which_persona rubric
for at least distinct_rate of distinct_sample of its replies, and the panel
spreads across categories. stage preview pushes preview_pairs pairs in four
shapes, on the set's last preview_prompts prompts: a handoff at reply token
handoff_at, alternation every alternate_every tokens, a pulse of the first
for pulse_length tokens, and the second from reply token onset_at. read_selection
checks every value and refuses a key it does not know.
"""

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

from pushlab.analysis.panel import PANEL_FILE
from pushlab.schedules.yaml_text import load_yaml

# A category's name: lowercase letters.
CATEGORY = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Selection:
    """configs/panel.yaml's selection section, checked.

    Example:
        >>> selection = read_selection()
        >>> selection.stage_b_size, selection.ladder_size, selection.panel_size
        (120, 40, 24)
    """

    prompts: str
    max_new_tokens: int
    batch_size: int
    screen_prompts: int
    stage_b_size: int
    ladder_size: int
    ladder_prompts: int
    ladder_doses: tuple[float, ...]
    panel_size: int
    panel_minimum: int
    distinct_rate: float
    distinct_sample: int
    categories: tuple[str, ...]
    preview_pairs: int
    preview_prompts: int
    handoff_at: int
    alternate_every: int
    pulse_length: int
    onset_at: int


# The whole numbers and each one's least value.
_WHOLE = {
    "max_new_tokens": 1,
    "batch_size": 1,
    "screen_prompts": 1,
    "stage_b_size": 1,
    "ladder_size": 1,
    "ladder_prompts": 1,
    "panel_size": 1,
    "panel_minimum": 1,
    "distinct_sample": 1,
    "preview_pairs": 1,
    "preview_prompts": 1,
    "handoff_at": 1,
    "alternate_every": 1,
    "pulse_length": 1,
    "onset_at": 1,
}


def _whole(where: str, name: str, value: object) -> int:
    if type(value) is not int or value < _WHOLE[name]:
        raise ValueError(
            f"{where}: {name} must be a whole number from {_WHOLE[name]}, not {value!r}"
        )
    return value


def _doses(where: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: ladder_doses must be a list of numbers")
    doses = []
    for dose in value:
        if type(dose) not in (int, float) or not (math.isfinite(dose) and dose > 0):
            raise ValueError(f"{where}: ladder_doses must be numbers above 0: {dose!r}")
        doses.append(float(dose))
    if doses != sorted(set(doses)):
        raise ValueError(f"{where}: ladder_doses must increase, each once: {doses}")
    return tuple(doses)


def _categories(where: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{where}: categories must list two or more names")
    for name in value:
        if not isinstance(name, str) or not CATEGORY.fullmatch(name):
            raise ValueError(
                f"{where}: a category's name must be lowercase letters: {name!r}"
            )
    if len(set(value)) != len(value):
        raise ValueError(f"{where}: categories must each be named once: {value}")
    return tuple(value)


def read_selection(path: Path | None = None) -> Selection:
    """configs/panel.yaml's selection section, or that of the file path
    names, checked.

    ValueError names the problem: a missing section or key, an unknown key,
    a value of the wrong kind or out of range, a panel_minimum above
    panel_size, or prompt counts the stages cannot share.

    Example:
        >>> read_selection().ladder_doses
        (0.5, 0.75, 1.0, 1.25, 1.5)
    """
    path = PANEL_FILE if path is None else Path(path)
    try:
        data = load_yaml(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {path}: {error}") from None
    section = data.get("selection") if isinstance(data, dict) else None
    where = f"{path}, selection"
    if not isinstance(section, dict):
        raise ValueError(f"{path} has no selection section (decision 0023)")
    names = [field.name for field in fields(Selection)]
    if missing := [name for name in names if name not in section]:
        raise ValueError(f"{where} lacks {', '.join(missing)}")
    if unknown := sorted(str(key) for key in section if key not in names):
        raise ValueError(f"{where} has unknown keys: {', '.join(unknown)}")
    values: dict[str, object] = {name: section[name] for name in names}
    if not isinstance(values["prompts"], str) or not values["prompts"].strip():
        raise ValueError(f"{where}: prompts must be a prompt set's path")
    for name in _WHOLE:
        values[name] = _whole(where, name, values[name])
    values["ladder_doses"] = _doses(where, values["ladder_doses"])
    values["categories"] = _categories(where, values["categories"])
    rate = values["distinct_rate"]
    if type(rate) not in (int, float) or not 0 < rate <= 1:
        raise ValueError(f"{where}: distinct_rate must be above 0 to 1, not {rate!r}")
    values["distinct_rate"] = float(rate)
    selection = Selection(**values)  # type: ignore[arg-type]
    if selection.panel_minimum > selection.panel_size:
        raise ValueError(
            f"{where}: panel_minimum, {selection.panel_minimum}, is above "
            f"panel_size, {selection.panel_size}"
        )
    if selection.stage_b_size < selection.ladder_size:
        raise ValueError(f"{where}: ladder_size is above stage_b_size")
    return selection
