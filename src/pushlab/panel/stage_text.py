"""A stage's experiment file as text (decision 0023): its settings as YAML,
with comments that say what it is, for pushlab panel stage.

stage_text writes a Stage's settings (pushlab.panel.stage_files) so that
load_yaml reads them back exactly: each on its own line, prompt_ids, the
candidates' keys and the conditions as lists, a condition's segments one to
a line, and every value on one line by flow, plain where that is safe and
in double quotes where not. Comments go above the settings, above
max_gpu_hours, beside each candidate's key and above conditions. Nothing
here reads or writes a file.
"""

import math
import re
import textwrap
from collections.abc import Mapping
from typing import Any

from pushlab.panel.stage_files import Stage

# Text that YAML reads as itself without quotes, in flow and block alike.
_PLAIN = re.compile(r"[A-Za-z/][A-Za-z0-9_./+-]*(?::[A-Za-z0-9_./+-]+)*")
# Words some YAML readers take for true, false or null: always quoted.
_WORDS = frozenset({"true", "false", "null", "yes", "no", "on", "off", "y", "n"})


def _quoted(text: str) -> str:
    """text in double quotes, every character outside printable ASCII
    escaped, as YAML reads it back."""
    shown = []
    for character in text:
        code = ord(character)
        if character in '"\\':
            shown.append("\\" + character)
        elif 0x20 <= code < 0x7F:
            shown.append(character)
        elif code <= 0xFF:
            shown.append(f"\\x{code:02x}")
        elif code <= 0xFFFF:
            shown.append(f"\\u{code:04x}")
        else:
            shown.append(f"\\U{code:08x}")
    return '"' + "".join(shown) + '"'


def flow(value: object) -> str:
    """value as YAML on one line, which load_yaml reads back as it is: text
    plain when that is safe, else in double quotes; floats by their repr.

    Example:
        >>> flow({"unit": "qwen3-8b:L9:N4242", "alpha": -1.125, "end": None})
        '{unit: qwen3-8b:L9:N4242, alpha: -1.125, end: null}'
        >>> flow(["library/L9N4242-", "a, b", "true", "1e3"])
        '[library/L9N4242-, "a, b", "true", "1e3"]'
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"an experiment file holds finite numbers, not {value!r}")
        return repr(value)
    if isinstance(value, str):
        plain = _PLAIN.fullmatch(value) and value.casefold() not in _WORDS
        return value if plain else _quoted(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{flow(k)}: {flow(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(flow(item) for item in value) + "]"
    raise TypeError(f"cannot write a {type(value).__name__} in an experiment file")


def plain(text: str) -> str:
    """text for a comment: on one line, with only printable characters.

    Example:
        >>> plain("Brass\\nOwl  ")
        'Brass Owl'
    """
    shown = "".join(c if c.isprintable() else " " for c in str(text))
    return " ".join(shown.split())


def comment(text: str, indent: str = "") -> list[str]:
    """text as comment lines of at most about 79 characters.

    Example:
        >>> comment("Stage b of the persona panel.", "  ")
        ['  # Stage b of the persona panel.']
    """
    lines = textwrap.wrap(
        plain(text), 77 - len(indent), break_long_words=False, break_on_hyphens=False
    )
    return [f"{indent}# {line}" for line in lines]


def _beside(line: str, note: str) -> str:
    """line with note as a comment at its end, if there is one."""
    return f"{line}  # {plain(note)}" if plain(note) else line


def _condition_lines(condition: Mapping[str, Any]) -> list[str]:
    """A condition as an item of conditions:, its segments one to a line."""
    lines = []
    for place, (key, value) in enumerate(condition.items()):
        head = "  - " if place == 0 else "    "
        if key == "segments" and value:
            lines.append(f"{head}segments:")
            lines += [f"      - {flow(segment)}" for segment in value]
        else:
            lines.append(f"{head}{key}: {flow(value)}")
    return lines


def stage_text(stage: Stage, mapping: Mapping[str, Any], hours_note: str = "") -> str:
    """The experiment file: stage.about as comments, then mapping's settings,
    with hours_note above max_gpu_hours, stage.key_notes beside the
    candidates' keys and stage.condition_notes above the conditions.
    load_yaml reads it back as mapping.

    Example:
        >>> stage = Stage(
        ...     "qwen3-8b", None, None, ({"label": "none", "segments": []},), 1, ""
        ... )
        >>> print(
        ...     stage_text(stage, {"name": "x", "conditions": list(stage.conditions)})
        ... )
        name: x
        conditions:
          - label: none
            segments: []
        <BLANKLINE>
    """
    lines = comment(stage.about) if stage.about else []
    for key, value in mapping.items():
        if key == "max_gpu_hours" and hours_note:
            lines += comment(hours_note)
        if key == "prompt_ids" and value:
            lines += ["prompt_ids:", *(f"  - {flow(prompt)}" for prompt in value)]
        elif key == "candidates":
            lines.append("candidates:")
            for name, setting in value.items():
                if name != "keys" or not setting:
                    lines.append(f"  {name}: {flow(setting)}")
                    continue
                notes = list(stage.key_notes) + [""] * len(setting)
                lines.append("  keys:")
                lines += [
                    _beside(f"    - {flow(item)}", note)
                    for item, note in zip(setting, notes, strict=False)
                ]
        elif key == "conditions" and value:
            lines.append("conditions:")
            for place, condition in enumerate(value):
                if note := stage.condition_notes.get(place):
                    lines += comment(note, "  ")
                lines += _condition_lines(condition)
        else:
            lines.append(f"{key}: {flow(value)}")
    return "\n".join(lines) + "\n"
