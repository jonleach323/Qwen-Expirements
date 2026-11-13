"""The panel file: the personas pushlab panel select chose (decision 0023).

units/panel_<model>.yaml, by default, holds the panel of one model: the
candidates list it was chosen from, with that file's SHA-256, what it was
made from, and one entry per persona, strongest first:

    version: 1
    model: qwen3-8b
    candidates: {file: units/candidates_qwen3-8b.csv, sha256: ...}
    made_from: {confirm: s3-confirm, ladder: s3-ladder, judge: api:..., ...}
    personas:
      - key: library/L9N4242-
        persona: Brass Owl
        category: animals
        kind: neuron
        source: library
        unit: qwen3-8b:L9:N4242
        direction: -1
        reference_dose: 1.5
        working_dose: 1.125
        collapse_dose: 1.875
        reliability: {k: 28, n: 32, rate: 0.875, low: 0.719, high: 0.95}
        coherence: {n: 32, mean: 7.6, low: 7.2, high: 8.0}
        distinct: {k: 8, n: 8, rate: 1.0, low: 0.676, high: 1.0}

key is the candidate's in the candidates list, where a set's neurons are;
working_dose and collapse_dose are doses as a ladder's runs push them, its
direction aside, collapse_dose None (null) when the ladder never collapsed
it. reliability is its speaks_as share at its working dose over the held-out
prompts, coherence its replies' mean there, and distinct the share of its
sampled replies the which_persona rubric named it for; each with its
interval. A panel of the Neuron Persona Library's personas holds its data,
so the file stays out of git (decision 0020; .gitignore).
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pushlab.schedules.yaml_text import dump_yaml, load_yaml

PANEL_FORMAT = 1
KINDS = ("neuron", "set")
HEADER = (
    "# A persona panel, written by pushlab panel select (decision 0023).\n"
    "# It may hold the Neuron Persona Library's data, which states no\n"
    "# licence: keep it out of git (decision 0020).\n"
)


@dataclass(frozen=True)
class Share:
    """k of n, their rate, and its interval, low to high.

    Example:
        >>> Share(28, 32, 0.875, 0.72, 0.95).rate
        0.875
    """

    k: int
    n: int
    rate: float
    low: float
    high: float


@dataclass(frozen=True)
class Mean:
    """The mean of n values, and its interval, low to high."""

    n: int
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class PanelPersona:
    """One persona of the panel, as the module's docstring describes it."""

    key: str
    persona: str
    category: str
    kind: str
    source: str
    unit: str
    direction: int
    reference_dose: float
    working_dose: float
    collapse_dose: float | None
    reliability: Share
    coherence: Mean
    distinct: Share


@dataclass(frozen=True)
class PanelFile:
    """A whole panel: its model, the candidates list it came from, what it
    was made from (each a name and a text), and its personas in order."""

    model: str
    candidates_file: str
    candidates_sha256: str
    made_from: dict[str, str]
    personas: tuple[PanelPersona, ...]


def _number(where: str, value: object, *, whole: bool = False) -> Any:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where} must be a number, not {value!r}")
    if whole and type(value) is not int:
        raise ValueError(f"{where} must be a whole number, not {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite, not {value!r}")
    return value if whole else float(value)


def _text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be text, not {value!r}")
    return value


def _mapping(where: str, value: object, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{where} must have exactly {', '.join(keys)}")
    return value


def _share(where: str, value: object) -> Share:
    data = _mapping(where, value, ("k", "n", "rate", "low", "high"))
    k, n = (_number(f"{where}.{key}", data[key], whole=True) for key in ("k", "n"))
    if not 0 <= k <= n or n < 1:
        raise ValueError(f"{where} needs 0 <= k <= n and n >= 1, not {k} of {n}")
    rate, low, high = (
        _number(f"{where}.{key}", data[key]) for key in ("rate", "low", "high")
    )
    return Share(k, n, rate, low, high)


def _mean(where: str, value: object) -> Mean:
    data = _mapping(where, value, ("n", "mean", "low", "high"))
    n = _number(f"{where}.n", data["n"], whole=True)
    mean, low, high = (
        _number(f"{where}.{key}", data[key]) for key in ("mean", "low", "high")
    )
    return Mean(n, mean, low, high)


def _persona(where: str, value: object) -> PanelPersona:
    names = [name for name in PanelPersona.__dataclass_fields__]
    data = _mapping(where, value, tuple(names))
    kind = data["kind"]
    if kind not in KINDS:
        raise ValueError(f"{where}.kind must be neuron or set, not {kind!r}")
    if data["direction"] not in (1, -1) or type(data["direction"]) is not int:
        raise ValueError(f"{where}.direction must be 1 or -1")
    collapse = data["collapse_dose"]
    return PanelPersona(
        key=_text(f"{where}.key", data["key"]),
        persona=_text(f"{where}.persona", data["persona"]),
        category=_text(f"{where}.category", data["category"]),
        kind=kind,
        source=_text(f"{where}.source", data["source"]),
        unit=_text(f"{where}.unit", data["unit"]),
        direction=data["direction"],
        reference_dose=_number(f"{where}.reference_dose", data["reference_dose"]),
        working_dose=_number(f"{where}.working_dose", data["working_dose"]),
        collapse_dose=None
        if collapse is None
        else _number(f"{where}.collapse_dose", collapse),
        reliability=_share(f"{where}.reliability", data["reliability"]),
        coherence=_mean(f"{where}.coherence", data["coherence"]),
        distinct=_share(f"{where}.distinct", data["distinct"]),
    )


def panel_mapping(panel: PanelFile) -> dict[str, Any]:
    """The panel as plain data, as the file holds it."""
    return {
        "version": PANEL_FORMAT,
        "model": panel.model,
        "candidates": {
            "file": panel.candidates_file,
            "sha256": panel.candidates_sha256,
        },
        "made_from": dict(panel.made_from),
        "personas": [asdict(persona) for persona in panel.personas],
    }


def panel_from_mapping(data: object, where: str = "the panel") -> PanelFile:
    """A panel from plain data, every field checked.

    ValueError names the first problem: a key missing or unknown, a value of
    the wrong kind, a key given to two personas.
    """
    keys = ("version", "model", "candidates", "made_from", "personas")
    data = _mapping(where, data, keys)
    if data["version"] != PANEL_FORMAT:
        raise ValueError(
            f"{where}: version must be {PANEL_FORMAT}, not {data['version']!r}"
        )
    candidates = _mapping(f"{where}.candidates", data["candidates"], ("file", "sha256"))
    made_from = data["made_from"]
    if not isinstance(made_from, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in made_from.items()
    ):
        raise ValueError(f"{where}.made_from must map names to texts")
    if not isinstance(data["personas"], list):
        raise ValueError(f"{where}.personas must be a list")
    personas = tuple(
        _persona(f"{where}.personas[{i}]", item)
        for i, item in enumerate(data["personas"])
    )
    seen: set[str] = set()
    for persona in personas:
        if persona.key in seen:
            raise ValueError(f"{where}: {persona.key} is in the panel twice")
        seen.add(persona.key)
    return PanelFile(
        _text(f"{where}.model", data["model"]),
        _text(f"{where}.candidates.file", candidates["file"]),
        _text(f"{where}.candidates.sha256", candidates["sha256"]),
        dict(made_from),
        personas,
    )


def write_panel_file(path: Path, panel: PanelFile) -> Path:
    """Write the panel to path, with a header saying it stays out of git,
    making its folder; the path.

    Example:
        >>> write_panel_file(Path("units/panel_qwen3-8b.yaml"), panel)  # doctest: +SKIP
        PosixPath('units/panel_qwen3-8b.yaml')
    """
    path = Path(path)
    text = HEADER + dump_yaml(panel_mapping(panel))
    if panel_from_mapping(load_yaml(text)) != panel:  # a bug, never the user's
        raise ValueError("the panel does not read back as written")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def read_panel_file(path: Path) -> PanelFile:
    """The panel in the file at path, checked; ValueError says what is wrong.

    Example:
        >>> read_panel_file(Path("units/panel_qwen3-8b.yaml")).model  # doctest: +SKIP
        'qwen3-8b'
    """
    path = Path(path)
    try:
        data = load_yaml(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the panel {path}: {error}") from None
    return panel_from_mapping(data, str(path))
