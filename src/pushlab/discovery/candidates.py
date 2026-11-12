"""Persona candidates: what a source offers, one unit each, and their CSV file.

A Candidate is a unit that a source says gives a persona: a neuron from the
Neuron Persona Library, or a set of neurons from Mayner's persona-neurons
data. It keeps the persona's name, the direction and strength the source
pushes it at, whether the source calls it sensitive, the source's own
scores, and where it came from: the source, the file inside the data
folder, the place in that file, the file's URL and the day it was
downloaded (decision 0020).

write_candidates saves candidates as CSV, one row each, with the source's
licence and credit on every row; read_candidates reads them back, units and
all.
"""

import csv
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from pushlab.discovery.sources import SOURCES
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit, is_unit

COLUMNS = (
    "candidate",
    "source",
    "persona",
    "unit",
    "kind",
    "neurons",
    "direction",
    "strength",
    "sensitive",
    "category",
    "tags",
    "coherence",
    "weirdness",
    "verified_pool",
    "size",
    "size_from",
    "entries",
    "mayner_in_role",
    "mayner_scored",
    "mayner_share",
    "source_file",
    "source_ref",
    "source_url",
    "downloaded_at",
    "licence",
    "credit",
)


def _optional(value: object, kind: type, name: str) -> None:
    if value is not None and (not isinstance(value, kind) or isinstance(value, bool)):
        what = "an int" if kind is int else f"a {kind.__name__}"
        raise ValueError(f"{name} must be None or {what}, not {value!r}")


@dataclass(frozen=True)
class Candidate:
    """One unit a source offers as a persona, with what the source says of it.

    source is "library" or "mayner"; persona the name the source gives it;
    unit a NeuronUnit or a NeuronSet; direction +1 or -1, the sign of the
    push that gives the persona; strength the source's own dose for it (the
    library's alpha; 1.0 for Mayner's sets, his strength). source_file is
    the file it was read from, inside the data folder, and source_ref the
    place in it; for a set, source_ref is the set's own source. The rest are
    the source's labels and scores, None or empty where it has none.

    Example:
        >>> cat = Candidate(
        ...     "library",
        ...     "Cat",
        ...     NeuronUnit("qwen3-8b", 9, 4242),
        ...     -1,
        ...     1.0,
        ...     False,
        ...     "library/qwen/library.json",
        ...     "L9 N4242 s-1",
        ... )
        >>> cat.key, cat.kind
        ('library/L9N4242-', 'neuron')
    """

    source: str
    persona: str
    unit: Unit
    direction: int
    strength: float
    sensitive: bool
    source_file: str
    source_ref: str
    source_url: str = ""
    downloaded_at: str = ""
    category: str = ""
    tags: tuple[str, ...] = ()
    coherence: int | None = None
    weirdness: int | None = None
    verified_pool: bool | None = None
    size_from: str = ""
    mayner_in_role: float | None = None
    mayner_scored: int | None = None
    mayner_share: float | None = None

    def __post_init__(self) -> None:
        """Refuse a field that is not what the class's docstring says."""
        if self.source not in SOURCES:
            raise ValueError(
                f"source must be one of {', '.join(SOURCES)}, not {self.source!r}"
            )
        if not isinstance(self.persona, str) or not self.persona.strip():
            raise ValueError(f"persona must be a name, not {self.persona!r}")
        if not is_unit(self.unit):
            raise ValueError(f"unit must be a NeuronUnit or a NeuronSet: {self.unit!r}")
        if self.direction not in (1, -1) or isinstance(self.direction, bool):
            raise ValueError(f"direction must be +1 or -1, not {self.direction!r}")
        strength = self.strength
        if isinstance(strength, bool) or not isinstance(strength, int | float):
            raise ValueError(f"strength must be a number, not {strength!r}")
        if not (math.isfinite(strength) and strength > 0):
            raise ValueError(f"strength must be finite and above 0, not {strength!r}")
        object.__setattr__(self, "strength", float(strength))
        for name in ("sensitive",):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False")
        for name in ("source_file", "source_ref", "source_url", "downloaded_at"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be text, not {getattr(self, name)!r}")
        if not self.source_file or not self.source_ref:
            raise ValueError("source_file and source_ref must say where it came from")
        if isinstance(self.unit, NeuronSet) and self.source_ref != self.unit.source:
            raise ValueError("a set's source_ref must be the set's own source")
        tags = tuple(self.tags)
        if not all(isinstance(tag, str) and tag for tag in tags):
            raise ValueError(f"tags must be words, not {self.tags!r}")
        object.__setattr__(self, "tags", tags)
        _optional(self.coherence, int, "coherence")
        _optional(self.weirdness, int, "weirdness")
        _optional(self.mayner_scored, int, "mayner_scored")
        _optional(self.mayner_in_role, float, "mayner_in_role")
        _optional(self.mayner_share, float, "mayner_share")
        if self.verified_pool is not None and not isinstance(self.verified_pool, bool):
            raise ValueError("verified_pool must be None, True or False")

    @property
    def kind(self) -> str:
        """ "neuron" or "set"."""
        return "set" if isinstance(self.unit, NeuronSet) else "neuron"

    @property
    def key(self) -> str:
        """The candidate's name in the list: source/L<layer>N<neuron><sign> for a
        neuron, as the library writes its IDs, or source/<set name>."""
        sign = "+" if self.direction > 0 else "-"
        if isinstance(self.unit, NeuronSet):
            return f"{self.source}/{self.unit.name}" + ("" if sign == "+" else sign)
        return f"{self.source}/L{self.unit.layer}N{self.unit.index}{sign}"


def same_name(name: str) -> str:
    """A persona's name as duplicates are found: case and spacing ignored.

    Example:
        >>> same_name("  Lord   God ") == same_name("lord god")
        True
    """
    return re.sub(r"\s+", " ", name).strip().casefold()


def _rank(candidate: Candidate) -> tuple:
    """Which of two candidates of one name is kept: verified first, then the
    more coherent, the weirder, the stronger, then by key."""
    return (
        candidate.verified_pool is not True,
        -(candidate.coherence if candidate.coherence is not None else -1),
        -(candidate.weirdness if candidate.weirdness is not None else -1),
        -candidate.strength,
        candidate.key,
    )


def drop_duplicates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """One candidate per name and size within each source, in the input's order.

    Candidates of one source whose personas have the same name (same_name)
    and whose units have as many neurons are duplicates; the best of them is
    kept (_rank), so a verified one wins, then the most coherent.

    Example:
        >>> a = Candidate(
        ...     "library",
        ...     "Cat",
        ...     NeuronUnit("qwen3-8b", 9, 1),
        ...     1,
        ...     1.0,
        ...     False,
        ...     "f",
        ...     "r",
        ...     coherence=7,
        ... )
        >>> b = Candidate(
        ...     "library",
        ...     "cat",
        ...     NeuronUnit("qwen3-8b", 9, 2),
        ...     1,
        ...     1.0,
        ...     False,
        ...     "f",
        ...     "r",
        ...     coherence=9,
        ... )
        >>> [c.key for c in drop_duplicates([a, b])]
        ['library/L9N2+']
    """
    candidates = list(candidates)
    best: dict[tuple, Candidate] = {}
    for candidate in candidates:
        size = len(candidate.unit.entries) if candidate.kind == "set" else 1
        key = (candidate.source, same_name(candidate.persona), size)
        if key not in best or _rank(candidate) < _rank(best[key]):
            best[key] = candidate
    kept = {id(candidate) for candidate in best.values()}
    return [candidate for candidate in candidates if id(candidate) in kept]


def count_by_source(candidates: Iterable[Candidate]) -> dict[str, int]:
    """How many candidates each source has, every source listed.

    Example:
        >>> count_by_source([])
        {'library': 0, 'mayner': 0}
    """
    counts = Counter(candidate.source for candidate in candidates)
    return {name: counts.get(name, 0) for name in SOURCES}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def to_row(candidate: Candidate) -> dict[str, str]:
    """The candidate as one CSV row, every value text."""
    unit = candidate.unit
    is_set = isinstance(unit, NeuronSet)
    source = SOURCES[candidate.source]
    values = {
        name: getattr(candidate, name) for name in (f.name for f in fields(Candidate))
    }
    values |= {
        "candidate": candidate.key,
        "unit": str(unit),
        "kind": candidate.kind,
        "neurons": len(unit.entries) if is_set else 1,
        "tags": ";".join(candidate.tags),
        "size": unit.size if is_set else None,
        "entries": json.dumps([list(entry) for entry in unit.entries])
        if is_set
        else "",
        "licence": source.licence,
        "credit": source.credit,
    }
    return {column: _text(values[column]) for column in COLUMNS}


def write_candidates(path: Path, candidates: Sequence[Candidate]) -> Path:
    """Write candidates to path as CSV, in one step; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(to_row(candidate) for candidate in candidates)
    partial.replace(path)
    return path


def _value(row: dict[str, str], name: str, kind: type) -> object:
    text = row[name]
    if text == "":
        return None
    if kind is bool:
        return {"true": True, "false": False}[text]
    return kind(text)


def from_row(row: dict[str, str]) -> Candidate:
    """A CSV row back as the Candidate it was written from."""
    unit_id = row["unit"]
    if row["kind"] == "set":
        model, _, name = unit_id.partition(":set:")
        entries = [tuple(entry) for entry in json.loads(row["entries"])]
        unit: Unit = NeuronSet(
            model, name, entries, row["source_ref"], float(row["size"])
        )
    else:
        unit = NeuronUnit.parse(unit_id)
    return Candidate(
        source=row["source"],
        persona=row["persona"],
        unit=unit,
        direction=int(row["direction"]),
        strength=float(row["strength"]),
        sensitive=_value(row, "sensitive", bool),  # type: ignore[arg-type]
        source_file=row["source_file"],
        source_ref=row["source_ref"],
        source_url=row["source_url"],
        downloaded_at=row["downloaded_at"],
        category=row["category"],
        tags=tuple(tag for tag in row["tags"].split(";") if tag),
        coherence=_value(row, "coherence", int),  # type: ignore[arg-type]
        weirdness=_value(row, "weirdness", int),  # type: ignore[arg-type]
        verified_pool=_value(row, "verified_pool", bool),  # type: ignore[arg-type]
        size_from=row["size_from"],
        mayner_in_role=_value(row, "mayner_in_role", float),  # type: ignore[arg-type]
        mayner_scored=_value(row, "mayner_scored", int),  # type: ignore[arg-type]
        mayner_share=_value(row, "mayner_share", float),  # type: ignore[arg-type]
    )


def read_candidates(path: Path) -> list[Candidate]:
    """The candidates of a file write_candidates wrote, units and all."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path} is not a candidates file: its columns differ")
        return [from_row(row) for row in reader]
