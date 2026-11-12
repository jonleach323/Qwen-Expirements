"""`pushlab candidates`: one list of persona candidates from both sources.

It imports the candidates of the Neuron Persona Library
(pushlab.discovery.library_import) and of Will Mayner's persona-neurons data
(pushlab.discovery.mayner_import) from the data folder, where `pushlab fetch`
put them, and checks that every candidate's unit is one of the model's. It
drops the candidates a source calls sensitive, unless asked to keep them,
and then the duplicate names within each source (candidates.drop_duplicates),
in that order, so that a name a sensitive and a safe persona share keeps the
safe one. What is left goes into one CSV file, each row saying where its
candidate came from and carrying its source's licence and credit
(candidates.write_candidates). It prints how many candidates each source
gave, dropped and kept, and each source's credit, since Mayner's data is
under CC BY 4.0 (decision 0020). It loads no model.
"""

import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from pushlab.cli_discovery import SOURCES_MODEL_ID, models_with_sources
from pushlab.discovery import sources
from pushlab.discovery.candidates import (
    Candidate,
    count_by_source,
    drop_duplicates,
    write_candidates,
)
from pushlab.discovery.library_import import import_library
from pushlab.discovery.mayner_import import import_mayner
from pushlab.discovery.sources import LIBRARY, MAYNER, SOURCES, Source, source_dir
from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY

# The most problems with units it names; the count says how many there are.
PROBLEMS_SHOWN = 10
# The candidate lists git ignores (.gitignore): the library's data stays out.
IGNORED = "units/candidates_*.csv"


class Tally(NamedTuple):
    """How many of a source's candidates were read, dropped as sensitive,
    dropped as duplicate names, and kept."""

    read: int
    sensitive: int
    duplicates: int
    kept: int


def _refuse(message: str) -> int:
    print(f"pushlab candidates: {message}", file=sys.stderr)
    return 1


def check_units(candidates: Iterable[Candidate], model: str) -> list[str]:
    """What is wrong with the candidates' units for model, one line each.

    Each unit must be one of model's, and pass validate against the
    registry's shape of model. An empty list means every unit is right.

    Example:
        >>> from pushlab.units.neuron import NeuronUnit
        >>> deep = NeuronUnit("qwen3-8b", 36, 0)  # Qwen3-8B's layers are 0 to 35
        >>> mole = Candidate("library", "Mole", deep, 1, 1.0, False, "f", "L36 N0 s+1")
        >>> print(*check_units([mole], "qwen3-8b"))  # doctest: +ELLIPSIS
        library/L36N0+, from f: qwen3-8b:L36:N0: layer 36 is out of range; ...
        >>> print(*check_units([mole], "qwen3.5-4b"))
        library/L36N0+, from f: qwen3-8b:L36:N0 is a unit of qwen3-8b, not qwen3.5-4b
    """
    spec = REGISTRY[model]
    shape = ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size)
    problems = []
    for candidate in candidates:
        unit, where = candidate.unit, f"{candidate.key}, from {candidate.source_file}"
        if unit.model_name != model:
            problems.append(
                f"{where}: {unit} is a unit of {unit.model_name}, not {model}"
            )
            continue
        try:
            unit.validate(shape)
        except ValueError as error:
            problems.append(f"{where}: {error}")
    return problems


def clean(
    found: Sequence[Candidate], include_sensitive: bool = False
) -> tuple[list[Candidate], dict[str, Tally]]:
    """The candidates of found that are kept, and each source's Tally.

    The sensitive ones are dropped first, unless include_sensitive, and then
    the duplicate names within each source, so that a name a sensitive and a
    safe persona share keeps the safe one. The kept keep found's order.

    Example:
        >>> from dataclasses import replace
        >>> from pushlab.units.neuron import NeuronUnit
        >>> teapot = Candidate(
        ...     "library",
        ...     "Teapot",
        ...     NeuronUnit("qwen3-8b", 3, 1),
        ...     1,
        ...     1.0,
        ...     False,
        ...     "f",
        ...     "r",
        ... )
        >>> found = [
        ...     replace(teapot, sensitive=True, coherence=9),
        ...     replace(teapot, unit=NeuronUnit("qwen3-8b", 3, 2), coherence=7),
        ...     replace(teapot, unit=NeuronUnit("qwen3-8b", 3, 3), coherence=5),
        ... ]
        >>> kept, tallies = clean(found)
        >>> [candidate.key for candidate in kept]
        ['library/L3N2+']
        >>> tallies["library"]
        Tally(read=3, sensitive=1, duplicates=1, kept=1)
    """
    safe = [c for c in found if include_sensitive or not c.sensitive]
    kept = drop_duplicates(safe)
    read, left, final = (count_by_source(group) for group in (found, safe, kept))
    tallies = {
        name: Tally(
            read[name], read[name] - left[name], left[name] - final[name], final[name]
        )
        for name in SOURCES
    }
    return kept, tallies


def tally_lines(tallies: Mapping[str, Tally]) -> list[str]:
    """The counts by source, under their columns' names and above their total.

    Example:
        >>> tallies = {"library": Tally(8, 2, 1, 5), "mayner": Tally(550, 0, 0, 550)}
        >>> print("\\n".join(tally_lines(tallies)))
        source   read  sensitive  duplicates  kept
        library     8          2           1     5
        mayner    550          0           0   550
        total     558          2           1   555
    """
    total = Tally(*(sum(tally[i] for tally in tallies.values()) for i in range(4)))
    rows = [("source", *Tally._fields)]
    rows += [
        (name, *(f"{count:,}" for count in tally))
        for name, tally in [*tallies.items(), ("total", total)]
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        row[0].ljust(widths[0])
        + "".join(
            f"  {cell:>{width}}"
            for cell, width in zip(row[1:], widths[1:], strict=True)
        )
        for row in rows
    ]


def credit_lines(credited: Iterable[Source]) -> list[str]:
    """Each source's licence and the credit a list made from it gives it.

    Example:
        >>> print(credit_lines([MAYNER])[0])  # doctest: +ELLIPSIS
        mayner: licence: CC BY 4.0. Credit: Will Mayner, ...; adapted by pushlab: ...
    """
    return [
        f"{source.name}: licence: {source.licence}. Credit: {source.list_credit}"
        for source in credited
    ]


def data_warnings(candidates: Sequence[Candidate]) -> list[str]:
    """What the user should know about the data the candidates came from.

    A source whose candidates have no download time has no FETCHED.json, or
    one that no longer matches its files, as after a failed fetch; Mayner's
    sets sized from the split's shares lack his role vectors, which give
    exactly his strength.

    Example:
        >>> data_warnings([])
        []
    """
    warnings = []
    for source in SOURCES.values():
        mine = [
            candidate for candidate in candidates if candidate.source == source.name
        ]
        if undated := sum(not candidate.downloaded_at for candidate in mine):
            warnings.append(
                f"{undated:,} of {len(mine):,} {source.name} candidates have no "
                f"download time: the record of where {source.title}'s files came "
                "from is missing or does not match them, as after a failed or "
                f"unfinished fetch. Run `uv run pushlab fetch {source.name}` on the "
                "GPU pod until it finishes, then make the list again"
            )
    shares = sum(candidate.size_from == "split shares" for candidate in candidates)
    if shares:
        warnings.append(
            f"{shares:,} of Mayner's sets are sized from the split's shares, 0.1 to "
            "0.2% off his strength, as their role vectors are missing; run `uv run "
            "pushlab fetch mayner` on the GPU pod to get them"
        )
    return warnings


def run_candidates(
    model: str,
    out: Path,
    data_dir: Path | None = None,
    include_sensitive: bool = False,
    overwrite: bool = False,
) -> int:
    """Write both sources' candidates for model to out; return the exit status.

    The sources are read from data_dir ($PUSHLAB_DATA_DIR, else
    /workspace/data, by default), where `pushlab fetch` puts them. Returns 0
    once out is written, and 1 with the reasons on stderr when it refuses:
    before anything is imported, a model the sources list no neurons of, and
    an out that is a folder, or that exists unless overwrite; then a source
    that is not fetched or not in its format, and a unit that is not one of
    the model's.

    Example:
        >>> out = Path("units/candidates_qwen3-8b.csv")
        >>> run_candidates("qwen3-8b", out)  # doctest: +SKIP
        0
    """
    if model not in (models := models_with_sources()):
        return _refuse(
            f"the persona sources list no neurons of {model!r}; their neurons are "
            f"{SOURCES_MODEL_ID}'s, so give --model {' or '.join(models)}"
        )
    out = Path(out)
    if out.is_dir():
        return _refuse(f"--out must be a file, but {out} is a folder")
    if out.exists() and not overwrite:
        return _refuse(f"{out} already exists; pass --overwrite to replace it")
    root = data_dir if data_dir is not None else sources.data_dir()
    print(f"Reading both sources' candidates from {root}", flush=True)
    found: list[Candidate] = []
    problems = []
    for source, importer in ((LIBRARY, import_library), (MAYNER, import_mayner)):
        try:
            found += importer(source_dir(source, root), model)
        except (OSError, ValueError) as error:  # each names the file and the fault
            problems.append(str(error))
    if not problems and (wrong := check_units(found, model)):
        problems = [f"a unit is not one of {model}'s: {line}" for line in wrong]
        if len(problems) > PROBLEMS_SHOWN:
            more = len(problems) - PROBLEMS_SHOWN
            problems = [*problems[:PROBLEMS_SHOWN], f"and {more:,} more like them"]
    if problems:
        for problem in problems:
            _refuse(problem)
        return 1
    for warning in data_warnings(found):
        print(f"pushlab candidates: warning: {warning}", file=sys.stderr)
    kept, tallies = clean(found, include_sensitive)
    try:
        write_candidates(out, kept)
    except OSError as error:
        return _refuse(f"cannot write {out}: {error}")
    print(
        "By source: the candidates read, those dropped as sensitive and then as "
        "duplicate names, and those kept:"
    )
    for line in tally_lines(tallies):
        print(f"  {line}")
    if include_sensitive:
        flagged = sum(candidate.sensitive for candidate in kept)
        print(
            f"Kept {flagged:,} candidates a source calls sensitive "
            "(--include-sensitive); their rows say sensitive true."
        )
    print(f"Saved {len(kept):,} candidates in {out}")
    print("Credit the sources wherever the list is used; each row names its own:")
    for line in credit_lines(SOURCES.values()):
        print(f"  {line}")
    if tallies[LIBRARY.name].kept:
        print(
            "The library states no licence, so keep the list out of git (git "
            f"ignores {IGNORED})."
        )
    return 0
