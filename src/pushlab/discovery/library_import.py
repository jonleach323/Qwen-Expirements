"""Persona candidates from the Neuron Persona Library's files for Qwen3-8B.

The library (neuronpersonas.com) labels what Qwen3-8B says when one of its
MLP neurons is pushed up or down: the persona it becomes, if any, and the
library's tags and scores for it. `pushlab fetch library` downloads three of
its files into the data folder's library/qwen, with FETCHED.json, its record
of where and when it got each (decision 0020):

- library.json, a table: "fields" names the values of each of its "rows". L,
  n and s are the layer, the neuron and the direction (1 up, -1 down);
  persona the name; cat an index into its "categories" (default_assistant,
  altered_assistant, persona, incoherent, other); w and c the weirdness and
  the coherence, from 0 to 10; tags; a the library's strength, the strongest
  alpha it tried whose reply its judge did not call incoherent; and text, an
  excerpt of the reply, which is never kept here.
- combos.json, whose "pool" holds the personas the library verified by
  running them again ("pool_fields" names their values), and combinations of
  two of them, which are not candidates here.
- sensitive.json, whose "neurons" lists the sensitive neurons, each with its
  direction, as "L:n:s", such as "18:2468:-1", and whose "from" stamp gives
  the size of each file it was computed from (library_files.stamped_sizes).

import_library reads each value by the name its file gives it, not by its
place, so a file whose columns move still reads right. Its candidates are
the rows of category "persona" and the verified pool's personas, and each
says where it came from: the file inside the data folder, the place in it
(L12 N4410 s-1 for a row of library.json, pool L33N4096- for a pool
persona), the file's URL and when it was downloaded, unknown ("") for a file
whose bytes or SHA-256 are not those its record gives. It refuses a
sensitive.json computed from a library.json or combos.json of another size
than the one beside it, as its flags may then miss that file's personas.
Those steps, which read the files as files, are in library_files.py.
"""

import math
from pathlib import Path
from typing import Any, NamedTuple

from pushlab.discovery.candidates import Candidate
from pushlab.discovery.library_files import (
    COMBOS_FILE,
    FETCH_AGAIN,
    LIBRARY_FILE,
    NEW_FORMAT,
    SENSITIVE_FILE,
    load_json,
    provenance,
    sensitive_neurons,
)
from pushlab.discovery.sources import LIBRARY, LIBRARY_FILES
from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY
from pushlab.units.neuron import NeuronUnit

# The category of library.json's rows that are candidates.
PERSONA = "persona"
# The values read, by the names that library.json's "fields" and combos.json's
# "pool_fields" give them; the rest, the replies' text among them, are not.
ROW_FIELDS = ("L", "n", "s", "persona", "cat", "w", "c", "tags", "a")
POOL_FIELDS = ("id", "L", "n", "s", "a", "name", "w", "c", "verified")
# The library scores weirdness and coherence from 0 to this.
TOP_SCORE = 10


class _Persona(NamedTuple):
    """A row of library.json or of the pool, its values checked; ref is where
    it is in its file, and category "" for a pool row, which has none."""

    layer: int
    index: int
    sign: int
    name: str
    strength: float
    weirdness: int
    coherence: int
    category: str
    tags: tuple[str, ...]
    ref: str

    @property
    def key(self) -> tuple[int, int, int]:
        """The neuron and direction, (layer, index, sign), as files match them."""
        return (self.layer, self.index, self.sign)


def _is_whole(value: object) -> bool:
    """True for an int; False for True, which Python counts as one, or 2.0."""
    return isinstance(value, int) and not isinstance(value, bool)


def _table(
    path: Path, data: Any, names_key: str, rows_key: str, needed: tuple[str, ...]
) -> list[tuple[str, dict[str, Any]]]:
    """Each row of data[rows_key] as its needed values by name, with where it is.

    data[names_key] names each row's values in order, as library.json's
    "fields" and combos.json's "pool_fields" do.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must hold a JSON object with {names_key!r} and {rows_key!r}; "
            f"{FETCH_AGAIN}"
        )
    names, rows = data.get(names_key), data.get(rows_key)
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(
            f"{path} lacks {names_key!r}, the list that names the values of each "
            f"of its {rows_key!r}; {NEW_FORMAT}"
        )
    if missing := [name for name in needed if name not in names]:
        raise ValueError(
            f"{path} lacks the field{'s' if len(missing) > 1 else ''} "
            f"{', '.join(missing)}: its {names_key!r} are {', '.join(names)}; "
            f"{NEW_FORMAT}"
        )
    if not isinstance(rows, list):
        raise ValueError(f"{path} lacks {rows_key!r}, a list of rows; {NEW_FORMAT}")
    columns = {name: names.index(name) for name in needed}
    table = []
    for i, row in enumerate(rows):
        where = f"{path}: {rows_key}[{i}]"
        if not isinstance(row, list) or len(row) != len(names):
            raise ValueError(
                f"{where} must be a list of {len(names)} values, one for each "
                f"name in {names_key!r}; {NEW_FORMAT}"
            )
        table.append((where, {name: row[at] for name, at in columns.items()}))
    return table


def _neuron(
    where: str, values: dict[str, Any], model: str, shape: ModelInfo
) -> tuple[int, int, int]:
    """The row's layer, neuron and direction, refused unless the model has it."""
    layer, index, sign = values["L"], values["n"], values["s"]
    for what, value in (("L, the layer,", layer), ("n, the neuron,", index)):
        if not _is_whole(value) or value < 0:
            raise ValueError(
                f"{where}: {what} must be a whole number from 0, not {value!r}; "
                f"{NEW_FORMAT}"
            )
    if not _is_whole(sign) or sign not in (1, -1):
        raise ValueError(
            f"{where}: s, the direction, must be 1 or -1, not {sign!r}; {NEW_FORMAT}"
        )
    try:
        NeuronUnit(model, layer, index).validate(shape)
    except ValueError as error:
        raise ValueError(f"{where}: {error}; are the files another model's?") from None
    return layer, index, sign


def _score(where: str, what: str, value: object) -> int:
    """A weirdness or coherence score, a whole number from 0 to TOP_SCORE."""
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= TOP_SCORE:
            return value
    raise ValueError(
        f"{where}: {what} must be a whole number from 0 to {TOP_SCORE}, not "
        f"{value!r}; {NEW_FORMAT}"
    )


def _strength(where: str, value: object) -> float:
    """The library's strength, a, as a float: a finite number above 0."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        try:
            if math.isfinite(value) and value > 0:
                return float(value)
        except OverflowError:  # an int too large for a float
            pass
    raise ValueError(
        f"{where}: a, the library's strength, must be a number above 0, not "
        f"{value!r}; {NEW_FORMAT}"
    )


def _name(where: str, field: str, value: object) -> str:
    """The persona's name, which must be text that is not blank."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{where}: {field}, the persona's name, must be text, not {value!r}; "
            f"{NEW_FORMAT}"
        )
    return value


def _row(
    where: str,
    values: dict[str, Any],
    model: str,
    shape: ModelInfo,
    categories: list[str],
) -> _Persona:
    """A row of library.json, its values checked."""
    layer, index, sign = _neuron(where, values, model, shape)
    cat, tags = values["cat"], values["tags"]
    if not _is_whole(cat) or not 0 <= cat < len(categories):
        raise ValueError(
            f"{where}: cat must be the index of one of 'categories', 0 to "
            f"{len(categories) - 1}, not {cat!r}; {NEW_FORMAT}"
        )
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        raise ValueError(
            f"{where}: tags must be a list of words, not {tags!r}; {NEW_FORMAT}"
        )
    return _Persona(
        layer,
        index,
        sign,
        name=_name(where, "persona", values["persona"]),
        strength=_strength(where, values["a"]),
        weirdness=_score(where, "w, the weirdness,", values["w"]),
        coherence=_score(where, "c, the coherence,", values["c"]),
        category=categories[cat],
        tags=tuple(tags),
        ref=f"L{layer} N{index} s{sign:+d}",
    )


def _library_rows(path: Path, model: str, shape: ModelInfo) -> list[_Persona]:
    """library.json's rows in order, the first row of each neuron and direction."""
    data = load_json(path)
    table = _table(path, data, "fields", "rows", ROW_FIELDS)
    categories = data.get("categories")
    if (
        not isinstance(categories, list)
        or not all(isinstance(name, str) for name in categories)
        or PERSONA not in categories
    ):
        raise ValueError(
            f"{path} lacks 'categories', the names its rows' cat values point to, "
            f"{PERSONA!r} among them; {NEW_FORMAT}"
        )
    rows: dict[tuple[int, int, int], _Persona] = {}
    for where, values in table:
        row = _row(where, values, model, shape, categories)
        rows.setdefault(row.key, row)
    return list(rows.values())


def _pool(path: Path, model: str, shape: ModelInfo) -> list[_Persona]:
    """The pool's verified personas in order, the first of each neuron and direction."""
    pool: dict[tuple[int, int, int], _Persona] = {}
    table = _table(path, load_json(path), "pool_fields", "pool", POOL_FIELDS)
    for where, values in table:
        layer, index, sign = _neuron(where, values, model, shape)
        ident, verified = values["id"], values["verified"]
        if not isinstance(ident, str) or not ident.strip():
            raise ValueError(
                f"{where}: id must be the persona's ID, such as 'L33N4096-', not "
                f"{ident!r}; {NEW_FORMAT}"
            )
        if not isinstance(verified, bool):
            raise ValueError(
                f"{where}: verified must be true or false, not {verified!r}; "
                f"{NEW_FORMAT}"
            )
        entry = _Persona(
            layer,
            index,
            sign,
            name=_name(where, "name", values["name"]),
            strength=_strength(where, values["a"]),
            weirdness=_score(where, "w, the weirdness,", values["w"]),
            coherence=_score(where, "c, the coherence,", values["c"]),
            category="",
            tags=(),
            ref=f"pool {ident}",
        )
        if verified:
            pool.setdefault(entry.key, entry)
    return list(pool.values())


def _shape(model: str) -> ModelInfo:
    """The registry's shape of model, which must be in the registry."""
    spec = REGISTRY.get(model) if isinstance(model, str) else None
    if spec is None:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"{model!r} is not in the model registry ({known})")
    return ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size)


def import_library(folder: Path, model: str = "qwen3-8b") -> list[Candidate]:
    """The library's persona candidates, read from its files in folder.

    folder holds library.json, combos.json and sensitive.json, as `pushlab
    fetch library` downloads them into the data folder's library/qwen, and
    FETCHED.json, its record of them. model names the model the files
    describe, Qwen3-8B for the library's files under /data/qwen/; every row's
    neuron must be one of its.

    A candidate is made of each row of library.json whose category is
    "persona", and of each persona of combos.json's pool that the library
    verified and that library.json has no persona row for; that one is built
    from its pool row, with no category or tags. A neuron and direction that
    a file gives twice count once, as its first row says. The candidates come
    in library.json's order, then the pool's. Each says whether it is in the
    verified pool, and whether sensitive.json lists its neuron and direction;
    see the module's docstring for where it came from. A file not in the
    library's format is refused with ValueError, which names it, and so is a
    sensitive.json whose "from" stamp gives library.json or combos.json
    another size than the one in folder.

    Example:
        >>> pod = Path("/workspace/data/library/qwen")
        >>> found = import_library(pod)  # doctest: +SKIP
        >>> import_library(Path("/no/such/folder"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: /no/such/folder has no library.json, combos.json, ...
    """
    folder = Path(folder)
    shape = _shape(model)
    if missing := [file for file in LIBRARY_FILES if not (folder / file).is_file()]:
        raise FileNotFoundError(
            f"{folder} has no {', '.join(missing)}; run `uv run pushlab fetch "
            "library` (on the GPU pod), which downloads the Neuron Persona "
            f"Library's files into the data folder's {LIBRARY.folder}"
        )
    rows = _library_rows(folder / LIBRARY_FILE, model, shape)
    pool = _pool(folder / COMBOS_FILE, model, shape)
    sensitive = sensitive_neurons(folder / SENSITIVE_FILE)
    origins = provenance(folder)
    personas = [row for row in rows if row.category == PERSONA]
    has_row = {row.key for row in personas}
    chosen = [(row, LIBRARY_FILE) for row in personas]
    chosen += [(entry, COMBOS_FILE) for entry in pool if entry.key not in has_row]
    in_pool = {entry.key for entry in pool}
    return [
        Candidate(
            source=LIBRARY.name,
            persona=persona.name,
            unit=NeuronUnit(model, persona.layer, persona.index),
            direction=persona.sign,
            strength=persona.strength,
            sensitive=persona.key in sensitive,
            source_file=f"{LIBRARY.folder}/{file}",
            source_ref=persona.ref,
            source_url=origins[file][0],
            downloaded_at=origins[file][1],
            category=persona.category,
            tags=persona.tags,
            coherence=persona.coherence,
            weirdness=persona.weirdness,
            verified_pool=persona.key in in_pool,
        )
        for persona, file in chosen
    ]
