"""The Neuron Persona Library's files as files: their JSON, sensitive.json, the record.

library_import reads the library's three files for Qwen3-8B (decision 0020).
These are its steps that do not depend on what a persona row says:

- load_json reads a file's JSON. A file that is not JSON is refused with the
  next step: download it again on the GPU pod with `uv run pushlab fetch
  library --refresh`.
- sensitive_neurons reads sensitive.json's sensitive neurons once its "from"
  stamp agrees with the files beside it. The library computes sensitive.json
  from library.json and combos.json, among others, and stamps it with their
  sizes (stamped_sizes). A sensitive.json whose stamp gives either file
  another size than the one beside it was computed from another copy of that
  file, so it may miss sensitive personas of the copy beside it, and is
  refused.
- provenance reads each file's URL and download time from FETCHED.json, the
  record `pushlab fetch library` writes; the time is unknown ("") for a file
  that is not the copy its entry describes.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pushlab.discovery.sources import LIBRARY, LIBRARY_FILES, RECORD_FILE, read_record

# The three files, as sources.LIBRARY_FILES names them.
LIBRARY_FILE, COMBOS_FILE, SENSITIVE_FILE = LIBRARY_FILES
# The next step for a file that is not JSON, or not a JSON object.
FETCH_AGAIN = (
    "download it again on the GPU pod with `uv run pushlab fetch library --refresh`"
)
# What a field or a value the importer does not expect asks.
NEW_FORMAT = "has the library changed its format?"
# One of sensitive.json's neurons: its layer, neuron and direction.
SENSITIVE_NEURON = re.compile(r"([0-9]+):([0-9]+):([+-]?1)")
# A file in sensitive.json's "from" stamp: its name, bytes and time in ms.
STAMP_PART = re.compile(r"([^:]+):([0-9]+):([0-9]+(?:\.[0-9]+)?)")
# The files whose sizes the stamp must give as they are beside sensitive.json.
STAMPED_FILES = (LIBRARY_FILE, COMBOS_FILE)


def load_json(path: Path) -> Any:
    """The JSON in the file at path, refused with what to do if it is not JSON.

    Example:
        >>> load_json(Path("/dev/null"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: /dev/null is not valid JSON (...); download it again on the GPU ...
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not valid JSON ({error}); {FETCH_AGAIN}") from None


def stamped_sizes(stamp: object) -> list[tuple[str, int]]:
    """The sizes sensitive.json's "from" stamp gives library.json and combos.json.

    The library stamps sensitive.json with what it was computed from: the
    time, then "<file>:<bytes>:<time>" for each file, all joined by "|", the
    times in milliseconds. Each part for library.json or combos.json gives
    (file, bytes), in the stamp's order. The other files' parts give nothing,
    and so does a stamp that is not text or names neither file, as it cannot
    be checked; a part for either file in another shape is refused.

    Example:
        >>> stamp = (
        ...     "1794390012000.25|library.json:1811:1794330900000"
        ...     "|combos.json:1394:1794330900000|detox.json:512:1794390000000.5"
        ... )
        >>> stamped_sizes(stamp)
        [('library.json', 1811), ('combos.json', 1394)]
        >>> stamped_sizes(None), stamped_sizes("made on a Tuesday")
        ([], [])
        >>> stamped_sizes("1794390012000|library.json:big:1794330900000")
        Traceback (most recent call last):
        ...
        ValueError: 'from' gives library.json as 'library.json:big:1794330900000', not library.json:<bytes>:<time>; has the library changed its format?
    """  # noqa: E501
    if not isinstance(stamp, str):
        return []
    sizes = []
    for given in stamp.split("|"):
        part = given.strip()
        name = part.split(":", 1)[0]
        if name not in STAMPED_FILES:
            continue
        match = STAMP_PART.fullmatch(part)
        if match is None:
            raise ValueError(
                f"'from' gives {name} as {part!r}, not {name}:<bytes>:<time>; "
                f"{NEW_FORMAT}"
            )
        sizes.append((name, int(match[2])))
    return sizes


def _check_stamp(path: Path, stamp: object) -> None:
    """Refuse sensitive.json at path if its "from" stamp gives library.json or
    combos.json another size than the file of that name beside it has."""
    try:
        sizes = stamped_sizes(stamp)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    for file, stamped in sizes:
        here = path.parent / file
        if (size := here.stat().st_size) != stamped:
            raise ValueError(
                f"{path} was computed from a {file} of {stamped:,} bytes, but "
                f"{here} has {size:,} bytes, so it may miss sensitive personas of "
                "this one. Download the library's files again on the GPU pod with "
                "`uv run pushlab fetch library --refresh`; if this persists after "
                "a refresh, the library's own files disagree"
            )


def sensitive_neurons(path: Path) -> set[tuple[int, int, int]]:
    """sensitive.json's neurons, each as (layer, neuron, direction).

    path is sensitive.json, beside the library.json and combos.json it was
    computed from. If its "from" stamp gives either of them another size than
    the file there has, it is refused: it may miss sensitive personas of that
    file. A stamp that is missing, not text or names neither is not checked.

    Example:
        >>> pod = Path("/workspace/data/library/qwen/sensitive.json")
        >>> len(sensitive_neurons(pod))  # doctest: +SKIP
        344
    """
    path = Path(path)
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must hold a JSON object with 'neurons'; {FETCH_AGAIN}"
        )
    _check_stamp(path, data.get("from"))
    neurons = data.get("neurons")
    if not isinstance(neurons, list):
        raise ValueError(
            f"{path} lacks 'neurons', the list of sensitive neurons written "
            f'L:n:s, such as "18:2468:-1"; {NEW_FORMAT}'
        )
    found = set()
    for i, item in enumerate(neurons):
        match = SENSITIVE_NEURON.fullmatch(item) if isinstance(item, str) else None
        if match is None:
            raise ValueError(
                f"{path}: neurons[{i}] must be a neuron written L:n:s, its layer, "
                f'neuron and direction, such as "18:2468:-1", not {item!r}; '
                f"{NEW_FORMAT}"
            )
        found.add((int(match[1]), int(match[2]), int(match[3])))
    return found


def _describes(entry: dict[str, Any], path: Path) -> bool:
    """True if a record's entry gives the bytes and SHA-256 of the file at path."""
    if entry.get("bytes") != path.stat().st_size:
        return False
    with path.open("rb") as file:  # hashed a block at a time, never whole
        return entry.get("sha256") == hashlib.file_digest(file, "sha256").hexdigest()


def provenance(folder: Path) -> dict[str, tuple[str, str]]:
    """Each library file's URL and download time, from the fetch record in folder.

    A file the record lacks, or every file when there is no record, gets the
    library's own URL for it and "" for the time, which is not known. So does
    the time of a file whose bytes or SHA-256 are not those its entry gives:
    the entry describes another copy of it, as `pushlab fetch library` finds
    too. Its URL is kept.

    Example:
        >>> provenance(Path("/no/such/folder"))["combos.json"]
        ('https://neuronpersonas.com/data/qwen/combos.json', '')
    """
    folder = Path(folder)
    path = folder / RECORD_FILE
    files = read_record(folder).get("files") or {}
    if not isinstance(files, dict):
        raise ValueError(f"{path}: 'files' must hold what was recorded of each file")
    found = {}
    for file in LIBRARY_FILES:
        entry = files.get(file) or {}
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: files[{file!r}] must be an object with its url and "
                "downloaded_at"
            )
        url = entry.get("url") or f"{LIBRARY.url}/{file}"
        downloaded_at = entry.get("downloaded_at") or ""
        if not isinstance(url, str) or not isinstance(downloaded_at, str):
            raise ValueError(
                f"{path}: files[{file!r}]'s url and downloaded_at must be text"
            )
        if downloaded_at and not _describes(entry, folder / file):
            downloaded_at = ""
        found[file] = (url, downloaded_at)
    return found
