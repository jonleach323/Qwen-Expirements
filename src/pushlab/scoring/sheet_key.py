"""A labeled sheet's key: key.json, which `pushlab label export`
(pushlab.scoring.label_export) writes beside the sheet, never for the
labeler (decision 0021).

read_key reads what `pushlab label check` (pushlab.scoring.label_check)
needs of it: the sheet's rubrics, each with its version and its file's
SHA-256 at export, in the sheet's columns' order; the persona speaks_as
asks about and the personas which_persona chooses among; the SHA-256 of
labels.csv as exported; and each item's experiment file, experiment,
runner folder version and run, with the SHA-256 of the run's reply then.
It refuses a key that is not as export writes it, naming the first field
that is not.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab.scoring.personas import Persona

_ITEM_KEYS = {
    "item_id": str,
    "spec": str,
    "experiment": str,
    "version": int,
    "run_id": str,
    "output_sha256": str,
}


@dataclass(frozen=True)
class KeyItem:
    """An item of the sheet, as key.json names it: the experiment file's path
    as given at export, its experiment, the runner folder's version, and
    the run whose reply it is, with the reply's SHA-256 then."""

    item_id: str
    spec: str
    experiment: str
    version: int
    run_id: str
    output_sha256: str


@dataclass(frozen=True)
class SheetKey:
    """key.json, as label check needs it.

    rubrics gives each rubric of the sheet, in its columns' order, its
    version and its file's SHA-256 at export; persona is the one speaks_as
    asks about, and personas those which_persona chooses among; sha256 is
    key.json's own.
    """

    sheet: str
    rubrics: dict[str, tuple[int, str]]
    persona: Persona | None
    personas: tuple[Persona, ...]
    labels_csv_sha256: str
    items: tuple[KeyItem, ...]
    sha256: str


def _not_key(path: Path, why: str) -> ValueError:
    return ValueError(f"{path} is not a sheet's key: {why}")


def _persona(entry: object, where: str) -> Persona:
    if not (
        isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and isinstance(entry.get("description", ""), str)
    ):
        raise ValueError(f"{where} must be a name and a description")
    try:
        return Persona(entry["name"], entry.get("description", ""))
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def _item(entry: object, place: int) -> KeyItem:
    where = f"items[{place}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a mapping")
    for name, kind in _ITEM_KEYS.items():
        value = entry.get(name)
        if kind is int and (type(value) is not int or value < 1):
            raise ValueError(
                f"{where}'s {name} must be a whole number from 1, not {value!r}"
            )
        if kind is str and (not isinstance(value, str) or not value):
            raise ValueError(f"{where}'s {name} must be text, not {value!r}")
    return KeyItem(*(entry[name] for name in _ITEM_KEYS))


def _parse_key(data: object) -> dict[str, Any]:
    """key.json's fields that label check reads; ValueError says what is
    wrong with the first one that is not as export writes it.

    Example:
        >>> _parse_key({"sheet": "coherence-1", "rubrics": {}})
        Traceback (most recent call last):
        ...
        ValueError: it lacks persona
    """
    if not isinstance(data, dict):
        raise ValueError("it is not a mapping")
    for name in ("sheet", "rubrics", "persona", "personas", "labels_csv_sha256"):
        if name not in data:
            raise ValueError(f"it lacks {name}")
    if "items" not in data:
        raise ValueError("it lacks items")
    if not isinstance(data["sheet"], str):
        raise ValueError("sheet must be text")
    if not isinstance(data["labels_csv_sha256"], str):
        raise ValueError("labels_csv_sha256 must be text")
    rubrics = data["rubrics"]
    if (
        not isinstance(rubrics, dict)
        or not rubrics
        or not all(
            isinstance(entry, dict)
            and type(entry.get("version")) is int
            and entry["version"] >= 1
            and isinstance(entry.get("sha256"), str)
            for entry in rubrics.values()
        )
    ):
        raise ValueError(
            "rubrics must map each rubric's name to its version, a whole number "
            "from 1, and its sha256"
        )
    persona = None
    if data["persona"] is not None:
        persona = _persona(data["persona"], "persona")
    if not isinstance(data["personas"], list):
        raise ValueError(
            "personas must be a list of personas, each a name and a description"
        )
    personas = tuple(
        _persona(entry, f"personas[{place}]")
        for place, entry in enumerate(data["personas"])
    )
    if not isinstance(data["items"], list) or not data["items"]:
        raise ValueError("items must list at least one item")
    items = tuple(_item(entry, place) for place, entry in enumerate(data["items"]))
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"it gives the item {item.item_id} twice")
        seen.add(item.item_id)
    return {
        "sheet": data["sheet"],
        "rubrics": {
            name: (entry["version"], entry["sha256"]) for name, entry in rubrics.items()
        },
        "persona": persona,
        "personas": personas,
        "labels_csv_sha256": data["labels_csv_sha256"],
        "items": items,
    }


def read_key(path: Path) -> SheetKey:
    """The sheet's key, key.json at path.

    ValueError when it cannot be read, is not JSON, or is not a sheet's key
    as `pushlab label export` writes one, naming the first field that is
    not: a rubric's version or hash, a persona, an item's run, ID or
    version, and an item given twice.

    Example:
        >>> read_key(Path("/workspace/labels/coherence-1/key.json"))  # doctest: +SKIP
        SheetKey(sheet='coherence-1', rubrics={'coherence': (1, '...')}, ...)
    """
    path = Path(path)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error.strerror or error}") from None
    try:
        data = json.loads(content.decode("utf-8"))
    except ValueError as error:  # not UTF-8, or not JSON
        raise ValueError(f"{path} is not JSON: {error}") from None
    try:
        fields = _parse_key(data)
    except ValueError as error:
        raise _not_key(path, str(error)) from None
    return SheetKey(**fields, sha256=hashlib.sha256(content).hexdigest())
