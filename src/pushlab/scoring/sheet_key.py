"""A labeled sheet's key: key.json, which `pushlab label export`
(pushlab.scoring.label_export) writes beside the sheet, never for the
labeler (decision 0021).

read_key reads what `pushlab label check` (pushlab.scoring.label_check)
needs of it: the sheet's rubrics, each with its version and its file's
SHA-256 at export, in the sheet's columns' order; the persona speaks_as
asks about and the personas which_persona chooses among; the SHA-256 of
labels.csv as exported; the runs folder export read the replies in, which
a key written before export recorded it lacks; and each item's experiment
file, experiment, runner folder version and run, with the SHA-256 of the
run's reply then. It refuses a key that is not as export writes it, naming
the first field that is not.

reply_texts reads the items' replies back from their runner folders, and
refuses a sheet whose replies cannot be found there or changed since the
export, naming the runner folder it read: the export may have read another.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab.runs.folder import default_runs_dir
from pushlab.runs.spec import read_spec
from pushlab.scoring.filled_sheet import listing
from pushlab.scoring.items import judged_runs
from pushlab.scoring.personas import Persona
from pushlab.store.scores import output_sha256

NEW_SHEET = "export a new sheet (pushlab label export)"
ELSEWHERE = "if the export read another runs folder, give it with --runs-dir"

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
    key.json's own; runs_dir is the runs folder export read, an absolute
    path, or None in a key written before export recorded it.
    """

    sheet: str
    rubrics: dict[str, tuple[int, str]]
    persona: Persona | None
    personas: tuple[Persona, ...]
    labels_csv_sha256: str
    items: tuple[KeyItem, ...]
    sha256: str
    runs_dir: Path | None = None


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
    runs_dir = data.get("runs_dir")  # absent from a key of an older export
    if runs_dir is not None and not (
        isinstance(runs_dir, str) and Path(runs_dir).is_absolute()
    ):
        raise ValueError(f"runs_dir must be a folder's absolute path, not {runs_dir!r}")
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
        "runs_dir": None if runs_dir is None else Path(runs_dir),
    }


def read_key(path: Path) -> SheetKey:
    """The sheet's key, key.json at path.

    ValueError when it cannot be read, is not JSON, or is not a sheet's key
    as `pushlab label export` writes one, naming the first field that is
    not: a rubric's version or hash, a persona, a runs folder that is not
    an absolute path, an item's run, ID or version, and an item given
    twice.

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


def reply_texts(
    key: SheetKey, runs_dir: Path | None = None
) -> tuple[dict[str, str], list[str]]:
    """Each item's reply, by item ID, read back from its experiment's runner
    folder in runs_dir (None: the default), and the experiments' models.

    ValueError refuses an experiment file that is not there (key.json gives
    its path as export was given it) or cannot be read; a runner folder that
    cannot be read (judged_runs); and a sheet whose replies changed since
    the export: an experiment file that names another experiment now, a
    runner folder at another version, a run with no reply now, and a reply
    that is not the one exported. A refusal of what a runner folder holds
    names the folder, and says to give --runs-dir if the export read
    another.

    Example:
        >>> reply_texts(read_key(Path("sheet/key.json")))  # doctest: +SKIP
        ({'sheet-3f2a-001': 'Hello!', ...}, ['qwen3-8b'])
    """
    runs_dir = default_runs_dir() if runs_dir is None else Path(runs_dir)
    texts: dict[str, str] = {}
    models: list[str] = []
    problems: list[str] = []
    folders: list[str] = []  # the runner folders the problems were found in
    for spec in dict.fromkeys(item.spec for item in key.items):
        if not Path(spec).exists():  # a path relative to where export ran
            raise ValueError(
                f"{spec}, an experiment file the sheet was exported from, is not "
                "there: run pushlab label check from the folder pushlab label "
                "export ran in"
            )
        read_spec(Path(spec))  # its own problems, which no runs folder solves
        try:
            experiment = judged_runs(Path(spec), runs_dir)
        except ValueError as error:
            raise ValueError(f"{error}; {ELSEWHERE}") from None
        models.append(experiment.spec.model)
        items = [item for item in key.items if item.spec == spec]
        name, found = experiment.spec.name, len(problems)
        versions = sorted({item.version for item in items} - {experiment.version})
        if others := sorted({item.experiment for item in items} - {name}):
            problems.append(
                f"{spec} is the experiment {name} now, not {listing(others)}"
            )
        elif versions:
            exported = listing([str(version) for version in versions])
            problems.append(
                f"{name}'s runner folder is at version {experiment.version} now, but "
                f"the sheet's replies are of version {exported}"
            )
        else:
            runs = {run.run_id: run for run in experiment.runs}
            gone = [item.run_id for item in items if item.run_id not in runs]
            changed = [
                item.run_id
                for item in items
                if item.run_id in runs
                and output_sha256(runs[item.run_id]) != item.output_sha256
            ]
            if gone:
                verb = "has" if len(gone) == 1 else "have"
                problems.append(f"{listing(gone)} {verb} no saved reply now")
            if changed:
                replies = "reply" if len(changed) == 1 else "replies"
                problems.append(
                    f"the {replies} of {listing(changed)} changed since the export"
                )
            texts |= {
                item.item_id: runs[item.run_id].output_text
                for item in items
                if item.run_id in runs
            }
        if len(problems) > found:
            folders.append(str(experiment.folder))
    if problems:
        where = "folder" if len(folders) == 1 else "folders"
        raise ValueError(
            f"{'; '.join(problems)} (in the runner {where} {listing(folders)}): "
            f"{ELSEWHERE}; else {NEW_SHEET}"
        )
    return texts, models
