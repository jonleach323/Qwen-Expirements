"""The categories file of pushlab panel select: each finalist's category,
which you fill in (decision 0023).

The persona sources give no categories (decision 0020), so select asks you:
a CSV with a header line, key, persona and category, and a row per finalist.
Its key and category columns are read, however a spreadsheet saved the file
(pushlab.scoring.filled_sheet: a byte-order mark or none; commas, semicolons
or tabs; UTF-8 or cp1252), the header's names and a category in any case
and with spaces around; the persona column is there to help you. A category
must be one of the selection's (configs/panel.yaml), each key given once; a
row of a key that is no finalist is not used. When the file is
missing, or leaves a finalist without a category, write_template writes it
again: a row per finalist, in order, with the category the file gave it or
a blank one, then the file's other rows that give one, so that nothing you
filled in is lost. It names candidates by key, the Neuron Persona Library's
among them, so it belongs outside the repository (decision 0020).
"""

import csv
import io
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pushlab.runs.ladder_command import inside_repository
from pushlab.scoring.filled_sheet import decode_sheet, delimiter_of, listing
from pushlab.scoring.label_sheet import guarded_csv

COLUMNS = ("key", "persona", "category")
OUTSIDE = "/workspace/experiments/categories.csv"  # where one goes on the pod


def check_categories(path: Path) -> None:
    """Refuse a categories file that is a folder or inside the repository.

    Example:
        >>> from pushlab import PROJECT_ROOT
        >>> check_categories(PROJECT_ROOT / "categories.csv")
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: --categories .../categories.csv is in the repository: ...
    """
    if Path(path).is_dir():
        raise ValueError(f"--categories must be a file, but {path} is a folder")
    if inside_repository(path):
        raise ValueError(
            f"--categories {path} is in the repository: it names candidates by key, "
            "the Neuron Persona Library's among them, whose data stays out of git "
            "(decision 0020), and the pod never changes the repository; give one "
            f"outside it, such as {OUTSIDE}"
        )


@dataclass(frozen=True)
class CategoriesFile:
    """A categories file, read: each key's category, in the selection's
    spelling, of the rows that give one, and each row's persona, by key."""

    path: Path
    categories: dict[str, str]
    personas: dict[str, str]

    def unfilled(self, keys: Sequence[str]) -> list[str]:
        """Those of keys the file gives no category, in order."""
        return [key for key in keys if key not in self.categories]


def _places(header: Sequence[str], path: Path) -> dict[str, int]:
    """Where each column of COLUMNS is in header; ValueError without key or
    category, or with a column named twice."""
    names = [cell.replace("﻿", "").strip().casefold() for cell in header]
    places = {}
    for name in COLUMNS:
        found = [place for place, cell in enumerate(names) if cell == name]
        if len(found) > 1:
            raise ValueError(f"{path} has two columns named {name}: keep one")
        if found:
            places[name] = found[0]
    if "key" not in places or "category" not in places:
        raise ValueError(
            f"{path} needs a key and a category column, as select writes it: "
            f"{', '.join(COLUMNS)}"
        )
    return places


def _category(cell: str, categories: Sequence[str]) -> str | None:
    """The category a cell names, in the selection's spelling, or None."""
    wanted = " ".join(cell.split()).casefold()
    return next((c for c in categories if c.casefold() == wanted), None)


def read_categories(path: Path, categories: Sequence[str]) -> CategoriesFile | None:
    """The categories file at path, None when there is none; categories are
    the selection's.

    ValueError when it cannot be read, has no key or category column, gives
    a key twice, or a category that is not one of categories, naming the
    rows as a spreadsheet numbers them, the header being row 1.

    Example:
        >>> read_categories(Path("/no/such/categories.csv"), ["people"]) is None
        True
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error.strerror or error}") from None
    text = decode_sheet(data, path)
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter_of(text))
        table = list(reader)
    except csv.Error as error:
        raise ValueError(f"cannot read {path} as CSV: {error}") from None
    if not table:
        return CategoriesFile(path, {}, {})
    places = _places(table[0], path)
    found: dict[str, str] = {}
    personas: dict[str, str] = {}
    seen: dict[str, int] = {}
    bad: list[str] = []

    def cell(row: Sequence[str], name: str) -> str:
        place = places.get(name)
        return row[place].strip() if place is not None and place < len(row) else ""

    for number, row in enumerate(table[1:], start=2):
        key = cell(row, "key")
        if not key:
            continue
        if key in seen:
            raise ValueError(
                f"{path} gives {key} in rows {seen[key]} and {number}: give it once"
            )
        seen[key] = number
        personas[key] = cell(row, "persona")
        if not (given := cell(row, "category")):
            continue
        if (category := _category(given, categories)) is None:
            bad.append(f"{given!r} (row {number})")
        else:
            found[key] = category
    if bad:
        verb = "is not a category" if len(bad) == 1 else "are not categories"
        raise ValueError(
            f"{path}: {listing(bad)} {verb}; a category is one of "
            f"{', '.join(categories)}"
        )
    return CategoriesFile(path, found, personas)


def write_template(
    path: Path, finalists: Sequence[tuple[str, str]], known: CategoriesFile | None
) -> Path:
    """Write the categories file at path for finalists, each a key and its
    persona, in order, as the module's docstring says, keeping what known,
    the file as read, gives; the path. The file is written whole beside path
    and then put in its place, so a failed write leaves the old one.

    Example:
        >>> write_template(Path("categories.csv"), [("library/L9N1-", "Owl")], None)
        ... # doctest: +SKIP
        PosixPath('categories.csv')
    """
    path = Path(path)
    given = {} if known is None else known.categories
    rows = [[*COLUMNS]]
    rows += [[key, persona, given.get(key, "")] for key, persona in finalists]
    keys = {key for key, _ in finalists}
    if known is not None:
        rows += [
            [key, known.personas.get(key, ""), category]
            for key, category in known.categories.items()
            if key not in keys
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(guarded_csv(rows))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
