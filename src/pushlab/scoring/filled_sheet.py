"""Your labels, read back from the sheet as a spreadsheet saved it
(decision 0021), for `pushlab label check` (pushlab.scoring.label_check).

read_labels reads your labeled labels.csv, for the items and rubrics of
the sheet's key (pushlab.scoring.sheet_key), however a spreadsheet saved
it: with or without a byte-order mark; in UTF-8, else Windows' cp1252; its
cells cut by commas, semicolons or tabs, whichever its header line holds
most of; its lines ended by \\r\\n or \\n; its header's names in any case
and with spaces around. Only the item_id column and the key's rubrics'
columns are read: not text, note or any other.

A label is matched ignoring case and the space around it, and given in the
rubric's own spelling: one of speaks_as's labels; a persona's name or
"none"; a whole number, such as "7" or " 7 " (or "7.0", as a spreadsheet
may format it), from 0 to 10. An empty cell is not a label: that reply is
skipped for that rubric. A blank row is passed over. Every problem is
refused at once, each naming its rows as a spreadsheet numbers them, the
header being row 1: a row with no item_id, an item_id the key does not
have, an item given twice, and a label the rubric does not have.
"""

import codecs
import csv
import hashlib
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab.scoring.sheet_key import SheetKey

ITEM_ID = "item_id"
DELIMITERS = (",", ";", "\t")  # a comma wins a tie
SHOWN_ROWS = 10  # rows a refusal names; the rest it counts
# A whole number as a spreadsheet may save it: 7, or 7.0 or 7,0.
_WHOLE = re.compile(r"([0-9]+)(?:[.,]0+)?")


@dataclass(frozen=True)
class FilledSheet:
    """Your labels, read: for each of the key's rubrics that has a column
    (columns, in the key's order), its labels by item ID, in the rubric's
    spelling, and how many of its cells were empty. items are the item IDs
    of the rows, in their order; sha256 is the file's."""

    path: Path
    sha256: str
    items: tuple[str, ...]
    columns: tuple[str, ...]
    labels: dict[str, dict[str, Any]]
    empty: dict[str, int]


def listing(words: Sequence[str], limit: int = SHOWN_ROWS) -> str:
    """words in a sentence: "a", "a and b", "a, b and c"; past limit, the
    first limit of them and how many more.

    Example:
        >>> listing(["5"]), listing(["5", "9"]), listing(["5", "9", "12"])
        ('5', '5 and 9', '5, 9 and 12')
        >>> listing([str(n) for n in range(1, 14)], limit=3)
        '1, 2, 3 and 10 more'
    """
    if len(words) > limit:
        return f"{', '.join(words[:limit])} and {len(words) - limit} more"
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def _either(options: Sequence[Any]) -> str:
    """Options as alternatives: "a, b or c"."""
    words = [str(option) for option in options]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


def column_name(place: int) -> str:
    """A column's name in a spreadsheet, from its place counted from 0.

    Example:
        >>> column_name(0), column_name(25), column_name(26), column_name(701)
        ('A', 'Z', 'AA', 'ZZ')
    """
    name, place = "", place + 1
    while place:
        place, rest = divmod(place - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def decode_sheet(data: bytes, path: Path) -> str:
    """A saved sheet's text: UTF-8, with or without a byte-order mark, else
    Windows' cp1252; ValueError for UTF-16, which spreadsheets also save, a
    file that starts as UTF-8 but is not, and one that is neither.

    Example:
        >>> decode_sheet(b"\\xef\\xbb\\xbfitem_id", Path("l.csv"))
        'item_id'
        >>> decode_sheet(b"caf\\xe9", Path("l.csv"))  # cp1252
        'café'
    """
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        raise ValueError(
            f"{path} is in UTF-16: save it as CSV, in UTF-8 (in Excel, CSV UTF-8)"
        )
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        if data.startswith(codecs.BOM_UTF8):
            raise ValueError(f"{path} starts as UTF-8 but is not: {error}") from None
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        raise ValueError(
            f"{path} is neither UTF-8 nor cp1252 text: save it as CSV, in UTF-8"
        ) from None


def delimiter_of(text: str) -> str:
    """The delimiter of a saved sheet: the one of a comma, a semicolon and a
    tab that its header line holds most of; a comma on a tie.

    Example:
        >>> delimiter_of("item_id;text;coherence;note\\r\\nsheet-1-001;Hi, you.;7;")
        ';'
    """
    line = text.split("\n", 1)[0]
    return max(DELIMITERS, key=line.count)


def match_label(cell: str, options: Sequence[Any]) -> Any:
    """The option a cell holds, in the option's own spelling, or None.

    Options of whole numbers take a whole number written in digits 0 to 9,
    as "7", " 7 " or "7.0"; others match ignoring case and the space around
    and within the cell.

    Example:
        >>> match_label(" Speaks_As ", ["speaks_as", "assistant"])
        'speaks_as'
        >>> match_label(" 7 ", range(11)), match_label("7.5", range(11))
        (7, None)
    """
    if all(type(option) is int for option in options):
        found = _WHOLE.fullmatch(cell.strip())
        number = int(found[1]) if found else None
        return number if number in options else None
    wanted = " ".join(cell.split()).casefold()
    for option in options:
        if " ".join(str(option).split()).casefold() == wanted:
            return option
    return None


def _header(cells: Sequence[str], path: Path, names: Sequence[str]) -> dict[str, int]:
    """Where each of names is in the header, its names compared in any case
    and with spaces around; ValueError for one given twice."""
    found = [cell.replace("\ufeff", "").strip().casefold() for cell in cells]
    places = {}
    for name in names:
        at = [place for place, cell in enumerate(found) if cell == name]
        if len(at) > 1:
            columns = listing([column_name(place) for place in at])
            raise ValueError(
                f"{path} has two columns named {name} (columns {columns}): keep one"
            )
        if at:
            places[name] = at[0]
    return places


def _rows(numbers: Sequence[int]) -> str:
    return ("row " if len(numbers) == 1 else "rows ") + listing(
        [str(number) for number in numbers]
    )


def _cells(found: Sequence[tuple[int, str]]) -> str:
    """Cells and their rows: "'x' (row 5) and 'y' (row 9)"."""
    return listing([f"{cell!r} (row {number})" for number, cell in found])


def _bad(name: str, found: list[tuple[int, str]], options: Sequence[Any]) -> str:
    """A rubric's cells that hold none of its labels, and what it takes."""
    verb = "is" if len(found) == 1 else "are"
    if all(type(option) is int for option in options):
        low, high = min(options), max(options)
        wanted = "a whole number" if verb == "is" else "whole numbers"
        return f"{name}: {_cells(found)} {verb} not {wanted} from {low} to {high}"
    return f"{name}: {_cells(found)} {verb} not one of {_either(options)}"


def read_labels(
    path: Path, key: SheetKey, options: Mapping[str, Sequence[Any]]
) -> FilledSheet:
    """Your labels in the sheet at path, for the key's items and rubrics;
    options gives each rubric's labels, in its own spelling, such as
    Rubric.options gives them.

    ValueError when the file cannot be read, has no header or no item_id
    column, or names a column twice; and, every one at once, with their
    rows: a row with no item_id, an item_id the key does not have, an item
    in more than one row, and a cell that holds none of its rubric's labels.

    Example:
        >>> read_labels(Path("labels.csv"), key, options)  # doctest: +SKIP
        FilledSheet(path=PosixPath('labels.csv'), ...)
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error.strerror or error}") from None
    text = decode_sheet(data, path)
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter_of(text))
        table = list(reader)
    except csv.Error as error:
        raise ValueError(f"cannot read {path} as CSV: {error}") from None
    if not table or not any(cell.strip() for cell in table[0]):
        raise ValueError(
            f"{path} has no header: its first row must name its columns, item_id, "
            "text and the rubrics"
        )
    places = _header(table[0], path, [ITEM_ID, *key.rubrics])
    if ITEM_ID not in places:
        named = ", ".join(cell.strip() for cell in table[0] if cell.strip())
        raise ValueError(f"{path} has no item_id column: its header holds {named}")
    columns = tuple(name for name in key.rubrics if name in places)
    known = {item.item_id for item in key.items}
    rows_of: dict[str, list[int]] = {}
    no_id: list[int] = []
    unknown: list[tuple[int, str]] = []
    labels: dict[str, dict[str, Any]] = {name: {} for name in columns}
    empty = dict.fromkeys(columns, 0)
    bad: dict[str, list[tuple[int, str]]] = {name: [] for name in columns}

    def cell(row: Sequence[str], name: str) -> str:
        place = places[name]
        return row[place].strip() if place < len(row) else ""

    for number, row in enumerate(table[1:], start=2):  # the header is row 1
        if not any(value.strip() for value in row):
            continue
        item_id = cell(row, ITEM_ID)
        if not item_id:
            no_id.append(number)
            continue
        if item_id not in known:
            unknown.append((number, item_id))
            continue
        rows_of.setdefault(item_id, []).append(number)
        if len(rows_of[item_id]) > 1:
            continue
        for name in columns:
            value = cell(row, name)
            if not value:
                empty[name] += 1
            elif (label := match_label(value, options[name])) is None:
                bad[name].append((number, value))
            else:
                labels[name][item_id] = label
    problems = []
    if no_id:
        problems.append(
            f"{_rows(no_id)} {'has' if len(no_id) == 1 else 'have'} no item_id"
        )
    if unknown:
        items = "is not an item" if len(unknown) == 1 else "are not items"
        problems.append(f"{_cells(unknown)} {items} of the sheet")
    repeated = {item_id: rows for item_id, rows in rows_of.items() if len(rows) > 1}
    if repeated:
        found = [f"{item_id!r} ({_rows(rows)})" for item_id, rows in repeated.items()]
        verb = (
            "is in more than one row"
            if len(found) == 1
            else "are each in more than one row"
        )
        problems.append(f"{listing(found)} {verb}: give each item once")
    problems += [_bad(name, bad[name], options[name]) for name in columns if bad[name]]
    if problems:
        raise ValueError(
            f"fix these in {path}, then check again:\n"
            + "\n".join(f"  {problem}" for problem in problems)
        )
    return FilledSheet(
        path,
        hashlib.sha256(data).hexdigest(),
        tuple(rows_of),
        columns,
        labels,
        empty,
    )
