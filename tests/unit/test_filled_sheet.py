"""Your labels read back (pushlab.scoring.filled_sheet): labels.csv as a
spreadsheet saves it, every problem refused naming its rows as a
spreadsheet numbers them, the header being row 1. test_sheet_key.py tests
key.json's reading.

The sheets here are written by the tests, with invented replies, as
spreadsheets save them: with or without a byte-order mark, in UTF-8 or
cp1252, cut by commas, semicolons or tabs, with \\r\\n or \\n.
"""

import codecs
import csv
import doctest
import hashlib
import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pushlab.scoring import filled_sheet
from pushlab.scoring.filled_sheet import delimiter_of, match_label, read_labels
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import RUBRIC_NAMES, load_rubric
from pushlab.scoring.sheet_key import KeyItem, SheetKey

OWL = Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles.")
SNOW = Persona("Snow Owl", "A quiet white owl of the far north.")
RUBRICS = {
    name: (load_rubric(name).version, load_rubric(name).sha256) for name in RUBRIC_NAMES
}
ITEMS = tuple(
    KeyItem(f"sheet-1-{k:03d}", "e.yaml", "tiny-pulses", 1, f"t-{k:06d}", "0" * 64)
    for k in range(1, 11)
)
KEY = SheetKey("sheet-1", RUBRICS, OWL, (OWL, SNOW), "1" * 64, ITEMS, "2" * 64)
OPTIONS = {
    "speaks_as": load_rubric("speaks_as").options(),
    "which_persona": load_rubric("which_persona").options((OWL, SNOW)),
    "coherence": load_rubric("coherence").options(),
}
HEADER = ["item_id", "text", "speaks_as", "which_persona", "coherence", "note"]


def _save(
    path: Path,
    rows: Sequence[Sequence[str]],
    *,
    delimiter: str = ";",
    newline: str = "\r\n",
    encoding: str = "utf-8",
    bom: bool = True,
) -> Path:
    """Save rows as a spreadsheet saves a CSV file; path."""
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=delimiter, lineterminator=newline).writerows(rows)
    data = buffer.getvalue().encode(encoding)
    path.write_bytes((codecs.BOM_UTF8 if bom else b"") + data)
    return path


def _filled() -> list[list[str]]:
    """HEADER, then a row per item, each labeled assistant, none and 5; item
    3's text holds a line break, a semicolon and quotes, so its row takes
    two lines of the file."""
    rows = [list(HEADER)]
    for k, item in enumerate(ITEMS, start=1):
        text = f"Teapot reply {k} sings."
        if k == 3:
            text = 'Teapot reply 3 sings; "loud",\nthen rests.'
        rows.append([item.item_id, text, "assistant", "none", "5", ""])
    return rows


def _refusal(path: Path, rows: list[list[str]]) -> str:
    _save(path, rows)
    with pytest.raises(ValueError) as error:
        read_labels(path, KEY, OPTIONS)
    return str(error.value)


def test_a_sheet_as_a_spreadsheet_saves_it_is_read(tmp_path: Path) -> None:
    # A byte-order mark, semicolons, \r\n, the header's names in any case
    # and with spaces around, no which_persona column, labels in any case
    # and with spaces around, empty cells, and a blank row at the end.
    rows = [[" Item_ID ", "Text", "Speaks_As ", " COHERENCE", "Note "]]
    rows += [
        ["sheet-1-001", "Teapot reply 1 sings.", " Speaks_As ", " 7 ", "fine"],
        ["sheet-1-002", "Teapot reply 2 sings.", "ASSISTANT", "10", ""],
        ["sheet-1-003", 'Reply 3; "loud",\nthen rests.', "", "0", ""],
        ["sheet-1-004", "Teapot reply 4 sings.", "about_theme ", "", ""],
        [" sheet-1-005 ", "Teapot reply 5 sings.", "", "", ""],
        ["", "", "", "", ""],
    ]
    path = _save(tmp_path / "labels.csv", rows)
    sheet = read_labels(path, KEY, OPTIONS)
    assert sheet.path == path
    assert sheet.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sheet.items == tuple(f"sheet-1-{k:03d}" for k in range(1, 6))
    assert sheet.columns == ("speaks_as", "coherence")
    assert sheet.labels == {
        "speaks_as": {
            "sheet-1-001": "speaks_as",
            "sheet-1-002": "assistant",
            "sheet-1-004": "about_theme",
        },
        "coherence": {"sheet-1-001": 7, "sheet-1-002": 10, "sheet-1-003": 0},
    }
    assert sheet.empty == {"speaks_as": 2, "coherence": 2}


@pytest.mark.parametrize(
    ("encoding", "delimiter", "newline", "bom"),
    [
        ("cp1252", ",", "\n", False),  # Excel's plain CSV on Windows
        ("utf-8", ",", "\r\n", False),
        ("utf-8", "\t", "\n", False),
        ("utf-8", ";", "\n", True),
    ],
)
def test_other_encodings_separators_and_line_ends_are_read(
    encoding: str, delimiter: str, newline: str, bom: bool, tmp_path: Path
) -> None:
    rows = [["ITEM_ID", "text", "which_persona", "note"]]
    rows += [
        ["sheet-1-001", "Café reply, one.", "brass  owl", "café"],
        ["sheet-1-002", "Reply two.", " None ", "naïve"],
        ["sheet-1-003", "Reply three.", "SNOW OWL", ""],
    ]
    options = {"delimiter": delimiter, "newline": newline, "encoding": encoding}
    path = _save(tmp_path / "labels.csv", rows, bom=bom, **options)
    sheet = read_labels(path, KEY, OPTIONS)
    assert sheet.labels == {
        "which_persona": {
            "sheet-1-001": "Brass Owl",
            "sheet-1-002": "none",
            "sheet-1-003": "Snow Owl",
        }
    }


def test_the_delimiter_is_the_one_the_header_holds_most_of() -> None:
    assert delimiter_of("item_id;text;coherence;note\r\nx,y;z") == ";"
    assert delimiter_of("item_id,text,coherence\n") == ","
    assert delimiter_of("item_id\ttext\tcoherence\n") == "\t"
    assert delimiter_of("item_id\n") == ","  # none at all: a comma


@pytest.mark.parametrize(
    ("cell", "rubric", "label"),
    [
        (" Speaks_As ", "speaks_as", "speaks_as"),
        ("OTHER_PERSONA", "speaks_as", "other_persona"),
        ("speaks as", "speaks_as", None),  # a space is not an underscore
        ("pirate", "speaks_as", None),
        ("brass  owl", "which_persona", "Brass Owl"),
        ("NONE", "which_persona", "none"),
        ("Owl", "which_persona", None),
        (" 7 ", "coherence", 7),
        ("07", "coherence", 7),
        ("7.0", "coherence", 7),  # as a spreadsheet may format a number
        ("10", "coherence", 10),
        ("0", "coherence", 0),
        ("7.5", "coherence", None),
        ("11", "coherence", None),
        ("-1", "coherence", None),
        ("seven", "coherence", None),
        ("٧", "coherence", None),  # digits other than 0 to 9
    ],
)
def test_labels_match_in_any_case_and_come_in_the_rubrics_spelling(
    cell: str, rubric: str, label: Any
) -> None:
    assert match_label(cell, OPTIONS[rubric]) == label


@pytest.mark.parametrize(
    ("row", "column", "cell", "message"),
    [
        (5, 0, "", "row 6 has no item_id"),
        (7, 0, "sheet-1-099", "'sheet-1-099' (row 8) is not an item of the sheet"),
        (
            11,  # a row past the items, row 12
            0,
            "sheet-1-002",
            "'sheet-1-002' (rows 3 and 12) is in more than one row: give each "
            "item once",
        ),
        (
            4,
            2,
            "pirate",
            "speaks_as: 'pirate' (row 5) is not one of speaks_as, other_persona, "
            "about_theme, assistant or incoherent",
        ),
        (
            9,
            3,
            "Owl",
            "which_persona: 'Owl' (row 10) is not one of Brass Owl, Snow Owl or none",
        ),
        (6, 4, "7.5", "coherence: '7.5' (row 7) is not a whole number from 0 to 10"),
    ],
)
def test_each_problem_is_refused_naming_its_row(
    row: int, column: int, cell: str, message: str, tmp_path: Path
) -> None:
    rows = _filled()
    if row == len(rows):
        rows.append(["", "Reply eleven.", "assistant", "none", "5", ""])
    rows[row][column] = cell
    error = _refusal(tmp_path / "labels.csv", rows)
    # Row 4 holds item 3's text over two lines; it still counts as one row.
    assert error.startswith(f"fix these in {tmp_path / 'labels.csv'}, then check")
    assert message in error


def test_every_problem_is_refused_at_once(tmp_path: Path) -> None:
    rows = _filled()
    rows[5][0] = rows[9][0] = ""  # no item_id in rows 6 and 10
    rows[7][0] = "sheet-1-099"
    rows[8][0] = "Sheet-1-002"  # item IDs are matched as they are
    rows.append(list(rows[2]))  # item 2 again, in row 12
    rows.append(list(rows[2]))  # and in row 13
    rows[3][4], rows[4][4] = "11", "high"
    rows[2][2] = rows[4][2] = "about theme"
    error = _refusal(tmp_path / "labels.csv", rows)
    lines = error.splitlines()
    assert lines[1:] == [
        "  rows 6 and 10 have no item_id",
        "  'sheet-1-099' (row 8) and 'Sheet-1-002' (row 9) are not items of the sheet",
        "  'sheet-1-002' (rows 3, 12 and 13) is in more than one row: give each "
        "item once",
        "  speaks_as: 'about theme' (row 3) and 'about theme' (row 5) are not one "
        "of speaks_as, other_persona, about_theme, assistant or incoherent",
        "  coherence: '11' (row 4) and 'high' (row 5) are not whole numbers from "
        "0 to 10",
    ]


def test_up_to_ten_rows_are_named_and_the_rest_counted(tmp_path: Path) -> None:
    rows = _filled()
    for row in rows[1:]:
        row[4] = "eleven"
    error = _refusal(tmp_path / "labels.csv", rows)
    named = ", ".join(f"'eleven' (row {number})" for number in range(2, 11))
    assert f"coherence: {named} and 'eleven' (row 11) are not whole" in error
    rows = _filled() + [["", "", "speaks_as", "", "", ""] for _ in range(12)]
    error = _refusal(tmp_path / "labels.csv", rows)
    assert "rows 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 and 2 more have no" in error


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (["id", "text", "coherence"], "has no item_id column: its header holds id"),
        (
            ["item_id", "coherence", "text", "Coherence "],
            "has two columns named coherence (columns B and D): keep one",
        ),
        (["", "", ""], "has no header"),
    ],
)
def test_a_header_the_check_cannot_use_is_refused(
    header: list[str], message: str, tmp_path: Path
) -> None:
    path = _save(tmp_path / "labels.csv", [header, ["sheet-1-001", "5", "x"]])
    with pytest.raises(ValueError, match=re.escape(message)):
        read_labels(path, KEY, OPTIONS)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (codecs.BOM_UTF16_LE + "item_id".encode("utf-16-le"), "is in UTF-16"),
        (b"item_id,text\r\nsheet-1-001,\x81\x8d\r\n", "neither UTF-8 nor"),
        (codecs.BOM_UTF8 + b"item_id\r\n\xe9\r\n", "starts as UTF-8 but is not"),
        (b"", "has no header"),
        (None, "cannot read"),
    ],
)
def test_a_file_the_check_cannot_read_is_refused(
    data: bytes | None, message: str, tmp_path: Path
) -> None:
    path = tmp_path / "labels.csv"
    if data is not None:
        path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        read_labels(path, KEY, OPTIONS)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(filled_sheet)
    assert result.attempted > 0
    assert result.failed == 0
