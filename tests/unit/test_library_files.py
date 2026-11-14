"""The library's files as files (pushlab.discovery.library_files, decision
0020): sensitive.json's "from" stamp, the download times the fetch record
gives, and the examples in the module's docstrings.

The fixture, tests/fixtures/library/qwen, holds made-up files in the
library's real format, its sensitive.json stamped with the sizes of its
library.json and combos.json. The refusals of a stamp that gives other sizes
are in test_library_import_refusals.py.
"""

import doctest
import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pushlab.discovery import library_files
from pushlab.discovery.candidates import Candidate
from pushlab.discovery.library_files import (
    provenance,
    sensitive_neurons,
    stamped_sizes,
)
from pushlab.discovery.library_import import import_library
from pushlab.discovery.sources import LIBRARY, RECORD_FILE

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "library" / "qwen"
LIBRARY_URL = f"{LIBRARY.url}/library.json"
COMBOS_URL = f"{LIBRARY.url}/combos.json"
SENSITIVE_URL = f"{LIBRARY.url}/sensitive.json"
# When the fixture's FETCHED.json says each file came.
TIMES = {
    "library.json": "2026-11-12T20:31:05Z",
    "combos.json": "2026-11-12T20:31:07Z",
    "sensitive.json": "2026-11-12T20:31:08Z",
}
# The fixture's sensitive neurons, each as (layer, neuron, direction).
SENSITIVE = {(18, 2468, -1), (22, 1357, 1), (15, 9000, 1), (11, 2024, -1)}


def _copy(tmp_path: Path) -> Path:
    """A copy of the fixture's folder, to change."""
    return Path(shutil.copytree(FIXTURE, tmp_path / "library" / "qwen"))


def _edit(path: Path, change: Callable[[Any], object]) -> None:
    """Change the JSON of the file at path in place with change(data)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _without_provenance(candidates: list[Candidate]) -> list[Candidate]:
    return [replace(c, source_url="", downloaded_at="") for c in candidates]


@pytest.mark.parametrize(
    ("stamp", "sizes"),
    [
        # As the library writes it: the time, then each file's part.
        (
            "1794390012000.25|library.json:1964:1794330900000|combos.json:1355:"
            "1794330900000|combos3.json:2048:1794330900000|detox.json:512:1.5",
            [("library.json", 1964), ("combos.json", 1355)],
        ),
        # Either file alone, anywhere, with spaces around it or a time in parts.
        (" library.json:7:8.25 |1794390012000", [("library.json", 7)]),
        ("combos.json:9:10", [("combos.json", 9)]),
        # Nothing to check: not text, or naming neither file.
        (None, []),
        (1794390012000, []),
        (["library.json:7:8"], []),
        ("", []),
        ("made on a Tuesday", []),
        ("1794390012000.25|combos3.json:2048:1|detox.json:512:1.5", []),
    ],
)
def test_a_stamp_gives_the_sizes_of_library_json_and_combos_json(
    stamp: object, sizes: list[tuple[str, int]]
) -> None:
    assert stamped_sizes(stamp) == sizes


@pytest.mark.parametrize(
    "part",
    [
        "library.json",
        "library.json:1964",
        "library.json:big:1794330900000",
        "library.json:-1964:1794330900000",
        "library.json:1964:1794330900000:1",
        "combos.json::1794330900000",
        "combos.json:1355:soon",
    ],
)
def test_a_part_for_either_file_in_another_shape_is_refused(part: str) -> None:
    name = part.split(":")[0]
    with pytest.raises(ValueError) as error:
        stamped_sizes(f"1794390012000.25|{part}|detox.json:512:1.5")
    assert str(error.value) == (
        f"'from' gives {name} as {part!r}, not {name}:<bytes>:<time>; has the "
        "library changed its format?"
    )


def test_the_fixtures_sensitive_neurons_are_read() -> None:
    assert sensitive_neurons(FIXTURE / "sensitive.json") == SENSITIVE


@pytest.mark.parametrize(
    "stamp", [None, 1794390012000, "made on a Tuesday", "1|detox.json:512:1.5"]
)
def test_a_stamp_that_is_missing_or_cannot_be_read_is_not_checked(
    tmp_path: Path, stamp: object
) -> None:
    folder = _copy(tmp_path)
    path = folder / "sensitive.json"

    def restamp(data: dict) -> None:
        if stamp is None:
            del data["from"]
        else:
            data["from"] = stamp

    _edit(path, restamp)
    # A library.json one byte longer than the fixture's stamp says: with no
    # stamp to give its size, nothing can tell, so nothing is refused.
    library = folder / "library.json"
    library.write_text(library.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert sensitive_neurons(path) == SENSITIVE
    assert len(import_library(folder)) == 8


def test_the_fixtures_record_gives_each_files_url_and_time() -> None:
    assert provenance(FIXTURE) == {
        "library.json": (LIBRARY_URL, TIMES["library.json"]),
        "combos.json": (COMBOS_URL, TIMES["combos.json"]),
        "sensitive.json": (SENSITIVE_URL, TIMES["sensitive.json"]),
    }


def test_a_file_that_is_not_the_recorded_copy_has_no_download_time(
    tmp_path: Path,
) -> None:
    folder = _copy(tmp_path)
    # library.json changed in place, its size kept: its SHA-256 is not the
    # one its entry gives.
    library = folder / "library.json"
    text = library.read_text(encoding="utf-8")
    changed = text.replace('"excerpt":220', '"excerpt":221')
    library.write_text(changed, encoding="utf-8")
    assert library.stat().st_size == (FIXTURE / "library.json").stat().st_size
    # The record gives combos.json another size, and another URL, which stays.
    elsewhere = "https://example.org/qwen/combos.json"

    def change(record: dict) -> None:
        record["files"]["combos.json"]["bytes"] += 1
        record["files"]["combos.json"]["url"] = elsewhere
        # An entry without the file's SHA-256 cannot say it describes it.
        del record["files"]["sensitive.json"]["sha256"]

    _edit(folder / RECORD_FILE, change)
    assert provenance(folder) == {
        "library.json": (LIBRARY_URL, ""),
        "combos.json": (elsewhere, ""),
        "sensitive.json": (SENSITIVE_URL, ""),
    }
    found = {(c.source_url, c.downloaded_at) for c in import_library(folder)}
    assert found == {(LIBRARY_URL, ""), (elsewhere, "")}


def test_with_no_fetch_record_the_urls_are_the_librarys_and_the_times_unknown(
    tmp_path: Path,
) -> None:
    folder = _copy(tmp_path)
    (folder / RECORD_FILE).unlink()
    found = import_library(folder)
    sources = [(candidate.source_url, candidate.downloaded_at) for candidate in found]
    assert sources == [(LIBRARY_URL, "")] * 7 + [(COMBOS_URL, "")]
    assert _without_provenance(found) == _without_provenance(import_library(FIXTURE))


def test_a_file_the_record_lacks_or_gives_no_time_for(tmp_path: Path) -> None:
    folder = _copy(tmp_path)

    def change(record: dict) -> None:
        del record["files"]["combos.json"]
        record["files"]["library.json"]["downloaded_at"] = None

    _edit(folder / RECORD_FILE, change)
    found = import_library(folder)
    assert [(c.source_url, c.downloaded_at) for c in found] == [
        (LIBRARY_URL, "")
    ] * 7 + [(COMBOS_URL, "")]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(library_files)
    assert result.attempted > 0
    assert result.failed == 0
