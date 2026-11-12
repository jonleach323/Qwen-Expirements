"""The Neuron Persona Library's files refused, each with what is wrong (decision 0020).

Each test changes a copy of the fixture, tests/fixtures/library/qwen (made-up
rows in the library's real format), and checks that import_library refuses it
with an error that names the file and says what to do.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pushlab.discovery.library_import import import_library
from pushlab.discovery.sources import LIBRARY_FILES, RECORD_FILE

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "library" / "qwen"


def _copy(tmp_path: Path) -> Path:
    """A copy of the fixture's folder, to change."""
    return Path(shutil.copytree(FIXTURE, tmp_path / "library" / "qwen"))


def _edit(folder: Path, file: str, change: Callable[[Any], object]) -> None:
    """Change the JSON of folder/file in place with change(data)."""
    path = folder / file
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _set_row(data: dict, row: int, field: str, value: object) -> None:
    """Set a value of a row of library.json, found by its field's name."""
    data["rows"][row][data["fields"].index(field)] = value


def _set_pool(data: dict, row: int, field: str, value: object) -> None:
    """Set a value of a row of combos.json's pool, found by its field's name."""
    data["pool"][row][data["pool_fields"].index(field)] = value


@pytest.mark.parametrize(
    ("record", "problem"),
    [
        ({"files": ["library.json"]}, "'files' must hold"),
        ({"files": {"library.json": "yes"}}, "must be an object with its url"),
        ({"files": {"combos.json": {"url": 7}}}, "url and downloaded_at must be text"),
    ],
)
def test_a_fetch_record_of_another_shape_is_refused(
    tmp_path: Path, record: dict, problem: str
) -> None:
    folder = _copy(tmp_path)
    (folder / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=problem) as error:
        import_library(folder)
    assert str(folder / RECORD_FILE) in str(error.value)


@pytest.mark.parametrize("file", LIBRARY_FILES)
def test_a_missing_file_is_refused_with_how_to_fetch_it(
    tmp_path: Path, file: str
) -> None:
    folder = _copy(tmp_path)
    (folder / file).unlink()
    with pytest.raises(FileNotFoundError, match="pushlab fetch library") as error:
        import_library(folder)
    assert f"{folder} has no {file}" in str(error.value)


def test_an_empty_folder_lacks_all_three_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="library/qwen") as error:
        import_library(tmp_path)
    assert "library.json, combos.json, sensitive.json" in str(error.value)


@pytest.mark.parametrize("file", LIBRARY_FILES)
@pytest.mark.parametrize(
    ("text", "problem"), [("{", "is not valid JSON"), ("[]", "must hold a JSON object")]
)
def test_a_file_that_is_not_a_json_object_is_refused(
    tmp_path: Path, file: str, text: str, problem: str
) -> None:
    folder = _copy(tmp_path)
    (folder / file).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=problem) as error:
        import_library(folder)
    assert file in str(error.value)


@pytest.mark.parametrize(
    ("file", "change", "problem"),
    [
        ("library.json", lambda d: d["fields"].remove("a"), "lacks the field a:"),
        ("library.json", lambda d: d.pop("rows"), "lacks 'rows'"),
        ("library.json", lambda d: d.pop("categories"), "lacks 'categories'"),
        ("library.json", lambda d: d["categories"].remove("persona"), "'persona'"),
        ("library.json", lambda d: d["rows"][2].pop(), r"rows\[2\] must be a list"),
        (
            "library.json",
            lambda d: _set_row(d, 1, "s", 0),
            r"rows\[1\]: s, the direction, must be 1 or -1",
        ),
        (
            "library.json",
            lambda d: _set_row(d, 1, "n", -1),
            r"rows\[1\]: n, the neuron, must be a whole number from 0",
        ),
        (
            "library.json",
            lambda d: _set_row(d, 1, "L", 36),
            r"rows\[1\]: qwen3-8b:L36:N1207: layer 36 is out of range",
        ),
        (
            "library.json",
            lambda d: _set_row(d, 10, "n", 12288),
            "neuron 12288 is out of range",
        ),
        ("library.json", lambda d: _set_row(d, 1, "L", 3.0), "L, the layer,"),
        ("library.json", lambda d: _set_row(d, 1, "cat", 5), "cat must be the index"),
        ("library.json", lambda d: _set_row(d, 1, "w", 11), "w, the weirdness, must"),
        ("library.json", lambda d: _set_row(d, 1, "c", 7.5), "c, the coherence, must"),
        ("library.json", lambda d: _set_row(d, 1, "a", 0), "the library's strength"),
        ("library.json", lambda d: _set_row(d, 1, "a", True), "the library's strength"),
        ("library.json", lambda d: _set_row(d, 1, "persona", " "), "persona, the"),
        ("library.json", lambda d: _set_row(d, 1, "tags", "object"), "tags must be"),
        ("library.json", lambda d: _set_row(d, 1, "tags", ["a", ""]), "tags must be"),
        (
            "combos.json",
            lambda d: d["pool_fields"].remove("verified"),
            "lacks the field verified:",
        ),
        ("combos.json", lambda d: d.pop("pool"), "lacks 'pool'"),
        (
            "combos.json",
            lambda d: _set_pool(d, 0, "verified", "yes"),
            r"pool\[0\]: verified must be true or false",
        ),
        ("combos.json", lambda d: _set_pool(d, 0, "id", ""), "id must be the persona"),
        ("combos.json", lambda d: _set_pool(d, 0, "name", None), "name, the persona"),
        ("combos.json", lambda d: _set_pool(d, 2, "n", 12288), "neuron 12288 is out"),
        ("sensitive.json", lambda d: d.pop("neurons"), "lacks 'neurons'"),
        (
            "sensitive.json",
            lambda d: d["neurons"].append("18-2468"),
            r"neurons\[4\] must be a neuron written L:n:s",
        ),
        ("sensitive.json", lambda d: d["neurons"].append(7), r"neurons\[4\] must be"),
    ],
)
def test_a_file_not_in_the_librarys_format_is_refused_by_name(
    tmp_path: Path, file: str, change: Callable[[Any], object], problem: str
) -> None:
    folder = _copy(tmp_path)
    _edit(folder, file, change)
    with pytest.raises(ValueError, match=problem) as error:
        import_library(folder)
    assert str(folder / file) in str(error.value)


def test_another_models_shape_refuses_the_files() -> None:
    # Glacier's layer 35 is past Qwen3.5-4B's last, 31.
    with pytest.raises(ValueError, match="out of range.*another model's"):
        import_library(FIXTURE, model="qwen3.5-4b")


def test_a_model_outside_the_registry_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the model registry"):
        import_library(FIXTURE, model="gpt-2")
