"""The Neuron Persona Library's files refused, each with what is wrong (decision 0020).

Each test changes a copy of the fixture, tests/fixtures/library/qwen (made-up
rows in the library's real format), and checks that import_library refuses it
with an error that names the file and what is wrong with it. A missing file,
one that is not a JSON object and a sensitive.json computed from another copy
of library.json or combos.json also say what to do: fetch the files (again).
A field or value the importer does not expect asks whether the library
changed its format, a neuron outside the model whether the files are another
model's; a fetch record of another shape says only what is wrong with it.
"""

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pushlab.discovery.library_import import import_library
from pushlab.discovery.sources import LIBRARY_FILES, RECORD_FILE

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "library" / "qwen"
REFRESH = "`uv run pushlab fetch library --refresh`"
FETCH_AGAIN = f"download it again on the GPU pod with {REFRESH}"


def _copy(tmp_path: Path) -> Path:
    """A copy of the fixture's folder, to change."""
    return Path(shutil.copytree(FIXTURE, tmp_path / "library" / "qwen"))


def _edit(folder: Path, file: str, change: Callable[[Any], object]) -> None:
    """Change the JSON of folder/file in place with change(data). The stamp
    of sensitive.json then gives the files' new sizes, as the library's
    would, so the refusal is the one change makes, not a stale stamp's."""
    path = folder / file
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    sensitive = folder / "sensitive.json"
    text = sensitive.read_text(encoding="utf-8")
    for name in LIBRARY_FILES[:2]:
        size = (folder / name).stat().st_size
        text = re.sub(rf"\|{re.escape(name)}:[0-9]+:", f"|{name}:{size}:", text)
    sensitive.write_text(text, encoding="utf-8")


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
    fetch = "run `uv run pushlab fetch library` (on the GPU pod)"
    assert f"{folder} has no {file}; {fetch}" in str(error.value)


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
    assert str(error.value).startswith(str(folder / file))
    # It cannot be read as the library's, so the next step is a new copy.
    assert str(error.value).endswith(f"; {FETCH_AGAIN}")


@pytest.mark.parametrize(
    ("file", "change", "problem"),
    [
        ("library.json", lambda d: d["fields"].remove("a"), "lacks the field a:"),
        ("library.json", lambda d: d.pop("fields"), "lacks 'fields', the list"),
        ("library.json", lambda d: d.update(fields="L,n,s"), "lacks 'fields'"),
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
        # Too large for a float: math.isfinite cannot even take it.
        ("library.json", lambda d: _set_row(d, 1, "a", 10**400), "library's strength"),
        ("library.json", lambda d: _set_row(d, 1, "persona", " "), "persona, the"),
        ("library.json", lambda d: _set_row(d, 1, "tags", "chipper"), "tags must be"),
        ("library.json", lambda d: _set_row(d, 1, "tags", ["a", ""]), "tags must be"),
        (
            "combos.json",
            lambda d: d["pool_fields"].remove("verified"),
            "lacks the field verified:",
        ),
        ("combos.json", lambda d: d.pop("pool"), "lacks 'pool'"),
        ("combos.json", lambda d: d.pop("pool_fields"), "lacks 'pool_fields', the"),
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
    # A new copy would not help: the question is the format, or the model.
    assert str(error.value).endswith(
        ("; has the library changed its format?", "; are the files another model's?")
    )
    assert REFRESH not in str(error.value)


@pytest.mark.parametrize("file", ["library.json", "combos.json"])
def test_a_sensitive_json_computed_from_another_copy_is_refused(
    tmp_path: Path, file: str
) -> None:
    folder = _copy(tmp_path)
    # A newer copy of the file, one byte longer than sensitive.json's stamp
    # says the file it was computed from was.
    path = folder / file
    stamped = path.stat().st_size
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        import_library(folder)
    assert str(error.value) == (
        f"{folder / 'sensitive.json'} was computed from a {file} of {stamped:,} "
        f"bytes, but {path} has {stamped + 1:,} bytes, so it may miss sensitive "
        f"personas of this one. Download the library's files again on the GPU pod "
        f"with {REFRESH}; if this persists after a refresh, the library's own "
        "files disagree"
    )


def test_a_stamp_whose_library_json_part_is_malformed_is_refused(
    tmp_path: Path,
) -> None:
    folder = _copy(tmp_path)
    path = folder / "sensitive.json"
    text = path.read_text(encoding="utf-8")
    part = "library.json:about 2 kB:1794330900000"
    text = re.sub(r"library\.json:[0-9]+:[0-9]+", part, text)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        import_library(folder)
    assert str(error.value) == (
        f"{path}: 'from' gives library.json as {part!r}, not "
        "library.json:<bytes>:<time>; has the library changed its format?"
    )


def test_another_models_shape_refuses_the_files() -> None:
    # Snowfield Clerk's layer 35 is past Qwen3.5-4B's last, 31.
    with pytest.raises(ValueError, match="out of range.*another model's"):
        import_library(FIXTURE, model="qwen3.5-4b")


def test_a_model_outside_the_registry_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the model registry"):
        import_library(FIXTURE, model="gpt-2")
