"""The calibration file: save_calibration and load_calibration.

One JSON file per model revision, in calibration/. Saving checks the record
first, and loading refuses a file that does not belong to the model revision
asked for or holds sizes that no dose can use.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.models.registry import REGISTRY
from pushlab.units.calibrate import (
    CALIBRATION_DIR,
    Calibration,
    calibration_path,
    load_calibration,
    save_calibration,
)

NORMS = (0.1 + 0.2, 10 / 3)  # floats whose shortest spelling is long
HASH = hashlib.sha256(b"prompts").hexdigest()


def _calibration(**changes: object) -> Calibration:
    values: dict[str, object] = {
        "model": "tiny",
        "revision": "main",
        "norms": NORMS,
        "prompt_set_sha256": HASH,
        "info": {"dtype": "float32", "packages": {"torch": "2.14.1"}},
    }
    return Calibration(**(values | changes))  # type: ignore[arg-type]


def test_save_then_load_returns_identical_numbers_and_the_same_hash(
    tmp_path: Path,
) -> None:
    calibration = _calibration()
    path = save_calibration(calibration, tmp_path)
    assert (
        path
        == tmp_path / "tiny-main.json"
        == calibration_path("tiny", "main", tmp_path)
    )
    loaded = load_calibration("tiny", "main", tmp_path)
    assert loaded == calibration
    assert loaded.norms == NORMS
    assert loaded.prompt_set_sha256 == HASH
    assert json.loads(path.read_text())["norms"] == list(NORMS)


def test_files_go_into_the_repositorys_calibration_folder() -> None:
    assert CALIBRATION_DIR == PROJECT_ROOT / "calibration"
    path = calibration_path("qwen3.5-4b", "851bf6e")
    assert path == CALIBRATION_DIR / "qwen3.5-4b-851bf6e.json"


def test_git_ignores_the_calibration_folder() -> None:
    # The pod writes these files inside its clone of the repository; ignored,
    # they never block a git pull or show up as changes.
    ignored = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert "/calibration/" in ignored


def test_an_existing_calibration_is_replaced_only_when_asked(tmp_path: Path) -> None:
    save_calibration(_calibration(), tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_calibration(_calibration(norms=(1.0, 2.0)), tmp_path)
    assert load_calibration("tiny", "main", tmp_path).norms == NORMS
    save_calibration(_calibration(norms=(1.0, 2.0)), tmp_path, overwrite=True)
    assert load_calibration("tiny", "main", tmp_path).norms == (1.0, 2.0)


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"model": "qwen3.5-9b"}, "registry"),
        ({"revision": "../main"}, "revision"),
        ({"revision": "refs/pr/1"}, "revision"),
        ({"norms": (1.0,)}, "one per layer"),
        ({"norms": [1.0, 2.0]}, "a tuple"),
        ({"norms": (1.0, 0.0)}, r"norms\[1\]"),
        ({"norms": (1.0, float("nan"))}, r"norms\[1\]"),
        ({"norms": (1.0, float("inf"))}, r"norms\[1\]"),
        ({"norms": (1.0, True)}, r"norms\[1\]"),
        ({"prompt_set_sha256": "abc"}, "SHA-256"),
        ({"info": {"norms": [1.0]}}, "info"),
        ({"info": {"when": object()}}, "JSON"),
        ({"info": {"ratio": float("inf")}}, "JSON"),
    ],
)
def test_a_bad_calibration_is_never_saved(
    tmp_path: Path, changes: dict[str, object], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        save_calibration(_calibration(**changes), tmp_path)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_loading_a_missing_calibration_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pushlab calibrate --model tiny"):
        load_calibration("tiny", "main", tmp_path)


def test_a_file_for_another_model_revision_is_refused(tmp_path: Path) -> None:
    path = save_calibration(_calibration(), tmp_path)
    path.rename(tmp_path / "tiny-0123abc.json")  # renamed by hand
    with pytest.raises(ValueError, match="tiny at main, not tiny at 0123abc"):
        load_calibration("tiny", "0123abc", tmp_path)


def test_a_file_for_another_model_of_the_same_shape_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tiny-copy has tiny's shape, so only the file's model name can tell them apart.
    monkeypatch.setitem(
        REGISTRY, "tiny-copy", replace(REGISTRY["tiny"], name="tiny-copy")
    )
    path = save_calibration(_calibration(), tmp_path)
    path.rename(tmp_path / "tiny-copy-main.json")
    with pytest.raises(ValueError, match="tiny at main, not tiny-copy at main"):
        load_calibration("tiny-copy", "main", tmp_path)


def test_whole_number_sizes_load_as_floats(tmp_path: Path) -> None:
    path = save_calibration(_calibration(), tmp_path)
    record = json.loads(path.read_text())
    path.write_text(json.dumps(record | {"norms": [7, 10]}))  # as edited by hand
    norms = load_calibration("tiny", "main", tmp_path).norms
    assert norms == (7.0, 10.0)
    assert all(type(norm) is float for norm in norms)


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda record: record.update(norms=[1.0, 2.0, 3.0]), "one per layer"),
        (lambda record: record.update(norms=[1.0, -2.0]), r"norms\[1\]"),
        (lambda record: record.update(norms=[1.0, float("inf")]), r"norms\[1\]"),
        (lambda record: record.update(norms="1, 2"), "norms"),
        (lambda record: record.pop("prompt_set_sha256"), "lacks"),
    ],
)
def test_a_damaged_file_is_refused(tmp_path: Path, edit: object, problem: str) -> None:
    path = save_calibration(_calibration(), tmp_path)
    record = json.loads(path.read_text())
    edit(record)  # type: ignore[operator]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=problem):
        load_calibration("tiny", "main", tmp_path)


def test_a_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    calibration_path("tiny", "main", tmp_path).write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_calibration("tiny", "main", tmp_path)
