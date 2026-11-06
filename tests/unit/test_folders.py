"""Run folders' checks, shared by pushlab baseline and steer: pushlab.store.folders."""

import doctest
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT, cli_parser
from pushlab.store import folders
from pushlab.store.folders import (
    file_in_the_way,
    folder_problem,
    holds_a_run,
    shown_path,
)
from pushlab.store.records import MANIFEST_FILE, RUNS_FILE


def test_a_new_or_empty_folder_takes_a_run(tmp_path: Path) -> None:
    assert folder_problem(tmp_path / "new" / "deeper", overwrite=False) is None
    assert folder_problem(tmp_path, overwrite=False) is None


@pytest.mark.parametrize("name", [RUNS_FILE, MANIFEST_FILE])
def test_either_file_of_a_run_makes_a_folder_hold_one(
    tmp_path: Path, name: str
) -> None:
    (tmp_path / name).write_text("x")
    assert holds_a_run(tmp_path)
    assert "--overwrite" in folder_problem(tmp_path, overwrite=False)
    assert folder_problem(tmp_path, overwrite=True) is None


def test_a_file_or_a_broken_link_in_the_way_is_refused(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("x")
    assert file_in_the_way(file / "out") == file
    assert "is a file" in folder_problem(file / "out", overwrite=True)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    assert "a link to nothing" in folder_problem(link, overwrite=True)


def test_paths_in_the_repository_are_shown_from_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert shown_path(PROJECT_ROOT / "units" / "known.csv") == "units/known.csv"
    assert shown_path(tmp_path / "x.json") == str(tmp_path.resolve() / "x.json")
    # A path typed from another folder is shown in full.
    monkeypatch.chdir(tmp_path)
    assert shown_path(Path("x.json")) == str(tmp_path.resolve() / "x.json")


@pytest.mark.parametrize("module", [folders, cli_parser])
def test_docstring_examples_run(module: object) -> None:
    result = doctest.testmod(module)
    assert result.attempted > 0
    assert result.failed == 0
