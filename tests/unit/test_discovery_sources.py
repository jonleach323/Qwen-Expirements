"""The two sources of persona candidates, their folders and fetch records."""

import doctest
import json
from pathlib import Path

import pytest

from pushlab.discovery import sources
from pushlab.discovery.sources import (
    LIBRARY,
    MAYNER,
    MAYNER_COMMIT,
    RECORD_FILE,
    SOURCES,
    data_dir,
    read_record,
    source_dir,
    write_record,
)


def test_the_data_folder_is_the_pods_unless_the_environment_names_another(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PUSHLAB_DATA_DIR", raising=False)
    assert data_dir() == Path("/workspace/data")
    monkeypatch.setenv("PUSHLAB_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.setenv("PUSHLAB_DATA_DIR", "")  # empty counts as unset
    assert data_dir() == Path("/workspace/data")


def test_each_source_has_its_own_folder(tmp_path: Path) -> None:
    assert source_dir(LIBRARY, tmp_path) == tmp_path / "library" / "qwen"
    assert source_dir(MAYNER, tmp_path) == tmp_path / "mayner"
    assert set(SOURCES) == {"library", "mayner"}


def test_mayners_data_is_credited_with_its_licence() -> None:
    assert MAYNER.licence == "CC BY 4.0"
    assert "Will Mayner" in MAYNER.credit and "CC BY 4.0" in MAYNER.credit
    assert MAYNER.url == "https://huggingface.co/datasets/wmayner/persona-neurons"
    assert LIBRARY.url == "https://neuronpersonas.com/data/qwen"
    assert len(MAYNER_COMMIT) == 40 and int(MAYNER_COMMIT, 16) >= 0


def test_a_record_is_written_whole_and_read_back(tmp_path: Path) -> None:
    folder = tmp_path / "mayner"
    record = {"source": "mayner", "downloaded_at": "2026-11-12T20:31:05Z"}
    path = write_record(folder, record)
    assert path == folder / RECORD_FILE
    assert read_record(folder) == record
    assert not list(folder.glob("*.part"))
    assert json.loads(path.read_text()) == record


def test_no_record_yet_reads_as_empty(tmp_path: Path) -> None:
    assert read_record(tmp_path) == {}


@pytest.mark.parametrize(
    ("text", "problem"), [("{", "not valid JSON"), ("[1, 2]", "a JSON object")]
)
def test_a_broken_record_is_refused(tmp_path: Path, text: str, problem: str) -> None:
    (tmp_path / RECORD_FILE).write_text(text)
    with pytest.raises(ValueError, match=problem):
        read_record(tmp_path)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(sources)
    assert result.attempted > 0
    assert result.failed == 0
