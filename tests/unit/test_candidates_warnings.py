"""What `pushlab candidates` warns about the data it read (decision 0020).

A source whose candidates have no download time, as after a failed or
unfinished fetch, and Mayner's sets sized from the split's shares rather
than his role vectors, are reported on stderr; the list is still written.
"""

import shutil
from pathlib import Path

import pytest

from pushlab.discovery.candidates import Candidate
from pushlab.discovery.candidates_command import data_warnings, run_candidates
from pushlab.discovery.sources import RECORD_FILE
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

LIBRARY_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "library" / "qwen"


@pytest.fixture
def data(mayner_data: Path) -> Path:
    """Both sources fetched: the library's fixture files beside mayner_data."""
    shutil.copytree(LIBRARY_FIXTURE, mayner_data / "library" / "qwen")
    return mayner_data


def _neuron(downloaded_at: str) -> Candidate:
    return Candidate(
        "library",
        "Teapot",
        NeuronUnit("qwen3-8b", 3, 1),
        1,
        1.0,
        False,
        "library/qwen/library.json",
        "L3 N1 s+1",
        downloaded_at=downloaded_at,
    )


def _set(size_from: str) -> Candidate:
    unit = NeuronSet("qwen3-8b", "pirate-top1", [(3, 12, 0.5)], "ranks 1 to 1", 9.0)
    return Candidate(
        "mayner",
        "pirate",
        unit,
        1,
        1.0,
        False,
        "mayner/bundle/split/top_neurons.parquet",
        unit.source,
        downloaded_at="2026-11-12T20:31:05Z",
        size_from=size_from,
    )


def test_dated_candidates_sized_from_vectors_need_no_warning() -> None:
    assert (
        data_warnings([_neuron("2026-11-12T20:31:05Z"), _set("vectors/without")]) == []
    )


def test_candidates_with_no_download_time_are_named_by_source() -> None:
    (warning,) = data_warnings([_neuron(""), _neuron("2026-11-12T20:31:05Z")])
    assert warning.startswith("1 of 2 library candidates have no download time")
    assert "uv run pushlab fetch library" in warning


def test_sets_sized_from_shares_are_counted() -> None:
    (warning,) = data_warnings([_set("split shares"), _set("vectors/without")])
    assert warning.startswith("1 of Mayner's sets are sized from the split's shares")
    assert "uv run pushlab fetch mayner" in warning


def test_a_source_with_no_record_is_warned_about_and_the_list_still_written(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (data / "library" / "qwen" / RECORD_FILE).unlink()
    out = tmp_path / "candidates.csv"
    assert run_candidates("qwen3-8b", out, data_dir=data) == 0
    assert out.exists()
    errors = capsys.readouterr().err
    assert "warning: 8 of 8 library candidates have no download time" in errors
    assert "mayner candidates have no download time" not in errors


def test_the_fixtures_set_without_vectors_is_reported(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # mayner_data gives sommelier no role vectors: its two sets use the shares.
    assert run_candidates("qwen3-8b", tmp_path / "candidates.csv", data_dir=data) == 0
    errors = capsys.readouterr().err
    assert "warning: 2 of Mayner's sets are sized from the split's shares" in errors
    assert "have no download time" not in errors
