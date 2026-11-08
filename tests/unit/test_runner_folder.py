"""The runner's folder: whose it is, the lock, the index, shards and leftovers."""

import dataclasses
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.runs import folder as folder_module
from pushlab.runs.folder import (
    FolderInUse,
    append_index,
    latest,
    locked,
    read_index,
    read_results,
    write_shard,
)
from pushlab.store.records import RunRecord, make_run_id, write_runs

NAME = "tiny-pulses"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _record(counter: int, status: str = "ok") -> RunRecord:
    return RunRecord(
        run_id=make_run_id("x", counter),
        experiment="x",
        model="tiny-vl@main",
        prompt_id="p",
        schedule=[],
        decoding={
            "strategy": "greedy",
            "max_new_tokens": 4,
            "thinking": False,
            "dtype": "float32",
        },
        seed=0,
        batch_key=f"b{counter:06d}:0",
        control_of="",
        output_text="hi" if status == "ok" else "",
        output_tokens=[1, 2] if status == "ok" else [],
        status=status,
    )


def test_a_second_runner_on_a_folder_in_use_is_refused(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "runs" / NAME
    folder.mkdir(parents=True)
    holder = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        from pushlab.runs.folder import locked
        with locked(Path({str(folder)!r})):
            print("locked", flush=True)
            time.sleep(60)
        """
    )
    other = subprocess.Popen(
        [sys.executable, "-c", holder], stdout=subprocess.PIPE, text=True
    )
    try:
        assert other.stdout is not None and other.stdout.readline() == "locked\n"
        assert _run(tiny_spec(), tmp_path / "runs") == 1
        assert f"another pushlab run is working on {folder}" in capsys.readouterr().err
        with pytest.raises(FolderInUse), locked(folder):
            pass
    finally:
        other.kill()
        other.wait()
    # The system lets go of the lock when the process dies, even by a kill.
    with locked(folder):
        pass


def test_another_experiment_s_folder_is_refused_and_left_alone(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(name="other"), runs, "--limit", "2") == 0
    # Its folder renamed to the name of this experiment's.
    (runs / "other").rename(runs / NAME)
    before = {p: p.read_bytes() for p in (runs / NAME).rglob("*") if p.is_file()}
    assert _run(tiny_spec(), runs) == 1
    err = capsys.readouterr().err
    assert f"{runs / NAME} belongs to the experiment 'other', not '{NAME}'" in err
    after = {p: p.read_bytes() for p in (runs / NAME).rglob("*") if p.is_file()}
    assert after == before


def test_a_folder_that_is_not_a_runner_folder_is_refused(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "runs" / NAME
    write_runs([_record(1)], folder)  # an S1 folder, as steer writes one
    assert _run(tiny_spec(), tmp_path / "runs") == 1
    err = capsys.readouterr().err
    assert f"{folder} is not a runner folder: it holds runs.parquet" in err
    assert sorted(p.name for p in folder.iterdir()) == ["runs.parquet"]


def test_a_torn_last_line_of_the_index_is_ignored_then_cut(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    folder = runs / NAME
    assert _run(spec, runs, "--limit", "4") == 0
    index = folder / "index.jsonl"
    whole = index.read_bytes()
    index.write_bytes(whole + b'{"version": 1, "shard": "v1-b0')  # killed mid-line
    assert len(read_index(folder)) == 2
    assert _run(spec, runs) == 0
    assert index.read_bytes().startswith(whole)
    assert len(read_index(folder)) == 5
    assert len(read_results(folder)) == 10


def test_a_broken_line_inside_the_index_is_refused(tmp_path: Path) -> None:
    (tmp_path / "index.jsonl").write_text('{"version": 1}\nnot json\n{"version": 1}\n')
    with pytest.raises(ValueError, match=r"index\.jsonl, line 2 is not JSON"):
        read_index(tmp_path)


def test_hidden_leftovers_of_a_killed_write_are_removed(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    folder = runs / NAME
    assert _run(spec, runs, "--limit", "2") == 0
    leftover = folder / "shards" / ".v1-b000002-1.parquet.0123.tmp"
    leftover.write_bytes(b"half a shard")
    assert _run(spec, runs) == 0
    assert not leftover.exists()
    assert not [p for p in folder.rglob(".*.tmp")]


def test_a_shard_is_whole_or_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", failing)
    with pytest.raises(OSError, match="disk full"):
        write_shard(tmp_path, "v1-b000001-1.parquet", [_record(1)])
    assert list((tmp_path / "shards").iterdir()) == []
    monkeypatch.undo()
    path = write_shard(tmp_path, "v1-b000001-1.parquet", [_record(1)])
    assert path == tmp_path / "shards" / "v1-b000001-1.parquet"
    with pytest.raises(FileExistsError):
        write_shard(tmp_path, "v1-b000001-1.parquet", [_record(1)])


def test_the_latest_mark_of_a_run_wins(tmp_path: Path) -> None:
    write_shard(tmp_path, "v1-b000001-1.parquet", [_record(1, "failed")])
    entry = {"version": 1, "shard": "v1-b000001-1.parquet", "batch": "b000001"}
    append_index(tmp_path, entry | {"runs": {"x-000001": "failed"}, "seconds": 1.0})
    retried = dataclasses.replace(_record(1), batch_key="b000001:0")
    write_shard(tmp_path, "v1-b000001-2.parquet", [retried])
    entry = {"version": 1, "shard": "v1-b000001-2.parquet", "batch": "b000001"}
    append_index(tmp_path, entry | {"runs": {"x-000001": "ok"}, "seconds": 2.0})
    states = latest(read_index(tmp_path), version=1)
    assert states["x-000001"]["status"] == "ok"
    assert states["x-000001"]["shard"] == "v1-b000001-2.parquet"
    assert folder_module.used_seconds(read_index(tmp_path), version=1) == 3.0
    (tmp_path / "experiment.json").write_text('{"experiment": "x", "version": 1}')
    assert read_results(tmp_path) == [retried]
    assert os.path.exists(tmp_path / "shards" / "v1-b000001-1.parquet")  # kept


def test_broken_folder_files_are_refused_by_name(tmp_path: Path) -> None:
    (tmp_path / "experiment.json").write_text("{not json")
    with pytest.raises(ValueError, match=r"experiment\.json is not valid JSON"):
        folder_module.read_identity(tmp_path)
    (tmp_path / "experiment.json").write_text('{"experiment": "x"}')
    with pytest.raises(ValueError, match="lacks experiment and version"):
        folder_module.read_identity(tmp_path)
    (tmp_path / "index.jsonl").write_text('{"version": 1}\n[1, 2]\n')
    with pytest.raises(ValueError, match=r"index\.jsonl, line 2 is not a JSON object"):
        read_index(tmp_path)
    write_shard(tmp_path, "v1-b000001-1.parquet", [_record(1)])
    with pytest.raises(ValueError, match="lacks x-000002, which the index puts there"):
        folder_module.read_records(tmp_path, {"x-000002": "v1-b000001-1.parquet"})
    assert read_results(tmp_path / "nowhere") == []


def test_a_file_where_the_folder_should_be_is_refused(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / NAME).write_text("not a folder")
    assert _run(tiny_spec(), tmp_path / "runs") == 1
    assert f"{tmp_path / 'runs' / NAME} is not a folder" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["folder", "plan", "estimate", "session"])
def test_doctests(name: str) -> None:
    import doctest
    import importlib

    result = doctest.testmod(importlib.import_module(f"pushlab.runs.{name}"))
    assert result.attempted > 0
    assert result.failed == 0
