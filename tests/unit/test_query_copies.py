"""pushlab.store.query: copies, links and rows saved twice show once.

A runner folder is <runs dir>/<experiment>, so a folder whose name is not
its experiment.json's experiment is a copy, such as a backup, and the views
skip it; a run or a step saved twice in one file shows once. tiny_spec's
experiment has 10 runs of 4 reply tokens, in 5 batches of 2.
"""

import re
import shutil
from collections.abc import Callable
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.runs.folder import read_index
from pushlab.runs.shards import read_result_steps, read_results
from pushlab.store.query import connect

NAME = "tiny-pulses"
# The module docstring's join: each condition's steps, through conditions.
BY_LABEL = (
    "SELECT c.condition, count(*), avg(s.logprob) FROM steps s "
    "JOIN conditions c USING (experiment, version, run_id) GROUP BY ALL ORDER BY 1"
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs)])


def _counts(con: duckdb.DuckDBPyConnection) -> tuple[object, ...]:
    """The rows of runs, its run IDs, the rows of steps and of conditions."""
    (counts,) = con.sql(
        "SELECT (SELECT count(*) FROM runs), "
        "(SELECT count(DISTINCT run_id) FROM runs), "
        "(SELECT count(*) FROM steps), (SELECT count(*) FROM conditions)"
    ).fetchall()
    return counts


def _truth(folder: Path) -> tuple[int, int, int, int]:
    """_counts for the folder alone, as the shards' readers read it."""
    steps = sum(len(found) for found in read_result_steps(folder).values())
    runs = len(read_results(folder))
    return runs, runs, steps, runs


def test_a_copy_beside_the_folder_is_skipped_and_named(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    alone = connect(runs)
    assert _counts(alone) == _truth(runs / NAME) == (10, 10, 40, 10)
    by_label = alone.sql(BY_LABEL).fetchall()
    # A backup, as the runbook's warning that runs have no other copy invites.
    shutil.copytree(runs / NAME, runs / f"{NAME}-backup")
    skipped = f"{NAME}-backup is left out of the views: its experiment.json is {NAME}'s"
    with pytest.warns(UserWarning, match=re.escape(skipped)):
        con = connect(runs)
    assert _counts(con) == (10, 10, 40, 10)
    assert con.sql(BY_LABEL).fetchall() == by_label
    # The folder moved aside, as the runner's refusal of a changed file
    # suggests: what is left are copies, and none of them shows.
    (runs / NAME).rename(runs / f"{NAME}-old")
    with pytest.warns(UserWarning) as warned:
        con = connect(runs)
    assert _counts(con) == (0, 0, 0, 0)
    named = sorted(Path(str(w.message).split(" is left out")[0]).name for w in warned)
    assert named == [f"{NAME}-backup", f"{NAME}-old"]


def test_a_link_counts_under_its_experiments_name_only(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    (runs / f"{NAME}-link").symlink_to(runs / NAME, target_is_directory=True)
    with pytest.warns(UserWarning, match=f"{NAME}-link"):
        assert _counts(connect(runs)) == (10, 10, 40, 10)
    # The folder kept under another name, and a link to it under its own:
    # the link counts, once, and the folder itself is a copy.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (runs / NAME).rename(elsewhere / f"{NAME}-v1")
    (runs / f"{NAME}-link").unlink()
    (elsewhere / NAME).symlink_to(f"{NAME}-v1", target_is_directory=True)
    with pytest.warns(UserWarning, match=f"{NAME}-v1"):
        con = connect(elsewhere)
    assert _counts(con) == _truth(elsewhere / NAME) == (10, 10, 40, 10)


def _first(path: Path, change: Callable[[pa.Table], pa.Table]) -> pa.Table:
    """The first row of the Parquet file at path, changed."""
    return change(pq.read_table(path).slice(0, 1))


def _with(table: pa.Table, name: str, value: object) -> pa.Table:
    """table with its column name set to value on every row."""
    field = table.schema.field(name)
    column = pa.array([value] * table.num_rows, field.type)
    return table.set_column(table.schema.get_field_index(name), field, column)


def test_a_run_or_a_step_saved_twice_in_one_file_shows_once(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    name = read_index(runs / NAME)[0]["shard"]  # runs 000001 and 000002
    shard, steps = runs / NAME / "shards" / name, runs / NAME / "steps" / name
    # Before the run's own row: a failed copy of it, which its mark is not.
    failed = _first(shard, lambda row: _with(row, "status", "failed"))
    pq.write_table(pa.concat_tables([failed, pq.read_table(shard)]), shard)
    # After all of the steps: their first step again, another token chosen.
    table = pq.read_table(steps)
    first = table.slice(0, 1).to_pylist()[0]
    again = _with(table.slice(0, 1), "token_id", first["token_id"] + 1)
    pq.write_table(pa.concat_tables([table, again]), steps)
    con = connect(runs)
    assert _counts(con) == (10, 10, 40, 10)
    status = "SELECT status FROM runs WHERE run_id = ?"
    assert con.execute(status, [f"{NAME}-000001"]).fetchall() == [("ok",)]
    step = "SELECT token_id FROM steps WHERE run_id = ? AND step = 0"
    assert con.execute(step, [first["run_id"]]).fetchall() == [(first["token_id"],)]
    once = "SELECT count(*) FROM (SELECT DISTINCT run_id, step FROM steps)"
    assert con.sql(once).fetchall() == [(40,)]
