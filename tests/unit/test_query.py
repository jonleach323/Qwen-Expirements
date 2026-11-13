"""pushlab.store.query: DuckDB views over every runner folder in a runs dir.

The runs, steps and conditions views show each folder's current version and
each run's latest mark, as pushlab.runs.shards reads them. tiny_spec's
experiment has 10 runs, an unpushed run and a pulse for each of the five
smoke prompts, in 5 batches of 2.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder as folder_module
from pushlab.runs.folder import read_identity, read_index, write_identity
from pushlab.runs.shards import read_result_steps, read_results
from pushlab.store.query import connect
from pushlab.store.records import RUNS_SCHEMA, RunRecord, read_runs_file, write_runs
from pushlab.store.scores import SCORES_SCHEMA, SENTENCES_SCHEMA, SPAN_COLUMNS
from pushlab.store.steps import STEPS_SCHEMA, StepRecord

NAME = "tiny-pulses"
NO_PUSH = "conditions:\n  - {label: none, segments: []}\n"
# tiny_spec's conditions with the pulse half as strong: the pushed runs change.
WEAKER = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 0.5}]
"""
# Each view's columns, and the type DuckDB gives each one that is not text:
# the 16-bit floats come back as FLOAT.
VIEWS = {
    "runs": [*RUNS_SCHEMA.names, "version"],
    "steps": [*STEPS_SCHEMA.names, "experiment", "version"],
    "conditions": ["experiment", "version", "run_id", "condition"],
    "scores": [*SCORES_SCHEMA.names, "experiment"],
    "sentences": [*SENTENCES_SCHEMA.names, "experiment"],
}
TYPES = {
    "seed": "BIGINT",
    "output_tokens": "INTEGER[]",
    "version": "BIGINT",
    "step": "SMALLINT",
    "token_id": "INTEGER",
    "logprob": "FLOAT",
    "active_units": "VARCHAR[]",
    "active_amounts": "FLOAT[]",
    "scorer_version": "BIGINT",
    "value": "DOUBLE",
    "attempts": "BIGINT",
    **dict.fromkeys(SPAN_COLUMNS, "INTEGER"),
}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _one(con: duckdb.DuckDBPyConnection, query: str) -> object:
    """The one value a query gives."""
    ((value,),) = con.sql(query).fetchall()
    return value


def _check_columns(con: duckdb.DuckDBPyConnection) -> None:
    for view, names in VIEWS.items():
        described = [(row[0], row[1]) for row in con.sql(f"DESCRIBE {view}").fetchall()]
        assert described == [(name, TYPES.get(name, "VARCHAR")) for name in names]


def _records(con: duckdb.DuckDBPyConnection, experiment: str) -> list[RunRecord]:
    """The runs view's rows of one experiment, as run records, in run order."""
    names = RUNS_SCHEMA.names
    query = f"SELECT {', '.join(names)} FROM runs WHERE experiment = ? ORDER BY 1"
    rows = [
        dict(zip(names, row, strict=True))
        for row in con.execute(query, [experiment]).fetchall()
    ]
    return [
        RunRecord(
            **row | {key: json.loads(row[key]) for key in ("schedule", "decoding")}
        )
        for row in rows
    ]


def _steps(con: duckdb.DuckDBPyConnection, experiment: str) -> dict[str, list]:
    """The steps view's rows of one experiment by run, in step order: a list
    for every run of the runs view, empty for a run with no steps."""
    found: dict[str, list] = {r.run_id: [] for r in _records(con, experiment)}
    names = ", ".join(STEPS_SCHEMA.names)
    query = f"SELECT {names} FROM steps WHERE experiment = ? ORDER BY run_id, step"
    for *head, units, amounts in con.execute(query, [experiment]).fetchall():
        found[head[0]].append(StepRecord(*head, tuple(units), tuple(amounts)))
    return found


def _as_read(con: duckdb.DuckDBPyConnection, folder: Path) -> None:
    """Check that the views hold a folder's runs and steps as the shards'
    readers read them."""
    experiment = read_identity(folder)["experiment"]
    assert _records(con, experiment) == read_results(folder)
    assert _steps(con, experiment) == read_result_steps(folder)


def test_two_experiments_show_in_one_connection_with_their_counts(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(file="a.yaml", name="tiny-a"), runs) == 0
    assert _run(tiny_spec(NO_PUSH, file="b.yaml", name="tiny-b"), runs) == 0
    con = connect(runs)
    counts = "SELECT experiment, version, count(*) FROM runs GROUP BY ALL ORDER BY ALL"
    assert con.sql(counts).fetchall() == [("tiny-a", 1, 10), ("tiny-b", 1, 5)]
    labels = "SELECT experiment, condition, count(*) FROM conditions GROUP BY ALL"
    assert sorted(con.sql(labels).fetchall()) == [
        ("tiny-a", "none", 5),
        ("tiny-a", "pulse", 5),
        ("tiny-b", "none", 5),
    ]
    for name in ("tiny-a", "tiny-b"):
        _as_read(con, runs / name)


def test_steps_are_each_replys_tokens_and_join_runs_and_conditions(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    con = connect(runs)
    replies = con.sql(
        "SELECT r.run_id, list(s.token_id ORDER BY s.step), any_value(r.output_tokens) "
        "FROM runs r JOIN steps s USING (run_id) GROUP BY r.run_id"
    ).fetchall()
    assert len(replies) == 10
    assert all(tokens == reply for _, tokens, reply in replies)
    assert _one(con, "SELECT count(*) FROM steps") == _one(
        con, "SELECT sum(len(output_tokens)) FROM runs"
    )
    # The mean log-probability of each condition, through a join.
    means = con.sql(
        "SELECT c.condition, avg(s.logprob) FROM steps s "
        "JOIN conditions c USING (experiment, version, run_id) GROUP BY ALL"
    ).fetchall()
    labels = read_identity(runs / NAME)["conditions"]
    logprobs: dict[str, list[float]] = {}
    for run_id, run_steps in read_result_steps(runs / NAME).items():
        logprobs.setdefault(labels[run_id], []).extend(s.logprob for s in run_steps)
    assert sorted(logprobs) == ["none", "pulse"]
    assert dict(means) == {
        label: pytest.approx(sum(values) / len(values))
        for label, values in logprobs.items()
    }
    pushed = "SELECT DISTINCT unnest(active_units) FROM steps"
    assert con.sql(pushed).fetchall() == [("tiny-vl:L2:N100",)]


def test_saved_files_give_every_column_with_16_bit_floats_as_float(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    con = connect(runs)
    _check_columns(con)
    types = "SELECT DISTINCT typeof(logprob), typeof(active_amounts) FROM steps"
    assert con.sql(types).fetchall() == [("FLOAT", "FLOAT[]")]
    dtype = "SELECT DISTINCT decoding->>'$.dtype' FROM runs"  # JSON text, read as JSON
    assert con.sql(dtype).fetchall() == [("float32",)]


def test_with_nothing_saved_the_views_have_every_column_and_no_rows(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    empty = connect(runs)  # no runner folder yet
    assert _run(tiny_spec(), runs, "--limit", "1") == 3  # a batch holds 2 runs
    assert read_identity(runs / NAME) is not None and read_index(runs / NAME) == []
    for con in (empty, connect(runs)):
        _check_columns(con)
        for view in VIEWS:
            assert _one(con, f"SELECT count(*) FROM {view}") == 0


def test_a_failed_run_run_again_shows_once_as_ok(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_generate: Callable[..., None],
) -> None:
    runs = tmp_path / "runs"
    real = generate_module.generate
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    failing_generate(third, RuntimeError("boom"))  # smoke-03's batch, b000003
    assert _run(tiny_spec(), runs) == 2
    failed = "SELECT list(run_id ORDER BY run_id) FROM runs WHERE status = 'failed'"
    retried = [f"{NAME}-000005", f"{NAME}-000006"]
    assert _one(connect(runs), failed) == retried
    monkeypatch.setattr(generate_module, "generate", real)
    assert _run(tiny_spec(), runs, "--retry-failed") == 0
    con = connect(runs)
    counts = "SELECT count(*), count(DISTINCT run_id) FROM runs WHERE status = 'ok'"
    assert con.sql(counts).fetchall() == [(10, 10)]
    keys = "SELECT batch_key FROM runs WHERE run_id IN ($1, $2) ORDER BY run_id"
    assert con.execute(keys, retried).fetchall() == [
        ("v1-b000003-2:0",),
        ("v1-b000003-2:1",),
    ]
    # The first attempt's shard still holds both runs, failed; none of it shows.
    first = read_runs_file(runs / NAME / "shards" / "v1-b000003-1.parquet")
    assert [record.status for record in first] == ["failed", "failed"]
    _as_read(con, runs / NAME)


def test_after_a_new_version_only_its_runs_show_kept_ones_included(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    unpushed = [r.run_id for r in read_results(runs / NAME) if not r.schedule]
    weaker = tiny_spec(WEAKER)
    assert _run(weaker, runs, "--overwrite-version", "--limit", "1") == 3
    # Version 2 so far: the 5 unpushed runs it keeps and 1 weaker pulse.
    # Version 1's other 4 pulses are still saved, and do not show.
    con = connect(runs)
    assert _one(con, "SELECT count(*) FROM runs") == 6
    _as_read(con, runs / NAME)
    assert _run(weaker, runs) == 0
    con = connect(runs)
    for view in ("runs", "steps", "conditions"):  # scores and sentences: none
        assert con.sql(f"SELECT DISTINCT version FROM {view}").fetchall() == [(2,)]
    rows = con.sql("SELECT run_id, batch_key FROM runs ORDER BY run_id").fetchall()
    assert len(rows) == 10
    # The unpushed runs are kept, from version 1's shards; the weaker pulses
    # are version 2's own.
    assert [run_id for run_id, key in rows if key.startswith("v1-")] == unpushed
    assert all(key.startswith("v2-") for run_id, key in rows if run_id not in unpushed)
    alphas = "SELECT DISTINCT schedule->>'$[0].alpha' FROM runs WHERE schedule <> '[]'"
    assert con.sql(alphas).fetchall() == [("0.5",)]
    _as_read(con, runs / NAME)


def test_a_shard_no_mark_names_never_shows(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    real = folder_module.append_index

    def killed(path: Path, entry: dict) -> None:
        if entry["batch"] == "b000002":
            raise KeyboardInterrupt  # both files are in place, the mark is not
        real(path, entry)

    monkeypatch.setattr(folder_module, "append_index", killed)
    assert _run(tiny_spec(), runs) == 130
    assert (runs / NAME / "shards" / "v1-b000002-1.parquet").exists()
    con = connect(runs)
    keys = "SELECT list(batch_key ORDER BY batch_key) FROM runs"
    assert _one(con, keys) == ["v1-b000001-1:0", "v1-b000001-1:1"]
    _as_read(con, runs / NAME)
    monkeypatch.setattr(folder_module, "append_index", real)
    assert _run(tiny_spec(), runs) == 0
    con = connect(runs)
    stray = "SELECT count(*) FROM runs WHERE batch_key LIKE 'v1-b000002-1:%'"
    assert _one(con, stray) == 0
    _as_read(con, runs / NAME)


def test_the_views_hold_what_was_saved_when_connect_was_called(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    con = connect(runs)
    assert _run(tiny_spec(), runs) == 0
    assert _one(con, "SELECT count(*) FROM runs") == 4  # a snapshot
    monkeypatch.setenv("PUSHLAB_RUNS_DIR", str(runs))  # the default runs dir
    assert _one(connect(), "SELECT count(*) FROM runs") == 10


def test_files_and_folders_that_are_not_runner_folders_are_skipped(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    assert (runs / "speeds.json").is_file()  # the runner's own file in the runs dir
    write_runs(read_results(runs / NAME)[:2], runs / "s1-steer")  # S1's layout
    (runs / "notes.txt").write_text("not a folder")
    (runs / "empty").mkdir()
    con = connect(runs)
    by_experiment = "SELECT experiment, count(*) FROM runs GROUP BY ALL"
    assert con.sql(by_experiment).fetchall() == [(NAME, 10)]
    _as_read(con, runs / NAME)


def test_a_batch_saved_without_steps_has_its_runs_but_no_steps(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    # As a batch saved before steps were kept.
    (runs / NAME / "steps" / "v1-b000001-1.parquet").unlink()
    con = connect(runs)
    assert _one(con, "SELECT count(*) FROM runs") == 10
    without = "SELECT list(run_id ORDER BY 1) FROM runs ANTI JOIN steps USING (run_id)"
    assert _one(con, without) == [f"{NAME}-000001", f"{NAME}-000002"]


def test_a_folder_without_condition_labels_shows_runs_with_no_label(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    identity = read_identity(runs / NAME)
    del identity["conditions"]  # as a folder saved before labels were kept
    write_identity(runs / NAME, identity)
    con = connect(runs)
    labelled = "SELECT count(*), count(condition) FROM conditions"
    assert con.sql(labelled).fetchall() == [(10, 0)]
    _as_read(con, runs / NAME)


def test_a_path_with_quotes_brackets_or_key_value_folders_reads_as_it_is(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    # DuckDB reads [1] as a pattern and step=3 as a column, unless told not to.
    runs = tmp_path / "step=3" / "it's [1]"
    assert _run(tiny_spec(), runs) == 0
    shutil.copytree(runs, tmp_path / "step=3" / "it's 1")  # what [1] would match
    _as_read(connect(runs), runs / NAME)


def test_a_path_duckdb_cannot_read_is_refused(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    odd = tmp_path / "back\\slash [1]"
    shutil.copytree(runs, odd)
    with pytest.raises(ValueError, match="a backslash"):
        connect(odd)


def test_a_missing_runs_dir_or_a_run_folder_in_its_place_is_refused(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="no folder"):
        connect(tmp_path / "missing")
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    with pytest.raises(ValueError, match="experiment.json"):
        connect(runs / NAME)
    write_runs(read_results(runs / NAME)[:1], tmp_path / "s1-steer")
    with pytest.raises(ValueError, match="runs.parquet"):
        connect(tmp_path / "s1-steer")


def test_a_shard_the_index_names_that_is_missing_is_refused(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    (runs / NAME / "shards" / "v1-b000004-1.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="v1-b000004-1.parquet"):
        connect(runs)
