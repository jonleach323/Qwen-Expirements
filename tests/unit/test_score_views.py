"""pushlab.store.score_views: the scores and sentences views over the judges'
verdicts in every runner folder.

connect() (pushlab.store.query) makes them beside the runs, steps and
conditions views. A row shows when the runs view shows its run, in the
row's version, with the reply the row scored; of the rows of one key, the
newest. tiny_spec's experiment has 10 runs, an unpushed run and a pulse for
each of the five smoke prompts, in 5 batches of 2. The verdicts are made
up, and written as pushlab score writes them.
"""

import doctest
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.runs.folder import SCORES, SENTENCES, locked, read_identity
from pushlab.runs.shards import read_results
from pushlab.scoring.judge import text_sha256
from pushlab.store import score_views
from pushlab.store.query import connect
from pushlab.store.records import RunRecord
from pushlab.store.scores import (
    SCHEMAS,
    SCORES_SCHEMA,
    SENTENCES_SCHEMA,
    SPAN_COLUMNS,
    ScoreRow,
    Span,
    output_sha256,
    rows_table,
    write_rows,
)

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
# Each view's columns, and the type DuckDB gives each one that is not text.
COLUMNS = {
    SCORES: [*SCORES_SCHEMA.names, "experiment"],
    SENTENCES: [*SENTENCES_SCHEMA.names, "experiment"],
}
TYPES = {
    "version": "BIGINT",
    "scorer_version": "BIGINT",
    "value": "DOUBLE",
    "attempts": "BIGINT",
    **dict.fromkeys(SPAN_COLUMNS, "INTEGER"),
}
EARLY, LATE = "2026-11-13T09:00:00Z", "2026-11-13T10:00:00Z"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


@pytest.fixture
def runs(tiny_spec: Callable[..., Path], tmp_path: Path) -> Path:
    """A runs dir in which tiny_spec's experiment is saved, all 10 runs."""
    path = tmp_path / "runs"
    assert _run(tiny_spec(), path) == 0
    return path


def _replies(folder: Path) -> dict[str, RunRecord]:
    """The saved runs of the folder's version that have a reply, by run ID."""
    return {run.run_id: run for run in read_results(folder) if run.status != "failed"}


def _verdicts(folder: Path, table: str = SCORES, **changes: object) -> list[ScoreRow]:
    """A made-up coherence verdict on each reply of the folder's version, the
    nth valued n: on the whole reply, or for sentences on each half of it,
    as a window of one sentence; changed as given."""
    version = read_identity(folder)["version"]
    rows = []
    for number, run in enumerate(_replies(folder).values()):
        reply = output_sha256(run)
        row = ScoreRow(
            run_id=run.run_id,
            version=version,
            scorer="coherence",
            scorer_version=1,
            judge="fake:fake-judge",
            question_sha256="c" * 64,
            persona="",
            label=None,
            value=float(number),
            status="ok",
            error="",
            attempts=1,
            details={"reason": f"Verdict {number}."},
            text_sha256=reply,
            output_sha256=reply,
            created_at=EARLY,
        )
        if table == SCORES:
            rows.append(replace(row, **changes))
            continue
        tokens, chars = len(run.output_tokens), len(run.output_text)
        halves = [
            Span(0, 1, 0, tokens // 2, 0, chars // 2, 1),
            Span(1, 1, tokens // 2, tokens, chars // 2, chars, 1),
        ]
        for span in halves:
            text = run.output_text[span.start_char : span.end_char]
            window = replace(row, span=span, text_sha256=text_sha256(text))
            rows.append(replace(window, **changes))
    return rows


def _write(
    folder: Path, rows: list[ScoreRow], replies: dict[str, RunRecord] | None = None
) -> Path:
    """Write rows into the folder holding its lock, as pushlab score does,
    checked against the folder's replies or those given."""
    with locked(folder):
        found = _replies(folder) if replies is None else replies
        return write_rows(folder, rows, found)


def _sorted(rows: list[ScoreRow]) -> list[ScoreRow]:
    return sorted(rows, key=lambda row: row.key)


def _shown(
    con: duckdb.DuckDBPyConnection, table: str, experiment: str = NAME
) -> list[ScoreRow]:
    """A view's rows of one experiment, as score rows in key order."""
    names = SCHEMAS[table].names
    listed = ", ".join(f'"{name}"' for name in names)
    query = f"SELECT {listed} FROM {table} WHERE experiment = ?"
    rows = []
    for values in con.execute(query, [experiment]).fetchall():
        row = dict(zip(names, values, strict=True))
        span = None
        if table == SENTENCES:
            span = Span(*(row.pop(name) for name in SPAN_COLUMNS))
        details = json.loads(row.pop("details"))
        rows.append(ScoreRow(**row, details=details, span=span))
    return _sorted(rows)


def _values(con: duckdb.DuckDBPyConnection) -> list[float]:
    """The scores view's values, in run order."""
    found = con.sql("SELECT value FROM scores ORDER BY run_id").fetchall()
    return [value for (value,) in found]


def _counts(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """The rows of the scores and sentences views."""
    (counts,) = con.sql(
        "SELECT (SELECT count(*) FROM scores), (SELECT count(*) FROM sentences)"
    ).fetchall()
    return counts


def _check_columns(con: duckdb.DuckDBPyConnection) -> None:
    for view, names in COLUMNS.items():
        described = [(row[0], row[1]) for row in con.sql(f"DESCRIBE {view}").fetchall()]
        assert described == [(name, TYPES.get(name, "VARCHAR")) for name in names]


def test_rows_of_both_tables_show_with_every_column_and_their_values(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    # A path DuckDB would read as a pattern and a column, unless told not to.
    runs = tmp_path / "step=3" / "it's [1]"
    assert _run(tiny_spec(), runs) == 0
    assert _run(tiny_spec(NO_PUSH, file="b.yaml", name="tiny-b"), runs) == 0
    folder, other = runs / NAME, runs / "tiny-b"
    whole = _verdicts(folder)
    windows = _verdicts(
        folder,
        SENTENCES,
        scorer="speaks_as",
        question_sha256="d" * 64,
        persona="Owl",
        label="speaks_as",
        value=None,
    )
    windows[3] = replace(windows[3], status="failed", label=None, error="busy")
    theirs = _verdicts(other, details={"reason": "Fine — «hoot».", "cached": True})
    _write(folder, whole)
    _write(folder, windows)
    copied = _write(other, theirs)
    con = connect(runs)
    _check_columns(con)
    assert _shown(con, SCORES) == _sorted(whole)
    assert _shown(con, SENTENCES) == _sorted(windows)
    assert _shown(con, SCORES, "tiny-b") == _sorted(theirs)
    assert _shown(con, SENTENCES, "tiny-b") == []
    by_label = (
        "SELECT c.condition, count(*) FROM sentences s "
        "JOIN conditions c USING (experiment, version, run_id) GROUP BY ALL ORDER BY 1"
    )
    assert con.sql(by_label).fetchall() == [("none", 10), ("pulse", 10)]
    # tiny-b's file put in this folder: it names runs this experiment lacks.
    shutil.copy(copied, folder / SCORES / copied.name)
    con = connect(runs)
    assert _shown(con, SCORES) == _sorted(whole)
    assert _shown(con, SCORES, "tiny-b") == _sorted(theirs)


def test_of_the_rows_of_one_key_the_newest_shows(runs: Path) -> None:
    folder = runs / NAME
    first = _verdicts(folder)
    _write(folder, first)
    # Four replies judged again an hour later: their new verdicts show.
    again = _verdicts(folder, value=7.0, created_at=LATE)[:4]
    _write(folder, again)
    assert _shown(connect(runs), SCORES) == _sorted(again + first[4:])
    # Verdicts made earlier, though written later, do not: created_at first.
    _write(folder, _verdicts(folder, value=3.0, created_at="2026-11-13T08:00:00Z"))
    assert _shown(connect(runs), SCORES) == _sorted(again + first[4:])
    # Of one created_at, the later file by name.
    tie = _write(folder, _verdicts(folder, value=5.0, created_at=LATE)[:2])
    assert _values(connect(runs))[:4] == [5.0, 5.0, 7.0, 7.0]
    tie.rename(tie.with_name(f"a-{tie.name}"))  # now the first file by name
    assert _values(connect(runs))[:4] == [7.0, 7.0, 7.0, 7.0]
    # Of one file, the later row: a file holding a key twice, as by hand.
    twice = [replace(again[0], value=1.5), replace(again[0], value=2.5)]
    pq.write_table(rows_table(twice, SCORES), folder / SCORES / "zz-by-hand.parquet")
    con = connect(runs)
    assert _values(con) == [2.5, 7.0, 7.0, 7.0, *(row.value for row in first[4:])]
    once = "SELECT count(*), count(DISTINCT run_id) FROM scores"
    assert con.sql(once).fetchall() == [(10, 10)]


def test_after_a_new_version_only_rows_of_its_own_replies_show(
    tiny_spec: Callable[..., Path], runs: Path
) -> None:
    folder = runs / NAME
    old = _replies(folder)
    _write(folder, _verdicts(folder))
    _write(folder, _verdicts(folder, SENTENCES))
    assert _counts(connect(runs)) == (10, 20)
    weaker = tiny_spec(WEAKER)
    assert _run(weaker, runs, "--overwrite-version", "--limit", "1") == 3
    # Version 2 so far: the 5 unpushed runs it keeps and 1 weaker pulse.
    # Version 1's rows all drop out, the kept runs' too, though their
    # replies are the same: a row is of the version it was judged in.
    kept = [run for run in _replies(folder).values() if not run.schedule]
    assert len(kept) == 5 and all(old[run.run_id] == run for run in kept)
    assert _counts(connect(runs)) == (0, 0)
    judged = _verdicts(folder)  # version 2's 6 replies, judged again
    _write(folder, judged)
    by_version = "SELECT version, count(*) FROM scores GROUP BY ALL"
    assert connect(runs).sql(by_version).fetchall() == [(2, 6)]
    # The rest of version 2: its verdicts stay, and its 4 new pulses have none.
    assert _run(weaker, runs) == 0
    con = connect(runs)
    assert _shown(con, SCORES) == _sorted(judged) and _counts(con) == (6, 0)
    assert con.sql(by_version).fetchall() == [(2, 6)]


def test_a_row_of_another_reply_than_its_runs_is_left_out(runs: Path) -> None:
    folder = runs / NAME
    rows = _verdicts(folder)
    _write(folder, rows)
    # A newer verdict on run 1, as if its reply had been another text.
    run = _replies(folder)[rows[0].run_id]
    other = replace(run, output_text=f"{run.output_text} Hoot.")
    sha = output_sha256(other)
    stale = replace(rows[0], text_sha256=sha, output_sha256=sha, created_at=LATE)
    _write(folder, [stale], {run.run_id: other})
    assert _shown(connect(runs), SCORES) == _sorted(rows)


def test_with_no_verdicts_the_views_have_their_columns_and_no_rows(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    empty = connect(runs)  # no runner folder yet
    assert _run(tiny_spec(), runs) == 0
    unjudged = connect(runs)
    for table in (SCORES, SENTENCES):
        (runs / NAME / table).mkdir()  # made, with nothing written yet
    for con in (empty, unjudged, connect(runs)):
        _check_columns(con)
        assert _counts(con) == (0, 0)


def test_a_copy_of_a_runner_folder_adds_no_rows(runs: Path) -> None:
    folder = runs / NAME
    rows = _verdicts(folder)
    _write(folder, rows)
    backup = runs / f"{NAME}-backup"
    shutil.copytree(folder, backup)
    # The copy judged again on its own: newer verdicts, and windows.
    _write(backup, _verdicts(backup, value=9.0, created_at=LATE))
    _write(backup, _verdicts(backup, SENTENCES))
    skipped = f"{NAME}-backup is left out of the views"
    with pytest.warns(UserWarning, match=re.escape(skipped)) as warned:
        con = connect(runs)
    assert len(warned) == 1  # named once, for all five views
    assert _shown(con, SCORES) == _sorted(rows) and _counts(con) == (10, 0)


def test_hidden_files_are_skipped(runs: Path) -> None:
    folder = runs / NAME
    rows = _verdicts(folder)
    _write(folder, rows)
    # Newer verdicts in a hidden file, which no view reads; nor a hidden
    # file that is not Parquet, such as macOS leaves beside a copied one;
    # nor folder.write_whole's temporary file of a write a crash stopped.
    newer = _write(folder, _verdicts(folder, value=9.0, created_at=LATE))
    newer.rename(newer.with_name(f".{newer.name}"))
    (folder / SCORES / f"._{newer.name}").write_bytes(b"\x00\x05\x16\x07")
    (folder / SENTENCES).mkdir()
    (folder / SENTENCES / f".{newer.name}.0123abcd.tmp").write_bytes(b"PAR1")
    con = connect(runs)
    assert _shown(con, SCORES) == _sorted(rows) and _counts(con) == (10, 0)


NARROW = SCORES_SCHEMA.set(1, pa.field("version", pa.int32(), nullable=False))


@pytest.mark.parametrize(
    ("table", "schema", "message"),
    [
        (SCORES, SENTENCES_SCHEMA, "its columns are run_id, version, sentence_idx, "),
        (SENTENCES, SCORES_SCHEMA, "its columns are run_id, version, scorer, "),
        (SCORES, NARROW, "version is int32, not int64"),
        (SCORES, None, "Parquet magic bytes not found"),
    ],
)
def test_a_file_of_another_schema_is_refused_naming_it(
    runs: Path, table: str, schema: pa.Schema | None, message: str
) -> None:
    folder = runs / NAME
    _write(folder, _verdicts(folder, table))  # a good file beside it
    odd = folder / table / "odd.parquet"
    if schema is None:
        odd.write_text("not a Parquet file", encoding="utf-8")
    else:
        pq.write_table(schema.empty_table(), odd)
    with pytest.raises(ValueError, match=re.escape(message)) as refused:
        connect(runs)
    assert str(refused.value).startswith(f"{odd} is not a {table} file: ")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(score_views)
    assert result.attempted > 0
    assert result.failed == 0
