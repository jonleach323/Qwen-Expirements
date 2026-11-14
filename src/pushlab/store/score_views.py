"""The scores and sentences views: the judges' verdicts on the runs' replies.

pushlab.store.query's connect() makes these two views beside runs, steps
and conditions, over the files that pushlab score writes into each runner
folder's scores/ and sentences/ (pushlab.store.scores):
- scores: one row per verdict on a whole reply, with the scores table's
  columns and the experiment;
- sentences: one row per verdict on a window of a reply's sentences, with
  the sentences table's columns, the window's span among them, and the
  experiment.

A row shows only when the runs view shows its run in the row's version,
with the reply the row scored: its output_sha256 is the SHA-256 of the
run's output_text. So the rows of an older version drop out, those of a run
the new version kept included, and so do those of a reply that a new
version replaced: judge the new version's replies again with pushlab
score, whose cache answers for a text and question judged before.

Of the rows of one key, the newest shows: the latest created_at, then the
last file by name, then the last row in the file. A key is the experiment
and KEYS: the run and version, the scorer and its version, the judge and
the question, and for sentences the window. So judging again replaces a
verdict, and the earlier file stays as it was.

The files are those of the runner folders that the runs view counts, so a
copy of a folder adds none. Hidden files, such as write_whole's temporary
ones, are skipped, and every file is checked before DuckDB reads any: one
whose columns are not its table's is refused, naming it.

Example:
    >>> from pushlab.store.query import connect
    >>> con = connect(Path("/workspace/runs"))  # doctest: +SKIP
    >>> con.sql(
    ...     "SELECT c.condition, s.label, count(*) FROM scores s "
    ...     "JOIN conditions c USING (experiment, version, run_id) "
    ...     "WHERE s.scorer = 'speaks_as' GROUP BY ALL ORDER BY ALL"
    ... ).fetchall()  # doctest: +SKIP
    [('none', 'assistant', 40), ('pulse length=16', 'speaks_as', 23), ...]
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pushlab.runs.folder import SCORES, SENTENCES
from pushlab.store.scores import SCHEMAS, _differences

# What makes rows of a table one verdict, besides the experiment: the key
# of pushlab.store.scores.ScoreRow.
_WHOLE = ("run_id", "version", "scorer", "scorer_version", "judge", "question_sha256")
KEYS = {
    SCORES: _WHOLE,
    SENTENCES: (*_WHOLE, "sentence_idx", "sentence_count", "splitter_version"),
}
# A table's files, as connect registers them: each one's runner folder's
# experiment, and its absolute path, as DuckDB names it in filename.
FILES_SCHEMA = pa.schema(
    [
        pa.field("experiment", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=False),
    ]
)


def mismatch(schema: pa.Schema, table: str) -> str | None:
    """How a file's schema differs from a table's, scores or sentences, or
    None when it is the table's: other columns, or a column of another
    type, or that may be null where the table's may not, or the reverse.

    Example:
        >>> from pushlab.store.scores import SCORES_SCHEMA, SENTENCES_SCHEMA
        >>> mismatch(SCORES_SCHEMA, "scores") is None
        True
        >>> mismatch(SENTENCES_SCHEMA, "scores")  # doctest: +ELLIPSIS
        'its columns are run_id, version, sentence_idx, sentence_count, ...'
        >>> narrow = pa.field("version", pa.int32(), nullable=False)
        >>> mismatch(SCORES_SCHEMA.set(1, narrow), "scores")
        'version is int32, not int64'
    """
    want = SCHEMAS[table]
    if schema.names != want.names:
        return f"its columns are {', '.join(schema.names)}"
    if not schema.equals(want):
        return _differences(schema, want)
    return None


def check_file(path: Path, table: str) -> None:
    """Refuse a file that is not a Parquet file of a table's columns, scores
    or sentences, with a ValueError that names it and how it differs.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = Path(folder) / "odd.parquet"
        ...     _ = path.write_text("not Parquet")
        ...     check_file(path, "scores")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: /.../odd.parquet is not a scores file: Parquet magic bytes ...
    """
    try:
        # Absolute, so pyarrow never reads a folder name such as run:1 as a URL.
        schema = pq.read_schema(Path(path).absolute())
    except pa.ArrowInvalid as error:  # not a Parquet file at all
        raise ValueError(f"{path} is not a {table} file: {error}") from None
    if (difference := mismatch(schema, table)) is not None:
        raise ValueError(f"{path} is not a {table} file: {difference}")


def table_files(
    folders: Sequence[tuple[Path, Mapping[str, Any]]], table: str
) -> pa.Table:
    """The files of a table, scores or sentences, in runner folders given
    with their experiment.json, as FILES_SCHEMA rows: each folder's in name
    order, hidden files skipped, and each one checked (check_file).

    Example:
        >>> folders = [(Path("/no/such/folder"), {"experiment": "x", "version": 1})]
        >>> table_files(folders, "scores").num_rows
        0
    """
    rows = []
    for path, identity in folders:
        for file in sorted((path / table).glob("[!.]*.parquet")):
            if not file.is_file():
                continue  # such as a folder of that name
            filename = file.absolute()
            check_file(filename, table)
            rows.append(
                {"experiment": identity["experiment"], "filename": str(filename)}
            )
    return pa.Table.from_pylist(rows, schema=FILES_SCHEMA)


def view(table: str, source: str, files: str) -> str:
    """The SQL that makes a table's view, as the module's docstring says.

    source is SQL for the rows of the table's files with filename and
    file_row_number (pushlab.store.query._source), and files names the
    files and their experiments, registered as FILES_SCHEMA. The view joins
    the runs view, which must exist.

    Example:
        >>> print(view("scores", "_no_scores", "_scores_files"))  # doctest: +ELLIPSIS
        CREATE VIEW scores AS SELECT s."run_id", s."version", ... = 1
    """
    columns = ", ".join(f's."{name}"' for name in SCHEMAS[table].names)
    key = ", ".join(f's."{name}"' for name in KEYS[table])
    return (
        f"CREATE VIEW {table} AS SELECT {columns}, f.experiment FROM {source} AS s "
        f"JOIN {files} AS f ON s.filename = f.filename "
        # The run as the runs view shows it, with the reply the row scored.
        "JOIN runs AS r ON r.experiment = f.experiment AND r.run_id = s.run_id "
        "AND r.version = s.version AND sha256(r.output_text) = s.output_sha256 "
        # The newest row of each key; a key's files are all in one folder.
        f"QUALIFY row_number() OVER (PARTITION BY f.experiment, {key} "
        "ORDER BY s.created_at DESC, s.filename DESC, s.file_row_number DESC) = 1"
    )
