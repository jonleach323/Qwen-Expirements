"""DuckDB views across every experiment's runs, steps and judged replies.

connect() opens an in-memory DuckDB connection with five views over every
runner folder in a runs dir: a folder with experiment.json, as
pushlab.runs.folder describes, whose name is its experiment's, as the
runner names it. Each view holds a folder's current version only, and each
run as its latest mark in the index says, as pushlab.runs.shards reads them:
- runs: one row per saved run, ok or failed, with the runs table's columns
  (pushlab.store.records) and the version. schedule and decoding are JSON
  text, which DuckDB's JSON functions read, such as decoding->>'$.dtype'.
- steps: one row per step of those runs (pushlab.store.steps), with the
  experiment and the version. A batch saved before steps were kept has none.
- conditions: each saved run's experiment, version, run ID and condition
  label from experiment.json, such as "pulse length=16", to group runs or
  steps by with a join; NULL in a folder saved before labels were kept.
- scores and sentences: the judges' verdicts on those runs' replies, whole
  or by windows of their sentences, with the scores and sentences tables'
  columns (pushlab.store.scores) and the experiment. Each verdict shows
  once, the newest, and only while the runs view shows the reply it
  judged, in its version (pushlab.store.score_views).
So each run of an experiment shows once, and each of its steps once, and
(experiment, version, run_id) names one run.

The marks are read in Python, from each folder's index, and DuckDB reads
only the files they name, matching each row to its mark by file and run ID.
So an older version's runs, a shard no mark names and an earlier attempt at
a run never show; a run a new version kept shows from the shard holding it.
A run written twice into one shard shows once, as the copy whose status is
its mark's, the first such in the file; a step written twice, its first
copy. pushlab verify reports both.

A folder whose name is not the experiment its experiment.json names is a
copy, such as a backup beside the folder or a link to it under another
name: the runner keeps an experiment's runs in <runs dir>/<experiment> only.
Copies are left out, with a warning that names them, so that no run counts
twice; to query one, give connect a runs dir in which it has its
experiment's name. A link under the experiment's own name counts as the
folder, once, as no runs dir holds two entries of one name.

Example:
    >>> con = connect(Path("/workspace/runs"))  # doctest: +SKIP
    >>> con.sql(
    ...     "SELECT c.condition, avg(s.logprob) FROM steps s "
    ...     "JOIN conditions c USING (experiment, version, run_id) GROUP BY ALL"
    ... ).fetchall()  # doctest: +SKIP
    [('none', -1.83...), ('pulse length=16', -2.41...)]
"""

import re
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from pushlab.runs import folder
from pushlab.store import score_views
from pushlab.store.records import RUNS_FILE, RUNS_SCHEMA
from pushlab.store.scores import SCHEMAS
from pushlab.store.steps import STEPS_SCHEMA

# Each saved run of a runner folder's version, and the files that hold it.
MARKS_SCHEMA = pa.schema(
    [
        pa.field("experiment", pa.string(), nullable=False),
        pa.field("version", pa.int64(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("status", pa.string()),  # as its mark says
        pa.field("condition", pa.string()),  # None without a label
        pa.field("shard", pa.string(), nullable=False),  # its runs file's path
        pa.field("steps", pa.string()),  # its steps file's path; None without one
    ]
)
# What DuckDB reads as a pattern in a file's path, such as runs[1].
_PATTERN = re.compile(r"[*?\[]")


def connect(runs_dir: Path | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection with the views runs, steps,
    conditions, scores and sentences over every runner folder in runs_dir,
    default_runs_dir() when None, as the module's docstring says.

    Other folders, such as S1's with one runs.parquet, and files are
    skipped; with no runner folder, or nothing saved, the views have their
    columns and no rows. A copy of a runner folder, whose name is not its
    experiment's, is skipped with a UserWarning. The views are a snapshot
    of what was saved when connect was called: connect again to see later
    batches and verdicts.

    FileNotFoundError says when runs_dir is not a folder or a shard that an
    index names is missing; ValueError when runs_dir is a run folder itself,
    a folder's experiment.json or index is broken, or a file in its scores/
    or sentences/ is not that table's.

    Example:
        >>> con = connect(Path("/workspace/runs"))  # doctest: +SKIP
        >>> con.sql(
        ...     "SELECT experiment, count(*) FROM runs GROUP BY ALL"
        ... ).fetchall()  # doctest: +SKIP
        [('s2-pulse-lengths', 1200)]
        >>> connect(Path("/no/such/folder"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: there is no folder /no/such/folder; ...
    """
    runs_dir = folder.default_runs_dir() if runs_dir is None else Path(runs_dir)
    folders = _runner_folders(runs_dir)
    marks = _marks(folders)
    # The judges' files, each checked before DuckDB reads any.
    judged = {table: score_views.table_files(folders, table) for table in SCHEMAS}
    con = duckdb.connect()
    con.register("_marks", marks)
    runs = _source(con, "_no_runs", RUNS_SCHEMA, marks["shard"].to_pylist())
    steps = _source(con, "_no_steps", STEPS_SCHEMA, marks["steps"].to_pylist())
    run_columns = ", ".join(f'r."{name}"' for name in RUNS_SCHEMA.names)
    step_columns = ", ".join(f's."{name}"' for name in STEPS_SCHEMA.names)
    # Each run once, and each step once, should a file hold one twice.
    con.execute(
        f"CREATE VIEW runs AS SELECT {run_columns}, m.version FROM {runs} AS r "
        "JOIN _marks AS m ON r.filename = m.shard AND r.run_id = m.run_id "
        "QUALIFY row_number() OVER (PARTITION BY m.experiment, r.run_id "
        "ORDER BY r.status = m.status DESC, r.file_row_number) = 1"
    )
    con.execute(
        f"CREATE VIEW steps AS SELECT {step_columns}, m.experiment, m.version "
        f"FROM {steps} AS s "
        "JOIN _marks AS m ON s.filename = m.steps AND s.run_id = m.run_id "
        "QUALIFY row_number() OVER (PARTITION BY m.experiment, s.run_id, s.step "
        "ORDER BY s.file_row_number) = 1"
    )
    con.execute(
        "CREATE VIEW conditions AS "
        "SELECT experiment, version, run_id, condition FROM _marks"
    )
    for table, files in judged.items():
        con.register(f"_{table}_files", files)
        paths = files["filename"].to_pylist()
        source = _source(con, f"_no_{table}", SCHEMAS[table], paths)
        con.execute(score_views.view(table, source, f"_{table}_files"))
    return con


def _runner_folders(runs_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Each runner folder in runs_dir, in name order, with its
    experiment.json; a copy of one is left out, with a warning (see the
    module's docstring)."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(
            f"there is no folder {runs_dir}; the runs dir is the folder that "
            "holds experiments' folders, such as /workspace/runs"
        )
    names = (folder.IDENTITY_FILE, RUNS_FILE)
    if held := [name for name in names if (runs_dir / name).exists()]:
        raise ValueError(
            f"{runs_dir} is a run folder (it holds {', '.join(held)}); give the "
            "folder that holds experiments' folders, such as /workspace/runs"
        )
    found = []
    for path in sorted(runs_dir.iterdir()):
        identity = folder.read_identity(path) if path.is_dir() else None
        if identity is None:
            continue  # not a runner folder
        if path.name != identity["experiment"]:
            warnings.warn(_copy(path, identity["experiment"]), stacklevel=3)
            continue
        found.append((path, identity))
    return found


def _marks(folders: list[tuple[Path, dict[str, Any]]]) -> pa.Table:
    """Each saved run of the runner folders, as MARKS_SCHEMA."""
    rows = []
    for path, identity in folders:
        # The identity was read first: a new version's marks are all in the
        # index before its experiment.json is.
        experiment, version = identity["experiment"], identity["version"]
        labels = identity.get("conditions") or {}
        states = folder.latest(folder.read_index(path), version)
        files: dict[str, tuple[str, str | None]] = {}
        for run_id, state in sorted(states.items()):
            name = state["shard"]
            if name not in files:
                files[name] = _files(path, name)
            mark = (experiment, version, run_id, state["status"], labels.get(run_id))
            rows.append((*mark, *files[name]))
    named = [dict(zip(MARKS_SCHEMA.names, row, strict=True)) for row in rows]
    return pa.Table.from_pylist(named, schema=MARKS_SCHEMA)


def _copy(path: Path, experiment: str) -> str:
    """Why the views leave out path, a copy of experiment's runner folder.

    Example:
        >>> _copy(Path("/workspace/runs/x-old"), "x")  # doctest: +NORMALIZE_WHITESPACE
        "/workspace/runs/x-old is left out of the views: its experiment.json is
        x's, whose runner folder is /workspace/runs/x, so it is a copy"
    """
    return (
        f"{path} is left out of the views: its experiment.json is {experiment}'s, "
        f"whose runner folder is {path.with_name(experiment)}, so it is a copy"
    )


def _files(path: Path, name: str) -> tuple[str, str | None]:
    """The runs and steps files of a runner folder's saved batch, as absolute
    paths; the steps file is None when the batch has none."""
    shard = (path / folder.SHARDS / name).absolute()
    if not shard.is_file():
        raise FileNotFoundError(
            f"{shard} is missing, though {path / folder.INDEX_FILE} names it"
        )
    steps = (path / folder.STEPS / name).absolute()
    return str(shard), (str(steps) if steps.is_file() else None)


def _source(
    con: duckdb.DuckDBPyConnection,
    name: str,
    schema: pa.Schema,
    paths: list[str | None],
) -> str:
    """SQL for a table of schema's columns, filename, the file each row is
    from, and file_row_number, its place there: read_parquet over the paths,
    each once, or with no paths an empty table registered as name, typed as
    read_parquet would type it."""
    files = sorted({path for path in paths if path is not None})
    if files:
        listed = ", ".join(_literal(_pattern(file)) for file in files)
        # Without hive_partitioning off, a folder such as step=3 is a column.
        return (
            f"read_parquet([{listed}], filename = true, file_row_number = true, "
            "hive_partitioning = false)"
        )
    fields = [field.with_type(_readable(field.type)) for field in schema]
    fields += [
        pa.field("filename", pa.string()),
        pa.field("file_row_number", pa.int64()),
    ]
    empty = pa.schema(fields).empty_table()
    con.register(name, empty)
    return name


def _readable(type_: pa.DataType) -> pa.DataType:
    """type_ with 16-bit floats as 32-bit ones: DuckDB reads both as FLOAT
    from Parquet, but only 32-bit ones from Arrow."""
    if pa.types.is_float16(type_):
        return pa.float32()
    if pa.types.is_list(type_):
        return pa.list_(_readable(type_.value_type))
    return type_


def _pattern(path: str) -> str:
    """path as a read_parquet pattern that matches that file alone.

    DuckDB reads *, ? and [ in a path as a pattern, so each one goes in
    brackets, which match it alone. DuckDB cannot match a backslash in such
    a pattern, so a path with both is refused.
    """
    if not _PATTERN.search(path):
        return path
    if "\\" in path:
        raise ValueError(
            f"DuckDB cannot read {path}: its path has a backslash and one of "
            "*, ? and [; move the runs dir to a path without them"
        )
    return _PATTERN.sub(lambda match: f"[{match[0]}]", path)


def _literal(text: str) -> str:
    """text as an SQL string: in single quotes, with each one inside doubled.

    Example:
        >>> _literal("it's")
        "'it''s'"
    """
    return "'" + text.replace("'", "''") + "'"
