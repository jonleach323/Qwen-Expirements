"""The scores and sentences tables: judges' verdicts, kept in runner folders.

pushlab score (decision 0021) writes a row for each verdict:
- scores: one row per run, scorer, scorer version, judge and question, on
  the whole reply;
- sentences: one row per window of a reply's sentences besides, with its
  span: its first sentence (sentence_idx) and how many it holds, the reply
  tokens start_step to end_step and the characters start_char to end_char,
  ends not included, and the version of the sentence splitter.
A row holds a label, or a value for a score such as coherence, when its
status is "ok", and neither, with the error, when the judge failed: a
label is never guessed. question_sha256 names everything the judge was
asked but the text (pushlab.scoring.rubrics.Request), so verdicts on one
persona and on another never mix; details is a JSON object of the rest,
such as the judge's reason, which may quote the reply.

The rows of a runner folder go in its scores/ and sentences/ folders, each
write a new Parquet file written whole (folder.write_whole) and never
replaced, by a process holding the folder's lock. A row records the
folder's version and the SHA-256 of the run's output_text when it was
scored, so a reply that a new version replaced loses its rows; of rows of
one key, the newest counts (pushlab.store.score_views).
"""

import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pushlab.runs.folder import SCORES, SENTENCES, write_whole
from pushlab.store.records import RunRecord
from pushlab.store.steps import _kind

STATUSES = ("ok", "failed")
SCORER = re.compile(r"[a-z][a-z0-9_]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
MAX_INT32 = 2**31 - 1

_COMMON = [
    pa.field("scorer", pa.string(), nullable=False),
    pa.field("scorer_version", pa.int64(), nullable=False),
    pa.field("judge", pa.string(), nullable=False),
    pa.field("question_sha256", pa.string(), nullable=False),
    pa.field("persona", pa.string(), nullable=False),  # speaks_as's; else ""
    pa.field("label", pa.string()),
    pa.field("value", pa.float64()),
    pa.field("status", pa.string(), nullable=False),
    pa.field("error", pa.string(), nullable=False),
    pa.field("attempts", pa.int64(), nullable=False),
    pa.field("details", pa.string(), nullable=False),  # JSON object
    pa.field("text_sha256", pa.string(), nullable=False),
    pa.field("output_sha256", pa.string(), nullable=False),
    pa.field("created_at", pa.string(), nullable=False),  # UTC
]
_RUN = [
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("version", pa.int64(), nullable=False),
]
SPAN_COLUMNS = (
    "sentence_idx",
    "sentence_count",
    "start_step",
    "end_step",
    "start_char",
    "end_char",
    "splitter_version",
)
SCORES_SCHEMA = pa.schema(_RUN + _COMMON)
SENTENCES_SCHEMA = pa.schema(
    _RUN
    + [pa.field(name, pa.int32(), nullable=False) for name in SPAN_COLUMNS]
    + _COMMON
)
SCHEMAS = {SCORES: SCORES_SCHEMA, SENTENCES: SENTENCES_SCHEMA}


@dataclass(frozen=True)
class Span:
    """A window of a reply: sentences sentence_idx to sentence_idx +
    sentence_count - 1, reply tokens start_step to end_step and characters
    start_char to end_char, ends not included, by splitter_version."""

    sentence_idx: int
    sentence_count: int
    start_step: int
    end_step: int
    start_char: int
    end_char: int
    splitter_version: int


@dataclass(frozen=True)
class ScoreRow:
    """One verdict, as the scores table holds it, or with a span, the
    sentences table.

    Example:
        >>> row = ScoreRow(
        ...     run_id="x-000001",
        ...     version=1,
        ...     scorer="coherence",
        ...     scorer_version=1,
        ...     judge="api:claude-haiku-5-5",
        ...     question_sha256="a" * 64,
        ...     persona="",
        ...     label=None,
        ...     value=8.0,
        ...     status="ok",
        ...     error="",
        ...     attempts=1,
        ...     details={"reason": "Clear."},
        ...     text_sha256="b" * 64,
        ...     output_sha256="b" * 64,
        ...     created_at="2026-11-13T09:00:00Z",
        ... )
        >>> row.table
        'scores'
    """

    run_id: str
    version: int
    scorer: str
    scorer_version: int
    judge: str
    question_sha256: str
    persona: str
    label: str | None
    value: float | None
    status: str
    error: str
    attempts: int
    details: dict[str, Any] = field(compare=True)
    text_sha256: str = ""
    output_sha256: str = ""
    created_at: str = ""
    span: Span | None = None

    @property
    def table(self) -> str:
        """The table it belongs to: scores for a whole reply, else sentences."""
        return SCORES if self.span is None else SENTENCES

    @property
    def key(self) -> tuple[Any, ...]:
        """What makes two rows the same verdict, the later replacing the
        earlier: the run and version, the scorer and its version, the judge,
        the question, and the window, if any."""
        window = None
        if self.span is not None:
            span = self.span
            window = (span.sentence_idx, span.sentence_count, span.splitter_version)
        return (
            self.run_id,
            self.version,
            self.scorer,
            self.scorer_version,
            self.judge,
            self.question_sha256,
            window,
        )


def now() -> str:
    """The time now, in UTC, as rows record it.

    Example:
        >>> len(now())
        20
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def output_sha256(run: RunRecord) -> str:
    """The SHA-256 of a run's output_text's UTF-8 bytes, in hex, as DuckDB's
    sha256 gives it."""
    return hashlib.sha256(run.output_text.encode("utf-8")).hexdigest()


def _whole(value: object, low: int, name: str) -> None:
    if type(value) is not int or not low <= value <= MAX_INT32:
        raise ValueError(f"{name} must be a whole number from {low}, not {value!r}")


def check_row(row: ScoreRow) -> None:
    """Refuse a row whose fields are not what ScoreRow's docstring and the
    module's say, naming the field and the run."""
    where = f"the row of {row.run_id!r}"
    if not isinstance(row.run_id, str) or not row.run_id:
        raise ValueError(f"{where}: run_id must be a run's ID")
    _whole(row.version, 1, f"{where}: version")
    if not isinstance(row.scorer, str) or not SCORER.fullmatch(row.scorer):
        raise ValueError(f"{where}: scorer must be lowercase letters, digits and _")
    _whole(row.scorer_version, 1, f"{where}: scorer_version")
    if not isinstance(row.judge, str) or ":" not in row.judge.strip(":"):
        raise ValueError(f"{where}: judge must be <backend>:<model>, not {row.judge!r}")
    for name in ("question_sha256", "text_sha256", "output_sha256"):
        if not isinstance(getattr(row, name), str) or not SHA256.fullmatch(
            getattr(row, name)
        ):
            raise ValueError(f"{where}: {name} must be a SHA-256 in hex")
    if not isinstance(row.persona, str) or not isinstance(row.error, str):
        raise ValueError(f"{where}: persona and error must be text")
    if row.status not in STATUSES:
        raise ValueError(f"{where}: status must be ok or failed, not {row.status!r}")
    if row.label is not None and not isinstance(row.label, str):
        raise ValueError(f"{where}: label must be text or None")
    if row.value is not None and (
        type(row.value) is not float or not math.isfinite(row.value)
    ):
        raise ValueError(f"{where}: value must be a finite float or None")
    if row.status == "ok" and ((row.label is None and row.value is None) or row.error):
        raise ValueError(f"{where}: an ok row has a label or value and no error")
    if row.status == "failed" and (
        row.label is not None or row.value is not None or not row.error
    ):
        raise ValueError(f"{where}: a failed row has an error and no label or value")
    _whole(row.attempts, 0, f"{where}: attempts")
    if not isinstance(row.details, dict):
        raise ValueError(f"{where}: details must be a mapping")
    try:  # as rows_table saves them
        saved = json.dumps(row.details, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: details must be JSON: {error}") from None
    if json.loads(saved) != row.details:  # such as a tuple, or a number as a key
        raise ValueError(f"{where}: details change when saved as JSON: {saved}")
    if not isinstance(row.created_at, str) or not TIME.fullmatch(row.created_at):
        raise ValueError(f"{where}: created_at must be a UTC time such as {now()}")
    if row.span is not None:
        span = row.span
        for name in SPAN_COLUMNS:
            low = 1 if name in ("sentence_count", "splitter_version") else 0
            _whole(getattr(span, name), low, f"{where}: {name}")
        if span.start_step > span.end_step or span.start_char > span.end_char:
            raise ValueError(f"{where}: a span's start must not pass its end")


def _check_against(row: ScoreRow, runs: Mapping[str, RunRecord]) -> None:
    """Refuse a row of a run that runs lacks, of another reply, or whose
    span passes the reply's tokens or text."""
    run = runs.get(row.run_id)
    if run is None:
        raise ValueError(f"{row.run_id} is not a saved run of the folder's version")
    if run.status == "failed":
        raise ValueError(f"{row.run_id} failed, so it has no reply to score")
    if row.output_sha256 != output_sha256(run):
        raise ValueError(f"{row.run_id}'s reply is not the one its row scored")
    span = row.span
    if span is not None and (
        span.end_step > len(run.output_tokens) or span.end_char > len(run.output_text)
    ):
        raise ValueError(f"{row.run_id}: a span passes the end of its reply")


def rows_table(rows: Sequence[ScoreRow], table: str) -> pa.Table:
    """Rows of one table as an Arrow table of its schema."""
    columns: dict[str, list[Any]] = {name: [] for name in SCHEMAS[table].names}
    for row in rows:
        values = {
            name: getattr(row, name)
            for name in columns
            if name not in SPAN_COLUMNS and name != "details"
        }
        values["details"] = json.dumps(row.details, sort_keys=True, ensure_ascii=False)
        if row.span is not None:
            values |= {name: getattr(row.span, name) for name in SPAN_COLUMNS}
        for name in columns:
            columns[name].append(values[name])
    return pa.table(columns, schema=SCHEMAS[table])


def write_rows(
    folder: Path, rows: Sequence[ScoreRow], runs: Mapping[str, RunRecord]
) -> Path:
    """Write rows into a new file of their table in the runner folder; its path.

    The rows are of one table and one scorer version, of runs saved in the
    folder's version (runs, by run ID), each key once. ValueError refuses
    anything else, and a malformed row, before anything is written. Call
    it holding the folder's lock (folder.locked), as pushlab score does.
    """
    if not rows:
        raise ValueError("there are no rows to write")
    tables = {row.table for row in rows}
    scorers = {(row.scorer, row.scorer_version) for row in rows}
    if len(tables) > 1 or len(scorers) > 1:
        raise ValueError("write one table's rows, of one scorer version, at a time")
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        check_row(row)
        _check_against(row, runs)
        if row.key in seen:
            raise ValueError(f"two rows of {row.run_id} have one key: {row.key}")
        seen.add(row.key)
    (table,), ((scorer, version),) = tables, scorers
    place = Path(folder) / table
    place.mkdir(parents=True, exist_ok=True)
    # To the microsecond, so that a scorer version's files sort in the order
    # they were written, as the views take them: the later file wins a tie.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = place / f"{scorer}-v{version}-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
    if path.exists():  # a name is never used twice
        raise FileExistsError(f"{path} already exists")
    data = rows_table(rows, table)
    write_whole(path, lambda file: pq.write_table(data, file))
    return path


def _row(values: dict[str, Any], table: str) -> ScoreRow:
    span = None
    if table == SENTENCES:
        span = Span(*(values.pop(name) for name in SPAN_COLUMNS))
    try:
        details = json.loads(values.pop("details"))
    except ValueError as error:
        raise ValueError(f"a row's details are not JSON: {error}") from None
    return ScoreRow(**values, details=details, span=span)


def read_rows(folder: Path, table: str) -> list[ScoreRow]:
    """Every row of a table, scores or sentences, in a runner folder, file by
    file in name order; none when the folder has no such table yet.

    ValueError refuses a file whose columns are not the table's, naming the
    differences, and a row whose details are not JSON.
    """
    schema = SCHEMAS[table]
    rows = []
    for path in sorted((Path(folder) / table).glob("[!.]*.parquet")):
        # Absolute, so pyarrow never reads a folder name such as run:1 as a URL.
        data = pq.read_table(path.absolute())
        if data.schema.names != schema.names:
            raise ValueError(
                f"{path} is not a {table} file: its columns are "
                f"{', '.join(data.schema.names)}"
            )
        if not data.schema.equals(schema):
            raise ValueError(
                f"{path} is not a {table} file: {_differences(data.schema, schema)}"
            )
        rows += [_row(values, table) for values in data.to_pylist()]
    return rows


def _differences(found: pa.Schema, schema: pa.Schema) -> str:
    """How the columns of found, which has schema's names, differ from
    schema's: in type, or in whether they may be null.

    Example:
        >>> wide = SCORES_SCHEMA.set(1, pa.field("version", pa.int32(), False))
        >>> _differences(wide, SCORES_SCHEMA)
        'version is int32, not int64'
    """
    differences = []
    for column, want in zip(found, schema, strict=True):
        if column.equals(want):
            continue
        if _kind(column.type) != _kind(want.type):
            differences.append(
                f"{want.name} is {_kind(column.type)}, not {_kind(want.type)}"
            )
        else:
            null = "may be null" if column.nullable else "may not be null"
            differences.append(f"{want.name} {null}, unlike the table's")
    return "; ".join(differences)
