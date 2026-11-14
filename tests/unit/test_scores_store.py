"""pushlab.store.scores: the scores and sentences tables in a runner folder.

write_rows checks every row, and each row against the folder's runs, then
writes them into a new file of their table, whole or not at all; read_rows
gives them back. The runs are made by hand, with invented replies (decision
0020): x-000001 and x-000002 have one, and x-000003 failed.
"""

import doctest
import math
import re
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pushlab.runs import shards
from pushlab.runs.folder import SCORES, SENTENCES, locked
from pushlab.scoring.judge import text_sha256
from pushlab.store import scores
from pushlab.store.records import RunRecord, make_run_id
from pushlab.store.scores import (
    SCORES_SCHEMA,
    SENTENCES_SCHEMA,
    ScoreRow,
    Span,
    check_row,
    output_sha256,
    read_rows,
    write_rows,
)

REPLY = "An owl hoots twice. Then it sleeps."  # 7 words, 35 characters
FIRST = "An owl hoots twice."  # its first sentence: words 0 to 3
DECODING = {"strategy": "greedy", "max_new_tokens": 8, "thinking": False}


def _record(counter: int, text: str = REPLY, status: str = "ok") -> RunRecord:
    """Run counter of experiment x, a reply of text with one token per word;
    a failed run has none."""
    if status == "failed":
        text = ""
    return RunRecord(
        run_id=make_run_id("x", counter),
        experiment="x",
        model="tiny-vl@0123456789abcdef0123456789abcdef01234567",
        prompt_id="smoke-01",
        schedule=[],
        decoding=DECODING | {"dtype": "float32"},
        seed=0,
        batch_key=f"v1-b000001-1:{counter - 1}",
        control_of="",
        output_text=text,
        output_tokens=list(range(len(text.split()))),
        status=status,
    )


RUNS = {
    run.run_id: run
    for run in (_record(1), _record(2, "Hoot."), _record(3, status="failed"))
}
# The first sentence of x-000001's reply: words 0 to 3, characters 0 to 19.
SPAN = Span(0, 1, 0, 4, 0, 19, 1)


def _row(run_id: str = "x-000001", **changes: object) -> ScoreRow:
    """An ok coherence verdict on a run's whole reply, changed as given."""
    reply = output_sha256(RUNS[run_id])
    row = ScoreRow(
        run_id=run_id,
        version=1,
        scorer="coherence",
        scorer_version=1,
        judge="fake:fake-judge",
        question_sha256="a" * 64,
        persona="",
        label=None,
        value=8.0,
        status="ok",
        error="",
        attempts=1,
        details={"reason": "Clear."},
        text_sha256=reply,
        output_sha256=reply,
        created_at="2026-11-13T09:00:00Z",
    )
    return replace(row, **changes)


def _write(folder: Path, rows: list[ScoreRow]) -> Path:
    """Write rows holding the folder's lock, as pushlab score does."""
    with locked(folder):
        return write_rows(folder, rows, RUNS)


def _files(folder: Path) -> list[str]:
    """Every file in the folder's scores/ and sentences/, hidden ones too."""
    places = [folder / SCORES, folder / SENTENCES]
    return sorted(p.name for place in places if place.is_dir() for p in place.iterdir())


def test_rows_written_are_read_back_unchanged_in_both_tables(tmp_path: Path) -> None:
    folder = tmp_path / "x"
    assert read_rows(folder, SCORES) == [] == read_rows(folder, SENTENCES)
    coherence = [
        _row(),
        _row("x-000002", status="failed", value=None, error="busy", attempts=3),
    ]
    speaks_as = [
        _row(scorer="speaks_as", persona="Owl", label="speaks_as", value=None),
        _row(
            "x-000002", scorer="speaks_as", persona="Owl", label="assistant", value=None
        ),
    ]
    windows = [
        _row(span=SPAN, text_sha256=text_sha256(FIRST), value=9.0),
        _row(
            span=Span(1, 1, 4, 7, 19, 35, 1),
            text_sha256=text_sha256(" Then it sleeps."),
            details={"reason": "Calm — «hoot».", "cached": True, "tokens": [3, 1]},
        ),
    ]
    paths = [_write(folder, rows) for rows in (coherence, speaks_as, windows)]
    assert [path.parent.name for path in paths] == [SCORES, SCORES, SENTENCES]
    assert re.fullmatch(r"coherence-v1-.+\.parquet", paths[0].name)
    assert pq.read_schema(paths[0]).equals(SCORES_SCHEMA)
    assert pq.read_schema(paths[2]).equals(SENTENCES_SCHEMA)
    # Each table file by file, in name order: coherence before speaks_as.
    assert read_rows(folder, SCORES) == coherence + speaks_as
    assert read_rows(folder, SENTENCES) == windows


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_id": ""}, "run_id must be a run's ID"),
        ({"version": 0}, "version must be a whole number from 1, not 0"),
        ({"version": True}, "version must be a whole number from 1, not True"),
        ({"version": 2**31}, "version must be a whole number from 1"),
        ({"scorer": "Coherence"}, "scorer must be lowercase letters, digits and _"),
        ({"scorer_version": 1.0}, "scorer_version must be a whole number from 1"),
        ({"judge": "fake"}, "judge must be <backend>:<model>, not 'fake'"),
        ({"judge": "fake:"}, "judge must be <backend>:<model>, not 'fake:'"),
        ({"question_sha256": "A" * 64}, "question_sha256 must be a SHA-256 in hex"),
        ({"text_sha256": ""}, "text_sha256 must be a SHA-256 in hex"),
        ({"output_sha256": "a" * 63}, "output_sha256 must be a SHA-256 in hex"),
        ({"persona": None}, "persona and error must be text"),
        ({"error": None}, "persona and error must be text"),
        ({"status": "collapsed"}, "status must be ok or failed, not 'collapsed'"),
        ({"label": 3}, "label must be text or None"),
        ({"value": 8}, "value must be a finite float or None"),
        ({"value": math.nan}, "value must be a finite float or None"),
        ({"value": math.inf}, "value must be a finite float or None"),
        ({"value": -math.inf}, "value must be a finite float or None"),
        ({"value": None}, "an ok row has a label or value and no error"),
        ({"error": "slow"}, "an ok row has a label or value and no error"),
        ({"status": "failed", "error": "busy"}, "a failed row has an error and no"),
        (
            {"status": "failed", "error": "busy", "value": None, "label": "owl"},
            "a failed row has an error and no label or value",
        ),
        ({"status": "failed", "value": None}, "a failed row has an error and no"),
        ({"attempts": -1}, "attempts must be a whole number from 0, not -1"),
        ({"details": "Clear."}, "details must be a mapping"),
        ({"details": {"reason": math.nan}}, "details must be JSON"),
        ({"details": {"seen": {1, 2}}}, "details must be JSON"),
        ({"details": {1: "one", "two": 2}}, "details must be JSON"),
        ({"details": {1: "one"}}, "details change when saved as JSON"),
        ({"details": {"window": (0, 1)}}, "details change when saved as JSON"),
        ({"created_at": "2026-11-13 09:00:00"}, "created_at must be a UTC time"),
        ({"span": replace(SPAN, sentence_idx=-1)}, "sentence_idx must be a whole"),
        ({"span": replace(SPAN, sentence_count=0)}, "sentence_count must be a whole"),
        ({"span": replace(SPAN, start_step=None)}, "start_step must be a whole"),
        ({"span": replace(SPAN, splitter_version=0)}, "splitter_version must be a"),
        ({"span": replace(SPAN, start_step=5)}, "a span's start must not pass its"),
        ({"span": replace(SPAN, start_char=20)}, "a span's start must not pass its"),
    ],
)
def test_check_row_refuses_a_malformed_row_naming_its_run(
    changes: dict[str, object], message: str
) -> None:
    check_row(_row())  # well formed as it is
    row = replace(_row(), **changes)
    with pytest.raises(ValueError, match=re.escape(message)) as refused:
        check_row(row)
    assert str(refused.value).startswith(f"the row of {row.run_id!r}: ")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "there are no rows to write"),
        ([_row(), _row(span=SPAN)], "write one table's rows, of one scorer version"),
        ([_row(), _row(scorer="speaks_as", label="owl")], "write one table's rows"),
        ([_row(), _row(scorer_version=2)], "write one table's rows"),
        ([_row(), _row(value=3.0)], "two rows of x-000001 have one key"),
        ([replace(_row(), run_id="x-000004")], "x-000004 is not a saved run of"),
        ([_row("x-000003")], "x-000003 failed, so it has no reply to score"),
        (
            [_row(output_sha256=output_sha256(RUNS["x-000002"]))],
            "x-000001's reply is not the one its row scored",
        ),
        ([_row(span=replace(SPAN, end_step=8))], "a span passes the end of its"),
        ([_row(span=replace(SPAN, end_char=36))], "a span passes the end of its"),
        ([_row("x-000002"), _row(value=math.nan)], "value must be a finite float"),
    ],
)
def test_write_rows_refuses_before_writing_anything(
    tmp_path: Path, rows: list[ScoreRow], message: str
) -> None:
    folder = tmp_path / "x"
    with pytest.raises(ValueError, match=re.escape(message)):
        _write(folder, rows)
    assert _files(folder) == []


def test_a_file_is_never_replaced_and_names_follow_the_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "x"
    paths = [_write(folder, [_row(value=float(n))]) for n in range(3)]
    # A new file each time, named after the one before, however quick: so
    # of rows of one key and one created_at, the later write's file is the
    # later file by name, as the views take it (pushlab.store.score_views).
    assert sorted(paths) == paths and len(set(paths)) == 3
    assert [row.value for row in read_rows(folder, SCORES)] == [0.0, 1.0, 2.0]

    class Frozen(datetime):
        """A clock stopped at 9:00 UTC on 13 November 2026."""

        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return datetime(2026, 11, 13, 9, 0, tzinfo=UTC)

    # With the clock stopped and every random name the same, the second
    # write's name is taken: it is refused, and the file stays as it was.
    monkeypatch.setattr(scores, "datetime", Frozen)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=7))
    other = tmp_path / "y"
    first = _write(other, [_row()])
    with pytest.raises(FileExistsError, match=re.escape(f"{first} already exists")):
        _write(other, [_row(value=1.0)])
    assert read_rows(other, SCORES) == [_row()]
    assert _files(other) == [first.name]


def _schema(name: str, field: pa.Field) -> pa.Schema:
    """SCORES_SCHEMA with its column name replaced by field."""
    return SCORES_SCHEMA.set(SCORES_SCHEMA.get_field_index(name), field)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            SCORES_SCHEMA.remove(SCORES_SCHEMA.get_field_index("created_at")),
            "its columns are run_id, version, scorer, ",
        ),
        (SENTENCES_SCHEMA, "its columns are run_id, version, sentence_idx, "),
        (
            _schema("version", pa.field("version", pa.int32(), False)),
            "version is int32",
        ),
        (_schema("value", pa.field("value", pa.float32())), "value is float32, not "),
        (
            _schema("label", pa.field("label", pa.string(), False)),
            "label may not be null, unlike the table's",
        ),
        (
            _schema("scorer", pa.field("scorer", pa.string())),
            "scorer may be null, unlike the table's",
        ),
    ],
)
def test_read_rows_refuses_a_file_of_another_schema_naming_it(
    tmp_path: Path, schema: pa.Schema, message: str
) -> None:
    folder = tmp_path / "x"
    _write(folder, [_row()])
    odd = folder / SCORES / "odd.parquet"
    pq.write_table(schema.empty_table(), odd)
    with pytest.raises(ValueError, match=re.escape(message)) as refused:
        read_rows(folder, SCORES)
    assert str(refused.value).startswith(f"{odd} is not a scores file: ")


def test_hidden_and_other_files_are_not_read(tmp_path: Path) -> None:
    folder = tmp_path / "x"
    written = _write(folder, [_row()])
    hidden = written.with_name(f".{written.name}")  # as a copy a tool left
    hidden.write_bytes(written.read_bytes())
    (folder / SCORES / "notes.txt").write_text("not a table", encoding="utf-8")
    assert read_rows(folder, SCORES) == [_row()]


def test_a_write_that_fails_midway_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "x"

    def full(table: pa.Table, where: object, **options: object) -> None:
        where.write(b"PAR1")  # type: ignore[attr-defined]  # a start, then
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pq, "write_table", full)
    for rows in ([_row()], [_row(span=SPAN, text_sha256=text_sha256(FIRST))]):
        with pytest.raises(OSError, match="No space left on device"):
            _write(folder, rows)
    assert _files(folder) == []  # neither the file nor its temporary one
    monkeypatch.undo()
    assert read_rows(folder, SCORES) == []
    _write(folder, [_row()])  # and the next write works
    assert read_rows(folder, SCORES) == [_row()]


def test_remove_leftovers_sweeps_a_crashed_writes_temporary_files(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "x"
    kept = _write(folder, [_row()])
    window = _write(folder, [_row(span=SPAN, text_sha256=text_sha256(FIRST))])
    # What a write killed before its rename leaves: its temporary file, as
    # folder.write_whole names it, beside the file it was to be.
    leftovers = [
        path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        for path in (kept, window)
    ]
    for path in leftovers:
        path.write_bytes(b"PAR1")
    with locked(folder):
        assert shards.remove_leftovers(folder) == 2
    assert _files(folder) == sorted([kept.name, window.name])
    assert read_rows(folder, SCORES) == [_row()]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(scores)
    assert result.attempted > 0
    assert result.failed == 0
