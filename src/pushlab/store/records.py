"""The two files every experiment writes: runs.parquet and manifest.json.

runs.parquet holds one row per run, with the runs fields of the development
plan ("Data model"). manifest.json records what produced the runs: the code,
the lockfile, the model, the machine and the times. Both are checked before
anything is written, and an existing file is only replaced when asked.
"""

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

RUNS_FILE = "runs.parquet"
MANIFEST_FILE = "manifest.json"

STATUSES = ("ok", "failed", "collapsed")

# Every run's decoding settings include these. The data type is among them
# because the plan records it with every run.
DECODING_KEYS = ("strategy", "max_new_tokens", "thinking", "dtype")

# Every manifest has these fields; it may have more.
REQUIRED_MANIFEST_FIELDS = (
    "code_commit",
    "lockfile_sha256",
    "model_id",
    "model_revision",
    "gpu",
    "cuda",
    "torch",
    "transformers",
    "started_at",
    "finished_at",
    "run_count",
)


@dataclass(frozen=True)
class RunRecord:
    """One run: what the model was asked, how, and what it replied.

    run_id is `<experiment>-<counter>`. schedule lists the pushes (empty for
    an unsteered run) and decoding the settings, including DECODING_KEYS;
    both are saved as JSON text. control_of names the run a random control
    mirrors and is empty otherwise. output_tokens are the reply's token IDs.

    Example:
        >>> run = RunRecord(
        ...     run_id="s0-baseline-000001",
        ...     experiment="s0-baseline",
        ...     model="qwen3.5-4b@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ...     prompt_id="s0-neuron",
        ...     schedule=[],
        ...     decoding={
        ...         "strategy": "greedy",
        ...         "max_new_tokens": 200,
        ...         "thinking": False,
        ...         "dtype": "bfloat16",
        ...     },
        ...     seed=0,
        ...     batch_key="b000001:0",
        ...     control_of="",
        ...     output_text="A neuron is one number inside the model.",
        ...     output_tokens=[32, 48273, 374],
        ...     status="ok",
        ... )
        >>> run.status
        'ok'
    """

    run_id: str
    experiment: str
    model: str
    prompt_id: str
    schedule: list[dict[str, Any]]
    decoding: dict[str, Any]
    seed: int
    batch_key: str
    control_of: str
    output_text: str
    output_tokens: list[int]
    status: str


def _text(name: str) -> pa.Field:
    return pa.field(name, pa.string(), nullable=False)


RUNS_SCHEMA = pa.schema(
    [
        _text("run_id"),
        _text("experiment"),
        _text("model"),
        _text("prompt_id"),
        _text("schedule"),  # JSON
        _text("decoding"),  # JSON
        pa.field("seed", pa.int64(), nullable=False),
        _text("batch_key"),
        _text("control_of"),
        _text("output_text"),
        pa.field("output_tokens", pa.list_(pa.int32()), nullable=False),
        _text("status"),
    ]
)


def make_run_id(experiment: str, counter: int) -> str:
    """Return the run ID `<experiment>-<counter>`, counting from 1.

    Example:
        >>> make_run_id("s0-baseline", 1)
        's0-baseline-000001'
    """
    if counter < 1:
        raise ValueError(f"the run counter starts at 1, not {counter}")
    return f"{experiment}-{counter:06d}"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _to_json(name: str, value: object) -> str:
    """value as compact JSON text, refusing anything JSON would change."""
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be plain JSON: {error}") from error
    if json.loads(text) != value:
        raise ValueError(f"{name} changes when saved as JSON: {value!r}")
    return text


def _check(record: RunRecord) -> dict[str, Any]:
    """Raise ValueError unless record is well formed; return it as a row."""
    for name in ("run_id", "experiment", "model", "prompt_id", "batch_key"):
        value = getattr(record, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty text, not {value!r}")
    for name in ("control_of", "output_text", "status"):
        value = getattr(record, name)
        if not isinstance(value, str):
            raise ValueError(f"{name} must be text, not {value!r}")
    if not re.fullmatch(re.escape(record.experiment) + r"-\d+", record.run_id):
        raise ValueError(
            f"run_id {record.run_id!r} must be '{record.experiment}-<counter>'"
        )
    if record.status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, not {record.status!r}")
    if not _is_int(record.seed) or not 0 <= record.seed < 2**63:
        raise ValueError(
            f"seed must be a whole number from 0 to 2**63-1, not {record.seed!r}"
        )
    tokens = record.output_tokens
    if not isinstance(tokens, list) or not all(
        _is_int(token) and 0 <= token < 2**31 for token in tokens
    ):
        raise ValueError(f"output_tokens must be a list of token IDs, not {tokens!r}")
    if not isinstance(record.schedule, list):
        raise ValueError(
            f"schedule must be a list of segments, not {record.schedule!r}"
        )
    if not isinstance(record.decoding, dict):
        raise ValueError(f"decoding must be a dict, not {record.decoding!r}")
    if missing := [key for key in DECODING_KEYS if key not in record.decoding]:
        raise ValueError(f"decoding lacks {', '.join(missing)}")
    row = asdict(record)
    row["schedule"] = _to_json("schedule", record.schedule)
    row["decoding"] = _to_json("decoding", record.decoding)
    return row


def _refuse_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it"
        )


def _replace_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Write a temporary file next to path, then rename it to path.

    A reader sees the old file or the new one, never half of one, and a
    failed write leaves nothing behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        write(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_runs(
    records: Sequence[RunRecord], out_dir: Path, *, overwrite: bool = False
) -> Path:
    """Check records and save them as out_dir/runs.parquet; return its path.

    A malformed record raises ValueError and nothing is written. An existing
    runs.parquet raises FileExistsError unless overwrite is True.

    Example:
        >>> write_runs([run], Path("outputs/s0-baseline"))  # doctest: +SKIP
        PosixPath('outputs/s0-baseline/runs.parquet')
    """
    if not records:
        raise ValueError("no records to write")
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, RunRecord):
            name = type(record).__name__
            raise ValueError(f"records[{index}] is a {name}, not a RunRecord")
        try:
            rows.append(_check(record))
        except ValueError as error:
            raise ValueError(f"records[{index}]: {error}") from None
    run_ids = [row["run_id"] for row in rows]
    if duplicates := sorted({i for i in run_ids if run_ids.count(i) > 1}):
        raise ValueError(f"run_id {', '.join(duplicates)} appears more than once")
    path = Path(out_dir) / RUNS_FILE
    _refuse_existing(path, overwrite)
    table = pa.Table.from_pylist(rows, schema=RUNS_SCHEMA)
    _replace_atomically(path, lambda temporary: pq.write_table(table, temporary))
    return path


def read_runs(out_dir: Path) -> list[RunRecord]:
    """Read out_dir/runs.parquet back into run records.

    Example:
        >>> read_runs(Path("outputs/s0-baseline"))[0].status  # doctest: +SKIP
        'ok'
    """
    path = Path(out_dir) / RUNS_FILE
    table = pq.read_table(path)
    if not table.schema.equals(RUNS_SCHEMA):
        raise ValueError(
            f"{path} does not have the runs table's columns and types: "
            f"{table.schema.names}"
        )
    return [
        RunRecord(
            **row | {key: json.loads(row[key]) for key in ("schedule", "decoding")}
        )
        for row in table.to_pylist()
    ]


def write_manifest(
    out_dir: Path, info: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    """Save info as out_dir/manifest.json; return its path.

    info must have every field in REQUIRED_MANIFEST_FIELDS; one that does
    not apply, such as the GPU on a CPU run, is None. Times are ISO 8601
    text. Extra fields are kept. An existing manifest.json raises
    FileExistsError unless overwrite is True.

    Example:
        >>> write_manifest(Path("outputs/s0-baseline"), info)  # doctest: +SKIP
        PosixPath('outputs/s0-baseline/manifest.json')
    """
    if missing := [name for name in REQUIRED_MANIFEST_FIELDS if name not in info]:
        raise ValueError(f"the manifest lacks {', '.join(missing)}")
    count = info["run_count"]
    if not _is_int(count) or count < 0:
        raise ValueError(f"run_count must be a number of runs, not {count!r}")
    for name in ("started_at", "finished_at"):
        try:
            datetime.fromisoformat(info[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be an ISO 8601 time, such as "
                f"2026-10-29T10:00:00+00:00, not {info[name]!r}"
            ) from None
    try:
        text = json.dumps(dict(info), indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the manifest must be plain JSON: {error}") from error
    path = Path(out_dir) / MANIFEST_FILE
    _refuse_existing(path, overwrite)
    _replace_atomically(path, lambda temporary: temporary.write_text(text + "\n"))
    return path
