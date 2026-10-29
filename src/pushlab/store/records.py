"""The two files every experiment writes: runs.parquet and manifest.json.

runs.parquet holds one row per run, with the runs fields of the development
plan ("Data model"). manifest.json records what produced the runs: the code,
the lockfile, the model, the machine and the times. Both are checked before
anything is written, and an existing file is only replaced when asked.
"""

import json
import os
import re
import uuid
from collections import Counter
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

# Experiment names start every run ID, so they stay simple: lowercase letters
# and digits joined by hyphens, such as s0-baseline.
EXPERIMENT_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")

# The largest seed the runs table can hold (a signed 64-bit integer).
MAX_SEED = 2**63 - 1

# Every run's decoding settings include these. The data type is among them
# because the plan records it with every run.
DECODING_KEYS = ("strategy", "max_new_tokens", "thinking", "dtype")

# Every manifest has these fields; it may have more. Only the ones that may
# not apply (no git checkout, no uv.lock, no GPU) may be None.
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


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _check_decoding(decoding: object) -> None:
    if not isinstance(decoding, dict):
        raise ValueError(f"decoding must be a dict, not {decoding!r}")
    if missing := [key for key in DECODING_KEYS if key not in decoding]:
        raise ValueError(f"decoding lacks {', '.join(missing)}")
    for key in ("strategy", "dtype"):
        if not _is_text(decoding[key]):
            raise ValueError(f"decoding's {key} must be non-empty text")
    count = decoding["max_new_tokens"]
    if not _is_int(count) or count < 1:
        raise ValueError(f"decoding's max_new_tokens must be 1 or more, not {count!r}")
    if not isinstance(decoding["thinking"], bool):
        raise ValueError("decoding's thinking must be True or False")


def _check(record: RunRecord) -> dict[str, Any]:
    """Raise ValueError unless record is well formed; return it as a row."""
    for name in ("run_id", "experiment", "model", "prompt_id", "batch_key"):
        value = getattr(record, name)
        if not _is_text(value):
            raise ValueError(f"{name} must be non-empty text, not {value!r}")
    for name in ("control_of", "output_text", "status"):
        value = getattr(record, name)
        if not isinstance(value, str):
            raise ValueError(f"{name} must be text, not {value!r}")
    if not EXPERIMENT_NAME.fullmatch(record.experiment):
        raise ValueError(
            "experiment must be lowercase letters and digits joined by hyphens, "
            f"such as s0-baseline, not {record.experiment!r}"
        )
    # The one spelling make_run_id gives, so one counter has one run ID.
    counter = re.fullmatch(re.escape(record.experiment) + "-([0-9]+)", record.run_id)
    if (
        not counter
        or int(counter[1]) < 1
        or record.run_id != make_run_id(record.experiment, int(counter[1]))
    ):
        raise ValueError(
            f"run_id {record.run_id!r} must be '{record.experiment}-' and a "
            "counter from 000001, as make_run_id writes it"
        )
    if record.status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, not {record.status!r}")
    if not _is_int(record.seed) or not 0 <= record.seed <= MAX_SEED:
        raise ValueError(
            f"seed must be a whole number from 0 to {MAX_SEED}, not {record.seed!r}"
        )
    tokens = record.output_tokens
    if not isinstance(tokens, list) or not all(
        _is_int(token) and 0 <= token < 2**31 for token in tokens
    ):
        raise ValueError(f"output_tokens must be a list of token IDs, not {tokens!r}")
    schedule = record.schedule
    if not isinstance(schedule, list) or not all(isinstance(s, dict) for s in schedule):
        raise ValueError(f"schedule must be a list of segments, not {schedule!r}")
    _check_decoding(record.decoding)
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
    failed write leaves nothing behind. write creates the temporary file
    itself, so it gets the usual permissions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Absolute, so pyarrow never reads a folder name such as run:1 as a URL.
    temporary = path.absolute().with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
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
    counts = Counter(row["run_id"] for row in rows)
    if duplicates := sorted(run_id for run_id, n in counts.items() if n > 1):
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
    table = pq.read_table(path.absolute())
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


def check_manifest(info: Mapping[str, Any]) -> str:
    """Raise ValueError unless info is a valid manifest; return it as JSON text.

    info must have every field in REQUIRED_MANIFEST_FIELDS. The model ID and
    revision and the torch and transformers versions are text; the code
    commit, lockfile hash, GPU and CUDA may be None when they do not apply,
    such as the GPU on a CPU run. Times are ISO 8601 text with a time zone,
    and the run cannot finish before it starts. Extra fields are kept.

    Example:
        >>> check_manifest({"run_count": 1})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: the manifest lacks code_commit, lockfile_sha256, ...
    """
    if missing := [name for name in REQUIRED_MANIFEST_FIELDS if name not in info]:
        raise ValueError(f"the manifest lacks {', '.join(missing)}")
    for name in ("model_id", "model_revision", "torch", "transformers"):
        if not _is_text(info[name]):
            raise ValueError(f"{name} must be non-empty text, not {info[name]!r}")
    count = info["run_count"]
    if not _is_int(count) or count < 0:
        raise ValueError(f"run_count must be a number of runs, not {count!r}")
    times = {}
    for name in ("started_at", "finished_at"):
        try:
            times[name] = datetime.fromisoformat(info[name])
        except (TypeError, ValueError):
            times[name] = None
        if times[name] is None or times[name].tzinfo is None:
            raise ValueError(
                f"{name} must be an ISO 8601 time with a time zone, such as "
                f"2026-10-29T10:00:00+00:00, not {info[name]!r}"
            )
    if times["finished_at"] < times["started_at"]:
        raise ValueError("finished_at is before started_at")
    try:
        text = json.dumps(dict(info), indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the manifest must be plain JSON: {error}") from error
    return text + "\n"


def write_manifest(
    out_dir: Path, info: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    """Check info with check_manifest and save it as out_dir/manifest.json.

    Returns the file's path. An existing manifest.json raises
    FileExistsError unless overwrite is True.

    Example:
        >>> write_manifest(Path("outputs/s0-baseline"), info)  # doctest: +SKIP
        PosixPath('outputs/s0-baseline/manifest.json')
    """
    text = check_manifest(info)
    path = Path(out_dir) / MANIFEST_FILE
    _refuse_existing(path, overwrite)
    _replace_atomically(path, lambda temporary: temporary.write_text(text))
    return path
