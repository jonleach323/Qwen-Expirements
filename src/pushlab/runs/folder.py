"""A runner folder: where `pushlab run` saves one experiment, batch by batch.

The folder is <runs dir>/<experiment name>, and holds:
- experiment.json: what the folder is, its identity. The experiment's name
  and version, and the hashes of what its runs were made from: the
  experiment file, the prompt set, units/known.csv, the model download and
  the calibration. A new version starts when one of them changes.
- shards/ and steps/: two Parquet files per saved batch, of one name, its
  runs and its runs' steps (pushlab.runs.shards).
- index.jsonl: the progress index, one line for each shard a version
  counts, saying which runs it holds and how each ended, with the batch's
  time and tokens.
- heartbeat.json and progress.json, rewritten after every batch, and the
  experiment's manifest.json; .lock, held while a runner works on the
  folder.
- scores/ and sentences/, once pushlab score has judged the runs: Parquet
  files of verdicts on whole replies and on windows of their sentences
  (pushlab.store.scores), written whole while pushlab score holds the lock.

A batch's files are written to hidden temporary files, flushed to the disk
and renamed into place (write_whole), and only then is the batch marked in
the index. So a crash at any moment leaves each batch either marked, and
saved, or not marked: a shard the index does not name is ignored, and its
batch is run again under the next attempt's name. A line of the index
that a crash cut short is ignored when read, and cut off by the next
runner. Only a runner holding the lock writes, and the system lets go of
the lock when the process ends, however it ends.
"""

import fcntl
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# The pod's volume, where the docs put run folders.
RUNS_DIR = Path("/workspace/runs")
IDENTITY_FILE = "experiment.json"
INDEX_FILE = "index.jsonl"
SHARDS = "shards"
STEPS = "steps"
SCORES = "scores"
SENTENCES = "sentences"
HEARTBEAT_FILE = "heartbeat.json"
PROGRESS_FILE = "progress.json"
LOCK_FILE = ".lock"


def default_runs_dir() -> Path:
    """The folder that holds one runner folder per experiment.

    It is /workspace/runs unless PUSHLAB_RUNS_DIR names another.

    Example:
        >>> default_runs_dir()  # doctest: +SKIP
        PosixPath('/workspace/runs')
    """
    return Path(os.environ.get("PUSHLAB_RUNS_DIR") or RUNS_DIR)


class FolderInUse(RuntimeError):
    """Another process holds the folder's lock: a runner is working on it."""


@contextmanager
def locked(folder: Path) -> Iterator[None]:
    """Hold the folder's lock for the with block, making the folder if need be.

    The lock is an exclusive flock on folder/.lock, refused with FolderInUse
    when another process holds it for more than a moment (pushlab status
    holds a shared one for an instant). The file itself is never written,
    and the system lets go of the lock when the process ends.

    Example:
        >>> with locked(Path("/workspace/runs/s2-pulse-lengths")):  # doctest: +SKIP
        ...     pass
    """
    folder.mkdir(parents=True, exist_ok=True)
    handle = os.open(folder / LOCK_FILE, os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        for _attempt in range(20):  # about a tenth of a second
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.005)
        else:
            raise FolderInUse(f"another pushlab run is working on {folder}")
        yield
    finally:
        os.close(handle)  # which lets go of the lock


def _sync_folder(path: Path) -> None:
    """Flush a folder's entries to the disk, so a rename in it survives a crash."""
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def write_whole(path: Path, write: Any) -> None:
    """Write path through a hidden temporary file, flushed, then renamed.

    write(file) writes the content to the open temporary file. A crash
    leaves the old path as it was, or none, and a hidden .tmp file that
    the next runner deletes.

    Example:
        >>> write_whole(
        ...     Path("runs/x/progress.json"), lambda file: file.write(b"{}")
        ... )  # doctest: +SKIP
    """
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _sync_folder(path.parent)


def write_json(path: Path, data: object) -> None:
    """Replace path with data as JSON, whole or not at all.

    Example:
        >>> write_json(Path("runs/x/progress.json"), {"ok": 2})  # doctest: +SKIP
    """
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_whole(path, lambda file: file.write(text.encode()))


def read_json(path: Path) -> Any:
    """The JSON in path, or None when there is no such file.

    Example:
        >>> read_json(Path("/no/such/file.json")) is None
        True
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None


def read_identity(folder: Path) -> dict[str, Any] | None:
    """The folder's experiment.json, or None for a folder without one.

    Example:
        >>> read_identity(Path("/no/such/folder")) is None
        True
    """
    identity = read_json(folder / IDENTITY_FILE)
    if identity is not None and not (
        isinstance(identity, dict) and {"experiment", "version"} <= set(identity)
    ):
        raise ValueError(f"{folder / IDENTITY_FILE} lacks experiment and version")
    return identity


def write_identity(folder: Path, identity: dict[str, Any]) -> None:
    """Replace the folder's experiment.json.

    Example:
        >>> write_identity(folder, {"experiment": "x", "version": 1})  # doctest: +SKIP
    """
    write_json(folder / IDENTITY_FILE, identity)


def read_index(folder: Path) -> list[dict[str, Any]]:
    """Every whole line of the folder's index, oldest first.

    A last line without its line feed is one a crash cut short, or one a
    runner is writing, and is left out. Any other line that is not a JSON
    object with a version, a shard and its runs is refused.

    Example:
        >>> read_index(Path("/no/such/folder"))
        []
    """
    path = folder / INDEX_FILE
    try:
        lines = path.read_bytes().split(b"\n")
    except FileNotFoundError:
        return []
    lines.pop()  # what follows the last line feed: nothing, or a torn line
    entries = []
    for number, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{path}, line {number} is not JSON: {error}") from None
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("version"), int)
            and isinstance(entry.get("shard"), str)
            and isinstance(entry.get("runs"), dict)
        ):
            raise ValueError(
                f"{path}, line {number} is not a JSON object with a version, a "
                "shard and its runs"
            )
        entries.append(entry)
    return entries


def repair_index(folder: Path) -> bool:
    """Cut off a last line that a crash left unfinished; True if there was one.

    Call it holding the lock, so that no runner is writing the line.

    Example:
        >>> repair_index(Path("/no/such/folder"))
        False
    """
    path = folder / INDEX_FILE
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return False
    keep = content.rfind(b"\n") + 1
    if keep == len(content):
        return False
    with open(path, "r+b") as file:
        file.truncate(keep)
        file.flush()
        os.fsync(file.fileno())
    return True


def append_index(folder: Path, entry: dict[str, Any]) -> None:
    """Add one line to the folder's index, flushed to the disk.

    Example:
        >>> append_index(
        ...     folder, {"version": 1, "shard": "v1-b000001-1.parquet"}
        ... )  # doctest: +SKIP
    """
    line = (json.dumps(entry, sort_keys=True, allow_nan=False) + "\n").encode()
    handle = os.open(folder / INDEX_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        while line:
            line = line[os.write(handle, line) :]
        os.fsync(handle)
    finally:
        os.close(handle)


def latest(entries: list[dict[str, Any]], version: int) -> dict[str, dict[str, Any]]:
    """For each run of a version, how its latest mark says it ended.

    Each value has the run's status, the shard that holds it, its batch and
    the batch's error, if it failed.

    Example:
        >>> entry = {
        ...     "version": 1,
        ...     "shard": "s",
        ...     "batch": "b1",
        ...     "runs": {"x-000001": "ok"},
        ... }
        >>> latest([entry], version=1)["x-000001"]["status"]
        'ok'
    """
    states: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("version") != version:
            continue
        for run_id, status in entry["runs"].items():
            states[run_id] = {
                "status": status,
                "shard": entry["shard"],
                "batch": entry.get("batch"),
                "error": entry.get("error"),
            }
    return states


def used_seconds(entries: list[dict[str, Any]], version: int) -> float:
    """The time a version's saved batches took, kept ones included.

    Example:
        >>> used_seconds([{"version": 1, "seconds": 2.5}], version=1)
        2.5
    """
    return sum(e.get("seconds", 0.0) for e in entries if e.get("version") == version)


def lock_held(folder: Path) -> bool:
    """Whether some process holds the folder's lock, as a runner does.

    It tries a shared lock for an instant and lets it go, and never makes
    the lock file.

    Example:
        >>> lock_held(Path("/no/such/folder"))
        False
    """
    try:
        handle = os.open(folder / LOCK_FILE, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(handle)  # which lets go of the shared lock, if it was taken
    return False
