"""A runner folder's batches: each one's runs shard and steps file.

Each saved batch is two Parquet files of one name: shards/<name>, its runs
in the runs table's schema (pushlab.store.records), and steps/<name>, its
runs' steps (pushlab.store.steps), named v<version>-<batch>-<attempt>.parquet.
Both are written to hidden temporary files, flushed to the disk and renamed
into place, and only then is the batch marked in the index
(pushlab.runs.folder), so a batch counts with both files whole or not at
all. A name is never used twice, and a file is never replaced: a batch run
again takes the next attempt's name.
"""

from pathlib import Path

import pyarrow.parquet as pq

from pushlab.runs.folder import (
    SCORES,
    SENTENCES,
    SHARDS,
    STEPS,
    latest,
    read_identity,
    read_index,
    write_whole,
)
from pushlab.store.records import RunRecord, read_runs_file, runs_table
from pushlab.store.steps import StepRecord, read_steps_file, steps_table


def next_shard_name(folder: Path, version: int, batch: str) -> str:
    """The name for the next attempt at a batch: one no shard or steps file
    has yet.

    Example:
        >>> next_shard_name(Path("/no/such/folder"), 1, "b000001")
        'v1-b000001-1.parquet'
    """
    pattern = f"v{version}-{batch}-*.parquet"
    places = (folder / SHARDS, folder / STEPS)
    taken = {path.name for place in places for path in place.glob(pattern)}
    attempt = 1
    while f"v{version}-{batch}-{attempt}.parquet" in taken:
        attempt += 1
    return f"v{version}-{batch}-{attempt}.parquet"


def _new_file(folder: Path, place: str, name: str) -> Path:
    """folder/place/name, which must not exist yet; the folder is made."""
    (folder / place).mkdir(parents=True, exist_ok=True)
    path = folder / place / name
    if path.exists():
        raise FileExistsError(f"{path} already exists; {place} are never replaced")
    return path


def write_shard(folder: Path, name: str, records: list[RunRecord]) -> Path:
    """Check records and save them as folder/shards/name, whole or not at all.

    An existing shard of that name raises FileExistsError: shards are never
    replaced.

    Example:
        >>> write_shard(folder, "v1-b000001-1.parquet", records)  # doctest: +SKIP
        PosixPath('.../shards/v1-b000001-1.parquet')
    """
    table = runs_table(records)
    path = _new_file(folder, SHARDS, name)
    write_whole(path, lambda file: pq.write_table(table, file))
    return path


def write_steps(folder: Path, name: str, steps: list[StepRecord]) -> Path:
    """Check steps and save them as folder/steps/name, whole or not at all.

    name is the shard's: a batch's steps file shares its runs shard's name.
    A batch with no reply, all of its runs failed, has an empty one.

    Example:
        >>> write_steps(folder, "v1-b000001-1.parquet", steps)  # doctest: +SKIP
        PosixPath('.../steps/v1-b000001-1.parquet')
    """
    table = steps_table(steps)
    path = _new_file(folder, STEPS, name)
    write_whole(path, lambda file: pq.write_table(table, file))
    return path


def remove_leftovers(folder: Path) -> int:
    """Delete the hidden temporary files of writes a crash stopped; how many.

    Call it holding the lock, so that no runner is writing them.

    Example:
        >>> remove_leftovers(Path("/no/such/folder"))
        0
    """
    places = [folder] + [folder / name for name in (SHARDS, STEPS, SCORES, SENTENCES)]
    leftovers = [path for place in places for path in place.glob(".*.tmp")]
    for path in leftovers:
        path.unlink()
    return len(leftovers)


def _by_shard(shards: dict[str, str]) -> dict[str, set[str]]:
    wanted: dict[str, set[str]] = {}
    for run_id, shard in shards.items():
        wanted.setdefault(shard, set()).add(run_id)
    return wanted


def read_records(folder: Path, shards: dict[str, str]) -> dict[str, RunRecord]:
    """The records of the given runs, each from the shard named for it.

    A run a shard holds twice (which only a hand edit makes, and verify
    reports) is read as its first copy; the DuckDB views take the same one
    unless the copies' statuses differ, when they take the index's.

    Example:
        >>> read_records(folder, {"x-000001": "v1-b000001-1.parquet"})  # doctest: +SKIP
        {'x-000001': RunRecord(run_id='x-000001', ...)}
    """
    records: dict[str, RunRecord] = {}
    for shard, run_ids in _by_shard(shards).items():
        found: dict[str, RunRecord] = {}
        for record in read_runs_file(folder / SHARDS / shard):
            if record.run_id in run_ids:
                found.setdefault(record.run_id, record)
        if missing := sorted(run_ids - set(found)):
            raise ValueError(
                f"{shard} lacks {', '.join(missing)}, which the index puts there"
            )
        records |= found
    return records


def read_steps(folder: Path, shards: dict[str, str]) -> dict[str, list[StepRecord]]:
    """The steps of the given runs, each from the steps file of the shard
    named for it, in step order. A run with no steps, such as a failed one,
    has an empty list; a step a file holds twice is read as its first copy.

    FileNotFoundError says when a shard has no steps file, as one saved
    before steps were kept has not.

    Example:
        >>> read_steps(folder, {"x-000001": "v1-b000001-1.parquet"})  # doctest: +SKIP
        {'x-000001': [StepRecord(run_id='x-000001', step=0, ...), ...]}
    """
    steps: dict[str, list[StepRecord]] = {run_id: [] for run_id in shards}
    seen: set[tuple[str, int]] = set()
    for shard, run_ids in _by_shard(shards).items():
        for step in read_steps_file(folder / STEPS / shard):
            if step.run_id in run_ids and (step.run_id, step.step) not in seen:
                seen.add((step.run_id, step.step))
                steps[step.run_id].append(step)
    return {
        run_id: sorted(found, key=lambda step: step.step)
        for run_id, found in steps.items()
    }


def _counted(folder: Path) -> dict[str, str]:
    """Each saved run of the folder's version, with the shard its latest
    mark names; empty for a folder without experiment.json."""
    identity = read_identity(folder)
    if identity is None:
        return {}
    states = latest(read_index(folder), identity["version"])
    return {run_id: state["shard"] for run_id, state in states.items()}


def read_results(folder: Path) -> list[RunRecord]:
    """Every saved run of the folder's version, once each, in run order.

    A run saved more than once, such as a failed run run again, counts as
    its latest mark says.

    Example:
        >>> read_results(Path("/workspace/runs/s2-pulse-lengths"))[
        ...     0
        ... ].status  # doctest: +SKIP
        'ok'
    """
    records = read_records(folder, _counted(folder))
    return [records[run_id] for run_id in sorted(records)]


def read_result_steps(folder: Path) -> dict[str, list[StepRecord]]:
    """The steps of every saved run of the folder's version, by run ID.

    FileNotFoundError names the first counted batch without a steps file, as
    one saved before steps were kept has none; the report reads such a
    folder batch by batch instead (pushlab.analysis.report_folder).

    Example:
        >>> read_result_steps(Path("/no/such/folder"))
        {}
    """
    return read_steps(folder, _counted(folder))
