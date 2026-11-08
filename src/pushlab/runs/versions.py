"""New versions of a runner folder: which saved runs one keeps, and starting it.

A folder's runs are made from an experiment file, a prompt set,
units/known.csv, a model download and a calibration (pushlab.runs.plan).
When one of them changes, `pushlab run --overwrite-version` starts the next
version. It keeps every saved run, ok or failed, that the new plan would
make the same way (same_run), from the same prompt set and calibration,
and under sampling, if it ended ok, on the same row; the index marks those
again for the new version, with the session that made them and their
share of their batch's time, and the rest are to run. The marks go first
and experiment.json last, under a number no mark has used, so a crash in
between changes nothing that counts.
"""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from pushlab.runs import folder, shards
from pushlab.runs.expand import Expansion
from pushlab.runs.plan import IDENTITY_KEYS, new_identity
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.records import RunRecord

# What a run is made of: a saved run with all of these equal is the same run.
# Its batch_key says where it ran, so a run saved in another layout is too.
SAME_RUN = ("run_id", "experiment", "model", "prompt_id", "schedule", "decoding")
SAME_RUN += ("seed", "control_of")


def same_run(saved: RunRecord, planned: RunRecord) -> bool:
    """Whether a saved run is the planned one: made of the same, wherever it ran.

    Example:
        >>> same_run(record, record)  # doctest: +SKIP
        True
    """
    return all(getattr(saved, name) == getattr(planned, name) for name in SAME_RUN)


def next_version(identity: dict[str, Any], entries: list[dict[str, Any]]) -> int:
    """A version number no mark in the index has used yet.

    A crash while a version starts can leave marks of a number that never
    got its experiment.json; those are never adopted.

    Example:
        >>> next_version({"version": 1}, [{"version": 2}])
        3
    """
    used = [entry.get("version", 0) for entry in entries]
    return max([identity["version"], *used]) + 1


def kept_runs(
    path: Path,
    identity: dict[str, Any],
    expansion: Expansion,
    inputs: dict[str, Any],
    rows: dict[str, int],
    sampled: bool,
) -> dict[str, dict[str, Any]]:
    """The saved runs of the folder's version that a new version keeps.

    A run is kept, ok or failed as it ended, when the new plan has a run of
    the same ID made of the same (same_run), from the same prompt set and
    calibration. Under sampling a run that ended ok must also be on the row
    the new plan gives it (rows), since a sampled reply depends on its row;
    a failed run has no reply, so it is kept on any row, for --retry-failed
    to find.

    Example:
        >>> len(
        ...     kept_runs(path, identity, expansion, inputs, rows, False)
        ... )  # doctest: +SKIP
        4
    """
    if any(
        identity.get(k) != inputs.get(k)
        for k in ("prompts_sha256", "calibration_sha256")
    ):
        return {}
    states = folder.latest(folder.read_index(path), identity["version"])
    saved = shards.read_records(path, {k: v["shard"] for k, v in states.items()})
    planned = {run.record.run_id: run.record for run in expansion.runs}
    kept = {}
    for run_id, record in saved.items():
        plan = planned.get(run_id)
        if plan is None or not same_run(record, replace(plan, model=inputs["model"])):
            continue
        row = int(record.batch_key.rsplit(":", 1)[1])
        if sampled and record.status == "ok" and row != rows.get(run_id):
            continue
        kept[run_id] = states[run_id]
    return kept


def kept_seconds(
    entries: list[dict[str, Any]], version: int, kept: dict[str, dict[str, Any]]
) -> float:
    """The time the kept runs bring along: each one's share of its batch's.

    Example:
        >>> entry = {
        ...     "version": 1,
        ...     "shard": "s",
        ...     "runs": {"a": "ok", "b": "ok"},
        ...     "seconds": 4.0,
        ... }
        >>> kept_seconds([entry], 1, {"a": {"shard": "s"}})
        2.0
    """
    sources = {e["shard"]: e for e in entries if e.get("version") == version}
    shards = Counter(state["shard"] for state in kept.values())
    return sum(
        sources[shard].get("seconds", 0.0) * count / len(sources[shard]["runs"])
        for shard, count in shards.items()
    )


def start_version(
    path: Path,
    identity: dict[str, Any],
    spec: ExperimentSpec,
    spec_path: Path,
    expansion: Expansion,
    inputs: dict[str, Any],
    rows: dict[str, int],
) -> tuple[dict[str, Any], int]:
    """Make the folder's next version; return its identity and how many saved
    runs it keeps.

    The index marks each kept run again for the new version (kept_runs), in
    the shard that holds it, ok or failed with its error, with the session
    that made it and with its share of its batch's time. The marks go first
    and experiment.json last, under a version number never used, so a crash
    in between changes nothing that counts.

    Example:
        >>> start_version(path, identity, spec, spec_path, expansion, inputs, rows)[
        ...     1
        ... ]  # doctest: +SKIP
        4
    """
    sampled = spec.decoding["strategy"] == "sample"
    kept = kept_runs(path, identity, expansion, inputs, rows, sampled)
    entries = folder.read_index(path)
    old, new = identity["version"], next_version(identity, entries)
    sources = {e["shard"]: e for e in entries if e.get("version") == old}
    by_shard: dict[str, dict[str, str]] = {}
    for run_id, state in kept.items():
        by_shard.setdefault(state["shard"], {})[run_id] = state["status"]
    for shard, statuses in by_shard.items():
        source = sources[shard]
        share = len(statuses) / len(source["runs"])
        folder.append_index(
            path,
            {
                "version": new,
                "shard": shard,
                "batch": source.get("batch"),
                "runs": statuses,
                "error": source.get("error") if "failed" in statuses.values() else None,
                "kept_from": old,
                "session": source.get("session"),  # the one that made the runs
                "seconds": source.get("seconds", 0.0) * share,
                "reply_tokens": 0,
            },
        )
    fresh = new_identity(spec, spec_path, expansion, inputs, new)
    fresh["previous"] = [*identity.get("previous", []), _summary(identity)]
    folder.write_identity(path, fresh)
    return fresh, len(kept)


def _summary(identity: dict[str, Any]) -> dict[str, Any]:
    keys = ("version", "created_at", *IDENTITY_KEYS)
    return {key: identity.get(key) for key in keys}
