"""What a run of an experiment folder will do, worked out before any model loads.

- Whose folder it is: the runner works only in a folder that is empty or
  holds a runner folder of the same experiment.
- What its runs were made from: the hashes of the experiment file, the
  prompt set, units/known.csv, the model download and the calibration. A
  folder made from other ones is refused, unless a new version is asked
  for; a new version keeps every saved run that would be made the same way
  again, and plans the rest afresh.
- What is left: the planned batches with runs still to run, or with runs
  that failed, in plan order.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.runs import folder
from pushlab.runs.expand import Expansion, PlannedRun
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.folders import shown_path
from pushlab.store.records import RunRecord

# Each input a folder's runs are made from, and its name in messages.
IDENTITY_KEYS = {
    "spec_sha256": "experiment file",
    "prompts_sha256": "prompt set",
    "known_sha256": "known units",
    "model": "model download",
    "calibration_sha256": "calibration",
}
# What a run is made of: a saved run with all of these equal is the same run.
# Its batch_key says where it ran, so a run saved in another layout is too.
SAME_RUN = ("run_id", "experiment", "model", "prompt_id", "schedule", "decoding")
SAME_RUN += ("seed", "control_of")


def differences(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """The names of the inputs that differ between two identities, in order.

    Example:
        >>> differences({"spec_sha256": "a"}, {"spec_sha256": "b"})
        ['experiment file']
    """
    return [
        name
        for key, name in IDENTITY_KEYS.items()
        if key in current and recorded.get(key) != current[key]
    ]


def folder_problem(path: Path, name: str) -> str | None:
    """Why the runner must not work in path for the experiment name, or None.

    Example:
        >>> folder_problem(Path("/no/such/folder"), "x") is None
        True
    """
    if not path.exists():
        return None
    if not path.is_dir():
        return f"{path} is not a folder"
    never = "the runner never writes into another experiment's folder"
    identity = folder.read_identity(path)
    if identity is not None:
        if identity["experiment"] != name:
            return (
                f"{path} belongs to the experiment {identity['experiment']!r}, "
                f"not {name!r}; {never}"
            )
        return None
    names = sorted(p.name for p in path.iterdir() if p.name != folder.LOCK_FILE)
    if names:
        held = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
        return f"{path} is not a runner folder: it holds {held}; {never}"
    return None


def current_inputs(spec: ExperimentSpec, expansion: Expansion) -> dict[str, Any]:
    """The hashes of what the spec's runs would be made from now.

    It reads the model's download record and, when anything is pushed, its
    calibration, but loads no model. FileNotFoundError or ValueError says
    what is missing and how to make it.
    """
    from pushlab.models import load
    from pushlab.store.environment import file_sha256
    from pushlab.units import calibrate

    revision = load.downloaded_revision(spec.model)
    calibration_sha256 = None
    if any(run.record.schedule for run in expansion.runs):
        calibrate.load_calibration(spec.model, revision)  # checked, or refused
        calibration_sha256 = file_sha256(
            calibrate.calibration_path(spec.model, revision)
        )
    return {
        "spec_sha256": expansion.spec_sha256,
        "prompts_sha256": expansion.prompts_sha256,
        "known_sha256": expansion.known_sha256,
        "model": f"{spec.model}@{revision}",
        "calibration_sha256": calibration_sha256,
    }


def changed_message(path: Path, identity: dict[str, Any], changed: list[str]) -> str:
    """Why a folder made from other inputs is refused, and what to do."""
    version = identity["version"]
    return (
        f"{path} holds version {version} of {identity['experiment']}, made from "
        f"another {' and '.join(changed)}. Run with --overwrite-version to start "
        f"version {version + 1}, which keeps the saved runs that did not change, "
        "or move the folder away to start afresh"
    )


def new_identity(
    spec: ExperimentSpec,
    spec_path: Path,
    expansion: Expansion,
    inputs: dict[str, Any],
    version: int,
) -> dict[str, Any]:
    """What experiment.json says about a new version of the folder."""
    return inputs | {
        "experiment": spec.name,
        "version": version,
        "spec_file": shown_path(spec_path),
        "spec": spec.to_mapping(),
        "runs": len(expansion.runs),
        "max_gpu_hours": spec.max_gpu_hours,
        "max_new_tokens": spec.max_new_tokens,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


@dataclass(frozen=True)
class Batch:
    """A planned batch's runs that are to run: all of it, or the part left."""

    key: str
    runs: tuple[PlannedRun, ...]


def work(
    expansion: Expansion, states: dict[str, dict[str, Any]], retry_failed: bool
) -> list[Batch]:
    """The planned batches with runs to run, in plan order.

    Those are the runs not saved yet, or with retry_failed the runs whose
    latest mark is failed, and nothing else.
    """
    batches: dict[str, list[PlannedRun]] = {}
    for run in expansion.runs:
        state = states.get(run.record.run_id)
        if retry_failed:
            due = state is not None and state["status"] == "failed"
        else:
            due = state is None
        if due:
            batches.setdefault(run.record.batch_key.split(":")[0], []).append(run)
    return [Batch(key, tuple(runs)) for key, runs in batches.items()]


def within_limit(batches: list[Batch], limit: int | None) -> list[Batch]:
    """The first batches whose runs add up to at most limit, if one is given.

    Example:
        >>> batches = [Batch("b1", (1, 2)), Batch("b2", (3, 4))]  # runs, as numbers
        >>> within_limit(batches, 3)  # type: ignore[arg-type]
        [Batch(key='b1', runs=(1, 2))]
    """
    if limit is None:
        return batches
    chosen, count = [], 0
    for batch in batches:
        if count + len(batch.runs) > limit:
            break
        chosen.append(batch)
        count += len(batch.runs)
    return chosen


def same_run(saved: RunRecord, planned: RunRecord) -> bool:
    """Whether a saved run is the planned one: made of the same, wherever it ran."""
    return all(getattr(saved, name) == getattr(planned, name) for name in SAME_RUN)


def start_version(
    path: Path,
    identity: dict[str, Any],
    spec: ExperimentSpec,
    spec_path: Path,
    expansion: Expansion,
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Make the folder's next version; return its identity and how many saved
    runs it keeps.

    A saved run is kept when the new plan has a run of the same ID made of
    the same (same_run), from the same prompt set and calibration: the index
    marks it again for the new version, in the shard that holds it, with its
    share of that batch's time. The marks go first, then experiment.json,
    so a crash in between leaves the old version as it was.
    """
    old = identity["version"]
    entries = folder.read_index(path)
    states = folder.latest(entries, old)
    kept: dict[str, list[str]] = {}
    same_inputs = all(
        identity.get(key) == inputs[key]
        for key in ("prompts_sha256", "calibration_sha256")
    )
    if same_inputs:
        planned = {run.record.run_id: run.record for run in expansion.runs}
        ok = {k: v["shard"] for k, v in states.items() if v["status"] == "ok"}
        for run_id, saved in folder.read_records(path, ok).items():
            plan = planned.get(run_id)
            if plan is not None and same_run(
                saved, replace(plan, model=inputs["model"])
            ):
                kept.setdefault(ok[run_id], []).append(run_id)
    seconds = {
        e["shard"]: (e.get("seconds", 0.0), len(e["runs"]))
        for e in entries
        if e.get("version") == old
    }
    for shard, run_ids in kept.items():
        taken, count = seconds[shard]
        folder.append_index(
            path,
            {
                "version": old + 1,
                "shard": shard,
                "batch": states[run_ids[0]]["batch"],
                "runs": dict.fromkeys(run_ids, "ok"),
                "kept_from": old,
                "seconds": taken * len(run_ids) / count,
                "reply_tokens": 0,
            },
        )
    fresh = new_identity(spec, spec_path, expansion, inputs, old + 1)
    fresh["previous"] = [*identity.get("previous", []), _summary(identity)]
    folder.write_identity(path, fresh)
    return fresh, sum(len(run_ids) for run_ids in kept.values())


def _summary(identity: dict[str, Any]) -> dict[str, Any]:
    keys = ("version", "created_at", *IDENTITY_KEYS)
    return {key: identity.get(key) for key in keys}
