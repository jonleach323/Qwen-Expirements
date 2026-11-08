"""`pushlab run`: an experiment file run batch by batch, each saved before the next.

Before any model loads, the runner reads and expands the experiment file,
checks that its folder, <runs dir>/<name>, is this experiment's (or new),
takes the folder's lock, and works out what is left (pushlab.runs.plan):
the planned batches with runs not saved yet, or with --retry-failed those
with failed runs. It loads the model and the calibration once, then runs
each batch with one generate() call. Every batch expand plans has one
prompt, one seed and one set of decoding settings, so its runs can share a
batch and its prompts are of equal length: no row is padded.

After each batch it saves the batch's records as a shard and marks them in
the index (pushlab.runs.folder), then rewrites the heartbeat and progress
files. A batch that raises is saved with its runs failed and the error,
and the runner goes on. It stops cleanly after the batch that brings the
GPU time of the folder's version to max_gpu_hours, after --limit runs (in
whole batches), or at Ctrl-C, which loses only the batch it stops. At the
end it saves the speed it measured, for later estimates.

The exit status is 0 when the runs it ran were saved ok, or it stopped
cleanly; 1 when it refuses; 2 when some runs failed; 130 at Ctrl-C.
"""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pushlab.runs import folder, plan
from pushlab.runs.estimate import exact
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.plan import differences
from pushlab.runs.session import GO_ON, Session, runs_text
from pushlab.runs.spec import ExperimentSpec, read_spec

# The wall clock of a session, for the GPU-hours: tests stand a clock in.
_clock = time.monotonic


def _refuse(message: str) -> int:
    print(f"pushlab run: {message}", file=sys.stderr)
    return 1


def run_experiment(
    spec_path: Path,
    *,
    runs_dir: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    retry_failed: bool = False,
    overwrite_version: bool = False,
) -> int:
    """Run the experiment file at spec_path, or with dry_run say what it would do.

    Returns the exit status (see the module's docstring).

    Example:
        >>> run_experiment(
        ...     Path("experiments/p0/s2_pulse_lengths.yaml")
        ... )  # doctest: +SKIP
        0
    """
    try:
        spec = read_spec(spec_path)
    except ValueError as error:
        return _refuse(str(error))
    try:
        expansion = expand(spec)
    except ValueError as error:
        return _refuse(f"{spec_path}: {error}")
    for warning in expansion.warnings:
        print(f"pushlab run: warning: {warning}", file=sys.stderr)
    runs_dir = folder.default_runs_dir() if runs_dir is None else runs_dir
    path = runs_dir / spec.name
    if problem := plan.folder_problem(path, spec.name):
        return _refuse(problem)
    if dry_run:
        from pushlab.runs.dry_run import print_dry_run

        return print_dry_run(spec, expansion, path, runs_dir, limit, retry_failed)
    try:
        inputs = plan.current_inputs(spec, expansion)
    except (FileNotFoundError, ValueError) as error:
        return _refuse(str(error))
    try:
        with folder.locked(path):
            return _start(
                spec,
                spec_path,
                expansion,
                path,
                inputs,
                limit,
                retry_failed,
                overwrite_version,
            )
    except folder.FolderInUse as error:
        beat = folder.read_json(path / folder.HEARTBEAT_FILE) or {}
        pid = f" (pid {beat['pid']})" if beat.get("state") == "running" else ""
        return _refuse(f"{error}{pid}")


def _start(
    spec: ExperimentSpec,
    spec_path: Path,
    expansion: Expansion,
    path: Path,
    inputs: dict[str, Any],
    limit: int | None,
    retry_failed: bool,
    overwrite_version: bool,
) -> int:
    """Holding the lock: check the folder's version, then run what is left."""
    if problem := plan.folder_problem(path, spec.name):  # again, now it is ours
        return _refuse(problem)
    folder.remove_leftovers(path)
    folder.repair_index(path)
    identity = folder.read_identity(path)
    if identity is None:
        identity = plan.new_identity(spec, spec_path, expansion, inputs, 1)
        folder.write_identity(path, identity)
    elif changed := differences(identity, inputs):
        if not overwrite_version:
            return _refuse(plan.changed_message(path, identity, changed))
        identity, kept = plan.start_version(
            path, identity, spec, spec_path, expansion, inputs
        )
        print(
            f"Started version {identity['version']}; it keeps {runs_text(kept)} saved."
        )
    version, total = identity["version"], len(expansion.runs)
    entries = folder.read_index(path)
    states = folder.latest(entries, version)
    todo = plan.work(expansion, states, retry_failed)
    counts = Counter(state["status"] for state in states.values())
    if not todo:
        if retry_failed:
            print(f"Nothing to retry: no run of version {version} failed.")
            return 0
        ok, failed = counts["ok"], counts["failed"]
        retry = "; rerun them with --retry-failed" if failed else ""
        print(
            f"Nothing to run: all {total} runs of version {version} are saved, "
            f"{ok} ok and {failed} failed{retry}."
        )
        return 0
    used = folder.used_seconds(entries, version)
    left = sum(len(batch.runs) for batch in todo)
    if used >= spec.max_gpu_hours * 3600:
        print(
            f"max_gpu_hours {exact(spec.max_gpu_hours)} is used up: "
            f"{used / 3600:.2f} GPU-hours are saved for version {version}; "
            f"{runs_text(left)} left. {GO_ON}"
        )
        return 0
    chosen = plan.within_limit(todo, limit)
    if not chosen:
        print(
            f"Nothing run: the next batch holds {len(todo[0].runs)} runs, more "
            f"than --limit {limit}."
        )
        return 0
    session = Session(spec, expansion, path, identity, states, used, left, _clock)
    return session.run(chosen, limit)
