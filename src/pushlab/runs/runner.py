"""`pushlab run`: an experiment file run batch by batch, each saved before the next.

Before any model loads, the runner reads and expands the experiment file,
checks that its folder, <runs dir>/<name>, is this experiment's (or new),
takes the folder's lock, and works out what is left (pushlab.runs.plan):
the calls with runs not saved yet, or with --retry-failed those with
failed runs. A call is one generate() call: planned batches that share
the model, the decoding settings and the seed, and with group_by_length
(the default) prompts of one length in tokens, so that no row is padded,
up to batch_size runs. It loads the model and the calibration once, then
runs the calls in plan order.

After each call it saves the records as a shard and marks them in the
index (pushlab.runs.folder), then rewrites the heartbeat and progress
files (pushlab.runs.session). A call that raises is saved with its runs
failed and the error, and the runner goes on. It stops cleanly after the
call that brings the version's GPU time to max_gpu_hours, after --limit
runs (in whole calls), or at Ctrl-C, which loses only the call it stops.

The exit status is 0 when it finished, or had nothing to do; 1 when it
refuses or cannot save; 2 when a run failed this time; 3 when it stopped
with runs left, at max_gpu_hours or --limit; 130 at Ctrl-C. The first of
130, 1, 2 and 3 that applies wins.
"""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pushlab.runs import folder, plan, versions
from pushlab.runs.estimate import exact
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.plan import Call, differences
from pushlab.runs.session import EXIT_STOPPED, GO_ON, Session, refuse, runs_text
from pushlab.runs.spec import ExperimentSpec, read_spec

# The wall clock of a session, for the GPU-hours: tests stand a clock in.
_clock = time.monotonic


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
        return refuse(str(error))
    try:
        expansion = expand(spec)
    except ValueError as error:
        return refuse(f"{spec_path}: {error}")
    for warning in expansion.warnings:
        print(f"pushlab run: warning: {warning}", file=sys.stderr)
    runs_dir = folder.default_runs_dir() if runs_dir is None else runs_dir
    path = runs_dir / spec.name
    try:
        if problem := plan.runs_dir_problem(runs_dir) or plan.folder_problem(
            path, spec.name
        ):
            return refuse(problem)
        if dry_run:
            from pushlab.runs.dry_run import print_dry_run

            return print_dry_run(
                spec, expansion, path, runs_dir, limit, retry_failed, overwrite_version
            )
        inputs = plan.current_inputs(spec, expansion)
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
        return refuse(f"{error}{pid}")
    except (FileNotFoundError, ValueError) as error:  # the inputs or the folder
        return refuse(str(error))


def layout(
    spec: ExperimentSpec, expansion: Expansion, prompts_sha256: str
) -> list[Call]:
    """Every call that runs the expansion, grouped as the spec says.

    With group_by_length it reads the prompts' lengths with the model's
    tokenizer, which must be downloaded; prompts_sha256 is the hash the
    prompt set must still have.
    """
    lengths = None
    if spec.group_by_length:
        from pushlab.models.load import load_tokenizer
        from pushlab.prompts import read_prompt_set

        prompt_set = read_prompt_set(spec.prompts_path)
        if prompt_set.sha256 != prompts_sha256:
            raise ValueError(
                f"{spec.prompts_path} changed while it was read; run again"
            )
        tokenizer = load_tokenizer(spec.model)
        lengths = plan.prompt_lengths(
            tokenizer, list(prompt_set.prompt_ids), list(prompt_set.prompts)
        )
    return plan.calls(expansion, spec.batch_size, lengths)


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
        return refuse(problem)
    folder.remove_leftovers(path)
    folder.repair_index(path)
    identity = folder.read_identity(path)
    planned: list[Call] | None = None
    if identity is None:
        identity = plan.new_identity(spec, spec_path, expansion, inputs, 1)
        folder.write_identity(path, identity)
    elif changed := differences(identity, inputs):
        if not overwrite_version:
            return refuse(plan.changed_message(path, identity, changed))
        planned = layout(spec, expansion, inputs["prompts_sha256"])
        rows = {
            run.record.run_id: row
            for call in planned
            for row, run in enumerate(call.runs)
        }
        try:
            identity, kept = versions.start_version(
                path, identity, spec, spec_path, expansion, inputs, rows
            )
        except KeyboardInterrupt:
            return refuse("stopped by Ctrl-C before the new version started; run again")
        print(
            f"Started version {identity['version']}; it keeps {runs_text(kept)} saved."
        )
    version, total = identity["version"], len(expansion.runs)
    entries = folder.read_index(path)
    states = folder.latest(entries, version)
    counts = Counter(state["status"] for state in states.values())
    left = sum(
        plan.is_due(states.get(run.record.run_id), retry_failed)
        for run in expansion.runs
    )
    if not left:
        if retry_failed:
            print(f"Nothing to retry: no run of version {version} failed.")
            return 0
        ok, failed = counts["ok"], counts["failed"]
        retry = f"; rerun {'it' if failed == 1 else 'them'} with --retry-failed"
        print(
            f"Nothing to run: all {total} runs of version {version} are saved, "
            f"{ok} ok and {failed} failed{retry if failed else ''}."
        )
        return 0
    used = folder.used_seconds(entries, version)
    if used >= spec.max_gpu_hours * 3600:
        print(
            f"max_gpu_hours {exact(spec.max_gpu_hours)} is used up: "
            f"{used / 3600:.2f} GPU-hours are saved for version {version}; "
            f"{runs_text(left)} left. {GO_ON}"
        )
        return EXIT_STOPPED
    if planned is None:
        planned = layout(spec, expansion, identity["prompts_sha256"])
    todo = plan.work(planned, states, retry_failed)
    chosen = plan.within_limit(todo, limit)
    if not chosen:
        print(
            f"Nothing run: the next batch holds {runs_text(len(todo[0].due))}, more "
            f"than --limit {limit}."
        )
        return EXIT_STOPPED
    session = Session(spec, expansion, path, identity, states, used, left, _clock)
    return session.run(chosen, limit)
