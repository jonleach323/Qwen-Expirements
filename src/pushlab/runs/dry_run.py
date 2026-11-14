"""`pushlab run --dry-run`: what a run would do and cost, with nothing run.

It prints the experiment's settings, its calibration among them, and its
runs by condition (as `pushlab expand --summary` does), then the runs still
to run in its folder as the same command would take them (with --limit,
--retry-failed and --overwrite-version), their reply tokens, the speed it
estimates with, the GPU-hours against what is left of max_gpu_hours, and
the cost (pushlab.runs.estimate). It loads no model (only its tokenizer,
for the prompts' lengths) and writes nothing, so it works in a cloud
session too. There, where the model is not downloaded, it cannot read the
prompts' lengths in tokens, nor check the model download and the fixed
calibration's file, and says so; the per_prompt calibration has no file.
"""

from pathlib import Path
from typing import Any

from pushlab.runs import folder, plan, versions
from pushlab.runs.estimate import (
    estimate_lines,
    read_costs,
    speed_for,
    speeds_problem,
)
from pushlab.runs.expand import Expansion
from pushlab.runs.expand_command import summary
from pushlab.runs.session import refuse, runs_text
from pushlab.runs.spec import ExperimentSpec


def _inputs(
    spec: ExperimentSpec, expansion: Expansion, identity: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    """The inputs' hashes now, and a note when some cannot be read.

    Where the model is not downloaded, as in a cloud session, the download
    and the calibration are taken to be as the folder has them; but the
    per_prompt calibration needs no file, so it is known here too. Where
    the model is downloaded but the rest cannot be read, such as a
    calibration not made yet, the note says why pushlab run would refuse.
    """
    from pushlab.models import load

    known = {
        "spec_sha256": expansion.spec_sha256,
        "prompts_sha256": expansion.prompts_sha256,
        "known_sha256": expansion.known_sha256,
    }
    if identity is not None:  # until read: as the folder has them
        known |= {k: identity.get(k) for k in ("model", "calibration_sha256")}
    try:
        revision = load.downloaded_revision(spec.model)
    except FileNotFoundError:
        unchecked = "the model is not downloaded here: its download"
        if spec.calibration != plan.PER_PROMPT:
            return known, f"{unchecked} and calibration were not checked"
        pushed = any(run.record.schedule for run in expansion.runs)
        known["calibration_sha256"] = plan.PER_PROMPT if pushed else None
        return known, f"{unchecked} was not checked"
    except ValueError as error:
        return known, f"pushlab run would refuse this: {error}"
    try:
        return plan.current_inputs(spec, expansion), None
    except (OSError, ValueError) as error:
        known["model"] = f"{spec.model}@{revision}"
        return known, f"pushlab run would refuse this: {error}"


def _layout(
    spec: ExperimentSpec, expansion: Expansion
) -> tuple[list[plan.Call], str | None]:
    """The calls the run would make, and a note when they are a guess."""
    from pushlab.runs.runner import layout

    try:
        return layout(spec, expansion, expansion.prompts_sha256), None
    except FileNotFoundError:  # no tokenizer here: each prompt counts as its own length
        alone = {run.record.prompt_id: n for n, run in enumerate(expansion.runs)}
        note = "without the model's tokenizer, batches are counted one prompt each"
        return plan.calls(expansion, spec.batch_size, alone), note


def print_dry_run(
    spec: ExperimentSpec,
    expansion: Expansion,
    path: Path,
    runs_dir: Path,
    limit: int | None,
    retry_failed: bool,
    overwrite_version: bool,
) -> int:
    """Print what `pushlab run` would do in path; return the exit status.

    Example:
        >>> print_dry_run(
        ...     spec, expand(spec), path, runs_dir, None, False, False
        ... )  # doctest: +SKIP
        0
    """
    try:
        costs = read_costs()
    except ValueError as error:
        return refuse(str(error))
    print(summary(spec, expansion), end="")
    identity = folder.read_identity(path)
    inputs, unchecked = _inputs(spec, expansion, identity)
    planned, guessed = _layout(spec, expansion)
    notes = [note for note in (unchecked, guessed, speeds_problem(runs_dir)) if note]
    states: dict[str, dict[str, Any]] = {}
    used, saved = 0.0, "none is saved yet"
    if identity is not None:
        version = identity["version"]
        entries = folder.read_index(path)
        if changed := plan.differences(identity, inputs):
            rows = {r.record.run_id: i for c in planned for i, r in enumerate(c.runs)}
            sampled = spec.decoding["strategy"] == "sample"
            states = versions.kept_runs(
                path, identity, expansion, inputs, rows, sampled
            )
            used = versions.kept_seconds(entries, version, states)
            saved = (
                f"a new version would keep {runs_text(len(states))} of version "
                f"{version}"
            )
            if not overwrite_version:  # what the advice would do
                upcoming = versions.next_version(identity, entries)
                why = plan.changed_message(path, identity, changed, upcoming)
                notes.append(f"pushlab run would refuse this: {why}")
                count = len(folder.latest(entries, version))
                saved = (
                    f"{runs_text(count)} saved in version {version}; with "
                    f"--overwrite-version, {saved}"
                )
            if sampled and guessed and states:
                notes.append(
                    "under sampling a new version keeps a saved run only on its "
                    "row, which needs the tokenizer here: the runs it keeps are a "
                    "guess"
                )
        else:
            states = folder.latest(entries, version)
            used = folder.used_seconds(entries, version)
            verb = "is" if len(states) == 1 else "are"
            saved = f"{runs_text(len(states))} {verb} saved in version {version}"
    todo = plan.work(planned, states, retry_failed)
    chosen = plan.within_limit(todo, limit)
    due = sum(len(call.due) for call in chosen)
    rows = sum(len(call.runs) for call in chosen)
    to_run = f"{due} of {len(expansion.runs)} runs; {saved}"
    if limit is not None and chosen != todo:
        to_run += f" (--limit {limit})"
    if used >= spec.max_gpu_hours * 3600 and due:
        notes.append("max_gpu_hours is used up: the run would stop at once")
    speed = speed_for(spec.model, spec.decoding["dtype"], runs_dir, costs)
    lines = [("to run", to_run)]
    if chosen:
        sizes = sorted({len(call.runs) for call in chosen})
        each = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
        how = "prompts of one length together" if spec.group_by_length else "padded"
        batches = "1 batch" if len(chosen) == 1 else f"{len(chosen)} batches"
        runs = "run" if sizes[-1] == 1 else "runs"
        lines.append(("run in", f"{batches} of {each} {runs}, {how}"))
    lines += estimate_lines(
        rows, spec.max_new_tokens, speed, costs, spec.max_gpu_hours, used / 3600
    )
    width = max(len(name) for name, _ in lines) + 2
    print()
    for name, value in lines:
        print(f"{name:<{width}}{value}")
    for note in notes:
        print(f"\nnote: {note.rstrip('.')}.")
    print("\nThis was a dry run: no model was loaded, and nothing was run or written.")
    return 0
