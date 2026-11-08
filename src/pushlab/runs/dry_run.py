"""`pushlab run --dry-run`: what a run would do and cost, with nothing run.

It prints the experiment's runs by condition (as `pushlab expand --summary`
does), then the runs still to run in its folder as the same command would
take them (with --limit, --retry-failed and --overwrite-version), their
reply tokens, the speed it estimates with, the GPU-hours against what is
left of max_gpu_hours, and the cost (pushlab.runs.estimate). It loads no
model and writes nothing, so it works in a cloud session too. There, where
the model is not downloaded, it cannot read the prompts' lengths in
tokens, nor check the model download and calibration, and says so.
"""

from pathlib import Path
from typing import Any

from pushlab.runs import folder, plan, versions
from pushlab.runs.estimate import estimate_lines, read_costs, speed_for
from pushlab.runs.expand import Expansion
from pushlab.runs.expand_command import summary
from pushlab.runs.session import refuse, runs_text
from pushlab.runs.spec import ExperimentSpec


def _inputs(
    spec: ExperimentSpec, expansion: Expansion, identity: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    """The inputs' hashes now, and a note when the download cannot be checked."""
    try:
        return plan.current_inputs(spec, expansion), None
    except (FileNotFoundError, ValueError):
        known = {
            "spec_sha256": expansion.spec_sha256,
            "prompts_sha256": expansion.prompts_sha256,
            "known_sha256": expansion.known_sha256,
        }
        if identity is not None:  # taken to be as the folder has them
            known |= {k: identity.get(k) for k in ("model", "calibration_sha256")}
        unchecked = "the model is not downloaded here: its download and calibration"
        return known, f"{unchecked} were not checked"


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
    notes = [note for note in (unchecked, guessed) if note]
    states: dict[str, dict[str, Any]] = {}
    used, saved = 0.0, "none is saved yet"
    if identity is not None:
        version = identity["version"]
        entries = folder.read_index(path)
        if changed := plan.differences(identity, inputs):
            if not overwrite_version:
                why = plan.changed_message(path, identity, changed)
                notes.append(f"pushlab run would refuse this: {why}")
            else:
                rows = {
                    r.record.run_id: i for c in planned for i, r in enumerate(c.runs)
                }
                sampled = spec.decoding["strategy"] == "sample"
                states = versions.kept_runs(
                    path, identity, expansion, inputs, rows, sampled
                )
                saved = (
                    f"a new version would keep {runs_text(len(states))} of version "
                    f"{version}"
                )
                used = versions.kept_seconds(entries, version, states)
        else:
            states = folder.latest(entries, version)
            used = folder.used_seconds(entries, version)
            saved = f"{len(states)} are saved in version {version}"
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
        lines.append(("run in", f"{batches} of {each} runs, {how}"))
    lines += estimate_lines(
        rows, spec.max_new_tokens, speed, costs, spec.max_gpu_hours, used / 3600
    )
    width = max(len(name) for name, _ in lines) + 2
    print()
    for name, value in lines:
        print(f"{name:<{width}}{value}")
    for note in notes:
        print(f"\nnote: {note}.")
    print("\nThis was a dry run: nothing was loaded, run or written.")
    return 0
