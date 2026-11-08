"""`pushlab run --dry-run`: what a run would do and cost, with nothing run.

It prints the experiment's runs by condition (as `pushlab expand --summary`
does), then how many runs are still to run in its folder, their reply
tokens, the speed it estimates with, the GPU-hours against max_gpu_hours
and the cost (pushlab.runs.estimate). It loads no model, reads no
calibration and writes nothing, so it works in a cloud session too, where
the speed comes from configs/costs.yaml.
"""

import sys
from pathlib import Path

from pushlab.runs import folder, plan
from pushlab.runs.estimate import estimate_lines, read_costs, speed_for
from pushlab.runs.expand import Expansion
from pushlab.runs.expand_command import summary
from pushlab.runs.spec import ExperimentSpec


def print_dry_run(
    spec: ExperimentSpec,
    expansion: Expansion,
    path: Path,
    runs_dir: Path,
    limit: int | None,
    retry_failed: bool,
) -> int:
    """Print what `pushlab run` would do in path; return the exit status.

    Example:
        >>> print_dry_run(
        ...     spec, expand(spec), path, runs_dir, None, False
        ... )  # doctest: +SKIP
        0
    """
    try:
        costs = read_costs()
    except ValueError as error:
        print(f"pushlab run: {error}", file=sys.stderr)
        return 1
    print(summary(spec, expansion), end="")
    identity = folder.read_identity(path)
    entries: list[dict] = []
    states: dict[str, dict] = {}
    note = None
    if identity is not None:
        # The inputs a dry run can check without the model's download.
        current = {
            "spec_sha256": expansion.spec_sha256,
            "prompts_sha256": expansion.prompts_sha256,
            "known_sha256": expansion.known_sha256,
        }
        if changed := plan.differences(identity, current):
            note = plan.changed_message(path, identity, changed)
        else:
            entries = folder.read_index(path)
            states = folder.latest(entries, identity["version"])
    todo = plan.work(expansion, states, retry_failed)
    chosen = plan.within_limit(todo, limit)
    runs = sum(len(batch.runs) for batch in chosen)
    total = len(expansion.runs)
    if states:
        saved = f"{len(states)} are saved in version {identity['version']}"
    else:
        saved = "none is saved yet"
    to_run = f"{runs} of {total} runs; {saved}"
    if limit is not None and chosen != todo:
        to_run += f" (--limit {limit})"
    used = folder.used_seconds(entries, identity["version"]) / 3600 if states else 0.0
    speed = speed_for(spec.model, spec.decoding["dtype"], runs_dir, costs)
    lines = [("to run", to_run)]
    lines += estimate_lines(
        runs, spec.max_new_tokens, speed, costs, spec.max_gpu_hours, used
    )
    width = max(len(name) for name, _ in lines) + 2
    print()
    for name, value in lines:
        print(f"{name:<{width}}{value}")
    if note:
        print(f"\nnote: pushlab run would refuse this: {note}.")
    print("\nThis was a dry run: nothing was loaded, run or written.")
    return 0
