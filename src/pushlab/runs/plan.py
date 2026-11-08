"""What a run of an experiment folder will do, worked out before any model loads.

- Whose folder it is: the runner works only in a folder that is empty or
  holds a runner folder of the same experiment.
- What its runs were made from: the hashes of the experiment file, the
  prompt set, units/known.csv, the model download and the calibration. A
  folder made from other ones is refused, unless a new version is asked
  for (pushlab.runs.versions).
- How the runs run: in calls, each one generate() call of planned batches
  that can share a batch; and what is left: the calls with runs still to
  run, or with runs that failed, in plan order.
"""

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.runs import folder
from pushlab.runs.expand import Expansion, PlannedRun
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.folders import shown_path
from pushlab.store.records import RUNS_FILE

# Each input a folder's runs are made from, and its name in messages.
IDENTITY_KEYS = {
    "spec_sha256": "experiment file",
    "prompts_sha256": "prompt set",
    "known_sha256": "known units",
    "model": "model download",
    "calibration_sha256": "calibration",
}


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
    names = sorted(
        p.name
        for p in path.iterdir()
        if p.name != folder.LOCK_FILE  # the runner's own, as are hidden .tmp files
        and not (p.name.startswith(".") and p.name.endswith(".tmp"))
    )
    if names:
        held = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
        return f"{path} is not a runner folder: it holds {held}; {never}"
    return None


def runs_dir_problem(runs_dir: Path) -> str | None:
    """Why runs_dir cannot hold experiments' folders, or None.

    It must be a folder, or not exist yet, and neither it nor a folder
    above it may be a run folder: a runner folder, with experiment.json, or
    one of S0's and S1's, with runs.parquet. The path is resolved first, so
    a relative path or one with .. is checked where it really leads.

    Example:
        >>> runs_dir_problem(Path("/no/such/folder")) is None
        True
    """
    resolved = runs_dir.resolve()
    for place in (resolved, *resolved.parents):
        if place.exists() and not place.is_dir():
            return f"the runs dir {runs_dir} must be a folder, but {place} is a file"
        names = (folder.IDENTITY_FILE, RUNS_FILE)
        if held := [name for name in names if (place / name).exists()]:
            return (
                f"{place} is a run folder (it holds {', '.join(held)}), and the "
                "runs dir must be the folder above experiments' folders, such as "
                "/workspace/runs; the runner never writes into another "
                "experiment's folder"
            )
    return None


def current_inputs(spec: ExperimentSpec, expansion: Expansion) -> dict[str, Any]:
    """The hashes of what the spec's runs would be made from now.

    It reads the model's download record and, when anything is pushed, its
    calibration, but loads no model. FileNotFoundError or ValueError says
    what is missing and how to make it.

    Example:
        >>> current_inputs(spec, expand(spec))["model"]  # doctest: +SKIP
        'qwen3.5-4b@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
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


# A change to one of these makes every saved run another run: none is kept.
KEEPS_NONE = ("model download", "prompt set", "calibration")


def changed_message(
    path: Path, identity: dict[str, Any], changed: list[str], upcoming: int
) -> str:
    """Why a folder made from other inputs is refused, and what to do.

    upcoming is the number a new version would take (versions.next_version).

    Example:
        >>> identity = {"experiment": "x", "version": 1}
        >>> message = changed_message(Path("runs/x"), identity, ["prompt set"], 3)
        >>> message[:59]
        'runs/x holds version 1 of x, made from another prompt set. '
        >>> message[59:]  # doctest: +NORMALIZE_WHITESPACE
        'Run with --overwrite-version to start version 3, which keeps no saved
        run, since the prompt set changed, or move the folder away to start afresh'
    """
    if lost := [name for name in changed if name in KEEPS_NONE]:
        keeps = f"which keeps no saved run, since the {' and '.join(lost)} changed"
    else:
        keeps = "which keeps the saved runs that did not change"
    return (
        f"{path} holds version {identity['version']} of {identity['experiment']}, "
        f"made from another {' and '.join(changed)}. Run with --overwrite-version "
        f"to start version {upcoming}, {keeps}, or move the folder away to start "
        "afresh"
    )


def new_identity(
    spec: ExperimentSpec,
    spec_path: Path,
    expansion: Expansion,
    inputs: dict[str, Any],
    version: int,
) -> dict[str, Any]:
    """What experiment.json says about a new version of the folder.

    Its conditions map each run ID to its condition's label, such as
    "pulse length=16", which no run record holds; a control has its
    condition's label, and its record's control_of names the run it mirrors.

    Example:
        >>> new_identity(spec, path, expand(spec), inputs, 1)[
        ...     "version"
        ... ]  # doctest: +SKIP
        1
    """
    return inputs | {
        "experiment": spec.name,
        "version": version,
        "spec_file": shown_path(spec_path),
        "spec": spec.to_mapping(),
        "runs": len(expansion.runs),
        "conditions": {run.record.run_id: run.condition for run in expansion.runs},
        "max_gpu_hours": spec.max_gpu_hours,
        "max_new_tokens": spec.max_new_tokens,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


@dataclass(frozen=True)
class Call:
    """One generate() call: runs that can share a batch, in their rows' order.

    due names the runs a session saves from it, which work() fills in: all
    of them, unless some are saved already (then the call still runs whole,
    so that every row is where the plan puts it, and only the due ones are
    saved). calls() leaves it empty.
    """

    key: str
    runs: tuple[PlannedRun, ...]
    due: frozenset[str] = frozenset()


def prompt_lengths(
    tokenizer: Any, prompt_ids: list[str], prompts: list[str]
) -> dict[str, int]:
    """Each prompt's length in tokens, as generate() renders it.

    Example:
        >>> prompt_lengths(tokenizer, ["p-1"], ["Say hi."])  # doctest: +SKIP
        {'p-1': 12}
    """
    from pushlab.models.chat import chat_prompt_ids

    return {
        prompt_id: len(chat_prompt_ids(tokenizer, prompt, thinking=False))
        for prompt_id, prompt in zip(prompt_ids, prompts, strict=True)
    }


def calls(
    expansion: Expansion, batch_size: int, lengths: dict[str, int] | None
) -> list[Call]:
    """The generate() calls that run every planned batch, in plan order.

    expand plans each batch from one prompt and seed. Planned batches share
    a call when their runs have the same model and decoding settings, seed
    included, and, given lengths (each prompt's length in tokens, when
    grouping by length is on), prompts of one length; a call holds at most
    batch_size runs, and a planned batch stays whole, so a run stays with
    its controls. Calls are numbered in the plan order of their first run,
    and the same plan always gives the same calls.

    Example:
        >>> [len(call.runs) for call in calls(expansion, 8, None)]  # doctest: +SKIP
        [8, 2]
    """
    planned: dict[str, list[PlannedRun]] = {}
    for run in expansion.runs:
        planned.setdefault(run.record.batch_key.split(":")[0], []).append(run)
    filling: dict[tuple[Any, ...], list[PlannedRun]] = {}
    opened: list[list[PlannedRun]] = []
    for runs in planned.values():
        first = runs[0].record
        length = None if lengths is None else lengths[first.prompt_id]
        group = (
            first.model,
            json.dumps(first.decoding, sort_keys=True),
            first.seed,
            length,
        )
        call = filling.get(group)
        if call is None or len(call) + len(runs) > batch_size:
            call = filling[group] = []
            opened.append(call)
        call.extend(runs)
    return [Call(f"b{n:06d}", tuple(runs)) for n, runs in enumerate(opened, start=1)]


def is_due(state: dict[str, Any] | None, retry_failed: bool) -> bool:
    """Whether a run is to run: not saved yet, or with retry_failed failed.

    Example:
        >>> is_due(None, False), is_due({"status": "failed"}, True)
        (True, True)
    """
    if retry_failed:
        return state is not None and state["status"] == "failed"
    return state is None


def work(
    planned: list[Call], states: dict[str, dict[str, Any]], retry_failed: bool
) -> list[Call]:
    """The calls with runs to run, each with its due runs, in plan order.

    Example:
        >>> [
        ...     call.key for call in work(calls(expansion, 8, None), {}, False)
        ... ]  # doctest: +SKIP
        ['b000001', 'b000002']
    """
    todo = []
    for call in planned:
        due = {
            run.record.run_id
            for run in call.runs
            if is_due(states.get(run.record.run_id), retry_failed)
        }
        if due:
            todo.append(replace(call, due=frozenset(due)))
    return todo


def within_limit(planned: list[Call], limit: int | None) -> list[Call]:
    """The first calls whose due runs add up to at most limit, if one is given.

    Example:
        >>> todo = [Call("b1", (), frozenset("ab")), Call("b2", (), frozenset("cd"))]
        >>> [call.key for call in within_limit(todo, 3)]
        ['b1']
    """
    if limit is None:
        return planned
    chosen, count = [], 0
    for call in planned:
        if count + len(call.due) > limit:
            break
        chosen.append(call)
        count += len(call.due)
    return chosen
