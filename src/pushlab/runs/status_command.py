"""`pushlab status`: how far an experiment has got, and how long the rest will take.

It reads the experiment file for the experiment's name, then its runner
folder: experiment.json, the index and the heartbeat. It prints the runs
saved ok, failed and pending, the calibration the runs use (fixed or
per_prompt), the batches saved, the speed measured over them, the
GPU-hours used of max_gpu_hours and the time left at that speed
(or, before any batch, at the speed estimates use, with the model's load).
It loads no model and writes nothing, so it works while a run is going; to
see whether a runner holds the folder, it tries a shared lock for an instant.
"""

import os
import socket
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pushlab.runs import folder
from pushlab.runs.estimate import (
    Speed,
    duration,
    exact,
    read_costs,
    speed_for,
    speeds_problem,
)
from pushlab.runs.spec import ExperimentSpec, read_spec

# The most failed runs it names; the count says how many there are.
FAILED_SHOWN = 5


def _refuse(message: str) -> int:
    print(f"pushlab status: {message}", file=sys.stderr)
    return 1


def pid_alive(pid: int) -> bool:
    """Whether a process with this ID runs on this machine.

    Example:
        >>> pid_alive(os.getpid())
        True
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # someone else's, but it is there
        return True
    return True


def _runner(path: Path, beat: dict[str, Any] | None) -> str:
    """Whether a runner works on the folder: its heartbeat, its process and
    the folder's lock must all say so."""
    if beat is None:
        return "not running: no session has started"
    if beat.get("state") != "running":
        ended = f"{beat['updated_at']} ({beat['state']})"
        return f"not running; its last session ended at {ended}"
    pid, host = beat["pid"], beat.get("host")
    if host != socket.gethostname():
        return f"running on {host}, pid {pid}, as of {beat['updated_at']}"
    go_on = "run the same command to go on"
    if not pid_alive(pid):
        return f"stopped without finishing: pid {pid} is gone; {go_on}"
    if not folder.lock_held(path):
        return (
            f"stopped without finishing: no process holds the folder's lock, so "
            f"pid {pid} is another program's now; {go_on}"
        )
    return f"running, pid {pid}; the last batch was saved at {beat['updated_at']}"


def _estimated(runs: int, max_new_tokens: int, speed: Speed) -> str:
    """The time runs would take at an estimate's speed, the model's load in."""
    tokens = runs * max_new_tokens
    seconds = speed.load_seconds + tokens / speed.reply_tokens_per_second
    return f"about {duration(seconds)}, at {speed.text()}"


def _changed(spec: ExperimentSpec, identity: dict[str, Any]) -> list[str]:
    """The inputs that differ from those the folder's version was made from,
    of the ones status can check without the model's download."""
    from pushlab.runs.expand import expand
    from pushlab.runs.plan import differences

    current = {"spec_sha256": spec.sha256}
    try:
        expansion = expand(spec)
    except ValueError:  # pushlab run would say what is wrong with it
        return differences(identity, current)
    current |= {
        "prompts_sha256": expansion.prompts_sha256,
        "known_sha256": expansion.known_sha256,
    }
    return differences(identity, current)


def status_lines(spec_path: Path, runs_dir: Path) -> list[tuple[str, str]]:
    """What `pushlab status` prints, as (name, value) lines.

    Example:
        >>> status_lines(
        ...     Path("experiments/p0/s2_pulse_lengths.yaml"), RUNS
        ... )  # doctest: +SKIP
        [('experiment', 's2-pulse-lengths, version 1, in /workspace/runs/...'), ...]
    """
    from pushlab.runs.plan import (
        PER_PROMPT,
        calibration_text,
        folder_problem,
        runs_dir_problem,
    )

    spec = read_spec(spec_path)
    path = runs_dir / spec.name
    if problem := runs_dir_problem(runs_dir) or folder_problem(path, spec.name):
        raise ValueError(problem)
    identity = folder.read_identity(path)
    costs = read_costs()
    speed = speed_for(spec.model, spec.decoding["dtype"], runs_dir, costs)
    if identity is None:
        from pushlab.runs.expand import expand  # for the runs it would make

        runs = expand(spec).runs
        total, pushed = len(runs), any(run.record.schedule for run in runs)
        where = "does not exist yet" if not path.exists() else "holds no runs yet"
        lines = [
            ("experiment", spec.name),
            ("runner", f"not started: {path} {where}"),
            ("runs", f"{total}: 0 ok, 0 failed, {total} pending"),
            ("calibration", calibration_text(spec.calibration, pushed)),
        ]
        if speed is not None:
            lines.append(("time left", _estimated(total, spec.max_new_tokens, speed)))
        if problem := speeds_problem(runs_dir):
            lines.append(("note", problem))
        return lines
    version, total = identity["version"], identity["runs"]
    entries = [e for e in folder.read_index(path) if e.get("version") == version]
    states = folder.latest(entries, version)
    counts = Counter(state["status"] for state in states.values())
    pending = total - len(states)
    ran = [e for e in entries if "kept_from" not in e]
    kept = len(entries) - len(ran)
    batches = f"{len(ran)} saved" + (
        f", and {kept} kept from earlier versions" if kept else ""
    )
    # The version's own: its file's SHA-256 (fixed), PER_PROMPT, or None.
    mark = identity.get("calibration_sha256")
    mode = PER_PROMPT if mark == PER_PROMPT else "fixed"
    lines = [
        ("experiment", f"{spec.name}, version {version}, in {path}"),
        ("runner", _runner(path, folder.read_json(path / folder.HEARTBEAT_FILE))),
        (
            "runs",
            f"{total}: {counts['ok']} ok, {counts['failed']} failed, {pending} pending",
        ),
        ("calibration", calibration_text(mode, mark is not None)),
        ("batches", batches),
    ]
    good = [e for e in ran if not e.get("error")]
    tokens = sum(e.get("reply_tokens", 0) for e in good)
    seconds = sum(e.get("generate_seconds", 0.0) for e in good)
    used = folder.used_seconds(entries, version) / 3600
    if tokens and seconds:
        rate = tokens / seconds
        over = "1 batch" if len(good) == 1 else f"{len(good)} batches"
        lines.append(
            ("speed", f"{rate:.3g} reply tokens a second, measured over {over}")
        )
    cap = exact(identity.get("max_gpu_hours", spec.max_gpu_hours))
    lines.append(("gpu-hours", f"{used:.2f} used of max_gpu_hours {cap}"))
    if not pending:
        lines.append(("time left", "none: every run is saved"))
    elif tokens and seconds:
        rows = sum(e.get("rows", len(e["runs"])) for e in good)
        left = pending * seconds / rows  # each run takes a row's share
        lines.append(
            ("time left", f"about {duration(left)} for {pending} runs at that speed")
        )
    elif speed is not None:
        lines.append(
            ("time left", _estimated(pending, identity["max_new_tokens"], speed))
        )
    failed = sorted(run_id for run_id, s in states.items() if s["status"] == "failed")
    for run_id in failed[:FAILED_SHOWN]:
        lines.append(("failed", f"{run_id}: {states[run_id]['error']}"))
    if len(failed) > FAILED_SHOWN:
        lines.append(("failed", f"and {len(failed) - FAILED_SHOWN} more"))
    for name in _changed(spec, identity):
        lines.append(
            (
                "note",
                f"the {name} changed since version {version}: pushlab run will "
                "refuse it without --overwrite-version",
            )
        )
    if problem := speeds_problem(runs_dir):
        lines.append(("note", problem))
    return lines


def run_status(spec_path: Path, runs_dir: Path | None = None) -> int:
    """Print how far the experiment at spec_path has got; return the exit status.

    The status is 0, or 1 with the reason on stderr when the experiment file
    or its folder cannot be read.

    Example:
        >>> run_status(Path("experiments/p0/s2_pulse_lengths.yaml"))  # doctest: +SKIP
        0
    """
    runs_dir = folder.default_runs_dir() if runs_dir is None else runs_dir
    try:
        lines = status_lines(spec_path, runs_dir)
    except ValueError as error:
        return _refuse(str(error))
    width = max(len(name) for name, _ in lines) + 2
    for name, value in lines:
        print(f"{name:<{width}}{value}")
    return 0
